#!/usr/bin/env python3
"""Experimental one-shower/static-frontier coordinator; no MPI or shared stack.

Input JSON: command (loader prefix + worker executable), physics_args (ordinary
c8_air_shower options, including fixed -E), seed, devices, output, optional env.
Outputs are NEW directories. A failed part can never produce COMPLETE.json.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time

HEADER = 'C8_STATIC_FRONTIER_V1'
RESERVED = {'-N', '--nevent', '-f', '--filename', '-s', '--seed', '--em-backend',
            '--radio-backend', '--kokkos-execution', '--kokkos-device',
            '--kokkos-num-threads', '--hadronic-workers', '--hadronic-backend',
            '--gpu-memory-fraction', '--gpu-resident-batch-limit'}

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

def save(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)

def memory_fraction(value):
    fraction = float(value)
    if not math.isfinite(fraction) or not 0 < fraction <= .9:
        raise ValueError('GPU memory fraction must be finite and in (0, 0.90]')
    return fraction

def validate_frontier_feed(folder, expected):
    feed = json.loads((folder / 'FRONTIER_FEED.json').read_text())
    if (feed.get('mode') != 'bounded-high-energy-first' or
        feed.get('roots_consumed') != expected or
        not 0 < feed.get('maximum_imported_batch', 0) <= 65536 or
        feed.get('batches', 0) < 1):
        raise ValueError('Incomplete/invalid bounded frontier feed: ' + str(folder))

def process_identity(pid):
    return dict(pid=pid, start=Path('/proc/%d/stat' % pid).read_text().rsplit(')', 1)[1].split()[19])

def validate_closed_outputs(folder):
    import numpy as np
    import pyarrow.parquet as pq
    import yaml
    for rel in ('profile/profile.parquet', 'CoREAS/observers.parquet', 'ZHS/observers.parquet'):
        meta = pq.read_metadata(folder / rel)
        if meta.num_rows == 0:
            raise ValueError('Empty required output (use --ring 1 or an antenna file): ' + rel)
    for path in folder.rglob('*.parquet'):
        for batch in pq.ParquetFile(path).iter_batches(batch_size=16384, use_threads=False):
            for column in batch.columns:
                if column.null_count:
                    raise ValueError('Nulls in output: ' + str(path))
                values = column.to_numpy(zero_copy_only=False)
                if values.dtype.kind in 'fiu' and not np.isfinite(values).all():
                    raise ValueError('Non-finite output: ' + str(path))
    statistics_path = folder / 'gpu_em' / 'summary.yaml'
    if statistics_path.exists():
        statistics = yaml.safe_load(statistics_path.read_text())['shower_0']['statistics']
        if statistics['queue_overflows'] or statistics['radio']['fixed_point_overflows']:
            raise ValueError('Queue/fixed-point overflow: ' + str(folder))

def partition(source, destinations):
    """Largest-energy-first scheduling is a cost proxy, NEVER a physics weight.

    No roots are cloned or reweighted. Descendants never migrate after this split.
    """
    lines = Path(source).read_text().splitlines()
    if not lines or lines.pop(0) != HEADER:
        raise ValueError('Invalid frontier header')
    roots, seen = [], set()
    for line in lines:
        cols = line.split()
        if len(cols) != 14:
            raise ValueError('Invalid frontier column count')
        history, energy, weight = int(cols[1]), float(cols[5]), float(cols[6])
        if history in seen or not math.isfinite(energy * weight) or energy < 0 or weight < 0:
            raise ValueError('Duplicate/invalid root')
        seen.add(history)
        roots.append((energy, history, weight, line))
    shards = [[] for _ in destinations]
    cost = [0.] * len(shards)
    for energy, history, weight, line in sorted(roots, reverse=True):
        k = min(range(len(shards)), key=lambda j: (cost[j], j))
        shards[k].append(line)
        cost[k] += energy
    for path, rows in zip(destinations, shards):
        with Path(path).open('x') as f:
            f.write(HEADER + '\n' + ''.join(row + '\n' for row in rows))
    return dict(roots=len(roots), counts=list(map(len, shards)),
                weighted_energy_GeV=math.fsum(e*w for e, _, w, _ in roots),
                cost_energy_GeV=cost, frontier_sha256=digest(source),
                shard_sha256=[digest(p) for p in destinations])

def merge(parts, destination):
    """Merge linear observables on IDENTICAL grids, not powers or aligned peaks.

    V1 sums closed writer arrays in float64. Device fixed-point accumulators stay
    private to each existing backend; cross-worker integer merging is NOT claimed.
    Component outputs are retained to audit rounding and every contribution.
    """
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    import yaml
    destination.mkdir()
    products = {}
    grid_products = {
        'profile/profile.parquet': ('shower', 'X'),
        'production_profile/profile.parquet': ('shower', 'X'),
        'energyloss/dEdX.parquet': ('shower', 'X'),
        'CoREAS/observers.parquet': ('shower', 'Time'),
        'ZHS/observers.parquet': ('shower', 'Time'),
    }
    for name, grid in grid_products.items():
        reference = pq.read_table(parts[0] / name)
        columns = reference.column_names
        arrays = {k: reference[k].to_numpy() for k in columns}
        accum = {k: arrays[k].astype(np.float64) for k in columns if k not in grid}
        correction = {k: np.zeros_like(v) for k, v in accum.items()}
        # Check observer names/order, absolute timing, units and positions too.
        configs = [yaml.safe_load((p / name).with_name('config.yaml').read_text()) for p in parts]
        if any(c != configs[0] for c in configs[1:]):
            raise ValueError('Output configuration mismatch: ' + name)
        for part in parts[1:]:
            t = pq.read_table(part / name)
            if t.column_names != columns or t.num_rows != reference.num_rows:
                raise ValueError('Schema/shape mismatch: ' + name)
            if any(not np.array_equal(t[k].to_numpy(), arrays[k]) for k in grid):
                raise ValueError('Grid mismatch: ' + name)
            for k in accum:
                value = t[k].to_numpy().astype(np.float64)
                if not np.isfinite(value).all():
                    raise ValueError('Non-finite output: ' + name)
                y = value - correction[k]
                total = accum[k] + y
                correction[k] = (total - accum[k]) - y
                accum[k] = total
        if not all(np.isfinite(v).all() for v in accum.values()):
            raise ValueError('Non-finite merged output')
        target = destination / name
        target.parent.mkdir(parents=True)
        pq.write_table(pa.table({k: arrays[k] if k in grid else accum[k] for k in columns}), target)
        target.with_name('config.yaml').write_text(yaml.safe_dump(configs[0], sort_keys=False))
        products[name] = dict(rows=reference.num_rows, sha256=digest(target))
        if name.startswith(('CoREAS/', 'ZHS/')):
            products[name]['nonzero_field_entries'] = sum(int(np.count_nonzero(v)) for v in accum.values())
            products[name]['maximum_absolute_field'] = max(
                (float(np.max(np.abs(v))) for v in accum.values() if v.size), default=0.)
    # Ground particles are disjoint. InteractionWriter records ONLY the FIRST
    # interaction, not a complete event log: keep the GLOBAL prefix record,
    # never mislabel each worker's first local reaction as another primary.
    for name in ('particles/particles.parquet', 'interactions/interactions.parquet'):
        target = destination / name
        target.parent.mkdir(parents=True)
        schema = pq.read_schema(parts[0] / name)
        rows = 0
        with pq.ParquetWriter(target, schema) as writer:
            record_parts = parts if name.startswith('particles/') else parts[:1]
            for part in record_parts:
                f = pq.ParquetFile(part / name)
                if f.schema_arrow != schema:
                    raise ValueError('Record schema mismatch')
                for batch in f.iter_batches(batch_size=65536):
                    writer.write_batch(batch)
                    rows += batch.num_rows
        products[name] = dict(rows=rows, sha256=digest(target))
        configs = [yaml.safe_load((p / name).with_name('config.yaml').read_text()) for p in parts]
        if any(c != configs[0] for c in configs[1:]):
            raise ValueError('Record configuration mismatch: ' + name)
        target.with_name('config.yaml').write_text(yaml.safe_dump(configs[0], sort_keys=False))
        if name.startswith('interactions/'):
            summary = (parts[0] / name).with_name('summary.yaml')
            if summary.exists():
                target.with_name('summary.yaml').write_bytes(summary.read_bytes())
    save(destination / 'MERGE.json', dict(parts=[str(p) for p in parts], products=products,
         arithmetic='compensated float64 sum of closed arrays; no cross-worker fixed-point claim',
         statistical_acceptance=False))
    return products

def run(config):
    cfg = json.loads(Path(config).read_text())
    root = Path(cfg['output']).resolve()
    args = list(map(str, cfg['physics_args']))
    for a in args:
        if a.split('=')[0] in RESERVED or a.startswith('--frontier-'):
            raise ValueError('Coordinator owns option: ' + a)
    if '-E' not in args or '--energy_range' in args or '--force-interaction' in args or '--force-decay' in args:
        raise ValueError('Require fixed -E and unforced primary')
    energy = float(args[args.index('-E') + 1])
    if not math.isfinite(energy) or energy <= 0 or not 0 < int(cfg['seed']) < 2**63:
        raise ValueError('Finite positive energy and explicit positive seed required')
    devices = list(map(str, cfg['devices']))
    if not devices or len(set(devices)) != len(devices) or len(devices) > 255:
        raise ValueError('Nonempty distinct GPU IDs required')
    inventory = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid',
                                         '--format=csv,noheader,nounits'], text=True)
    aliases = {}
    for row in inventory.splitlines():
        index, uuid = map(str.strip, row.split(','))
        aliases[index] = aliases[uuid] = uuid
    devices = [aliases[d] for d in devices]
    if len(set(devices)) != len(devices):
        raise ValueError('GPU aliases refer to the same physical device')
    # Default stays conservative; the explicitly authorized UHE campaign uses
    # 90%. This is an allocator budget, not a promise of physical occupancy.
    fraction = memory_fraction(cfg.get('gpu_memory_fraction', .5))
    root.mkdir(parents=True, exist_ok=False)
    save(root / 'CONFIG.json', cfg)
    save(root / 'DEVICE_UUIDS.json', devices)
    env = dict(os.environ, **cfg.get('env', {}), OMP_NUM_THREADS='1', OMP_THREAD_LIMIT='1',
               OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    running = []
    records = []
    start = time.monotonic()
    common = args + ['-N', '1', '--hadronic-workers', '1', '--hadronic-backend', 'scalar']

    def launch(name, options, childenv):
        folder = root / name
        folder.mkdir()
        debugger = cfg.get('worker_debugger', []) if name.startswith('worker_') else []
        argv = debugger + cfg['command'] + common + ['-f', str(folder / 'shower')] + options
        save(folder / 'COMMAND.json', argv)
        log = (folder / 'run.log').open('x')
        p = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                             env=childenv, cwd=folder, start_new_session=True)
        log.close()
        save(folder / 'OWNED_PROCESS.json', process_identity(p.pid))
        item = dict(name=name, pid=p.pid, start_s=time.monotonic()-start)
        running.append((p, folder, item))
        return p

    def finish(p):
        rc = p.wait(timeout=float(cfg.get('timeout_s', 7200)))
        entry = next(x for x in running if x[0] is p)
        _, folder, item = entry
        item.update(returncode=rc, end_s=time.monotonic()-start)
        save(folder / 'PROCESS_EXIT.json', item)
        records.append(item)
        if rc:
            raise RuntimeError('Worker failed: ' + item['name'])
        # A closed parquet footer and a normal exit are both mandatory.
        validate_closed_outputs(folder / 'shower')
        if item['name'].startswith('worker_') and not cfg.get('legacy_frontier_import', False):
            index = int(item['name'].split('_')[1])
            validate_frontier_feed(folder, audit['counts'][index])
        running.remove(entry)

    try:
        frontier = root / 'frontier.txt'
        p = launch('prefix', ['-s', str(cfg['seed']), '--em-backend', 'proposal', '--radio-backend', 'cpu',
                '--frontier-out', str(frontier), '--frontier-max-energy',
                str(cfg.get('frontier_max_energy_GeV', energy / (8 * len(devices))))], env)
        finish(p)
        shards = [root / ('roots_%d.txt' % i) for i in range(len(devices))]
        audit = partition(frontier, shards)
        save(root / 'PARTITION.json', audit)
        workers = []
        for i, device in enumerate(devices):
            if audit['counts'][i] == 0:
                continue  # no work is not an error; don't create artificial particles
            worker_seed = int.from_bytes(hashlib.sha256(('%s:%d' % (cfg['seed'], i)).encode()).digest()[:4], 'little') or 1
            options = ['-s', str(worker_seed), '--em-backend', 'kokkos-proposal', '--radio-backend', 'kokkos',
                       '--kokkos-execution', 'cuda', '--kokkos-num-threads', '1', '--kokkos-device', '0',
                       '--gpu-memory-fraction', str(fraction), '--gpu-resident-batch-limit', '0',
                       '--frontier-in', str(shards[i]), '--frontier-worker-id', str(i+1)]
            if cfg.get('legacy_frontier_import', False):
                options.append('--frontier-legacy-import')
            p = launch('worker_%d' % i, options, dict(env, CUDA_VISIBLE_DEVICES=device))
            workers.append(p)
        # Workers were all submitted first. Reap independently (failure is
        # fail-fast; fast workers must not inherit the slowest one's duration).
        pending = list(workers)
        deadline = time.monotonic() + float(cfg.get('timeout_s', 7200))
        while pending:
            for p in list(pending):
                if p.poll() is not None:
                    finish(p)
                    pending.remove(p)
            if pending:
                if time.monotonic() > deadline:
                    raise TimeoutError('Independent worker deadline exceeded')
                time.sleep(.1)
        if digest(frontier) != audit['frontier_sha256'] or [digest(s) for s in shards] != audit['shard_sha256']:
            raise ValueError('Frontier/shards changed while workers were running')
        save(root / 'TIMING.json', dict(wall_s=time.monotonic()-start, parts=records))
        # Completion order may vary; keep numerical merging in worker-ID order.
        completed = sorted((r for r in records if r['name'].startswith('worker_')),
                           key=lambda r: int(r['name'].split('_')[1]))
        parts = [root / 'prefix' / 'shower'] + [root / r['name'] / 'shower' for r in completed]
        products = merge(parts, root / 'merged')
        import yaml
        merged = root / 'merged'
        # There is ONE primary, not one primary per GPU. Keep only the original
        # record, and don't copy misleading partial timing/energy summaries.
        (merged / 'primary').mkdir()
        for name in ('config.yaml', 'summary.yaml'):
            (merged / 'primary' / name).write_bytes((parts[0] / 'primary' / name).read_bytes())
        (merged / 'config.yaml').write_text(yaml.safe_dump(dict(
            creator='CORSIKA8 experimental static multi-GPU', physics_args=args,
            coordinator_config=str(root / 'CONFIG.json'), merge=str(merged / 'MERGE.json'))))
        (merged / 'summary.yaml').write_text(yaml.safe_dump(dict(
            showers=1, seed=cfg['seed'], runtime_seconds=time.monotonic()-start,
            experimental=True, statistical_acceptance=False,
            output_dirs=['profile', 'production_profile', 'energyloss', 'particles',
                         'interactions', 'primary', 'CoREAS', 'ZHS'])))
        save(root / 'COMPLETE.json', dict(products=products, wall_s=time.monotonic()-start,
             partition_sha256=digest(root / 'PARTITION.json'), process_complete=True,
             physics_validated_against_beta2=False, experimental=True))
    except BaseException as e:
        save(root / 'INCOMPLETE.json', dict(error=repr(e), parts=records))
        raise
    finally:
        # Only our own process groups; never touch production or other GPU jobs.
        for p, _, _ in running:
            if p.poll() is None:
                os.killpg(p.pid, signal.SIGTERM)
        for p, _, _ in running:
            try:
                p.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL)
                p.wait()

if __name__ == '__main__':
    def stop(signum, frame):
        raise KeyboardInterrupt('Termination requested')
    signal.signal(signal.SIGTERM, stop)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config')
    run(parser.parse_args().config)

#!/usr/bin/env python3
"""Replay the selector on three fully recorded reference events, on PSR only."""
import argparse
import copy
import importlib.util
import json
import os
import pathlib
import shutil
import socket

os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import pandas as pd
import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True, type=pathlib.Path)
    root = parser.parse_args().root
    assert socket.gethostname() == 'psrpku2025'
    reference = root.parent / 'beta5_material_doublebang_openmp256_20260913'
    spec = importlib.util.spec_from_file_location('selector', root / 'code/classify_psr.py')
    selector = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(selector)
    results = []
    for seed, expected in [(158, False), (946, True), (3605, True)]:
        folder = reference / 'runs' / ('openmp_silica_SiO2_seed%d' % seed)
        summary = yaml.safe_load((folder / 'output/terrain_run.yaml').read_text())
        assert summary['complete']
        target = root / 'preflight' / ('selector_reference_seed%d' % seed)
        target.mkdir(exist_ok=True)
        for name in ['tau_tracks.json', 'command.json']:
            shutil.copy2(folder / name, target / name)
        wanted = {int(d['history_id']) for v in summary['neutrino']['interactions'] + summary['tau']['decays'] for d in v['daughters']}
        witnesses = {}
        for chunk in pd.read_csv(folder / 'output/terrain/tracks.csv', chunksize=250000,
                                  usecols=['history_id', 'pdg', 'medium', 't0_s', 't1_s']):
            relevant = chunk[chunk.history_id.isin(wanted)]
            for hid, rows in relevant.groupby('history_id', sort=False):
                key = str(hid)
                if key not in witnesses:
                    witnesses[key] = dict(first=rows.iloc[0].to_dict(), steps=0)
                witnesses[key]['steps'] += len(rows)
        (target / 'vertex_daughter_tracks.json').write_text(json.dumps(witnesses, indent=2))
        actual = selector.classify(target, summary)
        assert actual['topology_passed'] == expected, (seed, actual)
        assert not actual['qualified_doublebang'], 'Original seed must not fill a new-seed quota'
        if expected:
            # A real decay at an unrelated history ID cannot be counted as the
            # second bang of this CC tau, even with the same position/energy.
            disconnected = copy.deepcopy(summary)
            for decay in disconnected['tau']['decays']:
                decay['history_id'] = 2**63 + 17
            assert not selector.classify(target, disconnected)['topology_passed']
            # Restore the actual reference record after the negative control.
            selector.classify(target, summary)
        results.append(dict(seed=seed, expected_topology=expected, passed=True,
                            source=str(folder), classification=actual))
        print('SELECTOR VERIFIED', seed, 'physical topology:', expected, flush=True)
    (root / 'preflight/selector_verification.json').write_text(json.dumps(dict(passed=True, results=results), indent=2) + '\n')


if __name__ == '__main__':
    main()

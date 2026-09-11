#!/usr/bin/env python3
"""Summarize per-event memory slopes and read-back physics comparisons."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    root = args.root
    report = json.loads((root / 'regression.json').read_text())
    if not report['complete']:
        raise RuntimeError('Regression still incomplete')
    metrics = []
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    plt.rcParams.update({'font.family': 'serif', 'font.size': 11, 'savefig.dpi': 180})
    for label in report['runs']:
        r = json.loads((root / (label + '_monitor.json')).read_text())
        ends = [x for x in r['checkpoints'] if x['kind'] == 'end_log']
        if not ends:
            raise RuntimeError('No memory checkpoints: ' + label)
        n = ends[-1]['observed_shower']
        late = [x for x in ends if x['observed_shower'] >= max(2, n // 2)]
        x = [v['observed_shower'] for v in late]
        y = [v['rss_mib'] for v in late]
        slope = float(np.polyfit(x, y, 1)[0]) if len(set(x)) > 1 else None
        warm = next(v for v in ends if v['observed_shower'] >= min(8, n))
        samples = r['samples'] + r['checkpoints']
        metrics.append(dict(label=label, n=n, pid=r['pid'],
                            warmup_shower=warm['observed_shower'],
                            rss_at_8_mib=warm['rss_mib'], rss_last_mib=ends[-1]['rss_mib'],
                            peak_rss_mib=max(v['rss_mib'] for v in samples),
                            late_slope_mib_per_shower=slope,
                            process_swap_max_mib=max(v.get('swap_mib', 0) for v in samples),
                            host_available_min_mib=min(v['host_available_mib'] for v in samples),
                            elapsed_s=r['elapsed_s']))
        for ax, backend in zip(axes, ('openmp', 'cuda')):
            if label.startswith(backend + '_photon_'):
                version = 'Before' if '_before_' in label else 'After'
                ax.plot([v['observed_shower'] for v in ends],
                        [v['rss_mib'] for v in ends], label=version,
                        color='#D55E00' if version == 'Before' else '#0072B2')
                ax.set(title=backend.upper() + f': one process, N={n}',
                       xlabel='Shower ordinal', ylabel='Process RSS [MiB]')
                ax.grid(alpha=.2)
    for ax in axes:
        ax.legend()
    fig.suptitle('1 GeV photons, CoREAS + ZHS: event-boundary memory cleanup')
    fig.savefig(root / 'event_memory_before_after.png')
    fig.savefig(root / 'event_memory_before_after.pdf')
    plt.close(fig)
    result = dict(metrics=metrics,
                  all_arrays_equal=all(x['pass'] for x in report['comparisons'].values()),
                  compared_files=sum(len(x['files']) for x in report['comparisons'].values()),
                  note='Approximate end-log checkpoints; allocator caching is not a leak. '
                       'Timing is not an isolated performance benchmark when production runs concurrently.')
    (root / 'memory_review.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Plot the frozen edge-artifact reproduction against the corrected projection."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--before', type=Path, required=True)
    ap.add_argument('--after', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.grid': True, 'grid.alpha': .2, 'savefig.dpi': 170})
    all_rows = []
    data = {}
    for backend in ('cuda', 'openmp'):
        old = pd.read_csv(args.before / f'fixed_track_{backend}.csv')
        new = pd.read_csv(args.after / f'fixed_track_{backend}.csv')
        data[backend] = (old, new)
        keys = ['detection_start_ns', 'window_start_ns', 'window_end_ns',
                'deterministic', 'tiled', 'A_time_ns', 'E_time_ns']
        assert old[keys].equals(new[keys])
        for key, g in new.groupby(['detection_start_ns', 'window_start_ns', 'deterministic', 'tiled']):
            phase, start, fixed, tiled = key
            e = g.dropna(subset=['E_cpu_Vpm'])
            if start != -10:
                assert (e.E_cpu_Vpm == 0).all() and (e.E_kokkos_Vpm == 0).all()
            else:
                # True physical endpoint pulses must remain, not be smoothed away.
                assert np.max(np.abs(e.E_cpu_Vpm)) > 1e-12
            err = np.abs(e.E_cpu_Vpm - e.E_kokkos_Vpm).max()
            assert err <= 4e-18 + 2e-8 * np.abs(e.E_cpu_Vpm).max()
            all_rows.append(dict(backend=backend, detection_start_ns=phase,
                window_start_ns=int(start), deterministic=int(fixed), tiled=int(tiled),
                scalar_peak_Vpm=float(np.abs(e.E_cpu_Vpm).max()),
                kokkos_peak_Vpm=float(np.abs(e.E_kokkos_Vpm).max()),
                scalar_kokkos_max_difference_Vpm=float(err)))
    old, new = data['cuda']
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 6.8), sharex=True)
    for col, phase in enumerate((-5.25, -5.75)):
        def select(frame, start):
            return frame[(frame.detection_start_ns == phase) &
                         (frame.window_start_ns == start) &
                         (frame.deterministic == 0) & (frame.tiled == 0)]
        a, b, w = select(old, 0), select(new, 0), select(new, -10)
        # Crop only for displaying identical physical times; numerical
        # assertions above check ALL bins, including the true endpoint pulses.
        w = w[(w.A_time_ns >= 0) & (w.A_time_ns <= 10)]
        for row, (time, value, scale) in enumerate((('A_time_ns', 'A_cpu_Vsm', 1e21),
                                                    ('E_time_ns', 'E_cpu_Vpm', 1e12))):
            ax = axes[row, col]
            ax.plot(a[time], a[value]*scale, 'o-', color='#D55E00', label='Before')
            ax.plot(b[time], b[value]*scale, 'o-', color='#0072B2', label='Fixed')
            ax.plot(w[time], w[value]*scale, '--', color='.25', label='Wide-window reference')
            ax.set_xlim(-.1, 10.1)
            ax.axvline(0, color='.6', ls=':')
            ax.axvline(10, color='.6', ls=':')
        axes[0, col].set_title('First-point artifact' if col == 0 else 'Last-point artifact')
        axes[1, col].set_xlabel('Arrival time [ns]')
    axes[0, 0].set_ylabel(r'$A_x$ [$10^{-21}$ V s/m]')
    axes[1, 0].set_ylabel(r'$E_x$ [pV/m]')
    axes[0, 0].legend(loc='center', fontsize=9)
    fig.suptitle('Same track: fill complete potential bins before differentiating\nActual scalar ZHS shown; CUDA and OpenMP independently verified', fontsize=11)
    fig.tight_layout()
    fig.savefig(args.output / 'zhs_edge_fix_before_after.png')
    fig.savefig(args.output / 'zhs_edge_fix_before_after.pdf')
    plt.close(fig)
    summary = dict(pass_all=True, cases=len(all_rows), cases_detail=all_rows,
                   scope='Fixed-track window invariance; not an ensemble acceptance')
    (args.output / 'fixed_track_acceptance.json').write_text(json.dumps(summary, indent=2) + '\n')
    print('PASS', len(all_rows), 'fixed-track configurations')


if __name__ == '__main__':
    main()

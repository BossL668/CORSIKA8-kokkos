#!/usr/bin/env python3
"""Compare the two measured radio presets; preserve the strict oracle verdicts."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path('/data/yhlu/CorsikaData/corsika_validation_results/beta5_doublebang_radio_audit_20260912')

def main():
    if not ROOT.is_dir():
        raise RuntimeError('Run numerical analysis on PSR only')
    reports = [json.loads((ROOT / name / 'coreas_summary.json').read_text()) for name in
               ['optical_oracle_precision', 'optical_oracle_run_resolution']]
    rows = []
    for label, report in zip(['Fine reference preset', 'Production preset'], reports):
        rows.append(dict(preset=label, strict_suite_passed=report['passed'],
            comparisons=len(report['cases']),
            max_current_oracle_l2=max(c['absolute_complex_l2'] for c in report['cases']),
            max_coreas_zhs_l2=max(c['zhs_relative_l2'] for c in report['cases']),
            max_backend_l2=max(c['relative_l2'] for c in report['backend_pairs']),
            max_matched_split_l2=max(c['relative_l2'] for c in report['matched_medium_cancellation']),
            failed_case_checks=[dict(case=c['case'], backend=c['backend'],
                checks=[k for k,v in c['checks'].items() if not v])
                for c in report['cases'] if not all(c['checks'].values())]))
    out = ROOT / 'report'
    (out / 'figures').mkdir(parents=True, exist_ok=True)
    result = dict(presets=rows,
        production_current_oracle_gate=5e-3,
        production_paired_gate=1e-5,
        production_current_oracle_passed=rows[1]['max_current_oracle_l2'] < 5e-3,
        production_paired_passed=rows[1]['max_coreas_zhs_l2'] < 1e-5,
        strict_matched_split_gate=1e-10,
        note='Original strict summaries are retained. Production preset fails the strict '
             '1e-9 paired and 1e-10 matched-split criteria. The measured split dependence '
             'is reported as a finite-segment approximation, not relabeled an exact identity. '
             'The independent current comparison excludes the Nyquist bin. '
             'Both models use the same radiation-only real-ray geometrical-optics scope.')
    (out / 'radio_resolution.json').write_text(json.dumps(result, indent=2) + '\n')
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), constrained_layout=True)
    fields = ['max_current_oracle_l2', 'max_coreas_zhs_l2', 'max_matched_split_l2']
    titles = ['Independent current integral', 'CoREAS vs ZHS', 'Artificial matched-medium split']
    for ax, key, title in zip(axes, fields, titles):
        values = [r[key] for r in rows]
        ax.bar([0, 1], values, color=['#5179a3', '#d18a3d'])
        ax.set_yscale('log')
        ax.set_xticks([0, 1]); ax.set_xticklabels(['Fine preset', 'Production preset'])
        ax.set(title=title, ylabel='Maximum relative complex L2 error')
        ax.set_ylim(1e-16, 1e-1); ax.grid(axis='y', alpha=.2)
        for x, value in enumerate(values):
            ax.text(x, max(value, 1e-16)*2, '%.3g' % value, ha='center', fontsize=9)
    axes[0].axhline(5e-3, color='#ae3535', ls='--', lw=1, label='Current-oracle criterion')
    axes[1].axhline(1e-9, color='#777777', ls=':', lw=1, label='Strict fine-preset criterion')
    axes[1].axhline(1e-5, color='#ae3535', ls='--', lw=1, label='Production paired criterion')
    axes[2].axhline(1e-10, color='#777777', ls=':', lw=1, label='Strict identity criterion')
    for ax in axes: ax.legend(fontsize=7, loc='lower right')
    fig.suptitle('Measured numerical accuracy | 18 source geometries, OpenMP and CUDA\n'
                 'Production: 256 MHz, order 12 | Fine: 1 GHz, order 16')
    for ext in ['png', 'pdf']:
        fig.savefig(out / 'figures' / ('radio_resolution.' + ext), dpi=180)
    plt.close(fig)
    print(json.dumps(result))

if __name__ == '__main__':
    main()

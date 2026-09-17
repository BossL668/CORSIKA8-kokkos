#!/usr/bin/env python3
"""Redraw the existing ray-tube diagnostic on PSR; no new ray tracing."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def render_raytube_error(data_dir, output_stem):
    reference = np.genfromtxt(data_dir/'beta5_plane.csv', delimiter=',', names=True)
    beam = np.genfromtxt(data_dir/'radiopropa_beam.csv', delimiter=',', names=True)
    residual = []
    style = {'font.family': 'DejaVu Sans', 'font.size': 12,
             'axes.titlesize': 15, 'axes.labelsize': 12,
             'legend.fontsize': 10, 'pdf.fonttype': 42,
             'axes.spines.top': False, 'axes.spines.right': False,
             'axes.grid': True, 'grid.alpha': .2}
    with plt.rc_context(style):
        fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.7))
        fig.subplots_adjust(left=.100, right=.985, bottom=.235, top=.90, wspace=.38)
        for ax, n1, n2 in zip(axes, [2., 1.0003], [1., 2.]):
            for angle in [0., 10., 20., 29.]:
                mask = (beam['n1'] == n1) & (beam['n2'] == n2) & (beam['angle_deg'] == angle)
                match = (reference['n1'] == n1) & (reference['n2'] == n2) & (reference['angle_deg'] == angle)
                j = reference['jacobian_m2'][match]
                if len(j) != 1:
                    raise ValueError('Expected one beta5 reference per geometry')
                error = abs(beam['jacobian_m2'][mask]/j[0]-1)
                h = beam['h_rad'][mask]
                ax.loglog(h, np.maximum(error, 1.e-16), 'o-', label=str(int(angle))+'°')
                residual.append(dict(n1=n1, angle_deg=angle, h_rad=h.tolist(),
                                     relative_error=error.tolist()))
            ax.set_xlabel(r'Launch-direction perturbation step $h$ (rad)')
            ax.set_ylabel('Relative Jacobian error (dimensionless)\n'
                          r'$\varepsilon_J(h)=\left|\frac{J_{\mathrm{RadioPropa}}(h)}{J_{\mathrm{beta5}}}-1\right|$')
            ax.set_title(r'$n_1=%g\ \longrightarrow\ n_2=%g$' % (n1, n2))
            ax.invert_xaxis()
            ax.legend(title='Incidence from normal', title_fontsize=10, loc='upper right')
        fig.text(.5, .045,
                 r'$J=dA_{\perp}/d\Omega_i$   |   RadioPropa: finite-difference estimate; beta5: analytic reference',
                 ha='center', fontsize=11, color='#333333')
        for ext in ['png', 'pdf']:
            fig.savefig(str(output_stem)+'.'+ext, dpi=180)
        plt.close(fig)
    return residual


if __name__ == '__main__':
    root = Path(__file__).resolve().parents[1]
    rows = render_raytube_error(root/'data', root/'figures/03_public_raytube')
    print('Existing diagnostic redrawn; max error at h=1e-6:',
          max(r['relative_error'][-1] for r in rows))

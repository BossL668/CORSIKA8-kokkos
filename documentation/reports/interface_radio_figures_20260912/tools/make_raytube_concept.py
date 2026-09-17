#!/usr/bin/env python3
"""PSR-only report illustration; reuse saved probe data, do not run a shower."""
from pathlib import Path
import csv
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Polygon

ROOT = Path(__file__).resolve().parents[1]
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 14,
                     'mathtext.fontset': 'dejavusans', 'pdf.fonttype': 42})
blue, orange, ink = '#2576ad', '#dc8735', '#173f60'
fig, axes = plt.subplots(1, 2, figsize=(16, 5.4))
fig.subplots_adjust(left=.025, right=.98, bottom=.16, top=.85, wspace=.13)

ax = axes[0]
ax.set(xlim=(0, 10), ylim=(-2.15, 2.2))
ax.axis('off')
ax.set_title('A ray tube: a narrow bundle of neighboring rays', color=ink, pad=20)
source, near, far = .6, 4.7, 8.8
ax.add_patch(Polygon([(source, 0), (far, 1.6), (far, -1.6)],
                     color=blue, alpha=.09, zorder=0))
for slope in [-1.6, -.8, 0, .8, 1.6]:
    ax.plot([source, far], [0, slope], color=blue, lw=1.6,
            ls='--' if slope == 0 else '-', alpha=.8)
for x, radius in [(near, .8), (far, 1.6)]:
    ax.add_patch(Ellipse((x, 0), .42 * radius, 2 * radius,
                        facecolor=orange, edgecolor=orange, alpha=.3, lw=2))
ax.scatter([source], [0], color=ink, s=45, zorder=5)
ax.text(source-.15, -.48, 'Source', ha='center', color=ink)
ax.text(near, 1.05, r'$A_{\perp}$', ha='center', fontsize=20, color=orange)
ax.text(far, 1.82, r'$4A_{\perp}$', ha='center', fontsize=20, color=orange)
for x, label in [(near, r'$R$'), (far, r'$2R$')]:
    ax.annotate('', (x, -1.87), (source, -1.87),
                arrowprops={'arrowstyle': '<->', 'color': '#777', 'lw': 1})
    ax.text(x, -2.14, label, ha='center', fontsize=16)
ax.text(2, 1.77, r'$d\Omega_i$: fixed angular patch', fontsize=14, color=blue)
ax.text(near, -.1, r'$P$', ha='center', fontsize=18, color=ink)
ax.text(far, -.1, r'$P$', ha='center', fontsize=18, color=ink)

ax = axes[1]
ax.set(xlim=(-.6, 3.5), ylim=(-1.38, 2.45))
ax.set_aspect('equal', adjustable='box')
ax.axis('off')
ax.set_title('Refraction changes how the bundle spreads', color=ink, pad=20)
ax.axhspan(-1.38, 0, facecolor='#eee6d8', zorder=0)
ax.axhline(0, color='#80725d', lw=1.5)
ax.text(2.15, -.55, r'$n_1=2$', color='#735732')
ax.text(-.42, .65, r'$n_2=1$', color=blue)
ax.text(2.85, -.16, 'Interface', ha='center', fontsize=12)
s = np.array([0., -1.])
theta = np.deg2rad(20.)
out_theta = np.arcsin(2*np.sin(theta))
normal = np.array([np.sin(out_theta), np.cos(out_theta)])
center = np.array([np.tan(theta) + 1.8*np.tan(out_theta), 1.8])
ends, hits = [], []
for delta in [-3., 0., 3.]:
    angle = theta + np.deg2rad(delta)
    hit = np.array([np.tan(angle), 0.])
    out = np.arcsin(2*np.sin(angle))
    direction = np.array([np.sin(out), np.cos(out)])
    end = hit + direction * np.dot(center-hit, normal) / np.dot(direction, normal)
    hits.append(hit); ends.append(end)
    ax.plot([s[0], hit[0], end[0]], [s[1], hit[1], end[1]],
            color=blue, lw=2 if delta == 0 else 1.3,
            ls='--' if delta == 0 else '-', zorder=3)
ax.add_patch(Polygon([s, hits[0], ends[0], ends[2], hits[2]],
                     color=blue, alpha=.1, zorder=1))
tangent = np.array([normal[1], -normal[0]])
edge1, edge2 = center-.55*tangent, center+.55*tangent
ax.plot([edge1[0], edge2[0]], [edge1[1], edge2[1]], color=orange, lw=5, alpha=.65)
corner = np.array([center-.14*normal, center-.14*normal+.14*tangent,
                   center+.14*tangent])
ax.plot(corner[:, 0], corner[:, 1], color=orange, lw=1.2)
ax.scatter([s[0]], [s[1]], s=40, color=ink, zorder=4)
ax.text(-.03, -1.28, 'Source', ha='center', color=ink, fontsize=12)
ax.annotate(r'$d\Omega_i$', (.10, -.69), (-.48, -.50), color=blue,
            fontsize=18, arrowprops={'arrowstyle': '->', 'color': blue})
ax.annotate(r'$dA_{\perp}$', center, (2.78, 2.07), color=orange,
            fontsize=20, arrowprops={'arrowstyle': '->', 'color': orange})
ax.text(-.40, 2.05, r'$J=\dfrac{dA_{\perp}}{d\Omega_i}$', fontsize=24, color=ink)
ax.text(1.2, -1.12, 'Sections are perpendicular\nto the central ray', fontsize=12, color=ink)

fig.text(.25, .080, r'Same lossless medium: $I=P/A_{\perp}$,  $|E|\propto A_{\perp}^{-1/2}$',
         ha='center', fontsize=16, color=ink)
fig.text(.25, .025, 'Distance x2  ->  Area x4  ->  Intensity /4  ->  Field /2',
         ha='center', fontsize=13, color=ink)
fig.text(.75, .080, 'Geometric spreading; Fresnel transmission and absorption are separate',
         ha='center', fontsize=12, color=ink)
fig.text(.75, .025, 'Illustration: 2D views of 3D bundles; angular width enlarged',
         ha='center', fontsize=12, color='#555')
for ext in ['png', 'pdf']:
    fig.savefig(ROOT/'figures'/('12_raytube_concept.'+ext), dpi=160)
plt.close(fig)

# Recompute only the published diagnostic from the existing CSVs.
with (ROOT/'data/beta5_plane.csv').open() as f:
    reference = list(csv.DictReader(f))
with (ROOT/'data/radiopropa_beam.csv').open() as f:
    beam = list(csv.DictReader(f))
rows = []
for b in beam:
    matches = [a for a in reference if
               all(float(a[k]) == float(b[k]) for k in ['n1', 'n2', 'angle_deg'])]
    if len(matches) != 1:
        raise ValueError('Expected one beta5 reference per probe sample')
    j = float(matches[0]['jacobian_m2'])
    rows.append({**{k: float(b[k]) for k in ['n1','n2','angle_deg','h_rad']},
                 'relative_error': abs(float(b['jacobian_m2'])/j-1)})
assert len(rows) == 32
metrics = {
    'operation': 'Illustration and re-reading existing CSVs on PSR; no new shower or ray tracing',
    'definition': 'J = dA_perpendicular / dOmega_source; screen normal is the central transmitted ray',
    'error_definition': 'abs(J_RadioPropa(h)/J_beta5 - 1)',
    'perturbation': 'normalize(k +/- h*u), normalize(k +/- h*v); central differences; actual rotation atan(h)',
    'max_relative_error_h1e6': max(r['relative_error'] for r in rows if r['h_rad'] == 1e-6),
    'samples': rows,
    'input_sha256': {name: hashlib.sha256((ROOT/'data'/name).read_bytes()).hexdigest()
                     for name in ['beta5_plane.csv','radiopropa_beam.csv']}}
(ROOT/'data/raytube_explanation_metrics.json').write_text(json.dumps(metrics, indent=2)+'\n')
print(json.dumps({'samples': len(rows), 'max_error_h1e6': metrics['max_relative_error_h1e6']}))

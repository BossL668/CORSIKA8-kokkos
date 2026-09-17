#!/usr/bin/env python3
"""Render the paired CUDA trace with a separate row for every final grid copy."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
def plot(root):
    path=root.parent/'interface-radio-pair-20260912/profile-app/electron_rock_resident.jsonl'
    events=[json.loads(line) for line in path.read_text().splitlines()]
    kernels=[e for e in events if e['event']=='gpu_kernel' and any(k in e['name'] for k in ['CoreasEndpointKernel','AccumulateKernel'])]
    t0=min(e['start'] for e in kernels);end=max(e['end'] for e in kernels)
    fig,ax=plt.subplots(figsize=(11.8,3.8),constrained_layout=True)
    for e in kernels:
        coreas='CoreasEndpointKernel' in e['name']
        ax.broken_barh([((e['start']-t0)/1.e9,(e['end']-e['start'])/1.e9)],
                      ((4 if coreas else 3)-.20,.40),facecolors='#1769aa' if coreas else '#d55e00')
    labels={'interface_radio_moments':2,'interface_radio_coreas_regularized_moments':1,'interface_radio_zhs_moments':0}
    downloads=[e for e in events if e['event']=='copy_begin' and e['src_label'] in labels and e['dst_space']=='Host']
    assert len(downloads)==3 and all(e['time']>end for e in downloads)
    for e in downloads:
        ax.scatter([(e['time']-t0)/1.e9],[labels[e['src_label']]],color='#16826c',marker='v',s=65,zorder=5)
    ax.axvline((end-t0)/1.e9,color='gray',ls=':',label='Last radio kernel ends')
    ax.set(yticks=range(5),yticklabels=['ZHS A grid → host','CoREAS auxiliary A → host','CoREAS E grid → host','ZHS kernel','CoREAS kernel'],
           xlabel='Seconds from first radio GPU kernel',title='Actual CUDA execution; three separate final downloads')
    ax.set_xlim(-.3,(end-t0)/1.e9+.7);ax.grid(alpha=.2,axis='x')
    ax.spines['top'].set_visible(False);ax.spines['right'].set_visible(False)
    fig.savefig(root/'figures/17_gpu_residency.png',dpi=180,bbox_inches='tight')
    fig.savefig(root/'figures/17_gpu_residency.pdf',bbox_inches='tight');plt.close(fig)
if __name__=='__main__':
    plt.rcParams.update({'font.size':12,'axes.titlesize':14,'font.family':'DejaVu Sans','pdf.fonttype':42})
    plot(Path(__file__).resolve().parents[1])

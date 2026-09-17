#!/usr/bin/env python3
"""Draw actual CUPTI GPU kernels and final radio-grid downloads."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path('/data/yhlu/CorsikaData/corsika_validation_results/beta5_doublebang_radio_audit_20260912')

def main():
    summaries=json.loads((ROOT/'residency_summary.json').read_text())
    names=['magnetic_cuda_electron_on','magnetic_cuda_tau_on']
    patterns=['InterfaceStepKernel','CoreasEndpointKernel','AccumulateKernel']
    grids={'interface_radio_moments','interface_radio_coreas_regularized_moments','interface_radio_zhs_moments'}
    colors=['#467cad','#db8742','#53946b']
    fig,axes=plt.subplots(2,1,figsize=(12,6.4),constrained_layout=True)
    for ax,name in zip(axes,names):
        assert all(summaries[name]['checks'].values())
        events=[];downloads=[]
        with (ROOT/'runs'/name/'cupti.jsonl').open() as stream:
            for line in stream:
                e=json.loads(line)
                if e['event']=='gpu_kernel':
                    for row,pattern in enumerate(patterns):
                        if pattern in e['name']:events.append((row,e['start'],e['end']))
                if e['event']=='copy_begin' and e.get('src_label') in grids and e.get('dst_space')=='Host' and e.get('bytes',0)>0:
                    downloads.append(e['time'])
        origin=min(e[1] for e in events)
        for row in range(3):
            intervals=[((start-origin)*1e-9,(end-start)*1e-9) for r,start,end in events if r==row]
            ax.broken_barh(intervals,(row-.25,.5),facecolors=colors[row])
        ax.scatter([(t-origin)*1e-9 for t in downloads],[3]*len(downloads),color='#ae3838',marker='v',s=35,zorder=4)
        ax.set_yticks(range(4));ax.set_yticklabels(['EM transport','CoREAS','ZHS','Final grid D2H'])
        counts=summaries[name]['gpu_kernel_counts'];label='1 GeV electron' if 'electron' in name else '1 TeV tau'
        title=label+' | kernels: transport %d, CoREAS %d, ZHS %d' % (counts['transport'],counts['CoREAS'],counts['ZHS'])
        ax.set(xlabel='Time since first EM transport kernel (s)',ylim=(-.6,3.6),title=title)
        ax.grid(axis='x',alpha=.2)
    fig.suptitle('Actual CUDA trace: transport and both radio algorithms execute on the GPU\n'
                 'Final radio-grid downloads are marked; diagnostic/control copies are omitted from this view',fontsize=11)
    out=ROOT/'report/figures';out.mkdir(parents=True,exist_ok=True)
    for ext in ['png','pdf']:fig.savefig(out/('cuda_residency.'+ext),dpi=180)
    plt.close(fig)
    print('Saved current CUDA residency diagnostic')

if __name__=='__main__':main()

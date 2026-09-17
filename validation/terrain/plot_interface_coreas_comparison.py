#!/usr/bin/env python3
"""Plot recorded c2 rejection versus accepted c4; no new simulation."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from analyze_interface_coreas import reconstruct

def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--phase',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    old=a.phase/'fixtures-c2/CUDA_boundary_CoREAS';new=a.phase/'fixtures-c4/CUDA_boundary_CoREAS';zhs=a.phase/'fixtures-c4/CUDA_boundary_ZHS'
    assert (old/'tracks.csv').read_bytes()==(new/'tracks.csv').read_bytes()==(zhs/'tracks.csv').read_bytes()
    fields=[]
    for root in (old,new,zhs):
        c=json.loads((root/'config.json').read_text());f,_,field=reconstruct(root,c);fields.append(field[0])
    norms=[np.linalg.norm(v[:,1:],axis=0) for v in fields];denom=np.maximum(norms[2],max(norms[2])*1.e-12)
    residuals=[np.linalg.norm(v[:,1:]-fields[2][:,1:],axis=0)/denom for v in fields[:2]]
    fig,axes=plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
    for i,label,style,color in ((2,'ZHS','-','black'),(0,'Rejected c2',':','tab:red'),(1,'Adapted c4 CoREAS','--','tab:blue')):
        axes[0].loglog(f[1:]/1.e6,norms[i],style,color=color,label=label)
    axes[0].set(xlabel='Frequency (MHz)',ylabel='|E(f)| (V s/m)',title='Identical crossing track and optical inputs');axes[0].legend(fontsize=8)
    for values,label,color in zip(residuals,('Rejected c2 / ZHS','Adapted c4 / ZHS'),('tab:red','tab:blue')):
        axes[1].loglog(f[1:]/1.e6,np.maximum(values,1.e-16),color=color,label=label)
    axes[1].set(xlabel='Frequency (MHz)',ylabel='Relative complex residual',title='Absolute phase and amplitude; no fitted shifts');axes[1].legend(fontsize=8)
    for ax in axes:ax.grid(alpha=.25,which='both')
    fig.savefig(a.output/'boundary_before_after.png',dpi=170);fig.savefig(a.output/'boundary_before_after.pdf');plt.close(fig)
    np.savez_compressed(a.output/'boundary_before_after.npz',frequency_Hz=f,rejected_c2=fields[0],coreas_c4=fields[1],zhs=fields[2])

if __name__=='__main__':main()

#!/usr/bin/env python3
"""Prepare read-only transmission diagnostics on PSR; production keeps running."""
import os
os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
from pathlib import Path
import datetime
import hashlib
import json
import shlex
import shutil
import socket
import subprocess
import numpy as np
import pandas as pd
import yaml

assert socket.gethostname()=='psrpku2025'
BASE=Path('/data/yhlu/CorsikaData/corsika_validation_results')
os.environ.update(json.loads((BASE/'beta5_doublebang_100PeV_screen_thin1e3_130cores_20260916/campaign.json').read_text())['runtime_environment'])
ROOT=BASE/'beta5_1PeV_transmission_diagnostic_20260916'
ROOT.mkdir(exist_ok=True)
R1=BASE/'beta5_doublebang_1PeV_thin1e4_queue4M_20260915'
TAG='openmp_SiO2_21CMA80_1PeV_seed22309'
run=R1/'runs'/TAG
summary=yaml.safe_load((run/'output/terrain_run.yaml').read_text())
config=json.loads((run/'output/radio/ZHS/config.json').read_text())
(ROOT/'config.json').write_text(json.dumps(config))
mesh=Path(summary['scene']['geometry']['mesh_path'])
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(name,value):(ROOT/name).write_text(json.dumps(value,indent=2)+'\n')
oldroot=BASE/'beta5_material_doublebang_openmp256_20260913'
old=[]
for p in sorted((oldroot/'runs').glob('*/output/terrain_run.yaml')):
    s=yaml.safe_load(p.read_text());r=s['radio_result']['ZHS']
    old.append(dict(tag=p.parents[1].name,transmitted_paths=r['transmitted_paths'],
        direct_paths=r['direct_paths'],errors=r['errors'],complete=s['complete']))
save('historical_transmission.json',old)
if not (ROOT/'sources.json').exists():
    origin=np.array(summary['position_m']);direction=np.array(summary['direction']);direction/=np.linalg.norm(direction)
    sv=(np.array(summary['neutrino']['interactions'][0]['position_enu_m'])-origin)@direction
    best={};total=0;low=np.ones(3)*np.inf;high=-low
    columns=['step','pdg','medium','weight','E0_GeV','history_id','x0_m','y0_m','z0_m','x1_m','y1_m','z1_m']
    for f in pd.read_csv(run/'output/terrain/tracks.csv.gz',usecols=columns,dtype={'history_id':'uint64'},chunksize=200000):
        total+=len(f)
        a=f[(f.medium=='rock')&(abs(f.pdg)==11)].copy()
        if not len(a):continue
        points=.5*(a[['x0_m','y0_m','z0_m']].to_numpy()+a[['x1_m','y1_m','z1_m']].to_numpy())
        s=(points-origin)@direction
        low=np.minimum(low,points.min(axis=0));high=np.maximum(high,points.max(axis=0))
        criteria={f'axis_{target:g}m':-abs(s-sv-target) for target in [0,1.7,5,17,18.4]}
        criteria.update(highest_weighted_energy=a.E0_GeV.to_numpy()*a.weight.to_numpy(),
            min_east=-points[:,0],max_east=points[:,0],min_up=-points[:,2],max_up=points[:,2])
        for label,score in criteria.items():
            i=int(np.argmax(score));value=float(score[i])
            if label not in best or value>best[label]['score']:
                row=a.iloc[i]
                best[label]=dict(source_id=label,score=value,source_m=points[i].tolist(),
                    step=int(row.step),history_id=int(row.history_id),pdg=int(row.pdg),
                    energy_GeV=float(row.E0_GeV),weight=float(row.weight),
                    distance_from_vertex_m=float(s[i]-sv),event=TAG,kind='current_actual_rock_electron')
    assert total==summary['diagnostics']['steps']
    sources=list(best.values())
    for seed in [946,3605]:
        event=f'openmp_silica_SiO2_seed{seed}'
        file=oldroot/'runs'/event/'output/terrain/tracks.csv'
        if not file.exists():file=file.with_suffix('.csv.gz')
        for f in pd.read_csv(file,usecols=columns,dtype={'history_id':'uint64'},chunksize=50000):
            a=f[(f.medium=='rock')&(abs(f.pdg)==11)]
            if len(a):
                row=a.iloc[0];point=[.5*(float(row[f'{axis}0_m'])+float(row[f'{axis}1_m'])) for axis in 'xyz']
                sources.append(dict(source_id=f'old_seed{seed}',source_m=point,step=int(row.step),history_id=int(row.history_id),
                    pdg=int(row.pdg),energy_GeV=float(row.E0_GeV),weight=float(row.weight),event=event,kind='historical_actual_rock_electron'))
                break
    save('sources.json',sources)
    save('sampling.json',dict(all_tracks_scanned=total,electron_midpoint_bounds_m=[low.tolist(),high.tolist()],
        mesh_sha256=sha(mesh),source_selection='Real rock electron/positron segment midpoints near longitudinal targets, extrema and highest weighted energy; plus two actual historical rock segments. Not an exhaustive audit of every emitting segment.'))
else:sources=json.loads((ROOT/'sources.json').read_text())
observers={x['name']:x['position_enu_m'] for x in summary['scene']['radio']['observers']}
queries=[]
for source in sources:
    names=list(observers) if source['source_id']=='axis_1.7m' else ['E18','E20','N10','W20','S10']
    for name in names:
        queries.append(dict(label=source['source_id']+'_'+name,source_id=source['source_id'],source_m=source['source_m'],observer=name,observer_m=observers[name]))
    point=list(source['source_m']);point[2]+=1500
    queries.append(dict(label=source['source_id']+'_overhead_control',source_id=source['source_id'],source_m=source['source_m'],observer='overhead_control',observer_m=point))
save('queries.json',queries)
build=BASE/'beta5_doublebang_radio_audit_20260912/build-openmp/tests/accelerator'
flags=(build/'CMakeFiles/testKokkosInterfaceRadioBvh.dir/flags.make').read_text()
includes=shlex.split(next(x.split('=',1)[1] for x in flags.splitlines() if x.startswith('CXX_INCLUDES =')))
compiler='/home/yuhanglu/miniconda/envs/corsika_venv/bin/g++'
native=BASE/'beta5_interface_batch_performance_20260915/source'
compile_cmd=[compiler,'-O3','-std=c++17','-fopenmp','-I'+str(native)]+includes+['-c',str(ROOT/'TransmissionAudit.cpp'),'-o',str(ROOT/'probe.o')]
link=shlex.split((build/'CMakeFiles/testKokkosInterfaceRadioBvh.dir/link.txt').read_text())
link=[str(ROOT/'probe.o') if x=='CMakeFiles/testKokkosInterfaceRadioBvh.dir/testKokkosInterfaceRadioBvh.cpp.o' else x for x in link]
link[link.index('-o')+1]=str(ROOT/'probe')
for label,command in [('compile',compile_cmd),('link',link)]:
    with (ROOT/(label+'.log')).open('w') as log:subprocess.run(command,cwd=build,stdout=log,stderr=subprocess.STDOUT,check=True)
save('build_commands.json',dict(compile=compile_cmd,link=link,cwd=str(build),native_propagation_sha256=sha(native/'corsika/modules/radio/interface/Propagation.hpp')))
save('provenance.json',dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),mesh=str(mesh),mesh_sha256=sha(mesh),
    binary_sha256=sha(ROOT/'probe'),queries=len(queries),sources=len(sources),production_files_changed=False))
print('PREPARED',len(queries),'queries;',len(sources),'source points',flush=True)
with (ROOT/'native.log').open('w') as log:
    subprocess.run([str(ROOT/'probe'),str(mesh),str(ROOT/'config.json'),str(ROOT/'queries.json'),str(ROOT/'native.json')],stdout=log,stderr=subprocess.STDOUT,check=True)
print('NATIVE_FINISHED',flush=True)

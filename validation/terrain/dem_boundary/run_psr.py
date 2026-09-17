#!/usr/bin/env python3
"""Complete, small boundary showers at the requested air cuts/thinning on PSR."""
import argparse
import copy
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import struct
import subprocess
import time
import yaml

BASE = Path('/data/yhlu/CorsikaData/corsika_validation_results')
OLD = BASE / 'beta5_material_models_20260913'
PREVIOUS = BASE / 'beta5_SiO2_high_energy_aircuts_thin1e-5_20260914'

def save(path, data):
    path.write_text(json.dumps(data, indent=2) + '\n')

def rows(path):
    with gzip.open(path, 'rt') as f:
        return list(csv.DictReader(f))

def audit(folder):
    summary = yaml.safe_load((folder / 'output/terrain_run.yaml').read_text())
    d = summary['diagnostics']
    assert summary['complete'] and not d['csv_truncated'] and not d['deposition_csv_truncated']
    tracks = rows(folder / 'output/terrain/tracks.csv.gz')
    exits = rows(folder / 'output/terrain/domain_exits.csv.gz')
    deposits = rows(folder / 'output/terrain/deposits.csv.gz')
    assert len(tracks) == d['steps']
    assert [int(t['step']) for t in tracks] == list(range(1, len(tracks) + 1))
    assert len(exits) == d['domain_exits']
    assert len({x['history_id'] for x in exits}) == len(exits), 'duplicate terminal histories'
    last = {}
    for t in tracks:
        last[t['history_id']] = t
        if d['dem_coverage_boundary'] and folder.name.find('real_') < 0:
            for end in ['0', '1']:
                assert abs(float(t['x' + end + '_m'])) <= 10 + 2e-8
                assert abs(float(t['y' + end + '_m'])) <= 10 + 2e-8
    for x in exits:
        t = last[x['history_id']]
        assert t['pdg'] == x['pdg'] and t['parent_history_id'] == x['parent_history_id']
        for k in ['x', 'y', 'z']:
            assert abs(float(t[k+'1_m'])-float(x[k+'_m'])) < 1e-10, 'missing last track segment'
        assert abs(float(t['E1_GeV'])-float(x['total_GeV'])) < 1e-10
        assert float(t['weight']) == float(x['weight'])
    energy = sum(float(x['weight'])*float(x['total_GeV']) for x in exits)
    assert abs(energy-d['domain_escaped_total_GeV']) <= 1e-10*max(1.,energy)
    deposition = sum(float(x['weighted_deposited_GeV']) for x in deposits)
    assert abs(deposition-d['deposited_energy_GeV']) <= 1e-10*max(1.,deposition)
    ledger = summary['energy_ledger']
    assert abs(ledger['unexplained_over_initial']) < 1e-9, ledger
    accelerator = summary.get('accelerator', {})
    assert accelerator.get('pending_particles', 0) == 0
    radio = summary.get('radio_result', {})
    observer_names=[o['name'] for o in summary['scene']['radio']['observers']]
    if folder.name.endswith('_80stations'):
        assert len(observer_names)==80 and set(observer_names)=={arm+('%02d'%i) for arm in 'EWNS' for i in range(1,21)}
    if radio:
        assert radio['complete'] and radio['errors'] == 0 and radio['out_of_window'] == 0
        charged = sum(int(t['pdg']) not in [22, 12, -12, 14, -14, 16, -16, 2112, -2112] for t in tracks)
        assert radio['device_tracks']+radio['cpu_tracks'] == charged, 'radio lost a terminal charged segment'
        assert radio['track_observer_pairs']==charged*len(observer_names), 'observer-source pairs missing'
    return dict(complete=True, steps=len(tracks), exits=len(exits), deposited_GeV=deposition,
                escaped_GeV=energy, unexplained_over_initial=ledger['unexplained_over_initial'],
                accelerated_steps=d['accelerated_steps'], radio=radio, accelerator=accelerator,observer_names=observer_names)

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--mode',choices=['openmp','cuda'],required=True)
    p.add_argument('--only',default='');a=p.parse_args()
    assert socket.gethostname()=='psrpku2025'
    root=a.root;scenes=root/'scenes';scenes.mkdir(exist_ok=True)
    report=root/'report';report.mkdir(exist_ok=True)
    mesh=scenes/'flat_SiO2_test.ply'
    vertices=[(-10,-10,-10),(10,-10,-10),(10,10,-10),(-10,10,-10),(-10,-10,0),(10,-10,0),(10,10,0),(-10,10,0)]
    faces=[(0,2,1),(0,3,2),(4,5,6),(4,6,7),(0,1,5),(0,5,4),(1,2,6),(1,6,5),(2,3,7),(2,7,6),(3,0,4),(3,4,7)]
    mesh.write_bytes(b'ply\nformat binary_little_endian 1.0\nelement vertex 8\nproperty double x\nproperty double y\nproperty double z\nelement face 12\nproperty list uchar int vertex_indices\nend_header\n'+
        b''.join(struct.pack('<ddd',*v) for v in vertices)+b''.join(struct.pack('<Biii',3,*f) for f in faces))
    original=yaml.safe_load((PREVIOUS/'scenes/silica_SiO2.yaml').read_text())
    scene=copy.deepcopy(original);scene['geometry'].update(mesh_path=str(mesh),rock_reference_enu_m=[0,0,-1])
    scene['provenance']['mesh_sha256']=hashlib.sha256(mesh.read_bytes()).hexdigest()
    scene['radio'].update(enabled=False,samples=4096,sample_rate_GHz=.256,memory_MiB=512)
    scene['radio']['observers']=[dict(name='BOUNDARY_TEST',position_enu_m=[0,0,10])]
    # The scene uses sample_rate_Hz in some historical adapters; preserve the
    # accepted card's sampling convention and reduce only the reception grid.
    cases=[]
    def add(tag,primary,energy,position,boundary=True,backend='kokkos',radio=False,field='none',scheduler='resident',window=10000):
        cases.append(dict(tag=tag,primary=primary,energy=energy,position=position,boundary=boundary,backend=backend,radio=radio,field=field,scheduler=scheduler,window=window))
    add('rock_electron_exit','electron',.02,[9.999,0,-1],radio=True)
    add('rock_electron_exit_batched','electron',.02,[9.999,0,-1],radio=True,scheduler='batched')
    add('air_electron_exit','electron',1.,[9.999,0,10],radio=True)
    add('air_electron_field_exit','electron',1.,[9.999,0,10],radio=True,field='igrf14')
    add('contained_on','electron',.05,[0,0,-1])
    add('contained_off','electron',.05,[0,0,-1],boundary=False)
    add('profile_crossing','electron',.1,[9.98,0,-1])
    if a.mode=='openmp':
        for primary,energy in [('electron',.02),('positron',.02),('photon',.02),('mu_minus',10.),('proton',10.),('pi_minus',10.),('tau_minus',100.),('nu_tau',10000.)]:
            add('cpu_'+primary,primary,energy,[9.999999,0,-1],backend='proposal')
        add('cpu_air_field','electron',1.,[9.999,0,10],backend='proposal',field='igrf14')
        add('cpu_contained_on','electron',.05,[0,0,-1],backend='proposal')
        add('cpu_contained_off','electron',.05,[0,0,-1],backend='proposal',boundary=False)
    if (report/'perimeter.csv').exists():
        import numpy as np
        with (report/'perimeter.csv').open() as f:edges=list(csv.DictReader(f))
        edge=max(edges,key=lambda e:float(e['x0_m'])+float(e['x1_m']))
        x=.5*(float(edge['x0_m'])+float(edge['x1_m']));y=.5*(float(edge['y0_m'])+float(edge['y1_m']))
        dx=float(edge['y1_m'])-float(edge['y0_m']);dy=float(edge['x0_m'])-float(edge['x1_m']);length=(dx*dx+dy*dy)**.5
        direction=[dx/length,dy/length,0.];position=[x-direction[0]*.001,y-direction[1]*.001,10000.]
        add('real_air_electron_80stations','electron',1.,position,radio=True,field='igrf14');cases[-1]['direction']=direction
        if a.mode=='openmp':
            add('real_cpu_nu_tau_80stations','nu_tau',10000.,position,backend='proposal');cases[-1]['direction']=direction
        with Path(original['geometry']['mesh_path']).open('rb') as f:
            header=[]
            while True:
                line=f.readline().decode().strip();header.append(line)
                if line=='end_header':break
            assert 'property double x' in header
            count=int(next(s for s in header if s.startswith('element vertex ')).split()[-1])
            vertices=np.frombuffer(f.read(count*24),dtype='<f8').reshape(-1,3)
        heights=[]
        for k in [0,1]:
            near=(abs(vertices[:,0]-float(edge['x%d_m'%k]))<1e-8)&(abs(vertices[:,1]-float(edge['y%d_m'%k]))<1e-8)
            heights.append(float(vertices[near,2].max()))
        position=position[:2]+[sum(heights)/2-.1]
        add('real_rock_electron_80stations','electron',.02,position,radio=True);cases[-1]['direction']=direction
    results={}
    summary_file=report/('runs_'+a.mode+'.json')
    if summary_file.exists():results=json.loads(summary_file.read_text())
    for case in cases:
        tag=case['tag']
        if a.only and tag not in a.only.split(','):continue
        if results.get(tag,{}).get('complete'):continue
        folder=root/'runs'/a.mode/tag
        if folder.exists():
            # Keep failed attempts, including their exact input and exit status.
            failed=root/'failed_attempts';failed.mkdir(exist_ok=True)
            folder.rename(failed/(a.mode+'_'+tag+'_'+str(time.time_ns())))
        folder.mkdir(parents=True)
        local=copy.deepcopy(original if tag.startswith('real_') else scene)
        if tag.startswith('real_'):
            assert json.loads((report/'stations_80_audit.json').read_text())['count']==80
            local['radio'].update(enabled=False,samples=32768 if a.mode=='openmp' else 16384,memory_MiB=8192 if a.mode=='openmp' else 3072)
            local['radio']['observers']=yaml.safe_load((root/'stations/observers.yaml').read_text())['radio']['observers']
        local['geometry']['transport_boundary']={'type':'dem_coverage' if case['boundary'] else 'none'}
        scene_file=folder/'scene.yaml';scene_file.write_text(yaml.safe_dump(local,sort_keys=False))
        command=['taskset','-c','0-255',str(root/('build-'+a.mode)/'c8_terrain_cascade'),
            '--scene',str(scene_file),'--output',str(folder/'output'),'--primary',case['primary'],'--energy-GeV',str(case['energy']),
            '--position-m']+list(map(str,case['position']))+['--direction']+list(map(str,case.get('direction',[1,0,0])))+['--seed','39101',
            '--emcut-GeV','.0005','--hadcut-GeV','.3','--mucut-GeV','.3','--emthin','1e-5',
            '--transport-window-ns',str(case['window']),'--magnetic-field',case['field'],
            '--em-backend',case['backend'],'--threads','256','--batch','4096','--device-memory-MiB',('8192' if a.mode=='openmp' else '4096') if tag.startswith('real_') else '2048',
            '--track-row-limit','1000000','--transport-step-limit','1000000','--aux-cache',str(OLD/'auxiliary'),
            '--energy-ledger','--em-scheduler',case['scheduler'],'--resident-capacity','65536','--compress-terrain-csv']
        if case['radio']:command+=['--radio']
        save(folder/'command.json',command);start=time.monotonic()
        env=os.environ.copy();env['CORSIKA_DATA']=str(OLD/'source/modules/data')
        print('START',a.mode,tag,flush=True)
        with (folder/'run.log').open('w') as log:
            process=subprocess.Popen(command,cwd=folder,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            try:code=process.wait(timeout=600)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGTERM)
                try:process.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
                save(folder/'status.json',dict(complete=False,timeout=True,wall_s=time.monotonic()-start));raise
        save(folder/'status.json',dict(exit_code=code,wall_s=time.monotonic()-start))
        if code:raise RuntimeError('Simulation failed: '+str(folder))
        result=audit(folder);result['wall_s']=time.monotonic()-start;result['case']=case
        results[tag]=result;save(summary_file,results)
        print('PASS',a.mode,tag,result['steps'],'steps',result['exits'],'exits',flush=True)
    # An inactive boundary must preserve complete stochastic profiles exactly.
    for prefix in ['', 'cpu_']:
        tags=[prefix+'contained_on',prefix+'contained_off']
        if all(t in results for t in tags):
            for filename in ['tracks.csv.gz','deposits.csv.gz','window_survivors.csv.gz']:
                contents=[gzip.open(root/'runs'/a.mode/t/'output/terrain'/filename,'rb').read() for t in tags]
                assert contents[0]==contents[1], (prefix,filename,'contained profile changed')
            assert all(results[t]['exits']==0 for t in tags)
            save(report/(a.mode+'_'+prefix+'contained_profile.json'),dict(exact_decompressed_csv_match=True,files=3))

if __name__=='__main__':main()

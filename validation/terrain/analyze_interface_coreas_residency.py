#!/usr/bin/env python3
"""CUPTI evidence for the independent interface CoREAS device buffers and both resident grids."""
import argparse
from collections import Counter
import json
from pathlib import Path
import yaml
import numpy as np

def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--profiles',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--acceptance',type=Path,required=True)
    args=p.parse_args();rows=[]
    accepted=json.loads(args.acceptance.read_text())
    for run in json.loads(args.profiles.read_text()):
        events=[json.loads(line) for line in Path(run['trace']).read_text().splitlines()]
        command=run['command'];root=Path(command[command.index('--output')+1])
        summary=yaml.safe_load((root/'terrain_run.yaml').read_text());radio=summary['radio_result']
        config=json.loads((root/'radio/config.json').read_text())
        kernels=[e for e in events if e['event']=='gpu_kernel' and 'CoreasEndpointKernel' in e['name']]
        labels=Counter(e['name'] for e in events if e['event']=='for_begin')
        copies=[e for e in events if e['event']=='copy_begin' and e['src_label']!='(none)' and e['bytes']]
        grids=['interface_radio_moments','interface_radio_coreas_regularized_moments']
        moments=[e for e in copies if e['src_label'] in grids and e['dst_space']=='Host']
        cpu=[e for e in copies if e['dst_label']=='interface_radio_cpu_tracks' and e['src_space']=='Host' and e['dst_space']=='Cuda']
        resident_tracks=['interface_radio_transport_tracks','interface_radio_device_pending','interface_radio_cpu_pending']
        buffers=['interface_radio_coreas_regularized_moments','interface_radio_moments','interface_radio_counters','interface_radio_observers','interface_radio_index_table','interface_radio_optical_breaks']+resident_tracks
        allocations=[e for e in events if e['event']=='allocate' and e['label'] in buffers and e['space']=='Cuda']
        finals=[e for e in events if e['event']=='finalize']
        expected_bytes=config['samples']*len(config['observers'])*6*(config['moment_order']+1)*8
        checks=dict(accepted_run=all(run['checks'].values()),
            actual_radio_cuda_kernels=len(kernels)==radio['wavefronts']>0,
            labeled_launches=labels['interface_radio_coreas_endpoints']==len(kernels),
            device_buffers=all(any(a['label']==label for a in allocations) for label in buffers),
            no_device_tracks_roundtrip=not any(any(label in [e['src_label'],e['dst_label']] for label in resident_tracks) and e['src_space']!=e['dst_space'] for e in copies),
            device_buffer_appends=any(e['dst_label']=='interface_radio_device_pending' and e['src_label']=='interface_radio_transport_tracks' and e['src_space']==e['dst_space']=='Cuda' for e in copies),
            one_final_moment_download=len(moments)==2 and set(e['src_label'] for e in moments)==set(grids) and all(e['bytes']==expected_bytes and e['time']>max(k['end'] for k in kernels) for e in moments),
            no_moment_upload_after_initialization=not any(e['dst_label'] in grids and e['src_space']=='Host' for e in copies),
            optical_tables_uploaded_once=not any(e['dst_label'] in ['interface_radio_index_table','interface_radio_optical_breaks','interface_radio_observers'] and e['src_space']=='Host' and e['time']>min(k['start'] for k in kernels) for e in copies),
            cpu_sources_uploaded_only_when_present=bool(cpu)==bool(radio['cpu_tracks']),
            trace_complete=len(finals)==1 and not finals[0]['failed'] and finals[0]['dropped_records']==0 and not any(e['event']=='error' for e in events))
        checks['cpu_upload_track_accounting']=sum(e['bytes'] for e in cpu)==radio['cpu_tracks']*104
        previous=next(r for r in accepted if (r['case'],r['variant'])==(run['case'],run['variant']))
        reference=Path(previous['radio']['directory'])
        checks['unchanged_radio_counters']=all(v==previous['radio'][k] for k,v in radio.items() if k!='directory')
        checks['unchanged_radio_config']=(root/'radio/config.json').read_bytes()==(reference/'config.json').read_bytes()
        moment_errors={}
        for filename in ['moments.bin','regularized_moments.bin']:
            a=np.fromfile(root/'radio'/filename,dtype=np.float64);b=np.fromfile(reference/filename,dtype=np.float64)
            moment_errors[filename]=float(np.linalg.norm(a-b)/max(np.linalg.norm(b),1.e-100)) if a.shape==b.shape else float('inf')
        checks['unchanged_both_moment_arrays']=all(x<1.e-11 for x in moment_errors.values())
        actual=Counter((e['kind'],e['bytes']) for e in events if e['event']=='gpu_copy')
        checks['moment_copy_in_cupti']=actual[(2,expected_bytes)]>=2
        rows.append(dict(case=run['case'],variant=run['variant'],checks=checks,radio=radio,
            moment_relative_l2=moment_errors,gpu_radio_kernels=len(kernels),radio_kernel_time_ms=sum(k['end']-k['start'] for k in kernels)/1.e6,
            moment_array_downloads=len(moments),moment_download_bytes=2*expected_bytes,cpu_source_uploads=len(cpu),cpu_source_upload_bytes=sum(e['bytes'] for e in cpu),trace=run['trace']))
    report=dict(passed=bool(rows) and all(all(r['checks'].values()) for r in rows),runs=rows,
        scope='GPU-resident optical tables, captured EM segments and both CoREAS moment grids; host coordinates wavefronts, uploads charged CPU segments, and renders FFT/output after final download.')
    args.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
    if not report['passed']:raise SystemExit(1)

if __name__=='__main__':main()

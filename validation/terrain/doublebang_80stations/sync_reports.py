#!/usr/bin/env python3
"""Transfer-only local mirror; numerical analysis stays on PSR."""
import argparse
import json
from pathlib import Path
import subprocess
import time


def main():
    p=argparse.ArgumentParser();p.add_argument('--remote-root',required=True);p.add_argument('--destination',type=Path,required=True)
    p.add_argument('--socket',default='/tmp/c8-psr-figure-pack-20260914.sock');p.add_argument('--watch',action='store_true');a=p.parse_args()
    a.destination.mkdir(parents=True,exist_ok=True)
    terminal={'complete','failed','candidate_limit_reached','stopped_after_current'}
    while True:
        cmd=['rsync','-rt','--partial','--include=/report/','--include=/report/***',
            '--include=/README_CN.md','--include=/progress.json','--include=/campaign.json','--exclude=*',
            '-e','ssh -o BatchMode=yes -o ConnectTimeout=15 -o ServerAliveInterval=15 -o ServerAliveCountMax=2 -S '+a.socket,
            'psrpku2025_PKU:'+a.remote_root.rstrip('/')+'/',str(a.destination)+'/']
        try:
            subprocess.run(cmd,check=True,timeout=120)
            state=json.loads((a.destination/'progress.json').read_text())
            print(state['updated_utc'],state['state'],len(state['completed']),state.get('current',''),flush=True)
            handoff=a.destination/'report/radio_alignment_pending.json'
            awaiting_handoff=(state['state']=='stopped_after_current' and handoff.exists()
                and json.loads(handoff.read_text()).get('state')=='ready_for_stage_boundary')
            if (state['state'] in terminal and not awaiting_handoff) or not a.watch:return
        except (subprocess.SubprocessError,OSError,json.JSONDecodeError) as error:
            print('Transfer pending:',error,flush=True)
            if not a.watch:raise
        time.sleep(60)


if __name__=='__main__':main()

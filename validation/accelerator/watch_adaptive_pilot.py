#!/usr/bin/env python3
"""Refresh bounded diagnostic reports; never signal or start simulator jobs."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('root', type=Path)
    p.add_argument('--duration', type=float, default=8000.)
    a=p.parse_args()
    deadline=time.monotonic()+a.duration
    while time.monotonic()<deadline:
        status=a.root/'STATUS.json'
        if status.is_file() and (a.root/'PROVENANCE.json').is_file():
            state=json.loads(status.read_text())
            subprocess.run([sys.executable, str(Path(__file__).with_name(
                'summarize_adaptive_service_pilot.py')), str(a.root)],check=True)
            if state['complete'] or state['phase'].startswith('failed'):
                return
        time.sleep(30)


if __name__=='__main__':
    main()

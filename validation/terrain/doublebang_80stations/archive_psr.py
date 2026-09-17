#!/usr/bin/env python3
"""Lossless archive of this campaign's completed radio output, on PSR."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import socket
import subprocess


def hash_stream(stream):
    h=hashlib.sha256()
    for block in iter(lambda:stream.read(1048576),b''):h.update(block)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--folder',type=Path,required=True);folder=p.parse_args().folder
    assert socket.gethostname()=='psrpku2025'
    assert json.loads((folder.parent.parent/'report'/folder.name/'acceptance.json').read_text())['passed']
    records=[]
    for path in sorted((folder/'output/radio').rglob('*')):
        if not path.is_file() or path.suffix not in ['.csv','.bin']:continue
        packed=path.with_suffix(path.suffix+'.gz');temp=packed.with_suffix(packed.suffix+'.tmp')
        if packed.exists() or temp.exists():raise RuntimeError('Archive already exists: '+str(packed))
        with path.open('rb') as stream:original=hash_stream(stream)
        if shutil.which('pigz'):
            with temp.open('xb') as out:
                subprocess.run(['pigz','-1','-p','16','-c',str(path)],stdout=out,check=True)
        else:
            with path.open('rb') as src,temp.open('xb') as target:
                with gzip.GzipFile(filename='',mode='wb',compresslevel=1,fileobj=target,mtime=0) as dst:
                    shutil.copyfileobj(src,dst,1048576)
        with gzip.open(temp,'rb') as stream:restored=hash_stream(stream)
        assert restored==original
        temp.replace(packed)
        record=dict(original=str(path),archive=str(packed),bytes=path.stat().st_size,
                    gzip_bytes=packed.stat().st_size,uncompressed_sha256=original,roundtrip_verified=True)
        # Only remove the now redundant, reproducible uncompressed copy.
        path.unlink();records.append(record)
        (folder/'radio_archive.json').write_text(json.dumps(records,indent=2)+'\n')
    print('Losslessly archived',len(records),'radio files',flush=True)


if __name__=='__main__':main()

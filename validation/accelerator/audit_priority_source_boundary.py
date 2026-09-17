#!/usr/bin/env python3
"""Verify the frozen build differs only in the named priority-interface files."""
import argparse
import hashlib
import json
from pathlib import Path


ALLOWED = {
    'applications/c8_air_shower.cpp',
    'applications/detail/air_shower_kokkos/KokkosRunSession.cpp',
    'applications/detail/air_shower_kokkos/KokkosShowerReport.hpp',
    'corsika/accelerator/em/detail/CpuPrimarySubshowerPolicy.hpp',
    'corsika/accelerator/em/detail/IndependentSubshowerPump.hpp',
    'src/accelerator/em/kokkos/KokkosBackendSelection.cpp',
    'src/accelerator/em/kokkos/KokkosRuntime.cpp',
    'src/accelerator/em/kokkos/KokkosEmBackend.cpp',
    'src/accelerator/em/kokkos/KokkosCooperativeBackend.cpp',
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--before',type=Path,required=True)
    p.add_argument('--after',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();files=set()
    for root in (a.before,a.after):
        for sub in ('corsika','src','applications'):
            files.update(str(q.relative_to(root)) for q in (root/sub).rglob('*')
                         if q.is_file() and q.suffix in ('.hpp','.cpp','.cu','.h','.inl','.cxx','.hxx'))
    changed=[];unchanged=0
    for rel in sorted(files):
        x,y=digest(a.before/rel),digest(a.after/rel)
        if x==y:unchanged+=1
        else:changed.append(dict(path=rel,before=x,after=y,allowed=rel in ALLOWED))
    data=dict(before=str(a.before),after=str(a.after),unchanged_source_files=unchanged,
        changed=changed,pass_=bool(changed) and all(r['allowed'] for r in changed),
        scope='frozen source boundary only; changed scheduler behavior requires runtime tests')
    a.output.write_text(json.dumps(data,indent=2)+'\n')
    assert data['pass_'], 'unexpected physics/kernel/mountain source change'
    print(json.dumps(dict(pass_=True,unchanged=unchanged,allowed_changed=len(changed))))


if __name__=='__main__':main()

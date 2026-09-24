#!/usr/bin/env python3
"""Build a private data overlay; never alter the live production data directory."""
from pathlib import Path
import shutil

work = Path(__file__).resolve().parent
source = Path('/home/member/yuhanglu/workspace/corsika-beta5-fe-boundary-20260921/data')
target = work / 'runtime-data'
target.mkdir(exist_ok=True)
for entry in source.iterdir():
    if entry.name == 'GeoMag':
        continue
    link = target / entry.name
    if link.is_symlink():
        if link.resolve() != entry.resolve():
            raise RuntimeError('Unexpected data overlay target: ' + str(link))
    elif link.exists():
        raise RuntimeError('Refusing to replace data: ' + str(link))
    else:
        link.symlink_to(entry)
(target / 'GeoMag').mkdir(exist_ok=True)
destination = target / 'GeoMag/IGRF14.COF'
content = (work / 'IGRF14.COF').read_bytes()
if destination.exists():
    if destination.read_bytes() != content:
        raise RuntimeError('Existing magnetic data differ')
else:
    shutil.copyfile(work / 'IGRF14.COF', destination)
print('Private runtime data ready:', target)

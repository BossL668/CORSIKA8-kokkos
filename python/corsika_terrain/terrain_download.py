"""Bounded, verified HTTPS/cache access to GRAND SRTM or Mapzen Skadi HGT.

No GRAND import is required (its store protocol disables TLS verification).
Native HGT samples are point-posted, north-to-south, big-endian int16.
"""
from __future__ import annotations

import gzip
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
import time
import urllib.request

import numpy as np
from pyproj import Transformer

from .terrain import sha256_file

GEOID_URL = "https://cdn.proj.org/us_nga_egm96_15.tif"
HGT_BYTES = 3601 * 3601 * 2


def tile_name(lat, lon):
    return f"{'N' if lat >= 0 else 'S'}{abs(lat):02d}{'E' if lon >= 0 else 'W'}{abs(lon):03d}"


def validate_bounds(bounds):
    w, s, e, n = [float(bounds[k]) for k in ('west', 'south', 'east', 'north')]
    if not all(math.isfinite(v) for v in (w,s,e,n)):
        raise ValueError('geographic bounds must be finite')
    if not (-180 <= w < e <= 180 and -85 <= s < n <= 85):
        raise ValueError('require west < east, south < north; dateline/polar patches unsupported')
    # A fixed tangent plane and height-field triangulation are local models.
    if e-w > 2 or n-s > 2:
        raise ValueError('local ENU terrain supports at most 2 degrees per axis')
    return w,s,e,n


def required_tiles(bounds):
    w,s,e,n=validate_bounds(bounds)
    return [(lat,lon) for lat in range(math.floor(s),math.ceil(n))
            for lon in range(math.floor(w),math.ceil(e))]


def download(url, destination, max_bytes, offline=False):
    """Atomic cache admission; validate SHA on reuse; preserve corrupt files."""
    destination=Path(destination)
    receipt=destination.with_name(destination.name+'.download.json')
    if destination.exists():
        if not receipt.exists():
            raise ValueError(f'unverified cache entry (missing receipt): {destination}')
        meta=json.loads(receipt.read_text())
        if meta['url']!=url or sha256_file(destination)!=meta['sha256']:
            raise ValueError(f'cache URL/checksum mismatch: {destination}')
        return dict(meta,cache_hit=True,path=str(destination.resolve()))
    if offline:
        raise FileNotFoundError(f'offline cache miss: {destination.name}')
    destination.parent.mkdir(parents=True,exist_ok=True)
    if shutil.disk_usage(destination.parent).free < 2*max_bytes:
        raise ValueError('insufficient free space for bounded download')
    fd,name=tempfile.mkstemp(prefix=destination.name+'.partial-',dir=destination.parent)
    temporary=Path(name)
    try:
        request=urllib.request.Request(url,headers={'User-Agent':'CORSIKA-terrain-validation/1.0'})
        begin=time.monotonic()
        with os.fdopen(fd,'wb') as output, urllib.request.urlopen(request,timeout=30) as response:
            if int(response.headers.get('Content-Length','0'))>max_bytes:
                raise ValueError('download exceeds configured byte limit')
            count=0
            while True:
                block=response.read(1024*1024)
                if not block: break
                count+=len(block)
                if count>max_bytes or time.monotonic()-begin>180:
                    raise ValueError('download byte/time budget exceeded')
                output.write(block)
        meta=dict(url=url,sha256=sha256_file(temporary),bytes=count,
                  retrieved_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
        temporary.replace(destination)
        receipt.write_text(json.dumps(meta,indent=2)+'\n')
        return dict(meta,cache_hit=False,path=str(destination.resolve()))
    finally:
        if temporary.exists(): temporary.unlink() # only this owned partial file


class HgtSource:
    """Sample a bounded collection without materializing entire float64 tiles."""
    def __init__(self, tiles):
        self.tiles=tiles

    def sample(self, longitude, latitude):
        lon,lat=np.broadcast_arrays(np.asarray(longitude,dtype=float),np.asarray(latitude,dtype=float))
        result=np.full(lon.shape,np.nan)
        for south,west,path in self.tiles:
            selected=(lon>=west)&(lon<=west+1)&(lat>=south)&(lat<=south+1)&np.isnan(result)
            if not np.any(selected): continue
            size=Path(path).stat().st_size
            edge=math.isqrt(size//2)
            if size!=2*edge*edge or edge not in (1201,3601):
                raise ValueError(f'invalid HGT dimensions: {path}')
            dem=np.memmap(path,dtype='>i2',mode='r',shape=(edge,edge))
            x=(lon[selected]-west)*(edge-1)
            y=(south+1-lat[selected])*(edge-1)
            ix=np.minimum(np.floor(x).astype(int),edge-2)
            iy=np.minimum(np.floor(y).astype(int),edge-2)
            fx=x-ix;fy=y-iy
            heights=np.zeros_like(x)
            for dx,dy,weights in ((0,0,(1-fx)*(1-fy)),(1,0,fx*(1-fy)),
                                   (0,1,(1-fx)*fy),(1,1,fx*fy)):
                values=dem[iy+dy,ix+dx].astype(float)
                if np.any((values==-32768)&(weights>1e-12)):
                    raise ValueError('DEM nodata in requested footprint; no silent hole filling')
                heights+=values*weights
            result[selected]=heights
            del dem
        if not np.all(np.isfinite(result)):
            raise ValueError('DEM coverage incomplete; extrapolation forbidden')
        return result


def acquire_hgt(source, bounds, cache, offline=False):
    provider=source.get('provider','grand_srtmgl1')
    if provider not in ('grand_srtmgl1','skadi'):
        raise ValueError('source.provider must be grand_srtmgl1 or skadi')
    _,s,_,n=validate_bounds(bounds)
    if provider=='grand_srtmgl1' and not (-56<=s<n<=60):
        raise ValueError('SRTMGL1 latitude coverage is limited to [-56,60] degrees')
    tiles=[];provenance=[]
    for lat,lon in required_tiles(bounds):
        name=tile_name(lat,lon)
        # Explicitly permitted read-only caches, e.g. the user's GRAND installation.
        candidate=next((Path(d)/(name+'.hgt') for d in source.get('local_hgt_directories',[])
                        if (Path(d)/(name+'.hgt')).is_file()),None)
        if candidate is not None:
            if candidate.stat().st_size!=HGT_BYTES:
                raise ValueError('GRAND/Skadi source expects a 3601 x 3601 HGT tile')
            meta=dict(path=str(candidate.resolve()),sha256=sha256_file(candidate),
                      bytes=candidate.stat().st_size,cache_hit=True,
                      provenance='user_explicit_local_HGT; identity hash recorded, not remotely authenticated')
        else:
            url=(f'https://github.com/grand-mother/store/releases/download/101/{name}.SRTMGL1.hgt.gz'
                 if provider=='grand_srtmgl1' else
                 f'https://elevation-tiles-prod.s3.amazonaws.com/skadi/{name[:3]}/{name}.hgt.gz')
            archive=Path(cache)/provider/(name+'.hgt.gz')
            meta=download(url,archive,32*1024**2,offline)
            candidate=archive.with_suffix('')
            receipt=candidate.with_name(candidate.name+'.sha256')
            if candidate.exists():
                if not receipt.exists() or receipt.read_text().strip()!=sha256_file(candidate):
                    raise ValueError(f'unverified extracted HGT: {candidate}')
            else:
                fd,tmp=tempfile.mkstemp(prefix=name+'.extract-',dir=archive.parent)
                temporary=Path(tmp)
                try:
                    count=0
                    with os.fdopen(fd,'wb') as out,gzip.open(archive,'rb') as stream:
                        while True:
                            block=stream.read(1024*1024)
                            if not block: break
                            count+=len(block)
                            if count>HGT_BYTES: raise ValueError('HGT decompression size limit exceeded')
                            out.write(block)
                    if count!=HGT_BYTES: raise ValueError('truncated/unsupported HGT tile')
                    temporary.replace(candidate)
                    receipt.write_text(sha256_file(candidate)+'\n')
                finally:
                    if temporary.exists(): temporary.unlink()
            meta.update(uncompressed_path=str(candidate.resolve()),
                        uncompressed_sha256=sha256_file(candidate))
        provenance.append(dict(tile=name,**meta))
        tiles.append((lat,lon,candidate))
    return HgtSource(tiles),provenance


def geoid_transform(cache,offline=False):
    grid=Path(cache)/'geoid/us_nga_egm96_15.tif'
    meta=download(GEOID_URL,grid,4*1024**2,offline)
    # vgridshift without inv adds geoid undulation: h = H + N.
    transform=Transformer.from_pipeline(
        '+proj=pipeline +step +proj=unitconvert +xy_in=deg +xy_out=rad '
        f'+step +proj=vgridshift +grids={grid.resolve()} +multiplier=1 '
        '+step +proj=unitconvert +xy_in=rad +xy_out=deg')
    return transform,meta


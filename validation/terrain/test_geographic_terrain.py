from copy import deepcopy
import gzip
import io
import json
from pathlib import Path
import urllib.error

import numpy as np
import pytest
import yaml

from corsika_terrain import geographic_terrain as geo
from corsika_terrain import terrain_download as download


def settings(tmp_path):
    return dict(enabled=True,bounds_deg=dict(west=86.70,east=86.702,south=42.93,north=42.932),
        output_directory=str(tmp_path/'product'),cache_directory=str(tmp_path/'cache'),
        source=dict(provider='grand_srtmgl1'),mesh=dict(spacing_arcsec=1,bottom_depth_m=100),
        antennas={})


@pytest.mark.parametrize('bounds',[
    dict(west=1,east=0,south=0,north=1),dict(west=179,east=-179,south=0,north=1),
    dict(west=0,east=1,south=89,north=90),dict(west=0,east=4,south=0,north=1),
    dict(west=float('nan'),east=1,south=0,north=1)])
def test_bad_geographic_ranges(bounds):
    with pytest.raises(ValueError): download.required_tiles(bounds)


def test_tile_boundaries_and_southern_western_names():
    assert download.required_tiles(dict(west=-1.2,east=0,south=-.2,north=.2))==[(-1,-2),(-1,-1),(0,-2),(0,-1)]
    assert download.tile_name(-1,-2)=='S01W002'
    assert download.required_tiles(dict(west=86,east=87,south=42,north=43))==[(42,86)]


@pytest.mark.parametrize('override',[
    dict(mesh=dict(spacing_arcsec=.1)),dict(limits=dict(max_vertices=2)),
    dict(limits=dict(max_faces=1)),dict(limits=dict(max_estimated_memory_mb=1)),
    dict(limits=dict(max_tiles=0)),dict(source=dict(provider='not_real')),
    dict(medium=dict(density_g_cm3=-1)),dict(mesh=dict(bottom_depth_m=0))])
def test_preflight_resource_and_scope_guards(tmp_path,override):
    cfg=settings(tmp_path);cfg.update(override)
    with pytest.raises(ValueError): geo.estimate(cfg)
    assert not (tmp_path/'cache').exists()


def test_estimate_and_disabled_do_not_download_or_create_outputs(tmp_path,monkeypatch):
    path=tmp_path/'input.yaml';path.write_text(yaml.safe_dump(dict(terrain=settings(tmp_path))))
    monkeypatch.setattr(geo,'acquire_hgt',lambda *a,**k:pytest.fail('unexpected download'))
    assert geo.prepare(path,estimate_only=True)['network_requested'] is False
    path.write_text('terrain:\n  enabled: false\n')
    assert geo.prepare(path)['status']=='terrain_disabled'
    assert not (tmp_path/'product').exists() and not (tmp_path/'cache').exists()


def test_hgt_endian_orientation_bilinear_and_shared_boundary(tmp_path):
    # 1201 fixture is the documented SRTM3 shape; online adapters require 3601.
    path=tmp_path/'N00E000.hgt'
    r,c=np.indices((1201,1201))
    (1000+r+c).astype('>i2').tofile(path)
    source=download.HgtSource([(0,0,path)])
    np.testing.assert_allclose(source.sample([0,1,.25],[1,0,.75]),[1000,3400,1600])
    np.testing.assert_allclose(source.sample([.5/1200],[1-.5/1200]),1001)
    with pytest.raises(ValueError,match='extrapolation'): source.sample([1.001],[.5])


def test_hgt_void_and_truncation_rejected(tmp_path):
    path=tmp_path/'bad.hgt';np.full((1201,1201),-32768,dtype='>i2').tofile(path)
    with pytest.raises(ValueError,match='nodata'): download.HgtSource([(0,0,path)]).sample([.5],[.5])
    path.write_bytes(b'\0\0')
    with pytest.raises(ValueError,match='dimensions'): download.HgtSource([(0,0,path)]).sample([.5],[.5])


class Response(io.BytesIO):
    def __init__(self,data): super().__init__(data);self.headers={'Content-Length':str(len(data))}


def test_download_cache_checksum_offline_and_atomic_budget(tmp_path,monkeypatch):
    count=[]
    def urlopen(request,timeout): count.append(request.full_url);return Response(b'1234')
    monkeypatch.setattr(download.urllib.request,'urlopen',urlopen)
    p=tmp_path/'cache/test.bin'
    assert not download.download('https://example.test/file',p,10)['cache_hit']
    assert download.download('https://example.test/file',p,10,offline=True)['cache_hit']
    assert len(count)==1
    p.write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='checksum'): download.download('https://example.test/file',p,10)
    with pytest.raises(ValueError,match='limit'): download.download('https://example.test/file',tmp_path/'too_big',2)
    assert not list(tmp_path.glob('too_big*'))
    with pytest.raises(FileNotFoundError,match='offline'): download.download('https://example.test/file',tmp_path/'missing',10,True)


def test_geographic_csv_dms_selection_and_height_mode(tmp_path):
    path=tmp_path/'a.csv'
    path.write_text('name,longitude_deg,latitude_deg,height_m\nA,86.4302579725,42.5526742053,2598\nB,86.43,42.55,2600\n')
    rows,_=geo.read_antennas(dict(path=str(path),angle_format='packed_dms',selected_names=['A']))
    assert rows[0]['name']=='A'
    assert abs(rows[0]['longitude_deg']-(86+43/60+2.579725/3600))<1e-12
    with pytest.raises(ValueError,match='selection'): geo.read_antennas(dict(path=str(path),selected_names=['missing']))


def test_small_end_to_end_model_without_network(tmp_path,monkeypatch):
    cfg=settings(tmp_path);csv=tmp_path/'antennas.csv'
    csv.write_text('name,longitude_deg,latitude_deg,height_m\nA,86.7008,42.9312,2655\nB,86.7015,42.9305,2660\n')
    cfg['antennas']=dict(format='geodetic_csv',path=str(csv),height_mode='dem_agl',height_above_ground_m=1)
    class Source:
        def sample(self,lon,lat): return 2700+100*(np.asarray(lon)-86.7)+80*(np.asarray(lat)-42.93)
    class Geoid:
        def transform(self,lon,lat,h,**kw): return lon,lat,np.asarray(h)-40
    stub=tmp_path/'stub';stub.write_bytes(b'fixture')
    sha=download.sha256_file(stub)
    monkeypatch.setattr(geo,'acquire_hgt',lambda *a: (Source(),[dict(path=str(stub),sha256=sha)]))
    monkeypatch.setattr(geo,'geoid_transform',lambda *a: (Geoid(),dict(path=str(stub),sha256=sha)))
    monkeypatch.setattr(geo,'plot_product',lambda *a: None)
    path=tmp_path/'input.yaml';path.write_text(yaml.safe_dump(dict(terrain=cfg)))
    summary=geo.prepare(path)
    assert summary['mesh_qa']['watertight']
    assert summary['mesh_qa']['all_top_normals_up']
    assert summary['antenna_qa']['selected_count']==2
    np.testing.assert_allclose(summary['antenna_qa']['clearance_range_m'],[1,1],atol=1e-8)
    assert summary['coordinate_qa']['roundtrip_max_deg']<1e-9
    assert geo.prepare(path,offline=True)['reused'] is True
    ply=Path(cfg['output_directory'])/'terrain_enu.ply';ply.write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='checksum'): geo.prepare(path)


def test_outside_antenna_rejected_before_download(tmp_path,monkeypatch):
    cfg=settings(tmp_path);csv=tmp_path/'a.csv';csv.write_text('name,longitude_deg,latitude_deg\nA,87,43\n')
    cfg['antennas']=dict(path=str(csv))
    path=tmp_path/'input.yaml';path.write_text(yaml.safe_dump(dict(terrain=cfg)))
    monkeypatch.setattr(geo,'acquire_hgt',lambda *a:pytest.fail('unexpected download'))
    with pytest.raises(ValueError,match='outside requested bounds'): geo.prepare(path)


def test_run_preprocessor_never_silently_runs_default_uniform_shower(tmp_path):
    path=tmp_path/'run.yaml';path.write_text(yaml.safe_dump(dict(terrain=settings(tmp_path))))
    with pytest.raises(ValueError,match='geometry only'): geo.prepare_run_config(path)
    path.write_text('event: {backend: corsika}\nradio: {solver: corsika_zhs}\nterrain: {enabled: true}\n')
    with pytest.raises(ValueError,match='transport only'): geo.prepare_run_config(path)
    path.write_text('terrain: {enabled: false}\n')
    assert geo.prepare_run_config(path)==str(path.resolve())


def test_run_preprocessor_preserves_original_and_resolves_executable(tmp_path,monkeypatch):
    directory=tmp_path/'prepared';directory.mkdir()
    (directory/'corsika_geometry.yaml').write_text('site: {latitude_deg: 42, longitude_deg: 86, altitude_m: 2700}\ngeometry: {kind: terrain_mesh, mesh_path: /example/terrain.ply}\n')
    (directory/'observers.yaml').write_text('radio: {observers: [{name: A, position_enu_m: [1, 2, 3]}]}\n')
    monkeypatch.setattr(geo,'prepare',lambda p:dict(configuration=dict(output_directory=str(directory)),request_sha256='fixture'))
    path=tmp_path/'run.yaml';original='terrain: {enabled: true}\nevent: {backend: corsika, corsika: {executable: ../build/c8_mountain_corsika}}\nradio: {solver: none}\n'
    path.write_text(original)
    result=Path(geo.prepare_run_config(path));raw=yaml.safe_load(result.read_text())
    assert raw['geometry']['kind']=='terrain_mesh' and raw['terrain']['enabled'] is False
    assert raw['event']['corsika']['executable']==str((tmp_path/'../build/c8_mountain_corsika').resolve())
    assert raw['radio']['observers'][0]['name']=='A'
    assert path.read_text()==original
    assert geo.prepare_run_config(path)==str(result)


@pytest.mark.parametrize('supplied', [None,2750.,2700.,float('nan')])
def test_layered_run_preprocessor_geoid_datum_and_generated_world(tmp_path,monkeypatch,supplied):
    directory=tmp_path/'prepared';directory.mkdir()
    (directory/'corsika_geometry.yaml').write_text('site: {latitude_deg: 42, longitude_deg: 86, altitude_m: 2700}\ngeometry: {kind: terrain_mesh, mesh_path: /example/terrain.ply, air_world_radius_m: 100000}\n')
    (directory/'observers.yaml').write_text('radio: {observers: [{name: A, position_enu_m: [1, 2, 3]}]}\n')
    monkeypatch.setattr(geo,'prepare',lambda p:dict(configuration=dict(output_directory=str(directory),cache_directory=str(tmp_path/'cache')),request_sha256='fixture',frame=dict(origin_latitude_deg=42.,origin_longitude_deg=86.,origin_ellipsoidal_height_m=2700.)))
    class Geoid:
        def transform(self,*args,**kwargs):return 86.,42.,-50.
    monkeypatch.setattr(geo,'geoid_transform',lambda *args,**kwargs:(Geoid(),{}))
    raw=dict(terrain=dict(enabled=True),event=dict(backend='corsika',corsika=dict(executable='../build/c8_terrain_atmosphere_shower')),radio=dict(solver='none'),atmosphere=dict(model='us_standard_bk'))
    if supplied is not None:raw['atmosphere']['origin_altitude_asl_m']=supplied
    path=tmp_path/'run.yaml';original=yaml.safe_dump(raw);path.write_text(original)
    if supplied is not None and supplied!=2750.:
        with pytest.raises(ValueError,match='datum'):geo.prepare_run_config(path)
    else:
        result=yaml.safe_load(Path(geo.prepare_run_config(path)).read_text())
        assert result['atmosphere']['origin_altitude_asl_m']==2750.
        assert result['site']['altitude_m']==2700.
        assert 'air_world_radius_m' not in result['geometry']
        assert result['radio']['observers'][0]['name']=='A'
    assert path.read_text()==original


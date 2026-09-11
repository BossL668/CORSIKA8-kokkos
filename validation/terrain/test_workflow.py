"""Workflow regression tests are offline; real HTTPS/native tests are separate."""
import json
from pathlib import Path
import numpy as np
import pytest
import yaml
from corsika_terrain import workflow as flow
from corsika_terrain import geographic_terrain as geo
from corsika_terrain import scene


def fixture_request(tmp_path,monkeypatch):
    cfg=flow.region_settings([86.70,42.93,86.702,42.932],tmp_path/'region',tmp_path/'cache',bottom_depth_m=100)
    path=tmp_path/'input.yaml'
    path.write_text(yaml.safe_dump(dict(terrain=cfg)))
    class Source:
        def sample(self,lon,lat): return np.zeros_like(np.asarray(lon)+np.asarray(lat))+2700
    class Geoid:
        def transform(self,lon,lat,h,**kw): return lon,lat,np.asarray(h)-40
    stub=tmp_path/'stub';stub.write_bytes(b'fixture')
    meta=dict(path=str(stub),sha256=geo.sha256_file(stub))
    monkeypatch.setattr(geo,'acquire_hgt',lambda *a:(Source(),[meta]))
    monkeypatch.setattr(geo,'geoid_transform',lambda *a,**kw:(Geoid(),meta))
    monkeypatch.setattr(scene,'geoid_transform',lambda *a,**kw:(Geoid(),meta))
    monkeypatch.setattr(geo,'plot_product',lambda *a:None)
    return path,cfg


def test_estimate_bounds_no_side_effect(tmp_path,monkeypatch):
    monkeypatch.setattr(geo,'acquire_hgt',lambda *a:pytest.fail('unexpected network'))
    assert flow.main(['--bounds','86.7','42.93','86.702','42.932','--output',str(tmp_path/'new'),'--estimate-only'])==0
    assert not (tmp_path/'new').exists()


def test_workflow_native_scene_reuse_and_conflict(tmp_path,monkeypatch):
    config,cfg=fixture_request(tmp_path,monkeypatch)
    result=flow.prepare_scene(config,offline=True)
    scene_path=Path(result['scene']); before=scene_path.read_bytes()
    s=yaml.safe_load(before)
    assert s['geometry']['mesh_path']=='terrain_enu.ply'
    assert s['atmosphere']['origin_altitude_asl_m']==pytest.approx(2700)
    assert s['site']['origin_ellipsoidal_height_m']==pytest.approx(2660)
    assert s['atmosphere']['model']=='us_standard_bk'
    assert 'air_world_radius_m' not in s['geometry']
    assert s['radio']['observers'][0]['name']=='reference_centre'
    assert flow.prepare_scene(config,offline=True)['reused']
    assert scene_path.read_bytes()==before
    cfg['mesh']['spacing_arcsec']=4
    config.write_text(yaml.safe_dump(dict(terrain=cfg)))
    with pytest.raises(FileExistsError,match='request changed'):
        flow.prepare_scene(config,offline=True)


def test_scene_tamper_rejected(tmp_path,monkeypatch):
    config,_=fixture_request(tmp_path,monkeypatch)
    result=flow.prepare_scene(config)
    Path(result['scene']).write_text('tampered')
    with pytest.raises(FileExistsError,match='differs'):
        flow.prepare_scene(config)


def test_download_or_prepare_alone_never_runs_shower(tmp_path,monkeypatch):
    config,_=fixture_request(tmp_path,monkeypatch)
    monkeypatch.setattr(flow.subprocess,'run',lambda *a,**kw:pytest.fail('unexpected shower'))
    assert flow.main(['--config',str(config),'--offline'])==0


def test_run_command_no_shell_no_scene_override(tmp_path,monkeypatch):
    calls=[]
    monkeypatch.setattr(flow,'resolve_executable',lambda name:'/bin/test-application')
    class Done: returncode=7
    monkeypatch.setattr(flow.subprocess,'run',lambda argv,**kw:(calls.append((argv,kw)) or Done()))
    assert flow.run_scene(tmp_path/'scene.yaml','demo',['--em-backend','kokkos','--output','a path'])==7
    assert calls[0][0][-1]=='a path' and 'shell' not in calls[0][1]
    for args in [['--scene','evil'],['--scene=evil']]:
        with pytest.raises(ValueError,match='managed'):flow.run_scene(tmp_path/'scene.yaml','demo',args)


def test_lock_timeout_and_no_clobber(tmp_path):
    with flow.preparation_lock(tmp_path):
        with pytest.raises(TimeoutError):
            with flow.preparation_lock(tmp_path,timeout=.01):pass
    p=tmp_path/'artifact';flow.write_identical_or_new(p,'one')
    flow.write_identical_or_new(p,'one')
    with pytest.raises(FileExistsError):flow.write_identical_or_new(p,'two')
    assert p.read_text()=='one'


def test_inline_height_and_bounds_errors(tmp_path):
    cfg=flow.region_settings([86.7,42.93,86.702,42.932],tmp_path)
    rows,_=geo.read_antennas(cfg['antennas'])
    assert len(rows)==1 and rows[0]['measured_height_m'] is None
    cfg['antennas']['points'][0]['height_m']=2700.
    rows,_=geo.read_antennas(cfg['antennas'])
    assert rows[0]['measured_height_m']==2700.
    cfg['antennas']['points'][0]['height_m']='   '
    rows,_=geo.read_antennas(cfg['antennas'])
    assert rows[0]['measured_height_m'] is None
    with pytest.raises(ValueError):flow.region_settings([87,42,86,43],tmp_path)


def test_native_checker_admission(tmp_path,monkeypatch):
    config,_=fixture_request(tmp_path,monkeypatch)
    monkeypatch.setattr(flow,'resolve_executable',lambda _: '/bin/native-probe')
    def run(argv,**kw):
        assert argv[1]=='--config' and argv[3]=='--output'
        Path(argv[4]).write_text('complete: true\n')
    monkeypatch.setattr(flow.subprocess,'run',run)
    result=flow.prepare_scene(config,check_with='native-probe')
    assert result['native_geometry_checked']
    assert Path(result['native_check']).is_file()

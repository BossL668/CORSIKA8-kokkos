#!/usr/bin/env python3
"""Independent 80-station source, datum, coordinate, and mesh-height audit."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import socket
import numpy as np
from pyproj import Transformer
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);root=p.parse_args().root
    assert socket.gethostname()=='psrpku2025'
    source=root/'stations';out=root/'report';out.mkdir(exist_ok=True)
    manifest=yaml.safe_load((source/'terrain_manifest.yaml').read_text())
    for old,sha in manifest['antenna_source']['source_sha256'].items():assert digest(source/Path(old).name)==sha
    assert digest(source/'us_nga_egm96_15.tif')==manifest['geoid']['sha256']
    for name in ['terrain_grid.npz','antenna_positions_enu.csv','observers.yaml']:
        assert digest(source/name)==manifest['output_sha256'][name]
    with (source/'antenna_positions_enu.csv').open() as f:placed=list(csv.DictReader(f))
    observers=yaml.safe_load((source/'observers.yaml').read_text())['radio']['observers']
    with (source/'20160810.csv').open() as f:survey=list(csv.DictReader(f))
    raw={}
    for line in (source/'dump.txt').read_text().splitlines():
        n,*x=line.split();assert n not in raw;raw[n]=np.array(list(map(float,x)))
    for row in survey:
        if row['Node'].startswith(('N','S')):
            assert row['Node'] not in raw
            raw[row['Node']]=np.array([float(row[k]) for k in ['WCS84 x(m)','WCS84 y(m)','WCS84 z(m)']])
    measured={}
    for line in (source/'antenna_coord.dat').read_text().splitlines():
        name,*xyz=line.split();measured.setdefault(name,[]).append(list(map(float,xyz)))
    centroid_error=max(float(np.linalg.norm(np.mean(xyz,axis=0)-raw[name])) for name,xyz in measured.items() if name in raw)
    assert centroid_error<1e-6
    expected={arm+('%02d'%i) for arm in 'EWNS' for i in range(1,21)}
    for values in [list(raw),[r['name'] for r in placed],[r['name'] for r in observers]]:
        assert len(values)==80 and set(values)==expected and len(set(values))==80
    obs={r['name']:r['position_enu_m'] for r in observers}
    inverse=Transformer.from_crs(4978,4979,always_xy=True)
    forward=Transformer.from_crs(4979,4978,always_xy=True)
    geoid=Transformer.from_pipeline('+proj=pipeline +step +proj=unitconvert +xy_in=deg +xy_out=rad '+
        '+step +proj=vgridshift +grids='+str(source/'us_nga_egm96_15.tif')+' +multiplier=1 '+
        '+step +proj=unitconvert +xy_in=rad +xy_out=deg')
    frame=manifest['frame'];lat0,lon0=np.deg2rad([frame['origin_latitude_deg'],frame['origin_longitude_deg']])
    # Independently construct WGS84 ECEF and the ENU basis, not the producer's
    # projected_to_enu or placement/interpolation helpers.
    a=6378137.;f=1/298.257223563;e2=f*(2-f)
    def ecef(lon,lat,h):
        lon,lat=np.deg2rad([lon,lat]);n=a/np.sqrt(1-e2*np.sin(lat)**2)
        return np.array([(n+h)*np.cos(lat)*np.cos(lon),(n+h)*np.cos(lat)*np.sin(lon),(n*(1-e2)+h)*np.sin(lat)])
    origin=ecef(frame['origin_longitude_deg'],frame['origin_latitude_deg'],frame['origin_ellipsoidal_height_m'])
    rotation=np.array([[-np.sin(lon0),np.cos(lon0),0],[-np.sin(lat0)*np.cos(lon0),-np.sin(lat0)*np.sin(lon0),np.cos(lat0)],
        [np.cos(lat0)*np.cos(lon0),np.cos(lat0)*np.sin(lon0),np.sin(lat0)]])
    assert np.max(abs(origin-np.array(frame['origin_ecef_m'])))<1e-7
    assert np.max(abs(rotation-np.array(frame['ecef_to_enu'])))<1e-14
    with np.load(source/'terrain_grid.npz') as grid:
        east=grid['east_m'];north=grid['north_m'];up=grid['up_m'];lon_grid=grid['longitude_deg'];lat_grid=grid['latitude_deg'];H=grid['orthometric_height_m']
    vertices=np.column_stack([east.ravel(),north.ravel(),up.ravel()]);nr,nc=up.shape
    ii=np.arange((nr-1)*nc).reshape(nr-1,nc)[:,:-1].ravel()
    triangles=np.concatenate([np.column_stack([ii,ii+nc,ii+1]),np.column_stack([ii+1,ii+nc,ii+nc+1])])
    va,vb,vc=[vertices[triangles[:,i]] for i in range(3)]
    lower=np.minimum(np.minimum(va,vb),vc);upper=np.maximum(np.maximum(va,vb),vc)
    def surface(x,y):
        mask=(lower[:,0]-1e-8<=x)&(upper[:,0]+1e-8>=x)&(lower[:,1]-1e-8<=y)&(upper[:,1]+1e-8>=y)
        aa,bb,cc=va[mask].astype(np.longdouble),vb[mask].astype(np.longdouble),vc[mask].astype(np.longdouble)
        ab,ac=bb-aa,cc-aa;px,py=x-aa[:,0],y-aa[:,1]
        det=ab[:,0]*ac[:,1]-ab[:,1]*ac[:,0]
        u=(px*ac[:,1]-py*ac[:,0])/det;v=(ab[:,0]*py-ab[:,1]*px)/det
        valid=(u>=-1e-9)&(v>=-1e-9)&(u+v<=1+1e-9)
        z=(aa[:,2]+u*ab[:,2]+v*ac[:,2])[valid]
        assert len(z)>0 and np.ptp(z)<1e-7
        return float(z.mean())
    rows=[];dms_error=0.
    for r in survey:
        x=[float(r[k]) for k in ['WCS84 x(m)','WCS84 y(m)','WCS84 z(m)']];lon,lat,h=inverse.transform(*x)
        def dms(value):
            d=int(value);m=int((value-d)*100+1e-10);s=((value-d)*100-m)*100
            assert 0<=m<60 and 0<=s<60.000001
            return d+m/60+s/3600
        dms_error=max(dms_error,abs(lon-dms(float(r['Longitude']))),abs(lat-dms(float(r['Latitude']))))
    assert dms_error<1e-7
    for r in placed:
        name=r['name'];lon,lat,h=inverse.transform(*raw[name]);enu=np.array(obs[name])
        assert np.max(abs(enu-np.array([float(r[k]) for k in ['east_m','north_m','up_m']])))<1e-12
        model_h=float(r['model_ellipsoidal_height_m'])
        from_geodetic=(ecef(lon,lat,model_h)-origin)@rotation.T
        lon2,lat2,h2=inverse.transform(*(enu@rotation+origin))
        _,_,N=geoid.transform(lon,lat,0.,errcheck=True)
        top=surface(enu[0],enu[1]);raw_enu=(raw[name]-origin)@rotation.T;raw_top=surface(raw_enu[0],raw_enu[1])
        row=dict(name=name,longitude_deg=lon,latitude_deg=lat,measured_ellipsoidal_height_m=h,
            geoid_N_m=N,measured_orthometric_height_m=h-N,model_ellipsoidal_height_m=model_h,
            model_orthometric_height_m=model_h-N,model_minus_measured_height_m=model_h-h,
            east_m=enu[0],north_m=enu[1],up_m=enu[2],independent_mesh_surface_up_m=top,
            model_clearance_m=enu[2]-top,raw_survey_clearance_m=raw_enu[2]-raw_top,
            horizontal_source_error_deg=max(abs(lon-float(r['longitude_deg'])),abs(lat-float(r['latitude_deg']))),
            horizontal_roundtrip_error_deg=max(abs(lon2-lon),abs(lat2-lat)),
            enu_transform_error_m=float(np.linalg.norm(from_geodetic-enu)),
            height_roundtrip_error_m=abs(h2-model_h),geoid_error_m=abs(N-float(r['geoid_undulation_m'])))
        assert row['horizontal_source_error_deg']<1e-10 and row['horizontal_roundtrip_error_deg']<1e-9
        assert row['enu_transform_error_m']<1e-5 and row['height_roundtrip_error_m']<1e-5
        assert abs(row['model_clearance_m']-1.)<1e-7 and row['geoid_error_m']<1e-6
        assert abs(h-float(r['measured_height_m']))<1e-5
        rows.append(row)
    with (out/'stations_80_audit.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    e01=next(r for r in survey if r['Node']=='E01')
    reference=np.array([float(e01[k]) for k in ['WCS84 x(m)','WCS84 y(m)','WCS84 z(m)']])
    e01_delta=(raw['E01']-reference)@rotation.T
    # The published relative table uses two E01 origins, and EW entries rotate
    # at each station (as in its notebook), whereas NS uses the survey origin.
    relative={line.split()[0]:np.array(list(map(float,line.split()[1:]))) for line in (source/'antenna_location_21cma.txt').read_text().splitlines()}
    def rotation_at(x):
        lon,lat,_=inverse.transform(*x);lon,lat=np.deg2rad([lon,lat])
        return np.array([[-np.sin(lon),np.cos(lon),0],[-np.sin(lat)*np.cos(lon),-np.sin(lat)*np.sin(lon),np.cos(lat)],
            [np.cos(lat)*np.cos(lon),np.cos(lat)*np.sin(lon),np.sin(lat)]])
    relative_error={}
    for arm in 'EWNS':
        differences=[]
        for name,x in raw.items():
            if name[0]!=arm:continue
            anchor=raw['E01'] if arm in 'EW' else reference
            basis=rotation_at(x if arm in 'EW' else reference)
            differences.append(float(np.linalg.norm((x-anchor)@basis.T-relative[name])))
        relative_error[arm]=max(differences)
    assert max(relative_error.values())<1e-5
    result=dict(count=80,names_complete_unique=True,all_inside_dem=True,
        all_native_model_heights_dem_plus_1m=True,model_heights_are_survey_measurements=False,
        packed_dms_max_error_deg=dms_error,
        maximum_coordinate_error_m=max(r['enu_transform_error_m'] for r in rows),
        maximum_horizontal_roundtrip_error_deg=max(r['horizontal_roundtrip_error_deg'] for r in rows),
        clearance_range_m=[min(r['model_clearance_m'] for r in rows),max(r['model_clearance_m'] for r in rows)],
        model_minus_measured_range_m=[min(r['model_minus_measured_height_m'] for r in rows),max(r['model_minus_measured_height_m'] for r in rows)],
        raw_measurements_below_model_surface=[r['name'] for r in rows if r['raw_survey_clearance_m']<0],
        E01_dump_centroid_minus_survey_reference_ENU_m=e01_delta.tolist(),
        EW_centroid_recalculation_max_error_m=centroid_error,
        EW_station_measurement_counts={name:len(xyz) for name,xyz in measured.items() if name in raw},
        relative_table_reproduction_error_m=relative_error,
        relative_table_warning='EW values use the dump E01 centroid and each station basis; NS uses the 2016 E01 survey reference and its basis. Do not interpret the merged relative table as one common ENU frame.',
        raw_measurements_below_model_surface_count=int(sum(r['raw_survey_clearance_m']<0 for r in rows)),
        source_sha256={p.name:digest(p) for p in source.iterdir() if p.is_file()},
        height_policy='Preserve surveyed longitude/latitude; WGS84 ellipsoidal versus EGM96 orthometric heights distinguished; model antenna is 1 m above triangulated DEM along ENU Up.')
    (out/'stations_80_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    fig,ax=plt.subplots(figsize=(10,8),constrained_layout=True)
    im=ax.pcolormesh(east[::3,::3]/1000,north[::3,::3]/1000,up[::3,::3],shading='auto',cmap='terrain',rasterized=True)
    perimeter=np.genfromtxt(out/'perimeter.csv',delimiter=',',names=True)
    ax.plot(np.r_[perimeter['x0_m'],perimeter['x0_m'][0]]/1000,np.r_[perimeter['y0_m'],perimeter['y0_m'][0]]/1000,'k-',lw=1,label='DEM coverage boundary')
    colors=dict(E='crimson',W='darkorange',N='navy',S='purple')
    for arm in 'EWNS':
        selected=[r for r in rows if r['name'][0]==arm]
        ax.scatter([r['east_m']/1000 for r in selected],[r['north_m']/1000 for r in selected],s=17,c=colors[arm],label=arm+' arm: 20 stations')
        # All centres are drawn; label arm ends so the full-DEM view stays legible.
        # The complete names and coordinates are in stations_80_audit.csv.
        for r in selected:
            if r['name'].endswith('20'):
                offset=dict(E=(5,7),W=(-28,7),N=(5,7),S=(5,-12))[arm]
                ax.annotate(r['name'],(r['east_m']/1000,r['north_m']/1000),xytext=offset,textcoords='offset points',fontsize=8)
    ax.set(xlabel='East [km]',ylabel='North [km]',title='21CMA: all 80 station centres in the DEM frame',aspect='equal')
    ax.legend(fontsize=9);fig.colorbar(im,ax=ax,label='Terrain height, ENU Up [m]');fig.savefig(out/'01_21cma_80_stations.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(2,1,figsize=(12,7),constrained_layout=True)
    index=np.arange(80);axes[0].bar(index,[r['model_minus_measured_height_m'] for r in rows],color=[colors[r['name'][0]] for r in rows])
    axes[0].axhline(0,color='k',lw=.6);axes[0].set(ylabel='Model minus surveyed height [m]',title='Height policy audit: DEM + 1 m is not surveyed height')
    axes[1].plot(index,[r['model_clearance_m'] for r in rows],label='Model station above exact DEM',color='green')
    axes[1].plot(index,[r['raw_survey_clearance_m'] for r in rows],'.',label='Surveyed centre relative to DEM',color='black');axes[1].axhline(0,color='gray',lw=.6)
    axes[1].set(ylabel='ENU vertical clearance [m]');axes[1].legend()
    for ax in axes:ax.set_xticks(index[::2]);ax.set_xticklabels([r['name'] for r in rows][::2],rotation=90,fontsize=7);ax.grid(axis='y',alpha=.2)
    fig.savefig(out/'02_station_height_audit.png',dpi=180);plt.close(fig)
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()

import numpy as np
import pytest
from pyproj import Transformer
from corsika_terrain.terrain_array import enu_rotation, packed_dms_to_degrees, place_on_dem


def test_survey_angles_are_packed_dms_not_decimal_degrees():
    np.testing.assert_allclose(packed_dms_to_degrees([86.4302579725,42.5526742053]),
        [86+43/60+2.579725/3600,42+55/60+26.742053/3600],rtol=0,atol=1e-12)


def test_invalid_minutes_rejected():
    with pytest.raises(ValueError):
        packed_dms_to_degrees(42.65)


def test_fixed_enu_frame_preserves_lengths_and_orientation():
    matrix=enu_rotation(42.9321241707485,86.70041679000809)
    np.testing.assert_allclose(matrix@matrix.T,np.eye(3),atol=1e-14)
    assert abs(np.linalg.det(matrix)-1)<1e-14
    delta=np.array([2000.,-800.,432.])
    np.testing.assert_allclose(np.linalg.norm(matrix@delta),np.linalg.norm(delta),rtol=1e-14)


def test_model_height_preserves_geodetic_horizontal_position():
    lon,lat=np.array([86.70,86.71]),np.array([42.93,42.94])
    matrix=enu_rotation(42.93,86.70)
    origin=np.array(Transformer.from_crs(4979,4978,always_xy=True).transform(86.70,42.93,2675.))
    def surface(e,n):
        return np.ma.array(.01*e+.02*n)
    enu,h=place_on_dem(lon,lat,np.array([2590.,2600.]),matrix,origin,surface)
    np.testing.assert_allclose(enu[:,2]-surface(enu[:,0],enu[:,1]),1.,atol=1e-8)
    xyz=enu@matrix+origin
    longitude,latitude,_=Transformer.from_crs(4978,4979,always_xy=True).transform(*xyz.T)
    np.testing.assert_allclose(longitude,lon,atol=1e-10,rtol=0)
    np.testing.assert_allclose(latitude,lat,atol=1e-10,rtol=0)


def test_model_height_rejects_dem_extrapolation():
    matrix=enu_rotation(42.93,86.70)
    origin=np.array(Transformer.from_crs(4979,4978,always_xy=True).transform(86.70,42.93,2675.))
    with pytest.raises(ValueError,match="outside available DEM"):
        place_on_dem(np.array([86.7]),np.array([42.93]),np.array([2600.]),matrix,origin,
            lambda e,n: np.ma.masked_all(e.shape))


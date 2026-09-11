"""Audit 80 station centres in the DEM frame without silently changing heights."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
from pyproj import Transformer
import yaml
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from .terrain import sha256_file


def packed_dms_to_degrees(value):
    """Survey CSV uses DD.MMSSssss, not decimal degrees."""
    value = np.asarray(value, dtype=float)
    absolute = np.abs(value)
    degrees = np.floor(absolute)
    minutes_decimal = (absolute - degrees) * 100
    minutes = np.floor(minutes_decimal + 1e-10)
    seconds = (minutes_decimal - minutes) * 100
    if np.any(minutes >= 60) or np.any(seconds >= 60.000001):
        raise ValueError("invalid packed DMS")
    return np.sign(value) * (degrees + minutes / 60 + seconds / 3600)


def enu_rotation(latitude_deg, longitude_deg):
    lat, lon = np.deg2rad([latitude_deg, longitude_deg])
    return np.array([
        [-np.sin(lon), np.cos(lon), 0],
        [-np.sin(lat)*np.cos(lon), -np.sin(lat)*np.sin(lon), np.cos(lat)],
        [np.cos(lat)*np.cos(lon), np.cos(lat)*np.sin(lon), np.sin(lat)],
    ])


def place_on_dem(lon, lat, height, rotation, origin, surface, above_ground_m=1.0):
    """Keep geodetic horizontal coordinates; solve height for DEM + AGL."""
    transformer=Transformer.from_crs(4979,4978,always_xy=True)
    h=np.asarray(height,dtype=float).copy()
    for _ in range(8):
        xyz=np.column_stack(transformer.transform(lon,lat,h))
        enu=(xyz-origin)@rotation.T
        top=surface(enu[:,0],enu[:,1])
        if np.any(np.ma.getmaskarray(top)):
            raise ValueError("cannot place observer outside available DEM")
        residual=np.asarray(top)+above_ground_m-enu[:,2]
        h+=residual
        if np.max(np.abs(residual))<1.e-8:
            return enu,h
    raise ValueError("DEM observer height iteration did not converge")


def load_station_centres(directory):
    directory = Path(directory)
    ew_path = directory / "21cma_antenna_table/dump.txt"
    ns_path = directory / "20160810.csv"
    ew = np.loadtxt(ew_path, dtype=str)
    names, xyz = ew[:, 0].tolist(), ew[:, 1:].astype(float).tolist()
    sources = [str(ew_path.resolve())] * len(names)
    with ns_path.open() as stream:
        survey = list(csv.DictReader(stream))
    keys = ["WCS84 x(m)", "WCS84 y(m)", "WCS84 z(m)"]
    for row in survey:
        if row["Node"].startswith(("N", "S")):
            names.append(row["Node"])
            xyz.append([float(row[key]) for key in keys])
            sources.append(str(ns_path.resolve()))
    expected = {f"{arm}{i:02d}" for arm in "EWNS" for i in range(1, 21)}
    if len(names) != 80 or set(names) != expected:
        raise ValueError("expected exactly one centre for each E/W/N/S01–20")
    transform = Transformer.from_crs(4978, 4979, always_xy=True)
    survey_xyz = np.array([[float(row[key]) for key in keys] for row in survey])
    lon, lat, _ = transform.transform(*survey_xyz.T)
    residual = np.max(np.abs(np.column_stack([lon, lat]) - packed_dms_to_degrees(
        [[float(row["Longitude"]), float(row["Latitude"])] for row in survey])))
    if residual > 1e-7:
        raise ValueError("survey packed-DMS and ECEF coordinates disagree")
    return names, np.asarray(xyz), sources, {
        "source_sha256": {str(p.resolve()): sha256_file(p) for p in [ew_path, ns_path]},
        "csv_angle_encoding": "DD.MMSSssss; not decimal degrees",
        "csv_dms_vs_ecef_maximum_residual_deg": float(residual),
        "ew_definition": "mean station ECEF centres from dump.txt",
        "ns_definition": "survey station ECEF points from 20160810.csv",
        "height_policy": "raw ECEF retained; no geoid correction or DEM snapping",
    }



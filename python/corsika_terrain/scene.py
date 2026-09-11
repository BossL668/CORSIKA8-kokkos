"""Admit a prepared geographic product and export the beta5 terrain scene.

Uses the migrated mountain coordinate/height implementation, with no copying
of DEM tiles or simulation data. This command does NOT launch a shower.
"""
import argparse
from pathlib import Path
import math

import numpy as np
import yaml

from .terrain import sha256_file
from .terrain_download import geoid_transform
from .terrain_validation import _terrain_interpolator


def scene_from_product(product):
    product = Path(product).resolve()
    manifest_path = product / "terrain_manifest.yaml"
    manifest = yaml.safe_load(manifest_path.read_text())
    if manifest["status"] != "geometry_and_antenna_checks_passed":
        raise ValueError("terrain preparation has not passed its gates")
    for name, digest in manifest["output_sha256"].items():
        path = (product / name).resolve()
        if path.parent != product or not path.is_file() or sha256_file(path) != digest:
            raise ValueError(f"prepared terrain product checksum mismatch: {name}")
    frame = manifest["frame"]
    transform, _ = geoid_transform(manifest["configuration"]["cache_directory"], offline=True)
    _, _, undulation = transform.transform(frame["origin_longitude_deg"],
                                          frame["origin_latitude_deg"], 0., errcheck=True)
    altitude = frame["origin_ellipsoidal_height_m"] - undulation
    if not math.isfinite(altitude) or not 0 <= altitude < 7000:
        raise ValueError("prepared origin has unsupported ASL height")
    geometry = yaml.safe_load((product / "corsika_geometry.yaml").read_text())["geometry"]
    # Locate the reference against the exact triangulated surface, not a
    # bilinear DEM surface. Original preparation sets the ENU origin here.
    with np.load(product / "terrain_grid.npz", allow_pickle=False) as grid:
        surface = _terrain_interpolator(grid["east_m"], grid["north_m"], grid["up_m"])
        height = surface(0., 0.)
        if np.ma.is_masked(height) or not np.isfinite(height):
            raise ValueError("origin not covered by prepared mesh")
    geometry = dict(geometry)
    geometry["mesh_path"] = str((product / "terrain_enu.ply").resolve())
    geometry.pop("air_world_radius_m", None)
    # This belongs to the old uniform-air geometry product. In a native
    # atmosphere the authoritative refractivity is the per-layer model below.
    geometry.pop("air_refractive_index", None)
    geometry["rock_reference_enu_m"] = [0., 0., float(height)-1.]
    observers = yaml.safe_load((product / "observers.yaml").read_text())["radio"]["observers"]
    return dict(schema_version=1,
        site=dict(latitude_deg=frame["origin_latitude_deg"], longitude_deg=frame["origin_longitude_deg"],
                  origin_ellipsoidal_height_m=frame["origin_ellipsoidal_height_m"],
                  geoid_undulation_m=float(undulation), coordinate_frame="geographic_ENU"),
        atmosphere=dict(model="us_standard_bk", origin_altitude_asl_m=float(altitude),
                        sea_level_refractive_index=1.000327),
        geometry=geometry, radio=dict(observers=observers),
        provenance=dict(prepared_product=str(product), manifest_sha256=sha256_file(manifest_path),
                        mesh_sha256=sha256_file(product/"terrain_enu.ply"),
                        height_policy=manifest["configuration"]["antennas"].get("height_mode")))


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    scene = scene_from_product(args.prepared)
    # Exclusive creation; never rewrite an existing approved scene.
    with args.output.open("x") as stream:
        yaml.safe_dump(scene, stream, sort_keys=False)
    print(f"Validated terrain scene: {args.output} ({len(scene['radio']['observers'])} observers)")


if __name__ == "__main__":
    main()

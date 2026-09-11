"""Build a metre-scale, watertight CORSIKA terrain mesh from a GeoTIFF DEM.

The top surface follows the DEM.  Vertical skirts and a flat bottom close the
solid so CORSIKA can unambiguously choose between the uniform rock child and
the surrounding air node.  Geographic coordinates are transformed through
ECEF before rotation into the local East-North-Up coordinate system.
"""

from __future__ import annotations

import argparse
import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import yaml
from PIL import Image
from pyproj import CRS, Transformer


@dataclass(frozen=True)
class DemRaster:
    path: Path
    elevation_m: np.ndarray
    projected_x_m: np.ndarray
    projected_y_m: np.ndarray
    crs: CRS
    nodata: Optional[float]
    pixel_is_area: bool
    transform: np.ndarray


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def _geokey_values(directory: Iterable[int]) -> Dict[int, int]:
    values = list(directory)
    if len(values) < 4:
        raise ValueError("invalid GeoKeyDirectoryTag")
    number = int(values[3])
    result: Dict[int, int] = {}
    for offset in range(number):
        key, location, count, value = values[4 + 4 * offset : 8 + 4 * offset]
        if int(location) == 0 and int(count) == 1:
            result[int(key)] = int(value)
    return result


def read_geotiff(path: Path) -> DemRaster:
    path = path.resolve()
    with Image.open(path) as image:
        elevation = np.asarray(image, dtype=np.float64)
        transform_tag = image.tag_v2.get(34264)
        geokeys_tag = image.tag_v2.get(34735)
        nodata_tag = image.tag_v2.get(42113)
    if transform_tag is None or len(transform_tag) != 16:
        raise ValueError("GeoTIFF must contain a 4x4 ModelTransformationTag")
    if geokeys_tag is None:
        raise ValueError("GeoTIFF must contain a GeoKeyDirectoryTag")
    transform = np.asarray(transform_tag, dtype=np.float64).reshape(4, 4)
    keys = _geokey_values(geokeys_tag)
    epsg = keys.get(3072)
    if epsg is None:
        raise ValueError("GeoTIFF must declare ProjectedCSTypeGeoKey (3072)")
    crs = CRS.from_epsg(epsg)
    nodata = float(nodata_tag) if nodata_tag is not None else None
    invalid = ~np.isfinite(elevation)
    if nodata is not None:
        invalid |= elevation == nodata
    if np.any(invalid):
        raise ValueError(
            f"DEM contains {int(np.count_nonzero(invalid))} nodata/non-finite cells; "
            "fill them before constructing a closed terrain solid"
        )
    rows, cols = np.indices(elevation.shape, dtype=np.float64)
    # PixelIsArea GeoTIFFs locate the affine origin at the outer pixel corner.
    # DEM samples are represented at pixel centres.
    pixel_is_area = keys.get(1025, 1) == 1
    offset = 0.5 if pixel_is_area else 0.0
    homogeneous = np.stack(
        [cols + offset, rows + offset, np.zeros_like(cols), np.ones_like(cols)],
        axis=0,
    ).reshape(4, -1)
    projected = transform @ homogeneous
    x = projected[0].reshape(elevation.shape)
    y = projected[1].reshape(elevation.shape)
    return DemRaster(path, elevation, x, y, crs, nodata, pixel_is_area, transform)


def bilinear_dem_height(dem: DemRaster, x_m: float, y_m: float) -> float:
    # Many DEM GeoTIFFs deliberately use a zero Z row in their 4x4 model
    # transform.  Invert only the horizontal affine block.
    horizontal = dem.transform[:2, :2]
    pixel = np.linalg.solve(
        horizontal,
        np.array([x_m, y_m]) - dem.transform[:2, 3],
    )
    offset = 0.5 if dem.pixel_is_area else 0.0
    col = float(pixel[0] - offset)
    row = float(pixel[1] - offset)
    if not (0.0 <= row <= dem.elevation_m.shape[0] - 1 and
            0.0 <= col <= dem.elevation_m.shape[1] - 1):
        raise ValueError("ENU origin lies outside the DEM sample-centre footprint")
    r0 = min(int(math.floor(row)), dem.elevation_m.shape[0] - 2)
    c0 = min(int(math.floor(col)), dem.elevation_m.shape[1] - 2)
    fr = row - r0
    fc = col - c0
    block = dem.elevation_m[r0 : r0 + 2, c0 : c0 + 2]
    return float(
        (1 - fr) * (1 - fc) * block[0, 0]
        + (1 - fr) * fc * block[0, 1]
        + fr * (1 - fc) * block[1, 0]
        + fr * fc * block[1, 1]
    )


def projected_to_enu(
    crs: CRS,
    x_m: np.ndarray,
    y_m: np.ndarray,
    height_m: np.ndarray,
    origin_latitude_deg: float,
    origin_longitude_deg: float,
    origin_altitude_m: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    to_ecef = Transformer.from_crs(crs, CRS.from_epsg(4978), always_xy=True)
    geodetic_to_ecef = Transformer.from_crs(
        CRS.from_epsg(4979), CRS.from_epsg(4978), always_xy=True
    )
    x_ecef, y_ecef, z_ecef = to_ecef.transform(x_m, y_m, height_m)
    x0, y0, z0 = geodetic_to_ecef.transform(
        origin_longitude_deg, origin_latitude_deg, origin_altitude_m
    )
    dx = np.asarray(x_ecef) - x0
    dy = np.asarray(y_ecef) - y0
    dz = np.asarray(z_ecef) - z0
    latitude = math.radians(origin_latitude_deg)
    longitude = math.radians(origin_longitude_deg)
    sin_lat, cos_lat = math.sin(latitude), math.cos(latitude)
    sin_lon, cos_lon = math.sin(longitude), math.cos(longitude)
    east = -sin_lon * dx + cos_lon * dy
    north = -sin_lat * cos_lon * dx - sin_lat * sin_lon * dy + cos_lat * dz
    up = cos_lat * cos_lon * dx + cos_lat * sin_lon * dy + sin_lat * dz
    return np.asarray(east), np.asarray(north), np.asarray(up)


def _sample_indices(size: int, stride: int) -> np.ndarray:
    values = np.arange(0, size, stride, dtype=np.int64)
    if values[-1] != size - 1:
        values = np.append(values, size - 1)
    return values


def build_watertight_solid(
    east_m: np.ndarray,
    north_m: np.ndarray,
    up_m: np.ndarray,
    bottom_depth_m: float,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, int | float | bool]]:
    if east_m.shape != north_m.shape or east_m.shape != up_m.shape:
        raise ValueError("terrain coordinate arrays must have one common shape")
    n_rows, n_cols = up_m.shape
    if n_rows < 2 or n_cols < 2:
        raise ValueError("terrain grid must contain at least 2x2 samples")
    if not bottom_depth_m > 0:
        raise ValueError("bottom_depth_m must be positive")
    top = np.column_stack([east_m.ravel(), north_m.ravel(), up_m.ravel()])
    top_faces = []
    for row in range(n_rows - 1):
        for col in range(n_cols - 1):
            nw = row * n_cols + col
            ne = nw + 1
            sw = (row + 1) * n_cols + col
            se = sw + 1
            top_faces.extend(((nw, sw, ne), (ne, sw, se)))

    # Counter-clockwise perimeter as seen from +U: south, east, north, west.
    perimeter = []
    perimeter.extend((n_rows - 1) * n_cols + col for col in range(n_cols))
    perimeter.extend(row * n_cols + n_cols - 1 for row in range(n_rows - 2, -1, -1))
    perimeter.extend(col for col in range(n_cols - 2, -1, -1))
    perimeter.extend(row * n_cols for row in range(1, n_rows - 1))
    perimeter = np.asarray(perimeter, dtype=np.int64)
    bottom_u = float(np.min(up_m) - bottom_depth_m)
    bottom_ring = top[perimeter].copy()
    bottom_ring[:, 2] = bottom_u
    bottom_offset = len(top)
    centre_index = bottom_offset + len(perimeter)
    centre = np.array(
        [[np.mean(bottom_ring[:, 0]), np.mean(bottom_ring[:, 1]), bottom_u]],
        dtype=np.float64,
    )
    vertices = np.vstack([top, bottom_ring, centre])
    faces = list(top_faces)
    for index in range(len(perimeter)):
        following = (index + 1) % len(perimeter)
        top_i = int(perimeter[index])
        top_j = int(perimeter[following])
        bottom_i = bottom_offset + index
        bottom_j = bottom_offset + following
        faces.extend(((top_i, bottom_i, bottom_j), (top_i, bottom_j, top_j)))
        faces.append((centre_index, bottom_j, bottom_i))
    faces_array = np.asarray(faces, dtype=np.uint32)

    a = vertices[faces_array[:, 0]]
    b = vertices[faces_array[:, 1]]
    c = vertices[faces_array[:, 2]]
    signed_volume = float(np.sum(np.einsum("ij,ij->i", a, np.cross(b, c))) / 6.0)
    if signed_volume < 0:
        faces_array[:, [1, 2]] = faces_array[:, [2, 1]]
        signed_volume = -signed_volume
        b = vertices[faces_array[:, 1]]
        c = vertices[faces_array[:, 2]]
    twice_area = np.linalg.norm(np.cross(b - a, c - a), axis=1)
    edge_parts = []
    for left, right in ((0, 1), (1, 2), (2, 0)):
        edge_parts.append(np.sort(faces_array[:, [left, right]], axis=1))
    edges = np.concatenate(edge_parts, axis=0)
    edge_view = np.ascontiguousarray(edges).view(
        np.dtype([("left", np.uint32), ("right", np.uint32)])
    ).reshape(-1)
    _, edge_counts = np.unique(edge_view, return_counts=True)
    nonmanifold = int(np.count_nonzero(edge_counts != 2))
    qa: Dict[str, int | float | bool] = {
        "vertex_count": int(len(vertices)),
        "face_count": int(len(faces_array)),
        "top_face_count": int(len(top_faces)),
        "side_face_count": int(2 * len(perimeter)),
        "bottom_face_count": int(len(perimeter)),
        "unique_edge_count": int(len(edge_counts)),
        "nonmanifold_edge_count": nonmanifold,
        "watertight": nonmanifold == 0,
        "degenerate_face_count": int(np.count_nonzero(twice_area <= 1.0e-12)),
        "minimum_triangle_area_m2": float(0.5 * np.min(twice_area)),
        "signed_volume_m3": signed_volume,
        "bottom_u_m": bottom_u,
    }
    return vertices, faces_array, qa


def write_binary_ply(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        "comment metre-scale CORSIKA local ENU terrain solid\n"
        f"element vertex {len(vertices)}\n"
        "property double x\nproperty double y\nproperty double z\n"
        f"element face {len(faces)}\n"
        "property list uchar uint vertex_indices\n"
        "end_header\n"
    ).encode("ascii")
    vertex_data = np.asarray(vertices, dtype="<f8")
    face_dtype = np.dtype([("count", "u1"), ("indices", "<u4", (3,))])
    face_data = np.empty(len(faces), dtype=face_dtype)
    face_data["count"] = 3
    face_data["indices"] = faces
    with path.open("wb") as stream:
        stream.write(header)
        vertex_data.tofile(stream)
        face_data.tofile(stream)


def _plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_plain(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _plot_geometry(
    diagnostics: Path,
    elevation: np.ndarray,
    east: np.ndarray,
    north: np.ndarray,
    up: np.ndarray,
    bottom_u: float,
) -> None:
    diagnostics.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9.2, 7.2), constrained_layout=True)
    image = ax.pcolormesh(east, north, elevation, shading="auto", cmap="terrain")
    contours = ax.contour(east, north, elevation, levels=12, colors="k", linewidths=0.35, alpha=0.5)
    ax.clabel(contours, fontsize=6, fmt="%.0f")
    ax.scatter([0], [0], marker="+", s=120, linewidths=2, color="red", label="ENU origin")
    ax.set_aspect("equal")
    ax.set_xlabel("CORSIKA East [m]")
    ax.set_ylabel("CORSIKA North [m]")
    ax.set_title("S13-1500 DEM transformed to the local CORSIKA ENU frame")
    ax.legend(loc="best")
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("DEM elevation [m ASL]")
    fig.savefig(diagnostics / "01_dem_to_corsika_enu.png", dpi=180)
    plt.close(fig)

    fig = plt.figure(figsize=(12.0, 8.0), constrained_layout=True)
    ax3d = fig.add_subplot(2, 2, (1, 3), projection="3d")
    row_step = max(1, up.shape[0] // 60)
    col_step = max(1, up.shape[1] // 60)
    surface = ax3d.plot_surface(
        east[::row_step, ::col_step], north[::row_step, ::col_step],
        up[::row_step, ::col_step], cmap="terrain", linewidth=0, antialiased=True,
    )
    corners = [(0, 0), (0, -1), (-1, -1), (-1, 0), (0, 0)]
    for (r0, c0), (r1, c1) in zip(corners[:-1], corners[1:]):
        ax3d.plot(
            [east[r0, c0], east[r1, c1]],
            [north[r0, c0], north[r1, c1]],
            [bottom_u, bottom_u], color="0.25", lw=1.0,
        )
        ax3d.plot(
            [east[r0, c0], east[r0, c0]],
            [north[r0, c0], north[r0, c0]],
            [bottom_u, up[r0, c0]], color="0.25", lw=0.8,
        )
    ax3d.set_xlabel("E [m]")
    ax3d.set_ylabel("N [m]")
    ax3d.set_zlabel("U [m]")
    ax3d.set_title("Watertight terrain solid (top, skirts, flat bottom)")
    fig.colorbar(surface, ax=ax3d, shrink=0.55, label="U [m]")

    middle_row = up.shape[0] // 2
    middle_col = up.shape[1] // 2
    ax_e = fig.add_subplot(2, 2, 2)
    ax_e.plot(east[middle_row], up[middle_row], color="tab:brown")
    ax_e.fill_between(east[middle_row], bottom_u, up[middle_row], color="tab:brown", alpha=0.22)
    ax_e.set_xlabel("E [m]")
    ax_e.set_ylabel("U [m]")
    ax_e.set_title("Central east-west cross-section")
    ax_e.grid(alpha=0.25)
    ax_n = fig.add_subplot(2, 2, 4)
    ax_n.plot(north[:, middle_col], up[:, middle_col], color="tab:green")
    ax_n.fill_between(north[:, middle_col], bottom_u, up[:, middle_col], color="tab:green", alpha=0.22)
    ax_n.set_xlabel("N [m]")
    ax_n.set_ylabel("U [m]")
    ax_n.set_title("Central north-south cross-section")
    ax_n.grid(alpha=0.25)
    fig.savefig(diagnostics / "02_watertight_terrain_mesh.png", dpi=180)
    plt.close(fig)


def build_terrain_product(
    dem_path: Path,
    mesh_path: Path,
    manifest_path: Path,
    diagnostics_path: Path,
    grid_path: Path,
    stride: int,
    origin_latitude_deg: float,
    origin_longitude_deg: float,
    origin_altitude_m: Optional[float],
    bottom_depth_m: float,
) -> Dict[str, Any]:
    if stride < 1:
        raise ValueError("stride must be >= 1")
    dem = read_geotiff(dem_path)
    origin_to_projected = Transformer.from_crs(
        CRS.from_epsg(4326), dem.crs, always_xy=True
    )
    origin_x, origin_y = origin_to_projected.transform(
        origin_longitude_deg, origin_latitude_deg
    )
    terrain_origin_altitude = bilinear_dem_height(dem, origin_x, origin_y)
    automatic_origin_altitude = origin_altitude_m is None
    if automatic_origin_altitude:
        origin_altitude_m = terrain_origin_altitude

    rows = _sample_indices(dem.elevation_m.shape[0], stride)
    cols = _sample_indices(dem.elevation_m.shape[1], stride)
    elevation = dem.elevation_m[np.ix_(rows, cols)]
    projected_x = dem.projected_x_m[np.ix_(rows, cols)]
    projected_y = dem.projected_y_m[np.ix_(rows, cols)]
    east, north, up = projected_to_enu(
        dem.crs, projected_x, projected_y, elevation,
        origin_latitude_deg, origin_longitude_deg, origin_altitude_m,
    )
    # A coarse LOD top surface need not reproduce the full-resolution bilinear
    # DEM exactly at an arbitrary origin.  In automatic mode, shift the ENU
    # origin altitude so U=0 lies on the *actual mesh used by CORSIKA*.
    lod_origin_correction_m = 0.0
    if automatic_origin_altitude:
        n_rows, n_cols = elevation.shape
        triangles = []
        for row in range(n_rows - 1):
            for col in range(n_cols - 1):
                nw = row * n_cols + col
                ne = nw + 1
                sw = (row + 1) * n_cols + col
                se = sw + 1
                triangles.extend(((nw, sw, ne), (ne, sw, se)))
        interpolation = mtri.LinearTriInterpolator(
            mtri.Triangulation(
                east.ravel(), north.ravel(), np.asarray(triangles, dtype=np.int32)
            ),
            up.ravel(),
        )
        lod_origin_correction_m = float(interpolation(0.0, 0.0))
        origin_altitude_m += lod_origin_correction_m
        east, north, up = projected_to_enu(
            dem.crs, projected_x, projected_y, elevation,
            origin_latitude_deg, origin_longitude_deg, origin_altitude_m,
        )
    vertices, faces, qa = build_watertight_solid(
        east, north, up, bottom_depth_m
    )
    if not qa["watertight"] or qa["degenerate_face_count"]:
        raise RuntimeError(f"terrain mesh QA failed: {qa}")
    write_binary_ply(mesh_path, vertices, faces)
    grid_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        grid_path,
        east_m=east,
        north_m=north,
        up_m=up,
        elevation_asl_m=elevation,
        projected_x_m=projected_x,
        projected_y_m=projected_y,
        row_indices=rows,
        column_indices=cols,
    )
    _plot_geometry(diagnostics_path, elevation, east, north, up, float(qa["bottom_u_m"]))
    spacing_x = float(np.linalg.norm(vertices[1, :2] - vertices[0, :2]))
    spacing_y = float(
        np.linalg.norm(vertices[len(cols), :2] - vertices[0, :2])
    )
    manifest: Dict[str, Any] = {
        "schema_version": 1,
        "status": "geometry_qa_passed",
        "source": {
            "dem": str(dem.path),
            "sha256": sha256_file(dem.path),
            "crs": dem.crs.to_string(),
            "width_pixels": int(dem.elevation_m.shape[1]),
            "height_pixels": int(dem.elevation_m.shape[0]),
            "pixel_is_area": dem.pixel_is_area,
            "elevation_min_m": float(np.min(dem.elevation_m)),
            "elevation_max_m": float(np.max(dem.elevation_m)),
            "vertical_datum": "not_declared_in_source_GeoTIFF",
            "provenance": "JAXA ALOS AW3D30 V2.2; user-supplied S13-1500 patch",
        },
        "corsika_coordinate_system": {
            "kind": "local_ENU_via_ECEF_rotation",
            "origin_latitude_deg": origin_latitude_deg,
            "origin_longitude_deg": origin_longitude_deg,
            "origin_altitude_m": float(origin_altitude_m),
            "source_dem_bilinear_altitude_at_origin_m": terrain_origin_altitude,
            "lod_mesh_origin_altitude_correction_m": lod_origin_correction_m,
            "origin_altitude_semantics": (
                "actual_LOD_mesh_surface" if automatic_origin_altitude
                else "user_supplied"
            ),
            "projected_origin_x_m": float(origin_x),
            "projected_origin_y_m": float(origin_y),
        },
        "lod": {
            "source_grid_stride": stride,
            "selected_rows": int(len(rows)),
            "selected_columns": int(len(cols)),
            "approximate_east_spacing_m": spacing_x,
            "approximate_north_spacing_m": spacing_y,
            "reason": "source AW3D30 information scale and safe CORSIKA mesh memory",
        },
        "mesh": {
            "path": str(mesh_path.resolve()),
            "sha256": sha256_file(mesh_path),
            "format": "binary_little_endian_PLY",
            "coordinate_unit": "m",
            "east_bounds_m": [float(np.min(vertices[:, 0])), float(np.max(vertices[:, 0]))],
            "north_bounds_m": [float(np.min(vertices[:, 1])), float(np.max(vertices[:, 1]))],
            "up_bounds_m": [float(np.min(vertices[:, 2])), float(np.max(vertices[:, 2]))],
            **qa,
        },
        "grid_product": {
            "path": str(grid_path.resolve()),
            "sha256": sha256_file(grid_path),
        },
        "acceptance": {
            "watertight": bool(qa["watertight"]),
            "no_degenerate_faces": int(qa["degenerate_face_count"]) == 0,
            "positive_signed_volume": float(qa["signed_volume_m3"]) > 0,
            "origin_inside_sample_footprint": True,
        },
        "scope_warning": (
            "This is the supplied S13-1500 DEM patch centred near "
            "42.932124N, 86.700417E. It is not the full 21CMA array terrain, "
            "and it conflicts with the older beta4 nominal site coordinates."
        ),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        yaml.safe_dump(_plain(manifest), allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return manifest


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Convert a GeoTIFF DEM into a watertight CORSIKA ENU PLY solid"
    )
    parser.add_argument("--dem", required=True, type=Path)
    parser.add_argument("--output-mesh", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--grid", required=True, type=Path)
    parser.add_argument("--diagnostics", required=True, type=Path)
    parser.add_argument("--stride", type=int, default=6)
    parser.add_argument("--origin-latitude-deg", type=float, required=True)
    parser.add_argument("--origin-longitude-deg", type=float, required=True)
    parser.add_argument("--origin-altitude-m", type=float)
    parser.add_argument("--bottom-depth-m", type=float, default=100.0)
    args = parser.parse_args(argv)
    try:
        manifest = build_terrain_product(
            args.dem, args.output_mesh, args.manifest, args.diagnostics,
            args.grid, args.stride, args.origin_latitude_deg,
            args.origin_longitude_deg, args.origin_altitude_m,
            args.bottom_depth_m,
        )
    except Exception as error:
        print(f"c8-mountain-terrain-build: error: {error}")
        return 2
    print(yaml.safe_dump(_plain(manifest["acceptance"]), sort_keys=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


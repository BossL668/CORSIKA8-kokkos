"""Mesh-consistent linear terrain interpolation, migrated from mountain."""
import numpy as np
import matplotlib.tri as mtri

def _terrain_triangulation(east: np.ndarray, north: np.ndarray, up: np.ndarray):
    n_rows, n_cols = up.shape
    triangles = []
    for row in range(n_rows - 1):
        for col in range(n_cols - 1):
            nw = row * n_cols + col
            ne = nw + 1
            sw = (row + 1) * n_cols + col
            se = sw + 1
            triangles.extend(((nw, sw, ne), (ne, sw, se)))
    return mtri.Triangulation(
        east.ravel(), north.ravel(), np.asarray(triangles, dtype=np.int32)
    )


def _terrain_interpolator(east: np.ndarray, north: np.ndarray, up: np.ndarray):
    return mtri.LinearTriInterpolator(
        _terrain_triangulation(east, north, up), up.ravel()
    )



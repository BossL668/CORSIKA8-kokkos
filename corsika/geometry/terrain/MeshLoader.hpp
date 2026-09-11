/*
 * (c) Copyright 2020 CORSIKA Project, corsika8@kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/framework/core/PhysicalUnits.hpp>
#include <corsika/framework/geometry/Point.hpp>

#include <string>
#include <vector>
#include <array>
#include <utility>

namespace corsika::terrain {

  /**
   * Result structure for mesh loading operations.
   *
   * Contains the raw vertex coordinates and face indices loaded from a file.
   */
  struct MeshData {
    /**
     * Vertex coordinates as (x, y, z) tuples.
     * Coordinates are in the units specified during loading (default: meters).
     */
    std::vector<std::array<double, 3>> vertices;

    /**
     * Face indices as (v0, v1, v2) tuples.
     * Indices are 0-based and reference vertices in the vertices array.
     */
    std::vector<std::array<size_t, 3>> faces;
  };

  /**
   * Utility class for loading mesh data from various file formats.
   *
   * Supports:
   * - OBJ (Wavefront) format: Text-based format with v (vertex) and f (face) lines
   * - PLY (Polygon File Format): Binary little-endian format with x/y/z float vertex
   *   properties and a face list; polygons are fan-triangulated
   */
  class MeshLoader {

  public:
    /**
     * Load a mesh from a Wavefront OBJ file.
     *
     * The OBJ file should contain:
     * - Vertex lines: "v x y z" where x, y, z are coordinates
     * - Face lines: "f v1 v2 v3" where v1, v2, v3 are 1-based vertex indices
     *
     * Only triangular faces are supported. Non-triangular faces will be
     * triangulated using fan triangulation.
     *
     * @param filepath Path to the OBJ file.
     * @return MeshData structure containing vertices and faces.
     * @throws std::runtime_error if the file cannot be opened or parsed.
     */
    static MeshData loadOBJ(std::string const& filepath);

    /**
     * Load a mesh from a binary little-endian PLY file.
     *
     * Reads x, y, z float vertex properties and a face list. Polygonal
     * faces are fan-triangulated into triangles.
     *
     * @param filepath Path to the PLY file.
     * @return MeshData structure containing vertices and faces.
     * @throws std::runtime_error if the file cannot be opened or parsed.
     */
    static MeshData loadPLY(std::string const& filepath);

    /**
     * Detect the file format based on extension and load accordingly.
     *
     * @param filepath Path to the mesh file.
     * @return MeshData structure containing vertices and faces.
     * @throws std::runtime_error if the format is not supported.
     */
    static MeshData load(std::string const& filepath);

    /**
     * Convert MeshData to Point vector in a given coordinate system.
     *
     * @param data The loaded mesh data.
     * @param cs The coordinate system for the points.
     * @param scale Scale factor to apply to coordinates (default: 1_m).
     * @return Vector of Point objects.
     */
    static std::vector<Point> toPoints(MeshData const& data, CoordinateSystemPtr cs,
                                       LengthType scale = 1_m);
  };

} // namespace corsika

#include <corsika/geometry/terrain/detail/MeshLoader.inl>


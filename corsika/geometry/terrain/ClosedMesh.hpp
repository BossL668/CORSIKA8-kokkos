/*
 * (c) Copyright 2020 CORSIKA Project, corsika8@kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/geometry/terrain/TriangularMesh.hpp>

namespace corsika::terrain {

  /**
   * A closed (watertight) triangular mesh for use as a volume boundary or
   * observation shell.
   *
   * ClosedMesh represents a sealed 3D surface enclosing a well-defined volume,
   * analogous to Sphere. contains() uses the inherited ray-casting majority vote,
   * which is correct for truly 3D closed surfaces: from any point strictly outside
   * the mesh, all three test rays cross the surface an even number of times; from
   * inside, an odd number.
   *
   * Use cases:
   * - Geometry-tree volume boundary (replacing Sphere for complex shapes).
   * - ObservationMesh shell: records particles entering and/or exiting a closed
   *   surface. The direction check in ObservationMesh (direction · outward_normal)
   *   distinguishes entry from exit, preventing double-recording for non-absorbing
   *   use.
   *
   * A watertightness warning is issued at construction if any edge is not shared
   * by exactly two triangles. contains() may produce incorrect results for
   * non-watertight meshes, but construction is not blocked to allow near-watertight
   * meshes from external files to be used with a caveat.
   *
   * @ingroup Geometry
   */
  class ClosedMesh : public TriangularMesh {

  public:
    /**
     * Construct from vertex and face data.
     *
     * Issues a warning if the mesh is not watertight (any edge shared by a
     * number of triangles other than two).
     */
    ClosedMesh(std::vector<Point> vertices,
               std::vector<std::array<size_t, 3>> const& faces,
               LengthType boundaryPadding = 1e-6 * 1_m);

    /**
     * Load a closed mesh from a Wavefront OBJ file.
     */
    static ClosedMesh fromOBJ(std::string const& filepath, CoordinateSystemPtr cs,
                              LengthType scale = 1_m,
                              LengthType boundaryPadding = 1e-6 * 1_m);

    /**
     * Load a closed mesh from a binary little-endian PLY file.
     */
    static ClosedMesh fromPLY(std::string const& filepath, CoordinateSystemPtr cs,
                              LengthType scale = 1_m,
                              LengthType boundaryPadding = 1e-6 * 1_m);

    /**
     * Load a closed mesh from a file, auto-detecting the format.
     */
    static ClosedMesh fromFile(std::string const& filepath, CoordinateSystemPtr cs,
                               LengthType scale = 1_m,
                               LengthType boundaryPadding = 1e-6 * 1_m);

  private:
    /**
     * Check that every edge is shared by exactly two triangles.
     * Logs a warning for each non-manifold edge and a summary if the mesh
     * is not watertight.
     */
    static void validateWatertight(std::vector<std::array<size_t, 3>> const& faces);
  };

} // namespace corsika

#include <corsika/geometry/terrain/detail/ClosedMesh.inl>


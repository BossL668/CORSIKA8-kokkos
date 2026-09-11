/*
 * (c) Copyright 2020 CORSIKA Project, corsika8@kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/geometry/terrain/MeshLoader.hpp>
#include <corsika/framework/core/Logging.hpp>

#include <map>
#include <utility>

namespace corsika::terrain {

  inline ClosedMesh::ClosedMesh(std::vector<Point> vertices,
                                std::vector<std::array<size_t, 3>> const& faces,
                                LengthType boundaryPadding)
      : TriangularMesh(std::move(vertices), faces, boundaryPadding) {
    validateWatertight(faces);
  }

  inline ClosedMesh ClosedMesh::fromOBJ(std::string const& filepath,
                                        CoordinateSystemPtr cs, LengthType scale,
                                        LengthType boundaryPadding) {
    MeshData data = MeshLoader::loadOBJ(filepath);
    std::vector<Point> points = MeshLoader::toPoints(data, cs, scale);
    return ClosedMesh(std::move(points), data.faces, boundaryPadding);
  }

  inline ClosedMesh ClosedMesh::fromPLY(std::string const& filepath,
                                        CoordinateSystemPtr cs, LengthType scale,
                                        LengthType boundaryPadding) {
    MeshData data = MeshLoader::loadPLY(filepath);
    std::vector<Point> points = MeshLoader::toPoints(data, cs, scale);
    return ClosedMesh(std::move(points), data.faces, boundaryPadding);
  }

  inline ClosedMesh ClosedMesh::fromFile(std::string const& filepath,
                                         CoordinateSystemPtr cs, LengthType scale,
                                         LengthType boundaryPadding) {
    MeshData data = MeshLoader::load(filepath);
    std::vector<Point> points = MeshLoader::toPoints(data, cs, scale);
    return ClosedMesh(std::move(points), data.faces, boundaryPadding);
  }

  inline void ClosedMesh::validateWatertight(
      std::vector<std::array<size_t, 3>> const& faces) {
    // Count how many triangles share each undirected edge.
    // A watertight mesh has every edge shared by exactly two triangles.
    std::map<std::pair<size_t, size_t>, int> edgeCount;
    for (auto const& f : faces) {
      for (int i = 0; i < 3; ++i) {
        size_t const a = f[i];
        size_t const b = f[(i + 1) % 3];
        edgeCount[{std::min(a, b), std::max(a, b)}]++;
      }
    }

    int nonManifoldEdges = 0;
    for (auto const& [edge, count] : edgeCount) {
      if (count != 2) {
        ++nonManifoldEdges;
        CORSIKA_LOG_WARN("ClosedMesh: edge ({},{}) shared by {} triangle(s), expected 2",
                         edge.first, edge.second, count);
      }
    }

    if (nonManifoldEdges > 0) {
      CORSIKA_LOG_WARN(
          "ClosedMesh: mesh is not watertight ({} non-manifold edge(s)); "
          "contains() may return incorrect results",
          nonManifoldEdges);
    } else {
      CORSIKA_LOG_DEBUG("ClosedMesh: watertight validation passed ({} edges)",
                        edgeCount.size());
    }
  }

} // namespace corsika


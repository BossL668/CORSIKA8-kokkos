/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 * Distributed under the 3-clause BSD license; see LICENSE.
 *
 * The outward-winding, watertight-edge and half-space construction is ported
 * from corsika8-mountain/src/cma21_askaryan/geometry.py (MeshGeometry). This
 * standalone volume deliberately does not import the experimental mesh branch
 * or replace IVolume, TrackingStraight, or any atmospheric geometry.
 */

#pragma once

#include <corsika/framework/geometry/ConvexPolyhedronData.hpp>
#include <corsika/framework/geometry/IVolume.hpp>
#include <corsika/framework/geometry/PhysicalGeometry.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace corsika {

  /**
   * A finite, convex, closed triangular mesh with an explicit coordinate frame.
   *
   * Vertices use metres in that frame; query Points/Directions are transformed
   * into it. A validated copy of the outward planes can be uploaded to any
   * accelerator. Concave or open terrain is rejected rather than silently
   * replaced by its convex hull. This volume does not define transport/radio
   * physics outside its boundary.
   */
  class ConvexPolyhedron final : public IVolume {
  public:
    using VertexMeters = std::array<double, 3>;
    using TriangleIndices = std::array<std::size_t, 3>;
    using PlaneData = geometry_detail::ConvexPlane;
    using RayInterval = geometry_detail::ConvexRayInterval;

    ConvexPolyhedron(CoordinateSystemPtr cs, std::vector<VertexMeters> vertices_m,
                     std::vector<TriangleIndices> faces,
                     LengthType tolerance = 1e-8 * 1_m)
        : cs_(std::move(cs))
        , vertices_(std::move(vertices_m))
        , faces_(std::move(faces))
        , tolerance_m_(tolerance / 1_m) {
      validateAndBuild();
    }

    static ConvexPolyhedron fromVerticesMeters(
        CoordinateSystemPtr cs, std::vector<VertexMeters> vertices_m,
        std::vector<TriangleIndices> faces, LengthType tolerance = 1e-8 * 1_m) {
      return ConvexPolyhedron(std::move(cs), std::move(vertices_m), std::move(faces),
                              tolerance);
    }

    static ConvexPolyhedron tetrahedron(std::array<Point, 4> const& points,
                                        LengthType tolerance = 1e-8 * 1_m) {
      auto const cs = points[0].getCoordinateSystem();
      std::vector<VertexMeters> vertices;
      for (auto const& point : points) {
        auto const p = point.getCoordinates(cs);
        vertices.push_back({p[0] / 1_m, p[1] / 1_m, p[2] / 1_m});
      }
      return ConvexPolyhedron(cs, std::move(vertices),
                              {{0, 1, 2}, {0, 3, 1}, {0, 2, 3}, {1, 3, 2}},
                              tolerance);
    }

    /// Box half lengths and center are metres in cs, not full side lengths.
    static ConvexPolyhedron box(CoordinateSystemPtr cs, VertexMeters center_m,
                                VertexMeters half_lengths_m,
                                LengthType tolerance = 1e-8 * 1_m) {
      for (std::size_t axis = 0; axis < 3; ++axis) {
        if (!std::isfinite(center_m[axis]) ||
            !std::isfinite(half_lengths_m[axis]) || !(half_lengths_m[axis] > 0.)) {
          throw std::invalid_argument("ConvexPolyhedron box: invalid center/half length");
        }
      }
      std::vector<VertexMeters> vertices;
      for (int z : {-1, 1}) {
        for (int y : {-1, 1}) {
          for (int x : {-1, 1}) {
            vertices.push_back({center_m[0] + x * half_lengths_m[0],
                                center_m[1] + y * half_lengths_m[1],
                                center_m[2] + z * half_lengths_m[2]});
          }
        }
      }
      return ConvexPolyhedron(
          std::move(cs), std::move(vertices),
          {{0, 1, 3}, {0, 3, 2}, {4, 6, 7}, {4, 7, 5}, {0, 4, 5}, {0, 5, 1},
           {2, 3, 7}, {2, 7, 6}, {0, 2, 6}, {0, 6, 4}, {1, 5, 7}, {1, 7, 3}},
          tolerance);
    }

    /// Strict triangular Wavefront OBJ reader; index/slash and negative indices
    /// are accepted. No hidden triangulation, hull conversion or mesh repair.
    static ConvexPolyhedron fromOBJ(std::filesystem::path const& path,
                                    CoordinateSystemPtr cs, LengthType scale = 1_m,
                                    LengthType tolerance = 1e-8 * 1_m) {
      double const scale_m = scale / 1_m;
      if (!std::isfinite(scale_m) || !(scale_m > 0.)) {
        throw std::invalid_argument("ConvexPolyhedron OBJ: scale must be positive");
      }
      std::ifstream input(path);
      if (!input) {
        throw std::runtime_error("ConvexPolyhedron: cannot open OBJ " + path.string());
      }
      std::vector<VertexMeters> vertices;
      std::vector<TriangleIndices> faces;
      std::string line;
      std::size_t lineNumber = 0;
      while (std::getline(input, line)) {
        ++lineNumber;
        line = line.substr(0, line.find('#'));
        std::istringstream row(line);
        std::string tag;
        if (!(row >> tag)) continue;
        auto const invalid = [&]() {
          return std::invalid_argument("ConvexPolyhedron OBJ: invalid " + tag +
                                       " at line " + std::to_string(lineNumber));
        };
        if (tag == "v") {
          VertexMeters vertex;
          std::string extra;
          if (!(row >> vertex[0] >> vertex[1] >> vertex[2]) || (row >> extra)) {
            throw invalid();
          }
          for (auto& coordinate : vertex) coordinate *= scale_m;
          vertices.push_back(vertex);
        } else if (tag == "f") {
          TriangleIndices face;
          for (auto& index : face) {
            std::string token;
            if (!(row >> token)) throw invalid();
            token = token.substr(0, token.find('/'));
            std::size_t consumed = 0;
            long long raw = 0;
            try {
              raw = std::stoll(token, &consumed);
            } catch (std::exception const&) {
              throw invalid();
            }
            if (consumed != token.size() || raw == 0) throw invalid();
            long long const resolved = raw > 0 ? raw - 1 :
                static_cast<long long>(vertices.size()) + raw;
            if (resolved < 0 || static_cast<std::size_t>(resolved) >= vertices.size()) {
              throw invalid();
            }
            index = static_cast<std::size_t>(resolved);
          }
          std::string extra;
          if (row >> extra) throw invalid();
          faces.push_back(face);
        } else if (tag != "vn" && tag != "vt" && tag != "o" && tag != "g" &&
                   tag != "s" && tag != "usemtl" && tag != "mtllib") {
          throw invalid();
        }
      }
      if (input.bad()) throw std::runtime_error("ConvexPolyhedron: OBJ read failed");
      return ConvexPolyhedron(std::move(cs), std::move(vertices), std::move(faces),
                              tolerance);
    }

    bool contains(Point const& point) const override {
      auto const p = point.getCoordinates(cs_);
      for (std::size_t axis = 0; axis < 3; ++axis) {
        if (!std::isfinite(p[axis] / 1_m)) {
          throw std::invalid_argument("ConvexPolyhedron: non-finite query point");
        }
      }
      return geometry_detail::containsConvex(planes_.data(), planes_.size(),
                                              p[0] / 1_m, p[1] / 1_m, p[2] / 1_m,
                                              tolerance_m_);
    }

    /// Query a normalized forward ray. Non-unit input is normalized here once;
    /// entry_m and exit_m are still metres, not arbitrary ray parameters.
    RayInterval intersectRay(Point const& origin,
                              DirectionVector const& direction) const {
      auto const p = origin.getCoordinates(cs_);
      auto const d = direction.getComponents(cs_);
      double const dx = d[0].magnitude();
      double const dy = d[1].magnitude();
      double const dz = d[2].magnitude();
      double const norm = std::hypot(dx, dy, dz);
      if (!std::isfinite(norm) || !(norm > 0.) || !std::isfinite(p[0] / 1_m) ||
          !std::isfinite(p[1] / 1_m) || !std::isfinite(p[2] / 1_m)) {
        throw std::invalid_argument("ConvexPolyhedron: invalid ray origin/direction");
      }
      return geometry_detail::clipConvexRay(planes_.data(), planes_.size(),
                                             p[0] / 1_m, p[1] / 1_m, p[2] / 1_m,
                                             dx / norm, dy / norm, dz / norm,
                                             tolerance_m_);
    }

    CoordinateSystemPtr const& getCoordinateSystem() const noexcept { return cs_; }
    std::vector<PlaneData> const& exportPlanes() const noexcept { return planes_; }
    std::vector<VertexMeters> const& verticesMeters() const noexcept { return vertices_; }
    std::vector<TriangleIndices> const& triangles() const noexcept { return faces_; }
    LengthType boundaryTolerance() const noexcept { return tolerance_m_ * 1_m; }
    double volumeCubicMeters() const noexcept { return volume_m3_; }

  private:
    static VertexMeters subtract(VertexMeters const& a, VertexMeters const& b) {
      return {a[0] - b[0], a[1] - b[1], a[2] - b[2]};
    }
    static double dot(VertexMeters const& a, VertexMeters const& b) {
      return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    }
    static VertexMeters cross(VertexMeters const& a, VertexMeters const& b) {
      return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2],
              a[0] * b[1] - a[1] * b[0]};
    }

    void validateAndBuild() {
      auto reject = [](std::string const& reason) {
        throw std::invalid_argument("ConvexPolyhedron: " + reason);
      };
      if (!cs_ || !std::isfinite(tolerance_m_) || tolerance_m_ < 0.) {
        reject("invalid coordinate system or boundary tolerance");
      }
      if (vertices_.size() < 4 || faces_.size() < 4) reject("empty/undersized volume");
      VertexMeters center{0., 0., 0.};
      for (auto const& vertex : vertices_) {
        for (std::size_t axis = 0; axis < 3; ++axis) {
          if (!std::isfinite(vertex[axis])) reject("non-finite vertex");
          center[axis] += vertex[axis] / static_cast<double>(vertices_.size());
        }
      }
      double extent = 0.;
      for (auto const& vertex : vertices_) {
        auto const r = subtract(vertex, center);
        extent = std::max(extent, std::hypot(r[0], r[1], r[2]));
      }
      if (!std::isfinite(extent) || !(extent > 0.)) reject("zero/non-finite extent");
      double const precision = 128. * std::numeric_limits<double>::epsilon();
      double const geometryTolerance = precision * extent;
      // Classification padding must not turn a visibly non-convex mesh into a
      // convex volume. Convexity is checked using floating-point precision only.
      std::map<std::pair<std::size_t, std::size_t>, std::pair<int, int>> edges;
      std::set<TriangleIndices> uniqueFaces;
      std::set<std::size_t> usedVertices;
      planes_.reserve(faces_.size());
      for (auto& face : faces_) {
        for (auto index : face) {
          if (index >= vertices_.size()) reject("triangle vertex index out of range");
          usedVertices.insert(index);
        }
        auto sorted = face;
        std::sort(sorted.begin(), sorted.end());
        if (!uniqueFaces.insert(sorted).second) reject("duplicate triangle");
        auto const& a = vertices_[face[0]];
        auto const ab = subtract(vertices_[face[1]], a);
        auto const ac = subtract(vertices_[face[2]], a);
        auto normal = cross(ab, ac);
        double const twiceArea = std::hypot(normal[0], normal[1], normal[2]);
        double const edgeScale = std::max(dot(ab, ab), dot(ac, ac));
        if (!std::isfinite(twiceArea) || !(twiceArea > precision * edgeScale)) {
          reject("degenerate triangle");
        }
        for (auto& component : normal) component /= twiceArea;
        double inwardDistance = dot(normal, subtract(a, center));
        if (inwardDistance < 0.) {
          std::swap(face[1], face[2]);
          for (auto& component : normal) component = -component;
          inwardDistance = -inwardDistance;
        }
        if (!(inwardDistance > geometryTolerance)) reject("flat or non-convex volume");
        for (auto const& vertex : vertices_) {
          if (dot(normal, subtract(vertex, a)) > geometryTolerance) {
            reject("non-convex volume; concave terrain is not supported");
          }
        }
        double const offset = dot(normal, a);
        if (!std::isfinite(offset)) reject("non-finite plane");
        planes_.push_back({normal[0], normal[1], normal[2], offset});
        volume_m3_ += twiceArea * inwardDistance / 6.;
        for (std::size_t i = 0; i < 3; ++i) {
          auto const from = face[i];
          auto const to = face[(i + 1) % 3];
          auto const key = std::minmax(from, to);
          auto& edge = edges[{key.first, key.second}];
          ++edge.first;
          edge.second += from < to ? 1 : -1;
        }
      }
      if (usedVertices.size() != vertices_.size()) reject("unused vertex");
      for (auto const& edge : edges) {
        if (edge.second.first != 2 || edge.second.second != 0) {
          reject("surface is not a closed, consistently oriented manifold");
        }
      }
      if (vertices_.size() + faces_.size() != edges.size() + 2) {
        reject("surface does not have the topology of one convex volume");
      }
      if (!std::isfinite(volume_m3_) || !(volume_m3_ > 0.)) reject("invalid volume");
    }

    CoordinateSystemPtr cs_;
    std::vector<VertexMeters> vertices_;
    std::vector<TriangleIndices> faces_;
    std::vector<PlaneData> planes_;
    double tolerance_m_;
    double volume_m3_ = 0.;
  };

} // namespace corsika

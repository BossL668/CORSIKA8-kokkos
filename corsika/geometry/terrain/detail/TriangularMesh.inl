/*
 * (c) Copyright 2020 CORSIKA Project, corsika8@kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/framework/core/Logging.hpp>

#include <algorithm>
#include <sstream>
#include <cmath>

namespace corsika::terrain {

  inline TriangularMesh TriangularMesh::fromOBJ(std::string const& filepath,
                                                CoordinateSystemPtr cs, LengthType scale,
                                                LengthType boundaryPadding) {
    MeshData data = MeshLoader::loadOBJ(filepath);
    std::vector<Point> points = MeshLoader::toPoints(data, cs, scale);
    return TriangularMesh(std::move(points), data.faces, boundaryPadding);
  }

  inline TriangularMesh TriangularMesh::fromPLY(std::string const& filepath,
                                                CoordinateSystemPtr cs, LengthType scale,
                                                LengthType boundaryPadding) {
    MeshData data = MeshLoader::loadPLY(filepath);
    std::vector<Point> points = MeshLoader::toPoints(data, cs, scale);
    return TriangularMesh(std::move(points), data.faces, boundaryPadding);
  }

  inline TriangularMesh TriangularMesh::fromFile(std::string const& filepath,
                                                 CoordinateSystemPtr cs, LengthType scale,
                                                 LengthType boundaryPadding) {
    MeshData data = MeshLoader::load(filepath);
    std::vector<Point> points = MeshLoader::toPoints(data, cs, scale);
    return TriangularMesh(std::move(points), data.faces, boundaryPadding);
  }

  inline TriangularMesh::TriangularMesh(std::vector<Point> vertices,
                                        std::vector<std::array<size_t, 3>> const& faces,
                                        LengthType boundaryPadding)
      : cs_(vertices.empty() ? get_root_CoordinateSystem()
                             : vertices[0].getCoordinateSystem())
      , vertices_(std::move(vertices))
      , boundaryPadding_(boundaryPadding) {

    // Create triangles from face data
    triangles_.reserve(faces.size());
    for (auto const& face : faces) {
      triangles_.emplace_back(face[0], face[1], face[2], vertices_);
    }

    // Build acceleration structure
    computeBounds();
    buildBVH();

    CORSIKA_LOG_DEBUG("TriangularMesh created with {} vertices and {} triangles",
                      vertices_.size(), triangles_.size());
  }

  inline EInside TriangularMesh::inside(Point const& p) const {
    bool sawNearHit = false;
    bool const insideVote = classify(p, {}, sawNearHit);
    if (sawNearHit) return EInside::kSurface;
    return insideVote ? EInside::kInside : EInside::kOutside;
  }

  inline bool TriangularMesh::contains(Point const& p,
                                       std::vector<size_t> const& originTriangles) const {
    bool sawNearHit; // unused here: contains() keeps its historical binary answer
    return classify(p, originTriangles, sawNearHit);
  }

  inline bool TriangularMesh::classify(Point const& p,
                                       std::vector<size_t> const& originTriangles,
                                       bool& sawNearHit) const {
    sawNearHit = false;

    // AABB early-out: a point outside the mesh's axis-aligned bounding box
    // cannot be inside the mesh, so the three-ray cast is skipped entirely.
    // This is exact (the closed mesh surface is bounded by bounds_) and
    // physics-neutral -- it only removes the per-step ray-cast cost for
    // air-shower particles far from the terrain. The isValid() guard is
    // required: an empty/no-triangle mesh has an invalid bounds_, and there
    // contains() must fall through to the (empty) ray cast rather than
    // early-returning false on every query.
    if (bounds_.isValid() && !bounds_.contains(p)) return false;

    // Ray casting algorithm: count intersections along a test ray.
    // Odd count = inside, even count = outside.
    //
    // A single ray can give the wrong parity if it passes exactly through a
    // mesh vertex or edge (a degenerate case). To guard against this, we fire
    // three independent rays with irrational-ratio directions and take a
    // majority vote. Any single degenerate ray is overruled by the other two.
    static constexpr std::array<std::array<double, 3>, 3> kRayDirs = {
        {{1.0, 0.31784, 0.19273},   // primary
         {0.19273, 1.0, 0.31784},   // rotated permutation
         {0.31784, 0.19273, 1.0}}}; // second permutation

    // Identity-based self-hit suppression.
    //
    // originTriangles contains the indices of triangles seen by the tracker's
    // forward ray inside boundaryPadding.  boundaryPadding is a step-clamping
    // distance, not a numerical zero: a genuine crossing 10 cm ahead must still
    // participate in the parity vote when a deliberately large padding is used.
    // Suppress an origin-triangle hit only inside a much tighter numerical
    // self-hit tolerance; count every other positive hit.
    //
    // Passing all near-hit triangles (not just the closest one) fixes the
    // mesh-edge case where the forward ray and probe rays find different
    // adjacent triangles as the near hit and the single-index approach
    // would fail to suppress the probe ray's self-hit.
    auto const isSelfHit = [&](size_t triIdx) {
      return std::find(originTriangles.begin(), originTriangles.end(), triIdx) !=
             originTriangles.end();
    };
    LengthType const selfHitTolerance =
        std::min(boundaryPadding_, 1.e-9 * 1_m);

    int inside_count = 0;
    for (auto const& d : kRayDirs) {
      DirectionVector rayDir(cs_, {d[0], d[1], d[2]});
      auto hits = intersectRayAll(p, rayDir);
      size_t count = 0;
      for (auto const& hit : hits) {
        if (hit.distance > boundaryPadding_) {
          count++;
        } else if (hit.distance > 0_m) {
          sawNearHit = true; // p is within boundaryPadding_ of this triangle,
                              // self-hit or not -- geometrically "on the surface"
          if (originTriangles.empty() || !isSelfHit(hit.triangleIndex) ||
              hit.distance > selfHitTolerance) {
            count++;
          }
          // else: numerical t~=0 self-hit on a known origin face -> drop
        }
        // hit.distance <= 0_m: drop
      }
      if ((count % 2) == 1) { inside_count++; }
    }

    // Majority vote: at least 2 of 3 rays must agree on "inside"
    return inside_count >= 2;
  }

  inline MeshRayHit TriangularMesh::intersectRay(Point const& origin,
                                                 DirectionVector const& direction) const {
    MeshRayHit result;

    if (!bvh_) return result;

    size_t hitTriangle;
    LengthType hitDistance;

    if (bvh_->findClosestIntersection(origin, direction, triangles_, vertices_,
                                      hitTriangle, hitDistance)) {
      result.hit = true;
      result.distance = hitDistance;
      result.triangleIndex = hitTriangle;

      // Compute hit point: origin + direction * distance
      result.hitPoint = origin + direction * result.distance;

      result.normal = triangles_[hitTriangle].getNormal();
    }

    return result;
  }

  inline MeshRayHit TriangularMesh::intersectRay(Point const& origin,
                                                 VelocityVector const& velocity) const {
    return intersectRay(origin, velocity.normalized());
  }

  inline std::vector<MeshRayHit> TriangularMesh::intersectRayAll(
      Point const& origin, DirectionVector const& direction) const {
    std::vector<MeshRayHit> results;

    if (!bvh_) return results;

    auto intersections = bvh_->findAllIntersections(origin, direction, triangles_,
                                                    vertices_, boundaryPadding_);

    for (auto const& [triIdx, dist] : intersections) {
      MeshRayHit hit;
      hit.hit = true;
      hit.distance = dist;
      hit.triangleIndex = triIdx;

      // Compute hit point: origin + direction * distance
      hit.hitPoint = origin + direction * hit.distance;

      hit.normal = triangles_[triIdx].getNormal();
      results.push_back(hit);
    }

    return results;
  }

  inline LengthType TriangularMesh::getBoundaryPadding() const {
    return boundaryPadding_;
  }

  inline CoordinateSystemPtr TriangularMesh::getCoordinateSystem() const { return cs_; }

  inline AABB const& TriangularMesh::getBounds() const { return bounds_; }

  inline size_t TriangularMesh::getTriangleCount() const { return triangles_.size(); }

  inline size_t TriangularMesh::getVertexCount() const { return vertices_.size(); }

  inline Triangle const& TriangularMesh::getTriangle(size_t index) const {
    return triangles_.at(index);
  }

  inline Point const& TriangularMesh::getVertex(size_t index) const {
    return vertices_.at(index);
  }

  inline std::vector<Point> const& TriangularMesh::getVertices() const {
    return vertices_;
  }

  inline std::vector<Triangle> const& TriangularMesh::getTriangles() const {
    return triangles_;
  }

  inline std::string TriangularMesh::asString() const {
    std::ostringstream txt;
    txt << "TriangularMesh(vertices=" << vertices_.size()
        << ", triangles=" << triangles_.size() << ", bounds=" << bounds_.asString()
        << ")";
    return txt.str();
  }

  inline void TriangularMesh::buildBVH() {
    if (triangles_.empty()) {
      bvh_ = nullptr;
      return;
    }
    bvh_ = std::make_unique<BVH>(triangles_, vertices_);
  }

  inline void TriangularMesh::computeBounds() {
    bounds_ = AABB();
    for (auto const& vertex : vertices_) { bounds_.expand(vertex); }
  }

} // namespace corsika


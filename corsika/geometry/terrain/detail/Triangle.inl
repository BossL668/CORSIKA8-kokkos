/*
 * (c) Copyright 2020 CORSIKA Project, corsika8@kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <cmath>

namespace corsika::terrain {

  inline Triangle::Triangle(size_t v0, size_t v1, size_t v2,
                            std::vector<Point> const& vertices)
      : vertexIndices_({v0, v1, v2})
      , edge1_(vertices[v1] - vertices[v0])
      , edge2_(vertices[v2] - vertices[v0])
      , normal_((edge1_.cross(edge2_)).normalized()) {}

  inline std::array<size_t, 3> const& Triangle::getVertexIndices() const {
    return vertexIndices_;
  }

  inline Vector<length_d> const& Triangle::getEdge1() const { return edge1_; }

  inline Vector<length_d> const& Triangle::getEdge2() const { return edge2_; }

  inline DirectionVector const& Triangle::getNormal() const { return normal_; }

  inline bool Triangle::intersectRay(Point const& origin,
                                     DirectionVector const& direction,
                                     std::vector<Point> const& vertices, LengthType& t,
                                     double epsilon) const {
    // Möller-Trumbore algorithm
    // Ray: P(t) = origin + t * direction
    // Triangle: V0, V1, V2
    // Intersection: P(t) = (1-u-v)*V0 + u*V1 + v*V2
    //
    // Rearranged: origin - V0 = -t*direction + u*(V1-V0) + v*(V2-V0)
    //             T = -t*D + u*E1 + v*E2
    // where T = origin - V0, D = direction, E1 = edge1, E2 = edge2
    //
    // Using Cramer's rule:
    // det = -D . (E1 x E2) = D . (E2 x E1) = -(D x E2) . E1
    // Actually the standard form uses: det = D . (E1 x E2)
    // Let P = D x E2, then det = P . E1
    // u = (T . P) / det
    // Let Q = T x E1, then v = (D . Q) / det
    // t = (E2 . Q) / det

    Point const& v0 = vertices[vertexIndices_[0]];

    // P = direction x edge2
    auto const P = direction.cross(edge2_);

    // Determinant
    auto const det = edge1_.dot(P);

    // If determinant is near zero, ray is parallel to triangle
    if (std::abs(det.magnitude()) < epsilon) { return false; }

    auto const invDet = 1.0 / det;

    // T = origin - v0
    auto const T = origin - v0;

    // u = (T . P) / det
    auto const u = T.dot(P) * invDet;

    // Check u bounds (u is dimensionless)
    if (u < 0.0 || u > 1.0) { return false; }

    // Q = T x edge1
    auto const Q = T.cross(edge1_);

    // v = (direction . Q) / det
    auto const v = direction.dot(Q) * invDet;

    // Check v bounds and u + v <= 1 (v is dimensionless)
    if (v < 0.0 || (u + v) > 1.0) { return false; }

    // t = (edge2 . Q) / det
    auto const tVal = edge2_.dot(Q) * invDet;

    // Check that intersection is in front of ray origin
    if (tVal < 0_m) { return false; }

    t = tVal;
    return true;
  }

  inline AABB Triangle::computeAABB(std::vector<Point> const& vertices) const {
    Point const& v0 = vertices[vertexIndices_[0]];
    Point const& v1 = vertices[vertexIndices_[1]];
    Point const& v2 = vertices[vertexIndices_[2]];

    CoordinateSystemPtr cs = v0.getCoordinateSystem();

    LengthType const minX = std::min({v0.getX(cs), v1.getX(cs), v2.getX(cs)});
    LengthType const minY = std::min({v0.getY(cs), v1.getY(cs), v2.getY(cs)});
    LengthType const minZ = std::min({v0.getZ(cs), v1.getZ(cs), v2.getZ(cs)});

    LengthType const maxX = std::max({v0.getX(cs), v1.getX(cs), v2.getX(cs)});
    LengthType const maxY = std::max({v0.getY(cs), v1.getY(cs), v2.getY(cs)});
    LengthType const maxZ = std::max({v0.getZ(cs), v1.getZ(cs), v2.getZ(cs)});

    return AABB(Point(cs, minX, minY, minZ), Point(cs, maxX, maxY, maxZ));
  }

  inline Point Triangle::computeCentroid(std::vector<Point> const& vertices) const {
    Point const& v0 = vertices[vertexIndices_[0]];
    Point const& v1 = vertices[vertexIndices_[1]];
    Point const& v2 = vertices[vertexIndices_[2]];

    CoordinateSystemPtr cs = v0.getCoordinateSystem();

    return Point(cs, (v0.getX(cs) + v1.getX(cs) + v2.getX(cs)) / 3,
                 (v0.getY(cs) + v1.getY(cs) + v2.getY(cs)) / 3,
                 (v0.getZ(cs) + v1.getZ(cs) + v2.getZ(cs)) / 3);
  }

} // namespace corsika


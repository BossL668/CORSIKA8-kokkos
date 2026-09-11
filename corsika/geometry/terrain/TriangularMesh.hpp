/*
 * (c) Copyright 2020 CORSIKA Project, corsika8@kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/framework/core/PhysicalUnits.hpp>
#include <corsika/framework/geometry/Point.hpp>
#include <corsika/framework/geometry/Vector.hpp>
#include <corsika/framework/geometry/PhysicalGeometry.hpp>
#include <corsika/framework/geometry/RootCoordinateSystem.hpp>
#include <corsika/framework/geometry/IVolume.hpp>
#include <corsika/geometry/terrain/AABB.hpp>
#include <corsika/geometry/terrain/Triangle.hpp>
#include <corsika/geometry/terrain/BVH.hpp>
#include <corsika/geometry/terrain/MeshLoader.hpp>

#include <memory>
#include <vector>
#include <string>
#include <limits>

namespace corsika::terrain {

  enum class EInside { kOutside, kSurface, kInside };

  /**
   * Result of a ray-mesh intersection test.
   */
  struct MeshRayHit {
    bool hit;             ///< Whether an intersection occurred
    LengthType distance;  ///< Distance along ray to intersection point
    size_t triangleIndex; ///< Index of the intersected triangle
    Point hitPoint;       ///< World-space position of the intersection
    DirectionVector
        normal; ///< Surface normal at the intersection (from the intersected triangle)

    MeshRayHit()
        : hit(false)
        , distance(std::numeric_limits<double>::infinity() * 1_m)
        , triangleIndex(static_cast<size_t>(-1))
        , hitPoint(get_root_CoordinateSystem(), 0_m, 0_m, 0_m)
        , normal(get_root_CoordinateSystem(), {0, 0, 1}) {}
  };

  /**
   * A triangular mesh for CORSIKA geometry.
   *
   * TriangularMesh is the base class for OpenMesh (open surface) and ClosedMesh
   * (watertight volume). It stores vertices and triangles, builds a BVH for
   * efficient ray intersection, and implements the IVolume interface.
   *
   * The mesh can be loaded from OBJ or PLY files, or constructed directly
   * from vertex and face data.
   *
   * @ingroup Geometry
   */
  class TriangularMesh : public IVolume {

  public:
    /**
     * Load a mesh from a Wavefront OBJ file.
     *
     * @param filepath Path to the OBJ file.
     * @param cs Coordinate system for the mesh vertices.
     * @param scale Scale factor applied to vertex coordinates (default: 1 meter).
     * @param boundaryPadding Minimum intersection distance for contains() ray casting
     *        and TrackingStraight boundary crossing (default: 1 μm).
     * @return The constructed TriangularMesh object.
     */
    static TriangularMesh fromOBJ(std::string const& filepath, CoordinateSystemPtr cs,
                                  LengthType scale = 1_m,
                                  LengthType boundaryPadding = 1e-6 * 1_m);

    /**
     * Load a mesh from a binary little-endian PLY file.
     *
     * @param filepath Path to the PLY file.
     * @param cs Coordinate system for the mesh vertices.
     * @param scale Scale factor applied to vertex coordinates (default: 1 meter).
     * @param boundaryPadding Minimum intersection distance filter (default: 1 μm).
     * @return The constructed TriangularMesh object.
     */
    static TriangularMesh fromPLY(std::string const& filepath, CoordinateSystemPtr cs,
                                  LengthType scale = 1_m,
                                  LengthType boundaryPadding = 1e-6 * 1_m);

    /**
     * Load a mesh from a file, auto-detecting the format.
     *
     * @param filepath Path to the mesh file (.obj or .ply); format is auto-detected.
     * @param cs Coordinate system for the mesh vertices.
     * @param scale Scale factor applied to vertex coordinates (default: 1 meter).
     * @param boundaryPadding Minimum intersection distance for contains() ray casting
     *        and TrackingStraight boundary crossing (default: 1 μm).
     * @return The constructed TriangularMesh object.
     */
    static TriangularMesh fromFile(std::string const& filepath, CoordinateSystemPtr cs,
                                   LengthType scale = 1_m,
                                   LengthType boundaryPadding = 1e-6 * 1_m);

    /**
     * Construct a mesh directly from vertex and face data.
     *
     * @param vertices Vector of vertex positions.
     * @param faces Vector of face indices (each face is 3 vertex indices).
     * @param boundaryPadding Minimum intersection distance for contains() ray casting
     *        and TrackingStraight boundary crossing (default: 1 μm).  Mirrors the
     *        padding_ parameter of ObservationPlane.
     */
    TriangularMesh(std::vector<Point> vertices,
                   std::vector<std::array<size_t, 3>> const& faces,
                   LengthType boundaryPadding = 1e-6 * 1_m);

    /**
     * Classify a point relative to the mesh volume.
     *
     * Uses the ray casting algorithm: cast a ray from the point and count
     * intersections with the mesh surface. An odd count means inside.
     * Returns kSurface if the point is within boundaryPadding_ of any
     * triangle (self-hit or not), regardless of the parity-vote outcome;
     * otherwise kInside or kOutside per the vote. contains() ignores this
     * distinction and keeps its original parity-vote-only answer -- see
     * classify().
     *
     * @param p The point to test.
     */
    EInside inside(Point const& p) const;

    // TriangularMesh introduces its own two-arg contains() overload below,
    // which would otherwise hide the inherited IVolume::contains(Point const&)
    // wrapper from unqualified lookup (C++ name hiding across the two-arg
    // declaration in this class).
    bool contains(Point const& p) const override {
      return inside(p) != EInside::kOutside;
    }

    /**
     * Check if a point is inside the mesh volume, with identity-based self-hit
     * suppression for sub-boundaryPadding probe-ray hits.
     *
     * @param p The point to test.
     * @param originTriangles Indices of triangles the particle is currently
     *        sitting on (within boundaryPadding of).  Probe-ray hits on these
     *        triangles at sub-boundaryPadding distance are treated as self-hits
     *        and dropped; hits on any other triangle are counted as real
     *        crossings.  Pass an empty vector when the query point is not on a
     *        face (the blanket-drop rule then reverts to: count all
     *        sub-boundaryPadding hits as real, since there is no t~=0 self-hit
     *        to suppress).
     */
    virtual bool contains(Point const& p,
                          std::vector<size_t> const& originTriangles) const;

    /**
     * Find the first intersection of a ray with the mesh.
     *
     * @param origin Ray origin point.
     * @param direction Ray direction (will be normalized internally).
     * @return MeshRayHit containing intersection information.
     */
    MeshRayHit intersectRay(Point const& origin, DirectionVector const& direction) const;

    /**
     * Find the first intersection of a ray (defined by velocity) with the mesh.
     *
     * @param origin Ray origin point.
     * @param velocity Velocity vector defining ray direction.
     * @return MeshRayHit containing intersection information.
     */
    MeshRayHit intersectRay(Point const& origin, VelocityVector const& velocity) const;

    /**
     * Find all intersections of a ray with the mesh.
     *
     * @param origin Ray origin point.
     * @param direction Ray direction (will be normalized internally).
     * @return Vector of MeshRayHit objects sorted by distance.
     */
    std::vector<MeshRayHit> intersectRayAll(Point const& origin,
                                            DirectionVector const& direction) const;

    /**
     * Get the coordinate system of the mesh.
     */
    CoordinateSystemPtr getCoordinateSystem() const;

    /**
     * Get the overall bounding box of the mesh.
     */
    AABB const& getBounds() const;

    /**
     * Get the number of triangles in the mesh.
     */
    size_t getTriangleCount() const;

    /**
     * Get the number of vertices in the mesh.
     */
    size_t getVertexCount() const;

    /**
     * Get a specific triangle.
     *
     * @param index Triangle index.
     * @return Reference to the triangle.
     */
    Triangle const& getTriangle(size_t index) const;

    /**
     * Get a specific vertex.
     *
     * @param index Vertex index.
     * @return Reference to the vertex point.
     */
    Point const& getVertex(size_t index) const;

    /**
     * Get all vertices.
     */
    std::vector<Point> const& getVertices() const;

    /**
     * Get all triangles.
     */
    std::vector<Triangle> const& getTriangles() const;

    /// Read-only export of the already built hierarchy (no geometry resampling).
    BVH const& getBVH() const { return *bvh_; }

    std::string asString() const;

    /**
     * Get the boundary padding used in contains() and tracking intersection.
     */
    LengthType getBoundaryPadding() const;

  private:
    CoordinateSystemPtr cs_;
    std::vector<Point> vertices_;
    std::vector<Triangle> triangles_;
    std::unique_ptr<BVH> bvh_;
    AABB bounds_;
    LengthType boundaryPadding_;

    /**
     * Shared implementation for contains()/inside(): the 3-ray parity-vote
     * classification, plus whether any ray registered a hit within
     * boundaryPadding_ of p (sawNearHit), regardless of self-hit status.
     * contains() ignores sawNearHit (preserves its exact historical parity-
     * vote-only answer); inside() uses it to report kSurface.
     */
    bool classify(Point const& p, std::vector<size_t> const& originTriangles,
                  bool& sawNearHit) const;

    /**
     * Build the BVH acceleration structure.
     */
    void buildBVH();

    /**
     * Compute the overall bounding box.
     */
    void computeBounds();
  };

} // namespace corsika

#include <corsika/geometry/terrain/detail/TriangularMesh.inl>

/* Read-only terrain intersections. Real exports use original shared vertices;
 * edge-vector Moller--Trumbore remains only as a legacy diagnostic oracle. */
#pragma once
#include <corsika/geometry/terrain/IndexedTerrainIntersection.hpp>
#if defined(KOKKOS_INLINE_FUNCTION)
#define C8_TERRAIN_INLINE KOKKOS_INLINE_FUNCTION
#else
#define C8_TERRAIN_INLINE inline
#endif
namespace corsika::terrain::flat {
// orientation=0: closest hit; +1: outward exit; -1: inward entry.
// Logical-side boundary callers choose orientation, never infer it by nudging
// a physical particle across a surface. Input direction must be normalized.
C8_TERRAIN_INLINE Hit intersect(View const& data,Ray const& ray,int orientation=0) {
  if (data.indexed_triangles && data.vertices) {
    auto const hit = indexed::intersect(
        {data.nodes, data.indexed_triangles, data.indices, data.vertices, data.node_count},
        ray, orientation);
    return {hit.distance, hit.triangle};
  }
  Hit result;
  for(std::uint32_t i=0;i<data.node_count;) {
    auto const& node=data.nodes[i];
    if(!slab(node,ray)){i=node.skip;continue;}
    for(std::uint32_t j=0;j<node.count;++j) {
      auto id=data.indices[node.first+j];auto const& tri=data.triangles[id];
      if(orientation&&orientation*dot(tri.normal,ray.direction)<=1.e-12)continue;
      double d;
      if(triangleHit(tri,ray,d)&&d<result.distance)result={d,id};
    }
    ++i;
  }
  return result;
}
} // namespace corsika::terrain::flat
#undef C8_TERRAIN_INLINE

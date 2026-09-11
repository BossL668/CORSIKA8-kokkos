/* Portable, read-only geometry kernels. Units: m, unit direction vectors.
 * Moller--Trumbore and slab semantics follow the mountain CPU geometry.
 * No CUDA runtime, particle transport, or air-shower dependency. */
#pragma once
#include <cmath>
#include <cstdint>
#include <limits>
#include <type_traits>
#if defined(KOKKOS_INLINE_FUNCTION)
#define C8_TERRAIN_INLINE KOKKOS_INLINE_FUNCTION
#else
#define C8_TERRAIN_INLINE inline
#endif
namespace corsika::terrain::flat {
inline constexpr double NoHitDistance = std::numeric_limits<double>::infinity();
inline constexpr std::uint32_t NoTriangle = std::numeric_limits<std::uint32_t>::max();
struct Vec3 { double x,y,z; };
C8_TERRAIN_INLINE Vec3 sub(Vec3 a,Vec3 b){return {a.x-b.x,a.y-b.y,a.z-b.z};}
C8_TERRAIN_INLINE Vec3 cross(Vec3 a,Vec3 b){return {a.y*b.z-a.z*b.y,a.z*b.x-a.x*b.z,a.x*b.y-a.y*b.x};}
C8_TERRAIN_INLINE double dot(Vec3 a,Vec3 b){return a.x*b.x+a.y*b.y+a.z*b.z;}
struct Triangle { Vec3 origin,edge1,edge2,normal; };
struct IndexedTriangle { std::uint32_t vertex[3]; Vec3 normal; };
// Preorder nodes with a forward skip: no per-ray recursion or fixed-depth stack.
struct Node { Vec3 low,high; std::uint32_t skip,first,count; };
struct Ray { Vec3 origin,direction; int orientation{}; };
struct Hit {
  double distance=NoHitDistance;
  std::uint32_t triangle=NoTriangle;
  C8_TERRAIN_INLINE bool found()const{return triangle!=NoTriangle;}
};
struct View {
  Node const* nodes{}; Triangle const* triangles{};
  std::uint32_t const* indices{};
  std::uint32_t node_count{};
  // Optional only for legacy diagnostic fixtures. Real terrain uploads require
  // original vertices and topology, never origin + edge reconstruction.
  IndexedTriangle const* indexed_triangles{};
  Vec3 const* vertices{};
};
static_assert(std::is_trivially_copyable<View>::value,"device geometry must be POD");
C8_TERRAIN_INLINE double inverseDirection(double d) {
  return std::abs(d)>1.e-10?1./d:(d>=0.?1.e10:-1.e10);
}
C8_TERRAIN_INLINE bool slab(Node const& n,Ray const& r) {
  double lo=-NoHitDistance,hi=-lo;
  double const low[3]={n.low.x,n.low.y,n.low.z},high[3]={n.high.x,n.high.y,n.high.z};
  double const pos[3]={r.origin.x,r.origin.y,r.origin.z},dir[3]={r.direction.x,r.direction.y,r.direction.z};
  for(int i=0;i<3;++i) {
    double const inv=inverseDirection(dir[i]);
    double a=(low[i]-pos[i])*inv,b=(high[i]-pos[i])*inv;
    if(a>b){double tmp=a;a=b;b=tmp;}
    lo=lo>a?lo:a;hi=hi<b?hi:b;
  }
  return hi>=lo&&hi>=0.;
}
C8_TERRAIN_INLINE bool triangleHit(Triangle const& t,Ray const& r,double& distance) {
  auto p=cross(r.direction,t.edge2);auto det=dot(t.edge1,p);
  if(std::abs(det)<1.e-9)return false;
  auto inv=1./det;auto tv=sub(r.origin,t.origin);auto u=dot(tv,p)*inv;
  if(u<0.||u>1.)return false;
  auto q=cross(tv,t.edge1);auto v=dot(r.direction,q)*inv;
  if(v<0.||u+v>1.)return false;
  auto time=dot(t.edge2,q)*inv;if(time<0.)return false;
  distance=time;return true;
}
} // namespace corsika::terrain::flat
#undef C8_TERRAIN_INLINE

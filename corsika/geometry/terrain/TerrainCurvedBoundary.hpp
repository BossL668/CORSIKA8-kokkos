/* Intersection with the SAME quadratic leapfrog path used by CORSIKA air
 * tracking: r(L)=r0+L*d+L^2*q, L=|v|*dt (not chord length or helix arc).
 * The BVH encloses the whole curve, including coordinate extrema. Both plane
 * roots are checked against the finite triangle and the logical material side.
 * No trajectory subdivision, particle displacement, or boundary absorption. */
#pragma once
#include <corsika/geometry/terrain/TerrainBoundary.hpp>
#include <corsika/geometry/terrain/CurvedIntersectionNumerics.hpp>
#if defined(KOKKOS_INLINE_FUNCTION)
#define C8_TERRAIN_CURVE_INLINE KOKKOS_INLINE_FUNCTION
#else
#define C8_TERRAIN_CURVE_INLINE inline
#endif
namespace corsika::terrain::flat {
struct QuadraticPath {
  BoundaryQuery start;
  Vec3 quadratic; // inverse metres
  double maximum_length_m;
};
C8_TERRAIN_CURVE_INLINE Vec3 curvePosition(QuadraticPath const& p,double l) {
  return {p.start.origin.x+l*p.start.direction.x+l*l*p.quadratic.x,
          p.start.origin.y+l*p.start.direction.y+l*l*p.quadratic.y,
          p.start.origin.z+l*p.start.direction.z+l*l*p.quadratic.z};
}
C8_TERRAIN_CURVE_INLINE bool curveBox(Node const& n,QuadraticPath const& p,double end) {
  double const pos[3]={p.start.origin.x,p.start.origin.y,p.start.origin.z};
  double const dir[3]={p.start.direction.x,p.start.direction.y,p.start.direction.z};
  double const q[3]={p.quadratic.x,p.quadratic.y,p.quadratic.z};
  double const low[3]={n.low.x,n.low.y,n.low.z},high[3]={n.high.x,n.high.y,n.high.z};
  constexpr double begin=-1.e-8;
  for(int i=0;i<3;++i) {
    auto a=pos[i]+begin*dir[i]+begin*begin*q[i];
    auto b=pos[i]+end*dir[i]+end*end*q[i];
    double lo=a<b?a:b,hi=a>b?a:b;
    if(q[i]!=0.) {
      double stationary=-dir[i]/(2.*q[i]);
      if(stationary>begin&&stationary<end) {
        double v=pos[i]+stationary*dir[i]+stationary*stationary*q[i];
        lo=v<lo?v:lo;hi=v>hi?v:hi;
      }
    }
    if(hi<low[i]-1.e-8||lo>high[i]+1.e-8)return false;
  }
  return true;
}
C8_TERRAIN_CURVE_INLINE BoundaryCandidate nextCurvedBoundary(
    View const& view,QuadraticPath const& p) {
  if(!(p.maximum_length_m>=1.e-9))return {};
  if(p.quadratic.x==0.&&p.quadratic.y==0.&&p.quadratic.z==0.) {
    auto h=nextBoundary(view,p.start);
    return h.hit.distance<=p.maximum_length_m?h:BoundaryCandidate{};
  }
  Hit closest;
  for(std::uint32_t i=0;i<view.node_count;) {
    auto const& node=view.nodes[i];
    double end=closest.distance<p.maximum_length_m?closest.distance:p.maximum_length_m;
    if(!curveBox(node,p,end)){i=node.skip;continue;}
    for(std::uint32_t j=0;j<node.count;++j) {
      auto id=view.indices[node.first+j];Vec3 v[3],normal;
      curved::vertices(view,id,v,normal);
      auto plane=curved::plane(v,normal,p.start.origin,p.start.direction,p.quadratic);
      if(plane.orientation_scale==0.||!std::isfinite(plane.orientation_scale))continue;
      double roots[2];int count=curved::roots(plane.a,plane.b,plane.c,roots);
      for(int k=0;k<count;++k) {
        double l=roots[k];
        if(!std::isfinite(l)||l<-1.e-8||l>p.maximum_length_m)continue;
        double orientation=curved::value(curved::add(plane.b,curved::multiply(plane.a,{2.*l,0.})))
            /plane.orientation_scale;
        if(p.start.logically_inside?orientation<=1.e-12:orientation>=-1.e-12)continue;
        if(!curved::contains(v,normal,p.start.origin,p.start.direction,p.quadratic,l))continue;
        double flight=l>1.e-9?l:1.e-9;
        if(flight>p.maximum_length_m)continue;
        // A curve can hit the same feature twice: compare flight first, then
        // break EXACT ties by original face ID, not BVH traversal order.
        if(flight<closest.distance||(flight==closest.distance&&id<closest.triangle))
          closest={flight,id};
      }
    }
    ++i;
  }
  return {closest,closest.found()?(p.start.logically_inside?Crossing::ExitRock:Crossing::EnterRock):Crossing::None};
}
} // namespace corsika::terrain::flat
#undef C8_TERRAIN_CURVE_INLINE

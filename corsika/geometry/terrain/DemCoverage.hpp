// Horizontal DEM coverage, extruded vertically. This is a computational
// boundary, never a material interface. CPU and device use the same roots/BVH.
#pragma once
#include <corsika/geometry/terrain/TerrainCurvedBoundary.hpp>
#if defined(KOKKOS_INLINE_FUNCTION)
#define C8_COVERAGE_INLINE KOKKOS_INLINE_FUNCTION
#else
#define C8_COVERAGE_INLINE inline
#endif
namespace corsika::terrain::coverage {
using flat::Vec3;
struct Edge { Vec3 a,b; }; // counterclockwise perimeter: interior on the left
struct View {
  Edge const* edges{};
  flat::Node const* nodes{};
  std::uint32_t const* indices{};
  std::uint32_t node_count{};
  C8_COVERAGE_INLINE bool enabled() const { return node_count!=0; }
};
struct Hit {
  double distance{HUGE_VAL};
  std::uint32_t edge{UINT32_MAX};
  C8_COVERAGE_INLINE bool found() const { return edge!=UINT32_MAX; }
};
// Both zero-curvature infinite rays and finite quadratic paths are supported.
C8_COVERAGE_INLINE bool overlaps(flat::Node const& n,flat::QuadraticPath const& p,double end) {
  double o[2]{p.start.origin.x,p.start.origin.y},d[2]{p.start.direction.x,p.start.direction.y};
  double q[2]{p.quadratic.x,p.quadratic.y},lo[2]{n.low.x,n.low.y},hi[2]{n.high.x,n.high.y};
  if(q[0]==0.&&q[1]==0.) {
    double begin=-1.e-8;
    for(int i=0;i<2;++i) {
      if(d[i]==0.) {if(o[i]<lo[i]-1.e-8||o[i]>hi[i]+1.e-8)return false;continue;}
      double a=(lo[i]-1.e-8-o[i])/d[i],b=(hi[i]+1.e-8-o[i])/d[i];
      if(a>b){double t=a;a=b;b=t;}
      begin=begin>a?begin:a;end=end<b?end:b;
      if(begin>end)return false;
    }
    return true;
  }
  for(int i=0;i<2;++i) {
    double a=o[i]-1.e-8*d[i]+1.e-16*q[i],b=o[i]+end*d[i]+end*end*q[i];
    double low=a<b?a:b,high=a>b?a:b;
    if(q[i]!=0.) {
      double s=-d[i]/(2.*q[i]);
      if(s>0.&&s<end){double x=o[i]+s*d[i]+s*s*q[i];low=low<x?low:x;high=high>x?high:x;}
    }
    if(high<lo[i]-1.e-8||low>hi[i]+1.e-8)return false;
  }
  return true;
}
C8_COVERAGE_INLINE Hit nextExit(View const& v,flat::QuadraticPath const& p) {
  Hit hit;
  for(std::uint32_t i=0;i<v.node_count;) {
    auto const& n=v.nodes[i];double end=hit.distance<p.maximum_length_m?hit.distance:p.maximum_length_m;
    if(!overlaps(n,p,end)){i=n.skip;continue;}
    for(std::uint32_t j=0;j<n.count;++j) {
      auto id=v.indices[n.first+j];auto const& e=v.edges[id];
      double nx=e.b.y-e.a.y,ny=e.a.x-e.b.x,len=std::sqrt(nx*nx+ny*ny);
      using namespace corsika::terrain::curved;
      auto dot2=[&](Vec3 x){return add(multiply({nx,0.},{x.x,0.}),multiply({ny,0.},{x.y,0.}));};
      auto a=dot2(p.quadratic),b=dot2(p.start.direction);
      auto c=add(multiply({nx,0.},subtract({p.start.origin.x,0.},{e.a.x,0.})),
                 multiply({ny,0.},subtract({p.start.origin.y,0.},{e.a.y,0.})));
      double r[2];int count=roots(a,b,c,r);
      for(int k=0;k<count;++k) {
        double s=r[k];
        if(!std::isfinite(s)||s<-1.e-8||s>p.maximum_length_m||
            value(add(b,multiply(a,{2.*s,0.})))<=1.e-12*len)continue;
        auto x=flat::curvePosition(p,s);
        double along=(x.x-e.a.x)*(e.b.x-e.a.x)+(x.y-e.a.y)*(e.b.y-e.a.y);
        if(along<-1.e-8*len||along>len*len+1.e-8*len)continue;
        double flight=s>1.e-9?s:1.e-9;
        if(flight<=p.maximum_length_m&&(flight<hit.distance||(flight==hit.distance&&id<hit.edge)))hit={flight,id};
      }
    }
    ++i;
  }
  return hit;
}
// Host admission and propagation leg checks. Boundary points are inside.
C8_COVERAGE_INLINE bool contains(View const& v,Vec3 p) {
  if(!v.enabled())return true;
  bool inside=false;
  for(std::uint32_t i=0;i<v.node_count;) {
    auto const& n=v.nodes[i];
    if(p.y<n.low.y-1.e-8||p.y>n.high.y+1.e-8||p.x>n.high.x+1.e-8){i=n.skip;continue;}
    for(std::uint32_t j=0;j<n.count;++j) {
      auto const& e=v.edges[v.indices[n.first+j]];
      double dx=e.b.x-e.a.x,dy=e.b.y-e.a.y,len=std::sqrt(dx*dx+dy*dy);
      double cross=dx*(p.y-e.a.y)-dy*(p.x-e.a.x),along=dx*(p.x-e.a.x)+dy*(p.y-e.a.y);
      if(std::abs(cross)<=1.e-8*len&&along>=-1.e-8*len&&along<=len*len+1.e-8*len)return true;
      if((e.a.y>p.y)!=(e.b.y>p.y)) {
        double x=e.a.x+(p.y-e.a.y)*dx/dy;if(p.x<x)inside=!inside;
      }
    }
    ++i;
  }
  return inside;
}
C8_COVERAGE_INLINE bool segmentInside(View const& v,Vec3 a,Vec3 b) {
  if(!v.enabled())return true;
  if(!contains(v,a)||!contains(v,b))return false;
  auto d=flat::sub(b,a);double length=std::sqrt(flat::dot(d,d));
  if(length<=1.e-9)return true;
  auto h=nextExit(v,{{a,{d.x/length,d.y/length,d.z/length},true},{0.,0.,0.},length});
  return !h.found()||h.distance>=length-1.e-8;
}
} // namespace corsika::terrain::coverage
#undef C8_COVERAGE_INLINE

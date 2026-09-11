/* Numerical predicates for the existing quadratic leapfrog trajectory.
 * Twofold arithmetic protects cancellation, not a geometric padding. These
 * functions neither change the trajectory nor sample any random numbers.
 * This is compensated binary64 arithmetic, not an exact algebraic predicate.
 */
#pragma once
#include <corsika/geometry/terrain/IndexedTerrainIntersection.hpp>
#if defined(KOKKOS_INLINE_FUNCTION)
#define C8_CURVED_NUMERIC_INLINE KOKKOS_INLINE_FUNCTION
#else
#define C8_CURVED_NUMERIC_INLINE inline
#endif
namespace corsika::terrain::curved {
using indexed::Twofold;
using indexed::add;
using indexed::subtract;
using indexed::multiply;
using indexed::sum;
using indexed::component;
using flat::Vec3;

C8_CURVED_NUMERIC_INLINE double value(Twofold x) { return x.high + x.low; }
C8_CURVED_NUMERIC_INLINE Twofold scaled(Twofold x,int exponent) {
  return {std::ldexp(x.high,exponent),std::ldexp(x.low,exponent)};
}
// Stable quadratic formula with a compensated discriminant. In particular,
// b*b-4*a*c must not erase two real, near-tangent roots. Power-of-two scaling
// avoids overflow without perturbing normal-range coefficient significands.
C8_CURVED_NUMERIC_INLINE int roots(Twofold a,Twofold b,Twofold c,double* out) {
  double scale=std::fmax(std::abs(value(a)),std::fmax(std::abs(value(b)),std::abs(value(c))));
  if(!(scale>0.)||!std::isfinite(scale))return 0;
  int exponent=0;std::frexp(scale,&exponent);
  a=scaled(a,-exponent);b=scaled(b,-exponent);c=scaled(c,-exponent);
  double aa=value(a),bb=value(b),cc=value(c);int count=0;
  if(aa==0.) {
    if(bb==0.)return 0;
    out[count++]=-cc/bb;
  } else {
    auto discriminant=subtract(multiply(b,b),multiply({4.,0.},multiply(a,c)));
    double d=value(discriminant);
    if(!(d>=0.))return 0;
    double q=-.5*(bb+std::copysign(std::sqrt(d),bb));
    if(q==0.)out[count++]=-bb/(2.*aa);
    else {out[count++]=q/aa;out[count++]=cc/q;}
  }
  // Restore coefficient residuals lost by the final binary64 division. A
  // repeated root has zero derivative and is not Newton-refined.
  for(int i=0;i<count;++i)for(int step=0;step<2;++step) {
    double x=out[i];if(!std::isfinite(x))break;
    auto derivative=add(multiply(a,{2.*x,0.}),b);
    double slope=value(derivative);if(slope==0.||!std::isfinite(slope))break;
    auto residual=add(multiply(add(multiply(a,{x,0.}),b),{x,0.}),c);
    double next=x-value(residual)/slope;
    if(!std::isfinite(next)||next==x)break;
    out[i]=next;
  }
  if(count==2&&out[1]<out[0]){double x=out[0];out[0]=out[1];out[1]=x;}
  return count;
}

C8_CURVED_NUMERIC_INLINE void vertices(flat::View const& view,std::uint32_t id,
                                       Vec3* v,Vec3& normal) {
  if(view.vertices&&view.indexed_triangles) {
    auto const& t=view.indexed_triangles[id];
    std::uint32_t ids[3]={t.vertex[0],t.vertex[1],t.vertex[2]};
    for(int i=0;i<2;++i)for(int j=i+1;j<3;++j)
      if(ids[j]<ids[i]){auto k=ids[i];ids[i]=ids[j];ids[j]=k;}
    for(int i=0;i<3;++i)v[i]=view.vertices[ids[i]];
    normal=t.normal;
  } else {
    // Legacy diagnostic views only; production uploads require indexed data.
    auto const& t=view.triangles[id];v[0]=t.origin;
    v[1]={t.origin.x+t.edge1.x,t.origin.y+t.edge1.y,t.origin.z+t.edge1.z};
    v[2]={t.origin.x+t.edge2.x,t.origin.y+t.edge2.y,t.origin.z+t.edge2.z};
    normal=t.normal;
  }
}

struct PlanePolynomial { Twofold a,b,c;double orientation_scale; };
// A stored, rounded unit normal does not define a plane passing through all
// three original vertices exactly. Near a shared vertex that discrepancy can
// dominate the hit distance. Build the plane from compensated original edges;
// use the stored normal ONLY to normalize/orient the crossing derivative.
C8_CURVED_NUMERIC_INLINE PlanePolynomial plane(Vec3 const* v,Vec3 normal,
                                              Vec3 origin,Vec3 direction,Vec3 quadratic) {
  Twofold e1[3],e2[3],n[3];
  for(int k=0;k<3;++k) {
    e1[k]=sum(component(v[1],k),-component(v[0],k));
    e2[k]=sum(component(v[2],k),-component(v[0],k));
  }
  for(int k=0;k<3;++k) {
    int j=(k+1)%3,h=(k+2)%3;
    n[k]=subtract(multiply(e1[j],e2[h]),multiply(e1[h],e2[j]));
  }
  PlanePolynomial p{{0.,0.},{0.,0.},{0.,0.},0.};Twofold orient{0.,0.};
  for(int k=0;k<3;++k) {
    p.a=add(p.a,multiply(n[k],{component(quadratic,k),0.}));
    p.b=add(p.b,multiply(n[k],{component(direction,k),0.}));
    p.c=add(p.c,multiply(n[k],sum(component(origin,k),-component(v[0],k))));
    orient=add(orient,multiply(n[k],{component(normal,k),0.}));
  }
  p.orientation_scale=value(orient);return p;
}

// Signed projected areas using ORIGINAL shared vertices. Evaluate v-r(L)
// without first rounding the global hit point. Exact zeros include the shared
// edge; a point one representable input step outside is not snapped onto it.
C8_CURVED_NUMERIC_INLINE bool contains(Vec3 const* v,Vec3 normal,
                                       Vec3 origin,Vec3 direction,Vec3 quadratic,double l) {
  int kz=0;
  for(int k=1;k<3;++k)
    if(std::abs(component(normal,k))>std::abs(component(normal,kz)))kz=k;
  int kx=(kz+1)%3,ky=(kx+1)%3;
  auto ll=multiply({l,0.},{l,0.});
  auto dx=add(multiply({component(direction,kx),0.},{l,0.}),
              multiply({component(quadratic,kx),0.},ll));
  auto dy=add(multiply({component(direction,ky),0.},{l,0.}),
              multiply({component(quadratic,ky),0.},ll));
  Twofold x[3],y[3];
  for(int i=0;i<3;++i) {
    x[i]=subtract(sum(component(v[i],kx),-component(origin,kx)),dx);
    y[i]=subtract(sum(component(v[i],ky),-component(origin,ky)),dy);
  }
  bool positive=false,negative=false;
  for(int i=0;i<3;++i) {
    int j=(i+1)%3;
    double e=indexed::compensatedEdge(x[i],y[i],x[j],y[j]);
    if(!std::isfinite(e))return false;
    positive=positive||e>0.;negative=negative||e<0.;
  }
  return (positive||negative)&&!(positive&&negative);
}
} // namespace corsika::terrain::curved
#undef C8_CURVED_NUMERIC_INLINE

/* Device port of mountain's planar Snell/Fresnel/ray-tube and terrain visibility.
 * Source reference: corsika8-mountain/cpp/{MountainRadio,TerrainRadio}.hpp.
 * Single transmitted interface; straight homogeneous or radial-index legs.
 * No reflection, diffraction or repeated-interface ray branch is implied. */
#pragma once
#include <corsika/modules/radio/interface/Types.hpp>
#include <corsika/accelerator/ScalarPhysicalConstants.hpp>
#include <cmath>
#if defined(KOKKOS_INLINE_FUNCTION)
#define C8_INTERFACE_RADIO_INLINE KOKKOS_INLINE_FUNCTION
#else
#define C8_INTERFACE_RADIO_INLINE inline
#endif
namespace corsika::radio::interface::detail {
namespace flat = terrain::flat;
namespace indexed = terrain::indexed;
constexpr double light_speed = accelerator::scalar_constants::SpeedOfLightMPerS;
C8_INTERFACE_RADIO_INLINE Vec3 add(Vec3 a,Vec3 b){return {a.x+b.x,a.y+b.y,a.z+b.z};}
C8_INTERFACE_RADIO_INLINE Vec3 scale(Vec3 a,double s){return {a.x*s,a.y*s,a.z*s};}
using flat::sub; using flat::dot; using flat::cross;
C8_INTERFACE_RADIO_INLINE double norm(Vec3 a){return std::sqrt(dot(a,a));}
C8_INTERFACE_RADIO_INLINE Vec3 unit(Vec3 a){return scale(a,1./norm(a));}
C8_INTERFACE_RADIO_INLINE double max(double a,double b){return a>b?a:b;}
C8_INTERFACE_RADIO_INLINE double min(double a,double b){return a<b?a:b;}
C8_INTERFACE_RADIO_INLINE double clamp(double x,double lo,double hi){return max(lo,min(x,hi));}
C8_INTERFACE_RADIO_INLINE bool finite(Vec3 a){return std::isfinite(a.x)&&std::isfinite(a.y)&&std::isfinite(a.z);}
struct MediumView {
  double index{1.}, attenuation_length_m{HUGE_VAL};
  Vec3 center{}; double radius{};
  IndexSample const* samples{}; std::uint32_t count{};
  double const* breaks{};std::uint32_t break_count{};
  double minimum_index{1.},maximum_index{1.};
};
struct TransmissionNormalBounds { Vec3 low{},high{}; };
struct PropagationView {
  terrain::coverage::View coverage;
  Geometry geometry{Geometry::Uniform};
  MediumView media[2];
  Vec3 plane_point{}, outward{0.,0.,1.};
  flat::View mesh;
  std::uint32_t faces{}, integration_samples{64};
  double visibility_tolerance{1.e-7}, edge_tolerance{1.e-7};
  TransmissionNormalBounds const* transmission_normals{};
};

// Conservative interval test for a stationary transmitted ray in a BVH node.
// At a valid interface point x, n1*(x-S)/|x-S| + n2*(x-O)/|x-O|
// is parallel to the face normal (Snell's tangential continuity). Reject a
// whole node only if one component of their cross product excludes zero.
// No amplitude threshold, attenuation cutoff, or approximate ray is used.
struct OpticalInterval { double low{},high{}; };
C8_INTERFACE_RADIO_INLINE OpticalInterval intervalProduct(OpticalInterval a,OpticalInterval b){
  double aa=a.low*b.low,ab=a.low*b.high,ba=a.high*b.low,bb=a.high*b.high;
  return {min(min(aa,ab),min(ba,bb)),max(max(aa,ab),max(ba,bb))};
}
C8_INTERFACE_RADIO_INLINE OpticalInterval intervalDifference(OpticalInterval a,OpticalInterval b){
  return {a.low-b.high,a.high-b.low};
}
C8_INTERFACE_RADIO_INLINE void directionToBox(Vec3 source,Vec3 low,Vec3 high,OpticalInterval* out){
  double a[3]{low.x-source.x,low.y-source.y,low.z-source.z};
  double b[3]{high.x-source.x,high.y-source.y,high.z-source.z};
  double near2=0.,far2=0.;
  for(unsigned k=0;k<3;++k){
    double near=a[k]>0.?a[k]:(b[k]<0.?-b[k]:0.);
    double far=max(std::abs(a[k]),std::abs(b[k]));near2+=near*near;far2+=far*far;
  }
  if(!(near2>1.e-24)||!std::isfinite(far2)){
    for(unsigned k=0;k<3;++k)out[k]={-1.,1.};return;
  }
  OpticalInterval inverse{1./std::sqrt(far2),1./std::sqrt(near2)};
  for(unsigned k=0;k<3;++k){out[k]=intervalProduct({a[k],b[k]},inverse);out[k].low=max(-1.,out[k].low);out[k].high=min(1.,out[k].high);}
}
C8_INTERFACE_RADIO_INLINE bool transmissionNodeMayContain(PropagationView const& v,RayQuery q,std::uint32_t id){
  auto const& node=v.mesh.nodes[id];auto const& normal=v.transmission_normals[id];
  double magnitude=max(1.,max(norm(node.low),max(norm(node.high),max(norm(q.source),norm(q.observer)))));
  double margin=4*v.edge_tolerance+128*0x1p-52*magnitude;
  Vec3 low{node.low.x-margin,node.low.y-margin,node.low.z-margin};
  Vec3 high{node.high.x+margin,node.high.y+margin,node.high.z+margin};
  OpticalInterval fromSource[3],fromObserver[3],gradient[3];
  directionToBox(q.source,low,high,fromSource);directionToBox(q.observer,low,high,fromObserver);
  auto const& m1=v.media[q.source_region];auto const& m2=v.media[q.observer_region];
  OpticalInterval n1=m1.count?OpticalInterval{m1.minimum_index,m1.maximum_index}:OpticalInterval{m1.index,m1.index};
  OpticalInterval n2=m2.count?OpticalInterval{m2.minimum_index,m2.maximum_index}:OpticalInterval{m2.index,m2.index};
  OpticalInterval normals[3]{{normal.low.x,normal.high.x},{normal.low.y,normal.high.y},{normal.low.z,normal.high.z}};
  for(unsigned k=0;k<3;++k){
    auto a=intervalProduct(n1,fromSource[k]),b=intervalProduct(n2,fromObserver[k]);
    gradient[k]={a.low+b.low,a.high+b.high};
  }
  // Covers solver's 1e-10 Snell acceptance and interval arithmetic roundoff.
  double tolerance=1.e-9*max(1.,n1.high+n2.high);
  for(unsigned k=0;k<3;++k){
    auto a=(k+1)%3,b=(k+2)%3;
    auto crossComponent=intervalDifference(intervalProduct(gradient[a],normals[b]),intervalProduct(gradient[b],normals[a]));
    if(crossComponent.low>tolerance||crossComponent.high<-tolerance)return false;
  }
  return true;
}
// Exhaustive fallback remains available to host/reference callers that do not
// supply normal bounds. Threaded traversal has no finite candidate-list cap.
struct CandidateFaces {
  PropagationView const& v;RayQuery q;bool direct{};
  std::uint32_t node{},offset{},linear{};bool entered{};
  C8_INTERFACE_RADIO_INLINE bool next(std::uint32_t& faceId){
    if(direct||v.geometry==Geometry::Plane){faceId=0;return linear++==0;}
    if(!v.transmission_normals){faceId=linear++;return faceId<v.faces;}
    while(node<v.mesh.node_count){
      auto const& n=v.mesh.nodes[node];
      if(!entered){
        if(!transmissionNodeMayContain(v,q,node)){node=n.skip;continue;}
        entered=true;
      }
      if(offset<n.count){faceId=v.mesh.indices[n.first+offset++];return true;}
      ++node;offset=0;entered=false;
    }
    return false;
  }
};
C8_INTERFACE_RADIO_INLINE double indexAt(MediumView const& medium,Vec3 p) {
  if(!medium.count)return medium.index;
  double radial=norm(sub(p,medium.center)),h=radial-medium.radius;
  double tolerance=16*0x1p-52*max(1.,max(radial,std::abs(medium.radius)));
  if(h<medium.samples[0].height_m-tolerance || h>medium.samples[medium.count-1].height_m+tolerance)return NAN;
  h=clamp(h,medium.samples[0].height_m,medium.samples[medium.count-1].height_m);
  std::uint32_t lo=0,hi=medium.count-1;
  while(hi-lo>1){auto mid=(lo+hi)/2;if(h<medium.samples[mid].height_m)hi=mid;else lo=mid;}
  auto a=medium.samples[lo],b=medium.samples[hi];
  return a.index+(b.index-a.index)*(h-a.height_m)/(b.height_m-a.height_m);
}
C8_INTERFACE_RADIO_INLINE double opticalLength(MediumView const& m,Vec3 a,Vec3 b,unsigned samples) {
  auto delta=sub(b,a);double length=norm(delta);
  if(!m.count)return m.index*length;
  if(length==0.)return 0.;
  auto direction=scale(delta,1./length),relative=sub(a,m.center);
  double along=dot(relative,direction),perpendicular2=max(0.,dot(relative,relative)-along*along);
  double cuts[131];unsigned count=2;cuts[0]=0.;cuts[1]=length;
  if(-along>0.&&-along<length)cuts[count++]=-along;
  for(unsigned i=0;i<m.break_count;++i){
    double radius=m.radius+m.breaks[i],discriminant=radius*radius-perpendicular2;
    if(!(radius>0.)||discriminant<0.)continue;
    double root=std::sqrt(discriminant),crossings[2]{-along-root,-along+root};
    for(double x:crossings)if(x>0.&&x<length)cuts[count++]=x;
  }
  for(unsigned i=1;i<count;++i){double x=cuts[i];unsigned j=i;while(j&&cuts[j-1]>x){cuts[j]=cuts[j-1];--j;}cuts[j]=x;}
  // Composite eight-point Gauss integration within each radial layer piece.
  // A fixed midpoint grid can miss an entire narrow layer transition and
  // shift long-path pulses by nanoseconds even with a well-resolved n table.
  double nodes[4]{.18343464249564980494,.52553240991632898582,.79666647741362673959,.96028985649753623168};
  double weights[4]{.36268378337836198297,.31370664587788728734,.22238103445337447054,.10122853629037625915};
  unsigned panels=(samples+7)/8;double integral=0.;
  for(unsigned piece=1;piece<count;++piece){
    double width=(cuts[piece]-cuts[piece-1])/panels;
    if(!(width>0.))continue;
    for(unsigned panel=0;panel<panels;++panel){
      double center=cuts[piece-1]+(panel+.5)*width,sum=0.;
      for(unsigned j=0;j<4;++j){double offset=.5*width*nodes[j];
        sum+=weights[j]*(indexAt(m,add(a,scale(direction,center-offset)))+
                        indexAt(m,add(a,scale(direction,center+offset))));}
      integral+=.5*width*sum;
    }
  }
  return integral;
}
// Ignore only endpoint intersections. Traverse the original indexed BVH so an
// initial surface hit cannot conceal a later mountain blocking the same ray.
C8_INTERFACE_RADIO_INLINE bool segmentClear(PropagationView const& v,Vec3 a,Vec3 b) {
  double length=norm(sub(b,a));
  if(v.geometry!=Geometry::Mesh || length<=2*v.visibility_tolerance)return true;
  flat::Ray ray{a,scale(sub(b,a),1./length),0};
  indexed::View data{v.mesh.nodes,v.mesh.indexed_triangles,v.mesh.indices,v.mesh.vertices,v.mesh.node_count};
  for(std::uint32_t i=0;i<data.node_count;){
    auto const& node=data.nodes[i];
    if(!indexed::slab(node,ray)){i=node.skip;continue;}
    for(std::uint32_t j=0;j<node.count;++j){
      auto id=data.indices[node.first+j];
      if(v.coverage.enabled()&&v.mesh.indexed_triangles[id].normal.z<=0.)continue;
      auto h=indexed::triangleHit(data,id,ray);
      if(h.found() && h.distance>v.visibility_tolerance && h.distance<length-v.visibility_tolerance)return false;
    }
    ++i;
  }
  return true;
}
// Clip a linear source against a half-space f(s) >= 0.
C8_INTERFACE_RADIO_INLINE bool clipShadow(double a,double b,double& lo,double& hi){
  double slope=b-a;
  if(slope==0.)return a>=0.;
  double root=-a/slope;
  if(slope>0.)lo=max(lo,root);else hi=min(hi,root);
  return lo<hi;
}
C8_INTERFACE_RADIO_INLINE double directShadowBoundary(PropagationView const& v,
    Vec3 start,Vec3 end,Vec3 observer,double begin,double finish){
  double best=-1.,mid=.5*(begin+finish),distance=HUGE_VAL;
  double eps=max(1.e-13,v.visibility_tolerance/max(1.,norm(sub(end,start))));
  auto delta=sub(end,start),a=add(start,scale(delta,begin)),b=add(start,scale(delta,finish));
  Vec3 lower{min(observer.x,min(a.x,b.x)),min(observer.y,min(a.y,b.y)),min(observer.z,min(a.z,b.z))};
  Vec3 upper{max(observer.x,max(a.x,b.x)),max(observer.y,max(a.y,b.y)),max(observer.z,max(a.z,b.z))};
  for(std::uint32_t nodeId=0;nodeId<v.mesh.node_count;){
    auto const& node=v.mesh.nodes[nodeId];
    if(node.high.x<lower.x||node.low.x>upper.x||node.high.y<lower.y||node.low.y>upper.y||node.high.z<lower.z||node.low.z>upper.z){nodeId=node.skip;continue;}
    ++nodeId;
    for(std::uint32_t j=0;j<node.count;++j){
    auto id=v.mesh.indices[node.first+j];
    auto t=v.mesh.indexed_triangles[id];
    if(v.coverage.enabled()&&t.normal.z<=0.)continue;
    Vec3 vertex[3]{v.mesh.vertices[t.vertex[0]],v.mesh.vertices[t.vertex[1]],v.mesh.vertices[t.vertex[2]]};
    double side=dot(sub(observer,vertex[0]),t.normal);
    if(std::abs(side)<=v.visibility_tolerance)continue;
    Vec3 normal=scale(t.normal,side>0.?-1.:1.);
    double lo=0.,hi=1.;
    bool ok=clipShadow(dot(sub(start,vertex[0]),normal),dot(sub(end,vertex[0]),normal),lo,hi);
    for(int edge=0;edge<3&&ok;++edge){
      normal=cross(sub(vertex[edge],observer),sub(vertex[(edge+1)%3],observer));
      if(dot(normal,sub(vertex[(edge+2)%3],observer))<0.)normal=scale(normal,-1.);
      ok=clipShadow(dot(sub(start,observer),normal),dot(sub(end,observer),normal),lo,hi);
    }
    if(!ok)continue;
    double bounds[2]{lo,hi};
    for(double x:bounds)if(x>begin+eps&&x<finish-eps&&std::abs(x-mid)<distance){best=x;distance=std::abs(x-mid);}
    }
  }
  return best;
}
C8_INTERFACE_RADIO_INLINE bool containsTriangle(PropagationView const& v,std::uint32_t id,Vec3 p) {
  auto const& t=v.mesh.indexed_triangles[id];
  auto a=v.mesh.vertices[t.vertex[0]],b=v.mesh.vertices[t.vertex[1]],c=v.mesh.vertices[t.vertex[2]];
  auto u=sub(b,a),w=sub(c,a),q=sub(p,a);
  double uu=dot(u,u),uw=dot(u,w),ww=dot(w,w),qu=dot(q,u),qw=dot(q,w);
  double det=uu*ww-uw*uw;if(!(det>0.))return false;
  double x=(ww*qu-uw*qw)/det,y=(uu*qw-uw*qu)/det;
  double eps=v.edge_tolerance/max(1.,max(std::sqrt(uu),std::sqrt(ww)));
  return x>=-eps&&y>=-eps&&x+y<=1.+eps;
}
// Every positive-index planar Snell root lies between the two orthogonal
// endpoint projections. Reject only when that entire segment is outside one
// triangle half-plane. This avoids solving Snell on impossible DEM faces,
// without selecting a particular refractive index or rejecting edge roots.
C8_INTERFACE_RADIO_INLINE bool projectedSegmentMayHit(PropagationView const& v,
    std::uint32_t id,Vec3 a,Vec3 b) {
  auto const& t=v.mesh.indexed_triangles[id];auto origin=v.mesh.vertices[t.vertex[0]];
  auto u=sub(v.mesh.vertices[t.vertex[1]],origin),w=sub(v.mesh.vertices[t.vertex[2]],origin);
  double uu=dot(u,u),uw=dot(u,w),ww=dot(w,w),det=uu*ww-uw*uw;
  if(!(det>0.))return true; // Leave malformed/ill-conditioned geometry to the solver.
  auto qa=sub(a,origin),qb=sub(b,origin);
  double au=dot(qa,u),aw=dot(qa,w),bu=dot(qb,u),bw=dot(qb,w);
  double ax=ww*au-uw*aw,ay=uu*aw-uw*au,bx=ww*bu-uw*bw,by=uu*bw-uw*bu;
  double roundoff=128*0x1p-52*(std::abs(uu*ww)+std::abs(uw*uw)+
    std::abs(ww*au)+std::abs(uw*aw)+std::abs(uu*aw)+std::abs(uw*au)+
    std::abs(ww*bu)+std::abs(uw*bw)+std::abs(uu*bw)+std::abs(uw*bu));
  double margin=v.edge_tolerance/max(1.,max(std::sqrt(uu),std::sqrt(ww)))*det+roundoff;
  return !(max(ax,bx)<-margin||max(ay,by)<-margin||max(det-ax-ay,det-bx-by)<-margin);
}
C8_INTERFACE_RADIO_INLINE void face(PropagationView const& v,std::uint32_t id,Vec3& point,Vec3& normal) {
  if(v.geometry==Geometry::Plane){point=v.plane_point;normal=v.outward;return;}
  auto const& t=v.mesh.indexed_triangles[id];point=v.mesh.vertices[t.vertex[0]];normal=t.normal;
}
// Coplanar shared triangles describe one optical branch. Canonical ownership
// is the smallest face ID containing the solved point; adjacent crease faces
// are separate GO branches and must not be silently merged.
C8_INTERFACE_RADIO_INLINE bool ownsPoint(PropagationView const& v,std::uint32_t id,Vec3 p,Vec3 normal,bool use_bvh=true) {
  if(v.geometry!=Geometry::Mesh||id==0)return true;
  if(!use_bvh||!v.mesh.node_count){
    for(std::uint32_t k=0;k<id;++k){
      Vec3 q,n;face(v,k,q,n);
      if(dot(n,normal)>1.-1.e-12 && std::abs(dot(sub(p,q),n))<=v.edge_tolerance && containsTriangle(v,k,p))return false;
    }
    return true;
  }
  for(std::uint32_t nodeId=0;nodeId<v.mesh.node_count;){
    auto const& node=v.mesh.nodes[nodeId];
    // Barycentric acceptance extends at most two edge tolerances beyond the
    // vertex box; the plane test adds one. Include coordinate roundoff too.
    double magnitude=max(1.,max(std::abs(p.x),max(std::abs(p.y),std::abs(p.z))));
    magnitude=max(magnitude,max(std::abs(node.low.x),max(std::abs(node.low.y),std::abs(node.low.z))));
    magnitude=max(magnitude,max(std::abs(node.high.x),max(std::abs(node.high.y),std::abs(node.high.z))));
    double margin=4*v.edge_tolerance+128*0x1p-52*magnitude;
    if(p.x<node.low.x-margin||p.x>node.high.x+margin||p.y<node.low.y-margin||p.y>node.high.y+margin||p.z<node.low.z-margin||p.z>node.high.z+margin){nodeId=node.skip;continue;}
    ++nodeId;
    for(std::uint32_t j=0;j<node.count;++j){
      auto k=v.mesh.indices[node.first+j];if(k>=id)continue;
      Vec3 q,n;face(v,k,q,n);
      if(dot(n,normal)>1.-1.e-12 && std::abs(dot(sub(p,q),n))<=v.edge_tolerance && containsTriangle(v,k,p))return false;
    }
  }
  return true;
}
C8_INTERFACE_RADIO_INLINE double snellResidual(double x,double d1,double d2,double lateral,double n1,double n2) {
  return n1*x/std::hypot(d1,x)-n2*(lateral-x)/std::hypot(d2,lateral-x);
}
C8_INTERFACE_RADIO_INLINE bool snellRootNearStart(double d1,double d2,double lateral,double n1,double n2,double& x,unsigned& iterations) {
  if(lateral==0.){x=0.;return true;}
  double low=0.,high=.5*lateral;x=min(high,lateral*d1/(d1+d2));
  for(unsigned j=0;j<64;++j){
    ++iterations;double value=snellResidual(x,d1,d2,lateral,n1,n2);
    if(std::abs(value)<1.e-14)return true;
    if(value>0.)high=x;else low=x;
    double r1=std::hypot(d1,x),r2=std::hypot(d2,lateral-x);
    double derivative=n1*d1*d1/(r1*r1*r1)+n2*d2*d2/(r2*r2*r2);
    double candidate=x-value/derivative;
    if(!(candidate>low&&candidate<high)||!std::isfinite(candidate))candidate=.5*(low+high);
    if(candidate==x)break;
    x=candidate;
  }
  return std::abs(snellResidual(x,d1,d2,lateral,n1,n2))<1.e-11;
}
// Solve the smaller lateral coordinate directly. Recovering a micrometre
// receiver leg by subtracting two kilometre coordinates can make an otherwise
// valid root fail the Snell residual gate. Keep BOTH coordinates for geometry.
C8_INTERFACE_RADIO_INLINE bool snellCoordinates(double d1,double d2,double lateral,double n1,double n2,double& x,double& y,unsigned& iterations){
  if(lateral==0.){x=y=0.;return true;}
  bool reverse=snellResidual(.5*lateral,d1,d2,lateral,n1,n2)<0.;
  bool ok=reverse?snellRootNearStart(d2,d1,lateral,n2,n1,y,iterations):snellRootNearStart(d1,d2,lateral,n1,n2,x,iterations);
  if(reverse)x=lateral-y;else y=lateral-x;
  return ok;
}
C8_INTERFACE_RADIO_INLINE bool snellRoot(double d1,double d2,double lateral,double n1,double n2,double& x,unsigned& iterations){
  double y;return snellCoordinates(d1,d2,lateral,n1,n2,x,y,iterations);
}
C8_INTERFACE_RADIO_INLINE bool spreading(double d1,double d2,double n1,double n2,double invariant,double& jacobian,double& geometric) {
  if(!(invariant>=0.&&invariant<min(n1,n2)))return false;
  double r1=std::sqrt(n1*n1-invariant*invariant),r2=std::sqrt(n2*n2-invariant*invariant);
  if(invariant>1.e-12*min(n1,n2)){
    double horizontal=d1*invariant/r1+d2*invariant/r2;
    double derivative=d1*n1*n1/(r1*r1*r1)+d2*n2*n2/(r2*r2*r2);
    jacobian=horizontal*(r2/n2)*derivative*r1/(invariant/n1);
  }else{double axial=n1*(d1/n1+d2/n2);jacobian=axial*axial;}
  geometric=std::sqrt(n1/(n2*jacobian));
  return std::isfinite(geometric)&&geometric>0.;
}
struct Fresnel { bool transmitted{}; double cos_transmitted{},t_s{},t_p{}; };
C8_INTERFACE_RADIO_INLINE Fresnel fresnel(double n1,double n2,double ci) {
  Fresnel result;
  if(!(n1>0.&&n2>0.&&ci>=0.&&ci<=1.))return result;
  double st=n1/n2*std::sqrt(max(0.,1.-ci*ci));
  if(st>1.)return result;
  result.transmitted=true;result.cos_transmitted=std::sqrt(max(0.,1.-st*st));
  double ct=result.cos_transmitted;
  result.t_s=2*n1*ci/(n1*ci+n2*ct);result.t_p=2*n1*ci/(n2*ci+n1*ct);
  return result;
}
C8_INTERFACE_RADIO_INLINE void transfer(Path& p,double sFactor,double pFactor) {
  Vec3 s=cross(p.emit,p.normal);
  if(norm(s)<1.e-12){Vec3 trial{0.,0.,1.};if(std::abs(dot(trial,p.emit))>.9)trial={0.,1.,0.};s=cross(p.emit,trial);}
  s=unit(s);Vec3 ps=unit(cross(s,p.emit)),pd=unit(cross(s,p.receive_direction));
  double sv[3]{s.x,s.y,s.z},a[3]{ps.x,ps.y,ps.z},b[3]{pd.x,pd.y,pd.z};
  for(int i=0;i<3;++i)for(int j=0;j<3;++j)p.transfer_per_m[3*i+j]=sFactor*sv[i]*sv[j]+pFactor*b[i]*a[j];
}
C8_INTERFACE_RADIO_INLINE Vec3 applyTransfer(Path const& p,Vec3 source) {
  double x[3]{source.x,source.y,source.z},out[3]{};
  for(int i=0;i<3;++i)for(int j=0;j<3;++j)out[i]+=p.transfer_per_m[3*i+j]*x[j];
  return {out[0],out[1],out[2]};
}
C8_INTERFACE_RADIO_INLINE Path directPath(PropagationView const& v,RayQuery q) {
  Path p;double length=norm(sub(q.observer,q.source));
  if(q.source_region>1||q.observer_region>1||q.source_region!=q.observer_region||
     !finite(q.source)||!finite(q.observer)||!(length>0.)){p.status=PathStatus::Invalid;return p;}
  if(!terrain::coverage::segmentInside(v.coverage,q.source,q.observer)){p.status=PathStatus::OutsideCoverage;return p;}
  if(!segmentClear(v,q.source,q.observer)){p.status=PathStatus::Blocked;return p;}
  auto const& m=v.media[q.source_region];p.emit=unit(sub(q.observer,q.source));p.receive_direction=p.emit;
  p.source_index=indexAt(m,q.source);p.destination_index=indexAt(m,q.observer);
  p.interface_source_index=p.source_index;p.interface_destination_index=p.destination_index;
  p.source_length_m=length;p.time_s=opticalLength(m,q.source,q.observer,v.integration_samples)/light_speed;
  p.spreading_per_m=1./length;p.jacobian_m2=length*length;
  p.attenuation=std::exp(-length/m.attenuation_length_m);
  p.normal=p.emit;transfer(p,p.spreading_per_m*p.attenuation,p.spreading_per_m*p.attenuation);
  p.status=std::isfinite(p.time_s)&&std::isfinite(p.source_index)&&std::isfinite(p.destination_index)?PathStatus::Valid:PathStatus::Invalid;
  return p;
}
C8_INTERFACE_RADIO_INLINE Path transmittedPath(PropagationView const& v,RayQuery q,std::uint32_t id,bool unbounded=false,bool candidate_filter=true) {
  Path p;p.face=id;
  if(q.source_region>1||q.observer_region>1||q.source_region==q.observer_region||
     !finite(q.source)||!finite(q.observer)||v.geometry==Geometry::Uniform){p.status=PathStatus::Invalid;return p;}
  Vec3 point,outward;face(v,id,point,outward);
  if(v.coverage.enabled()&&outward.z<=0.)return p; // artificial closure, no optical surface
  Vec3 n=q.source_region==1?outward:scale(outward,-1.);
  double d1=dot(sub(point,q.source),n),d2=dot(sub(q.observer,point),n);
  if(!(d1>1.e-10&&d2>1.e-10))return p;
  auto a=add(q.source,scale(n,d1)),b=sub(q.observer,scale(n,d2)),delta=sub(b,a);
  if(candidate_filter&&!unbounded&&v.geometry==Geometry::Mesh&&!projectedSegmentMayHit(v,id,a,b))return p;
  // Restore tangency lost to rounding in the two global projections.
  delta=sub(delta,scale(n,dot(delta,n)));
  double lateral=norm(delta);Vec3 tangent=lateral>0.?scale(delta,1./lateral):Vec3{};
  auto const& m1=v.media[q.source_region];auto const& m2=v.media[q.observer_region];
  double n1=indexAt(m1,point),n2=indexAt(m2,point),x=0.,y=0.;bool converged=false;
  for(unsigned j=0;j<16;++j){
    if(!(n1>0.&&n2>0.)){p.status=PathStatus::Invalid;return p;}
    if(!snellCoordinates(d1,d2,lateral,n1,n2,x,y,p.iterations)){p.status=PathStatus::Unconverged;return p;}
    p.interface_point=x<=y?add(a,scale(tangent,x)):sub(b,scale(tangent,y));
    double next1=indexAt(m1,p.interface_point),next2=indexAt(m2,p.interface_point);
    if((!std::isfinite(next1)||!std::isfinite(next2))&&v.geometry==Geometry::Mesh&&
       !containsTriangle(v,id,p.interface_point))return p;
    if(std::abs(next1-n1)<1.e-13&&std::abs(next2-n2)<1.e-13){converged=true;break;}
    n1=next1;n2=next2;
  }
  if(!converged){p.status=PathStatus::Unconverged;return p;}
  if(!unbounded&&v.geometry==Geometry::Mesh&&(!containsTriangle(v,id,p.interface_point)||!ownsPoint(v,id,p.interface_point,outward,candidate_filter)))return p;
  if(!terrain::coverage::segmentInside(v.coverage,q.source,p.interface_point)||
      !terrain::coverage::segmentInside(v.coverage,p.interface_point,q.observer)){p.status=PathStatus::OutsideCoverage;return p;}
  if(!unbounded&&(!segmentClear(v,q.source,p.interface_point)||!segmentClear(v,p.interface_point,q.observer))){p.status=PathStatus::Blocked;return p;}
  p.source_length_m=std::hypot(d1,x);p.destination_length_m=std::hypot(d2,y);
  p.emit=unit(add(scale(n,d1),scale(tangent,x)));
  p.receive_direction=unit(add(scale(n,d2),scale(tangent,y)));p.normal=n;
  double ci=clamp(dot(p.emit,n),0.,1.),ct=clamp(dot(p.receive_direction,n),0.,1.);
  double si=norm(cross(p.emit,n)),st=norm(cross(p.receive_direction,n));
  p.snell_residual=n1*si-n2*st;
  if(std::abs(p.snell_residual)>1.e-10){p.status=PathStatus::Unconverged;return p;}
  p.t_s=2*n1*ci/(n1*ci+n2*ct);p.t_p=2*n1*ci/(n2*ci+n1*ct);
  if(!spreading(d1,d2,n1,n2,n1*si,p.jacobian_m2,p.spreading_per_m)){p.status=PathStatus::Invalid;return p;}
  p.source_index=indexAt(m1,q.source);p.destination_index=indexAt(m2,q.observer);
  p.interface_source_index=n1;p.interface_destination_index=n2;
  p.time_s=(opticalLength(m1,q.source,p.interface_point,v.integration_samples)+
            opticalLength(m2,p.interface_point,q.observer,v.integration_samples))/light_speed;
  p.attenuation=std::exp(-p.source_length_m/m1.attenuation_length_m-p.destination_length_m/m2.attenuation_length_m);
  double factor=p.spreading_per_m*std::sqrt(n2*ct/(n1*ci))*p.attenuation;
  transfer(p,factor*p.t_s,factor*p.t_p);
  p.status=std::isfinite(p.time_s)&&std::isfinite(p.source_index)&&std::isfinite(p.destination_index)?PathStatus::Valid:PathStatus::Invalid;
  return p;
}
} // namespace corsika::radio::interface::detail

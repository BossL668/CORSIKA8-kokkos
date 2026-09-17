/* Existing interface ZHS kernel, shared for the CoREAS Cherenkov limit. */
#pragma once
#include <corsika/modules/radio/interface/KokkosPropagation.hpp>
namespace corsika::radio::interface {
namespace detail {
struct RadioCounters {
  unsigned long long device_tracks{},cpu_tracks{},pairs{},leaves{},paths{},direct{},transmitted{},blocked{},missing{},out_of_window{};
  unsigned long long endpoint_contributions{},regularized_pairs{},boundary_endpoints{};
  unsigned long long roundoff_limited_tracks{},outside_coverage{};
  std::uint32_t error{};
};
// Endpoint coordinates and absolute clocks are rounded independently. Their
// differences can put a micrometre chord slightly outside the light cone at
// kilometre coordinates. Canonicalize beta only within a bound derived from
// those input ULPs and the source transport's declared chord accuracy; a
// source outside both budgets is still an error. Transport
// coordinates, times, energy and RNG are never modified by this radio helper.
C8_INTERFACE_RADIO_INLINE double inputUlp(double x){
  x=std::abs(x);return std::nextafter(x,HUGE_VAL)-x;
}
C8_INTERFACE_RADIO_INLINE bool sourceBeta(Track const& t,Vec3 displacement,double length,double dt,Vec3& beta,bool& limited){
  limited=false;double ct=light_speed*dt;
  if(!(dt>0.)||!std::isfinite(length)||!std::isfinite(ct)||!(ct>0.)||
     !std::isfinite(t.relative_chord_tolerance)||t.relative_chord_tolerance<0.||t.relative_chord_tolerance>1.e-9)return false;
  if(length>ct){
    double uncertainty=inputUlp(t.start_m.x)+inputUlp(t.start_m.y)+inputUlp(t.start_m.z)+
      inputUlp(t.end_m.x)+inputUlp(t.end_m.y)+inputUlp(t.end_m.z)+
      light_speed*(inputUlp(t.start_time_s)+inputUlp(t.end_time_s))+16*0x1p-52*(length+ct);
    uncertainty=max(uncertainty,ct*t.relative_chord_tolerance);
    if(!std::isfinite(uncertainty)||length-ct>uncertainty)return false;
    beta=scale(displacement,1./length);limited=true;
  }else beta=scale(displacement,1./ct);
  return finite(beta);
}
struct DeviceObserver {Vec3 position{};std::uint32_t region{};};
struct MomentGrid {double start_time{},sample_rate{};std::size_t samples{},channels{};unsigned order{};};
// Atomics operate on shared waves rather than 256 copies on OpenMP. Floating
// addition is compared by tolerances in waveform validation, not bit identity.
template<class Values,class Counters>
C8_INTERFACE_RADIO_INLINE void addInterval(Values const& values,Counters const& counters,
    MomentGrid g,double midpoint,double width,Vec3 area,std::size_t channel) {
  if(!std::isfinite(midpoint)||!std::isfinite(width)||width<0.||!finite(area)){
    Kokkos::atomic_max(&counters().error,std::uint32_t{1});return;}
  double x=(midpoint-g.start_time)*g.sample_rate,r=.5*width*g.sample_rate;
  double flo=std::floor(x-r+.5),fhi=std::floor(x+r+.5);
  if(flo<0.||fhi>=g.samples){Kokkos::atomic_fetch_add(&counters().out_of_window,1ULL);Kokkos::atomic_max(&counters().error,std::uint32_t{2});return;}
  auto first=static_cast<std::size_t>(flo),last=static_cast<std::size_t>(fhi);
  double a[3]{area.x,area.y,area.z};
  for(auto k=first;k<=last;++k){
    double u=x-k,lo=max(-.5,u-r),hi=min(.5,u+r);
    double fraction=first==last?1.:(hi-lo)/(2*r);
    if(!(fraction>0.))continue;
    double sum=1.,powerLo=1.,factorial=1.;
    for(unsigned p=0;p<=g.order;++p){
      if(p){powerLo*=lo;sum=hi*sum+powerLo;factorial*=p;}
      double moment=fraction*sum/((p+1)*factorial);
      for(unsigned axis=0;axis<3;++axis){auto i=(p*g.channels+channel+axis)*g.samples+k;
        double v=a[axis]*moment;
        if(!std::isfinite(v))Kokkos::atomic_max(&counters().error,std::uint32_t{1});
        else Kokkos::atomic_add(&values(i),v);
      }
    }
  }
}
template<class Space> struct AccumulateKernel {
  using Memory=typename Space::memory_space;
  Kokkos::View<Track*,Memory> tracks;
  Kokkos::View<DeviceObserver*,Memory> observers;
  Kokkos::View<double*,Memory> moments;
  Kokkos::View<RadioCounters,Memory> counters;
  PropagationView propagation;MomentGrid grid;
  double frequency{},limit{},mesh_segment{};unsigned maximum_depth{};bool cpu_source{};
  C8_INTERFACE_RADIO_INLINE void operator()(std::size_t pair)const {
    auto const observerId=pair%observers.extent(0),trackId=pair/observers.extent(0);
    auto const t=tracks(trackId);if(!t.valid)return;
    auto const o=observers(observerId);
    auto displacement=sub(t.end_m,t.start_m);double dt=t.end_time_s-t.start_time_s,length=norm(displacement);
    if(t.region>1||!finite(t.start_m)||!finite(t.end_m)||!std::isfinite(dt)||!std::isfinite(t.charge_e)||!std::isfinite(t.weight)||t.weight<0.){
      Kokkos::atomic_max(&counters().error,std::uint32_t{3});return;}
    if(t.weight==0.||t.charge_e==0.||length==0.)return;
    Vec3 beta;bool limited;
    if(!sourceBeta(t,displacement,length,dt,beta,limited)){Kokkos::atomic_max(&counters().error,std::uint32_t{3});return;}
    if(observerId==0){
      if(cpu_source)Kokkos::atomic_fetch_add(&counters().cpu_tracks,1ULL);
      else Kokkos::atomic_fetch_add(&counters().device_tracks,1ULL);
      if(limited)Kokkos::atomic_fetch_add(&counters().roundoff_limited_tracks,1ULL);
    }
    Kokkos::atomic_fetch_add(&counters().pairs,1ULL);
    double betaNorm=norm(beta);
    constexpr double pi=3.141592653589793238462643383279502884;
    double coefficient=t.charge_e*accelerator::scalar_constants::ElementaryChargeC*t.weight/
      (4*pi*accelerator::scalar_constants::VacuumPermittivityFPerM*light_speed);
    bool direct=t.region==o.region;
    unsigned faces=direct?1:(propagation.geometry==Geometry::Plane?1:propagation.faces);
    if(!faces){Kokkos::atomic_max(&counters().error,std::uint32_t{4});return;}
    struct Segment {double a,b;unsigned depth;};
    Segment stack[22];unsigned size=1;stack[0]={0.,1.,0};
    while(size){
      auto s=stack[--size];double mid=.5*(s.a+s.b),fraction=s.b-s.a,leafLength=length*fraction;
      RayQuery q{add(t.start_m,scale(displacement,mid)),o.position,t.region,o.region};
      bool split=!direct&&propagation.geometry==Geometry::Mesh&&leafLength>mesh_segment;
      // For straight direct rays, partition exactly at each triangle shadow
      // boundary. This captures partially visible and narrow visible sources,
      // including a blocked midpoint with visible endpoints.
      double splitAt=mid;
      if(direct&&propagation.geometry==Geometry::Mesh){
        double boundary=directShadowBoundary(propagation,t.start_m,t.end_m,o.position,s.a,s.b);
        if(boundary>s.a&&boundary<s.b){split=true;splitAt=boundary;}
      }
      CandidateFaces refineFaces{propagation,q,direct};std::uint32_t face;
      while(refineFaces.next(face)){
        auto p=direct?directPath(propagation,q):transmittedPath(propagation,q,face);
        if(p.status==PathStatus::Invalid||p.status==PathStatus::Unconverged){Kokkos::atomic_max(&counters().error,std::uint32_t{5});return;}
        if(p.status!=PathStatus::Valid)continue;
        double cosine=dot(beta,p.emit)/betaNorm;
        double distance=p.source_length_m+p.destination_length_m;
        // The Fraunhofer term vanishes on axis although 1/R and polarization
        // still vary along a long segment. Bound its relative geometric size
        // too; tightening the control must refine those segments as well.
        double criterion=max(leafLength/distance,
          max(0.,1.-cosine*cosine)*leafLength*leafLength/distance*(frequency/light_speed)*2*pi);
        if(criterion>limit)split=true;
      }
      if(split){
        if(s.depth==maximum_depth){Kokkos::atomic_max(&counters().error,std::uint32_t{6});return;}
        stack[size++]={splitAt,s.b,s.depth+1};stack[size++]={s.a,splitAt,s.depth+1};continue;
      }
      Kokkos::atomic_fetch_add(&counters().leaves,1ULL);
      CandidateFaces emitFaces{propagation,q,direct};
      while(emitFaces.next(face)){
        auto p=direct?directPath(propagation,q):transmittedPath(propagation,q,face);
        if(p.status==PathStatus::OutsideCoverage){Kokkos::atomic_fetch_add(&counters().outside_coverage,1ULL);continue;}
        if(p.status==PathStatus::Blocked){Kokkos::atomic_fetch_add(&counters().blocked,1ULL);continue;}
        if(p.status!=PathStatus::Valid){Kokkos::atomic_fetch_add(&counters().missing,1ULL);continue;}
        auto perpendicular=sub(beta,scale(p.emit,dot(beta,p.emit)));
        auto area=scale(applyTransfer(p,perpendicular),coefficient*dt*fraction);
        double denominator=std::fma(-p.source_index,dot(beta,p.emit),1.);
        double time=t.start_time_s+mid*dt+p.time_s,width=std::abs(denominator)*dt*fraction;
        // A straight path in n(h) is not a Fermat ray. Its optical-delay
        // derivative is not simply -n(source) k/c. Use the two optical times
        // to form its detection interval; retain the stable local formula for
        // homogeneous legs and for a source endpoint exactly on an interface.
        if(propagation.media[t.region].count||propagation.media[o.region].count){
          Vec3 a=add(t.start_m,scale(displacement,s.a)),b=add(t.start_m,scale(displacement,s.b));
          if(direct){
            auto const& medium=propagation.media[t.region];
            width=std::abs(dt*fraction+(opticalLength(medium,b,o.position,propagation.integration_samples)-
              opticalLength(medium,a,o.position,propagation.integration_samples))/light_speed);
          }else{
            auto first=transmittedPath(propagation,{a,o.position,t.region,o.region},face,true);
            auto last=transmittedPath(propagation,{b,o.position,t.region,o.region},face,true);
            if(first.status==PathStatus::Valid&&last.status==PathStatus::Valid)width=std::abs(dt*fraction+last.time_s-first.time_s);
          }
        }
        addInterval(moments,counters,grid,time,width,area,observerId*6+t.region*3);
        Kokkos::atomic_fetch_add(&counters().paths,1ULL);
        if(direct)Kokkos::atomic_fetch_add(&counters().direct,1ULL);
        else Kokkos::atomic_fetch_add(&counters().transmitted,1ULL);
      }
    }
  }
};
} // namespace detail

} // namespace corsika::radio::interface

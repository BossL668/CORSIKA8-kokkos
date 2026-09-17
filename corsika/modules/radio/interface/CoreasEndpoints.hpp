/* Independent CoREAS pairs for the interface radiation-only far-field model.
 * Adapted from mountain/cpp/c8_mountain_corsika.cpp: MountainCoREAS.
 * Each accepted subtrack uses one midpoint optical path for both endpoints.
 * This is the same Fraunhofer current model used by interface ZHS. Evaluating
 * two independent endpoint amplitudes without the omitted near-field terms
 * leaves a spurious DC/low-frequency component and is not used here.
 * Original air CoREAS code is neither included nor modified.
 */
#pragma once
#include <corsika/modules/radio/interface/ZhsIntervals.hpp>
namespace corsika::radio::interface::detail {
C8_INTERFACE_RADIO_INLINE bool sourceOnFace(PropagationView const& v,
    unsigned id,Vec3 source){
  Vec3 point,normal;face(v,id,point,normal);
  double tolerance=1.e-10+64*0x1p-52*max(1.,max(norm(point),norm(source)));
  return std::abs(dot(sub(source,point),normal))<=tolerance&&
    (v.geometry!=Geometry::Mesh||containsTriangle(v,id,source));
}
// Deposit two delta fields with opposite signed strengths area/span. For a
// common cell evaluate the polynomial divided difference directly: this
// preserves exact zero area and avoids subtracting two large close impulses.
// The recurrence follows from (b^p-a^p)/(b-a), not from a ZHS potential grid.
template<class Values,class Counters>
C8_INTERFACE_RADIO_INLINE void addEndpointPair(Values const& values,Counters const& counters,
    MomentGrid g,double midpoint,double span,Vec3 area,std::size_t channel){
  if(!std::isfinite(midpoint)||!std::isfinite(span)||!finite(area)){
    Kokkos::atomic_max(&counters().error,std::uint32_t{1});return;
  }
  double x=(midpoint-g.start_time)*g.sample_rate,r=.5*span*g.sample_rate;
  double firstBin=std::floor(x-r+.5),lastBin=std::floor(x+r+.5);
  if(min(firstBin,lastBin)<0.||max(firstBin,lastBin)>=g.samples){
    Kokkos::atomic_fetch_add(&counters().out_of_window,1ULL);
    Kokkos::atomic_max(&counters().error,std::uint32_t{2});return;
  }
  double components[3]{area.x,area.y,area.z};
  if(firstBin==lastBin){
    auto cell=static_cast<std::size_t>(firstBin);double u=x-firstBin;
    double a=u-r,b=u+r,sum=1.,powerA=1.,factorial=1.;
    // Order zero cancels exactly within this pair, including a vanishing span.
    for(unsigned order=1;order<=g.order;++order){
      if(order>1){powerA*=a;sum=b*sum+powerA;}
      factorial*=order;
      for(unsigned axis=0;axis<3;++axis){
        double value=components[axis]*g.sample_rate*sum/factorial;
        if(!std::isfinite(value))Kokkos::atomic_max(&counters().error,std::uint32_t{1});
        else Kokkos::atomic_add(&values((order*g.channels+channel+axis)*g.samples+cell),value);
      }
    }
  }else{
    for(unsigned endpoint=0;endpoint<2;++endpoint){
      double bin=endpoint?lastBin:firstBin;
      auto cell=static_cast<std::size_t>(bin);
      double offset=(x-bin)+(endpoint?r:-r),power=1.;
      for(unsigned order=0;order<=g.order;++order){
        if(order)power*=offset/order;
        for(unsigned axis=0;axis<3;++axis){
          double value=(endpoint?1.:-1.)*components[axis]/span*power;
          if(!std::isfinite(value))Kokkos::atomic_max(&counters().error,std::uint32_t{1});
          else Kokkos::atomic_add(&values((order*g.channels+channel+axis)*g.samples+cell),value);
        }
      }
    }
  }
}
template<class Space> struct CoreasEndpointKernel {
  using Memory=typename Space::memory_space;
  Kokkos::View<Track*,Memory> tracks;
  Kokkos::View<DeviceObserver*,Memory> observers;
  Kokkos::View<double*,Memory> moments,regularized;
  Kokkos::View<RadioCounters,Memory> counters;
  PropagationView propagation;MomentGrid grid;
  double frequency{},limit{},mesh_segment{};unsigned maximum_depth{};bool cpu_source{};
  double cherenkov_threshold{};
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
    bool usedLimit=false;
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
        double time=t.start_time_s+mid*dt+p.time_s,span=denominator*dt*fraction;
        // A straight path in n(h) is not a Fermat ray. Its optical-delay
        // derivative is not simply -n(source) k/c. Use the two optical times
        // to form its detection interval; retain the stable local formula for
        // homogeneous legs and for a source endpoint exactly on an interface.
        if(propagation.media[t.region].count||propagation.media[o.region].count){
          Vec3 a=add(t.start_m,scale(displacement,s.a)),b=add(t.start_m,scale(displacement,s.b));
          if(direct){
            auto const& medium=propagation.media[t.region];
            span=dt*fraction+(opticalLength(medium,b,o.position,propagation.integration_samples)-
              opticalLength(medium,a,o.position,propagation.integration_samples))/light_speed;
          }else{
            auto first=transmittedPath(propagation,{a,o.position,t.region,o.region},face,true);
            auto last=transmittedPath(propagation,{b,o.position,t.region,o.region},face,true);
            if(first.status==PathStatus::Valid&&last.status==PathStatus::Valid)span=dt*fraction+last.time_s-first.time_s;
          }
        }
        // Both algorithms use the same local current and optical linearization.
        // Ordinary CoREAS emits a pair of opposite endpoint impulses. Near a
        // zero arrival-time derivative, use its finite-current limit instead.
        double effectiveDoppler=span/(dt*fraction);
        double gridX=(time-grid.start_time)*grid.sample_rate;
        double halfSpan=.5*span*grid.sample_rate;
        bool crossesCell=std::floor(gridX-halfSpan+.5)!=std::floor(gridX+halfSpan+.5);
        bool limitPair=std::abs(effectiveDoppler)<cherenkov_threshold||
          (crossesCell&&std::abs(span)*grid.sample_rate<1.e-4);
        if(limitPair){
          addInterval(regularized,counters,grid,time,std::abs(span),area,observerId*6+t.region*3);
          usedLimit=true;
        }else{
          addEndpointPair(moments,counters,grid,time,span,area,observerId*6+t.region*3);
          Kokkos::atomic_fetch_add(&counters().endpoint_contributions,2ULL);
        }
        if(!direct){
          if(s.a==0.&&sourceOnFace(propagation,face,t.start_m))
            Kokkos::atomic_fetch_add(&counters().boundary_endpoints,1ULL);
          if(s.b==1.&&sourceOnFace(propagation,face,t.end_m))
            Kokkos::atomic_fetch_add(&counters().boundary_endpoints,1ULL);
        }
        Kokkos::atomic_fetch_add(&counters().paths,1ULL);
        if(direct)Kokkos::atomic_fetch_add(&counters().direct,1ULL);
        else Kokkos::atomic_fetch_add(&counters().transmitted,1ULL);
      }
    }
    if(usedLimit)Kokkos::atomic_fetch_add(&counters().regularized_pairs,1ULL);
  }
};
} // namespace corsika::radio::interface::detail

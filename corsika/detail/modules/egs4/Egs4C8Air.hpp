#pragma once
#include <corsika/detail/modules/egs4/OverheadOptions.hpp>
#include <corsika/detail/modules/egs4/Egs4C8Adapter.hpp>
#include <corsika/detail/modules/egs4/Egs4Geometry.hpp>
#include <corsika/detail/modules/egs4/Egs4Observation.hpp>

namespace c7_egs4::c8_adapter {
// An explicit choice, not a silent reinterpretation of C8's homogeneous fifth
// layer. EGSIN1 uses four exponential transport regions up to HLAY(6). HOWFAR's
// overburden still uses the original five-layer atmosphere.
enum class AirConvention { c7_egs4_four_exponentials };
struct AirEnvironment {
  unsigned overhead_mask{}; double full_column[5]{};
  Frame observer_frame;
  CurvedConfig curved;
  GeometryConfig geometry;
  AirRegion regions[4]; // C7 indices 2..5, highest first
  ::corsika::gpu::em::EnvironmentSnapshot source;
  MagneticField field;
  std::int32_t medium_id{};
};
KOKKOS_INLINE_FUNCTION double columnAbove(AirEnvironment const& e,double height_cm) {
  double radius=e.curved.earth_radius_cm*.01+height_cm*.01,answer=0.;
  for(unsigned i=0;i<e.source.number_of_layers;++i) {
    auto l=e.source.atmosphere_layers[i];double lo=larger(radius,l.inner_radius_m),hi=l.outer_radius_m;
    if(lo>=hi)continue;
    if((e.overhead_mask&overhead::column_cache)&&radius<=l.inner_radius_m) {answer+=e.full_column[i];continue;}
    if(l.density_model==::corsika::gpu::em::DensityModel::Homogeneous)
      answer+=100.*l.density_parameter_a*(hi-lo);
    else answer+=100.*l.density_parameter_a*::exp((lo-l.density_parameter_b)/l.density_parameter_c)*
      l.density_parameter_c*::expm1((hi-lo)/l.density_parameter_c);
  }
  return answer;
}
inline AirEnvironment makeAirEnvironment(::corsika::gpu::em::EnvironmentSnapshot const& snapshot,
    double earth_radius_m,double sterncor,AirConvention convention) {
  using namespace ::corsika::gpu::em;
  if(convention!=AirConvention::c7_egs4_four_exponentials||!finite(earth_radius_m)||earth_radius_m<=0.||
     !finite(sterncor)||snapshot.geometry!=EnvironmentGeometry::SphericalLayers||snapshot.number_of_layers!=5)
    throw std::invalid_argument("Native EGS4 requires an explicit four-exponential C7 air convention and a five-layer C8 snapshot");
  AirEnvironment e;e.source=snapshot;e.medium_id=snapshot.atmosphere_layers[0].medium_id;
  for(unsigned i=0;i<5;++i) {
    auto l=snapshot.atmosphere_layers[i];
    if(!finite(l.inner_radius_m)||!finite(l.outer_radius_m)||l.outer_radius_m<=l.inner_radius_m||
       !finite(l.density_parameter_a)||l.density_parameter_a<=0.||l.medium_id!=e.medium_id||
       (i&&::fabs(l.inner_radius_m-snapshot.atmosphere_layers[i-1].outer_radius_m)>1.e-7))
      throw std::invalid_argument("Non-contiguous or multi-material native EGS4 atmosphere");
    if(i<4) {
      if(l.density_model!=DensityModel::Exponential||!finite(l.density_parameter_b)||
         !finite(l.density_parameter_c)||l.density_parameter_c>=0.)
        throw std::invalid_argument("EGS4 air regions require exponential density");
      double rho=l.density_parameter_a*std::exp((earth_radius_m-l.density_parameter_b)/l.density_parameter_c);
      e.regions[3-i]={rho,-.01/l.density_parameter_c,sterncor};
      if(!validRegion(e.regions[3-i]))throw std::invalid_argument("Invalid EGS4 reference density");
    } else if(l.density_model!=DensityModel::Homogeneous)
      throw std::invalid_argument("C8 fifth atmosphere layer must be homogeneous for this C7 convention");
  }
  Direction up{snapshot.observation_plane_normal[0],snapshot.observation_plane_normal[1],snapshot.observation_plane_normal[2]};
  Direction center{snapshot.earth_center_m[0],snapshot.earth_center_m[1],snapshot.earth_center_m[2]};
  Direction plane{snapshot.observation_plane_point_m[0],snapshot.observation_plane_point_m[1],snapshot.observation_plane_point_m[2]};
  Direction b{snapshot.magnetic_field_T[0],snapshot.magnetic_field_T[1],snapshot.magnetic_field_T[2]};
  if(::fabs(dot(up,up)-1.)>1.e-12||!finite(dot(up,center))||!finite(dot(up,plane))||!finite(dot(b,b)))
    throw std::invalid_argument("Invalid native EGS4 observation plane or field");
  auto& f=e.observer_frame;f.z={-up.x,-up.y,-up.z};
  double vertical=dot(b,f.z);Direction horizontal{b.x-vertical*f.z.x,b.y-vertical*f.z.y,b.z-vertical*f.z.z};
  double norm=std::sqrt(dot(horizontal,horizontal)),magnitude=std::sqrt(dot(b,b));
  if(norm>1.e-20)f.x={horizontal.x/norm,horizontal.y/norm,horizontal.z/norm};
  else {
    Direction ref=::fabs(up.x)<.8?Direction{1.,0.,0.}:Direction{0.,1.,0.};
    double projection=dot(ref,f.z);ref={ref.x-projection*f.z.x,ref.y-projection*f.z.y,ref.z-projection*f.z.z};
    double n=std::sqrt(dot(ref,ref));f.x={ref.x/n,ref.y/n,ref.z/n};
  }
  f.y=cross(f.z,f.x);
  f.origin_m={center.x+earth_radius_m*up.x,center.y+earth_radius_m*up.y,center.z+earth_radius_m*up.z};
  if(!validFrame(f))throw std::invalid_argument("Native EGS4 observer frame is not right handed");
  double height=dot({plane.x-center.x,plane.y-center.y,plane.z-center.z},up)-earth_radius_m;
  e.curved={true,earth_radius_m*100.,height*100.,-1.};
  e.geometry={true,true,earth_radius_m*100.,{},height*100.,6.e5,2.e6,0.};
  e.geometry.bound_cm[0]=(snapshot.atmosphere_layers[4].outer_radius_m-earth_radius_m)*100.;
  for(int i=0;i<4;++i)e.geometry.bound_cm[i+1]=(snapshot.atmosphere_layers[3-i].inner_radius_m-earth_radius_m)*100.;
  e.geometry.bound_cm[5]=e.geometry.bound_cm[4]-1.;
  if(height*100.<e.geometry.bound_cm[4]||height*100.>=e.geometry.bound_cm[0])
    throw std::invalid_argument("Observation plane outside native EGS4 atmosphere");
  e.overhead_mask=overhead::options();
  e.curved.direct_azimuth=(e.overhead_mask&overhead::curved_azimuth)!=0;
  for(unsigned i=0;i<e.source.number_of_layers;++i) {
    auto l=e.source.atmosphere_layers[i];double lo=l.inner_radius_m,hi=l.outer_radius_m;
    e.full_column[i]=l.density_model==::corsika::gpu::em::DensityModel::Homogeneous?
      100.*l.density_parameter_a*(hi-lo):
      100.*l.density_parameter_a*std::exp((lo-l.density_parameter_b)/l.density_parameter_c)*
      l.density_parameter_c*std::expm1((hi-lo)/l.density_parameter_c);
  }
  double column=columnAbove(e,0.);
  if(!finite(column)||column<=0.)throw std::invalid_argument("Invalid atmosphere column");
  e.geometry.horizontal_slope_cm_per_g_cm2=(6.e5-2.e6)/column;
  e.field={magnitude*2.99792458,magnitude>0.?vertical/magnitude:0.,magnitude>0.?norm/magnitude:1.,
    snapshot.maximum_magnetic_deflection_rad};
  if(!validField(e.field))throw std::invalid_argument("Invalid C7 magnetic step constraint");
  return e;
}
KOKKOS_INLINE_FUNCTION int airRegion(AirEnvironment const& e,double local_z_cm) {
  double height=-local_z_cm;
  if(height>e.geometry.bound_cm[0])return 1;
  for(int i=1;i<5;++i)if(height>=e.geometry.bound_cm[i])return i+1;
  return 6;
}
// Minimal proper rotation from the observer downward axis to the local radial
// downward axis. This is an adapter for C8's Cartesian vectors, not a claim that
// C7's approximate, azimuth-preserving curved transport is an exact rotation.
KOKKOS_INLINE_FUNCTION Frame radialFrame(Point observer,double earth_radius_cm) {
  double z=earth_radius_cm-observer.z;
  double radius=::sqrt(observer.x*observer.x+observer.y*observer.y+z*z);
  Direction n{-observer.x/radius,-observer.y/radius,z/radius};double d=1.+n.z;
  Frame f;f.x={1.-n.x*n.x/d,-n.x*n.y/d,-n.x};f.y={-n.x*n.y/d,1.-n.y*n.y/d,-n.y};f.z=n;return f;
}
struct AirRecord {
  C8Particle metadata;
  CurvedTrack track;
  CounterStream random;
  CounterStream hadron_random;
  int region{};
  bool exported_position{}; // false on import: do not assume Cartesian round-trip is exact
};
KOKKOS_INLINE_FUNCTION bool importAirRecord(C8Particle p,AirEnvironment const& e,
    std::uint64_t seed,std::uint64_t shower,AirRecord& result) {
  TrackState observer;
  if(p.medium_id!=e.medium_id||!importParticle(p,e.observer_frame,observer))return false;
  Point pos=observer.position_cm;double z=e.curved.earth_radius_cm-pos.z;
  double lateral=::sqrt(pos.x*pos.x+pos.y*pos.y),radius=::sqrt(z*z+lateral*lateral);
  if(z<=0.||!finite(radius)||radius<=0.)return false;
  auto tangent=radialFrame(pos,e.curved.earth_radius_cm);if(!validFrame(tangent))return false;
  double arc=lateral>0.?e.curved.earth_radius_cm*::atan2(lateral,z)/lateral:1.;
  AirRecord r;r.metadata=p;r.track.particle=observer;r.track.observer_cm=pos;
  r.track.particle.position_cm={pos.x*arc,pos.y*arc,e.curved.earth_radius_cm-radius};
  r.track.particle.direction=toLocal(tangent,observer.direction);r.track.wa=z/radius;
  double dh=-pos.z-e.curved.observation_height_cm,d=::sqrt(lateral*lateral+dh*dh);
  r.track.wap=d>0.?dh/d:0.;r.region=airRegion(e,r.track.particle.position_cm.z);
  r.random.key={seed,shower,p.history_id,p.step_id,NativeEgs4RandomDomain,0};
  r.hadron_random.key={seed,shower,p.history_id,p.step_id,NativeEgs4HadronRandomDomain,0};
  if(!validTrack(r.track.particle))return false;result=r;return true;
}
KOKKOS_INLINE_FUNCTION bool exportAirParticle(CurvedTrack const& t,AirEnvironment const& e,C8Particle& p) {
  auto observer=t.particle;observer.position_cm=t.observer_cm;
  if(!t.detector_local) {
    auto tangent=radialFrame(t.observer_cm,e.curved.earth_radius_cm);
    if(!validFrame(tangent))return false;observer.direction=toGlobal(tangent,t.particle.direction);
  }
  return exportParticle(observer,e.observer_frame,p);
}
enum class AirAction { error,active,replaced,observed,discarded,needs_host_channel };
enum class AirDiscard { none,outside_atmosphere,angular,early_observation };
struct AirOutcome {
  AirAction action{AirAction::error};
  AirDiscard discard{AirDiscard::none};
  HistoryOutcome history;
  CurvedTrack children[2];
  double discarded_energy_MeV{};
};
struct GuardedFrame {
  CurvedFrameStep frame;GeometryConfig geometry;bool* outside;
  KOKKOS_INLINE_FUNCTION FrameStepResult operator()(TrackState previous,Point next,Direction direction,
      double step,double path,double mass,double speed)const {
    // Literal C7 UPWARD guard precedes displacement/normal time/radio output.
    if(-next.z>=geometry.bound_cm[0]-1.||-next.z<=geometry.bound_cm[5]) {
      *outside=true;return {};
    }
    return frame(previous,next,direction,step,path,mass,speed);
  }
};
KOKKOS_INLINE_FUNCTION void discardAir(AirOutcome& out,TrackState p,double mass,AirDiscard why) {
  out.action=AirAction::discarded;out.discard=why;out.discarded_energy_MeV=terminationEnergy(p,mass);
  out.history.action=HistoryAction::replaced;out.history.error=HistoryError::none;
  out.history.particle=p;out.history.child_count=0;
}
// Actual C8-carrier history step in a configured spherical atmosphere. Kept
// separate from the PROPOSAL router so no table flag or fallback is falsified.
KOKKOS_INLINE_FUNCTION AirOutcome advanceAirRecord(ModelView model,AirRecord& record,
    AirEnvironment const& e,TransportSettings settings) {
  settings.paired_scatter_trig=(e.overhead_mask&overhead::scatter_pair)!=0;
  PreparedCounterStream step_random(record.random,(e.overhead_mask&overhead::prepared_rng)!=0);
  AirOutcome out;auto previous=record.track;out.history.particle=previous.particle;
  if(record.region<1||record.region>6||record.metadata.step_id==~std::uint64_t{0})return out;
  if(record.region==1||record.region==6) {
    discardAir(out,previous.particle,model.thresholds.mass_MeV,AirDiscard::outside_atmosphere);return out;
  }
  auto region=e.regions[record.region-2];
  auto plan=planHistory(model,previous.particle,region,e.field,settings,step_random,(e.overhead_mask&overhead::reuse_query)!=0);
  if(plan.action==PlanAction::error){out.history.error=plan.error;return out;}
  GeometryResult geometry;CurvedHistoryOutcome result;int next_region=record.region;
  if(plan.action!=PlanAction::transport) {
    result=finishCurvedHistory(model,plan,previous,region,e.field,settings,{},e.curved,step_random);
  } else {
    geometry=howFar(e.geometry,{previous.particle.position_cm,previous.particle.direction,record.region,previous.wa,
      columnAbove(e,-previous.particle.position_cm.z),plan.proposed_cm,0.});
    if(!geometry.valid||geometry.step_cm<=0.)return out;
    if(geometry.discard>0) {
      discardAir(out,previous.particle,model.thresholds.mass_MeV,AirDiscard::outside_atmosphere);return out;
    }
    if(geometry.discard==-1) {
      auto detector=finishDetectorHistory(model,plan,previous,region,e.field,settings,e.curved,
        e.geometry.maximum_horizontal_cm,geometry.step_cm,step_random);
      if(detector.action==DetectorAction::error)return out;
      result=detector.advanced;
      if(detector.action==DetectorAction::angular_discard) {
        out.action=AirAction::discarded;out.discard=AirDiscard::angular;
        out.discarded_energy_MeV=detector.discarded_energy_MeV;
      } else if(detector.action==DetectorAction::observed)out.action=AirAction::observed;
    } else {
      bool outside=false;result.particle=previous;
      GuardedFrame frame{{e.curved,previous,&result.frame},e.geometry,&outside};
      plan.proposed_cm=geometry.step_cm;
      if(plan.particle.pdg==22)plan.photon.proposed_cm=geometry.step_cm;
      else plan.electron.proposed_cm=geometry.step_cm;
      // Defer the vertex until after final layer and angular checks, as in C7.
      // A synthetic boundary disposition changes no path arithmetic or RNG.
      result.history=finishHistory(model,plan,region,e.field,settings,
        {geometry.step_cm,BoundaryKind::density_layer},step_random,frame);
      if(outside||result.frame.status==CurvedStatus::angle_cut||result.frame.status==CurvedStatus::observation) {
        discardAir(out,previous.particle,model.thresholds.mass_MeV,outside?AirDiscard::outside_atmosphere:
          result.frame.status==CurvedStatus::angle_cut?AirDiscard::angular:AirDiscard::early_observation);
        return out; // no normally timed segment was emitted
      }
      if(result.history.action==HistoryAction::error){out.history=result.history;return out;}
      result.frame_applied=result.history.has_track;result.particle.particle=result.history.particle;
      result.particle.observer_cm=result.frame.observer;result.particle.wa=result.frame.wa;result.particle.wap=result.frame.wap;
      if(result.history.action!=HistoryAction::replaced) {
        next_region=geometry.region;
        double height=-result.history.particle.position_cm.z;
        if(next_region>=2&&next_region<=5&&geometry.discard>=0) {
          if(height<e.geometry.bound_cm[next_region-1])++next_region;
          else if(height>e.geometry.bound_cm[next_region-2]||
              (height==e.geometry.bound_cm[next_region-2]&&result.history.particle.direction.z<=.003))--next_region;
        }
        if(result.history.particle.direction.z<e.curved.minimum_cosine) {
          out.action=AirAction::discarded;out.discard=AirDiscard::angular;
          out.discarded_energy_MeV=terminationEnergy(result.history.particle,model.thresholds.mass_MeV);
        } else if(next_region<=1||next_region>=6||geometry.discard==-2) {
          out.action=AirAction::discarded;out.discard=AirDiscard::outside_atmosphere;
          out.discarded_energy_MeV=terminationEnergy(result.history.particle,model.thresholds.mass_MeV);
        } else {
          auto& h=result.history;h.boundary=next_region!=record.region?BoundaryKind::density_layer:BoundaryKind::none;
          bool due=h.particle.pdg==22?h.particle.photon_clock.remaining_mfp<=1.e-10:h.particle.clock.remaining_mfp<1.e-10;
          if(due)finishCollision(model,e.regions[next_region-2],h,step_random);
          else h.action=HistoryAction::active;
          result.particle.particle=h.particle;
        }
      }
    }
  }
  out.history=result.history;
  if(out.history.action==HistoryAction::error)return out;
  if(out.action==AirAction::error) {
    auto a=out.history.action;
    out.action=a==HistoryAction::replaced?AirAction::replaced:a==HistoryAction::needs_host_channel?AirAction::needs_host_channel:
      (a==HistoryAction::active||a==HistoryAction::boundary)?AirAction::active:AirAction::error;
  }
  if(out.action==AirAction::error)return out;
  for(int j=0;j<out.history.child_count;++j) {
    out.children[j]=result.particle;out.children[j].particle=out.history.children[j];
  }
  auto metadata=record.metadata;
  if(!exportAirParticle(result.particle,e,metadata)){out.action=AirAction::error;return out;}
  if(out.history.has_track)++metadata.step_id;
  record.metadata=metadata;record.exported_position=true;record.track=result.particle;record.region=next_region;return out;
}
} // namespace c7_egs4::c8_adapter

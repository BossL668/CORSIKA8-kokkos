#pragma once
#include "Egs4Model.hpp"

namespace c7_egs4 {
// Native C7 EGS coordinates and units: x north, y east, z down, cm/MeV/s.
// The C8 adapter must transform the frame, rather than relabel its axes.
struct Point {double x{},y{},z{};};
struct TrackState {
  int pdg{};
  double energy_MeV{},time_s{};
  Point position_cm;
  Direction direction;
  ElectronClock clock;
  PhotonClock photon_clock;
  Channel pending_channel{Channel::invalid}; // host request must be resolved before further transport
};
struct AirRegion {
  double reference_density_g_cm3{}; // RHOR, NOT density at the current height
  double inverse_scale_height_cm{}; // HBAROI; rho(z)=RHOR*exp(z*HBAROI)
  double sterncor{};
};
struct MagneticField {
  double bnorm_MeV_cm{},sinB{},cosB{1.},maximum_angle_rad{.2};
};
struct TransportSettings {
  double electron_cut_total_MeV{},photon_cut_MeV{};
  double stepfc{1.},vacuum_distance_cm{1.e9};
  double speed_cm_s{2.99792458e10};
  bool paired_scatter_trig{};
};
// HATCH's threshold floor is part of the backend, not an optional C8 cut.
inline TransportSettings transportSettings(ModelView model,double electron_kinetic_cut,
                                           double photon_cut,double stepfc=1.) {
  if(!finite(electron_kinetic_cut)||electron_kinetic_cut<0.||!finite(photon_cut)||photon_cut<0.||
     !finite(stepfc)||stepfc<=0.)throw std::runtime_error("Invalid EGS4 transport cut or STEPFC");
  TransportSettings s;
  s.electron_cut_total_MeV=larger(larger(electron_kinetic_cut+model.thresholds.mass_MeV,
                                        model.tables.medium.ae),model.tables.medium.ap+model.thresholds.mass_MeV);
  s.photon_cut_MeV=larger(photon_cut,model.tables.medium.ap);s.stepfc=stepfc;return s;
}
KOKKOS_INLINE_FUNCTION bool validTrack(TrackState s) {
  return (s.pdg==11||s.pdg==-11||s.pdg==22)&&finite(s.energy_MeV)&&s.energy_MeV>0.&&
    finite(s.time_s)&&finite(s.position_cm.x)&&finite(s.position_cm.y)&&finite(s.position_cm.z)&&validDirection(s.direction);
}
KOKKOS_INLINE_FUNCTION bool validRegion(AirRegion r) {
  return finite(r.reference_density_g_cm3)&&r.reference_density_g_cm3>0.&&
    finite(r.inverse_scale_height_cm)&&r.inverse_scale_height_cm>=0.&&finite(r.sterncor);
}
KOKKOS_INLINE_FUNCTION bool validField(MagneticField b) {
  return finite(b.bnorm_MeV_cm)&&b.bnorm_MeV_cm>=0.&&finite(b.sinB)&&finite(b.cosB)&&
    ::fabs(b.sinB*b.sinB+b.cosB*b.cosB-1.)<1.e-12&&finite(b.maximum_angle_rad)&&
    b.maximum_angle_rad>0.&&b.maximum_angle_rad<=.2;
}
struct ElectronPlan {
  Status status{Status::invalid_input};
  LocalStep local;
  double scattering_length_cm{},density_inverse{},true_physical_cm{};
  double unclipped_cm{},proposed_cm{};
};
// Two-phase interface: propose, ask geometry for a cap, then finish. No RNG
// is consumed by the proposal. A history's initial SIG0 stays in its clock.
KOKKOS_INLINE_FUNCTION ElectronPlan prepareElectronSegment(ModelView model,TrackState state,
    AirRegion region,MagneticField field,TransportSettings settings,ElectronQuery const* supplied=nullptr) {
  ElectronPlan p;
  double m=model.thresholds.mass_MeV;
  if(!validTrack(state)||state.pdg==22||!validRegion(region)||!validField(field)||
     !state.clock.initialized||settings.electron_cut_total_MeV<model.tables.medium.ae||
     settings.electron_cut_total_MeV<model.tables.medium.ap+m)return p;
  auto q=supplied?*supplied:electronQuery(model.tables,state.energy_MeV,m,state.pdg==11?-1:1);
  q=withSampledRate(q,state.clock);
  p.local=proposeLocalStep(model.tables,q,state.energy_MeV,m,settings.electron_cut_total_MeV,
    region.reference_density_g_cm3,state.position_cm.z,region.sterncor,settings.stepfc,
    state.clock.remaining_mfp,settings.vacuum_distance_cm);
  if(p.local.status!=Status::success){p.status=p.local.status;return p;}
  double beta2=larger(1.e-8,1.-m*m/(state.energy_MeV*state.energy_MeV));
  double beta3=state.energy_MeV*beta2*.094315;
  double rhofi=1./(region.reference_density_g_cm3/model.tables.medium.rho);
  p.scattering_length_cm=model.tables.medium.radiation_length_cm*beta3*beta3*rhofi;
  double ratio=p.local.true_step_cm/p.scattering_length_cm;
  p.density_inverse=::exp(-state.position_cm.z*region.inverse_scale_height_cm);
  double distance=p.local.projected_step_cm*p.density_inverse;
  double disc=state.direction.z*distance*region.inverse_scale_height_cm;
  if(::fabs(disc)<.0000007)
    distance*=1.-.5*disc*(1.-.666666666666667*disc*(1.-.75*disc*(1.-.8*disc)));
  else if(disc>-1.)distance=distance*::log(disc+1.)/disc;
  else distance=settings.vacuum_distance_cm;
  p.unclipped_cm=distance;p.true_physical_cm=distance/(1.-ratio);
  p.proposed_cm=field.bnorm_MeV_cm>0.?smaller(distance,(field.maximum_angle_rad/field.bnorm_MeV_cm)*state.energy_MeV):distance;
  if(!finite(p.density_inverse)||p.density_inverse<=0.||!finite(p.proposed_cm)||
     p.proposed_cm<=0.||!finite(p.true_physical_cm)||!finite(p.scattering_length_cm))return p;
  p.status=Status::success;return p;
}
KOKKOS_INLINE_FUNCTION double pathCorrection(double vstp) {
  int i=int(2.+18.182*vstp);if(i<1)i=1;if(i>6)i=6;
  double c0,c1,c2;
  if(i<=2){c0=1.;c1=.98875;c2=2.5026;}
  else if(i==3){c0=1.006;c1=.78657;c2=4.2387;}
  else if(i==4){c0=1.0657;c1=-.25051;c2=8.7681;}
  else {c0=1.6971;c1=-7.56;c2=29.946;}
  return c0+vstp*(c1+vstp*c2);
}
enum class BoundaryKind { none, density_layer, material_change, observation, escape };
struct BoundaryLimit {double distance_cm{};BoundaryKind kind{BoundaryKind::none};};
enum class TransportStatus { invalid, advanced, collision_due, energy_cut, boundary, random_failure, sampling_failure };
struct ElectronSegment {
  TransportStatus status{TransportStatus::invalid};
  TrackState particle;
  BoundaryKind boundary{BoundaryKind::none};
  double geometric_step_cm{},loss_path_cm{},time_path_cm{},deposit_MeV{},scattering_angle{},bend_angle{};
  bool path_corrected{};
  Direction transport_end_direction; // before MSCAT rotation; scatter is applied after displacement
};
struct FrameStepResult {
  bool valid{};
  Point position_cm;
  Direction direction;
  double time_increment_s{};
};
// The frame policy runs after magnetic displacement and BEFORE the MSCAT
// azimuth/rotation. In C7 CURVED these operations do not commute. The default
// policy preserves the previously tested flat arithmetic and random sequence.
struct FlatFrameStep {
  KOKKOS_INLINE_FUNCTION FrameStepResult operator()(TrackState previous,Point advanced,
      Direction direction,double /*geometric_cm*/,double time_path_cm,
      double mass_MeV,double speed_cm_s)const {
    double time=time_path_cm*(1./speed_cm_s);
    if(previous.pdg!=22)time/=::sqrt((1.-mass_MeV/previous.energy_MeV)*(1.+mass_MeV/previous.energy_MeV));
    return {finite(time)&&time>=0.,advanced,direction,time};
  }
};
// A non-default policy must explicitly reject unimplemented endpoint cases;
// they are NOT emitted as successfully timed normal radio tracks.
// Errors leave the returned particle unchanged. RNG exhaustion is fatal;
// callers must not substitute a PROPOSAL transport step.
template<class Random,class FrameStep=FlatFrameStep>
KOKKOS_INLINE_FUNCTION ElectronSegment finishElectronSegment(ModelView model,TrackState state,
    AirRegion region,MagneticField field,TransportSettings settings,ElectronPlan p,
    BoundaryLimit limit,Random& rng,FrameStep frame={}) {
  ElectronSegment r;r.particle=state;
  if(p.status!=Status::success||!validTrack(state)||state.pdg==22||!validRegion(region)||!validField(field)||
     !finite(settings.speed_cm_s)||settings.speed_cm_s<=0.||!finite(limit.distance_cm)||limit.distance_cm<=0.)return r;
  double step=smaller(p.proposed_cm,limit.distance_cm),tvstep,tvstpc;
  if(limit.distance_cm<=p.proposed_cm)r.boundary=limit.kind;
  if(step==p.unclipped_cm){tvstep=p.local.true_step_cm;tvstpc=p.true_physical_cm;}
  else {
    r.path_corrected=true;
    double equivalent=step,disc=state.direction.z*equivalent*region.inverse_scale_height_cm;
    if(disc!=0.)equivalent=equivalent*(::exp(disc)-1.)/(disc*p.density_inverse);
    else equivalent/=p.density_inverse;
    equivalent=larger(equivalent,.0001);
    double factor=pathCorrection(equivalent/p.scattering_length_cm);
    tvstep=factor*equivalent;tvstpc=factor*step;
  }
  int charge=state.pdg==11?-1:1;
  double alpha=step*charge*field.bnorm_MeV_cm/state.energy_MeV;
  tvstpc=tvstpc*(1.+.04166667*alpha*alpha);
  double deposit=p.local.loss_MeV_per_cm*tvstep,m=model.thresholds.mass_MeV;
  double new_energy=state.energy_MeV-deposit;
  if(!finite(tvstep)||tvstep<0.||!finite(tvstpc)||tvstpc<0.||!finite(new_energy)||new_energy<m)return r;
  double rhofac=region.reference_density_g_cm3/model.tables.medium.rho;
  auto scatter=sampleScattering(model.tables.medium,state.energy_MeV,m,tvstep*rhofac,rng,1024,settings.paired_scatter_trig);
  if(scatter.status!=InteractionStatus::success) {
    r.status=scatter.status==InteractionStatus::random_failure?TransportStatus::random_failure:TransportStatus::sampling_failure;
    return r;
  }
  double u0=state.direction.x,v0=state.direction.y,w0=state.direction.z;
  double f=1.-.5*alpha*alpha*(1.-.75*alpha*alpha);
  double fs=(1.-f)*field.sinB,fc=(1.-f)*field.cosB,v1=v0*alpha*f,usw=u0*field.sinB-w0*field.cosB;
  Direction bent{u0-fs*usw+v1*field.sinB,f*(v0-alpha*usw),w0+fc*usw-v1*field.cosB};
  double inv=1.5-.5*(bent.x*bent.x+bent.y*bent.y+bent.z*bent.z);
  bent.x*=inv;bent.y*=inv;bent.z*=inv;
  auto next=state;
  next.position_cm.x+=step*.5*(u0+bent.x);next.position_cm.y+=step*.5*(v0+bent.y);next.position_cm.z+=step*.5*(w0+bent.z);
  auto transformed=frame(state,next.position_cm,bent,step,tvstpc,m,settings.speed_cm_s);
  if(!transformed.valid)return r;
  next.position_cm=transformed.position_cm;bent=transformed.direction;
  next.time_s+=transformed.time_increment_s;
  double sp,cp;
  // ELECTR always consumes this azimuth, including the NOSCAT branch.
  if(!azimuth(rng,sp,cp)){r.status=TransportStatus::random_failure;return r;}
  next.direction=rotate(bent,scatter.sine,scatter.cosine,sp,cp);next.energy_MeV=new_energy;
  if(!validTrack(next))return r;
  if(new_energy<=settings.electron_cut_total_MeV)r.status=TransportStatus::energy_cut;
  else if(r.boundary==BoundaryKind::escape)r.status=TransportStatus::boundary;
  else if(r.boundary==BoundaryKind::material_change){next.clock={};r.status=TransportStatus::boundary;}
  else {
    auto cs=consumeElectronClock(next.clock,tvstep,rhofac);
    if(cs==ClockStatus::invalid)return r;
    r.status=cs==ClockStatus::collision_due?TransportStatus::collision_due:TransportStatus::advanced;
    if(r.boundary!=BoundaryKind::none)r.status=TransportStatus::boundary;
  }
  r.particle=next;r.geometric_step_cm=step;r.loss_path_cm=tvstep;r.time_path_cm=tvstpc;
  r.deposit_MeV=deposit;r.scattering_angle=scatter.angle;r.bend_angle=alpha;
  r.transport_end_direction=bent;return r;
}
} // namespace c7_egs4

#pragma once
#include <corsika/detail/modules/egs4/Egs4CurvedTransport.hpp>

namespace c7_egs4 {
enum class ObservationPreparationStatus { invalid, transport, angular_discard };
struct ObservationPreparation {
  ObservationPreparationStatus status{ObservationPreparationStatus::invalid};
  CurvedTrack particle;
  double step_cm{};
  bool transport_ready{};
};
// C7 ELECTR/PHOTON CURVED pre-transport IDISC=-1 block. This is not a
// post-step clipping operation. In particular ALTEXP and the sampled loss
// coefficients remain those of the ORIGINAL proposal. Save the OLD observer
// endpoint separately for the subsequent observer-speed correction.
KOKKOS_INLINE_FUNCTION ObservationPreparation prepareObservationStep(
    CurvedTrack previous,CurvedConfig c,double maximum_horizontal_cm,double original_step_cm=0.) {
  ObservationPreparation r;r.particle=previous;r.step_cm=original_step_cm;
  if(!validTrack(previous.particle)||!finite(c.earth_radius_cm)||c.earth_radius_cm<=0.||
     !finite(c.observation_height_cm)||!finite(c.minimum_cosine)||c.minimum_cosine< -1.||c.minimum_cosine>1.||
     !finite(maximum_horizontal_cm)||maximum_horizontal_cm<=0.||!finite(original_step_cm)||original_step_cm<0.)return r;
  auto& p=r.particle.particle;
  double lateral=::sqrt(p.position_cm.x*p.position_cm.x+p.position_cm.y*p.position_cm.y);
  r.particle.wa=smaller(1.,::cos(lateral/c.earth_radius_cm));
  double phi=(p.direction.x!=0.||p.direction.y!=0.)?::atan2(p.direction.y,p.direction.x):0.;
  double effective=-(::cos(phi)*p.position_cm.x+::sin(phi)*p.position_cm.y);
  if(effective<maximum_horizontal_cm) {
    double sd=::sin(effective/c.earth_radius_cm),cd=::sqrt((1.-sd)*(1.+sd));
    double w=p.direction.z*cd;
    if(::fabs(p.direction.z)<1.)w-=::sqrt((1.-p.direction.z)*(1.+p.direction.z))*sd;
    p.direction.z=larger(-1.,smaller(1.,w));
  }
  if(p.direction.z<c.minimum_cosine){r.status=ObservationPreparationStatus::angular_discard;return r;}
  double apparent_z=c.earth_radius_cm-(c.earth_radius_cm-p.position_cm.z)*r.particle.wa;
  // Expose C7 ZAP even for an early exit; observer x/y must not be advanced.
  r.particle.observer_cm.z=apparent_z;
  if(c.flat_output) {
    if(r.particle.wa!=1.) {
      if(r.particle.wa<=0.)return r;
      double phi1=(p.position_cm.x!=0.||p.position_cm.y!=0.)?::atan2(p.position_cm.y,p.position_cm.x):0.;
      double st=::sqrt((1.-r.particle.wa)*(1.+r.particle.wa));
      double radius=(-apparent_z+c.earth_radius_cm)*st/r.particle.wa;
      p.position_cm={radius*::cos(phi1),radius*::sin(phi1),apparent_z};
    }
    r.particle.wap=0.;
  }
  if(p.direction.x!=0.) {
    double t=p.direction.y/p.direction.x;
    p.direction.x=::fabs(p.direction.z)<1.?::copysign(1.,p.direction.x)*
      ::sqrt((1.-p.direction.z)*(1.+p.direction.z)/(1.+t*t)):0.;
    p.direction.y=t*p.direction.x;
  } else if(p.direction.y!=0.&&::fabs(p.direction.z)<1.) {
    p.direction.y=::copysign(1.,p.direction.y)*::sqrt((1.-p.direction.z)*(1.+p.direction.z));
  } else p.direction.y=0.;
  if(p.direction.z<=0.){r.status=ObservationPreparationStatus::angular_discard;return r;}
  r.step_cm=larger(-(p.position_cm.z+c.observation_height_cm)/p.direction.z,.0001);
  r.particle.detector_local=true;
  if(!finite(r.step_cm))return r;
  // Literal C7 updates W but leaves U=V=0 for an exactly vertical input away
  // from the origin. Do not silently normalize or invent a transverse sign.
  // Expose its arithmetic for the oracle, but reject such an unsupported
  // direction at the transport-controller boundary.
  r.transport_ready=validTrack(p);
  r.status=ObservationPreparationStatus::transport;return r;
}

enum class DetectorAction { error, observed, stopped, angular_discard };
struct DetectorHistoryOutcome {
  DetectorAction action{DetectorAction::error};
  ObservationPreparation preparation;
  CurvedHistoryOutcome advanced;
  Point observer_start_cm;
  double discarded_energy_MeV{}; // C7 angular-loss ledger, NOT continuous deposition
};
KOKKOS_INLINE_FUNCTION double terminationEnergy(TrackState p,double mass_MeV) {
  return p.energy_MeV+(p.pdg==11?-mass_MeV:p.pdg==-11?mass_MeV:0.);
}
// Finish an already planned normal-air step for which HOWFAR returned IDISC=-1.
// Keep the original density/loss/scattering coefficients: replanning after the
// detector-frame conversion changes C7's algorithm. Original C7 performs this
// correction after its geometry cap, so do not cap it a second time by the old
// proposal. This entry does not cover atmosphere escape or EM-primary FIXHEI.
template<class Random>
KOKKOS_INLINE_FUNCTION DetectorHistoryOutcome finishDetectorHistory(ModelView model,
    HistoryPlan plan,CurvedTrack previous,AirRegion air,MagneticField field,
    TransportSettings settings,CurvedConfig config,double maximum_horizontal_cm,
    double howfar_step_cm,Random& rng) {
  DetectorHistoryOutcome r;r.observer_start_cm=previous.observer_cm;
  r.advanced.particle=previous;
  if(plan.action!=PlanAction::transport||!validSettings(model,settings)||
     !validRegion(air)||!validField(field)||!validPoint(previous.observer_cm))return r;
  previous.particle=plan.particle; // include the clock sampled by the proposal
  r.preparation=prepareObservationStep(previous,config,maximum_horizontal_cm,howfar_step_cm);
  if(r.preparation.status==ObservationPreparationStatus::angular_discard) {
    r.action=DetectorAction::angular_discard;
    r.advanced.particle=r.preparation.particle;
    auto& h=r.advanced.history;h.particle=r.preparation.particle.particle;
    h.action=HistoryAction::replaced;h.error=HistoryError::none;
    r.discarded_energy_MeV=terminationEnergy(h.particle,model.thresholds.mass_MeV);
    return r; // label 420/1000: NO stopped-positron photons and NO track
  }
  if(r.preparation.status!=ObservationPreparationStatus::transport||!r.preparation.transport_ready)return r;
  auto start=r.preparation.particle;
  start.observer_cm=r.observer_start_cm; // XXXOLD/YYYOLD/ZAPOLD, before preprocessing
  plan.particle=start.particle;plan.proposed_cm=r.preparation.step_cm;
  if(start.particle.pdg==22)plan.photon.proposed_cm=plan.proposed_cm;
  else plan.electron.proposed_cm=plan.proposed_cm;
  r.advanced=finishCurvedHistory(model,plan,start,air,field,settings,
    {plan.proposed_cm,BoundaryKind::observation},config,rng);
  auto& h=r.advanced.history;
  if(h.action==HistoryAction::error)return r;
  // ELECTR checks the energy cutoff before the post-MSCAT angular cutoff and
  // the observation writer. Its stopped-positron photons therefore survive.
  if(h.action==HistoryAction::replaced){r.action=DetectorAction::stopped;return r;}
  if(h.action!=HistoryAction::boundary||h.boundary!=BoundaryKind::observation)return r;
  if(h.particle.direction.z<config.minimum_cosine) {
    r.action=DetectorAction::angular_discard;
    r.discarded_energy_MeV=terminationEnergy(h.particle,model.thresholds.mass_MeV);
    h.action=HistoryAction::replaced;return r; // keep the normally emitted track
  }
  r.action=DetectorAction::observed;return r;
}
} // namespace c7_egs4

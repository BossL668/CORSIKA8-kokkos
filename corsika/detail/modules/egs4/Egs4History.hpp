#pragma once
#include <corsika/detail/modules/egs4/Egs4PhotonTransport.hpp>

namespace c7_egs4 {
// Native, per-particle controller. The caller asks geometry for a BoundaryLimit
// between planning and finishing, then handles observations and returned children.
// Frame handling is a policy; the default is local flat. This is not yet the
// production C8 atmosphere/stack adapter.
enum class PlanAction { error, transport, collision, energy_cut };
enum class HistoryAction { error, active, boundary, escaped, replaced, needs_host_channel };
enum class HistoryError { none, invalid_input, table_range, random_failure, transport_failure, sampling_failure };
struct HistoryPlan {
  PlanAction action{PlanAction::error};
  HistoryError error{HistoryError::invalid_input};
  TrackState particle;
  ElectronPlan electron;
  PhotonPlan photon;
  double proposed_cm{};
};
struct HistoryOutcome {
  HistoryAction action{HistoryAction::error};
  HistoryError error{HistoryError::invalid_input};
  TrackState particle;
  TrackState children[2]{};
  int child_count{};
  Channel channel{Channel::invalid};
  BoundaryKind boundary{BoundaryKind::none};
  bool has_track{};
  Direction transport_end_direction;
  double geometric_step_cm{},loss_path_cm{},time_path_cm{};
  double continuous_deposit_MeV{},vertex_deposit_MeV{};
  DepositKind vertex_deposit_kind{DepositKind::none};
};
KOKKOS_INLINE_FUNCTION bool validSettings(ModelView model,TransportSettings s) {
  return finite(s.electron_cut_total_MeV)&&s.electron_cut_total_MeV>=model.tables.medium.ae&&
    s.electron_cut_total_MeV>=model.tables.medium.ap+model.thresholds.mass_MeV&&
    finite(s.photon_cut_MeV)&&s.photon_cut_MeV>=model.tables.medium.ap&&
    finite(s.stepfc)&&s.stepfc>0.&&finite(s.vacuum_distance_cm)&&s.vacuum_distance_cm>0.&&
    finite(s.speed_cm_s)&&s.speed_cm_s>0.;
}
template<class Random>
KOKKOS_INLINE_FUNCTION HistoryPlan planHistory(ModelView model,TrackState state,AirRegion region,
    MagneticField field,TransportSettings settings,Random& rng,bool reuse_query=false) {
  HistoryPlan p;p.particle=state;
  if(!validTrack(state)||!validRegion(region)||!validSettings(model,settings)||
     state.pending_channel!=Channel::invalid||(state.pdg!=22&&(!validField(field)||state.energy_MeV<model.thresholds.mass_MeV)))return p;
  p.error=HistoryError::none;
  double cut=state.pdg==22?settings.photon_cut_MeV:settings.electron_cut_total_MeV;
  if(state.energy_MeV<=cut){p.action=PlanAction::energy_cut;return p;}
  if(state.pdg==22) {
    auto q=photonQuery(model.tables,state.energy_MeV);
    if(q.status!=Status::success){p.error=HistoryError::table_range;return p;}
    if(state.photon_clock.initialized) {
      if(!finite(state.photon_clock.remaining_mfp)||state.photon_clock.remaining_mfp<0.){p.error=HistoryError::invalid_input;return p;}
      if(state.photon_clock.remaining_mfp<=1.e-10){p.action=PlanAction::collision;return p;}
    } else if(startPhotonClock(q,rng,state.photon_clock)!=ClockStatus::transporting) {
      p.error=HistoryError::random_failure;return p;
    }
    // A freshly sampled very small path still follows PHOTON's minimum step.
    p.photon=preparePhotonSegment(model,state,region,settings,reuse_query?&q:nullptr);
    if(p.photon.status!=Status::success){p.error=HistoryError::transport_failure;return p;}
    p.proposed_cm=p.photon.proposed_cm;
  } else {
    auto q=electronQuery(model.tables,state.energy_MeV,model.thresholds.mass_MeV,state.pdg==11?-1:1);
    if(q.status!=Status::success){p.error=HistoryError::table_range;return p;}
    if(state.clock.initialized) {
      if(!finite(state.clock.remaining_mfp)||state.clock.remaining_mfp<0.||!finite(state.clock.sampled_rate_per_cm)) {
        p.error=HistoryError::invalid_input;return p;
      }
      if(state.clock.remaining_mfp<1.e-10){p.action=PlanAction::collision;return p;}
    } else if(startElectronClock(q,rng,state.clock)!=ClockStatus::transporting) {
      p.error=HistoryError::random_failure;return p;
    }
    p.electron=prepareElectronSegment(model,state,region,field,settings,reuse_query?&q:nullptr);
    if(p.electron.status!=Status::success){p.error=HistoryError::transport_failure;return p;}
    p.proposed_cm=p.electron.proposed_cm;
  }
  p.particle=state;p.action=PlanAction::transport;return p;
}
KOKKOS_INLINE_FUNCTION void applyLocalOutcome(HistoryOutcome& r,LocalOutcome local) {
  if(local.secondaries.status!=InteractionStatus::success||local.secondaries.count<0||local.secondaries.count>2) {
    r.action=HistoryAction::error;
    r.error=local.secondaries.status==InteractionStatus::random_failure?HistoryError::random_failure:HistoryError::sampling_failure;
    return;
  }
  r.vertex_deposit_MeV=local.deposit_MeV;r.vertex_deposit_kind=local.deposit_kind;
  r.child_count=local.secondaries.count;
  for(int i=0;i<r.child_count;++i) {
    auto child=local.secondaries.particle[i];TrackState s;
    s.pdg=child.pdg;s.energy_MeV=child.energy_MeV;s.position_cm=r.particle.position_cm;
    s.time_s=r.particle.time_s;s.direction=child.direction;
    r.children[i]=s; // new histories, no inherited clocks or pending channel
  }
  r.action=HistoryAction::replaced;r.error=HistoryError::none;
}
template<class Random>
KOKKOS_INLINE_FUNCTION void finishEnergyCut(ModelView model,HistoryOutcome& r,Random& rng) {
  if(r.particle.pdg==22) {
    LocalOutcome local;local.secondaries.status=InteractionStatus::success;
    local.deposit_MeV=r.particle.energy_MeV;local.deposit_kind=DepositKind::subcut_kinetic;
    applyLocalOutcome(r,local);
  } else applyLocalOutcome(r,stopLepton(r.particle.energy_MeV,model.thresholds.mass_MeV,r.particle.pdg,rng));
}
template<class Random>
KOKKOS_INLINE_FUNCTION void finishCollision(ModelView model,AirRegion region,HistoryOutcome& r,Random& rng) {
  if(r.particle.pdg!=22) {
    auto q=electronQuery(model.tables,r.particle.energy_MeV,model.thresholds.mass_MeV,r.particle.pdg==11?-1:1);
    auto accepted=acceptElectronCollision(r.particle.clock,q,rng);
    if(accepted==ClockStatus::restart){r.action=HistoryAction::active;r.error=HistoryError::none;return;}
    if(accepted!=ClockStatus::accepted) {
      r.action=HistoryAction::error;r.error=accepted==ClockStatus::random_failure?HistoryError::random_failure:HistoryError::invalid_input;return;
    }
  } else r.particle.photon_clock={};
  double rho=region.reference_density_g_cm3*::exp(r.particle.position_cm.z*region.inverse_scale_height_cm);
  auto vertex=selectAndSample(model,{r.particle.pdg,r.particle.energy_MeV,rho,r.particle.direction},rng);
  r.channel=vertex.channel;
  if(vertex.disposition==Disposition::needs_host_channel) {
    r.particle.pending_channel=vertex.channel;
    r.action=HistoryAction::needs_host_channel;r.error=HistoryError::none;
  } else if(vertex.disposition==Disposition::keep_parent) {
    // Fictitious or LPM-suppressed vertex: retain particle at the new position,
    // but sample a NEW interaction distance on its next native history step.
    r.action=HistoryAction::active;r.error=HistoryError::none;
  } else if(vertex.disposition==Disposition::replace_parent)applyLocalOutcome(r,vertex.local);
  else {r.action=HistoryAction::error;r.error=vertex.local.secondaries.status==InteractionStatus::random_failure?
         HistoryError::random_failure:HistoryError::sampling_failure;}
}
template<class Random,class FrameStep=FlatFrameStep>
KOKKOS_INLINE_FUNCTION HistoryOutcome finishHistory(ModelView model,HistoryPlan p,AirRegion region,
    MagneticField field,TransportSettings settings,BoundaryLimit limit,Random& rng,FrameStep frame={}) {
  HistoryOutcome r;r.particle=p.particle;r.error=p.error;
  if(p.action==PlanAction::error)return r;
  if(!validTrack(p.particle)||!validRegion(region)||!validSettings(model,settings)||p.particle.pending_channel!=Channel::invalid) {
    r.error=HistoryError::invalid_input;return r;
  }
  if(p.action==PlanAction::energy_cut){finishEnergyCut(model,r,rng);return r;}
  if(p.action==PlanAction::collision){finishCollision(model,region,r,rng);return r;}
  TransportStatus status;
  if(p.particle.pdg==22) {
    auto segment=finishPhotonSegment(p.particle,region,settings,p.photon,limit,frame);
    status=segment.status;r.particle=segment.particle;r.boundary=segment.boundary;
    r.geometric_step_cm=segment.geometric_step_cm;r.time_path_cm=segment.geometric_step_cm;
    r.transport_end_direction=segment.transport_end_direction;
  } else {
    auto segment=finishElectronSegment(model,p.particle,region,field,settings,p.electron,limit,rng,frame);
    status=segment.status;r.particle=segment.particle;r.boundary=segment.boundary;
    r.geometric_step_cm=segment.geometric_step_cm;r.loss_path_cm=segment.loss_path_cm;
    r.time_path_cm=segment.time_path_cm;r.continuous_deposit_MeV=segment.deposit_MeV;
    r.transport_end_direction=segment.transport_end_direction;
  }
  if(status==TransportStatus::invalid||status==TransportStatus::random_failure||status==TransportStatus::sampling_failure) {
    r.error=status==TransportStatus::random_failure?HistoryError::random_failure:HistoryError::transport_failure;return r;
  }
  r.has_track=true;r.error=HistoryError::none;
  if(status==TransportStatus::energy_cut){finishEnergyCut(model,r,rng);return r;}
  if(status==TransportStatus::boundary) {
    // Caller first updates geometry/medium or records an observation. A due
    // collision can then be processed at the same point on the next call.
    r.action=r.boundary==BoundaryKind::escape?HistoryAction::escaped:HistoryAction::boundary;return r;
  }
  if(status==TransportStatus::collision_due){finishCollision(model,region,r,rng);return r;}
  r.action=HistoryAction::active;return r;
}
} // namespace c7_egs4

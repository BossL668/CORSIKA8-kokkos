#pragma once
#include <corsika/detail/modules/egs4/Egs4CurvedFrame.hpp>
#include <corsika/detail/modules/egs4/Egs4History.hpp>

namespace c7_egs4 {
struct CurvedConfig {
  bool flat_output{true};
  double earth_radius_cm{},observation_height_cm{},minimum_cosine{-1.};
  bool direct_azimuth{};
};
// Per-history observer coordinates must survive suspension and be inherited
// by all children. They are not derivable by relabelling the surface-arc x/y.
struct CurvedTrack {
  TrackState particle;
  Point observer_cm;
  double wa{1.},wap{};
  bool detector_local{};
};
struct CurvedFrameStep {
  CurvedConfig config;
  CurvedTrack previous;
  CurvedResult* result; // private to the calling particle/thread, never shared
  KOKKOS_INLINE_FUNCTION FrameStepResult operator()(TrackState state,Point advanced,
      Direction direction,double geometric_cm,double time_path_cm,
      double mass_MeV,double speed_cm_s)const {
    CurvedInput q;
    q.photon=state.pdg==22;q.detector_local=previous.detector_local;q.flat_output=config.flat_output;
    q.previous_local=state.position_cm;q.previous_observer=previous.observer_cm;
    q.advanced_local=advanced;q.advanced_direction=direction;
    q.earth_radius_cm=config.earth_radius_cm;q.observation_height_cm=config.observation_height_cm;
    q.minimum_cosine=config.minimum_cosine;q.geometric_step_cm=geometric_cm;q.time_path_cm=time_path_cm;
    q.total_energy_MeV=state.energy_MeV;q.mass_MeV=mass_MeV;q.speed_cm_s=speed_cm_s;
    q.direct_azimuth=config.direct_azimuth;
    *result=rebaseCurvedStep(q);
    return {result->status==CurvedStatus::advanced,result->local,result->direction,result->time_increment_s};
  }
};
struct CurvedHistoryOutcome {
  HistoryOutcome history;
  CurvedTrack particle,children[2];
  CurvedResult frame;
  bool frame_applied{};
};
// Normal air-step composition only. An early CURVED angle/observation exit is
// exposed in frame.status and history.error; it is NOT a normal track and has
// no committed loss or children. Resolving C7's label-420/498 endpoints is a
// separate integration task. Never retry a failed call with an advanced RNG.
template<class Random>
KOKKOS_INLINE_FUNCTION CurvedHistoryOutcome finishCurvedHistory(ModelView model,HistoryPlan plan,
    CurvedTrack previous,AirRegion region,MagneticField field,TransportSettings settings,
    BoundaryLimit limit,CurvedConfig config,Random& rng) {
  CurvedHistoryOutcome r;r.particle=previous;
  CurvedFrameStep frame{config,previous,&r.frame};
  r.history=finishHistory(model,plan,region,field,settings,limit,rng,frame);
  if(r.history.action==HistoryAction::error)return r;
  r.frame_applied=r.history.has_track;
  r.particle.particle=r.history.particle;
  if(r.frame_applied) {
    r.particle.observer_cm=r.frame.observer;
    if(!previous.detector_local||!config.flat_output) {
      r.particle.wa=r.frame.wa;r.particle.wap=r.frame.wap;
    }
  }
  for(int j=0;j<r.history.child_count;++j) {
    r.children[j]=r.particle;r.children[j].particle=r.history.children[j];
  }
  return r;
}
static_assert(std::is_trivially_copyable_v<CurvedTrack>);
} // namespace c7_egs4

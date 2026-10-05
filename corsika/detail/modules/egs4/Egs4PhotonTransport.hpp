#pragma once
#include <corsika/detail/modules/egs4/Egs4Transport.hpp>

namespace c7_egs4 {
struct PhotonPlan {
  Status status{Status::invalid_input};
  PhotonQuery coefficients;
  double mean_free_path_cm{},density_inverse{},proposed_cm{};
};
// Native PHOTON normal-air transport, no Rayleigh. Flat local EGS frame;
// geometry must cap this proposal before transport (not a HOWFAR replacement).
KOKKOS_INLINE_FUNCTION PhotonPlan preparePhotonSegment(ModelView model,TrackState state,
    AirRegion region,TransportSettings settings,PhotonQuery const* supplied=nullptr) {
  PhotonPlan p;
  if(!validTrack(state)||state.pdg!=22||!validRegion(region)||
     !finite(settings.photon_cut_MeV)||settings.photon_cut_MeV<model.tables.medium.ap||
     state.energy_MeV<=settings.photon_cut_MeV||!state.photon_clock.initialized||
     !finite(state.photon_clock.remaining_mfp)||state.photon_clock.remaining_mfp<0.||
     !finite(settings.vacuum_distance_cm)||settings.vacuum_distance_cm<=0.)return p;
  p.coefficients=supplied?*supplied:photonQuery(model.tables,state.energy_MeV);
  if(p.coefficients.status!=Status::success){p.status=p.coefficients.status;return p;}
  double rhofi=1./(region.reference_density_g_cm3/model.tables.medium.rho);
  p.mean_free_path_cm=p.coefficients.mean_free_path_cm*rhofi;
  if(!finite(p.mean_free_path_cm)||p.mean_free_path_cm<=0.)return p;
  p.density_inverse=::exp(-state.position_cm.z*region.inverse_scale_height_cm);
  double step=larger(p.mean_free_path_cm*state.photon_clock.remaining_mfp,1.e-3)*p.density_inverse;
  double disc=state.direction.z*step*region.inverse_scale_height_cm;
  if(::fabs(disc)<.0000007)
    step*=1.-.5*disc*(1.-.666666666666667*disc*(1.-.75*disc*(1.-.8*disc)));
  else if(disc>-1.)step=step*::log(disc+1.)/disc;
  else step=settings.vacuum_distance_cm;
  p.proposed_cm=step;
  if(!finite(p.density_inverse)||p.density_inverse<=0.||!finite(step)||step<=0.)return p;
  p.status=Status::success;return p;
}
struct PhotonSegment {
  TransportStatus status{TransportStatus::invalid};
  TrackState particle;
  BoundaryKind boundary{BoundaryKind::none};
  double geometric_step_cm{},optical_path_cm{};
  Direction transport_end_direction;
};
template<class FrameStep=FlatFrameStep>
KOKKOS_INLINE_FUNCTION PhotonSegment finishPhotonSegment(TrackState state,AirRegion region,
    TransportSettings settings,PhotonPlan p,BoundaryLimit limit,FrameStep frame={}) {
  PhotonSegment r;r.particle=state;
  if(p.status!=Status::success||!validTrack(state)||state.pdg!=22||!validRegion(region)||
     !finite(settings.speed_cm_s)||settings.speed_cm_s<=0.||!finite(limit.distance_cm)||limit.distance_cm<=0.||
     !state.photon_clock.initialized)return r;
  double step=smaller(p.proposed_cm,limit.distance_cm),equivalent=step;
  if(limit.distance_cm<=p.proposed_cm)r.boundary=limit.kind;
  double disc=state.direction.z*equivalent*region.inverse_scale_height_cm;
  if(disc!=0.)equivalent=equivalent*(::exp(disc)-1.)/(disc*p.density_inverse);
  else equivalent/=p.density_inverse;
  equivalent=larger(equivalent,.0001);
  auto next=state;
  next.position_cm.x+=state.direction.x*step;next.position_cm.y+=state.direction.y*step;
  next.position_cm.z+=state.direction.z*step;
  auto transformed=frame(state,next.position_cm,state.direction,step,step,0.,settings.speed_cm_s);
  if(!transformed.valid)return r;
  next.position_cm=transformed.position_cm;next.direction=transformed.direction;
  next.time_s+=transformed.time_increment_s;
  // Unlike ELECTR, PHOTON retains its optical depth across a material change.
  // The next proposal obtains the new mean free path, without drawing anew.
  next.photon_clock.remaining_mfp=larger(0.,state.photon_clock.remaining_mfp-equivalent/p.mean_free_path_cm);
  if(!validTrack(next)||!finite(equivalent)||!finite(next.photon_clock.remaining_mfp))return r;
  r.particle=next;r.geometric_step_cm=step;r.optical_path_cm=equivalent;
  r.transport_end_direction=next.direction;
  r.status=next.photon_clock.remaining_mfp<=1.e-10?TransportStatus::collision_due:TransportStatus::advanced;
  if(r.boundary!=BoundaryKind::none)r.status=TransportStatus::boundary;
  return r;
}
} // namespace c7_egs4

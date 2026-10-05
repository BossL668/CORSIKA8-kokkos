#pragma once

#include <corsika/detail/modules/egs4/Egs4Interactions.hpp>

namespace c7_egs4 {
// Physical channels are explicit. In particular, the low-rate hadronic and
// muon-pair branches must be handled by an adapter, never silently rerouted
// to a PROPOSAL e/gamma reaction when the EGS4 backend is selected.
enum class Channel {
  invalid, continue_transport, bremsstrahlung, moller, bhabha, annihilation,
  electronuclear, pair_production, compton, muon_pair, photonuclear, photoelectric
};
struct ChannelThresholds {
  double mass_MeV{};
  double electronuclear_MeV{}; // C7 MAX(ELCUT(4)*1.e3, PITHR)
  double photonuclear_MeV{};  // C7 PITHR
  double muon_pair_MeV{};     // C7 RMMUT4 + ELCUT(2)*1.e3
};
KOKKOS_INLINE_FUNCTION bool validThresholds(ChannelThresholds t) {
  return finite(t.mass_MeV)&&t.mass_MeV>0.&&
    finite(t.electronuclear_MeV)&&t.electronuclear_MeV>=0.&&
    finite(t.photonuclear_MeV)&&t.photonuclear_MeV>=0.&&
    finite(t.muon_pair_MeV)&&t.muon_pair_MeV>=0.;
}
// Preserve the order and equality convention of ELECTR's final branch.
// `ae` is the PEGS discrete-secondary threshold, not the transport cut.
KOKKOS_INLINE_FUNCTION Channel electronChannel(ElectronQuery q,int charge,
    double total,double ae,ChannelThresholds t,double u) {
  if(q.status!=Status::success||!finite(total)||total<=t.mass_MeV||
     !finite(ae)||ae<=t.mass_MeV||!validThresholds(t)||
     !finite(u)||u<0.||u>=1.||(charge!=-1&&charge!=1)) return Channel::invalid;
  for(int i=0;i<(charge<0?2:3);++i) if(!finite(q.branches[i]))return Channel::invalid;
  // Do not clamp/re-normalize HATCH's interpolated branching coefficients.
  if(charge<0) {
    if(u>=q.branches[1])return Channel::bremsstrahlung;
    if(u>=q.branches[0]) {
      double thmoll=(ae-t.mass_MeV)*2.+t.mass_MeV;
      if(total<=thmoll)return q.branches[0]<=0.?Channel::continue_transport:Channel::bremsstrahlung;
      return Channel::moller;
    }
  } else {
    if(u>=q.branches[2])return Channel::bremsstrahlung;
    if(u>=q.branches[1])return Channel::bhabha;
    if(u>=q.branches[0])return Channel::annihilation;
  }
  return total<t.electronuclear_MeV?Channel::bremsstrahlung:Channel::electronuclear;
}
KOKKOS_INLINE_FUNCTION Channel photonChannel(PhotonQuery q,double energy,
                                            ChannelThresholds t,double u) {
  if(q.status!=Status::success||!finite(energy)||energy<=0.||!validThresholds(t)||
     !finite(u)||u<0.||u>=1.)return Channel::invalid;
  for(int i=0;i<4;++i)if(!finite(q.branches[i]))return Channel::invalid;
  if(u>=q.branches[3]&&energy>2.*t.mass_MeV)return Channel::pair_production;
  if(u>=q.branches[2])return Channel::compton;
  if(u<=q.branches[0]&&energy>t.muon_pair_MeV)return Channel::muon_pair;
  if(u<=q.branches[1]&&energy>t.photonuclear_MeV)return Channel::photonuclear;
  return Channel::photoelectric;
}

struct ElectronClock {
  double remaining_mfp{},sampled_rate_per_cm{};
  bool initialized{};
};
struct PhotonClock {
  double remaining_mfp{};
  bool initialized{};
};
enum class ClockStatus { invalid, transporting, collision_due, restart, accepted, random_failure };
template<class Random>
KOKKOS_INLINE_FUNCTION ClockStatus startPhotonClock(PhotonQuery q,Random& rng,PhotonClock& clock) {
  clock={};
  if(q.status!=Status::success||!finite(q.mean_free_path_cm)||q.mean_free_path_cm<=0.)return ClockStatus::invalid;
  double u;
  if(!uniform(rng,u)||u==0.)return ClockStatus::random_failure;
  clock.remaining_mfp=larger(-::log(u),1.e-14);clock.initialized=true;
  return ClockStatus::transporting;
}
template<class Random>
KOKKOS_INLINE_FUNCTION ClockStatus startElectronClock(ElectronQuery q,Random& rng,
                                                     ElectronClock& clock) {
  clock={};
  if(q.status!=Status::success||!finite(q.rate_per_cm))return ClockStatus::invalid;
  double u;
  // RMMARD's production uniforms are nonzero. A zero supplied by an adapter
  // is an explicit error here, not a changed free-path distribution.
  if(!uniform(rng,u)||u==0.)return ClockStatus::random_failure;
  clock.remaining_mfp=larger(-::log(u),1.e-14);
  clock.sampled_rate_per_cm=q.rate_per_cm;clock.initialized=true;
  return ClockStatus::transporting;
}
// TVSTEP is C7's density/path-corrected length in the region-reference medium.
// It is NOT the geometric segment length and NOT an arbitrary C8 grammage.
KOKKOS_INLINE_FUNCTION ClockStatus consumeElectronClock(ElectronClock& clock,
                                                       double tvstep,double rhofac) {
  if(!clock.initialized||!finite(clock.remaining_mfp)||!finite(clock.sampled_rate_per_cm)||
     !finite(tvstep)||tvstep<0.||!finite(rhofac)||rhofac<=0.)return ClockStatus::invalid;
  double sig=clock.sampled_rate_per_cm*rhofac;
  clock.remaining_mfp=larger(0.,clock.remaining_mfp-tvstep*sig);
  return clock.remaining_mfp>=1.e-10?ClockStatus::transporting:ClockStatus::collision_due;
}
// The initial sigma is held fixed while continuous energy loss changes the
// current energy. This is C7's fictitious-sigma check, not per-step resampling.
template<class Random>
KOKKOS_INLINE_FUNCTION ClockStatus acceptElectronCollision(ElectronClock& clock,
                                                           ElectronQuery final,Random& rng) {
  if(!clock.initialized||clock.remaining_mfp>=1.e-10||clock.sampled_rate_per_cm<=0.||
     final.status!=Status::success||!finite(final.rate_per_cm))return ClockStatus::invalid;
  double u;
  if(!uniform(rng,u))return ClockStatus::random_failure;
  auto status=u>final.rate_per_cm/clock.sampled_rate_per_cm?ClockStatus::restart:ClockStatus::accepted;
  clock.initialized=false; // caller must begin a new history after either branch
  return status;
}
// Use up-to-date dE/dx/TMXS but the originally sampled sigma for the local
// step proposal. Merely querying the current sigma here changes C7's scheme.
KOKKOS_INLINE_FUNCTION ElectronQuery withSampledRate(ElectronQuery current,ElectronClock clock) {
  if(!clock.initialized||!finite(clock.sampled_rate_per_cm))current.status=Status::invalid_input;
  else current.rate_per_cm=clock.sampled_rate_per_cm;
  return current;
}
} // namespace c7_egs4

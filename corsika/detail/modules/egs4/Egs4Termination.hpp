#pragma once
#include <corsika/detail/modules/egs4/Egs4Interactions.hpp>

namespace c7_egs4 {
enum class DepositKind { none, ionization, subcut_kinetic };
struct LocalOutcome {
  Secondaries secondaries;
  double deposit_MeV{}; // unweighted; stack/output layer applies history weight
  DepositKind deposit_kind{DepositKind::none};
};

// C7 ANNIH, in flight, not the stopped-positron rule below. The literal
// source gives both photons a positive SINTHE and reuses the same phi.
// Preserve this for source-equivalence testing; it is a known transverse
// momentum issue and must NOT be described as a conservation-validated model.
template<class Random>
KOKKOS_INLINE_FUNCTION Secondaries annihilateInFlight(
    double E,double m,Direction direction,Random& rng,int max_trials=4096) {
  Secondaries r;
  if(!finite(E)||!finite(m)||m<=0.||E<=m||!validDirection(direction)||max_trials<1)return r;
  double available=E+m,a=available*(1./m),ai=1./a,g=a-1.,t=g-1.;
  double p=::sqrt(a*t),pot=p/t,ep0i=a+p;
  if(!finite(pot)||!finite(ep0i)||ep0i<=1.)return r;
  for(int trial=1;trial<=max_trials;++trial) {
    r.trials=trial;double u,v;
    if(!uniform(rng,u)||!uniform(rng,v)){r.status=InteractionStatus::random_failure;return r;}
    double ep=::exp(u*::log(ep0i-1.))/ep0i;
    double acceptance=1.-ep+ai*ai*(2.*g-1./ep);
    if(v>acceptance)continue;
    double e1=available*larger(ep,1.-ep),e2=available-e1,sp,cp;
    if(!azimuth(rng,sp,cp)){r.status=InteractionStatus::random_failure;return r;}
    double c1=(e1-m)*pot/e1,c2=(e2-m)*pot/e2;
    r.particle[0]={e1,rotate(direction,::sqrt(larger(0.,(1.-c1)*(1.+c1))),c1,sp,cp),22};
    r.particle[1]={e2,rotate(direction,::sqrt(larger(0.,(1.-c2)*(1.+c2))),c2,sp,cp),22};
    r.count=2;r.status=InteractionStatus::success;return r;
  }
  r.status=InteractionStatus::trial_limit;return r;
}

// PHOTO: binding energy deposited locally, forward photoelectron; no
// fluorescence or Auger cascade added to the source model.
KOKKOS_INLINE_FUNCTION LocalOutcome photoelectric(
    double E,double m,double binding,Direction direction) {
  LocalOutcome r;
  if(!finite(E)||E<=0.||!finite(m)||m<=0.||!finite(binding)||binding<0.||
     !validDirection(direction))return r;
  r.secondaries.status=InteractionStatus::success;r.deposit_kind=DepositKind::ionization;
  if(E<=binding){r.deposit_MeV=E;return r;}
  r.deposit_MeV=binding;r.secondaries.count=1;
  r.secondaries.particle[0]={E-binding+m,direction,11};return r;
}

// ELECTR label 390: caller has ALREADY selected the kinetic-energy cutoff.
// Do not use this for particles leaving the atmosphere or an angular cut.
// A stopped positron deposits kinetic energy and produces two opposite
// rest-mass photons, regardless of the photon cut. PHOTON later applies
// its own cut, preventing double deposition or premature photon deletion.
template<class Random>
KOKKOS_INLINE_FUNCTION LocalOutcome stopLepton(
    double E,double m,int pdg,Random& rng) {
  LocalOutcome r;
  if(!finite(E)||!finite(m)||m<=0.||E<m||(pdg!=11&&pdg!=-11))return r;
  r.deposit_kind=DepositKind::subcut_kinetic;r.deposit_MeV=E-m;
  if(pdg==11){r.secondaries.status=InteractionStatus::success;return r;}
  double ct,flip,sp,cp;
  if(!uniform(rng,ct)||!uniform(rng,flip)||!azimuth(rng,sp,cp)) {
    r.secondaries.status=InteractionStatus::random_failure;return r;
  }
  if(flip<=.5)ct=-ct;
  auto d=rotate({0.,0.,1.},::sqrt(larger(0.,(1.-ct)*(1.+ct))),ct,sp,cp);
  r.secondaries.particle[0]={m,d,22};r.secondaries.particle[1]={m,{-d.x,-d.y,-d.z},22};
  r.secondaries.count=2;r.secondaries.status=InteractionStatus::success;return r;
}
} // namespace c7_egs4

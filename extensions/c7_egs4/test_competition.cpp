#include "Egs4Competition.hpp"
#include <iostream>

namespace {
using namespace c7_egs4;
struct Fixed {
  double u;int used{};
  KOKKOS_INLINE_FUNCTION bool next(double& out){out=u;++used;return true;}
};
// Branch truth table from the inspected ELECTR and PHOTON control flow.
// This checks selectors/history contracts; the independent C7 physics
// sampler comparisons are separate tests, not this synthetic fixture.
KOKKOS_INLINE_FUNCTION int checkContracts() {
  int failures=0;
  ChannelThresholds t{.5,150.,140.,500.};
  ElectronQuery e;e.status=Status::success;e.rate_per_cm=2.;
  e.loss_MeV_per_cm=3.;e.maximum_step_cm=4.;
  e.branches[0]=.1;e.branches[1]=.6;
  failures+=electronChannel(e,-1,200.,.9,t,.6)!=Channel::bremsstrahlung;
  failures+=electronChannel(e,-1,200.,.9,t,.1)!=Channel::moller;
  failures+=electronChannel(e,-1,150.,.9,t,.05)!=Channel::electronuclear;
  failures+=electronChannel(e,-1,149.,.9,t,.05)!=Channel::bremsstrahlung;
  failures+=electronChannel(e,-1,1.3,.9,t,.2)!=Channel::bremsstrahlung;
  e.branches[0]=0.;
  failures+=electronChannel(e,-1,1.3,.9,t,.2)!=Channel::continue_transport;
  e.branches[0]=.1;e.branches[1]=.2;e.branches[2]=.6;
  failures+=electronChannel(e,1,200.,.9,t,.6)!=Channel::bremsstrahlung;
  failures+=electronChannel(e,1,200.,.9,t,.2)!=Channel::bhabha;
  failures+=electronChannel(e,1,200.,.9,t,.1)!=Channel::annihilation;
  failures+=electronChannel(e,1,200.,.9,t,0.)!=Channel::electronuclear;
  failures+=electronChannel(e,1,1.,.9,t,0.)!=Channel::bremsstrahlung;
  failures+=electronChannel(e,0,1.,.9,t,0.)!=Channel::invalid;
  failures+=electronChannel(e,-1,1.,.9,t,1.)!=Channel::invalid;
  PhotonQuery p;p.status=Status::success;
  p.branches[0]=.1;p.branches[1]=.2;p.branches[2]=.4;p.branches[3]=.7;
  failures+=photonChannel(p,2.,t,.7)!=Channel::pair_production;
  failures+=photonChannel(p,1.,t,.7)!=Channel::compton;
  failures+=photonChannel(p,1000.,t,.4)!=Channel::compton;
  failures+=photonChannel(p,501.,t,.1)!=Channel::muon_pair;
  failures+=photonChannel(p,500.,t,.1)!=Channel::photonuclear;
  failures+=photonChannel(p,141.,t,.2)!=Channel::photonuclear;
  failures+=photonChannel(p,140.,t,.2)!=Channel::photoelectric;
  failures+=photonChannel(p,1.,t,.1)!=Channel::photoelectric;
  failures+=photonChannel(p,1000.,t,.3)!=Channel::photoelectric;
  p.status=Status::outside_table;
  failures+=photonChannel(p,1000.,t,.3)!=Channel::invalid;
  ElectronClock clock;Fixed rng{::exp(-1.)};
  failures+=startElectronClock(e,rng,clock)!=ClockStatus::transporting;
  failures+=::fabs(clock.remaining_mfp-1.)>1.e-14||rng.used!=1;
  // Energy changes between steps, but sigma0 must not be replaced by sigmaf.
  ElectronQuery current=e;current.rate_per_cm=.5;current.loss_MeV_per_cm=9.;
  auto modified=withSampledRate(current,clock);
  failures+=modified.rate_per_cm!=2.||modified.loss_MeV_per_cm!=9.;
  failures+=consumeElectronClock(clock,.1,1.)!=ClockStatus::transporting;
  failures+=::fabs(clock.remaining_mfp-.8)>1.e-14;
  failures+=consumeElectronClock(clock,.4,1.)!=ClockStatus::collision_due;
  failures+=acceptElectronCollision(clock,current,rng)!=ClockStatus::restart;
  failures+=clock.initialized||rng.used!=2;
  rng.u=.5;startElectronClock(e,rng,clock);consumeElectronClock(clock,1.,1.);
  failures+=acceptElectronCollision(clock,e,rng)!=ClockStatus::accepted;
  failures+=withSampledRate(current,clock).status!=Status::invalid_input;
  rng.u=0.;failures+=startElectronClock(e,rng,clock)!=ClockStatus::random_failure;
  failures+=clock.initialized;
  return failures;
}
template<class Exec>struct Batch {
  Kokkos::View<int*,typename Exec::memory_space> output;
  KOKKOS_INLINE_FUNCTION void operator()(int i)const{output(i)=checkContracts();}
};
}
int main(int argc,char** argv) {
  Kokkos::ScopeGuard guard(argc,argv);using Exec=Kokkos::DefaultExecutionSpace;
  Kokkos::View<int*,typename Exec::memory_space> out("contracts",128);
  Kokkos::parallel_for("egs4_history_contracts",Kokkos::RangePolicy<Exec>(0,128),Batch<Exec>{out});
  Exec().fence();auto h=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},out);
  int errors=checkContracts();for(int i=0;i<128;++i)errors+=h(i);
  std::cout<<"execution_space="<<Exec::name()<<"\nchannel_and_history_contract_errors="<<errors<<'\n';
  return errors?1:0;
}

/* Muon bremsstrahlung and CPU pair-routing validation. BSD-3-Clause. */
#include "MuonBremsDriver.hpp"
#include <corsika/accelerator/em/detail/LeptonFinalStateStep.hpp>
#include <corsika/accelerator/em/kokkos/KokkosProposalNativeTable.hpp>
#include <Kokkos_Core.hpp>
#include <iostream>
#include <chrono>
namespace muon_test {
namespace impl = corsika::accelerator::em::detail;
namespace nt = em::tables;
using Exec = Kokkos::DefaultExecutionSpace;
void initialize(){Kokkos::initialize(); std::cout<<"Execution="<<Exec::name()<<std::endl;}
void finalize(){Kokkos::finalize();}

KOKKOS_INLINE_FUNCTION Output evaluate(Input q, nt::ProposalNativeDeviceView view,
                                      em::BremsLpmSnapshot snapshot) {
  em::EmInteractionRecord r{};
  r.status=em::EmInteractionStatus::Selected;
  r.interaction_vertex_reached=1;
  r.process_id=q.process; r.component_hash=q.component;
  r.energy_fraction=q.v; r.mass_density_g_per_cm3=q.density;
  r.particle_mass_GeV=(em::isMuonPid(q.pid)?snapshot.muon_mass_MeV:snapshot.electron_mass_MeV)/1000.;
  r.process_uniform=.5;
  r.particle.pid=q.pid; r.particle.energy_GeV=q.energy/1000.;
  r.particle.weight=1.; r.particle.history_id=q.history;
  // Include exact poles as well as a non-axis-aligned direction.
  if(q.history%3==0) r.particle.direction[2]=1.;
  else if(q.history%3==1) r.particle.direction[2]=-1.;
  else {r.particle.direction[0]=.6; r.particle.direction[2]=.8;}
  if(q.fault==1) snapshot.muon_e_lpm_MeV=0.;
  if(q.fault==2) snapshot.component_count=0;
  if(q.fault==3) r.particle_mass_GeV=snapshot.electron_mass_MeV/1000.;
  if(q.fault==4) r.energy_fraction=NAN;
  if(q.fault==5) r.mass_density_g_per_cm3=-1.;
  if(q.fault==6) r.component_hash=0;
  if(q.fault==7) r.energy_fraction=1.;
  if(q.fault==8) r.particle.energy_GeV=.1;
  em::EmThinningConfig thin{};
  if(q.thinning){thin.enabled=1;thin.threshold_GeV=2*q.energy/1000.;thin.maximum_weight=1000.;}
  auto c=impl::classifyLeptonFinalState(snapshot,thin,r,26101191,71);
  auto m=impl::materializeLeptonFinalState(r,c,0,100,snapshot.electron_mass_MeV/1000.);
  Output out{};
  out.fallback_event=c.fallback;
  // The orchestration layer attaches calculator hashes after the kernel.
  out.fallback_event.medium_hash=1;out.fallback_event.interaction_hash=2;
  out.fallback=c.fallback_flag; out.error=m.error; out.count=m.secondary_count;
  out.suppressed=c.suppression_flag; out.trials=c.electron_pair_rejection_trials;
  out.rho=c.parameters.energy_split_fraction; out.lpm=c.parameters.lpm_survival_probability;
  out.azimuth=c.parameters.azimuth_uniform; out.sign=c.parameters.azimuth_uniform;
  out.lpm_uniform=c.parameters.lpm_uniform;
  out.replay=em::proposalNativeSelectionRequiresReplay(q.pid,q.process,q.quantile);
  auto loss=nt::queryProposalNativeLossFraction(view,q.pid,q.process,q.component,q.energy,q.quantile);
  out.loss=loss.value; out.loss_status=unsigned(loss.status);
  for(unsigned j=0;j<m.secondary_count && j<3;++j){
    out.pids[j]=m.secondaries[j].pid;out.energies[j]=m.secondaries[j].energy_GeV;
    out.weights[j]=m.secondaries[j].weight;
    for(int k=0;k<3;++k)out.directions[j][k]=m.secondaries[j].direction[k];
  }
  return out;
}
std::vector<Output> run(nt::ProposalNativeTableSet const& tables,
    em::BremsLpmSnapshot const& snapshot,std::vector<Input> const& inputs,bool on_device) {
  std::vector<Output> result(inputs.size());
  if(!on_device){
    auto view=nt::makeProposalNativeHostView(tables);
    for(std::size_t i=0;i<inputs.size();++i)result[i]=evaluate(inputs[i],view,snapshot);
    return result;
  }
  corsika::accelerator::em::kokkos_detail::KokkosProposalNativeTable<Exec> owner;
  owner.initialize(tables);auto view=owner.deviceView();auto lpm=snapshot;
  Kokkos::View<Input*,Exec> in("muon_input",inputs.size());
  auto hi=Kokkos::create_mirror_view(in);
  for(std::size_t i=0;i<inputs.size();++i)hi(i)=inputs[i];
  Kokkos::deep_copy(in,hi);
  Kokkos::View<Output*,Exec> out("muon_output",inputs.size());
  Kokkos::parallel_for("muon_radiative_finalstates",Kokkos::RangePolicy<Exec>(0,inputs.size()),
    KOKKOS_LAMBDA(std::size_t i){out(i)=evaluate(in(i),view,lpm);});
  auto ho=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},out);
  for(std::size_t i=0;i<inputs.size();++i)result[i]=ho(i);
  return result;
}

}

/* (c) Copyright 2026 CORSIKA Project; BSD-3-Clause. */
#include "PhotonMuonPairDriver.hpp"
#include <corsika/accelerator/em/detail/PhotonFinalStateStep.hpp>
#include <corsika/accelerator/em/detail/InteractionSelection.hpp>
#include <corsika/accelerator/em/detail/LeptonTransportStep.hpp>
#include <corsika/accelerator/em/kokkos/KokkosProfileAccumulator.hpp>
#include <corsika/accelerator/em/kokkos/KokkosPhotonFinalState.hpp>
#include <corsika/accelerator/em/kokkos/KokkosProposalNativeTable.hpp>
#include <Kokkos_Core.hpp>
#include <algorithm>
#include <chrono>
#include <iostream>
#include <stdexcept>
namespace photon_muon_test {
namespace nt = em::tables;
namespace impl = corsika::accelerator::em::detail;
namespace kd = corsika::accelerator::em::kokkos_detail;
using Exec=Kokkos::DefaultExecutionSpace;
void initialize(){ Kokkos::initialize(); std::cout<<"Execution="<<Exec::name()<<std::endl; }
void finalize(){ Kokkos::finalize(); }
KOKKOS_INLINE_FUNCTION em::EmInteractionRecord interaction(Input q) {
  em::EmInteractionRecord r{};
  r.status=em::EmInteractionStatus::Selected;
  r.interaction_vertex_reached=1; r.process_id=em::PhotonMuonPairProcessId;
  r.component_hash=q.component; r.energy_fraction=1.; r.particle.pid=22;
  r.mass_density_g_per_cm3=.001;
  r.particle.energy_GeV=q.energy_MeV/1000.; r.particle.weight=1.;
  r.particle.history_id=q.history;
  r.particle.time_s=1.e-6; r.particle.position_m[0]=3.;
  if(q.history%3==0)r.particle.direction[2]=1.;
  else if(q.history%3==1)r.particle.direction[2]=-1.;
  else {r.particle.direction[0]=.6;r.particle.direction[2]=.8;}
  return r;
}
KOKKOS_INLINE_FUNCTION Output evaluate(Input q, nt::ProposalNativeDeviceView view) {
  auto r=interaction(q);
  if(q.fault==1)view.dndx_column_count=0;
  if(q.fault==2)r.component_hash=0;
  if(q.fault==3)r.particle.generation=0xffffffffU;
  if(q.fault==4)r.particle.weight=NAN;
  if(q.fault==5)r.particle.weight=-1.;
  nt::NativePhysicsView physics{}; physics.proposal_native=view;
  em::EmThinningConfig thin{};
  if(q.thinning){thin.enabled=1;thin.threshold_GeV=q.energy_MeV/500.;thin.maximum_weight=1000.;}
  auto c=impl::classifyPhotonFinalState(physics,{},thin,r,26101193,73);
  auto m=impl::materializePhotonFinalState(r,c,0,1000+2*q.history);
  Output out{};
  out.x=c.parameters.split_fraction;out.uniform=c.parameters.split_uniform;
  out.azimuth=c.parameters.azimuth_uniform;out.mask=c.parameters.thinning_keep_mask;
  out.fallback=c.fallback_flag;out.reason=unsigned(c.fallback.reason);out.error=m.error;
  out.diagnostic=c.fallback.diagnostic_status;
  out.count=m.secondary_count;out.mass_correction=m.record.weighted_mass_convention_correction_GeV;
  auto v=nt::queryProposalNativeLossFraction(view,22,em::PhotonMuonPairProcessId,q.component,q.energy_MeV,q.quantile);
  out.public_v=v.value;out.public_status=unsigned(v.status);
  auto x=nt::queryLossFraction(physics,22,em::PhotonMuonPairFinalStateProcessId,q.component,q.energy_MeV,q.quantile);
  out.probe=x.value;out.probe_status=unsigned(x.status);
  for(unsigned j=0;j<m.secondary_count && j<2;++j){
    out.children[j]=m.secondaries[j];
    if(em::isNativeMuonPairCut(out.children[j])) {
      auto selected=impl::selectDiscreteInteraction(physics,out.children[j],j,26101193,73);
      em::LeptonTransportRecord cut{};em::ProposalFallbackEvent fallback{};
      em::EnvironmentSnapshot environment{};
      out.cut_status[j]=impl::transportLepton(physics,true,environment,selected.interaction,cut,fallback);
      double axis[]={0.,10.};em::detail::DeviceProfileProjection projection{};
      projection.axis_direction[0]=1.;projection.axis_step_length_m=10.;
      projection.axis_grammage_g_per_cm2=axis;projection.axis_support_count=2;
      out.cuts[j]=impl::projectLeptonStep(projection,cut);
      if(selected.fallback_flag || selected.interaction.status!=em::EmInteractionStatus::ParticleCut ||
         cut.start.time_s!=cut.end.time_s || cut.start.position_m[0]!=cut.end.position_m[0] ||
         cut.continuous_deposited_energy_GeV!=0. || cut.multiple_scattering_applied)
        out.cut_status[j]=99;
    }
  }
  return out;
}
Result run(nt::ProposalNativeTableSet const& tables,std::vector<Input> const& inputs,bool device){
  Result result;result.outputs.resize(inputs.size());std::vector<double> times;
  auto seconds=[](auto start){return std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();};
  if(!device){
    auto view=nt::makeProposalNativeHostView(tables);
    for(int repeat=0;repeat<4;++repeat){auto start=std::chrono::steady_clock::now();
      for(std::size_t i=0;i<inputs.size();++i)result.outputs[i]=evaluate(inputs[i],view);
      if(repeat)times.push_back(seconds(start));}
  }else{
    kd::KokkosProposalNativeTable<Exec> owner;owner.initialize(tables);auto view=owner.deviceView();
    Kokkos::View<Input*,Exec> in("photomu_inputs",inputs.size());
    Kokkos::View<Output*,Exec> out("photomu_outputs",inputs.size());
    auto hi=Kokkos::create_mirror_view(in);for(std::size_t i=0;i<inputs.size();++i)hi(i)=inputs[i];
    Kokkos::deep_copy(in,hi);
    for(int repeat=0;repeat<4;++repeat){auto start=std::chrono::steady_clock::now();
      Kokkos::parallel_for("photomu_final_state",Kokkos::RangePolicy<Exec>(0,inputs.size()),
          KOKKOS_LAMBDA(std::size_t i){out(i)=evaluate(in(i),view);});Kokkos::fence();
      if(repeat)times.push_back(seconds(start));}
    auto ho=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},out);
    for(std::size_t i=0;i<inputs.size();++i)result.outputs[i]=ho(i);
  }
  std::sort(times.begin(),times.end());result.kernel_s=times[1];return result;
}
void checkBatch(nt::ProposalNativeTableSet const& tables,std::vector<Input> const& inputs){
  kd::KokkosProposalNativeTable<Exec> owner;owner.initialize(tables);
  nt::NativePhysicsView physics{};physics.proposal_native=owner.deviceView();
  std::vector<em::EmInteractionRecord> records;
  for(auto q:inputs){auto r=interaction(q);r.input_index=records.size();records.push_back(r);}
  auto result=kd::generatePhotonFinalStates<Exec>(physics,{},{},records,26101193,73,10000000);
  auto const& b=result.batch;
  if(b.secondaries.size()!=2*inputs.size() || !b.fallback_events.empty() ||
     b.photon_muon_pair_interactions!=inputs.size() || b.photon_pair_interactions!=0 || !b.lpm_suppressed.empty())
    throw std::runtime_error("Kokkos photon batch/counter/scan failed");
  for(std::size_t i=0;i<b.secondaries.size();++i)
    if(b.secondaries[i].pid!=(i%2?-13:13) || b.secondaries[i].history_id!=10000000+i)
      throw std::runtime_error("Kokkos scan changed muon order or history identity");
}
em::GpuProfileResult checkCutProfile(std::vector<em::EmParticleState> const& children){
  using Owner=kd::KokkosProfileAccumulator<Exec>;
  em::GpuEmConfig::ProfileProjection cfg{};
  cfg.enabled=cfg.accumulate_on_device=true;cfg.output_bin_count=2;
  cfg.output_bin_width_g_per_cm2=10.;cfg.fixed_point_weight_limit=1000.;
  cfg.fixed_point_energy_limit_GeV=1000.;
  Owner owner;owner.initialize(cfg);
  typename Owner::LeptonStepView steps("native_cuts",children.size());
  auto host=Kokkos::create_mirror_view(steps);
  em::EnvironmentSnapshot environment{};
  for(std::size_t i=0;i<children.size();++i){
    em::LeptonTransportRecord cut{};em::ProposalFallbackEvent fallback{};
    auto selected=impl::selectDiscreteInteraction({},children[i],i,1,1);
    if(selected.fallback_flag || impl::transportLepton({},true,environment,selected.interaction,cut,fallback)!=2)
      throw std::runtime_error("tagged child did not terminate locally");
    host(i)=cut;
    auto invalid=children[i];invalid.reserved=0;
    selected.interaction.particle=invalid;
    if(impl::transportLepton({},true,environment,selected.interaction,cut,fallback)!=1)
      throw std::runtime_error("untagged invalid energy unexpectedly transported");
  }
  Kokkos::deep_copy(steps,host);
  Kokkos::View<double*,Exec> axis("cut_axis",2);
  auto ha=Kokkos::create_mirror_view(axis);ha(0)=0.;ha(1)=10.;Kokkos::deep_copy(axis,ha);
  em::detail::DeviceProfileProjection projection{};projection.axis_direction[0]=1.;
  projection.axis_step_length_m=10.;projection.axis_support_count=2;
  projection.axis_grammage_g_per_cm2=axis.data();
  owner.accumulateLepton(projection,steps,children.size(),typename Owner::LeptonFinalView{},0);
  auto result=owner.download();
  if(result.invalid_records || result.fixed_point_overflows || result.particle_cuts!=children.size())
    throw std::runtime_error("native cut profile failed");
  return result;
}
}

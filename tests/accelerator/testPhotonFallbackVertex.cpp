/* (c) Copyright 2026 CORSIKA Project; BSD-3-Clause. */
#include <Kokkos_Core.hpp>
#include <corsika/accelerator/em/detail/InteractionSelection.hpp>
#include <corsika/accelerator/em/detail/PhotonTransportStep.hpp>
#include <corsika/accelerator/em/detail/PhotonFinalStateStep.hpp>
#include <iostream>
#include <stdexcept>
#include <string>

namespace em = corsika::gpu::em;
namespace nt = em::tables;
namespace physics = corsika::accelerator::em::detail;

// Synthetic constant-rate fixtures isolate transport ordering, not a PROPOSAL
// cross-section accuracy oracle. Arrays are created on the execution space.
struct Result { unsigned errors{}, completed{}, boundary{}, cut{}, pending{}; };
KOKKOS_FUNCTION Result exercise(std::size_t index) {
  Result out{};
  auto check = [&](bool ok) { if (!ok) ++out.errors; };
  double values[4]{0., 0., .02, .02}, zeros[4]{}, dy[4]{.02,.02,.02,.02};
  double cubic[2]{.02,.02}, derivative[2]{};
  std::uint32_t ordinal[1]{0};
  nt::NativeAxisDescriptor energy{nt::NativeAxisType::Linear,2,1.,1.e6,999999.};
  nt::NativeDndxColumn column{};
  column.pdg_id=22; column.component_hash=7;
  auto process = (index/7)%4;
  column.process_id=process==0 ? em::PhotoproductionProcessId :
                    process==1 ? em::PhotonMuonPairProcessId :
                    process==2 ? em::PhotonPairProcessId : em::ComptonProcessId;
  column.kinematic_model=process==3 ? nt::NativeKinematicModel::Unsupported :
                                                   nt::NativeKinematicModel::OnlyStochastic;
  column.spline.energy_axis=energy;
  column.spline.loss_axis={nt::NativeAxisType::Linear,2,0.,1.,1.};
  column.spline.rows=column.spline.columns=2;
  nt::NativeTotalRateColumn total{};
  total.pdg_id=22;total.spline.axis=energy;total.spline.coefficient_count=2;
  nt::NativePhysicsView table{};
  auto& v=table.proposal_native;
  v.dndx_columns=&column;v.dndx_column_count=1;
  v.total_rate_columns=&total;v.total_rate_column_count=1;
  v.selection_column_indices=ordinal;v.selection_column_index_count=1;
  v.bicubic_values=values;v.bicubic_derivative_energy=zeros;
  v.bicubic_derivative_loss=dy;v.bicubic_mixed_derivative=zeros;
  v.bicubic_coefficient_count=4;v.cubic_values=cubic;
  v.cubic_node_derivatives=derivative;v.cubic_coefficient_count=2;
  table.em_transport_cut_MeV=.5;
  constexpr double R=6371000.;
  em::EnvironmentSnapshot env{};
  env.number_of_layers=2;env.observation_radius_m=R;
  env.observation_plane_point_m[2]=R;env.observation_plane_normal[2]=1.;
  env.atmosphere_layers[0]={R,R+7000.,.001,0.,0.,0,em::DensityModel::Homogeneous};
  env.atmosphere_layers[1]={R+7000.,R+20000.,.001,0.,0.,0,em::DensityModel::Homogeneous};
  em::EmParticleState p{};
  p.pid=22;p.energy_GeV=1.;p.weight=1.;p.direction[2]=-1.;
  p.position_m[2]=R+7000.;p.history_id=index+1;
  auto scenario=index%7;
  if(scenario==1)p.position_m[2]+=1.;
  if(scenario==2)env.observation_plane_point_m[2]=R+6999.;
  if(scenario==4)p.time_s=em::ParticleCutMaximumTimeS+1.e-6;
  if(scenario==5)p.time_s=em::ParticleCutMaximumTimeS-1.e-12;
  if(scenario==6)p.energy_GeV=1.e-5; // energy cut wins before selection
  em::ExternalTransportBoundary boundary{};
  if(scenario==3){boundary.enabled=true;boundary.distance_m=.5;}
  auto selected=physics::selectDiscreteInteraction(table,p,index,2026110001,0);
  check(!selected.fallback_flag);
  check(!selected.interaction.interaction_vertex_reached);
  if(scenario!=4&&scenario!=6&&process!=2) {
    check(selected.interaction.deferred_photon_fallback_reason>=0);
    ++out.pending;
    // Even a syntactically complete selection must NOT be allowed to create
    // its CPU final state before transport. The classifier rejects this call.
    auto premature=physics::classifyPhotonFinalState(table,{}, {},selected.interaction,2026110001,0);
    check(premature.fallback_flag && premature.fallback.reason==em::ProposalFallbackReason::InvalidFinalState);
    check(!premature.fallback.interaction_vertex_reached);
  }
  auto transported=physics::transportPhoton(env,selected.interaction,boundary);
  check(!transported.fallback_flag);
  auto const& step=transported.record;
  if(step.limit!=em::PhotonTransportLimit::ParticleCut) {
    check(step.start.energy_GeV==step.end.energy_GeV);
    check(step.start.weight==step.end.weight);
    check(step.cut_deposited_energy_GeV==0.);
  } else {
    check(step.cut_deposited_energy_GeV==p.energy_GeV);
  }
  if(step.limit==em::PhotonTransportLimit::Interaction) {
    check(step.interaction.interaction_vertex_reached==1);
    auto expected=selected.interaction.interaction_grammage_g_per_cm2/.1;
    check(::fabs(step.distance_m-expected)<1.e-9*(1.+expected));
    check(step.end.time_s>p.time_s);
    check(step.end.position_m[2]<p.position_m[2]);
    if(selected.interaction.deferred_photon_fallback_reason>=0) {
      auto final=physics::classifyPhotonFinalState(table,{}, {},step.interaction,2026110001,0);
      check(final.fallback_flag && !final.record_flag && !final.child_count);
      check(final.fallback.interaction_vertex_reached==1);
      check(final.fallback.particle.position_m[2]==step.end.position_m[2]);
      check(final.fallback.particle.time_s==step.end.time_s);
      check(final.fallback.process_id==selected.interaction.process_id);
      check(final.fallback.component_hash==selected.interaction.component_hash);
      check(final.fallback.particle.history_id==p.history_id);
      check(final.fallback.selection_uniform==selected.interaction.proposal_selection_uniform);
      check(final.fallback.loss_quantile==selected.interaction.loss_quantile);
      // All supported selected-loss failure reasons obey the same vertex rule.
      for(int reason=6;reason<=8;++reason) {
        auto candidate=step.interaction;candidate.deferred_photon_fallback_reason=reason;
        auto f=physics::classifyPhotonFinalState(table,{}, {},candidate,2026110001,0);
        check(f.fallback_flag && f.fallback.interaction_vertex_reached==1);
        check(static_cast<int>(f.fallback.reason)==reason);
      }
    }
    ++out.completed;
  } else {
    check(!step.interaction.interaction_vertex_reached);
    if(step.limit==em::PhotonTransportLimit::ParticleCut)++out.cut;
    else ++out.boundary;
    if(scenario==1 && step.distance_m>=1.)check(step.limit==em::PhotonTransportLimit::LayerBoundary);
    if(scenario==2 && step.distance_m>=1.)check(step.limit==em::PhotonTransportLimit::ObservationSurface);
    if(scenario==3 && step.distance_m>=.5)check(step.limit==em::PhotonTransportLimit::MaterialBoundary);
  }
  // Charged leptons (including both muon charges) defer process selection
  // until continuous loss has completed. This ordering must remain intact.
  if(scenario==0) {
    for(int pid : {11,-11,13,-13}) {
      column.pdg_id=total.pdg_id=p.pid=pid;
      auto lepton=physics::selectDiscreteInteraction(table,p,index,2026110001,0);
      check(!lepton.fallback_flag);
      check(lepton.interaction.status==em::EmInteractionStatus::DistanceSampled);
      check(lepton.interaction.deferred_photon_fallback_reason<0);
      check(!lepton.interaction.interaction_vertex_reached);
    }
  }
  return out;
}

template<class Ex> void run() {
  constexpr std::size_t samples=28*2048;
  Kokkos::View<Result*,typename Ex::memory_space> results("vertex_results",samples);
  Kokkos::parallel_for("photon_fallback_vertex",Kokkos::RangePolicy<Ex>(0,samples),
      KOKKOS_LAMBDA(std::size_t i){results(i)=exercise(i);});
  auto host=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},results);
  Result total{};
  for(std::size_t i=0;i<samples;++i){
    if(host(i).errors)std::cerr<<"failed fixture="<<i<<" errors="<<host(i).errors<<'\n';
    total.errors+=host(i).errors;total.completed+=host(i).completed;
    total.boundary+=host(i).boundary;total.cut+=host(i).cut;total.pending+=host(i).pending;
  }
  std::cout<<Ex::name()<<": samples="<<samples<<" errors="<<total.errors
           <<" vertices="<<total.completed<<" boundary="<<total.boundary
           <<" cuts="<<total.cut<<" deferred="<<total.pending<<'\n';
  if(total.errors||!total.completed||!total.boundary||!total.cut||!total.pending)
    throw std::runtime_error("photon selected-loss vertex regression failed");
}
int main(int argc,char** argv) {
  Kokkos::ScopeGuard guard(Kokkos::InitializationSettings{}.set_num_threads(2));
#ifdef KOKKOS_ENABLE_OPENMP
  if(argc>1 && std::string(argv[1])=="openmp"){run<Kokkos::OpenMP>();return 0;}
#endif
  run<Kokkos::DefaultExecutionSpace>();
}

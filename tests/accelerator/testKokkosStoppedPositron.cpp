/* Production shared endpoint kernels, exercised on the compiled execution space. */
#include <Kokkos_Core.hpp>
#include <corsika/accelerator/em/detail/LeptonTransportStep.hpp>
#include <corsika/accelerator/em/detail/LeptonVertexSelection.hpp>
#include <corsika/accelerator/em/detail/LeptonFinalStateStep.hpp>
#include <corsika/accelerator/em/detail/InteractionSelection.hpp>
#include <iostream>
#include <stdexcept>
namespace em=corsika::gpu::em;
namespace phy=corsika::accelerator::em::detail;
struct Result { unsigned errors{}; double x{},y{},z{},xx{},yy{},zz{}; };
struct EndpointTest {
  Kokkos::View<Result*> results;
  KOKKOS_FUNCTION void operator()(std::size_t i) const {
    Result r{};
    em::LeptonTransportRecord cut{};
    cut.start.pid=-11;cut.start.energy_GeV=em::TransportElectronMassGeV+0.0002;
    cut.start.weight=7.;cut.start.history_id=i+1;cut.start.step_id=3;
    cut.start.direction[2]=1.;cut.start.position_m[2]=100.;
    cut.end=cut.start;cut.end.position_m[2]=101.;cut.end.time_s=1.e-9;
    cut.end.step_id=4;cut.input_index=i;cut.interaction.input_index=i;
    cut.limit=em::LeptonTransportLimit::ParticleCut;cut.cut_deposited_energy_GeV=.0002;
    phy::prepareStoppedAnnihilation(cut);
    auto vertex=phy::selectLeptonVertex({},cut.interaction,20260920,0);
    r.errors+=!vertex.interaction_flag||vertex.fallback_flag;
    em::EmThinningConfig thinning{};
    thinning.enabled=1;thinning.threshold_GeV=1.;thinning.maximum_weight=100.;
    // Even when ordinary stochastic vertices are thinned, the already
    // weighted stopped parent must yield two full-weight photons.
    auto classification=phy::classifyLeptonFinalState({},thinning,vertex.record,20260920,0);
    r.errors+=classification.child_count!=2||!classification.annihilation_flag;
    auto f=phy::materializeLeptonFinalState(vertex.record,classification,2*i,3000000,
                                            em::TransportElectronMassGeV);
    r.errors+=f.error||!f.has_record||f.secondary_count!=2;
    auto const& a=f.secondaries[0];auto const& b=f.secondaries[1];
    r.errors+=a.pid!=22||b.pid!=22||a.weight!=7.||b.weight!=7.;
    r.errors+=a.energy_GeV!=em::TransportElectronMassGeV||b.energy_GeV!=a.energy_GeV;
    r.errors+=a.history_id!=3000000+2*i||b.history_id!=a.history_id+1;
    r.errors+=a.parent_history_id!=i+1||b.parent_history_id!=i+1;
    r.errors+=a.position_m[2]!=101.||a.time_s!=1.e-9;
    for(int k=0;k<3;++k)r.errors+=a.direction[k]+b.direction[k]!=0.;
    r.x=a.direction[0];r.y=a.direction[1];r.z=a.direction[2];
    r.xx=r.x*r.x;r.yy=r.y*r.y;r.zz=r.z*r.z;
    r.errors+=::fabs(r.xx+r.yy+r.zz-1.)>2.e-14;
    r.errors+=f.record.final_state_draw_id!=corsika::accelerator::em::AtRestAnnihilationPolarDrawId;
    em::tables::NativePhysicsView no_tables{};no_tables.em_transport_cut_MeV=1.;
    auto photon=phy::selectDiscreteInteraction(no_tables,a,0,20260920,0);
    r.errors+=photon.fallback_flag||photon.interaction.status!=em::EmInteractionStatus::ParticleCut;
    // Administrative time cut and electron cut must never turn into this vertex.
    cut.end.time_s=em::ParticleCutMaximumTimeS+1.e-9;cut.interaction={};
    phy::prepareStoppedAnnihilation(cut);
    r.errors+=cut.interaction.status==em::EmInteractionStatus::AtRestAnnihilation;
    cut.end.time_s=0.;cut.end.pid=11;cut.interaction={};
    phy::prepareStoppedAnnihilation(cut);
    r.errors+=cut.interaction.status==em::EmInteractionStatus::AtRestAnnihilation;
    results(i)=r;
  }
};
int main() {
  Kokkos::ScopeGuard guard(Kokkos::InitializationSettings{}.set_num_threads(4));
  constexpr std::size_t n=1000000;
  Kokkos::View<Result*> output("stopped_positron_oracle",n);
  Kokkos::parallel_for("stopped_positron_oracle",n,EndpointTest{output});
  auto h=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},output);
  std::uint64_t errors=0;double sums[6]{};
  for(std::size_t i=0;i<n;++i){auto const& r=h(i);errors+=r.errors;
    sums[0]+=r.x;sums[1]+=r.y;sums[2]+=r.z;sums[3]+=r.xx;sums[4]+=r.yy;sums[5]+=r.zz;}
  for(auto& s:sums)s/=n;
  for(int k=0;k<3;++k)if(std::abs(sums[k])>.005||std::abs(sums[k+3]-1./3.)>.005)++errors;
  std::cout<<Kokkos::DefaultExecutionSpace::name()<<" samples="<<n<<" errors="<<errors
           <<" mean_direction="<<sums[0]<<","<<sums[1]<<","<<sums[2]
           <<" second_moments="<<sums[3]<<","<<sums[4]<<","<<sums[5]<<'\n';
  if(errors)throw std::runtime_error("stopped positron oracle failed");
}

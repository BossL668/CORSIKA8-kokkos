/* (c) Copyright 2026 CORSIKA Project; BSD-3-Clause. */
#include "PhotonPairFallbackDriver.hpp"
#include <corsika/accelerator/em/common/ProcessCapabilities.hpp>
#include <corsika/accelerator/em/detail/PhotonFinalStateStep.hpp>
#include <corsika/accelerator/em/kokkos/KokkosProposalNativeTable.hpp>
#include <Kokkos_Core.hpp>

namespace corsika::accelerator::em::testing {
  using Exec = Kokkos::DefaultExecutionSpace;
  void initializePhotonPairFallbackTest() { Kokkos::initialize(); }
  void finalizePhotonPairFallbackTest() { Kokkos::finalize(); }

  std::vector<PhotonPairFallbackOutput> runPhotonPairFallbackTest(
      gpu::em::tables::ProposalNativeTableSet const& tables,
      gpu::em::tables::ProposalNativeAuxData const& aux,
      std::vector<PhotonPairFallbackInput> const& inputs) {
    kokkos_detail::KokkosProposalNativeTable<Exec> owner;
    owner.initialize(tables);
    gpu::em::tables::NativePhysicsView physics{};
    physics.proposal_native = owner.deviceView();
    auto lpm = aux.photon_pair_lpm;
    Kokkos::View<PhotonPairFallbackInput*, Exec> in("pair_inputs", inputs.size());
    auto hi = Kokkos::create_mirror_view(in);
    for (std::size_t i=0; i<inputs.size(); ++i) hi(i)=inputs[i];
    Kokkos::deep_copy(in, hi);
    Kokkos::View<PhotonPairFallbackOutput*, Exec> out("pair_outputs", inputs.size());
    Kokkos::parallel_for("production_photon_pair_fallback", Kokkos::RangePolicy<Exec>(0, inputs.size()),
      KOKKOS_LAMBDA(std::size_t i) {
        auto q=in(i); auto view=physics; auto snapshot=lpm;
        if (q.fault==1) view.proposal_native.dndx_column_count=0;
        if (q.fault==2) snapshot.component_count=0;
        gpu::em::EmInteractionRecord interaction{};
        interaction.status=gpu::em::EmInteractionStatus::Selected;
        interaction.process_id=gpu::em::PhotonPairProcessId;
        interaction.component_hash=q.component;
        interaction.energy_fraction=1.; interaction.mass_density_g_per_cm3=q.density;
        interaction.particle.pid=22; interaction.particle.energy_GeV=q.energy_MeV/1000.;
        interaction.particle.weight=1.; interaction.particle.history_id=q.history;
        interaction.particle.direction[0]=.6; interaction.particle.direction[2]=.8;
        detail::PhotonFinalStateClassification classification;
        if (q.attempts==0)
          classification=detail::classifyPhotonFinalState<0>(view,snapshot,{},interaction,26101117,71);
        else if (q.attempts==1)
          classification=detail::classifyPhotonFinalState<1>(view,snapshot,{},interaction,26101117,71);
        else
          classification=detail::classifyPhotonFinalState(view,snapshot,{},interaction,26101117,71);
        auto children=detail::materializePhotonFinalState(interaction,classification,0,100);
        PhotonPairFallbackOutput y{};
        y.x=classification.parameters.split_fraction;
        y.uniform=classification.parameters.split_uniform;
        y.draw=classification.parameters.split_draw_id;
        y.fallback=classification.fallback_flag; y.suppressed=classification.suppression_flag;
        y.children=children.secondary_count; y.error=children.error;
        for(unsigned j=0; j<children.secondary_count; ++j) {
          y.energy_sum_GeV+=children.secondaries[j].energy_GeV;
          double norm=0.;
          for(int k=0;k<3;++k)norm+=children.secondaries[j].direction[k]*children.secondaries[j].direction[k];
          y.norm_error=Kokkos::max(y.norm_error,Kokkos::abs(norm-1.));
        }
        out(i)=y;
      });
    auto ho=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},out);
    std::vector<PhotonPairFallbackOutput> result(inputs.size());
    for(std::size_t i=0;i<inputs.size();++i)result[i]=ho(i);
    return result;
  }
} // namespace corsika::accelerator::em::testing

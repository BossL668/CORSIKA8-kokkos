#include "LiveFinalStateDriver.hpp"
#include <Kokkos_Core.hpp>
#include <corsika/accelerator/em/detail/LeptonFinalStateStep.hpp>
#include <corsika/accelerator/em/detail/PhotonFinalStateStep.hpp>

namespace migration_audit {
std::vector<Output> evaluate(std::vector<Input> const& input) {
  using Exec = Kokkos::DefaultExecutionSpace;
  Kokkos::View<Input*, Exec> in("live_final_input", input.size());
  Kokkos::View<Output*, Exec> out("live_final_output", input.size());
  auto staging = Kokkos::create_mirror_view(in);
  for (std::size_t i=0; i<input.size(); ++i) staging(i)=input[i];
  Kokkos::deep_copy(in, staging);
  Kokkos::parallel_for("live_final_oracle", Kokkos::RangePolicy<Exec>(0,input.size()),
    KOKKOS_LAMBDA(std::size_t i) {
      namespace em = corsika::gpu::em;
      namespace impl = corsika::accelerator::em::detail;
      auto const& x=in(i);
      auto& y=out(i);
      if(x.interaction.particle.pid==22) {
        impl::PhotonFinalStateClassification c{};
        c.record_flag=1;
        c.parameters.process_id=x.interaction.process_id;
        c.parameters.split_fraction=x.interaction.energy_fraction;
        c.parameters.azimuth_uniform=x.azimuth_uniform;
        c.parameters.thinning_keep_mask=3;
        c.parameters.thinning_first_weight=c.parameters.thinning_second_weight=1.;
        auto r=impl::materializePhotonFinalState(x.interaction,c,0,100);
        y.error=r.error; y.count=r.secondary_count;
        for(unsigned j=0;j<r.secondary_count;++j) y.children[j]=r.secondaries[j];
      } else {
        impl::LeptonFinalStateClassification c{};
        c.record_flag=1;
        c.parameters.process_id=x.interaction.process_id;
        c.parameters.azimuth_uniform=x.azimuth_uniform;
        c.parameters.thinning_keep_mask=3;
        c.parameters.thinning_first_weight=c.parameters.thinning_second_weight=1.;
        if(x.interaction.process_id==em::AnnihilationProcessId &&
           !impl::sampleAnnihilationRho(x.interaction.particle.energy_GeV,
              x.interaction.particle_mass_GeV,x.rho_uniform,
              c.parameters.energy_split_fraction)) { y.error=100; return; }
        auto r=impl::materializeLeptonFinalState(x.interaction,c,0,100,em::ElectronMassGeV);
        y.error=r.error; y.count=r.secondary_count;
        for(unsigned j=0;j<r.secondary_count;++j) y.children[j]=r.secondaries[j];
      }
    });
  auto host=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),out);
  std::vector<Output> result(input.size());
  for(std::size_t i=0;i<input.size();++i) result[i]=host(i);
  return result;
}
std::vector<RhoOutput> evaluateRho(corsika::gpu::em::BremsLpmSnapshot const& snapshot,
    double energy_MeV, double v, unsigned samples, unsigned group) {
  using Exec = Kokkos::DefaultExecutionSpace;
  Kokkos::View<RhoOutput*, Exec> out("live_epair_rho", samples);
  auto const physics = snapshot;
  Kokkos::parallel_for("live_epair_distribution", Kokkos::RangePolicy<Exec>(0, samples),
    KOKKOS_LAMBDA(unsigned i) {
      namespace em = corsika::gpu::em;
      em::RandomNumberKey key{2026090907, group, i + 1ULL, 0,
          static_cast<std::uint32_t>(em::ElectronPairProcessId), 0};
      auto r = em::sampleEpairRhoRejection(physics, physics.components[0].component_hash,
          energy_MeV, v, em::uniformOpen01(key), key);
      out(i) = {r.sample.rho, static_cast<unsigned>(r.sample.status), r.trial_count};
    });
  auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), out);
  std::vector<RhoOutput> result(samples);
  for(unsigned i=0; i<samples; ++i) result[i]=host(i);
  return result;
}
}

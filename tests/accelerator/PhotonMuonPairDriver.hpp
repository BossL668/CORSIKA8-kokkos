/* (c) Copyright 2026 CORSIKA Project; BSD-3-Clause. */
#pragma once
#include <corsika/accelerator/em/common/tables/ProposalNativeTable.hpp>
#include <corsika/accelerator/em/common/Types.hpp>
#include <vector>
namespace photon_muon_test {
namespace em = corsika::gpu::em;
struct Input {
  double energy_MeV{}, quantile{.5};
  std::uint64_t component{}, history{};
  unsigned fault{}, thinning{};
};
struct Output {
  em::EmParticleState children[2]{};
  em::ProjectedEmStepRecord cuts[2]{};
  unsigned cut_status[2]{};
  double x{}, uniform{}, azimuth{}, probe{}, public_v{}, mass_correction{};
  unsigned count{}, fallback{}, error{}, probe_status{}, public_status{}, mask{}, reason{}, diagnostic{};
};
struct Result { std::vector<Output> outputs; double kernel_s{}; };
void initialize();
void finalize();
Result run(em::tables::ProposalNativeTableSet const&, std::vector<Input> const&, bool device);
void checkBatch(em::tables::ProposalNativeTableSet const&, std::vector<Input> const&);
em::GpuProfileResult checkCutProfile(std::vector<em::EmParticleState> const&);
double nativeCutReference(em::EmParticleState const&, em::ProjectedEmStepRecord const&);
void checkCutOutput(em::GpuProfileResult const&);
}

/* (c) Copyright 2026 CORSIKA Project; BSD-3-Clause. */
#pragma once

#include <corsika/accelerator/em/common/tables/ProposalNativeTable.hpp>
#include <corsika/accelerator/em/common/tables/ProposalNativeAux.hpp>
#include <vector>

namespace corsika::accelerator::em::testing {
  struct PhotonPairFallbackInput {
    double energy_MeV{}, density{1.e-3};
    std::uint64_t component{}, history{};
    unsigned attempts{64}, fault{};
  };
  struct PhotonPairFallbackOutput {
    double x{}, uniform{}, energy_sum_GeV{}, norm_error{};
    unsigned draw{}, fallback{}, suppressed{}, children{}, error{};
  };
  void initializePhotonPairFallbackTest();
  void finalizePhotonPairFallbackTest();
  std::vector<PhotonPairFallbackOutput> runPhotonPairFallbackTest(
      gpu::em::tables::ProposalNativeTableSet const&,
      gpu::em::tables::ProposalNativeAuxData const&,
      std::vector<PhotonPairFallbackInput> const&);
} // namespace corsika::accelerator::em::testing

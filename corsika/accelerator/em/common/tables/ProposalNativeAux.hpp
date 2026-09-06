/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <cstdint>
#include <filesystem>
#include <vector>

#include <corsika/accelerator/em/common/BremsLpm.hpp>
#include <corsika/accelerator/em/common/MoliereScattering.hpp>
#include <corsika/accelerator/em/common/PhotonPairLpm.hpp>
#include <corsika/accelerator/em/common/tables/Sha256.hpp>

// Keep the host-only PROPOSAL calculator view out of headers compiled by
// nvcc.  A declaration is sufficient because this interface only accepts a
// reference to a vector; ProposalNativeAux.cpp includes the complete type.
namespace corsika::proposal {
  struct NativeInteractionCalculatorView;
}

namespace corsika::gpu::em::tables {

  inline constexpr std::uint32_t ProposalNativeAuxFormatVersion = 3;
  inline constexpr char ProposalNativeAuxAlgorithmVersion[] =
      "proposal-native-aux-v3-epair-device-rejection-v1";

  /**
   * Small auxiliary state that is not part of the native PROPOSAL spline
   * export. Epair rho deliberately remains on beta4's bounded device rejection
   * sampler because the PROPOSAL KKP oracle has isolated zero-density branches
   * that cannot satisfy a uniform 1e-3 interpolation bound.
   */
  struct ProposalNativeAuxData {
    std::uint32_t format_version{ProposalNativeAuxFormatVersion};
    PhotonPairLpmSnapshot photon_pair_lpm{};
    BremsLpmSnapshot brems_lpm{};
    MoliereSnapshot electron_moliere{};
    MoliereSnapshot muon_moliere{};
    std::uint32_t has_muon_moliere{};
    Sha256Digest key_hash{};
    Sha256Digest content_hash{};
    bool cache_hit{};
    std::filesystem::path cache_file{};
  };

  std::filesystem::path defaultProposalNativeAuxCacheDirectory();

  ProposalNativeAuxData loadOrCreateProposalNativeAux(
      std::vector<proposal::NativeInteractionCalculatorView> const&,
      std::filesystem::path const& cache_directory = {});

} // namespace corsika::gpu::em::tables

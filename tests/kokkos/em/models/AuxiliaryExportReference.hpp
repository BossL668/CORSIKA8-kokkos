/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once
#include "AuxiliaryModelReference.hpp"
#include <corsika/modules/proposal/NativeCalculatorView.hpp>
#include <corsika/kokkos/utility/Sha256.hpp>
#include <filesystem>

namespace corsika::kokkos::test::auxiliary_reference::tables {
  using Sha256Digest = corsika::kokkos::sha256_digest_type;
  inline constexpr std::uint32_t ProposalNativeAuxFormatVersion = 4;
  inline constexpr char ProposalNativeAuxAlgorithmVersion[] =
      "proposal-native-aux-v4-muon-brems-v1";
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
  ProposalNativeAuxData load_reference_aux(
      std::vector<corsika::proposal::NativeInteractionCalculatorView> const& views,
      std::filesystem::path const& directory);
} // namespace corsika::kokkos::test::auxiliary_reference::tables

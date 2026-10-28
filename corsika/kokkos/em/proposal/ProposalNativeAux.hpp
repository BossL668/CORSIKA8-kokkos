/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/kokkos/em/models/BremsLpm.hpp>
#include <corsika/kokkos/em/models/MoliereSnapshot.hpp>
#include <corsika/kokkos/em/models/PhotonPairLpm.hpp>
#include <corsika/kokkos/utility/Sha256.hpp>

#include <cstdint>
#include <filesystem>
#include <vector>

namespace corsika::proposal {
  class NativeInteractionCalculatorView;
} // namespace corsika::proposal

namespace corsika::kokkos {

  inline constexpr std::uint32_t ProposalNativeAuxFormatVersion = 4;
  inline constexpr char ProposalNativeAuxAlgorithmVersion[] =
      "proposal-native-aux-v4-muon-brems-v1";

  /**
   * Host-owned auxiliary state copied from existing PROPOSAL calculators.
   *
   * The numerical snapshots own their component and coefficient arrays and remain
   * valid after the source calculators are destroyed. Cache diagnostics are host-only.
   * Direct-pair energy-asymmetry data are not included; those final states retain
   * their CPU path. The version-four cache preserves the production byte layout
   * and is not a portable interchange format across ABIs or byte orders.
   */
  struct ProposalNativeAuxData {
    std::uint32_t formatVersion_{ProposalNativeAuxFormatVersion};
    PhotonPairLpmSnapshot photonPairLpm_{};
    BremsLpmSnapshot bremsLpm_{};
    MoliereSnapshot electronMoliere_{};
    MoliereSnapshot muonMoliere_{};
    std::uint32_t hasMuonMoliere_{};
    sha256_digest_type keyHash_{};
    sha256_digest_type contentHash_{};
    bool cacheHit_{};
    std::filesystem::path cacheFile_{};
  };

  /**
   * Resolve the existing auxiliary cache directory from XDG_CACHE_HOME or HOME.
   *
   * A nonempty environment value must be absolute. The production directory name
   * is retained so compatible version-four caches can be read without migration.
   */
  [[nodiscard]] std::filesystem::path default_proposal_native_aux_cache_directory();

  /**
   * Load a validated cache, or export and atomically publish auxiliary model state.
   *
   * Borrowed calculator references must remain alive throughout this host call.
   * Supported projectile and medium identities determine the key. A valid cache
   * hit precedes cold-generation checks on LPM calculators and material composition.
   * An empty directory selects the default location; a supplied path retains the
   * production path semantics. Corrupt cache reads are rebuilt under the existing
   * per-key lock; generation, locking and I/O failures retain their exceptions.
   */
  [[nodiscard]] ProposalNativeAuxData load_or_create_proposal_native_aux(
      std::vector<proposal::NativeInteractionCalculatorView> const& views,
      std::filesystem::path const& directory = {});

} // namespace corsika::kokkos

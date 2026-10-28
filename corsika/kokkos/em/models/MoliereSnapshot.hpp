/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/framework/core/HostDevice.hpp>

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace corsika::kokkos {

  inline constexpr std::size_t MaxMoliereComponents = 16;
  inline constexpr std::size_t MoliereSeriesCoefficientCount = 70;
  inline constexpr std::size_t MoliereLargeSeriesCoefficientCount = 13;
  inline constexpr std::size_t MoliereLargeIntegralCoefficientCount = 15;

  /** Static screening factors for one component of the exported medium. */
  struct MoliereComponentSnapshot {
    double nuclearCharge_{};
    double weightZz_{};
    // Static screening factors derived once from the versioned medium.
    double chi0SquaredMeV2_{};
    double coulombCorrection_{};
  };

  /** Pointer-free state copied or derived from the linked PROPOSAL Moliere model.
   * Coefficients remain versioned table data, not a second parametrization.
   * Member order and initialization retain the production auxiliary-cache layout. */
  struct MoliereSnapshot {
    double particleMassMeV_{};
    double electronMassMeV_{};
    double fineStructureConstant_{};
    double avogadroPerMol_{};
    double hbarMeVS_{};
    double speedOfLightCmPerS_{};
    double zSquaredOverAAverage_{};
    double inverseWeightZzSum_{};
    double eulerMascheroni_{};
    std::uint32_t componentCount_{};
    std::uint32_t maximumWeightIndex_{};
    MoliereComponentSnapshot components_[MaxMoliereComponents]{};
    double c1_[MoliereSeriesCoefficientCount]{};
    double c2_[MoliereSeriesCoefficientCount]{};
    double c2Large_[MoliereLargeSeriesCoefficientCount]{};
    double s2Large_[MoliereLargeSeriesCoefficientCount]{};
    double c1IntegralLarge_[MoliereLargeIntegralCoefficientCount]{};
  };

  static_assert(std::is_standard_layout_v<MoliereSnapshot>);
  static_assert(std::is_trivially_copyable_v<MoliereSnapshot>);

  namespace moliere_snapshot_detail {
    inline constexpr double Pi = 3.141592653589793238462643383279502884;
  } // namespace moliere_snapshot_detail

  /**
   * Check the production snapshot constraints without changing any input.
   * Only active components are inspected; all coefficient-array entries are checked.
   * This predicate does not evaluate or sample the scattering distribution.
   */
  [[nodiscard]] CORSIKA_HOST_DEVICE inline bool valid_moliere_snapshot(
      MoliereSnapshot const& snapshot);

} // namespace corsika::kokkos

#include <corsika/detail/kokkos/em/models/MoliereSnapshot.inl>

/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <cstdint>
#include <type_traits>

namespace corsika::kokkos {

  /// Production status of a magnetic step calculation, with unchanged values.
  enum class MagneticStepStatus : std::uint32_t {
    Success = 0,
    Linear = 1,
    InvalidInput = 2,
    NonFiniteResult = 3,
  };

  /**
   * Value-only result supplied by a magnetic step-limit calculation.
   *
   * Preserves the production field order, defaults and reserved word. Distances
   * are in metres; a Linear result can carry positive infinity for the distance
   * and gyroradius. This record does not compute a limit or validate its inputs.
   * Callers must handle InvalidInput and NonFiniteResult before using it in a
   * trajectory or boundary query. Padding is not a portable serialization format.
   */
  struct MagneticStepLimitResult {
    MagneticStepStatus status_{MagneticStepStatus::InvalidInput};
    std::uint32_t reserved_{};
    double distanceM_{};
    double gyroradiusM_{};
  };

  static_assert(std::is_standard_layout_v<MagneticStepLimitResult>);
  static_assert(std::is_trivially_copyable_v<MagneticStepLimitResult>);

} // namespace corsika::kokkos

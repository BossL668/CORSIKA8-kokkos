/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <cstddef>
#include <type_traits>
namespace corsika::kokkos {
  /** Borrowed production shower-axis support data. Positions/spacing use metres,
   * support values use g/cm^2 and the axis direction is dimensionless.
   * The owner must validate finite inputs, positive spacing, a unit axis and nonempty
   * support. A nonnegative projected support index must be representable by size_t.
   * The owner must keep device-addressable support alive until projection kernels finish.
   * This record performs no geometry search or density integration.
   */
  struct ProfileProjectionView {
    double axisStartPositionM_[3]{};
    double axisDirection_[3]{};
    double axisStepLengthM_{};
    double const* axisGrammageGPerCm2_{};
    std::size_t axisSupportCount_{};
  };
  static_assert(std::is_standard_layout_v<ProfileProjectionView>);
  static_assert(std::is_trivially_copyable_v<ProfileProjectionView>);
} // namespace corsika::kokkos

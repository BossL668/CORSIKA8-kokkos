/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <Kokkos_Core.hpp>

#include <cstddef>
#include <stdexcept>

#include <corsika/gpu/em/Types.hpp>
#include <corsika/gpu/em/detail/ProfileProjectionData.hpp>

namespace corsika::accelerator::em::kokkos_detail {

  /** Own the shower-axis support in the selected Kokkos memory space. */
  template <class ExecutionSpace>
  class KokkosProfileProjection {
  public:
    using memory_space = typename ExecutionSpace::memory_space;

    void initialize(
        gpu::em::GpuEmConfig::ProfileProjection const& source,
        ExecutionSpace const& execution = {}) {
      if (!source.enabled) return;
      if (!(source.axis_step_length_m > 0.) ||
          source.axis_grammage_g_per_cm2.size() < 2)
        throw std::invalid_argument(
            "Kokkos profile projection requires a valid ShowerAxis support");
      axis_ = Kokkos::View<double*, memory_space>(
          "c8_kokkos_profile_axis", source.axis_grammage_g_per_cm2.size());
      auto host = Kokkos::create_mirror_view(axis_);
      for (std::size_t i = 0; i < source.axis_grammage_g_per_cm2.size(); ++i)
        host(i) = source.axis_grammage_g_per_cm2[i];
      Kokkos::deep_copy(execution, axis_, host);
      for (int component = 0; component < 3; ++component) {
        view_.axis_start_position_m[component] =
            source.axis_start_position_m[component];
        view_.axis_direction[component] = source.axis_direction[component];
      }
      view_.axis_step_length_m = source.axis_step_length_m;
      view_.axis_grammage_g_per_cm2 = axis_.data();
      view_.axis_support_count = source.axis_grammage_g_per_cm2.size();
      enabled_ = true;
    }

    bool enabled() const noexcept { return enabled_; }
    gpu::em::detail::DeviceProfileProjection deviceView() const noexcept {
      return view_;
    }
    std::size_t deviceBytes() const noexcept {
      return axis_.extent(0) * sizeof(double);
    }

  private:
    Kokkos::View<double*, memory_space> axis_{};
    gpu::em::detail::DeviceProfileProjection view_{};
    bool enabled_{};
  };

} // namespace corsika::accelerator::em::kokkos_detail

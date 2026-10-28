/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>
#include <Kokkos_Core.hpp>
#include <corsika/kokkos/runtime/CheckedMemory.hpp>
#include <corsika/kokkos/output/ProfileProjectionConfig.hpp>
#include <corsika/kokkos/output/ProfileProjectionView.hpp>

namespace corsika::kokkos {

  /** Own copied shower-axis support in the selected execution space.
   * Configuration distances use metres, grammages use g/cm^2 and the direction
   * is dimensionless. The caller supplies validated finite axis data. Keep this
   * owner alive and unchanged until kernels using deviceView() have finished.
   */
  template <typename TExecutionSpace>
  class ProfileProjectionStorage {
  public:
    using memory_space_type = typename TExecutionSpace::memory_space;

    static std::size_t projectedDeviceBytes(ProfileProjectionConfig const& source);

    void initialize(ProfileProjectionConfig const& source,
                    TExecutionSpace const& execution = {});

    bool enabled() const noexcept;
    /** Borrow the uploaded support; this record does not own the axis array. */
    ProfileProjectionView deviceView() const noexcept;
    std::size_t deviceBytes() const noexcept;

  private:
    Kokkos::View<double*, memory_space_type> axis_{};
    ProfileProjectionView view_{};
    bool enabled_{};
  };

} // namespace corsika::kokkos

#include <corsika/detail/kokkos/output/ProfileProjectionStorage.inl>

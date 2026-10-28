/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <Kokkos_Core.hpp>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>
#include <corsika/kokkos/data/ParticleState.hpp>
#include <corsika/kokkos/runtime/RawView.hpp>
#include <corsika/kokkos/runtime/CheckedMemory.hpp>

namespace corsika::kokkos {

  struct ParticleSoARawView {
    std::int32_t* pid_{};
    std::int32_t* mediumId_{};
    std::int32_t* environmentNode_{};
    std::uint32_t* generation_{};
    std::uint32_t* reserved_{};
    double* energyGeV_{};
    RawView2D<double> positionM_{};
    RawView2D<double> direction_{};
    double* timeS_{};
    double* weight_{};
    std::uint64_t* historyId_{};
    std::uint64_t* parentHistoryId_{};
    std::uint64_t* stepId_{};

    KOKKOS_INLINE_FUNCTION ParticleState load(std::size_t const index) const;

    KOKKOS_INLINE_FUNCTION void store(std::size_t const index,
                                      ParticleState const& state) const;
  };
  template <class TMemorySpace>
  struct ParticleSoA {
    using execution_space_type = typename TMemorySpace::execution_space;

    Kokkos::View<std::int32_t*, TMemorySpace> pid_;
    Kokkos::View<std::int32_t*, TMemorySpace> mediumId_;
    Kokkos::View<std::int32_t*, TMemorySpace> environmentNode_;
    Kokkos::View<std::uint32_t*, TMemorySpace> generation_;
    Kokkos::View<std::uint32_t*, TMemorySpace> reserved_;
    Kokkos::View<double*, TMemorySpace> energyGeV_;
    Kokkos::View<double * [3], TMemorySpace> positionM_;
    Kokkos::View<double * [3], TMemorySpace> direction_;
    Kokkos::View<double*, TMemorySpace> timeS_;
    Kokkos::View<double*, TMemorySpace> weight_;
    Kokkos::View<std::uint64_t*, TMemorySpace> historyId_;
    Kokkos::View<std::uint64_t*, TMemorySpace> parentHistoryId_;
    Kokkos::View<std::uint64_t*, TMemorySpace> stepId_;

    ParticleSoA() = default;

    static std::size_t deviceBytesForCapacity(std::size_t const capacity);

    ParticleSoA(std::string const& label, std::size_t const capacity);

    ParticleSoA(std::string const& label, std::size_t const capacity,
                execution_space_type const& execution);

    KOKKOS_INLINE_FUNCTION ParticleState load(std::size_t const index) const;

    KOKKOS_INLINE_FUNCTION void store(std::size_t const index,
                                      ParticleState const& state) const;

    ParticleSoARawView rawDeviceView() const noexcept;

    std::size_t deviceBytes() const noexcept;
  };
  static_assert(std::is_trivially_copyable_v<ParticleSoARawView>);
  static_assert(std::is_standard_layout_v<ParticleSoARawView>);
  static_assert(sizeof(ParticleSoARawView) <= 160);

} // namespace corsika::kokkos

#include <corsika/detail/kokkos/data/ParticleSoA.inl>

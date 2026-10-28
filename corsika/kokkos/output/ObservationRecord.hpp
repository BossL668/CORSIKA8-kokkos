/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/kokkos/data/ParticleState.hpp>
#include <cstdint>
#include <type_traits>

namespace corsika::kokkos {
  /** Production particle-at-surface or escaped-environment output. */
  enum class ObservationStatus : std::int32_t {
    ReachedObservationSurface = 0,
    EscapedEnvironment = 1,
  };
  struct ObservationRecord {
    ParticleState particle_{};
    ObservationStatus status_{ObservationStatus::ReachedObservationSurface};
    std::int32_t reserved_{};
  };
  static_assert(std::is_standard_layout_v<ObservationRecord>);
  static_assert(std::is_trivially_copyable_v<ObservationRecord>);
} // namespace corsika::kokkos

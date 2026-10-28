/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/kokkos/environment/EnvironmentQuery.hpp>
#include <corsika/kokkos/environment/ObservationPlane.hpp>
#include <cmath>
#include <type_traits>

namespace corsika::kokkos {

  /** Per-step geometry configuration; the environment tree remains a borrowed view.
   * All coordinates use the snapshot's common system and lengths are metres.
   * earthCenterM_ is only the origin of the retained observation-radius diagnostic,
   * not a spherical-layer geometry definition. The caller supplies the observation
   * plane; a default zero normal is invalid and is never silently normalized.
   */
  struct TransportGeometryConfig {
    EnvironmentQueryConfig query_{};
    ObservationPlane observation_{};
    double earthCenterM_[3]{};
  };

  /** Optional non-absorbing straight-track material interface supplied by a caller.
   * It competes with tree boundaries and observation without changing the logical
   * node. The caller owns subsequent material routing. This does not provide an
   * alternative environment or a curved terrain-intersection implementation.
   * No legacy sphere-layer overflow option is needed in the NodeTree domain.
   */
  struct ExternalTransportBoundary {
    double distanceM_{HUGE_VAL};
    bool enabled_{};
    bool disableObservation_{};
  };
  static_assert(std::is_trivially_copyable_v<TransportGeometryConfig>);
  static_assert(std::is_trivially_copyable_v<ExternalTransportBoundary>);

} // namespace corsika::kokkos

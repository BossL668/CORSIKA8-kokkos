/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */
#pragma once

namespace corsika::kokkos {

  /** Production coordinator: one driver packet, coordinator-owned callbacks.
   * Endpoint instances and borrowed runtime leases outlive this pump.
   * The full private template implementation is in the central detail tree.
   */
  template <typename TEndpoint>
  class IndependentSubshowerPump;

} // namespace corsika::kokkos
#include <corsika/detail/kokkos/scheduling/IndependentSubshowerPump.inl>

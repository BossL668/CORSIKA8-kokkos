/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/framework/geometry/BoundaryResult.hpp>

#include <corsika/framework/core/HostDevice.hpp>

namespace corsika::kokkos {

  /**
   * Compile the shared ordered boundary search for numerical Kokkos accessors.
   *
   * This uses exactly the same implementation body as the native host entry
   * corsika::next_environment_boundary, including order, ties and clamping.
   * Access, node handles and parameter arithmetic must be device callable when
   * used in a kernel. Native units and host node pointers use the host entry.
   * The access object's geometry must remain alive and accessible throughout
   * the search; valid indices and supported shapes are caller preconditions.
   * Any accumulated shape-error flags must be checked on access after return.
   *
   * \param access Supplies intersections and ordered topology access methods.
   * \param current Current logical node handle.
   * \param limit Maximum trajectory parameter, in the intersection units.
   * \return Selected parameter and next logical node.
   */
  template <typename TAccess, typename TNode, typename TParameter>
  [[nodiscard]] CORSIKA_HOST_DEVICE inline BoundaryResult<TParameter, TNode>
  next_environment_boundary(TAccess const& access, TNode current, TParameter limit);

} // namespace corsika::kokkos

#include <corsika/detail/kokkos/environment/NextEnvironmentBoundary.inl>

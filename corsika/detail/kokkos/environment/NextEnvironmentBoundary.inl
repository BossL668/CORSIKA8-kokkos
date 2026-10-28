/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

namespace corsika::kokkos {

  template <typename TAccess, typename TNode, typename TParameter>
  CORSIKA_HOST_DEVICE inline BoundaryResult<TParameter, TNode> next_environment_boundary(
      TAccess const& access, TNode current, TParameter limit) {
#include <corsika/detail/framework/geometry/NextEnvironmentBoundaryBody.inl>
  }

} // namespace corsika::kokkos

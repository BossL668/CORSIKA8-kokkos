/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/framework/core/HostDevice.hpp>

namespace corsika {

  /**
   * Selected trajectory parameter and next logical environment node.
   *
   * The parameter has the same units as the supplied limit and intersections:
   * native tracking uses time, while numerical queries may also use distance.
   * A node handle is borrowed, not owned. Native trees use their node pointer;
   * numerical snapshots use an index. Keep the referenced tree alive.
   */
  template <typename TParameter, typename TNode>
  class BoundaryResult {
  public:
    /// Store a selected parameter and borrowed node handle without conversion.
    CORSIKA_HOST_DEVICE BoundaryResult(TParameter const& parameter, TNode const& node);

    /// Trajectory parameter selected by the ordered boundary search.
    [[nodiscard]] CORSIKA_HOST_DEVICE TParameter getParameter() const;

    /// Next logical node, or the input node when the step limit wins.
    [[nodiscard]] CORSIKA_HOST_DEVICE TNode getNode() const;

  private:
    TParameter parameter_;
    TNode node_;
  };

} // namespace corsika

#include <corsika/detail/framework/geometry/BoundaryResult.inl>

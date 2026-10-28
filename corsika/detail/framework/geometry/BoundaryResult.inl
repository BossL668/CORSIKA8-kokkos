/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

namespace corsika {

  template <typename TParameter, typename TNode>
  CORSIKA_HOST_DEVICE inline BoundaryResult<TParameter, TNode>::BoundaryResult(
      TParameter const& parameter, TNode const& node)
      : parameter_(parameter)
      , node_(node) {}

  template <typename TParameter, typename TNode>
  CORSIKA_HOST_DEVICE inline TParameter BoundaryResult<TParameter, TNode>::getParameter()
      const {
    return parameter_;
  }

  template <typename TParameter, typename TNode>
  CORSIKA_HOST_DEVICE inline TNode BoundaryResult<TParameter, TNode>::getNode() const {
    return node_;
  }

} // namespace corsika

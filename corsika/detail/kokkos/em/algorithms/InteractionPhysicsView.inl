/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

namespace corsika::kokkos {

  CORSIKA_HOST_DEVICE inline InteractionPhysicsView::InteractionPhysicsView() = default;
  CORSIKA_HOST_DEVICE inline InteractionPhysicsView::InteractionPhysicsView(
      ContinuousPhysicsView const& continuousView, std::uint32_t physicsSource)
      : continuousView_(continuousView)
      , physicsSource_(physicsSource) {}
  CORSIKA_HOST_DEVICE inline ContinuousPhysicsView const&
  InteractionPhysicsView::getContinuousView() const {
    return continuousView_;
  }
  CORSIKA_HOST_DEVICE inline std::uint32_t InteractionPhysicsView::getPhysicsSource()
      const {
    return physicsSource_;
  }

} // namespace corsika::kokkos

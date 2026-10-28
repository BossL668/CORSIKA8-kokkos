/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/kokkos/em/tables/ContinuousPhysicsQueries.hpp>

namespace corsika::kokkos {

  /**
   * Borrowed tables, cuts and original diagnostic-source tag for selection.
   * Owns no arrays. The tables must outlive every host/device selection using this
   * view. physicsSource is copied without validation: the production value 1 enables
   * native inverse-failure counting, but does not choose a different physics model.
   */
  class InteractionPhysicsView {
  public:
    CORSIKA_HOST_DEVICE InteractionPhysicsView();
    CORSIKA_HOST_DEVICE explicit InteractionPhysicsView(
        ContinuousPhysicsView const& continuousView, std::uint32_t physicsSource = 1u);
    [[nodiscard]] CORSIKA_HOST_DEVICE ContinuousPhysicsView const& getContinuousView()
        const;
    [[nodiscard]] CORSIKA_HOST_DEVICE std::uint32_t getPhysicsSource() const;

  private:
    ContinuousPhysicsView continuousView_{};
    std::uint32_t physicsSource_{1u};
  };
  static_assert(std::is_trivially_copyable_v<InteractionPhysicsView>);

} // namespace corsika::kokkos

#include <corsika/detail/kokkos/em/algorithms/InteractionPhysicsView.inl>

/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once
#include <corsika/framework/core/PhysicalUnits.hpp>
#include <corsika/kokkos/em/PhysicsRequirements.hpp>
#include <corsika/kokkos/em/execution/BackendConfig.hpp>
#include <corsika/kokkos/environment/EnvironmentSnapshotData.hpp>
#include <corsika/kokkos/environment/TransportGeometryConfig.hpp>
#include <memory>
namespace corsika::kokkos::application {
  /** Prepared host data for the production air runner.
   * Numeric views borrow environmentStorage_; copies share the same owner.
   * The original native environment and its model assignments must also stay
   * alive and unchanged until transport, fallback and output have completed.
   */
  template <typename TModelProperties>
  struct PreparedCascade {
    std::shared_ptr<EnvironmentSnapshot<TModelProperties> const> environmentStorage_;
    EnvironmentDataView environmentData_{};
    TransportGeometryConfig geometry_{};
    BackendConfig backendConfig_;
    PhysicsRequirements physicsRequirements_;
    HEPEnergyType proposalStochasticCut_{};
    bool radioEnabled_{};
  };

} // namespace corsika::kokkos::application

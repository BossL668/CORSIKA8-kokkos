/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/kokkos/scheduling/HadronicBatchProtocol.hpp>
#include <corsika/framework/geometry/CoordinateSystem.hpp>
#include <corsika/modules/fluka/InteractionModel.hpp>
#include <vector>

namespace corsika::kokkos {
  struct FlukaWorkerResult {
    HadronicInteractionResultHeader header_{};
    std::vector<HadronicSecondaryRecord> secondaries_;
  };

  /** Execute one production keyed request without changing its RNG or kinematics.
   * The caller initializes FLUKA once in this process using the environment's
   * target elements. This function does not initialize Kokkos or perform cuts.
   */
  FlukaWorkerResult execute_fluka_request(fluka::InteractionModel& model,
                                          HadronicInteractionRequest const& request,
                                          CoordinateSystemPtr const& coordinateSystem);

  /** Serve the production binary protocol until clean EOF; malformed streams throw.
   * The caller owns the descriptor and the process-local model. No target set is
   * hard-coded here, and no application entry point is provided by this header.
   */
  int serve_fluka_worker(fluka::InteractionModel& model,
                         CoordinateSystemPtr const& coordinateSystem,
                         int serverDescriptor, std::size_t workerId);
} // namespace corsika::kokkos

#include <corsika/detail/kokkos/scheduling/FlukaWorker.inl>

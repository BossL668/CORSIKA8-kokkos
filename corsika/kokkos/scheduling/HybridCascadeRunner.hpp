/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/framework/core/Logging.hpp>
#include <corsika/kokkos/em/ProcessSequenceCompatibility.hpp>
#include <corsika/kokkos/em/proposal/ProposalCpuFallbackHandler.hpp>
#include <corsika/kokkos/environment/EnvironmentBorrowedView.hpp>
#include <corsika/kokkos/environment/EnvironmentSnapshot.hpp>
#include <corsika/kokkos/environment/EnvironmentSnapshotData.hpp>
#include <corsika/kokkos/environment/TransportGeometryConfig.hpp>
#include <corsika/kokkos/scheduling/EmRouter.hpp>
#include <corsika/kokkos/scheduling/HybridCascade.hpp>

#ifdef CORSIKA_KOKKOS_STATIC_MULTIGPU
#include <corsika/kokkos/scheduling/StaticFrontier.hpp>
#endif

#include <stdexcept>
#include <type_traits>
#include <utility>

namespace corsika::kokkos {

  struct HybridRunOptions {
    bool retainRecords_{};
    bool failOnUnexpectedFallback_{true};
  };

  /**
   * Run the production hybrid cascade with caller-owned models and sequence.
   *
   * The typed fallback factory receives this snapshot's original topology.
   * A legacy no-argument factory remains supported for handlers with a matching
   * setEnvironmentTree() member. The standard ProposalCpuFallbackHandler binds
   * the topology in its constructor and requires the typed factory.
   *
   * Snapshot arrays, the native environment and all supplied models must outlive
   * this call. Completion runs while every typed transport object is alive.
   * No process sequence is assembled or reordered by this helper.
   */
  template <typename TProcessRegistry, typename TModelProperties, typename TEnvironment,
            typename TTracking, typename TSequence, typename TOutput, typename TStack,
            typename TBackendFactory, typename TFallbackFactory,
            typename TOutputSinkFactory, typename TConfigureCascade, typename TOnComplete>
  void run_hybrid_cascade(
      CoordinateSystemPtr const& coordinateSystem,
      EnvironmentSnapshot<TModelProperties> const& environmentSnapshot,
      TransportGeometryConfig const& geometry, TEnvironment& environment,
      TTracking& tracking, TSequence& sequence, TOutput& output, TStack& stack,
      HybridRunOptions const& options, TBackendFactory&& backendFactory,
      TFallbackFactory&& fallbackFactory, TOutputSinkFactory&& outputSinkFactory,
      TConfigureCascade&& configureCascade, TOnComplete&& onComplete);

  /// Forwarding entry corresponding to the original native-CUDA call spelling.
  /// Uses the new typed snapshot/geometry signature, not the old FullView ABI.
  template <typename TProcessRegistry, typename... TArguments>
  void run_cuda_hybrid_cascade(TArguments&&... arguments);

} // namespace corsika::kokkos

#include <corsika/detail/kokkos/scheduling/HybridCascadeRunner.inl>

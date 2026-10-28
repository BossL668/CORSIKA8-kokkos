/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */
#pragma once
#include <Kokkos_Core.hpp>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <vector>
#include <corsika/kokkos/em/execution/BatchResults.hpp>
#include <corsika/kokkos/em/algorithms/PhotonTransportStep.hpp>
#include <corsika/kokkos/environment/EnvironmentBorrowedView.hpp>

namespace corsika::kokkos {

  template <class TExecutionSpace>
  PhotonTransportBatchResult transport_photons_batch(
      EnvironmentBorrowedView const environment, TransportGeometryConfig const geometry,
      std::vector<InteractionRecord> const& input, TExecutionSpace const& execution = {});
}
#include <corsika/detail/kokkos/em/execution/PhotonTransportBatch.inl>

/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */
#pragma once
#include <corsika/kokkos/em/execution/IEmBackend.hpp>
#include <corsika/kokkos/em/execution/BatchResults.hpp>
#include <corsika/kokkos/output/CorsikaOutputSink.hpp>
#include <corsika/kokkos/em/models/PhotonPairKinematics.hpp>
#include <corsika/kokkos/em/ProcessCapabilities.hpp>
#include <corsika/kokkos/em/proposal/ProposalFallbackAdapter.hpp>
#include <corsika/kokkos/data/RouterParticleConversion.hpp>
#include <corsika/kokkos/environment/EnvironmentBorrowedView.hpp>
#include <corsika/kokkos/environment/TransportGeometryConfig.hpp>
#include <corsika/kokkos/em/algorithms/TransportMass.hpp>
#include <corsika/kokkos/em/algorithms/NativeMuonPairCutRecord.hpp>
#include <corsika/kokkos/em/algorithms/NativeMuonPairCut.hpp>
#include <corsika/kokkos/em/random/LeptonFinalStateRandomDomains.hpp>
#include <corsika/kokkos/scheduling/SubshowerCallbacks.hpp>
#include <corsika/framework/core/Logging.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>
#include <corsika/kokkos/scheduling/EmRouterStatistics.hpp>

namespace corsika::kokkos {
  struct ScalarProposalFallback {};
  /** Complete production EM router. The numeric snapshot arrays, topology and
   * native tree must remain alive and unchanged until the router is destroyed.
   * Only the backend uploads numeric arrays to device storage; this coordinator
   * keeps host-readable borrowed arrays and the original native node mapping.
   */
  template <typename TStack, typename TModelProperties,
            typename TProposalFallbackHandler = ScalarProposalFallback,
            typename TOutputSink = NullCorsikaOutputSink, typename TBackend = IEmBackend>
  class EmRouter;
  template <typename TStack, typename TModelProperties, typename TBackend,
            typename TProposalFallbackHandler = ScalarProposalFallback,
            typename TOutputSink = NullCorsikaOutputSink>
  using physical_em_router_type =
      EmRouter<TStack, TModelProperties, TProposalFallbackHandler, TOutputSink, TBackend>;

} // namespace corsika::kokkos
#include <corsika/detail/kokkos/scheduling/EmRouter.inl>

/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once
#include <corsika/kokkos/egs4/Session.hpp>
#include <corsika/detail/kokkos/egs4/Egs4ProposalMuonBackend.inl>
#include <corsika/framework/geometry/PhysicalGeometry.hpp>
#include <corsika/kokkos/data/RouterParticleConversion.hpp>
#include <stdexcept>
#include <utility>

namespace corsika::kokkos::egs4::application {
  /** Route e+/e-/gamma to EGS4 and hand fresh muon daughters to the same queue.
   * The caller owns the C8 topology, host handlers and optional muon backend.
   * All borrowed objects must remain alive until the router is drained.
   */
  template <class TStack, class TModelProperties, class THostHandler, class TRegionOf,
            class TMuonRouter = NoAcceleratedMuons>
  class Router;
} // namespace corsika::kokkos::egs4::application
#include <corsika/detail/kokkos/egs4/Router.inl>

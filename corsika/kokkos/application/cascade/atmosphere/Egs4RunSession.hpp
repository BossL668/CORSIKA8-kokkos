/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/kokkos/egs4/EmbeddedTables.hpp>
#include <corsika/kokkos/egs4/HostHandler.hpp>
#include <corsika/kokkos/egs4/Router.hpp>
#include <corsika/kokkos/egs4/AtmosphereExport.hpp>
#include <corsika/kokkos/environment/EnvironmentPreparation.hpp>
#include <corsika/kokkos/output/CorsikaOutputSink.hpp>
#include <corsika/kokkos/em/proposal/ProposalCpuFallbackHandler.hpp>
#include <corsika/kokkos/scheduling/EmRouter.hpp>
#include <corsika/kokkos/em/EmRunSession.hpp>
#include <corsika/kokkos/runtime/BackendRuntime.hpp>
#include <corsika/modules/proposal/ProposalProcessBase.hpp>
#include <corsika/modules/proposal/HadronicPhotonModel.hpp>
#include <corsika/kokkos/radio/atmosphere/FlatAtmosphereConfig.hpp>
#include <corsika/kokkos/application/cascade/CascadeEventConfig.hpp>
#include <corsika/kokkos/application/BackendOptions.hpp>
#include <corsika/kokkos/scheduling/StaticFrontier.hpp>
#include <corsika/kokkos/scheduling/HybridCascade.hpp>
#include <corsika/framework/geometry/StraightTrajectory.hpp>
#include <corsika/framework/process/ProcessSequence.hpp>
#include <corsika/output/BaseOutput.hpp>
#include <corsika/output/ShowerSummary.hpp>

namespace corsika::kokkos::application {
  /** Own the optional EGS4 atmospheric session and its PROPOSAL muon handoff.
   * C8 continues to own the scene, hadronic models, decay models and writers.
   */
  class Egs4RunSession;
} // namespace corsika::kokkos::application
#include <corsika/detail/kokkos/application/cascade/atmosphere/Egs4RunSession.inl>

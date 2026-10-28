/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/kokkos/application/cascade/CascadeBindings.hpp>
#include <corsika/kokkos/application/cascade/CascadeSupport.hpp>
#include <corsika/kokkos/application/cascade/CascadeRunner.hpp>
#include <corsika/kokkos/application/cascade/CascadeRunSession.hpp>
#include <CLI/App.hpp>
#include <memory>
#include <utility>

#ifdef CORSIKA8_WITH_NATIVE_EGS4
#include <corsika/kokkos/application/Egs4AirShowerRunner.hpp>
#endif

namespace corsika::kokkos::application {

  /** Bind the production application without introducing a second process sequence.
   * Models, writers and session remain caller-owned. Each returned cascade must
   * run before its native sequence, stack, inspector and thinning leave scope.
   * The existing EAS.run(), forceInteraction() and forceDecay() vocabulary remains.
   */
  template <typename TGeometry, typename TOutputs, typename TModels, typename TMonitors>
  class CascadeApplication {
    CascadeRunSession& session_;
    BackendOptions const& options_;
    CascadeEventConfig defaults_;
    TGeometry geometry_;
    TOutputs outputs_;
    TModels models_;
    TMonitors monitors_;
#ifdef CORSIKA8_WITH_NATIVE_EGS4
    std::unique_ptr<Egs4RunSession> egs4_;
#endif
  public:
    CascadeApplication(CascadeRunSession& session, BackendOptions const& options,
                       CascadeEventConfig defaults, TGeometry geometry, TOutputs outputs,
                       TModels models, TMonitors monitors);

    template <class TOutput>
    void registerBackendOutput(TOutput& output);

    template <typename TH, typename TL>
    auto captureDiagnostics(TH const& high, TL const& low) const;

    template <typename TTracking, typename TSequence, typename TOutput, typename TStack,
              typename TInspector, typename TThinning, typename TBefore>
    auto cascade(TTracking& tracking, TSequence& sequence, TOutput& output, TStack& stack,
                 TInspector& inspector, TThinning& thinning, std::uint64_t ordinal,
                 HEPEnergyType energy, double maximum_weight, TBefore const& before);
  };

  template <typename TG, typename TO, typename TM, typename TD>
  CascadeApplication(CascadeRunSession&, BackendOptions const&, CascadeEventConfig, TG,
                     TO, TM, TD)
      ->CascadeApplication<TG, TO, TM, TD>;
} // namespace corsika::kokkos::application

#include <corsika/detail/kokkos/application/cascade/CascadeApplication.inl>

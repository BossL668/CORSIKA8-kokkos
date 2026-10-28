/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/framework/core/Cascade.hpp>
#include <corsika/kokkos/application/cascade/CascadeBindings.hpp>
#include <corsika/kokkos/application/cascade/CascadeSupport.hpp>
#include <corsika/media/Environment.hpp>
#include <memory>
#include <optional>
#include <utility>

#ifdef CORSIKA_KOKKOS_STATIC_MULTIGPU
#include <corsika/kokkos/environment/EnvironmentSnapshot.hpp>
#include <corsika/kokkos/environment/EnvironmentSnapshotData.hpp>
#include <corsika/kokkos/scheduling/StaticFrontier.hpp>
#include <corsika/kokkos/scheduling/CpuFrontier.hpp>
#include <corsika/kokkos/scheduling/HybridCascade.hpp>
#endif
#ifdef CORSIKA8_WITH_KOKKOS_EM
#include <corsika/kokkos/application/cascade/CascadeApplication.hpp>
#include <corsika/kokkos/radio/RadioTrackSpoolWriter.hpp>
#include <corsika/kokkos/radio/ScopedRadioTrackSink.hpp>
#endif

namespace corsika::kokkos::application {

  /** Event-scoped particle import and native-node ownership; no partitioning changes. */
  template <typename TMedium>
  class CascadeParticleInput {
    CascadeWorkerOptions const& workers_;
    BackendOptions const& options_;
#ifdef CORSIKA_KOKKOS_STATIC_MULTIGPU
    std::unique_ptr<frontier::FrontierInput> source_;
    std::unique_ptr<EnvironmentSnapshot<TMedium>> environment_;
    EnvironmentDataView tree_{};
    std::optional<frontier::ScopedFrontierInput> scope_;
#endif

  public:
    template <typename TStack, typename TPrimary>
    CascadeParticleInput(CascadeWorkerOptions const&, BackendOptions const&,
                         Environment<TMedium>&, CoordinateSystemPtr const&, TStack&,
                         TPrimary const&);
    CascadeParticleInput(CascadeParticleInput const&) = delete;
    CascadeParticleInput& operator=(CascadeParticleInput const&) = delete;

    template <typename TTracking, typename TSequence, typename TOutput, typename TStack,
              typename TConfigure>
    bool runPrefix(Environment<TMedium>&, CoordinateSystemPtr const&, TTracking&,
                   TSequence&, TOutput&, TStack&, TConfigure&);
  };

  template <typename TM, typename TS, typename TP>
  CascadeParticleInput(CascadeWorkerOptions const&, BackendOptions const&,
                       Environment<TM>&, CoordinateSystemPtr const&, TS&, TP const&)
      ->CascadeParticleInput<TM>;

  /** Own the spool before the sink; keep both alive until all event reports finish. */
  template <typename TCoreas, typename TZhs>
  class CascadeRadioCapture {
#if defined(CORSIKA_KOKKOS_STATIC_MULTIGPU) && defined(CORSIKA8_WITH_KOKKOS_EM)
    std::unique_ptr<radio::RadioTrackSpoolWriter> spool_;
    std::optional<radio::ScopedRadioTrackSink<TCoreas, TZhs>> sink_;
#endif
  public:
    CascadeRadioCapture(CascadeWorkerOptions const&, TCoreas&, TZhs&,
                        CoordinateSystemPtr const&);
    CascadeRadioCapture(CascadeRadioCapture const&) = delete;
    CascadeRadioCapture& operator=(CascadeRadioCapture const&) = delete;
    void complete();
  };

  template <typename TC, typename TZ>
  CascadeRadioCapture(CascadeWorkerOptions const&, TC&, TZ&, CoordinateSystemPtr const&)
      ->CascadeRadioCapture<TC, TZ>;

  /** Select the existing native, prefix or accelerated runner around one sequence.
   * All models and event objects remain caller-owned. This class only assembles
   * existing operations; it contains no particle transport or batch algorithm.
   */
  template <typename TGeometry, typename TOutputs, typename TModels, typename TMonitors>
  class CascadeDriver {
    CascadeRuntime& runtime_;
    BackendOptions const& options_;
    CascadeWorkerOptions const& workers_;
    TGeometry geometry_;
    TModels models_;
    TMonitors monitors_;
#ifdef CORSIKA8_WITH_KOKKOS_EM
    CascadeApplication<TGeometry, TOutputs, TModels, TMonitors> accelerated_;
#endif
  public:
    CascadeDriver(CascadeRuntime&, BackendOptions const&, CascadeWorkerOptions const&,
                  CascadeEventConfig, TGeometry, TOutputs, TModels, TMonitors);
    template <typename TOutput>
    void registerBackendOutput(TOutput&);
    auto captureRadio();
    template <typename TH, typename TL>
    auto captureDiagnostics(TH const&, TL const&) const;
    template <typename TStack, typename TPrimary>
    auto prepareParticles(TStack&, TPrimary const&);
    template <typename TTracking, typename TSequence, typename TOutput, typename TStack,
              typename TInspector, typename TThinning, typename TInput, typename TBefore,
              typename TConfigure>
    auto cascade(TTracking&, TSequence&, TOutput&, TStack&, TInspector&, TThinning&,
                 TInput&, std::uint64_t ordinal, HEPEnergyType energy,
                 double maximumWeight, TBefore const&, TConfigure&);
  };

  template <typename TG, typename TO, typename TM, typename TD>
  CascadeDriver(CascadeRuntime&, BackendOptions const&, CascadeWorkerOptions const&,
                CascadeEventConfig, TG, TO, TM, TD)
      ->CascadeDriver<TG, TO, TM, TD>;
} // namespace corsika::kokkos::application

#include <corsika/detail/kokkos/application/cascade/CascadeDriver.inl>

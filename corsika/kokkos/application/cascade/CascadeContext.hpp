/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */
#pragma once

#include <array>
#include <boost/optional.hpp>
#include <cstddef>
#include <cstdint>
#include <utility>

namespace corsika::kokkos::application {

  // Non-owning, per-event bindings to application-owned objects. Every referred
  // object must outlive the run call. No model, writer, stack or callback is
  // copied. Only scalar configuration and the magnetic-field array are owned.
  // Diagnostics bind the snapshots already captured at the event boundary;
  // constructing a context never resamples counters or consumes random numbers.
  // C++17 deduction guides keep concrete process types out of the main program.

  template <typename TEnvironment, typename TCoordinates, typename TPosition,
            typename TSurface, typename TStep, typename TAxis, typename TDepth>
  struct CascadeGeometry {
    TEnvironment& environment_;
    TCoordinates const& coordinates_;
    TPosition const& injectionPosition_;
    TSurface const& surface_;
    TStep const& propagationStep_;
    TAxis const& showerAxis_;
    TDepth const& depthStep_;
    double observationHeightM_;
    double coreXM_;
    double coreYM_;
    std::array<double, 3> magneticFieldT_;
  };

  template <typename TEnvironment, typename TCoordinates, typename TPosition,
            typename TSurface, typename TStep, typename TAxis, typename TDepth>
  CascadeGeometry(TEnvironment&, TCoordinates const&, TPosition const&, TSurface const&,
                  TStep const&, TAxis const&, TDepth const&, double, double, double,
                  std::array<double, 3>)
      ->CascadeGeometry<TEnvironment, TCoordinates, TPosition, TSurface, TStep, TAxis,
                        TDepth>;

  template <typename TEnergyLoss, typename TLongitudinal, typename TProduction,
            typename TObservation, typename TInteractions, typename TCoreasDetector,
            typename TZhsDetector>
  struct CascadeOutputs {
    TEnergyLoss& energyLoss_;
    TLongitudinal& longitudinal_;
    TProduction& production_;
    TObservation& observation_;
    TInteractions& interactions_;
    TCoreasDetector& coreasDetector_;
    TZhsDetector& zhsDetector_;
  };

  template <typename TEnergyLoss, typename TLongitudinal, typename TProduction,
            typename TObservation, typename TInteractions, typename TCoreasDetector,
            typename TZhsDetector>
  CascadeOutputs(TEnergyLoss&, TLongitudinal&, TProduction&, TObservation&,
                 TInteractions&, TCoreasDetector&, TZhsDetector&)
      ->CascadeOutputs<TEnergyLoss, TLongitudinal, TProduction, TObservation,
                       TInteractions, TCoreasDetector, TZhsDetector>;

  template <typename TCoreas, typename TZhs, typename TEmCascade,
            typename TEmContinuousProposal, typename TInspector, typename TNeutrino,
            typename THadron, typename TDecay, typename TEmContinuous,
            typename TLongitudinal, typename TProduction, typename TThinning,
            typename TCut>
  struct CascadeProcesses {
    TCoreas& coreas_;
    TZhs& zhs_;
    TEmCascade& emCascade_;
    TEmContinuousProposal& emContinuousProposal_;
    TInspector& stackInspector_;
    TNeutrino& neutrinoPrimary_;
    THadron& hadronSequence_;
    TDecay& decaySequence_;
    TEmContinuous& emContinuous_;
    TLongitudinal& longitudinalProcess_;
    TProduction& productionProcess_;
    TThinning& thinning_;
    TCut& cut_;
  };

  template <typename TCoreas, typename TZhs, typename TEmCascade,
            typename TEmContinuousProposal, typename TInspector, typename TNeutrino,
            typename THadron, typename TDecay, typename TEmContinuous,
            typename TLongitudinal, typename TProduction, typename TThinning,
            typename TCut>
  CascadeProcesses(TCoreas&, TZhs&, TEmCascade&, TEmContinuousProposal&, TInspector&,
                   TNeutrino&, THadron&, TDecay&, TEmContinuous&, TLongitudinal&,
                   TProduction&, TThinning&, TCut&)
      ->CascadeProcesses<TCoreas, TZhs, TEmCascade, TEmContinuousProposal, TInspector,
                         TNeutrino, THadron, TDecay, TEmContinuous, TLongitudinal,
                         TProduction, TThinning, TCut>;

  template <typename TSequence, typename TTracking, typename TStack, typename TOutput,
            typename TPool, typename TConfigurePrimary>
  struct CascadeExecution {
    TSequence& sequence_;
    TTracking& tracking_;
    TStack& stack_;
    TOutput& output_;
    boost::optional<TPool&> hadronicPool_;
    TConfigurePrimary& configurePrimary_;
  };

  template <typename TSequence, typename TTracking, typename TStack, typename TOutput,
            typename TPool, typename TConfigurePrimary>
  CascadeExecution(TSequence&, TTracking&, TStack&, TOutput&, boost::optional<TPool&>,
                   TConfigurePrimary&)
      ->CascadeExecution<TSequence, TTracking, TStack, TOutput, TPool, TConfigurePrimary>;

  template <typename TPhotoHigh, typename TPhotoLow, typename TPhotoHighStats,
            typename TPhotoLowStats, typename THighCounter, typename TLowCounter,
            typename TPoolStatistics, typename TPhotoFallback>
  struct CascadeDiagnostics {
    TPhotoHigh& photoHigh_;
    TPhotoLow& photoLow_;
    TPhotoHighStats const& photoHighBefore_;
    TPhotoLowStats const& photoLowBefore_;
    THighCounter& highCounter_;
    TLowCounter& lowCounter_;
    std::uint64_t highInteractionsBefore_;
    std::uint64_t lowInteractionsBefore_;
    std::size_t highTimingsBefore_;
    std::size_t lowTimingsBefore_;
    TPoolStatistics const& poolBefore_;
    TPhotoFallback const& photoFallback_;
  };

  template <typename TPhotoHigh, typename TPhotoLow, typename TPhotoHighStats,
            typename TPhotoLowStats, typename THighCounter, typename TLowCounter,
            typename TPoolStatistics, typename TPhotoFallback>
  CascadeDiagnostics(TPhotoHigh&, TPhotoLow&, TPhotoHighStats const&,
                     TPhotoLowStats const&, THighCounter&, TLowCounter&, std::uint64_t,
                     std::uint64_t, std::size_t, std::size_t, TPoolStatistics const&,
                     TPhotoFallback const&)
      ->CascadeDiagnostics<TPhotoHigh, TPhotoLow, TPhotoHighStats, TPhotoLowStats,
                           THighCounter, TLowCounter, TPoolStatistics, TPhotoFallback>;

  template <typename TGeometry, typename TOutputs, typename TProcesses,
            typename TExecution, typename TDiagnostics>
  struct CascadeContext {
    TGeometry geometry_;
    TOutputs outputs_;
    TProcesses processes_;
    TExecution execution_;
    TDiagnostics diagnostics_;
  };

  template <typename TG, typename TO, typename TP, typename TE, typename TD>
  CascadeContext(TG, TO, TP, TE, TD)->CascadeContext<TG, TO, TP, TE, TD>;

  namespace detail {
    // The single, tested mapping to the established positional interface.
    // Keep the legacy computation unchanged, including its exception behavior.
    template <typename TContext, typename TFunction>
    decltype(auto) with_cascade_arguments(TContext const& context, TFunction&& function);
  } // namespace detail
} // namespace corsika::kokkos::application

#include <corsika/detail/kokkos/application/cascade/CascadeContext.inl>

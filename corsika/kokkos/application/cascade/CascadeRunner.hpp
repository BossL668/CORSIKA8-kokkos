/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/framework/geometry/CoordinateSystem.hpp>
#include <corsika/modules/proposal/NativeCalculatorView.hpp>
#include <corsika/kokkos/application/cascade/CascadeContext.hpp>
#include <corsika/kokkos/application/cascade/CascadeEventConfig.hpp>
#include <corsika/kokkos/application/cascade/CascadePreparation.hpp>
#include <corsika/kokkos/application/cascade/CascadeStepRegistry.hpp>
#include <corsika/kokkos/application/BackendOptions.hpp>
#include <corsika/kokkos/application/cascade/ShowerReport.hpp>
#include <corsika/kokkos/em/proposal/ProposalCpuFallbackHandler.hpp>
#include <corsika/kokkos/output/CorsikaOutputSink.hpp>
#include <corsika/kokkos/radio/HostRadioCollector.hpp>
#include <corsika/kokkos/radio/BorrowedRadioBackend.hpp>
#include <corsika/kokkos/radio/RadioTrackSpoolReader.hpp>
#include <corsika/kokkos/radio/ScopedRadioTrackSink.hpp>
#include <corsika/kokkos/scheduling/HybridCascadeRunner.hpp>
#include <corsika/output/ReplayTrace.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <set>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace corsika::kokkos::application {

  /** Assemble the production backend around the supplied native process sequence.
   * All models, writers and native environment nodes are caller-owned. The
   * prepared snapshot owns its numeric storage until transport, CPU fallback,
   * radio finalization and report publication have completed. Process order,
   * table reuse, random streams and scheduling policy are preserved.
   */
  template <typename TRunSession, typename TEnvironment, typename TInjectionPosition,
            typename TSurface, typename TPropagationStep, typename TDetectorCoREAS,
            typename TDetectorZHS, typename TShowerAxis, typename TDepthStep,
            typename TEnergyLossWriter, typename TLongitudinalWriter,
            typename TProductionWriter, typename TObservationLevel,
            typename TInteractionWriter, typename TCoreas, typename TZhs,
            typename TEmCascade, typename TEmContinuousProposal, typename TStackInspector,
            typename TNeutrinoPrimary, typename THadronSequence, typename TDecaySequence,
            typename TEmContinuous, typename TLongitudinalProfile,
            typename TProductionProfileProcess, typename TThinning, typename TCut,
            typename TSequence, typename TTracking, typename TStack, typename TOutput,
            typename TConfigureForcedPrimary, typename TPhotoHighEnergy,
            typename TPhotoLowEnergy, typename TPhotoHighStatistics,
            typename TPhotoLowStatistics, typename THighEnergyCounter,
            typename TLowEnergyCounter, typename TPhotoFallback>
  void run_cascade(TRunSession& cuda_session, BackendOptions const& gpu_cli,
                   CascadeEventConfig const& event, TEnvironment& env,
                   CoordinateSystemPtr const& rootCS,
                   TInjectionPosition const& injectionPos, TSurface const& surface_,
                   TPropagationStep const& step, TDetectorCoREAS& detectorCoREAS,
                   TDetectorZHS& detectorZHS, TShowerAxis const& showerAxis,
                   TDepthStep const& dX, double observationHeight_m, double showerCoreX_m,
                   double showerCoreY_m, std::array<double, 3> const& magnetic_field_T,
                   TEnergyLossWriter& dEdX, TLongitudinalWriter& profile,
                   TProductionWriter& prod_profile, TObservationLevel& observationLevel,
                   TInteractionWriter& inter_writer, TCoreas& coreas, TZhs& zhs,
                   TEmCascade& emCascade, TEmContinuousProposal& emContinuousProposal,
                   TStackInspector& stackInspect, TNeutrinoPrimary& neutrinoPrimaryPythia,
                   THadronSequence& hadronSequence, TDecaySequence& decaySequence,
                   TEmContinuous& emContinuous, TLongitudinalProfile& longprof,
                   TProductionProfileProcess& prodprof, TThinning& thinning, TCut& cut,
                   TSequence& sequence, TTracking& tracking, TStack& stack,
                   TOutput& output, HadronicProcessPool* hadronic_process_pool,
                   TConfigureForcedPrimary& configure_forced_primary,
                   TPhotoHighEnergy& photoHadronicHighEnergy,
                   TPhotoLowEnergy& photoHadronicLowEnergy,
                   TPhotoHighStatistics const& photo_hadronic_before,
                   TPhotoLowStatistics const& photo_hadronic_le_before,
                   THighEnergyCounter& heCounted, TLowEnergyCounter& leIntCounted,
                   std::uint64_t high_energy_hadronic_interactions_before,
                   std::uint64_t low_energy_hadronic_interactions_before,
                   std::size_t high_energy_hadronic_timings_before,
                   std::size_t low_energy_hadronic_timings_before,
                   HadronicProcessPoolStatistics const& hadronic_pool_statistics_before,
                   TPhotoFallback const& photoHadronicQgsjetFallback);

  template <typename TRunSession, typename TGeometry, typename TOutputs,
            typename TProcesses, typename TExecution, typename TDiagnostics>
  void run_cascade(TRunSession& session, BackendOptions const& options,
                   CascadeEventConfig const& event,
                   CascadeContext<TGeometry, TOutputs, TProcesses, TExecution,
                                  TDiagnostics> const& context);
} // namespace corsika::kokkos::application

#include <corsika/detail/kokkos/application/cascade/CascadeRunner.inl>

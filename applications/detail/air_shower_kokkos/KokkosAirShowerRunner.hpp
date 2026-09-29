/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include "KokkosAirShowerSetup.hpp"
#include "KokkosAirShowerContext.hpp"
#include "KokkosShowerReport.hpp"
#include "GpuCliOptions.hpp"

#include <corsika/accelerator/em/common/CorsikaOutputSink.hpp>
#include <corsika/accelerator/em/detail/AcceleratedHybridCascadeRunner.hpp>
#include <corsika/accelerator/em/common/ProposalCpuFallbackHandler.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <vector>

namespace corsika::applications::air_shower {

  /**
   * Application adapter that wires the existing CORSIKA process sequence to
   * the CUDA backend.  It owns no physics model and changes no random stream;
   * all models, writers and process ordering are supplied by c8_air_shower.
   */
  template <typename TRunSession,
      typename TEnvironment, typename TInjectionPosition, typename TSurface,
      typename TPropagationStep, typename TDetectorCoREAS,
      typename TDetectorZHS, typename TShowerAxis, typename TDepthStep,
      typename TEnergyLossWriter, typename TLongitudinalWriter,
      typename TProductionWriter, typename TObservationLevel,
      typename TInteractionWriter, typename TCoreas, typename TZhs,
      typename TEmCascade, typename TEmContinuousProposal,
      typename TStackInspector, typename TNeutrinoPrimary,
      typename THadronSequence, typename TDecaySequence,
      typename TEmContinuous, typename TLongitudinalProfile,
      typename TProductionProfileProcess, typename TThinning, typename TCut,
      typename TSequence, typename TTracking, typename TStack,
      typename TOutput, typename TConfigureForcedPrimary,
      typename TPhotoHighEnergy, typename TPhotoLowEnergy,
      typename TPhotoHighStatistics, typename TPhotoLowStatistics,
      typename THighEnergyCounter, typename TLowEnergyCounter,
      typename TPhotoFallback>
  void runKokkosAirShower(
      TRunSession& cuda_session, GpuCliOptions const& gpu_cli,
      KokkosEventConfig const& event, TEnvironment& env,
      CoordinateSystemPtr const& rootCS,
      TInjectionPosition const& injectionPos, TSurface const& surface_,
      TPropagationStep const& step, TDetectorCoREAS& detectorCoREAS,
      TDetectorZHS& detectorZHS, TShowerAxis const& showerAxis,
      TDepthStep const& dX, double observationHeight_m,
      double showerCoreX_m, double showerCoreY_m,
      std::array<double, 3> const& magnetic_field_T,
      TEnergyLossWriter& dEdX, TLongitudinalWriter& profile,
      TProductionWriter& prod_profile,
      TObservationLevel& observationLevel,
      TInteractionWriter& inter_writer, TCoreas& coreas, TZhs& zhs,
      TEmCascade& emCascade,
      TEmContinuousProposal& emContinuousProposal,
      TStackInspector& stackInspect,
      TNeutrinoPrimary& neutrinoPrimaryPythia,
      THadronSequence& hadronSequence, TDecaySequence& decaySequence,
      TEmContinuous& emContinuous,
      TLongitudinalProfile& longprof,
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
      TPhotoFallback const& photoHadronicQgsjetFallback) {
    using namespace corsika::gpu::em;
    using namespace corsika::gpu::em::tables;
    using EnvType = std::remove_reference_t<TEnvironment>;
    using StackType = std::remove_reference_t<TStack>;
    using Sequence = std::remove_reference_t<TSequence>;

    auto const hadronic_min_batch = gpu_cli.hadronic_min_batch;
    auto const hadronic_target_batch_ms =
        gpu_cli.hadronic_target_batch_ms;
    auto const hadronic_max_batch = gpu_cli.hadronic_max_batch;
    auto const hadronic_initial_cost_ms =
        gpu_cli.hadronic_initial_cost_ms;
    auto const cpu_detailed_step_timing =
        gpu_cli.cpu_detailed_step_timing;

    auto const seed = event.seed;
    auto const i_shower = event.shower_ordinal;
    auto const output_shower_id = event.output_shower_id;
    auto const primaryTotalEnergy = event.primary_total_energy;
    auto const heHadronModelThreshold =
        event.high_energy_hadronic_threshold;
    auto const emthinfrac = event.em_thinning_fraction;
    auto const maxWeight = event.maximum_weight;
    auto const automaticMaxWeight = event.automatic_maximum_weight;
    auto const thinningCanActivateFromUnitWeight =
        event.thinning_can_activate_from_unit_weight;
    auto const beamCode = event.beam_code;
    auto const& high_energy_hadronic_model =
        event.high_energy_hadronic_model;
    // These references deliberately participate in template deduction: their
    // types form the audited GPU process registry below.  Runtime access goes
    // through the already-constructed ProcessSequence, exactly as before the
    // application-only extraction.
    static_cast<void>(stackInspect);
    static_cast<void>(neutrinoPrimaryPythia);
    static_cast<void>(hadronSequence);
    static_cast<void>(decaySequence);
    static_cast<void>(emContinuous);
    static_cast<void>(longprof);
    static_cast<void>(prodprof);
    static_cast<void>(thinning);
    static_cast<void>(cut);

    auto prepared = prepareKokkosAirShower(
        cuda_session, gpu_cli, event, env, rootCS, injectionPos, surface_, step,
        detectorCoREAS, detectorZHS, showerAxis, dX, observationHeight_m,
        showerCoreX_m, showerCoreY_m, magnetic_field_T, dEdX, profile);
    auto& environment_snapshot = prepared.environment_snapshot;
    auto& gpu_config = prepared.gpu_config;
    auto const& physics_requirements = prepared.physics_requirements;
    auto const proposal_stochastic_cut = prepared.proposal_stochastic_cut;
    auto const gpu_radio_enabled = prepared.gpu_radio_enabled;
    using GpuEmStepRegistry = KokkosAirShowerStepRegistry<
        TStackInspector, TNeutrinoPrimary, THadronSequence, TDecaySequence,
        TEmCascade, TEmContinuous, TCoreas, TZhs, TLongitudinalProfile,
        TObservationLevel, TProductionProfileProcess, TInteractionWriter,
        TThinning, TCut>;
    using FallbackHandler =
        ProposalCpuFallbackHandler<StackType, TEmCascade, Sequence, EnvType>;
    auto& runtime_session = cuda_session.runtimeSession();
    bool backend_reused_for_shower = false;
    bool muon_transport_available = false;
    auto begin_backend = [&]() -> decltype(auto) {

        std::vector<proposal::NativeInteractionCalculatorView> native_views;
        std::vector<proposal::NativeContinuousCalculatorView>
            native_continuous_views;
        if (!runtime_session.initialized()) {
          native_views = emCascade.nativeCalculatorViews();
          native_continuous_views =
              emContinuousProposal.nativeCalculatorViews();
        }
        auto session_begin = runtime_session.beginProposalNative(
            environment_snapshot, gpu_config, physics_requirements,
            native_views, native_continuous_views);
        backend_reused_for_shower = session_begin.reused;
        muon_transport_available = session_begin.muon_transport_available;
        return *session_begin.backend;
      
    };

    HadronicWorkClassifierConfig hadronic_work_classifier;
    hadronic_work_classifier.transition_energy_GeV =
        heHadronModelThreshold / 1_GeV;
    gpu::em::detail::AcceleratedHybridRunOptions const run_options{
        validation::CudaReplayTrace::instance().enabled(), true};
    auto fallback_factory = [&]() -> FallbackHandler {
      return {emCascade, sequence, env, rootCS,
              static_cast<std::uint64_t>(seed),
              static_cast<std::uint64_t>(i_shower),
              proposal_stochastic_cut};
    };
    auto output_sink_factory = [&]() {
      return CorsikaOutputSink{
          rootCS, dEdX, profile, prod_profile, observationLevel,
          inter_writer, coreas, zhs, detectorCoREAS.size() != 0,
          gpu_radio_enabled};
    };
    auto configure_cascade = [&](auto& cascade) {
      cascade.configureHadronicWorkClassification(hadronic_work_classifier);
      cascade.enableScalarDetailedPhaseTiming(cpu_detailed_step_timing);
      if (hadronic_process_pool) {
        HybridHadronicWorkerConfig hadronic_worker_config;
        hadronic_worker_config.seed = static_cast<std::uint64_t>(seed);
        hadronic_worker_config.shower_id =
            static_cast<std::uint64_t>(output_shower_id);
        hadronic_worker_config.minimum_pending_interactions =
            hadronic_min_batch;
        hadronic_worker_config.target_batch_cost_ms =
            hadronic_target_batch_ms;
        hadronic_worker_config.maximum_batch_items = hadronic_max_batch;
        hadronic_worker_config.initial_interaction_cost_ms =
            hadronic_initial_cost_ms;
        cascade.configureHadronicProcessPool(*hadronic_process_pool,
                                             hadronic_worker_config);
      }
      configure_forced_primary(cascade);
    };
    auto record_report =
        [&](auto& completed_backend, auto& cascade, auto& router,
            auto& output_sink, auto& fallback_handler) {
          corsika::applications::air_shower::KokkosShowerReportBuilder::
              record<GpuEmStepRegistry, Sequence>(
                  cuda_session, gpu_cli, completed_backend, cascade, router,
                  output_sink, fallback_handler, photoHadronicHighEnergy,
                  photoHadronicLowEnergy, photo_hadronic_before,
                  photo_hadronic_le_before, heCounted, leIntCounted,
                  high_energy_hadronic_interactions_before,
                  low_energy_hadronic_interactions_before,
                  high_energy_hadronic_timings_before,
                  low_energy_hadronic_timings_before,
                  hadronic_work_classifier, hadronic_process_pool,
                  hadronic_pool_statistics_before,
                  photoHadronicQgsjetFallback, high_energy_hadronic_model,
                  heHadronModelThreshold, emthinfrac, maxWeight,
                  automaticMaxWeight, thinningCanActivateFromUnitWeight,
                  backend_reused_for_shower,
                  muon_transport_available, gpu_radio_enabled,
                  gpu_config, beamCode, primaryTotalEnergy, output_shower_id);
        };
    gpu::em::detail::runAcceleratedHybridCascade<GpuEmStepRegistry>(
        rootCS, environment_snapshot, env, tracking, sequence, output, stack,
        run_options, begin_backend, fallback_factory, output_sink_factory,
        configure_cascade, record_report);

  }

  // Compact application-facing overload. The legacy overload above remains the
  // computation and compatibility path for existing comparison applications.
  template <typename RunSession, typename G, typename O, typename P, typename E,
            typename D>
  void runKokkosAirShower(
      RunSession& session, GpuCliOptions const& options,
      KokkosEventConfig const& event,
      KokkosAirShowerContext<G, O, P, E, D> const& context) {
    detail::withKokkosAirShowerArguments(context, [&](auto&&... arguments) {
      runKokkosAirShower(session, options, event,
                        std::forward<decltype(arguments)>(arguments)...);
    });
  }

} // namespace corsika::applications::air_shower

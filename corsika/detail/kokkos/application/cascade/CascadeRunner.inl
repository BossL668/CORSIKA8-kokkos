/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

namespace corsika::kokkos::application {

  template <typename TRadio>
  void replay_prefix_radio(TRadio&, std::vector<std::string> const& paths,
                           CoordinateSystemPtr const&) {
    if (!paths.empty())
      throw std::invalid_argument("prefix radio tracks require a replay-capable backend");
  }

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
                   TPhotoFallback const& photoHadronicQgsjetFallback) {
    using namespace corsika::kokkos;

    using environment_type = std::remove_reference_t<TEnvironment>;
    using stack_type = std::remove_reference_t<TStack>;
    using sequence_type = std::remove_reference_t<TSequence>;

    auto const hadronic_min_batch = gpu_cli.hadronicMinBatch_;
    auto const hadronic_target_batch_ms = gpu_cli.hadronicTargetBatchMs_;
    auto const hadronic_max_batch = gpu_cli.hadronicMaxBatch_;
    auto const hadronic_initial_cost_ms = gpu_cli.hadronicInitialCostMs_;
    auto const cpu_detailed_step_timing = gpu_cli.cpuDetailedStepTiming_;

    auto const seed = event.seed_;
    auto const i_shower = event.showerOrdinal_;
    auto const output_shower_id = event.outputShowerId_;
    auto const primaryTotalEnergy = event.primaryTotalEnergy_;
    auto const heHadronModelThreshold = event.highEnergyHadronicThreshold_;
    auto const emthinfrac = event.emThinningFraction_;
    auto const maxWeight = event.maximumWeight_;
    auto const automaticMaxWeight = event.automaticMaximumWeight_;
    auto const thinningCanActivateFromUnitWeight =
        event.thinningCanActivateFromUnitWeight_;
    auto const beamCode = event.beamCode_;
    auto const& high_energy_hadronic_model = event.highEnergyHadronicModel_;
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

    auto const observation_plane =
        export_observation_plane(observationLevel.getPlane(), rootCS);
    auto prepared = prepare_cascade(
        cuda_session, gpu_cli, event, env, rootCS, injectionPos, surface_, step,
        detectorCoREAS, detectorZHS, showerAxis, dX, observationHeight_m, showerCoreX_m,
        showerCoreY_m, magnetic_field_T, dEdX, profile, &observation_plane);
    auto const& environment_snapshot = *prepared.environmentStorage_;
    auto& gpu_config = prepared.backendConfig_;
    auto const& physics_requirements = prepared.physicsRequirements_;
    auto const proposal_stochastic_cut = prepared.proposalStochasticCut_;
    auto const gpu_radio_enabled = prepared.radioEnabled_;
    using process_registry_type = cascade_step_registry_type<
        TStackInspector, TNeutrinoPrimary, THadronSequence, TDecaySequence, TEmCascade,
        TEmContinuous, TCoreas, TZhs, TLongitudinalProfile, TObservationLevel,
        TProductionProfileProcess, TInteractionWriter, TThinning, TCut>;
    using fallback_handler_type =
        ProposalCpuFallbackHandler<stack_type, TEmCascade, sequence_type,
                                   environment_type>;
    auto& runtime_session = cuda_session.getRuntimeSession();
    bool backend_reused_for_shower = false;
    bool muon_transport_available = false;
    using host_radio_collector_type =
        corsika::kokkos::radio::HostRadioCollector<corsika::kokkos::radio::IRadioBackend>;
    using scalar_radio_scope_type =
        corsika::kokkos::radio::ScopedRadioTrackSink<TCoreas, TZhs>;
    // Keep the collector alive through router finalization and output writing.
    // Its backend flushes the last partial batch before downloading waveforms.
    std::unique_ptr<radio::BorrowedRadioBackend<IEmBackend>> radio_backend_adapter;
    std::unique_ptr<host_radio_collector_type> host_radio_collector;
    std::unique_ptr<scalar_radio_scope_type> scalar_radio_scope;
    auto begin_backend = [&]() -> decltype(auto) {
      std::vector<proposal::NativeInteractionCalculatorView> native_views;
      std::vector<proposal::NativeContinuousCalculatorView> native_continuous_views;
      if (!runtime_session.isInitialized()) {
        native_views = emCascade.nativeCalculatorViews();
        native_continuous_views = emContinuousProposal.nativeCalculatorViews();
      }
      auto session_begin = runtime_session.beginProposalNative(
          prepared.environmentData_, prepared.geometry_, gpu_config, physics_requirements,
          native_views, native_continuous_views);
      backend_reused_for_shower = session_begin.reused_;
      muon_transport_available = session_begin.muonTransportAvailable_;
      if (gpu_radio_enabled) {
        radio_backend_adapter = std::make_unique<radio::BorrowedRadioBackend<IEmBackend>>(
            *session_begin.backend_);
        host_radio_collector =
            std::make_unique<host_radio_collector_type>(*radio_backend_adapter);
        scalar_radio_scope = std::make_unique<scalar_radio_scope_type>(
            coreas, zhs, *host_radio_collector, rootCS);
        // Prefix processes capture tracks without initializing a GPU. Each
        // closed spool belongs to exactly one final GPU worker.
        std::set<std::filesystem::path> inputs;
        for (auto const& path : gpu_cli.radioTrackInputs_) {
          if (!inputs.insert(std::filesystem::canonical(path)).second) {
            throw std::invalid_argument("duplicate prefix radio track input");
          }
          corsika::kokkos::radio::RadioTrackSpoolReader reader(path);
          std::vector<corsika::radio::RadioTrackSegment> tracks;
          while (reader.readBatch(tracks, 4096)) {
            for (auto const& track : tracks) host_radio_collector->append(track);
          }
          if (reader.consumed() != reader.count()) {
            throw std::runtime_error("prefix radio track input was not fully consumed");
          }
        }
      } else {
        replay_prefix_radio(coreas, gpu_cli.radioTrackInputs_, rootCS);
      }
      return *session_begin.backend_;
    };

    HadronicWorkClassifierConfig hadronic_work_classifier;
    hadronic_work_classifier.transitionEnergyGeV_ = heHadronModelThreshold / 1_GeV;
    corsika::kokkos::HybridRunOptions const run_options{
        corsika::ReplayTrace::instance().enabled(), true};
    auto fallback_factory =
        [&](typename fallback_handler_type::topology_type const& topology)
        -> fallback_handler_type {
      return {emCascade,
              sequence,
              env,
              topology,
              rootCS,
              static_cast<std::uint64_t>(seed),
              static_cast<std::uint64_t>(i_shower),
              proposal_stochastic_cut};
    };
    auto output_sink_factory = [&]() {
      return CorsikaOutputSink{rootCS,
                               dEdX,
                               profile,
                               prod_profile,
                               observationLevel,
                               inter_writer,
                               coreas,
                               zhs,
                               detectorCoREAS.size() != 0,
                               gpu_radio_enabled};
    };
    auto configure_cascade = [&](auto& cascade) {
      cascade.configureHadronicWorkClassification(hadronic_work_classifier);
      cascade.enableScalarDetailedPhaseTiming(cpu_detailed_step_timing);
      if (hadronic_process_pool) {
        HybridHadronicWorkerConfig hadronic_worker_config;
        hadronic_worker_config.seed_ = static_cast<std::uint64_t>(seed);
        hadronic_worker_config.showerId_ = static_cast<std::uint64_t>(output_shower_id);
        hadronic_worker_config.minimumPendingInteractions_ = hadronic_min_batch;
        hadronic_worker_config.targetBatchCostMs_ = hadronic_target_batch_ms;
        hadronic_worker_config.maximumBatchItems_ = hadronic_max_batch;
        hadronic_worker_config.initialInteractionCostMs_ = hadronic_initial_cost_ms;
        cascade.configureHadronicProcessPool(*hadronic_process_pool,
                                             hadronic_worker_config);
      }
      configure_forced_primary(cascade);
    };
    auto record_report = [&](auto& completed_backend, auto& cascade, auto& router,
                             auto& output_sink, auto& fallback_handler) {
      corsika::kokkos::application::ShowerReportBuilder::record<process_registry_type,
                                                                sequence_type>(
          cuda_session, gpu_cli, completed_backend, cascade, router, output_sink,
          fallback_handler, photoHadronicHighEnergy, photoHadronicLowEnergy,
          photo_hadronic_before, photo_hadronic_le_before, heCounted, leIntCounted,
          high_energy_hadronic_interactions_before,
          low_energy_hadronic_interactions_before, high_energy_hadronic_timings_before,
          low_energy_hadronic_timings_before, hadronic_work_classifier,
          hadronic_process_pool, hadronic_pool_statistics_before,
          photoHadronicQgsjetFallback, high_energy_hadronic_model, heHadronModelThreshold,
          emthinfrac, maxWeight, automaticMaxWeight, thinningCanActivateFromUnitWeight,
          backend_reused_for_shower, muon_transport_available, gpu_radio_enabled,
          gpu_config, beamCode, primaryTotalEnergy, output_shower_id);
    };
    corsika::kokkos::run_hybrid_cascade<process_registry_type>(
        rootCS, environment_snapshot, prepared.geometry_, env, tracking, sequence, output,
        stack, run_options, begin_backend, fallback_factory, output_sink_factory,
        configure_cascade, record_report);
  }

  template <typename TRunSession, typename TGeometry, typename TOutputs,
            typename TProcesses, typename TExecution, typename TDiagnostics>
  void run_cascade(TRunSession& session, BackendOptions const& options,
                   CascadeEventConfig const& event,
                   CascadeContext<TGeometry, TOutputs, TProcesses, TExecution,
                                  TDiagnostics> const& context) {
    detail::with_cascade_arguments(context, [&](auto&&... arguments) {
      run_cascade(session, options, event,
                  std::forward<decltype(arguments)>(arguments)...);
    });
  }
} // namespace corsika::kokkos::application

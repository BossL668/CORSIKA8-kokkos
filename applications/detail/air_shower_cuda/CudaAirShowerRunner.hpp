/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include "CudaRunSession.hpp"
#include "CudaShowerReport.hpp"
#include "GpuCliOptions.hpp"

#include <corsika/framework/core/HybridCascade.hpp>
#include <corsika/gpu/em/CorsikaOutputSink.hpp>
#include <corsika/gpu/em/EnvironmentSnapshotBuilder.hpp>
#include <corsika/gpu/em/PhysicalCudaEmRouter.hpp>
#include <corsika/gpu/em/ProcessSequenceCompatibility.hpp>
#include <corsika/gpu/em/ProposalCpuFallbackHandler.hpp>
#include <corsika/gpu/em/tables/ProposalNativeTableExporter.hpp>
#include <corsika/gpu/radio/RadioSnapshotBuilder.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace corsika::applications::air_shower {

  struct CudaEventConfig {
    std::uint64_t seed{};
    std::uint64_t shower_ordinal{};
    unsigned int output_shower_id{};
    HEPEnergyType configured_primary_energy_max{};
    HEPEnergyType primary_total_energy{};
    HEPEnergyType em_cut{};
    HEPEnergyType muon_cut{};
    HEPEnergyType production_threshold{};
    HEPEnergyType high_energy_hadronic_threshold{};
    double em_thinning_fraction{};
    double maximum_weight{};
    bool automatic_maximum_weight{};
    bool thinning_can_activate_from_unit_weight{};
    bool keep_zero_weight_particles{};
    Code beam_code{};
    std::string high_energy_hadronic_model;
    double maximum_magnetic_deflection_rad{};
  };

  /**
   * Application adapter that wires the existing CORSIKA process sequence to
   * the CUDA backend.  It owns no physics model and changes no random stream;
   * all models, writers and process ordering are supplied by c8_air_shower.
   */
  template <
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
  void runCudaAirShower(
      CudaRunSession& cuda_session, GpuCliOptions const& gpu_cli,
      CudaEventConfig const& event, TEnvironment& env,
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
    using TrackingType = std::remove_reference_t<TTracking>;
    using Sequence = std::remove_reference_t<TSequence>;

    auto const* loaded_gpu_table = cuda_session.loadedRateTable();
    auto const& gpu_physics_source = gpu_cli.gpu_physics_source;
    auto const gpu_device = gpu_cli.gpu_device;
    auto const gpu_min_batch = gpu_cli.gpu_min_batch;
    auto const gpu_memory_fraction = gpu_cli.gpu_memory_fraction;
    auto const& gpu_table_cache = gpu_cli.gpu_table_cache;
    auto const& gpu_aux_cache_dir = gpu_cli.gpu_aux_cache_dir;
    auto const gpu_table_tolerance = gpu_cli.gpu_table_tolerance;
    auto const gpu_deterministic = gpu_cli.gpu_deterministic;
    auto const gpu_detailed_stage_timing =
        gpu_cli.gpu_detailed_stage_timing;
    auto const gpu_full_step_records = gpu_cli.gpu_full_step_records;
    auto const gpu_resident_cross_species =
        gpu_cli.gpu_resident_cross_species;
    auto const& radio_backend = gpu_cli.radio_backend;
    auto const gpu_radio_field_limit = gpu_cli.gpu_radio_field_limit;
    auto const gpu_radio_track_diagnostics =
        gpu_cli.gpu_radio_track_diagnostics;
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
    auto const eMax = event.configured_primary_energy_max;
    auto const primaryTotalEnergy = event.primary_total_energy;
    auto const emcut = event.em_cut;
    auto const mucut = event.muon_cut;
    auto const prod_threshold = event.production_threshold;
    auto const heHadronModelThreshold =
        event.high_energy_hadronic_threshold;
    auto const emthinfrac = event.em_thinning_fraction;
    auto const maxWeight = event.maximum_weight;
    auto const automaticMaxWeight = event.automatic_maximum_weight;
    auto const thinningCanActivateFromUnitWeight =
        event.thinning_can_activate_from_unit_weight;
    auto const multithin = event.keep_zero_weight_particles;
    auto const beamCode = event.beam_code;
    auto const& high_energy_hadronic_model =
        event.high_energy_hadronic_model;
    auto const maximum_magnetic_deflection_rad =
        event.maximum_magnetic_deflection_rad;

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

    auto const showerCoreX = showerCoreX_m * 1_m;
    auto const showerCoreY = showerCoreY_m * 1_m;
    auto const observationHeight = observationHeight_m * 1_m;

      if (gpu_physics_source == "c8emrt" && !loaded_gpu_table) {
        throw std::logic_error(
            "CUDA EM table was not loaded during application initialization");
      }
      auto const requested_cut_MeV = emcut / 1_MeV;
      auto const proposal_stochastic_cut =
          proposal::optimized_proposal_energy_cut(prod_threshold);
      auto const proposal_stochastic_cut_MeV =
          proposal_stochastic_cut / 1_MeV;
      ProposalTableSet descriptor{};
      bool gpu_muon_transport_available =
          gpu_physics_source == "proposal-native";
      if (loaded_gpu_table) {
      auto const& table = *loaded_gpu_table;
      auto const cut_scale =
          std::max({1., std::abs(proposal_stochastic_cut_MeV),
                    std::abs(table.metadata.energy_cut_MeV)});
      if (std::abs(
              table.metadata.energy_cut_MeV -
              proposal_stochastic_cut_MeV) >
          32. * std::numeric_limits<double>::epsilon() * cut_scale) {
        throw std::runtime_error(
            "CUDA EM table stochastic cut does not match the scalar "
            "PROPOSAL cut resolved from the configured production threshold");
      }
      if (table.metadata.energy_min_MeV >
          std::min(
              requested_cut_MeV,
              proposal_stochastic_cut_MeV)) {
        throw std::runtime_error(
            "CUDA EM table energy domain does not cover the configured "
            "transport and stochastic cuts");
      }
      for (auto const pdg : {11, -11}) {
        auto const& continuous = findContinuousEnergyTable(table, pdg);
        auto const code = convert_from_PDG(static_cast<PDGCode>(pdg));
        auto const table_transport_cut_MeV =
            continuous.minimum_total_energy_MeV - get_mass(code) / 1_MeV;
        auto const expected_transport_cut_MeV =
            ContinuousCutSafetyFactor * requested_cut_MeV;
        if (std::abs(table_transport_cut_MeV - expected_transport_cut_MeV) >
            2.e-6 * cut_scale) {
          throw std::runtime_error(
              "CUDA EM table transport cut does not match --emcut");
        }
      }
      auto const tableHasPid = [&](std::int32_t pdg) {
        return std::any_of(
            table.particles.begin(), table.particles.end(),
            [pdg](auto const& particle) {
              return particle.pdg_id == pdg;
            });
      };
      gpu_muon_transport_available =
          tableHasPid(13) && tableHasPid(-13);
      if (tableHasPid(13) != tableHasPid(-13)) {
        throw std::runtime_error(
            "CUDA EM table contains only one muon charge state");
      }
      if (gpu_muon_transport_available) {
        auto const requested_muon_cut_MeV =
            mucut / 1_MeV;
        auto const expected_muon_transport_cut_MeV =
            ContinuousCutSafetyFactor *
            requested_muon_cut_MeV;
        for (auto const pdg : {13, -13}) {
          auto const& continuous =
              findContinuousEnergyTable(table, pdg);
          auto const code =
              convert_from_PDG(static_cast<PDGCode>(pdg));
          auto const table_transport_cut_MeV =
              continuous.minimum_total_energy_MeV -
              get_mass(code) / 1_MeV;
          auto const muon_cut_scale =
              std::max(
                  {1., std::abs(table_transport_cut_MeV),
                   std::abs(
                       expected_muon_transport_cut_MeV)});
          if (std::abs(
                  table_transport_cut_MeV -
                  expected_muon_transport_cut_MeV) >
              2.e-6 * muon_cut_scale) {
            throw std::runtime_error(
                "CUDA muon table transport cut does not match --mucut");
          }
        }
      }
      if (primaryTotalEnergy / 1_MeV > table.metadata.energy_max_MeV) {
        throw std::runtime_error(
            "primary energy exceeds the CUDA EM table energy domain");
      }

      descriptor.process_count = rateTableProcessCount(table);
      descriptor.content_hash = table.content_hash;
      }
      auto environment_snapshot = makeCorsika7AtmosphereSnapshot(
          AtmosphereId::USStdBK, {0., 0., 0.}, 17,
          observationHeight / 1_m,
          magnetic_field_T,
          maximum_magnetic_deflection_rad);
      setObservationPlane(
          environment_snapshot,
          {showerCoreX / 1_m, showerCoreY / 1_m,
           observationHeight / 1_m},
          {0., 0., 1.});
      GpuEmConfig gpu_config{};
      gpu_config.device = gpu_device;
      gpu_config.min_batch_size = gpu_min_batch;
      gpu_config.memory_fraction = gpu_memory_fraction;
      gpu_config.table_tolerance = gpu_table_tolerance;
      gpu_config.em_transport_cut_MeV = requested_cut_MeV;
      gpu_config.muon_transport_cut_MeV = mucut / 1_MeV;
      gpu_config.physics_source =
          gpu_physics_source == "proposal-native"
              ? GpuPhysicsSource::ProposalNative
              : GpuPhysicsSource::C8EmRt;
      gpu_config.auxiliary_cache_directory = gpu_aux_cache_dir;
      gpu_config.deterministic = gpu_deterministic;
      gpu_config.detailed_stage_timing =
          gpu_detailed_stage_timing;
      gpu_config.random_seed = static_cast<std::uint64_t>(seed);
      gpu_config.shower_id = static_cast<std::uint64_t>(i_shower);
      gpu_config.table_cache = gpu_table_cache;
      gpu_config.thinning.enabled = emthinfrac > 0. ? 1 : 0;
      gpu_config.thinning.threshold_GeV =
          emthinfrac * primaryTotalEnergy / 1_GeV;
      gpu_config.thinning.maximum_weight = maxWeight;
      gpu_config.thinning.erase_zero_weight = multithin ? 0 : 1;
      gpu_config.resident_cross_species =
          gpu_resident_cross_species;
      auto const gpu_radio_enabled =
          radio_backend == "cuda" && detectorCoREAS.size() != 0;
      if (gpu_radio_enabled &&
          !cuda_session.hasBackend()) {
        gpu_config.radio =
            gpu::radio::makeGpuRadioConfig(
                env, injectionPos, surface_, step,
                detectorCoREAS, detectorZHS);
        gpu_config.radio.deterministic =
            gpu_config.deterministic;
        gpu_config.radio.fixed_point_field_limit_V_per_m =
            gpu_radio_field_limit;
        gpu_config.radio.track_diagnostics =
            gpu_radio_track_diagnostics;
      }
      if ((detectorCoREAS.size() == 0 || gpu_radio_enabled) &&
          !gpu_full_step_records) {
        auto& projection =
            gpu_config.profile_projection;
        auto const primary_energy_GeV =
            primaryTotalEnergy / 1_GeV;
        auto const em_cut_GeV = emcut / 1_GeV;
        projection.fixed_point_weight_limit =
            2. * primary_energy_GeV / em_cut_GeV;
        projection.fixed_point_energy_limit_GeV =
            2. * primary_energy_GeV;
        if (!cuda_session.hasBackend()) {
          projection.enabled = true;
          auto const axis_start =
              showerAxis.getStart().getCoordinates(rootCS);
          auto const axis_direction =
              showerAxis.getDirection().getComponents(rootCS);
          for (int axis = 0; axis < 3; ++axis) {
            projection.axis_start_position_m[axis] =
                axis_start[axis] / 1_m;
            projection.axis_direction[axis] =
                axis_direction[axis].magnitude();
          }
          projection.axis_step_length_m =
              showerAxis.getSteplength() / 1_m;
          auto const& support =
              showerAxis.getGrammageSupport();
          projection.axis_grammage_g_per_cm2.reserve(
              support.size());
          for (auto const grammage : support) {
            projection.axis_grammage_g_per_cm2.push_back(
                grammage /
                (1_g / square(1_cm)));
          }
          if (dEdX.GetNBins() != profile.getNBins()) {
            throw std::runtime_error(
                "CUDA resident profile requires identical energy-loss and "
                "longitudinal bin counts");
          }
          projection.accumulate_on_device = true;
          projection.output_bin_count = dEdX.GetNBins();
          projection.output_bin_width_g_per_cm2 =
              dX / (1_g / square(1_cm));
          projection.energy_loss_threshold_g_per_cm2 =
              1.e-4;
        }
      }

      using GpuEmStepRegistry = GpuEmStepProcessRegistry<
          GpuEmStepProcessRegistration<
              TStackInspector,
              GpuEmStepProcessPolicy::DiagnosticOnly>,
          GpuEmStepProcessRegistration<
              TNeutrinoPrimary,
              GpuEmStepProcessPolicy::InapplicableToRoutedEm>,
          GpuEmStepProcessRegistration<
              THadronSequence,
              GpuEmStepProcessPolicy::InapplicableToRoutedEm>,
          GpuEmStepProcessRegistration<
              TDecaySequence,
              GpuEmStepProcessPolicy::DeferredToCpu>,
          GpuEmStepProcessRegistration<
              TEmCascade,
              GpuEmStepProcessPolicy::ReplacedOnDevice>,
          GpuEmStepProcessRegistration<
              TEmContinuous,
              GpuEmStepProcessPolicy::ReplacedOnDevice>,
          GpuEmStepProcessRegistration<
              TCoreas,
              GpuEmStepProcessPolicy::ReplayedFromDeviceRecord>,
          GpuEmStepProcessRegistration<
              TZhs,
              GpuEmStepProcessPolicy::ReplayedFromDeviceRecord>,
          GpuEmStepProcessRegistration<
              TLongitudinalProfile,
              GpuEmStepProcessPolicy::ReplayedFromDeviceRecord>,
          GpuEmStepProcessRegistration<
              TObservationLevel,
              GpuEmStepProcessPolicy::ReplayedFromDeviceRecord>,
          GpuEmStepProcessRegistration<
              TProductionProfileProcess,
              GpuEmStepProcessPolicy::ReplayedFromDeviceRecord>,
          GpuEmStepProcessRegistration<
              TInteractionWriter,
              GpuEmStepProcessPolicy::ReplayedFromDeviceRecord>,
          GpuEmStepProcessRegistration<
              TThinning,
              GpuEmStepProcessPolicy::ReplacedOnDevice>,
          GpuEmStepProcessRegistration<
              TCut,
              GpuEmStepProcessPolicy::ReplacedOnDevice>>;
      GpuEmStepRegistry::template validateOrThrow<Sequence>();
      CORSIKA_LOG_INFO(
          "CUDA EM process registry accepted {} process contracts "
          "({} device-replaced, {} record-replayed, {} deferred-to-CPU, "
          "{} inapplicable-to-routed-EM, {} diagnostic-only)",
          GpuEmStepRegistry::registrationCount(),
          GpuEmStepRegistry::replacedOnDeviceCount(),
          GpuEmStepRegistry::replayedFromDeviceRecordCount(),
          GpuEmStepRegistry::deferredToCpuCount(),
          GpuEmStepRegistry::inapplicableToRoutedEmCount(),
          GpuEmStepRegistry::diagnosticOnlyCount());

      auto const backend_reused_for_shower =
          cuda_session.hasBackend();
      if (!cuda_session.hasBackend()) {
        cuda_session.createBackend();
        if (gpu_config.physics_source ==
            GpuPhysicsSource::ProposalNative) {
          auto const native_views =
              emCascade.nativeCalculatorViews();
          auto const native_continuous_views =
              emContinuousProposal.nativeCalculatorViews();
          auto const native_tables = exportProposalNativeTables(
              native_views, native_continuous_views);
          // Fail before transporting the first particle when the immutable
          // PROPOSAL interpolants cannot cover any shower requested by this
          // process.  In --energy_range mode the reusable backend is created
          // only once, so checking the first randomly sampled primary would
          // let a later, more energetic shower bypass this startup gate.  A
          // hadronic primary can transfer almost all of its energy to one EM
          // or muonic secondary; therefore use the configured range upper
          // bound for every routed PID.
          auto const configured_primary_energy_max_MeV =
              eMax / 1_MeV;
          auto const native_domain_epsilon =
              64. * std::numeric_limits<double>::epsilon();
          for (auto const pdg : {22, 11, -11, 13, -13}) {
            auto const total = std::find_if(
                native_tables.total_rate_columns.begin(),
                native_tables.total_rate_columns.end(),
                [pdg](auto const& column) {
                  return column.pdg_id == pdg;
                });
            if (total == native_tables.total_rate_columns.end()) {
              throw std::runtime_error(
                  "PROPOSAL native table has no total-rate spline for a "
                  "routed particle");
            }
            double upper_energy_MeV = total->spline.axis.high;
            double lower_energy_MeV = total->spline.axis.low;
            for (auto const& column : native_tables.dndx_columns) {
              if (column.pdg_id == pdg &&
                  column.rate_model ==
                      NativeRateModel::BicubicSpline) {
                upper_energy_MeV = std::min(
                    upper_energy_MeV,
                    column.spline.energy_axis.high);
              }
            }
            if (pdg != 22) {
              auto const utility = std::find_if(
                  native_tables.utility_columns.begin(),
                  native_tables.utility_columns.end(),
                  [pdg](auto const& column) {
                    return column.pdg_id == pdg;
                  });
              if (utility == native_tables.utility_columns.end()) {
                throw std::runtime_error(
                    "PROPOSAL native table has no continuous range spline "
                    "for a routed charged lepton");
              }
              lower_energy_MeV = std::max(
                  lower_energy_MeV,
                  std::max(utility->lower_energy_limit_MeV,
                           utility->spline.axis.low));
              upper_energy_MeV = std::min(
                  upper_energy_MeV, utility->spline.axis.high);
              for (auto const& column : native_tables.dedx_columns) {
                if (column.pdg_id == pdg) {
                  upper_energy_MeV = std::min(
                      upper_energy_MeV,
                      column.spline.axis.high);
                }
              }
            }
            auto const code = convert_from_PDG(
                static_cast<PDGCode>(pdg));
            auto const minimum_transport_energy_MeV =
                pdg == 22
                    ? requested_cut_MeV
                    : get_mass(code) / 1_MeV +
                          ContinuousCutSafetyFactor *
                              (std::abs(pdg) == 13
                                   ? mucut / 1_MeV
                                   : requested_cut_MeV);
            auto const lower_scale = std::max(
                {1., std::abs(lower_energy_MeV),
                 std::abs(minimum_transport_energy_MeV)});
            if (minimum_transport_energy_MeV +
                    native_domain_epsilon * lower_scale <
                lower_energy_MeV) {
              throw std::runtime_error(
                  "PROPOSAL native interpolation domain does not cover the "
                  "configured transport cut");
            }
            auto const upper_scale = std::max(
                {1., std::abs(upper_energy_MeV),
                 std::abs(configured_primary_energy_max_MeV)});
            if (configured_primary_energy_max_MeV >
                upper_energy_MeV +
                    native_domain_epsilon * upper_scale) {
              throw std::runtime_error(
                  "configured maximum primary energy exceeds the common "
                  "PROPOSAL native interpolation domain");
            }
          }
          auto const native_aux = loadOrCreateProposalNativeAux(
              native_views, gpu_config.auxiliary_cache_directory);
          cuda_session.backend().initialize(
              environment_snapshot, native_tables, native_aux,
              gpu_config);
        } else {
          cuda_session.backend().initialize(
              environment_snapshot, descriptor, gpu_config);
        }
      } else {
        cuda_session.backend().beginShower(
            makeGpuEmShowerConfig(gpu_config));
      }
      auto& backend = cuda_session.backend();

      using FallbackHandler =
          ProposalCpuFallbackHandler<StackType, TEmCascade, Sequence, EnvType>;
      FallbackHandler fallback_handler{
          emCascade, sequence, env, rootCS, static_cast<std::uint64_t>(seed),
          static_cast<std::uint64_t>(i_shower),
          proposal_stochastic_cut};

      CorsikaOutputSink output_sink{
          rootCS, dEdX, profile, prod_profile, observationLevel,
          inter_writer, coreas, zhs,
          detectorCoREAS.size() != 0, gpu_radio_enabled};
      using OutputSink = decltype(output_sink);
      using Router = PhysicalCudaEmRouter<StackType, FallbackHandler, OutputSink>;
      Router router{backend, rootCS, environment_snapshot, fallback_handler, output_sink};
      router.setRetainRecords(
          validation::CudaReplayTrace::instance().enabled());
      router.setFailOnUnexpectedFallback(true);
      HybridCascade<TrackingType, Sequence, TOutput, StackType, Router> EAS(
          env, tracking, sequence, output, stack, router);
      HadronicWorkClassifierConfig hadronic_work_classifier;
      hadronic_work_classifier.transition_energy_GeV =
          heHadronModelThreshold / 1_GeV;
      EAS.configureHadronicWorkClassification(
          hadronic_work_classifier);
      EAS.enableScalarDetailedPhaseTiming(
          cpu_detailed_step_timing);
      if (hadronic_process_pool) {
        HybridHadronicWorkerConfig
            hadronic_worker_config;
        hadronic_worker_config.seed =
            static_cast<std::uint64_t>(seed);
        hadronic_worker_config.shower_id =
            static_cast<std::uint64_t>(
                output_shower_id);
        hadronic_worker_config
            .minimum_pending_interactions =
            hadronic_min_batch;
        hadronic_worker_config
            .target_batch_cost_ms =
            hadronic_target_batch_ms;
        hadronic_worker_config
            .maximum_batch_items =
            hadronic_max_batch;
        hadronic_worker_config
            .initial_interaction_cost_ms =
            hadronic_initial_cost_ms;
        EAS.configureHadronicProcessPool(
            *hadronic_process_pool,
            hadronic_worker_config);
      }
      configure_forced_primary(EAS);
      EAS.run();

      corsika::applications::air_shower::CudaShowerReportBuilder::
          record<GpuEmStepRegistry, Sequence>(
              cuda_session, gpu_cli, backend, EAS, router, output_sink,
              fallback_handler, photoHadronicHighEnergy,
              photoHadronicLowEnergy, photo_hadronic_before,
              photo_hadronic_le_before, heCounted, leIntCounted,
              high_energy_hadronic_interactions_before,
              low_energy_hadronic_interactions_before,
              high_energy_hadronic_timings_before,
              low_energy_hadronic_timings_before,
              hadronic_work_classifier, hadronic_process_pool,
              hadronic_pool_statistics_before,
              photoHadronicQgsjetFallback,
              high_energy_hadronic_model,
              heHadronModelThreshold, emthinfrac, maxWeight,
              automaticMaxWeight, thinningCanActivateFromUnitWeight,
              backend_reused_for_shower, gpu_muon_transport_available,
              gpu_radio_enabled, gpu_config, beamCode,
              primaryTotalEnergy, output_shower_id);

  }

} // namespace corsika::applications::air_shower

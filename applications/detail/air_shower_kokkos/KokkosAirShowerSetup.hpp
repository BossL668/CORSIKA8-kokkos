/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include "GpuCliOptions.hpp"

#include <corsika/accelerator/em/AcceleratedPhysicsRequirements.hpp>
#include <corsika/accelerator/em/common/EnvironmentSnapshotBuilder.hpp>
#include <corsika/accelerator/em/common/ProcessSequenceCompatibility.hpp>
#include <corsika/accelerator/radio/common/RadioSnapshotBuilder.hpp>
#include <corsika/modules/proposal/ProposalProcessBase.hpp>
#include <corsika/validation/CudaReplayTrace.hpp>

#include <array>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace corsika::applications::air_shower {

  struct KokkosEventConfig {
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
    // Appended to preserve existing aggregate initialization and air defaults.
    AtmosphereId atmosphere_id{AtmosphereId::USStdBK};
    // Comparison-only opt-out. Standard air callers retain both algorithms.
    bool radio_zhs_enabled{true};
  };

  struct PreparedKokkosAirShower {
    gpu::em::EnvironmentSnapshot environment_snapshot;
    gpu::em::GpuEmConfig gpu_config;
    accelerator::em::AcceleratedPhysicsRequirements physics_requirements;
    HEPEnergyType proposal_stochastic_cut{};
    bool gpu_radio_enabled{};
  };

  template <
      typename TStackInspector, typename TNeutrinoPrimary,
      typename THadronSequence, typename TDecaySequence,
      typename TEmCascade, typename TEmContinuous, typename TCoreas,
      typename TZhs, typename TLongitudinalProfile,
      typename TObservationLevel, typename TProductionProfileProcess,
      typename TInteractionWriter, typename TThinning, typename TCut>
  using KokkosAirShowerStepRegistry = gpu::em::GpuEmStepProcessRegistry<
      gpu::em::GpuEmStepProcessRegistration<
          TStackInspector, gpu::em::GpuEmStepProcessPolicy::DiagnosticOnly>,
      gpu::em::GpuEmStepProcessRegistration<
          TNeutrinoPrimary,
          gpu::em::GpuEmStepProcessPolicy::InapplicableToRoutedEm>,
      gpu::em::GpuEmStepProcessRegistration<
          THadronSequence,
          gpu::em::GpuEmStepProcessPolicy::InapplicableToRoutedEm>,
      gpu::em::GpuEmStepProcessRegistration<
          TDecaySequence, gpu::em::GpuEmStepProcessPolicy::DeferredToCpu>,
      gpu::em::GpuEmStepProcessRegistration<
          TEmCascade, gpu::em::GpuEmStepProcessPolicy::ReplacedOnDevice>,
      gpu::em::GpuEmStepProcessRegistration<
          TEmContinuous, gpu::em::GpuEmStepProcessPolicy::ReplacedOnDevice>,
      gpu::em::GpuEmStepProcessRegistration<
          TCoreas,
          gpu::em::GpuEmStepProcessPolicy::ReplayedFromDeviceRecord>,
      gpu::em::GpuEmStepProcessRegistration<
          TZhs, gpu::em::GpuEmStepProcessPolicy::ReplayedFromDeviceRecord>,
      gpu::em::GpuEmStepProcessRegistration<
          TLongitudinalProfile,
          gpu::em::GpuEmStepProcessPolicy::ReplayedFromDeviceRecord>,
      gpu::em::GpuEmStepProcessRegistration<
          TObservationLevel,
          gpu::em::GpuEmStepProcessPolicy::ReplayedFromDeviceRecord>,
      gpu::em::GpuEmStepProcessRegistration<
          TProductionProfileProcess,
          gpu::em::GpuEmStepProcessPolicy::ReplayedFromDeviceRecord>,
      gpu::em::GpuEmStepProcessRegistration<
          TInteractionWriter,
          gpu::em::GpuEmStepProcessPolicy::ReplayedFromDeviceRecord>,
      gpu::em::GpuEmStepProcessRegistration<
          TThinning, gpu::em::GpuEmStepProcessPolicy::ReplacedOnDevice>,
      gpu::em::GpuEmStepProcessRegistration<
          TCut, gpu::em::GpuEmStepProcessPolicy::ReplacedOnDevice>>;

  template <typename TRunSession,
      typename TEnvironment, typename TInjectionPosition, typename TSurface,
      typename TPropagationStep, typename TDetectorCoREAS,
      typename TDetectorZHS, typename TShowerAxis, typename TDepthStep,
      typename TEnergyLossWriter, typename TLongitudinalWriter>
  PreparedKokkosAirShower prepareKokkosAirShower(
      TRunSession const& cuda_session, GpuCliOptions const& gpu_cli,
      KokkosEventConfig const& event, TEnvironment& environment,
      CoordinateSystemPtr const& root_cs,
      TInjectionPosition const& injection_position,
      TSurface const& surface, TPropagationStep const& propagation_step,
      TDetectorCoREAS& detector_coreas, TDetectorZHS& detector_zhs,
      TShowerAxis const& shower_axis, TDepthStep const& depth_step,
      double observation_height_m, double shower_core_x_m,
      double shower_core_y_m,
      std::array<double, 3> const& magnetic_field_T,
      TEnergyLossWriter& energy_loss_writer,
      TLongitudinalWriter& longitudinal_writer) {
    using namespace corsika::gpu::em;
    using namespace corsika::units::si;

    PreparedKokkosAirShower prepared;
    prepared.proposal_stochastic_cut =
        proposal::optimized_proposal_energy_cut(event.production_threshold);
    prepared.physics_requirements = {
        (gpu_cli.gpu_physics_source == "proposal-native"
             ? event.configured_primary_energy_max
             : event.primary_total_energy) /
            1_MeV,
        event.em_cut / 1_MeV, event.muon_cut / 1_MeV,
        prepared.proposal_stochastic_cut / 1_MeV};
    prepared.environment_snapshot = makeCorsika7AtmosphereSnapshot(
        event.atmosphere_id, {0., 0., 0.}, 17, observation_height_m,
        magnetic_field_T, event.maximum_magnetic_deflection_rad);
    setObservationPlane(
        prepared.environment_snapshot,
        {shower_core_x_m, shower_core_y_m, observation_height_m},
        {0., 0., 1.});

    auto& config = prepared.gpu_config;
    config.device = gpu_cli.em_backend == "kokkos"
                        ? gpu_cli.kokkos_device
                        : gpu_cli.gpu_device;
    config.min_batch_size = gpu_cli.gpu_min_batch;
    config.resident_batch_limit = gpu_cli.gpu_resident_batch_limit;
    config.memory_fraction = gpu_cli.gpu_memory_fraction;
    config.em_transport_cut_MeV = event.em_cut / 1_MeV;
    config.muon_transport_cut_MeV = event.muon_cut / 1_MeV;
    config.physics_source = GpuPhysicsSource::ProposalNative;
    config.auxiliary_cache_directory = gpu_cli.gpu_aux_cache_dir;
    config.deterministic = gpu_cli.gpu_deterministic;
    config.detailed_stage_timing = gpu_cli.gpu_detailed_stage_timing;
    config.diagnostic_interaction_records =
        validation::CudaReplayTrace::instance().enabled();
    config.random_seed = event.seed;
    config.shower_id = event.shower_ordinal;
    config.thinning.enabled = event.em_thinning_fraction > 0. ? 1 : 0;
    config.thinning.threshold_GeV =
        event.em_thinning_fraction * event.primary_total_energy / 1_GeV;
    config.thinning.maximum_weight = event.maximum_weight;
    config.thinning.erase_zero_weight =
        event.keep_zero_weight_particles ? 0 : 1;
    config.resident_cross_species = gpu_cli.gpu_resident_cross_species;

    prepared.gpu_radio_enabled =
        gpu_cli.radio_backend == gpu_cli.em_backend &&
        gpu_cli.em_backend != "proposal" && detector_coreas.size() != 0;
    if (prepared.gpu_radio_enabled && !cuda_session.hasBackend()) {
      config.radio = gpu::radio::makeGpuRadioConfig(
          environment, injection_position, surface, propagation_step,
          detector_coreas, detector_zhs, true, event.radio_zhs_enabled);
      config.radio.deterministic = config.deterministic;
      config.radio.fixed_point_field_limit_V_per_m =
          gpu_cli.gpu_radio_field_limit;
      config.radio.track_diagnostics =
          gpu_cli.gpu_radio_track_diagnostics;
    }

    if ((detector_coreas.size() == 0 || prepared.gpu_radio_enabled) &&
        !gpu_cli.gpu_full_step_records) {
      auto& projection = config.profile_projection;
      projection.crossing_mode = longitudinal_writer.getCrossingMode();
      auto const primary_energy_GeV = event.primary_total_energy / 1_GeV;
      auto const em_cut_GeV = event.em_cut / 1_GeV;
      projection.fixed_point_weight_limit =
          2. * primary_energy_GeV / em_cut_GeV;
      // Gross atomic-target rest-mass entries can exceed the primary energy
      // (they cancel against cut electron rest masses). A stopped 0.7 MeV e+
      // followed by six Compton vertices is a concrete example. Keep ample
      // low-energy integer range; this changes neither deposits nor allocation
      // size, and leaves all >=0.5 GeV production scales unchanged.
      projection.fixed_point_energy_limit_GeV = std::max(2. * primary_energy_GeV, 1.);
      if (!cuda_session.hasBackend()) {
        projection.enabled = true;
        auto const axis_start =
            shower_axis.getStart().getCoordinates(root_cs);
        auto const axis_direction =
            shower_axis.getDirection().getComponents(root_cs);
        for (int axis = 0; axis < 3; ++axis) {
          projection.axis_start_position_m[axis] = axis_start[axis] / 1_m;
          projection.axis_direction[axis] = axis_direction[axis].magnitude();
        }
        projection.axis_step_length_m = shower_axis.getSteplength() / 1_m;
        auto const& support = shower_axis.getGrammageSupport();
        projection.axis_grammage_g_per_cm2.reserve(support.size());
        for (auto const grammage : support) {
          projection.axis_grammage_g_per_cm2.push_back(
              grammage / (1_g / square(1_cm)));
        }
        if (energy_loss_writer.GetNBins() !=
            longitudinal_writer.getNBins()) {
          throw std::runtime_error(
              "CUDA resident profile requires identical energy-loss and "
              "longitudinal bin counts");
        }
        projection.accumulate_on_device = true;
        projection.output_bin_count = energy_loss_writer.GetNBins();
        projection.output_bin_width_g_per_cm2 =
            depth_step / (1_g / square(1_cm));
        projection.energy_loss_threshold_g_per_cm2 = 1.e-4;
      }
    }
    return prepared;
  }

} // namespace corsika::applications::air_shower

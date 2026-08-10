/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include <cuda_runtime.h>

#include <cub/device/device_scan.cuh>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <sstream>
#include <stdexcept>

#include <corsika/gpu/em/CudaLeptonTransport.hpp>
#include <corsika/gpu/em/MoliereScattering.hpp>
#include <corsika/gpu/em/Philox.hpp>
#include <corsika/gpu/em/ProposalFallback.hpp>
#include <corsika/gpu/em/SphericalAtmosphere.hpp>
#include <corsika/gpu/em/UniformMagneticField.hpp>
#include <corsika/gpu/em/detail/DeviceBatchStages.hpp>

namespace corsika::gpu::em {

  namespace {

    constexpr unsigned int ThreadsPerBlock = 256;
    constexpr unsigned int TransportThreadsPerBlock = 64;
    constexpr double MaximumContinuousRelativeLoss = 0.10;

    void checkCuda(cudaError_t status, char const* operation) {
      if (status == cudaSuccess) {
        return;
      }
      std::ostringstream message;
      message << operation << " failed: "
              << cudaGetErrorString(status);
      throw std::runtime_error(message.str());
    }

    __device__ ProposalFallbackEvent makeLeptonFallback(
        EmInteractionRecord const& interaction,
        ProposalFallbackReason reason) {
      auto event = makeProcessFallbackEvent(interaction, reason);
      event.input_index = interaction.input_index;
      return event;
    }

    __device__ ProposalFallbackEvent makeTableFallback(
        EmInteractionRecord const& interaction,
        tables::TableLookupStatus status) {
      return makeLeptonFallback(
          interaction, proposalFallbackReason(status));
    }

    __device__ bool closeRadius(double left, double right) {
      auto const scale = left > right ? left : right;
      return ::fabs(left - right) <=
             128. * 2.22044604925031308085e-16 *
                 (scale > 1. ? scale : 1.);
    }

    template <std::size_t ComponentCapacity>
    __device__ bool applyMoliereScattering(
        MoliereSnapshot const& snapshot,
        MoliereInterpolationView const& interpolation,
        bool enabled,
        std::uint64_t random_seed, std::uint64_t shower_id,
        LeptonTransportRecord& record,
        ProposalFallbackEvent& fallback) {
      if (!enabled) {
        return true;
      }
      auto const& start = record.start;
      RandomNumberKey const first_key{
          random_seed, shower_id, start.history_id,
          start.step_id, ContinuousScatteringRandomProcessId,
          MoliereFirstAngleDrawId};
      RandomNumberKey const second_key{
          random_seed, shower_id, start.history_id,
          start.step_id, ContinuousScatteringRandomProcessId,
          MoliereSecondAngleDrawId};
      RandomNumberKey const azimuth_key{
          random_seed, shower_id, start.history_id,
          start.step_id, ContinuousScatteringRandomProcessId,
          MoliereAzimuthDrawId};
      record.multiple_scattering_first_uniform =
          uniformOpen01(first_key);
      record.multiple_scattering_second_uniform =
          uniformOpen01(second_key);
      record.multiple_scattering_azimuth_uniform =
          uniformOpen01(azimuth_key);
      auto const scattering =
          sampleMoliereScatteringAngle2DForCapacity<
              ComponentCapacity>(
          snapshot, interpolation,
          record.traversed_grammage_g_per_cm2,
          record.start.energy_GeV * 1000.,
          record.end.energy_GeV * 1000.,
          record.multiple_scattering_first_uniform,
          record.multiple_scattering_second_uniform,
          // CudaEmBackend accepts this snapshot only after
          // makeMoliereSnapshot() has validated every static coefficient.
          // Per-particle input and convergence checks remain enabled.
          false);
      record.multiple_scattering_status =
          static_cast<std::uint16_t>(scattering.status);
      record.multiple_scattering_iterations =
          scattering.iterations;
      record.multiple_scattering_angle_rad =
          scattering.angle_rad;
      if (scattering.status ==
          MoliereStatus::NoDeflection) {
        return true;
      }
      if (scattering.status != MoliereStatus::Success) {
        fallback = makeLeptonFallback(
            record.interaction,
            ProposalFallbackReason::MoliereSamplingFailed);
        fallback.final_state_uniform =
            record.multiple_scattering_first_uniform;
        fallback.final_state_draw_id =
            MoliereFirstAngleDrawId;
        return false;
      }
      auto const direction =
          applyMoliereDirectionCpuStepCompatible(
          record.start.direction, record.end.direction,
          scattering.angle_rad,
          record.multiple_scattering_azimuth_uniform);
      if (direction.status != MoliereStatus::Success) {
        fallback = makeLeptonFallback(
            record.interaction,
            ProposalFallbackReason::MoliereSamplingFailed);
        fallback.final_state_uniform =
            record.multiple_scattering_azimuth_uniform;
        fallback.final_state_draw_id =
            MoliereAzimuthDrawId;
        return false;
      }
      for (int axis = 0; axis < 3; ++axis) {
        record.end.direction[axis] =
            direction.direction[axis];
      }
      record.multiple_scattering_applied = 1;
      return true;
    }

    /**
     * Write-through flag view that counts only actual transport failures.
     *
     * Transport fallbacks are exceptional. Counting them at the branch that
     * creates the fallback removes a full O(N) flag-counting kernel from every
     * lepton wavefront while preserving the existing stable scan/compaction
     * path whenever the count is nonzero.
     */
    struct CountingFallbackFlagView {
      struct Reference {
        std::uint32_t* value{};
        std::uint32_t* count{};

        __device__ operator std::uint32_t() const {
          return *value;
        }

        __device__ void operator=(std::uint32_t next) const {
          *value = next;
          if (next == 1U) {
            atomicAdd(count, 1U);
          }
        }
      };

      std::uint32_t* values{};
      std::uint32_t* count{};

      __device__ Reference operator[](
          std::size_t index) const {
        return {values + index, count};
      }
    };

    __global__ void transportLeptonsKernel(
        tables::FlatRateTableView table,
        bool apply_moliere,
        EnvironmentSnapshot environment,
        EmInteractionRecord const* interactions,
        EmInteractionRecord const* raw_interactions,
        std::uint32_t const* selection_fallback_count,
        std::size_t count,
        LeptonTransportRecord* raw_records,
        ProposalFallbackEvent* raw_fallbacks,
        CountingFallbackFlagView fallback_flags) {
      auto const index =
          static_cast<std::size_t>(blockIdx.x) * blockDim.x +
          threadIdx.x;
      if (index >= count) {
        return;
      }

      auto const selection_fallbacks =
          selection_fallback_count == nullptr
              ? 0U
              : *selection_fallback_count;
      auto const selected_count =
          count -
          static_cast<std::size_t>(selection_fallbacks);
      if (index >= selected_count) {
        fallback_flags[index] = 0;
        return;
      }
      auto const& interaction =
          selection_fallback_count != nullptr &&
                  selection_fallbacks == 0
              ? raw_interactions[index]
              : interactions[index];
      auto const& start = interaction.particle;
      if (!isChargedLeptonPid(start.pid)) {
        raw_fallbacks[index] = makeLeptonFallback(
            interaction, ProposalFallbackReason::UnsupportedParticle);
        fallback_flags[index] = 1;
        return;
      }
      if (interaction.status != EmInteractionStatus::Selected &&
          interaction.status !=
              EmInteractionStatus::DistanceSampled &&
          interaction.status !=
              EmInteractionStatus::NoDiscreteInteraction &&
          interaction.status !=
              EmInteractionStatus::ParticleCut) {
        raw_fallbacks[index] = makeLeptonFallback(
            interaction, ProposalFallbackReason::InvalidFinalState);
        fallback_flags[index] = 1;
        return;
      }
      if ((interaction.status == EmInteractionStatus::Selected ||
           interaction.status ==
               EmInteractionStatus::DistanceSampled) &&
          (!::isfinite(interaction.interaction_grammage_g_per_cm2) ||
           interaction.interaction_grammage_g_per_cm2 < 0.)) {
        raw_fallbacks[index] = makeLeptonFallback(
            interaction, ProposalFallbackReason::InvalidFinalState);
        fallback_flags[index] = 1;
        return;
      }

      auto const mass = tables::queryContinuousMass(table, start.pid);
      auto const minimum_energy =
          tables::queryContinuousMinimumEnergy(table, start.pid);
      if (mass.status != tables::TableLookupStatus::Success) {
        raw_fallbacks[index] =
            makeTableFallback(interaction, mass.status);
        fallback_flags[index] = 1;
        return;
      }
      if (minimum_energy.status !=
          tables::TableLookupStatus::Success) {
        raw_fallbacks[index] =
            makeTableFallback(interaction, minimum_energy.status);
        fallback_flags[index] = 1;
        return;
      }
      auto const initial_energy_MeV = start.energy_GeV * 1000.;
      if (!(initial_energy_MeV > mass.value)) {
        raw_fallbacks[index] = makeLeptonFallback(
            interaction, ProposalFallbackReason::InvalidFinalState);
        fallback_flags[index] = 1;
        return;
      }
      auto const transport_cut_MeV =
          (minimum_energy.value - mass.value) /
          tables::ContinuousCutSafetyFactor;
      if (!::isfinite(transport_cut_MeV) ||
          !(transport_cut_MeV > 0.)) {
        raw_fallbacks[index] = makeLeptonFallback(
            interaction,
            ProposalFallbackReason::InvalidTableQuery);
        fallback_flags[index] = 1;
        return;
      }

      if (interaction.status ==
              EmInteractionStatus::ParticleCut ||
          initial_energy_MeV - mass.value <
              transport_cut_MeV) {
        auto const layer = queryAtmosphereLayer(
            environment, start.position_m, start.direction);
        if (layer.status != AtmosphereStatus::Success) {
          raw_fallbacks[index] = makeLeptonFallback(
              interaction,
              ProposalFallbackReason::UnsupportedGeometry);
          fallback_flags[index] = 1;
          return;
        }

        // This state is already at the lowest energy represented by the
        // continuous table.  It cannot make a finite transport step, so
        // terminate it in-place exactly as ParticleCut would after a
        // continuous step and deposit all remaining kinetic energy.
        auto& record = raw_records[index];
        record = LeptonTransportRecord{};
        record.interaction = interaction;
        record.interaction.particle_mass_GeV =
            mass.value / 1000.;
        record.start = start;
        record.end = start;
        record.end.step_id++;
        record.input_index = interaction.input_index;
        record.limit = LeptonTransportLimit::ParticleCut;
        record.start_layer_index = layer.layer_index;
        record.end_layer_index = layer.layer_index;
        record.start_density_g_per_cm3 =
            layer.density_g_per_cm3;
        record.end_density_g_per_cm3 =
            layer.density_g_per_cm3;
        record.cut_deposited_energy_GeV =
            (initial_energy_MeV - mass.value) / 1000.;
        // State 2 is a successful zero-length cut that must bypass the
        // split Moliere kernel. That kernel normalizes it to state 0 before
        // fallback counting.
        fallback_flags[index] = apply_moliere ? 2U : 0U;
        return;
      }

      auto const initial_range = tables::queryContinuousRange(
          table, start.pid, initial_energy_MeV);
      if (initial_range.status !=
          tables::TableLookupStatus::Success) {
        raw_fallbacks[index] =
            makeTableFallback(interaction, initial_range.status);
        fallback_flags[index] = 1;
        return;
      }
      auto const relative_limit =
          (1. - MaximumContinuousRelativeLoss) *
          initial_energy_MeV;
      auto const target_energy_MeV =
          relative_limit > minimum_energy.value
              ? relative_limit
              : minimum_energy.value;
      auto const target_range = tables::queryContinuousRange(
          table, start.pid, target_energy_MeV);
      if (target_range.status !=
          tables::TableLookupStatus::Success) {
        raw_fallbacks[index] =
            makeTableFallback(interaction, target_range.status);
        fallback_flags[index] = 1;
        return;
      }
      auto continuous_grammage =
          initial_range.value - target_range.value;
      auto const range_scale =
          initial_range.value > 1. ? initial_range.value : 1.;
      if (continuous_grammage < 0. &&
          continuous_grammage >
              -64. * 2.22044604925031308085e-16 *
                  range_scale) {
        continuous_grammage = 0.;
      }
      if (!::isfinite(continuous_grammage) ||
          continuous_grammage < 0.) {
        raw_fallbacks[index] = makeLeptonFallback(
            interaction, ProposalFallbackReason::InvalidTableQuery);
        fallback_flags[index] = 1;
        return;
      }

      auto const layer = queryAtmosphereLayer(
          environment, start.position_m, start.direction);
      if (layer.status != AtmosphereStatus::Success) {
        raw_fallbacks[index] = makeLeptonFallback(
            interaction, ProposalFallbackReason::UnsupportedGeometry);
        fallback_flags[index] = 1;
        return;
      }
      auto const charge_number = start.pid > 0 ? -1. : 1.;
      auto const magnetic_limit = maximumUniformMagneticStep(
          start, mass.value / 1000., charge_number,
          environment.magnetic_field_T,
          environment.maximum_magnetic_deflection_rad);
      if (magnetic_limit.status ==
              MagneticStepStatus::InvalidInput ||
          magnetic_limit.status ==
              MagneticStepStatus::NonFiniteResult) {
        raw_fallbacks[index] = makeLeptonFallback(
            interaction,
            ProposalFallbackReason::MagneticTransportFailed);
        fallback_flags[index] = 1;
        return;
      }

      auto geometry_distance_m = CUDART_INF;
      auto limiting_radius_m = 0.;
      auto magnetic_step_wins = false;
      auto observation_plane_wins = false;
      if (magnetic_limit.status ==
          MagneticStepStatus::Linear) {
        auto const atmosphere_boundary = distanceToAtmosphereBoundary(
            environment, start.position_m, start.direction);
        auto const observation = intersectObservationPlaneStraight(
            environment, start.position_m, start.direction);
        auto const has_atmosphere_boundary =
            atmosphere_boundary.status == AtmosphereStatus::Success;
        auto const has_observation =
            observation.status == ObservationPlaneStatus::Success;
        if (!has_atmosphere_boundary && !has_observation) {
          raw_fallbacks[index] = makeLeptonFallback(
              interaction,
              ProposalFallbackReason::UnsupportedGeometry);
          fallback_flags[index] = 1;
          return;
        }
        observation_plane_wins =
            has_observation &&
            (!has_atmosphere_boundary ||
             observation.distance_m < atmosphere_boundary.distance_m);
        geometry_distance_m =
            observation_plane_wins
                ? observation.distance_m
                : atmosphere_boundary.distance_m;
        limiting_radius_m =
            observation_plane_wins ? 0.
                                   : atmosphere_boundary.radius_m;
      } else {
        auto const& current_layer =
            environment.atmosphere_layers[layer.layer_index];
        auto const inner_radius_m = current_layer.inner_radius_m;
        auto const inner =
            intersectUniformMagneticSphere(
                start, mass.value / 1000., charge_number,
                environment.magnetic_field_T,
                environment.earth_center_m, inner_radius_m,
                magnetic_limit.distance_m,
                environment.maximum_magnetic_deflection_rad);
        auto const outer =
            intersectUniformMagneticSphere(
                start, mass.value / 1000., charge_number,
                environment.magnetic_field_T,
                environment.earth_center_m,
                current_layer.outer_radius_m,
                magnetic_limit.distance_m,
                environment.maximum_magnetic_deflection_rad);
        auto const observation =
            intersectUniformMagneticPlane(
                start, mass.value / 1000., charge_number,
                environment.magnetic_field_T, environment,
                magnetic_limit.distance_m,
                environment.maximum_magnetic_deflection_rad);
        if (inner.status ==
                MagneticIntersectionStatus::InvalidInput ||
            inner.status ==
                MagneticIntersectionStatus::NonFiniteResult ||
            outer.status ==
                MagneticIntersectionStatus::InvalidInput ||
            outer.status ==
                MagneticIntersectionStatus::NonFiniteResult ||
            observation.status ==
                MagneticIntersectionStatus::InvalidInput ||
            observation.status ==
                MagneticIntersectionStatus::NonFiniteResult) {
          raw_fallbacks[index] = makeLeptonFallback(
              interaction,
              ProposalFallbackReason::MagneticBoundaryFailed);
          fallback_flags[index] = 1;
          return;
        }
        auto const use_inner_sphere =
            inner.status ==
                MagneticIntersectionStatus::Success &&
            (outer.status !=
                 MagneticIntersectionStatus::Success ||
             inner.distance_m < outer.distance_m);
        auto const has_sphere_boundary =
            use_inner_sphere ||
            outer.status ==
                MagneticIntersectionStatus::Success;
        auto const sphere_distance_m =
            has_sphere_boundary
                ? (use_inner_sphere ? inner.distance_m
                                    : outer.distance_m)
                : CUDART_INF;
        auto const has_observation =
            observation.status ==
                MagneticIntersectionStatus::Success;
        observation_plane_wins =
            has_observation &&
            observation.distance_m < sphere_distance_m;
        if (observation_plane_wins) {
          geometry_distance_m = observation.distance_m;
          limiting_radius_m = 0.;
        } else if (has_sphere_boundary) {
          geometry_distance_m =
              sphere_distance_m;
          limiting_radius_m =
              use_inner_sphere ? inner_radius_m
                               : current_layer.outer_radius_m;
        } else {
          geometry_distance_m =
              magnetic_limit.distance_m;
          magnetic_step_wins = true;
        }
      }
      auto const geometry_grammage = atmosphereGrammage(
          environment, layer.layer_index, start.position_m,
          start.direction, geometry_distance_m);
      if (geometry_grammage.status != AtmosphereStatus::Success) {
        auto fallback = makeLeptonFallback(
            interaction, ProposalFallbackReason::AtmosphereGrammageFailed);
        fallback.diagnostic_status =
            static_cast<std::int32_t>(geometry_grammage.status);
        fallback.diagnostic_value0 = geometry_distance_m;
        fallback.diagnostic_value1 = layer.density_g_per_cm3;
        fallback.diagnostic_value2 = limiting_radius_m;
        raw_fallbacks[index] = fallback;
        fallback_flags[index] = 1;
        return;
      }

      auto const has_interaction =
          interaction.status == EmInteractionStatus::Selected ||
          interaction.status ==
              EmInteractionStatus::DistanceSampled;
      auto const interaction_grammage =
          has_interaction
              ? interaction.interaction_grammage_g_per_cm2
              : CUDART_INF;
      auto continuous_wins =
          continuous_grammage < geometry_grammage.value &&
          continuous_grammage < interaction_grammage;
      auto interaction_wins =
          !continuous_wins && has_interaction &&
          interaction_grammage <= geometry_grammage.value;

      // Construct the output directly in its device slot. Keeping this large
      // aggregate as a thread-local value caused local-memory spills in the
      // geometry-heavy kernel; failed slots are ignored by compaction.
      auto& record = raw_records[index];
      record = LeptonTransportRecord{};
      record.interaction = interaction;
      record.interaction.particle_mass_GeV =
          mass.value / 1000.;
      record.start = start;
      record.input_index = interaction.input_index;
      record.start_layer_index = layer.layer_index;
      record.start_density_g_per_cm3 =
          layer.density_g_per_cm3;
      record.continuous_step_grammage_g_per_cm2 =
          continuous_grammage;
      record.limiting_radius_m = limiting_radius_m;
      record.magnetic_step_status =
          static_cast<std::uint32_t>(magnetic_limit.status);
      record.magnetic_step_limit_m =
          magnetic_limit.distance_m;
      record.magnetic_gyroradius_m =
          magnetic_limit.gyroradius_m;

      double traversed_grammage = geometry_grammage.value;
      double distance_m = geometry_distance_m;
      if (continuous_wins) {
        traversed_grammage = continuous_grammage;
      } else if (interaction_wins) {
        traversed_grammage = interaction_grammage;
      }
      if (continuous_wins || interaction_wins) {
        auto const distance = atmosphereDistanceFromGrammage(
            environment, layer.layer_index, start.position_m,
            start.direction, traversed_grammage);
        if (distance.status != AtmosphereStatus::Success ||
            distance.value >
                geometry_distance_m * (1. + 1.e-12)) {
          auto fallback = makeLeptonFallback(
              interaction,
              ProposalFallbackReason::AtmosphereInverseGrammageFailed);
          fallback.diagnostic_status =
              static_cast<std::int32_t>(distance.status);
          fallback.diagnostic_value0 = distance.value;
          fallback.diagnostic_value1 = geometry_distance_m;
          fallback.diagnostic_value2 = traversed_grammage;
          raw_fallbacks[index] = fallback;
          fallback_flags[index] = 1;
          return;
        }
        distance_m = distance.value;
      }
      auto const has_decay =
          isMuonPid(start.pid) &&
          ::isfinite(interaction.decay_distance_m) &&
          interaction.decay_distance_m >= 0.;
      auto const decay_wins =
          has_decay &&
          interaction.decay_distance_m <= distance_m;
      if (decay_wins) {
        continuous_wins = false;
        interaction_wins = false;
        distance_m = interaction.decay_distance_m;
        auto const decay_grammage = atmosphereGrammage(
            environment, layer.layer_index, start.position_m,
            start.direction, distance_m);
        if (decay_grammage.status !=
            AtmosphereStatus::Success) {
          auto fallback = makeLeptonFallback(
              interaction,
              ProposalFallbackReason::AtmosphereGrammageFailed);
          fallback.diagnostic_status =
              static_cast<std::int32_t>(
                  decay_grammage.status);
          fallback.diagnostic_value0 = distance_m;
          raw_fallbacks[index] = fallback;
          fallback_flags[index] = 1;
          return;
        }
        traversed_grammage = decay_grammage.value;
      }
      auto const observation_reached =
          observation_plane_wins && !continuous_wins &&
          !interaction_wins && !decay_wins &&
          !magnetic_step_wins;

      auto const magnetic_advance =
          advanceUniformMagneticField(
              start, mass.value / 1000., charge_number,
              environment.magnetic_field_T, distance_m);
      if (magnetic_advance.status ==
              MagneticStepStatus::InvalidInput ||
          magnetic_advance.status ==
              MagneticStepStatus::NonFiniteResult) {
        raw_fallbacks[index] = makeLeptonFallback(
            interaction,
            ProposalFallbackReason::MagneticTransportFailed);
        fallback_flags[index] = 1;
        return;
      }
      record.end = magnetic_advance.particle;
      if (observation_reached) {
        double radial[3]{};
        record.limiting_radius_m =
            atmosphere_detail::radiusVector(
                environment, record.end.position_m, radial);
      }
      record.magnetic_bending_applied =
          magnetic_advance.status ==
                  MagneticStepStatus::Success &&
                  magnetic_advance.bend_parameter != 0.
              ? 1U
              : 0U;
      record.magnetic_bend_parameter =
          magnetic_advance.bend_parameter;
      record.magnetic_chord_length_m =
          magnetic_advance.chord_length_m;
      record.distance_m = distance_m;

      // Scalar CORSIKA uses the original leapfrog trajectory when it turns a
      // sampled grammage into the competing step length.  Once the endpoint
      // has been selected, however, Step retains only the displacement and
      // ContinuousProcess integrates the medium along
      // Step::getStraightTrack(): the chord between the two endpoints.  Keep
      // the start-tangent grammage above for distance competition, then match
      // the scalar continuous-process semantics here after magnetic advance.
      double chord_direction[3]{
          start.direction[0], start.direction[1],
          start.direction[2]};
      auto continuous_chord_length_squared = 0.;
      for (int axis = 0; axis < 3; ++axis) {
        auto const displacement =
            record.end.position_m[axis] - start.position_m[axis];
        chord_direction[axis] = displacement;
        continuous_chord_length_squared += displacement * displacement;
      }
      auto const continuous_chord_length_m =
          ::sqrt(continuous_chord_length_squared);
      if (continuous_chord_length_m > 0.) {
        for (int axis = 0; axis < 3; ++axis) {
          chord_direction[axis] /= continuous_chord_length_m;
        }
      } else {
        for (int axis = 0; axis < 3; ++axis) {
          chord_direction[axis] = start.direction[axis];
        }
      }
      auto const chord_grammage = atmosphereGrammage(
          environment, layer.layer_index, start.position_m,
          chord_direction, continuous_chord_length_m);
      if (chord_grammage.status != AtmosphereStatus::Success) {
        auto fallback = makeLeptonFallback(
            interaction,
            ProposalFallbackReason::AtmosphereGrammageFailed);
        fallback.diagnostic_status =
            static_cast<std::int32_t>(chord_grammage.status);
        fallback.diagnostic_value0 =
            continuous_chord_length_m;
        fallback.diagnostic_value1 = distance_m;
        fallback.diagnostic_value2 = traversed_grammage;
        raw_fallbacks[index] = fallback;
        fallback_flags[index] = 1;
        return;
      }
      traversed_grammage = chord_grammage.value;
      record.traversed_grammage_g_per_cm2 =
          traversed_grammage;
      auto const final_energy =
          tables::queryEnergyAfterContinuousLoss(
              table, start.pid, initial_energy_MeV,
              traversed_grammage);
      auto transport_cut_reached = false;
      double final_energy_MeV{};
      if (final_energy.status ==
          tables::TableLookupStatus::Success) {
        final_energy_MeV = final_energy.value;
      } else if (final_energy.status ==
                 tables::TableLookupStatus::TransportCutReached) {
        // The scalar sequence applies ParticleCut after continuous loss and
        // before any candidate discrete interaction.  The table stops just
        // below that cut, so clamp to its minimum represented energy and
        // deposit the remaining kinetic energy through the ordinary cut
        // record below.
        final_energy_MeV = minimum_energy.value;
        transport_cut_reached = true;
      } else {
        raw_fallbacks[index] =
            makeTableFallback(interaction, final_energy.status);
        fallback_flags[index] = 1;
        return;
      }
      auto const actual_continuous_loss_MeV =
          initial_energy_MeV - final_energy_MeV;
      if (continuous_wins &&
          !(actual_continuous_loss_MeV > 0.)) {
        auto fallback = makeLeptonFallback(
            interaction,
            ProposalFallbackReason::InvalidTableQuery);
        fallback.diagnostic_value0 = initial_energy_MeV;
        fallback.diagnostic_value1 = final_energy_MeV;
        fallback.diagnostic_value2 = traversed_grammage;
        raw_fallbacks[index] = fallback;
        fallback_flags[index] = 1;
        return;
      }
      record.end.energy_GeV = final_energy_MeV / 1000.;
      record.continuous_deposited_energy_GeV =
          (initial_energy_MeV - final_energy_MeV) / 1000.;

      // In the scalar sequence, continuous loss, multiple scattering,
      // profile/radio projection and ObservationPlane all see this completed
      // step before the final ParticleCut check.  Terminate at the endpoint
      // (rather than truncating at exactly 10 ms), suppress any pending
      // interaction/decay, and retain the dual observation+cut outcome when
      // the observation plane was the limiting process.
      if (exceedsParticleCutTime(record.end.time_s)) {
        record.end.step_id++;
        record.limit = LeptonTransportLimit::ParticleCut;
        record.end_layer_index = layer.layer_index;
        record.end_density_g_per_cm3 =
            layer.density_g_per_cm3;
        record.cut_deposited_energy_GeV =
            (final_energy_MeV - mass.value) / 1000.;
        record.observation_surface_reached_before_cut =
            observation_reached ? 1U : 0U;
        fallback_flags[index] = 0;
        return;
      }

      auto const reaches_cut =
          transport_cut_reached ||
          final_energy_MeV - mass.value <
              transport_cut_MeV;
      if (reaches_cut && !observation_reached) {
        auto const end_layer = queryAtmosphereLayer(
            environment, record.end.position_m,
            record.end.direction);
        if (end_layer.status != AtmosphereStatus::Success) {
          raw_fallbacks[index] = makeLeptonFallback(
              interaction,
              ProposalFallbackReason::UnsupportedGeometry);
          fallback_flags[index] = 1;
          return;
        }
        record.end.step_id++;
        record.limit = LeptonTransportLimit::ParticleCut;
        record.end_layer_index = end_layer.layer_index;
        record.end_density_g_per_cm3 =
            end_layer.density_g_per_cm3;
        record.cut_deposited_energy_GeV =
            (final_energy_MeV - mass.value) / 1000.;
        fallback_flags[index] = 0;
        return;
      }

      if (decay_wins) {
        auto const vertex = queryAtmosphereLayer(
            environment, record.end.position_m,
            record.end.direction);
        if (vertex.status != AtmosphereStatus::Success) {
          raw_fallbacks[index] = makeLeptonFallback(
              interaction,
              ProposalFallbackReason::
                  AtmosphereVertexLookupFailed);
          fallback_flags[index] = 1;
          return;
        }
        record.limit = LeptonTransportLimit::DecayCandidate;
        record.end_layer_index = vertex.layer_index;
        record.end_density_g_per_cm3 =
            vertex.density_g_per_cm3;
        record.interaction.particle = record.end;
        record.interaction.status =
            EmInteractionStatus::Selected;
        record.interaction.process_id = DecayProcessId;
        record.interaction.component_hash = 0;
        record.interaction.energy_fraction = 0.;
        record.interaction.loss_quantile = 0.;
        fallback_flags[index] = 0;
        return;
      }

      if (interaction_wins) {
        auto const vertex = queryAtmosphereLayer(
            environment, record.end.position_m,
            record.end.direction);
        if (vertex.status != AtmosphereStatus::Success ||
            !::isfinite(vertex.density_g_per_cm3) ||
            !(vertex.density_g_per_cm3 > 0.)) {
          auto fallback = makeLeptonFallback(
              interaction,
              ProposalFallbackReason::AtmosphereVertexLookupFailed);
          fallback.diagnostic_status =
              static_cast<std::int32_t>(vertex.status);
          fallback.diagnostic_value0 = vertex.radius_m;
          fallback.diagnostic_value1 = vertex.density_g_per_cm3;
          fallback.diagnostic_value2 = distance_m;
          raw_fallbacks[index] = fallback;
          fallback_flags[index] = 1;
          return;
        }
        record.limit =
            LeptonTransportLimit::InteractionCandidate;
        record.end_layer_index = vertex.layer_index;
        record.end_density_g_per_cm3 =
            vertex.density_g_per_cm3;
        record.interaction.particle = record.end;
        record.interaction.status =
            EmInteractionStatus::RequiresReselection;
        record.interaction.process_id = 0;
        record.interaction.component_hash = 0;
        record.interaction.energy_fraction = 0.;
        record.interaction.loss_quantile = 0.;
        record.interaction.mass_density_g_per_cm3 =
            vertex.density_g_per_cm3;
        fallback_flags[index] = 0;
        return;
      }

      record.end.step_id++;
      if (continuous_wins) {
        auto const end_layer = queryAtmosphereLayer(
            environment, record.end.position_m,
            record.end.direction);
        if (end_layer.status != AtmosphereStatus::Success) {
          raw_fallbacks[index] = makeLeptonFallback(
              interaction, ProposalFallbackReason::UnsupportedGeometry);
          fallback_flags[index] = 1;
          return;
        }
        record.limit =
            LeptonTransportLimit::ContinuousStep;
        record.end_layer_index = end_layer.layer_index;
        record.end_density_g_per_cm3 =
            end_layer.density_g_per_cm3;
        fallback_flags[index] = 0;
        return;
      }

      if (magnetic_step_wins) {
        auto const end_layer = queryAtmosphereLayer(
            environment, record.end.position_m,
            record.end.direction);
        if (end_layer.status != AtmosphereStatus::Success) {
          auto fallback = makeLeptonFallback(
              interaction,
              ProposalFallbackReason::MagneticBoundaryFailed);
          fallback.diagnostic_status =
              static_cast<std::int32_t>(end_layer.status);
          fallback.diagnostic_value0 = end_layer.radius_m;
          fallback.diagnostic_value1 =
              magnetic_limit.distance_m;
          raw_fallbacks[index] = fallback;
          fallback_flags[index] = 1;
          return;
        }
        if (end_layer.layer_index != layer.layer_index) {
          // queryAtmosphereLayer deliberately assigns a point within the
          // 0.1 mm tracking guard to the layer selected by its direction.
          // The bounded curved-sphere solver can correctly report no root
          // when the root lies only a few micrometres beyond the magnetic
          // step.  Accept that direction-owned adjacent-layer transition;
          // otherwise a valid maximum-deflection step is misclassified as
          // a numerical transport failure on the next ownership query.
          auto const outward =
              end_layer.layer_index == layer.layer_index + 1;
          auto const inward =
              end_layer.layer_index == layer.layer_index - 1;
          auto const shared_radius_m =
              outward
                  ? environment
                        .atmosphere_layers[layer.layer_index]
                        .outer_radius_m
                  : (inward
                         ? environment
                               .atmosphere_layers[layer.layer_index]
                               .inner_radius_m
                         : 0.);
          auto const floating_tolerance_m =
              128. * 2.22044604925031308085e-16 *
              (shared_radius_m > 1. ? shared_radius_m : 1.);
          auto const boundary_tolerance_m =
              floating_tolerance_m > AtmosphereBoundaryGuardM
                  ? floating_tolerance_m
                  : AtmosphereBoundaryGuardM;
          if ((outward || inward) &&
              ::fabs(end_layer.radius_m - shared_radius_m) <=
                  boundary_tolerance_m) {
            record.limit = LeptonTransportLimit::LayerBoundary;
            record.limiting_radius_m = shared_radius_m;
            record.end_layer_index = end_layer.layer_index;
            record.end_density_g_per_cm3 =
                end_layer.density_g_per_cm3;
            record.end.medium_id =
                environment
                    .atmosphere_layers[end_layer.layer_index]
                    .medium_id;
            fallback_flags[index] = 0;
            return;
          }
          auto fallback = makeLeptonFallback(
              interaction,
              ProposalFallbackReason::MagneticBoundaryFailed);
          fallback.diagnostic_status = end_layer.layer_index;
          fallback.diagnostic_value0 = end_layer.radius_m;
          fallback.diagnostic_value1 = shared_radius_m;
          fallback.diagnostic_value2 =
              magnetic_limit.distance_m;
          raw_fallbacks[index] = fallback;
          fallback_flags[index] = 1;
          return;
        }
        record.limit = LeptonTransportLimit::MagneticStep;
        record.end_layer_index = end_layer.layer_index;
        record.end_density_g_per_cm3 =
            end_layer.density_g_per_cm3;
        fallback_flags[index] = 0;
        return;
      }

      if (observation_reached) {
        record.limit =
            LeptonTransportLimit::ObservationSurface;
        record.end_layer_index = -1;
        record.end_density_g_per_cm3 =
            layer.density_g_per_cm3;
      } else {
        auto const outermost =
            environment.atmosphere_layers[
                environment.number_of_layers - 1]
                .outer_radius_m;
        if (closeRadius(limiting_radius_m, outermost)) {
          record.limit =
              LeptonTransportLimit::EscapedEnvironment;
          record.end_layer_index = -1;
          record.end_density_g_per_cm3 = 0.;
        } else {
          auto const next = queryAtmosphereLayer(
              environment, record.end.position_m,
              record.end.direction);
          if (next.status != AtmosphereStatus::Success ||
              next.layer_index == layer.layer_index) {
            raw_fallbacks[index] = makeLeptonFallback(
                interaction,
                ProposalFallbackReason::UnsupportedGeometry);
            fallback_flags[index] = 1;
            return;
          }
          record.limit =
              LeptonTransportLimit::LayerBoundary;
          record.end_layer_index = next.layer_index;
          record.end.medium_id =
              environment
                  .atmosphere_layers[next.layer_index]
                  .medium_id;
          record.end_density_g_per_cm3 =
              next.density_g_per_cm3;
        }
      }
      fallback_flags[index] = 0;
    }

    /**
     * Moliere sampling is intentionally isolated from atmospheric transport.
     *
     * Keeping both algorithms in transportLeptonsKernel forced all geometry,
     * magnetic-field and scattering temporaries to share one thread frame,
     * producing a large local stack and very high register pressure. This
     * second kernel consumes the already materialized transport record and
     * preserves the same Philox keys and record fields as the fused path.
     */
    template <std::size_t ComponentCapacity>
    __global__ void applyMoliereScatteringKernel(
        MoliereSnapshot snapshot,
        MoliereSnapshot muon_snapshot,
        MoliereInterpolationView interpolation,
        bool muon_snapshot_available,
        std::uint64_t random_seed,
        std::uint64_t shower_id,
        LeptonTransportRecord* raw_records,
        ProposalFallbackEvent* raw_fallbacks,
        CountingFallbackFlagView fallback_flags,
        std::size_t count,
        std::uint32_t const* selection_fallback_count) {
      auto const index =
          static_cast<std::size_t>(blockIdx.x) * blockDim.x +
          threadIdx.x;
      if (index >= count) {
        return;
      }
      if (selection_fallback_count != nullptr &&
          index >=
              count -
                  static_cast<std::size_t>(
                      *selection_fallback_count)) {
        return;
      }
      auto const state =
          static_cast<std::uint32_t>(
              fallback_flags[index]);
      if (state == 2U) {
        fallback_flags[index] = 0U;
        return;
      }
      if (state != 0U) {
        return;
      }
      auto& record = raw_records[index];
      auto const muon = isMuonPid(record.start.pid);
      if (muon && !muon_snapshot_available) {
        raw_fallbacks[index] = makeLeptonFallback(
            record.interaction,
            ProposalFallbackReason::
                MoliereParametersUnavailable);
        fallback_flags[index] = 1U;
        return;
      }
      auto const& particle_snapshot =
          muon ? muon_snapshot : snapshot;
      if (!applyMoliereScattering<ComponentCapacity>(
              particle_snapshot, interpolation, true,
              random_seed, shower_id,
              record, raw_fallbacks[index])) {
        fallback_flags[index] = 1U;
        return;
      }
      // Interaction reselection must see the scattered vertex direction.
      if (record.limit ==
              LeptonTransportLimit::InteractionCandidate ||
          record.limit ==
              LeptonTransportLimit::DecayCandidate) {
        record.interaction.particle = record.end;
      }
    }

    __global__ void compactLeptonTransportKernel(
        LeptonTransportRecord const* raw_records,
        ProposalFallbackEvent const* raw_fallbacks,
        std::uint32_t const* fallback_flags,
        std::uint32_t const* fallback_offsets, std::size_t count,
        LeptonTransportRecord* compact_records,
        ProposalFallbackEvent* compact_fallbacks) {
      auto const index =
          static_cast<std::size_t>(blockIdx.x) * blockDim.x +
          threadIdx.x;
      if (index >= count) {
        return;
      }
      auto const fallbacks_before = fallback_offsets[index];
      if (fallback_flags[index] != 0) {
        compact_fallbacks[fallbacks_before] =
            raw_fallbacks[index];
      } else {
        compact_records[index - fallbacks_before] =
            raw_records[index];
      }
    }

    __global__ void classifyLeptonInteractionsKernel(
        LeptonTransportRecord const* records, std::size_t count,
        std::uint32_t* interaction_flags) {
      auto const index =
          static_cast<std::size_t>(blockIdx.x) * blockDim.x +
          threadIdx.x;
      if (index < count) {
        interaction_flags[index] =
            records[index].limit ==
                    LeptonTransportLimit::InteractionCandidate
                ? 1U
                : 0U;
      }
    }

    __global__ void compactLeptonInteractionsKernel(
        LeptonTransportRecord const* records,
        std::uint32_t const* interaction_flags,
        std::uint32_t const* interaction_offsets,
        std::size_t count,
        EmInteractionRecord* compact_interactions) {
      auto const index =
          static_cast<std::size_t>(blockIdx.x) * blockDim.x +
          threadIdx.x;
      if (index < count && interaction_flags[index] != 0) {
        compact_interactions[interaction_offsets[index]] =
            records[index].interaction;
      }
    }

    __global__ void finalizeLeptonInteractionCountKernel(
        std::uint32_t const* interaction_flags,
        std::uint32_t const* interaction_offsets,
        std::size_t count,
        std::uint32_t* interaction_count) {
      if (blockIdx.x == 0 && threadIdx.x == 0) {
        auto const last = count - 1;
        *interaction_count =
            interaction_offsets[last] +
            interaction_flags[last];
      }
    }

  } // namespace

  namespace detail {

    void appendLeptonTransportWorkspace(
        WorkspaceSize& required, std::size_t count) {
      std::size_t scan_bytes = 0;
      checkCuda(cub::DeviceScan::ExclusiveSum(
                    nullptr, scan_bytes,
                    static_cast<std::uint32_t*>(nullptr),
                    static_cast<std::uint32_t*>(nullptr), count),
                "query lepton transport scan storage");
      required.add<LeptonTransportRecord>(count);
      required.add<ProposalFallbackEvent>(count);
      required.add<std::uint32_t>(count);
      required.add<std::uint32_t>(count);
      required.add<LeptonTransportRecord>(count);
      required.add<ProposalFallbackEvent>(count);
      required.add<std::uint32_t>(2);
      required.addBytes(scan_bytes);
    }

    DeviceLeptonTransportBatch launchLeptonTransportOnDevice(
        tables::FlatRateTableView device_table,
        MoliereSnapshot const& moliere_snapshot,
        MoliereSnapshot const& muon_moliere_snapshot,
        MoliereInterpolationView const& moliere_interpolation,
        bool apply_moliere, bool muon_moliere_available,
        EnvironmentSnapshot const& environment,
        EmInteractionRecord const* device_interactions,
        std::size_t count, std::uint64_t random_seed,
        std::uint64_t shower_id, DeviceWorkspace& workspace,
        DeviceInteractionSelectionBatch*
            deferred_selection,
        LeptonPipelineStageEvents const* stage_events) {
      if (count == 0 || device_interactions == nullptr) {
        throw std::invalid_argument(
            "device lepton transport requires a non-empty input");
      }
      std::size_t scan_bytes = 0;
      checkCuda(cub::DeviceScan::ExclusiveSum(
                    nullptr, scan_bytes,
                    static_cast<std::uint32_t*>(nullptr),
                    static_cast<std::uint32_t*>(nullptr), count),
                "query lepton transport scan storage");
      auto* raw_records =
          workspace.acquire<LeptonTransportRecord>(count);
      auto* raw_fallbacks =
          workspace.acquire<ProposalFallbackEvent>(count);
      auto* fallback_flags =
          workspace.acquire<std::uint32_t>(count);
      auto* fallback_offsets =
          workspace.acquire<std::uint32_t>(count);
      auto* compact_records =
          workspace.acquire<LeptonTransportRecord>(count);
      auto* compact_fallbacks =
          workspace.acquire<ProposalFallbackEvent>(count);
      auto* device_summary =
          workspace.acquire<std::uint32_t>(2);
      auto* device_fallback_count =
          device_summary + 1;
      auto* scan_temporary =
          workspace.acquireBytes(scan_bytes);

      auto const transport_block_count =
          (count + TransportThreadsPerBlock - 1) /
          TransportThreadsPerBlock;
      if (transport_block_count >
          std::numeric_limits<unsigned int>::max()) {
        throw std::length_error(
            "lepton transport launch is too large");
      }
      auto const transport_blocks =
          static_cast<unsigned int>(
              transport_block_count);
      auto const block_count =
          (count + ThreadsPerBlock - 1) / ThreadsPerBlock;
      if (block_count >
          std::numeric_limits<unsigned int>::max()) {
        throw std::length_error(
            "lepton transport compaction launch is too large");
      }
      auto const blocks = static_cast<unsigned int>(block_count);
      checkCuda(
          cudaMemset(
              device_summary, 0,
              2 * sizeof(std::uint32_t)),
          "clear lepton transport summary");
      auto const* selection_fallback_count =
          deferred_selection == nullptr
              ? nullptr
              : deferred_selection
                    ->device_fallback_count;
      auto const* selection_raw_interactions =
          deferred_selection == nullptr
              ? device_interactions
              : deferred_selection->raw_interactions;
      if (deferred_selection != nullptr &&
          (!deferred_selection->counts_deferred ||
           selection_fallback_count == nullptr ||
           selection_raw_interactions == nullptr)) {
        throw std::logic_error(
            "lepton transport received an invalid deferred selection");
      }
      CountingFallbackFlagView const counting_fallback_flags{
          fallback_flags, device_fallback_count};
      transportLeptonsKernel
          <<<transport_blocks, TransportThreadsPerBlock>>>(
          device_table, apply_moliere, environment,
          device_interactions, selection_raw_interactions,
          selection_fallback_count, count,
          raw_records, raw_fallbacks,
          counting_fallback_flags);
      checkCuda(cudaGetLastError(),
                "lepton transport kernel launch");
      if (stage_events != nullptr) {
        checkCuda(
            cudaEventRecord(
                stage_events->transport_physics_done),
            "record lepton transport physics stage");
      }
      if (apply_moliere) {
        constexpr std::size_t AirComponentCapacity = 4;
        if (moliere_snapshot.component_count <=
            AirComponentCapacity) {
          applyMoliereScatteringKernel<
              AirComponentCapacity>
              <<<transport_blocks,
                 TransportThreadsPerBlock>>>(
                  moliere_snapshot, muon_moliere_snapshot,
                  moliere_interpolation,
                  muon_moliere_available,
                  random_seed, shower_id,
                  raw_records, raw_fallbacks,
                  counting_fallback_flags,
                  count, selection_fallback_count);
        } else {
          applyMoliereScatteringKernel<
              MaxMoliereComponents>
              <<<transport_blocks,
                 TransportThreadsPerBlock>>>(
                  moliere_snapshot, muon_moliere_snapshot,
                  moliere_interpolation,
                  muon_moliere_available,
                  random_seed, shower_id,
                  raw_records, raw_fallbacks,
                  counting_fallback_flags,
                  count, selection_fallback_count);
        }
        checkCuda(
            cudaGetLastError(),
            "Moliere scattering kernel launch");
      }
      if (stage_events != nullptr) {
        checkCuda(
            cudaEventRecord(stage_events->moliere_done),
            "record lepton Moliere stage");
      }
      if (selection_fallback_count != nullptr) {
        checkCuda(
            cudaMemcpy(
                device_summary, selection_fallback_count,
                sizeof(std::uint32_t),
                cudaMemcpyDeviceToDevice),
            "copy selection count into lepton transport summary");
      }
      std::array<std::uint32_t, 2> host_summary{};
      checkCuda(
          cudaMemcpy(
              host_summary.data(), device_summary,
              sizeof(host_summary),
              cudaMemcpyDeviceToHost),
          "download lepton transport control summary");
      auto const selection_fallbacks =
          static_cast<std::size_t>(host_summary[0]);
      if (selection_fallbacks > count) {
        throw std::runtime_error(
            "lepton selection fallback count exceeds its input");
      }
      auto const selection_interactions =
          count - selection_fallbacks;
      auto const fallback_count =
          static_cast<std::size_t>(host_summary[1]);
      if (fallback_count > selection_interactions) {
        throw std::runtime_error(
            "lepton transport fallback count exceeds selected interactions");
      }
      auto const record_count =
          selection_interactions - fallback_count;
      if (stage_events != nullptr) {
        checkCuda(
            cudaEventRecord(
                stage_events->transport_control_done),
            "record lepton transport control stage");
      }
      if (deferred_selection != nullptr) {
        deferred_selection->fallback_count =
            selection_fallbacks;
        deferred_selection->interaction_count =
            selection_interactions;
        deferred_selection->counts_deferred = false;
        if (selection_fallbacks == 0) {
          deferred_selection->interactions =
              deferred_selection->raw_interactions;
        }
      }
      auto* selected_records = raw_records;
      if (fallback_count != 0) {
        checkCuda(cub::DeviceScan::ExclusiveSum(
                      scan_temporary, scan_bytes,
                      fallback_flags, fallback_offsets,
                      selection_interactions),
                  "scan lepton transport fallback flags");
        compactLeptonTransportKernel<<<blocks, ThreadsPerBlock>>>(
            raw_records, raw_fallbacks, fallback_flags,
            fallback_offsets, selection_interactions,
            compact_records,
            compact_fallbacks);
        checkCuda(cudaGetLastError(),
                  "compact lepton transport kernel launch");
        selected_records = compact_records;
      }
      return {selection_interactions, record_count,
              fallback_count,
              selected_records, compact_fallbacks};
    }

    void appendLeptonInteractionWorkspace(
        WorkspaceSize& required, std::size_t count) {
      std::size_t scan_bytes = 0;
      checkCuda(cub::DeviceScan::ExclusiveSum(
                    nullptr, scan_bytes,
                    static_cast<std::uint32_t*>(nullptr),
                    static_cast<std::uint32_t*>(nullptr), count),
                "query lepton-interaction scan storage");
      required.add<std::uint32_t>(count);
      required.add<std::uint32_t>(count);
      required.add<EmInteractionRecord>(count);
      required.add<std::uint32_t>(1);
      required.addBytes(scan_bytes);
    }

    DeviceTransportInteractionBatch
    extractLeptonInteractionsOnDevice(
        LeptonTransportRecord const* device_records,
        std::size_t count, DeviceWorkspace& workspace,
        bool defer_count_download) {
      if (count == 0 || device_records == nullptr) {
        throw std::invalid_argument(
            "lepton interaction extraction requires records");
      }
      std::size_t scan_bytes = 0;
      checkCuda(cub::DeviceScan::ExclusiveSum(
                    nullptr, scan_bytes,
                    static_cast<std::uint32_t*>(nullptr),
                    static_cast<std::uint32_t*>(nullptr), count),
                "query lepton-interaction scan storage");
      auto* flags = workspace.acquire<std::uint32_t>(count);
      auto* offsets = workspace.acquire<std::uint32_t>(count);
      auto* interactions =
          workspace.acquire<EmInteractionRecord>(count);
      auto* device_interaction_count =
          workspace.acquire<std::uint32_t>(1);
      auto* scan_temporary =
          workspace.acquireBytes(scan_bytes);
      auto const blocks = static_cast<unsigned int>(
          (count + ThreadsPerBlock - 1) / ThreadsPerBlock);
      classifyLeptonInteractionsKernel<<<blocks, ThreadsPerBlock>>>(
          device_records, count, flags);
      checkCuda(cudaGetLastError(),
                "classify lepton interactions launch");
      checkCuda(cub::DeviceScan::ExclusiveSum(
                    scan_temporary, scan_bytes, flags, offsets, count),
                "scan lepton interaction flags");
      finalizeLeptonInteractionCountKernel<<<1, 1>>>(
          flags, offsets, count, device_interaction_count);
      checkCuda(
          cudaGetLastError(),
          "finalize lepton transport interaction count launch");
      compactLeptonInteractionsKernel<<<blocks, ThreadsPerBlock>>>(
          device_records, flags, offsets, count, interactions);
      checkCuda(cudaGetLastError(),
                "compact lepton interactions launch");
      std::size_t interaction_count = 0;
      if (!defer_count_download) {
        std::uint32_t host_interaction_count = 0;
        checkCuda(
            cudaMemcpy(
                &host_interaction_count,
                device_interaction_count,
                sizeof(host_interaction_count),
                cudaMemcpyDeviceToHost),
            "download lepton interaction count");
        interaction_count = host_interaction_count;
      }
      return {count, interaction_count, interactions,
              device_interaction_count,
              defer_count_download};
    }

  } // namespace detail

  LeptonTransportBatchResult transportLeptonsStraightForValidation(
      tables::FlatRateTableView device_table,
      EnvironmentSnapshot const& environment,
      std::vector<EmInteractionRecord> const& interactions, int device,
      detail::DeviceWorkspace& workspace) {
    LeptonTransportBatchResult result{};
    result.input_interactions = interactions.size();
    if (interactions.empty()) {
      return result;
    }
    if (device < 0) {
      throw std::invalid_argument(
          "lepton transport CUDA device must be non-negative");
    }
    if (!atmosphere_detail::validEnvironment(environment)) {
      throw std::invalid_argument(
          "lepton transport received an invalid environment snapshot");
    }
    for (double component : environment.magnetic_field_T) {
      if (!std::isfinite(component) || component != 0.) {
        throw std::invalid_argument(
            "straight lepton validation transport requires zero magnetic field");
      }
    }
    if (interactions.size() >
        std::numeric_limits<std::uint32_t>::max()) {
      throw std::length_error(
          "lepton transport batch exceeds 32-bit scan offsets");
    }
    checkCuda(cudaSetDevice(device),
              "cudaSetDevice(lepton transport)");

    auto const count = interactions.size();
    detail::WorkspaceSize required;
    required.add<EmInteractionRecord>(count);
    detail::appendLeptonTransportWorkspace(required, count);
    workspace.prepare(required.bytes());

    auto* device_interactions =
        workspace.acquire<EmInteractionRecord>(count);
    checkCuda(cudaMemcpy(
                  device_interactions, interactions.data(),
                  count * sizeof(EmInteractionRecord),
                  cudaMemcpyHostToDevice),
              "upload lepton transport interactions");
    auto const batch = detail::launchLeptonTransportOnDevice(
        device_table, {}, {}, {}, false, false, environment,
        device_interactions, count, 0, 0, workspace);

    result.records.resize(batch.record_count);
    result.fallback_events.resize(batch.fallback_count);
    if (batch.record_count != 0) {
      checkCuda(cudaMemcpy(
                    result.records.data(), batch.records,
                    batch.record_count *
                        sizeof(LeptonTransportRecord),
                    cudaMemcpyDeviceToHost),
                "download lepton transport records");
    }
    if (batch.fallback_count != 0) {
      checkCuda(cudaMemcpy(
                    result.fallback_events.data(), batch.fallbacks,
                    batch.fallback_count *
                        sizeof(ProposalFallbackEvent),
                    cudaMemcpyDeviceToHost),
                "download lepton transport fallbacks");
    }
    return result;
  }

} // namespace corsika::gpu::em

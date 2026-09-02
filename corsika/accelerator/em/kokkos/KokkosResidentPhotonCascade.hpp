/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

#include <corsika/accelerator/em/detail/InteractionSelection.hpp>
#include <corsika/accelerator/em/detail/ProfileProjectionStep.hpp>
#include <corsika/accelerator/em/detail/PhotonFinalStateStep.hpp>
#include <corsika/accelerator/em/detail/PhotonTransportStep.hpp>
#include <corsika/accelerator/em/kokkos/KokkosProfileAccumulator.hpp>
#include <corsika/accelerator/em/kokkos/KokkosWavefrontQueue.hpp>

namespace corsika::accelerator::em::kokkos_detail {

  struct ResidentPhotonSourceOutcome {
    detail::InteractionSelectionOutcome selection{};
    detail::PhotonTransportOutcome transport{};
    detail::PhotonFinalStateClassification final_state{};
    std::uint32_t selected{};
    std::uint32_t transported{};
    std::uint32_t at_interaction{};
  };

  struct ResidentPhotonSourceCounts {
    std::uint32_t child{};
    std::uint32_t next{};
    std::uint32_t charged{};
    std::uint32_t fallback{};
    std::uint32_t observation{};
    std::uint32_t step{};
    std::uint32_t record{};
  };

  enum ResidentPhotonOffset : std::size_t {
    PhotonChildOffset = 0,
    PhotonNextOffset = 1,
    PhotonChargedOffset = 2,
    PhotonFallbackOffset = 3,
    PhotonObservationOffset = 4,
    PhotonStepOffset = 5,
    PhotonRecordOffset = 6,
    PhotonOffsetCount = 7,
  };

  template <class ExecutionSpace>
  gpu::em::ResidentPhotonCascadeResult runResidentPhotonCascade(
      gpu::em::tables::FlatRateTableView const physics,
      gpu::em::PhotonPairLpmSnapshot const lpm_snapshot,
      gpu::em::EmThinningConfig const thinning,
      gpu::em::EnvironmentSnapshot const environment,
      std::vector<gpu::em::EmParticleState> const& particles,
      std::uint64_t const random_seed, std::uint64_t const shower_id,
      std::uint64_t const first_secondary_history_id,
      std::size_t const maximum_wavefronts,
      std::size_t const minimum_resident_batch_size,
      std::optional<gpu::em::GpuFirstInteractionSnapshot>& first_interaction,
      bool const project_steps = false,
      gpu::em::detail::DeviceProfileProjection const profile_projection = {},
      KokkosProfileAccumulator<ExecutionSpace>* const profile_accumulator =
          nullptr,
      ExecutionSpace const& execution = {}) {
    using Memory = typename ExecutionSpace::memory_space;
    using Policy = Kokkos::RangePolicy<ExecutionSpace>;
    using namespace gpu::em;
    ResidentPhotonCascadeResult result{};
    result.input_particles = particles.size();
    if (particles.empty()) {
      result.completed = true;
      return result;
    }
    if (first_secondary_history_id == 0 || maximum_wavefronts == 0 ||
        minimum_resident_batch_size == 0)
      throw std::invalid_argument(
          "resident Kokkos photon cascade requires nonzero limits");

    auto const capacity = particles.size();
    KokkosWavefrontQueue<ExecutionSpace> queue(capacity);
    queue.upload(particles);
    Kokkos::View<ResidentPhotonSourceOutcome*, Memory> outcomes(
        "c8_kokkos_resident_photon_outcomes", capacity);
    Kokkos::View<ResidentPhotonSourceCounts*, Memory> counts(
        "c8_kokkos_resident_photon_counts", capacity);
    Kokkos::View<std::uint64_t*[PhotonOffsetCount], Memory> offsets(
        "c8_kokkos_resident_photon_offsets", capacity);
    Kokkos::View<PhotonTransportRecord*, Memory> steps(
        "c8_kokkos_resident_photon_steps", capacity);
    Kokkos::View<ProjectedEmStepRecord*, Memory> projected_steps(
        "c8_kokkos_resident_photon_projected_steps", capacity);
    Kokkos::View<PhotonPairFinalStateRecord*, Memory> records(
        "c8_kokkos_resident_photon_records", capacity);
    Kokkos::View<EmParticleState*, Memory> charged(
        "c8_kokkos_resident_photon_charged", 2 * capacity);
    Kokkos::View<ProposalFallbackEvent*, Memory> fallbacks(
        "c8_kokkos_resident_photon_fallbacks", capacity);
    Kokkos::View<ObservationRecord*, Memory> observations(
        "c8_kokkos_resident_photon_observations", capacity);
    Kokkos::View<GpuFirstInteractionSnapshot*, Memory> first_snapshots(
        "c8_kokkos_resident_photon_first_snapshots", capacity);
    Kokkos::View<std::uint32_t*, Memory> first_flags(
        "c8_kokkos_resident_photon_first_flags", capacity);
    Kokkos::View<std::uint32_t*, Memory> error(
        "c8_kokkos_resident_photon_error", 1);

    auto append = [&](auto& destination, auto const& source,
                      std::size_t const count) {
      if (count == 0) return;
      auto sub = Kokkos::subview(source, std::make_pair<std::size_t>(0, count));
      auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), sub);
      auto const old = destination.size();
      destination.resize(old + count);
      for (std::size_t i = 0; i < count; ++i)
        destination[old + i] = host(i);
    };

    std::uint64_t next_history_id = first_secondary_history_id;
    result.peak_resident_photons = queue.size();
    while (!queue.empty() && result.wavefronts < maximum_wavefronts) {
      auto const current_count = queue.size();
      if (result.wavefronts != 0 &&
          current_count < minimum_resident_batch_size) {
        result.below_minimum_batch_checkpoint = true;
        break;
      }
      Kokkos::deep_copy(execution, first_flags, std::uint32_t{0});
      Kokkos::deep_copy(execution, error, std::uint32_t{0});
      auto const current = queue.current();
      Kokkos::parallel_for(
          "c8_kokkos_resident_photon_classify",
          Policy(execution, 0, current_count),
          KOKKOS_LAMBDA(std::size_t const source) {
            ResidentPhotonSourceOutcome outcome{};
            ResidentPhotonSourceCounts source_counts{};
            auto const particle = current.load(source);
            outcome.selection = detail::selectDiscreteInteraction(
                physics, particle, source, random_seed, shower_id);
            if (outcome.selection.fallback_flag != 0) {
              source_counts.fallback = 1;
            } else {
              outcome.selected = 1;
              outcome.transport = detail::transportPhoton(
                  environment, outcome.selection.interaction);
              if (outcome.transport.fallback_flag != 0) {
                source_counts.fallback = 1;
              } else {
                outcome.transported = 1;
                source_counts.step = 1;
                auto const& step = outcome.transport.record;
                if (step.limit == PhotonTransportLimit::LayerBoundary) {
                  source_counts.next = 1;
                } else if (
                    step.limit == PhotonTransportLimit::ObservationSurface ||
                    step.limit == PhotonTransportLimit::EscapedEnvironment ||
                    (step.limit == PhotonTransportLimit::ParticleCut &&
                     step.observation_surface_reached_before_cut != 0)) {
                  source_counts.observation = 1;
                } else if (step.limit == PhotonTransportLimit::Interaction) {
                  outcome.at_interaction = 1;
                  outcome.final_state = detail::classifyPhotonFinalState(
                      physics, lpm_snapshot, thinning, step.interaction,
                      random_seed, shower_id);
                  auto const& final = outcome.final_state;
                  source_counts.child = final.child_count;
                  source_counts.record = final.record_flag;
                  source_counts.fallback = final.fallback_flag;
                  if (final.suppression_flag || final.continuation_flag) {
                    source_counts.next = 1;
                  } else if (final.record_flag) {
                    auto const process = final.parameters.process_id;
                    if (process == ComptonProcessId) {
                      source_counts.next =
                          (final.parameters.thinning_keep_mask & 0x1U) != 0;
                      source_counts.charged =
                          (final.parameters.thinning_keep_mask & 0x2U) != 0;
                    } else {
                      source_counts.charged = final.child_count;
                    }
                  }
                }
              }
            }
            outcomes(source) = outcome;
            counts(source) = source_counts;
          });

      std::uint64_t totals[PhotonOffsetCount]{};
#define C8_KOKKOS_PHOTON_SCAN(NAME, COLUMN, MEMBER)                         \
      Kokkos::parallel_scan(                                                \
          NAME, Policy(execution, 0, current_count),                        \
          KOKKOS_LAMBDA(std::size_t const source, std::uint64_t& update,     \
                        bool const final) {                                  \
            if (final) offsets(source, COLUMN) = update;                    \
            update += counts(source).MEMBER;                                \
          },                                                                \
          totals[COLUMN])
      C8_KOKKOS_PHOTON_SCAN("c8_kokkos_scan_photon_children",
                            PhotonChildOffset, child);
      C8_KOKKOS_PHOTON_SCAN("c8_kokkos_scan_photon_next", PhotonNextOffset,
                            next);
      C8_KOKKOS_PHOTON_SCAN("c8_kokkos_scan_photon_charged",
                            PhotonChargedOffset, charged);
      C8_KOKKOS_PHOTON_SCAN("c8_kokkos_scan_photon_fallback",
                            PhotonFallbackOffset, fallback);
      C8_KOKKOS_PHOTON_SCAN("c8_kokkos_scan_photon_observation",
                            PhotonObservationOffset, observation);
      C8_KOKKOS_PHOTON_SCAN("c8_kokkos_scan_photon_step", PhotonStepOffset,
                            step);
      C8_KOKKOS_PHOTON_SCAN("c8_kokkos_scan_photon_record",
                            PhotonRecordOffset, record);
#undef C8_KOKKOS_PHOTON_SCAN
      if (totals[PhotonNextOffset] > capacity ||
          totals[PhotonChargedOffset] > 2 * capacity)
        throw std::runtime_error(
            "resident Kokkos photon endpoint capacity exceeded");

      auto next = queue.next();
      Kokkos::parallel_for(
          "c8_kokkos_resident_photon_materialize",
          Policy(execution, 0, current_count),
          KOKKOS_LAMBDA(std::size_t const source) {
            auto const outcome = outcomes(source);
            auto const source_counts = counts(source);
            if (source_counts.step) {
              steps(offsets(source, PhotonStepOffset)) =
                  outcome.transport.record;
              if (project_steps)
                projected_steps(offsets(source, PhotonStepOffset)) =
                    detail::projectPhotonStep(
                        profile_projection, outcome.transport.record);
            }
            if (outcome.selection.fallback_flag) {
              fallbacks(offsets(source, PhotonFallbackOffset)) =
                  outcome.selection.fallback;
              return;
            }
            if (outcome.transport.fallback_flag) {
              fallbacks(offsets(source, PhotonFallbackOffset)) =
                  outcome.transport.fallback;
              return;
            }
            auto const& step = outcome.transport.record;
            if (source_counts.observation) {
              ObservationRecord observation{};
              observation.particle = step.end;
              observation.status =
                  step.limit == PhotonTransportLimit::EscapedEnvironment
                      ? ObservationStatus::EscapedEnvironment
                      : ObservationStatus::ReachedObservationSurface;
              observations(offsets(source, PhotonObservationOffset)) =
                  observation;
            }
            if (step.limit == PhotonTransportLimit::LayerBoundary) {
              next.store(offsets(source, PhotonNextOffset), step.end);
              return;
            }
            if (!outcome.at_interaction) return;
            auto const materialized = detail::materializePhotonFinalState(
                step.interaction, outcome.final_state,
                offsets(source, PhotonChildOffset), next_history_id);
            if (materialized.error) {
              Kokkos::atomic_compare_exchange(
                  error.data(), std::uint32_t{0}, materialized.error);
              return;
            }
            if (materialized.has_fallback) {
              fallbacks(offsets(source, PhotonFallbackOffset)) =
                  materialized.fallback;
              return;
            }
            if (materialized.has_continuation) {
              next.store(offsets(source, PhotonNextOffset),
                         materialized.continuation.particle);
              return;
            }
            if (materialized.has_suppression) {
              next.store(offsets(source, PhotonNextOffset),
                         materialized.suppression.particle);
              return;
            }
            if (!materialized.has_record) return;
            records(offsets(source, PhotonRecordOffset)) = materialized.record;
            std::uint32_t charged_index = 0;
            for (std::uint32_t child = 0;
                 child < materialized.secondary_count; ++child) {
              auto const particle = materialized.secondaries[child];
              if (particle.pid == static_cast<std::int32_t>(EmPid::Photon)) {
                next.store(offsets(source, PhotonNextOffset), particle);
              } else {
                charged(offsets(source, PhotonChargedOffset) +
                        charged_index++) = particle;
              }
            }
            if (materialized.has_first_interaction) {
              first_flags(source) = 1;
              first_snapshots(source) = materialized.first_interaction;
            }
          });
      execution.fence("complete resident Kokkos photon wavefront");
      auto host_error =
          Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), error);
      if (host_error(0) != 0)
        throw std::runtime_error(
            "resident Kokkos photon final-state materialization failed");

      if (profile_accumulator != nullptr)
        profile_accumulator->accumulatePhoton(
            profile_projection, steps, totals[PhotonStepOffset], records,
            totals[PhotonRecordOffset], execution);
      if (project_steps)
        append(result.projected_step_records, projected_steps,
               totals[PhotonStepOffset]);
      else if (profile_accumulator == nullptr)
        append(result.step_records, steps, totals[PhotonStepOffset]);
      // Match the native-CUDA resident-output contract: when the complete
      // profile ledger is accumulated on the execution space, final-state
      // records remain resident as well.  Returning them without their
      // matching transport records would make the host router account the
      // same vertex a second time and breaks the history ledger.
      if (profile_accumulator == nullptr)
        append(result.final_state_records, records,
               totals[PhotonRecordOffset]);
      append(result.electromagnetic_secondaries, charged,
             totals[PhotonChargedOffset]);
      append(result.fallback_events, fallbacks,
             totals[PhotonFallbackOffset]);
      append(result.observations, observations,
             totals[PhotonObservationOffset]);
      if (!first_interaction) {
        auto host_flags = Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace(), first_flags);
        auto host_snapshots = Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace(), first_snapshots);
        for (std::size_t source = 0; source < current_count; ++source) {
          if (host_flags(source)) {
            first_interaction = host_snapshots(source);
            break;
          }
        }
      }
      auto host_steps = Kokkos::create_mirror_view_and_copy(
          Kokkos::HostSpace(), Kokkos::subview(
              steps, std::make_pair<std::size_t>(0,
                                                 totals[PhotonStepOffset])));
      for (std::size_t i = 0; i < totals[PhotonStepOffset]; ++i) {
        auto const limit = host_steps(i).limit;
        if (limit == PhotonTransportLimit::Interaction)
          result.interaction_vertices++;
        else if (limit == PhotonTransportLimit::LayerBoundary)
          result.layer_boundaries++;
        else if (limit == PhotonTransportLimit::ParticleCut)
          result.particle_cuts++;
      }
      result.transport_records += totals[PhotonStepOffset];
      auto host_outcomes = Kokkos::create_mirror_view_and_copy(
          Kokkos::HostSpace(), Kokkos::subview(
              outcomes, std::make_pair<std::size_t>(0, current_count)));
      for (std::size_t source = 0; source < current_count; ++source) {
        auto const& final = host_outcomes(source).final_state;
        result.lpm_suppressions += final.suppression_flag;
        auto& statistics = result.process_statistics;
        statistics.gpu_final_states += final.record_flag;
        if (final.record_flag != 0) {
          statistics.physical_secondaries_generated += final.child_count;
          statistics.photon_pair_final_states += final.photon_pair_flag;
          statistics.compton_final_states += final.compton_flag;
          statistics.photoelectric_final_states += final.photoelectric_flag;
          auto const status = static_cast<EmThinningStatus>(
              final.parameters.thinning_status);
          statistics.thinning_hillas_vertices +=
              status == EmThinningStatus::Hillas;
          statistics.thinning_statistical_vertices +=
              status == EmThinningStatus::Statistical;
          auto const original_multiplicity =
              final.parameters.process_id == PhotoelectricProcessId ? 1U : 2U;
          if (final.child_count <= original_multiplicity)
            statistics.thinning_particles_discarded +=
                original_multiplicity - final.child_count;
        }
        statistics.photon_pair_lpm_trials +=
            final.photon_pair_flag + final.suppression_flag;
        statistics.photon_pair_lpm_suppressions += final.suppression_flag;
      }
      if (totals[PhotonChildOffset] >
          std::numeric_limits<std::uint64_t>::max() - next_history_id)
        throw std::overflow_error(
            "resident Kokkos photon history ID overflow");
      next_history_id += totals[PhotonChildOffset];
      queue.commitNext(totals[PhotonNextOffset]);
      result.wavefronts++;
      result.peak_resident_photons =
          std::max(result.peak_resident_photons, queue.size());
    }
    result.completed = queue.empty();
    if (!result.completed) result.remaining_photons = queue.download();
    return result;
  }

} // namespace corsika::accelerator::em::kokkos_detail

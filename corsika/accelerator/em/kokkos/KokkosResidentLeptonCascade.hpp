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
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

#include <corsika/accelerator/em/detail/InteractionSelection.hpp>
#include <corsika/accelerator/em/detail/LeptonFinalStateStep.hpp>
#include <corsika/accelerator/em/detail/LeptonTransportStep.hpp>
#include <corsika/accelerator/em/detail/LeptonVertexSelection.hpp>
#include <corsika/accelerator/em/detail/MoliereStep.hpp>
#include <corsika/accelerator/em/kokkos/KokkosWavefrontQueue.hpp>

namespace corsika::accelerator::em::kokkos_detail {

  struct ResidentLeptonSourceOutcome {
    detail::InteractionSelectionOutcome selection{};
    gpu::em::LeptonTransportRecord transport{};
    gpu::em::ProposalFallbackEvent transport_fallback{};
    detail::LeptonVertexSelectionOutcome vertex{};
    detail::LeptonFinalStateClassification final_state{};
    std::uint32_t transported{};
    std::uint32_t at_interaction{};
  };

  struct ResidentLeptonSourceCounts {
    std::uint32_t child{};
    std::uint32_t next{};
    std::uint32_t photon{};
    std::uint32_t fallback{};
    std::uint32_t observation{};
    std::uint32_t decay{};
    std::uint32_t step{};
    std::uint32_t record{};
  };

  enum ResidentLeptonOffset : std::size_t {
    LeptonChildOffset = 0,
    LeptonNextOffset = 1,
    LeptonPhotonOffset = 2,
    LeptonFallbackOffset = 3,
    LeptonObservationOffset = 4,
    LeptonDecayOffset = 5,
    LeptonStepOffset = 6,
    LeptonRecordOffset = 7,
    LeptonOffsetCount = 8,
  };

  template <std::size_t MoliereComponentCapacity>
  KOKKOS_INLINE_FUNCTION void classifyResidentLeptonSource(
      gpu::em::tables::FlatRateTableView const& physics,
      gpu::em::EnvironmentSnapshot const& environment,
      gpu::em::MoliereSnapshot const& electron_moliere,
      gpu::em::MoliereSnapshot const& muon_moliere,
      gpu::em::MoliereInterpolationView const& moliere_interpolation,
      bool const muon_moliere_available,
      gpu::em::BremsLpmSnapshot const& brems_lpm,
      gpu::em::EmThinningConfig const& thinning,
      gpu::em::EmParticleState const& particle, std::size_t const source,
      std::uint64_t const random_seed, std::uint64_t const shower_id,
      ResidentLeptonSourceOutcome& outcome,
      ResidentLeptonSourceCounts& source_counts) {
    using namespace gpu::em;
    outcome = {};
    source_counts = {};
    outcome.selection = detail::selectDiscreteInteraction(
        physics, particle, source, random_seed, shower_id);
    if (outcome.selection.fallback_flag != 0) {
      source_counts.fallback = 1;
      return;
    }

    auto transport_state = detail::transportLepton(
        physics, true, environment, outcome.selection.interaction,
        outcome.transport, outcome.transport_fallback);
    transport_state = detail::applyMoliereScatteringStage<
        MoliereComponentCapacity>(
        electron_moliere, muon_moliere, moliere_interpolation,
        muon_moliere_available, random_seed, shower_id, outcome.transport,
        outcome.transport_fallback, transport_state);
    if (transport_state != 0) {
      source_counts.fallback = 1;
      return;
    }

    outcome.transported = 1;
    source_counts.step = 1;
    auto const& step = outcome.transport;
    switch (step.limit) {
    case LeptonTransportLimit::ContinuousStep:
    case LeptonTransportLimit::LayerBoundary:
    case LeptonTransportLimit::MagneticStep:
      source_counts.next = 1;
      return;
    case LeptonTransportLimit::ObservationSurface:
    case LeptonTransportLimit::EscapedEnvironment:
      source_counts.observation = 1;
      return;
    case LeptonTransportLimit::ParticleCut:
      source_counts.observation =
          step.observation_surface_reached_before_cut != 0;
      return;
    case LeptonTransportLimit::DecayCandidate:
      source_counts.decay = 1;
      return;
    case LeptonTransportLimit::InteractionCandidate:
      break;
    }

    outcome.at_interaction = 1;
    outcome.vertex = detail::selectLeptonVertex(
        physics, step.interaction, random_seed, shower_id);
    if (outcome.vertex.fallback_flag != 0) {
      source_counts.fallback = 1;
      return;
    }
    if (outcome.vertex.continuation_flag != 0) {
      source_counts.next = 1;
      return;
    }
    if (outcome.vertex.interaction_flag == 0) {
      source_counts.fallback = 1;
      outcome.vertex.fallback = detail::vertexFallback(
          step.interaction, ProposalFallbackReason::InvalidFinalState);
      outcome.vertex.fallback_flag = 1;
      return;
    }

    outcome.final_state = detail::classifyLeptonFinalState(
        brems_lpm, thinning, outcome.vertex.record, random_seed, shower_id);
    auto const& final = outcome.final_state;
    source_counts.child = final.child_count;
    source_counts.record = final.record_flag;
    source_counts.fallback = final.fallback_flag;
    if (final.continuation_flag || final.suppression_flag) {
      source_counts.next = 1;
      return;
    }
    if (!final.record_flag) return;

    auto const process = final.parameters.process_id;
    if (process == BremsProcessId) {
      source_counts.next =
          (final.parameters.thinning_keep_mask & 0x1U) != 0;
      source_counts.photon =
          (final.parameters.thinning_keep_mask & 0x2U) != 0;
    } else if (process == AnnihilationProcessId) {
      source_counts.photon = final.child_count;
    } else if (process == IonizationProcessId) {
      source_counts.next = final.child_count;
    } else if (process == ElectronPairProcessId) {
      source_counts.next = final.child_count;
    }
  }

  template <class ExecutionSpace>
  gpu::em::ResidentLeptonCascadeResult runResidentLeptonCascade(
      gpu::em::tables::FlatRateTableView const physics,
      gpu::em::EnvironmentSnapshot const environment,
      gpu::em::MoliereSnapshot const electron_moliere,
      gpu::em::MoliereSnapshot const muon_moliere,
      gpu::em::MoliereInterpolationView const moliere_interpolation,
      bool const muon_moliere_available,
      gpu::em::BremsLpmSnapshot const brems_lpm,
      gpu::em::EmThinningConfig const thinning,
      std::vector<gpu::em::EmParticleState> const& particles,
      std::uint64_t const random_seed, std::uint64_t const shower_id,
      std::uint64_t const first_secondary_history_id,
      std::size_t const maximum_wavefronts,
      std::uint64_t const secondary_history_id_limit_exclusive,
      std::size_t const minimum_resident_batch_size,
      std::optional<gpu::em::GpuFirstInteractionSnapshot>& first_interaction,
      ExecutionSpace const& execution = {}) {
    using Memory = typename ExecutionSpace::memory_space;
    using Policy = Kokkos::RangePolicy<ExecutionSpace>;
    using namespace gpu::em;
    ResidentLeptonCascadeResult result{};
    result.input_particles = particles.size();
    if (particles.empty()) {
      result.completed = true;
      return result;
    }
    if (first_secondary_history_id == 0 || maximum_wavefronts == 0 ||
        secondary_history_id_limit_exclusive <= first_secondary_history_id ||
        minimum_resident_batch_size == 0)
      throw std::invalid_argument(
          "resident Kokkos lepton cascade requires valid limits");
    if (particles.size() > std::numeric_limits<std::size_t>::max() / 3)
      throw std::length_error(
          "resident Kokkos lepton input exceeds addressable capacity");

    // A source can produce at most three charged children.  Keeping three
    // input fronts gives ordinary Epair growth room while retaining a strict,
    // lossless checkpoint if a later front exceeds this per-call workspace.
    auto const capacity = particles.size() * 3;
    KokkosWavefrontQueue<ExecutionSpace> queue(capacity);
    queue.upload(particles);
    Kokkos::View<ResidentLeptonSourceOutcome*, Memory> outcomes(
        "c8_kokkos_resident_lepton_outcomes", capacity);
    Kokkos::View<ResidentLeptonSourceCounts*, Memory> counts(
        "c8_kokkos_resident_lepton_counts", capacity);
    Kokkos::View<std::uint64_t*[LeptonOffsetCount], Memory> offsets(
        "c8_kokkos_resident_lepton_offsets", capacity);
    Kokkos::View<LeptonTransportRecord*, Memory> steps(
        "c8_kokkos_resident_lepton_steps", capacity);
    Kokkos::View<BremsFinalStateRecord*, Memory> records(
        "c8_kokkos_resident_lepton_records", capacity);
    Kokkos::View<EmParticleState*, Memory> photons(
        "c8_kokkos_resident_lepton_photons", 2 * capacity);
    Kokkos::View<ProposalFallbackEvent*, Memory> fallbacks(
        "c8_kokkos_resident_lepton_fallbacks", capacity);
    Kokkos::View<ObservationRecord*, Memory> observations(
        "c8_kokkos_resident_lepton_observations", capacity);
    Kokkos::View<EmParticleState*, Memory> decays(
        "c8_kokkos_resident_lepton_decays", capacity);
    Kokkos::View<GpuFirstInteractionSnapshot*, Memory> first_snapshots(
        "c8_kokkos_resident_lepton_first_snapshots", capacity);
    Kokkos::View<std::uint32_t*, Memory> first_flags(
        "c8_kokkos_resident_lepton_first_flags", capacity);
    Kokkos::View<std::uint32_t*, Memory> error(
        "c8_kokkos_resident_lepton_error", 1);

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
    result.peak_resident_leptons = queue.size();
    while (!queue.empty() && result.wavefronts < maximum_wavefronts) {
      auto const current_count = queue.size();
      if (result.wavefronts != 0 &&
          current_count < minimum_resident_batch_size) {
        result.below_minimum_batch_checkpoint = true;
        break;
      }
      auto const remaining_history_ids =
          secondary_history_id_limit_exclusive - next_history_id;
      if (current_count > remaining_history_ids / 3) {
        result.history_range_exhausted = true;
        break;
      }

      Kokkos::deep_copy(execution, first_flags, std::uint32_t{0});
      Kokkos::deep_copy(execution, error, std::uint32_t{0});
      auto const current = queue.current();
      if (electron_moliere.component_count <= 4u) {
        Kokkos::parallel_for(
            "c8_kokkos_resident_lepton_classify_air",
            Policy(execution, 0, current_count),
            KOKKOS_LAMBDA(std::size_t const source) {
              auto outcome = ResidentLeptonSourceOutcome{};
              auto source_counts = ResidentLeptonSourceCounts{};
              classifyResidentLeptonSource<4>(
                  physics, environment, electron_moliere, muon_moliere,
                  moliere_interpolation, muon_moliere_available, brems_lpm,
                  thinning, current.load(source), source, random_seed,
                  shower_id, outcome, source_counts);
              outcomes(source) = outcome;
              counts(source) = source_counts;
            });
      } else {
        Kokkos::parallel_for(
            "c8_kokkos_resident_lepton_classify_general",
            Policy(execution, 0, current_count),
            KOKKOS_LAMBDA(std::size_t const source) {
              auto outcome = ResidentLeptonSourceOutcome{};
              auto source_counts = ResidentLeptonSourceCounts{};
              classifyResidentLeptonSource<MaxMoliereComponents>(
                  physics, environment, electron_moliere, muon_moliere,
                  moliere_interpolation, muon_moliere_available, brems_lpm,
                  thinning, current.load(source), source, random_seed,
                  shower_id, outcome, source_counts);
              outcomes(source) = outcome;
              counts(source) = source_counts;
            });
      }

      std::uint64_t totals[LeptonOffsetCount]{};
#define C8_KOKKOS_LEPTON_SCAN(NAME, COLUMN, MEMBER)                         \
      Kokkos::parallel_scan(                                                \
          NAME, Policy(execution, 0, current_count),                        \
          KOKKOS_LAMBDA(std::size_t const source, std::uint64_t& update,     \
                        bool const final) {                                  \
            if (final) offsets(source, COLUMN) = update;                    \
            update += counts(source).MEMBER;                                \
          },                                                                \
          totals[COLUMN])
      C8_KOKKOS_LEPTON_SCAN("c8_kokkos_scan_lepton_children",
                            LeptonChildOffset, child);
      C8_KOKKOS_LEPTON_SCAN("c8_kokkos_scan_lepton_next", LeptonNextOffset,
                            next);
      C8_KOKKOS_LEPTON_SCAN("c8_kokkos_scan_lepton_photons",
                            LeptonPhotonOffset, photon);
      C8_KOKKOS_LEPTON_SCAN("c8_kokkos_scan_lepton_fallbacks",
                            LeptonFallbackOffset, fallback);
      C8_KOKKOS_LEPTON_SCAN("c8_kokkos_scan_lepton_observations",
                            LeptonObservationOffset, observation);
      C8_KOKKOS_LEPTON_SCAN("c8_kokkos_scan_lepton_decays",
                            LeptonDecayOffset, decay);
      C8_KOKKOS_LEPTON_SCAN("c8_kokkos_scan_lepton_steps", LeptonStepOffset,
                            step);
      C8_KOKKOS_LEPTON_SCAN("c8_kokkos_scan_lepton_records",
                            LeptonRecordOffset, record);
#undef C8_KOKKOS_LEPTON_SCAN

      if (totals[LeptonNextOffset] > capacity ||
          totals[LeptonPhotonOffset] > 2 * capacity) {
        result.workspace_limit_checkpoint = true;
        break;
      }
      if (totals[LeptonChildOffset] >
          secondary_history_id_limit_exclusive - next_history_id) {
        result.history_range_exhausted = true;
        break;
      }

      auto next = queue.next();
      auto const electron_mass_GeV = brems_lpm.lepton_mass_MeV / 1000.;
      Kokkos::parallel_for(
          "c8_kokkos_resident_lepton_materialize",
          Policy(execution, 0, current_count),
          KOKKOS_LAMBDA(std::size_t const source) {
            auto const outcome = outcomes(source);
            auto const source_counts = counts(source);
            if (source_counts.step)
              steps(offsets(source, LeptonStepOffset)) = outcome.transport;
            if (outcome.selection.fallback_flag) {
              fallbacks(offsets(source, LeptonFallbackOffset)) =
                  outcome.selection.fallback;
              return;
            }
            if (!outcome.transported) {
              fallbacks(offsets(source, LeptonFallbackOffset)) =
                  outcome.transport_fallback;
              return;
            }

            auto const& step = outcome.transport;
            if (source_counts.observation) {
              ObservationRecord observation{};
              observation.particle = step.end;
              observation.status =
                  step.limit == LeptonTransportLimit::EscapedEnvironment
                      ? ObservationStatus::EscapedEnvironment
                      : ObservationStatus::ReachedObservationSurface;
              observations(offsets(source, LeptonObservationOffset)) =
                  observation;
            }
            if (source_counts.decay)
              decays(offsets(source, LeptonDecayOffset)) = step.end;
            if (step.limit == LeptonTransportLimit::ContinuousStep ||
                step.limit == LeptonTransportLimit::LayerBoundary ||
                step.limit == LeptonTransportLimit::MagneticStep) {
              next.store(offsets(source, LeptonNextOffset), step.end);
              return;
            }
            if (!outcome.at_interaction) return;
            if (outcome.vertex.fallback_flag) {
              fallbacks(offsets(source, LeptonFallbackOffset)) =
                  outcome.vertex.fallback;
              return;
            }
            if (outcome.vertex.continuation_flag) {
              next.store(offsets(source, LeptonNextOffset),
                         outcome.vertex.record.particle);
              return;
            }

            auto const materialized = detail::materializeLeptonFinalState(
                outcome.vertex.record, outcome.final_state,
                offsets(source, LeptonChildOffset), next_history_id,
                electron_mass_GeV);
            if (materialized.error) {
              Kokkos::atomic_compare_exchange(
                  error.data(), std::uint32_t{0}, materialized.error);
              return;
            }
            if (materialized.has_fallback) {
              fallbacks(offsets(source, LeptonFallbackOffset)) =
                  materialized.fallback;
              return;
            }
            if (materialized.has_continuation) {
              next.store(offsets(source, LeptonNextOffset),
                         materialized.continuation.particle);
              return;
            }
            if (materialized.has_suppression) {
              next.store(offsets(source, LeptonNextOffset),
                         materialized.suppression.particle);
              return;
            }
            if (!materialized.has_record) return;
            records(offsets(source, LeptonRecordOffset)) = materialized.record;
            std::uint32_t next_index = 0;
            std::uint32_t photon_index = 0;
            for (std::uint32_t child = 0;
                 child < materialized.secondary_count; ++child) {
              auto const secondary = materialized.secondaries[child];
              if (isChargedLeptonPid(secondary.pid)) {
                next.store(offsets(source, LeptonNextOffset) + next_index++,
                           secondary);
              } else if (secondary.pid ==
                         static_cast<std::int32_t>(EmPid::Photon)) {
                photons(offsets(source, LeptonPhotonOffset) + photon_index++) =
                    secondary;
              } else {
                Kokkos::atomic_compare_exchange(
                    error.data(), std::uint32_t{0}, std::uint32_t{2});
                return;
              }
            }
            if (materialized.has_first_interaction) {
              first_flags(source) = 1;
              first_snapshots(source) = materialized.first_interaction;
            }
          });
      execution.fence("complete resident Kokkos lepton wavefront");
      auto host_error =
          Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), error);
      if (host_error(0) != 0)
        throw std::runtime_error(
            "resident Kokkos lepton final-state materialization failed");

      append(result.step_records, steps, totals[LeptonStepOffset]);
      append(result.final_state_records, records, totals[LeptonRecordOffset]);
      append(result.generated_photons, photons, totals[LeptonPhotonOffset]);
      append(result.fallback_events, fallbacks,
             totals[LeptonFallbackOffset]);
      append(result.observations, observations,
             totals[LeptonObservationOffset]);
      append(result.decay_candidates, decays, totals[LeptonDecayOffset]);
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
                                                 totals[LeptonStepOffset])));
      for (std::size_t i = 0; i < totals[LeptonStepOffset]; ++i)
        result.interaction_vertices +=
            host_steps(i).limit == LeptonTransportLimit::InteractionCandidate;
      auto host_outcomes = Kokkos::create_mirror_view_and_copy(
          Kokkos::HostSpace(), Kokkos::subview(
              outcomes, std::make_pair<std::size_t>(0, current_count)));
      for (std::size_t source = 0; source < current_count; ++source)
        result.lpm_suppressions +=
            host_outcomes(source).final_state.suppression_flag;

      next_history_id += totals[LeptonChildOffset];
      queue.commitNext(totals[LeptonNextOffset]);
      result.wavefronts++;
      result.peak_resident_leptons =
          std::max(result.peak_resident_leptons, queue.size());
    }
    result.secondary_history_ids_used =
        next_history_id - first_secondary_history_id;
    result.completed = queue.empty();
    if (!result.completed) result.remaining_leptons = queue.download();
    return result;
  }

} // namespace corsika::accelerator::em::kokkos_detail

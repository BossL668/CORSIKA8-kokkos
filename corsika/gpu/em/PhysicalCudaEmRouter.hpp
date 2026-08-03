/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

#include <corsika/gpu/em/CudaEmBackend.hpp>
#include <corsika/gpu/em/CorsikaOutputSink.hpp>
#include <corsika/gpu/em/PhotonPairKinematics.hpp>
#include <corsika/gpu/em/ProcessCapabilities.hpp>
#include <corsika/gpu/em/ProposalFallback.hpp>
#include <corsika/gpu/em/RouterParticleConversion.hpp>
#include <corsika/gpu/em/SphericalAtmosphere.hpp>

namespace corsika::gpu::em {

  struct ScalarProposalFallback {};

  struct PhysicalCudaEmRouterStatistics {
    std::uint64_t particles_staged{};
    std::uint64_t photons_advanced{};
    std::uint64_t leptons_advanced{};
    std::uint64_t resident_photon_wavefronts{};
    std::uint64_t resident_lepton_wavefronts{};
    std::uint64_t history_range_checkpoints{};
    std::uint64_t below_minimum_batch_checkpoints{};
    std::uint64_t workspace_limit_checkpoints{};
    std::uint64_t input_batch_splits{};
    std::uint64_t electromagnetic_particles_produced{};
    std::uint64_t particles_returned_for_cpu_fallback{};
    std::uint64_t particles_returned_for_cpu_memory_spill{};
    std::uint64_t particles_returned_for_cpu_decay{};
    std::uint64_t specified_cpu_final_states{};
    std::uint64_t cpu_fallback_steps_executed{};
    std::uint64_t cpu_memory_spill_steps_executed{};
    std::uint64_t forced_cpu_decays_executed{};
    std::uint64_t cpu_wavefront_expansion_steps_executed{};
    std::uint64_t particles_returned_for_cpu_wavefront_expansion{};
    std::uint64_t small_batch_expansions{};
    std::uint64_t stalled_boundary_gpu_flushes{};
    std::uint64_t scalar_expansion_budget_gpu_flushes{};
    std::uint64_t particles_observed{};
    std::uint64_t particles_escaped{};
    std::uint64_t particles_cut{};
    std::uint64_t wavefronts{};
    std::uint64_t reserved_history_ids{};
    std::size_t maximum_input_batch{};
    double deposited_energy_GeV{};
    double medium_rest_mass_input_GeV{};
    double cut_rest_mass_energy_GeV{};
    double observed_total_energy_GeV{};
    double escaped_total_energy_GeV{};
    double specified_cpu_fallback_time_ms{};
    double photon_backend_wall_time_ms{};
    double lepton_backend_wall_time_ms{};
    double photon_host_postprocess_time_ms{};
    double lepton_host_postprocess_time_ms{};
    std::map<std::int32_t, std::uint64_t>
        cpu_fallbacks_by_process{};
    std::map<std::int32_t, std::uint64_t>
        cpu_fallbacks_by_reason{};
  };

  /**
   * HybridCascade adapter for the physical photon and lepton device pipelines.
   *
   * Electromagnetic continuations and secondaries stay owned by this router
   * between wavefronts. Unsupported processes are losslessly imported into the
   * CPU stack and bypass GPU routing for exactly one scalar step. History ID
   * ranges are reserved through the shared stack allocator immediately before
   * each branching kernel, preventing CPU/GPU identity collisions.
   *
   * Validation callers retain observation, fallback, step and radio records
   * by default. Production callers can disable retention after attaching an
   * output sink, so an ultra-high-energy shower streams records to the
   * existing writers without growing host memory with the complete history.
   */
  template <
      typename TStack,
      typename TProposalFallbackHandler =
          ScalarProposalFallback,
      typename TOutputSink = NullCorsikaOutputSink>
  class PhysicalCudaEmRouter {
  public:
    using particle_type = typename TStack::particle_type;

    PhysicalCudaEmRouter(
        CudaEmBackend& backend,
        CoordinateSystemPtr coordinate_system,
        EnvironmentSnapshot const& environment)
        : backend_(backend)
        , coordinate_system_(std::move(coordinate_system))
        , environment_(environment) {
      if (!coordinate_system_) {
        throw std::invalid_argument(
            "physical CUDA EM router requires a coordinate system");
      }
      if (!backend_.hasProposalTable()) {
        throw std::invalid_argument(
            "physical CUDA EM router requires an uploaded PROPOSAL table");
      }
      if (!atmosphere_detail::validEnvironment(environment_)) {
        throw std::invalid_argument(
            "physical CUDA EM router requires a valid atmosphere snapshot");
      }
    }

    PhysicalCudaEmRouter(
        CudaEmBackend& backend,
        CoordinateSystemPtr coordinate_system,
        EnvironmentSnapshot const& environment,
        TProposalFallbackHandler& fallback_handler)
        : PhysicalCudaEmRouter(
              backend, std::move(coordinate_system),
              environment) {
      static_assert(
          !std::is_same_v<
              TProposalFallbackHandler,
              ScalarProposalFallback>,
          "the scalar fallback marker cannot be supplied as a handler");
      fallback_handler_ = &fallback_handler;
    }

    PhysicalCudaEmRouter(
        CudaEmBackend& backend,
        CoordinateSystemPtr coordinate_system,
        EnvironmentSnapshot const& environment,
        TProposalFallbackHandler& fallback_handler,
        TOutputSink& output_sink)
        : PhysicalCudaEmRouter(
              backend, std::move(coordinate_system),
              environment, fallback_handler) {
      static_assert(
          !std::is_same_v<TOutputSink, NullCorsikaOutputSink>,
          "the null output marker cannot be supplied as a sink");
      output_sink_ = &output_sink;
    }

    bool canRoute(
        particle_type const& particle,
        transport::StepId step_id) {
      auto const fallback_key =
          std::make_pair(particle.getHistoryId(), step_id);
      if (forced_decay_steps_.find(fallback_key) !=
          forced_decay_steps_.end()) {
        return false;
      }
      auto const expansion =
          cpu_wavefront_expansion_steps_.find(fallback_key);
      if (expansion !=
          cpu_wavefront_expansion_steps_.end()) {
        cpu_wavefront_expansion_steps_.erase(expansion);
        ++statistics_
              .cpu_wavefront_expansion_steps_executed;
        return false;
      }
      auto const blocked =
          cpu_fallback_steps_.find(fallback_key);
      if (blocked != cpu_fallback_steps_.end()) {
        cpu_fallback_steps_.erase(blocked);
        ++statistics_.cpu_fallback_steps_executed;
        return false;
      }
      auto const memory_spill =
          cpu_memory_spill_steps_.find(fallback_key);
      if (memory_spill !=
          cpu_memory_spill_steps_.end()) {
        cpu_memory_spill_steps_.erase(memory_spill);
        ++statistics_.cpu_memory_spill_steps_executed;
        return false;
      }
      if (!is_em(particle.getPID()) &&
          !is_muon(particle.getPID())) {
        return false;
      }
      auto const state = router_detail::toDeviceState(
          particle, coordinate_system_,
          particle.getHistoryId(),
          particle.getParentHistoryId(),
          particle.getGeneration(), step_id);
      if (!backend_.canTransport(state)) {
        return false;
      }
      auto const layer = queryAtmosphereLayer(
          environment_, state.position_m, state.direction);
      return layer.status == AtmosphereStatus::Success;
    }

    bool consumeForcedDecay(
        transport::HistoryId history_id,
        transport::StepId step_id) {
      auto const found = forced_decay_steps_.find(
          std::make_pair(history_id, step_id));
      if (found == forced_decay_steps_.end()) {
        return false;
      }
      forced_decay_steps_.erase(found);
      ++statistics_.forced_cpu_decays_executed;
      return true;
    }

    void stage(
        particle_type const& particle,
        transport::HistoryId history_id,
        transport::HistoryId parent_history_id,
        transport::Generation generation,
        transport::StepId step_id) {
      auto state = router_detail::toDeviceState(
          particle, coordinate_system_, history_id,
          parent_history_id, generation, step_id);
      auto const layer = queryAtmosphereLayer(
          environment_, state.position_m, state.direction);
      if (layer.status != AtmosphereStatus::Success) {
        throw std::logic_error(
            "physical CUDA EM router staged a particle outside its environment");
      }
      state.medium_id =
          environment_
              .atmosphere_layers[layer.layer_index]
              .medium_id;
      auto const previous =
          last_cpu_expansion_states_.find(
              state.history_id);
      if (previous !=
              last_cpu_expansion_states_.end() &&
          sameTransportState(previous->second, state)) {
        force_small_batch_to_gpu_ = true;
      }
      staged_.push_back(state);
      ++statistics_.particles_staged;
    }

    bool pending() const {
      return !staged_.empty() ||
             backend_.pendingPhotonCount() != 0 ||
             backend_.pendingLeptonCount() != 0;
    }

    /**
     * Report whether the resident EM front should be advanced before the
     * scalar stack is exhausted.
     *
     * HybridCascade historically drained every runnable scalar history before
     * calling the router. At ultra-high energy this allowed routed gamma/e+/e-
     * states to accumulate without bound in staged_ while the CPU continued a
     * long hadronic front. Interleave only at the configured production batch
     * boundary (or when a device-resident continuation is waiting), preserving
     * the existing small-batch scalar-expansion policy.
     */
    bool readyForScalarInterleave() const noexcept {
      return force_small_batch_to_gpu_ ||
             backend_.pendingPhotonCount() != 0 ||
             backend_.pendingLeptonCount() != 0 ||
             staged_.size() >= backend_.minimumBatchSize();
    }

    void endOfShower() {
      if (radio_finalized_) {
        throw std::logic_error(
            "CUDA EM router output finalized more than once");
      }
      if (!staged_.empty() ||
          backend_.pendingPhotonCount() != 0 ||
          backend_.pendingLeptonCount() != 0) {
        throw std::logic_error(
            "CUDA EM router cannot finalize with an unconsumed host or device wavefront");
      }
      if (backend_.gpuProfileEnabled()) {
        if constexpr (
            !std::is_same_v<TOutputSink, NullCorsikaOutputSink>) {
          if (output_sink_ == nullptr) {
            throw std::logic_error(
                "CUDA resident profile requires a production output sink");
          }
          auto const profile = backend_.downloadProfile();
          output_sink_->onGpuProfile(profile);
          statistics_.particles_cut +=
              profile.particle_cuts;
          statistics_.deposited_energy_GeV +=
              profile.weighted_deposited_energy_GeV;
          statistics_.medium_rest_mass_input_GeV +=
              profile
                  .weighted_medium_rest_mass_input_GeV;
          statistics_.cut_rest_mass_energy_GeV +=
              profile.weighted_cut_rest_mass_energy_GeV;
          statistics_.observed_total_energy_GeV +=
              profile
                  .weighted_observed_total_energy_GeV;
          statistics_.escaped_total_energy_GeV +=
              profile
                  .weighted_escaped_total_energy_GeV;
        } else {
          throw std::logic_error(
              "CUDA resident profile cannot finalize through a null output sink");
        }
      }
      if (backend_.gpuRadioEnabled()) {
        if constexpr (
            !std::is_same_v<TOutputSink, NullCorsikaOutputSink>) {
          if (output_sink_ == nullptr) {
            throw std::logic_error(
                "CUDA radio requires a production output sink");
          }
          auto const waveforms =
              backend_.downloadRadioWaveforms();
          output_sink_->onGpuRadioWaveforms(
              waveforms,
              backend_.statistics().radio.lepton_tracks);
        } else {
          throw std::logic_error(
              "CUDA radio cannot finalize through a null output sink");
        }
      }
      radio_finalized_ = true;
    }

    void setRetainRecords(bool retain) noexcept {
      retain_records_ = retain;
    }

    bool retainsRecords() const noexcept {
      return retain_records_;
    }

    void setFailOnUnexpectedFallback(bool fail) noexcept {
      fail_on_unexpected_fallback_ = fail;
    }

    bool failsOnUnexpectedFallback() const noexcept {
      return fail_on_unexpected_fallback_;
    }

    static constexpr bool isPermittedScalarFallbackReason(
        ProposalFallbackReason reason) noexcept {
      return reason ==
                 ProposalFallbackReason::UnsupportedParticle ||
             reason ==
                 ProposalFallbackReason::UnsupportedMedium ||
             reason ==
                 ProposalFallbackReason::UnsupportedGeometry;
    }

    std::size_t advanceOneWavefrontAndReturn(TStack& stack) {
      auto const initial_device_pending =
          backend_.pendingPhotonCount() +
          backend_.pendingLeptonCount();
      if (staged_.empty() &&
          initial_device_pending == 0) {
        return 0;
      }
      if (staged_.size() <
              backend_.minimumBatchSize() &&
          !force_small_batch_to_gpu_ &&
          initial_device_pending == 0 &&
          consecutive_scalar_expansion_rounds_ <
              MaximumConsecutiveScalarExpansionRounds) {
        std::vector<EmParticleState> expansion;
        expansion.swap(staged_);
        last_cpu_expansion_states_.clear();
        for (auto const& particle : expansion) {
          auto const key = std::make_pair(
              particle.history_id, particle.step_id);
          if (!cpu_wavefront_expansion_steps_
                   .insert(key)
                   .second) {
            throw std::runtime_error(
                "duplicate CPU wavefront-expansion step");
          }
          router_detail::importParticle(
              stack, particle, coordinate_system_);
          last_cpu_expansion_states_.emplace(
              particle.history_id, particle);
        }
        force_small_batch_to_gpu_ = false;
        ++consecutive_scalar_expansion_rounds_;
        ++statistics_.small_batch_expansions;
        statistics_
            .particles_returned_for_cpu_wavefront_expansion +=
            expansion.size();
        auto const expansion_count =
            statistics_.small_batch_expansions;
        if (expansion_count <= 8 ||
            (expansion_count &
             (expansion_count - 1)) == 0) {
          CORSIKA_LOG_DEBUG(
              "CUDA EM scalar wavefront expansion {}: {} particles "
              "(threshold {}, first pid/energy/history/step={}/{}/{}/{})",
              expansion_count, expansion.size(),
              backend_.minimumBatchSize(),
              expansion.front().pid,
              expansion.front().energy_GeV,
              expansion.front().history_id,
              expansion.front().step_id);
        }
        return expansion.size();
      }
      if (force_small_batch_to_gpu_ &&
          staged_.size() <
              backend_.minimumBatchSize()) {
        ++statistics_.stalled_boundary_gpu_flushes;
      } else if (
          initial_device_pending == 0 &&
          staged_.size() <
              backend_.minimumBatchSize() &&
          consecutive_scalar_expansion_rounds_ >=
              MaximumConsecutiveScalarExpansionRounds) {
        ++statistics_
              .scalar_expansion_budget_gpu_flushes;
        CORSIKA_LOG_DEBUG(
            "CUDA EM forced a {}-particle wavefront after {} "
            "consecutive scalar expansion rounds",
            staged_.size(),
            consecutive_scalar_expansion_rounds_);
      }
      force_small_batch_to_gpu_ = false;
      consecutive_scalar_expansion_rounds_ = 0;
      last_cpu_expansion_states_.clear();
      std::vector<EmParticleState> current;
      current.swap(staged_);
      auto const input_limit =
          backend_.maximumResidentInputBatchSize();
      if (current.size() > input_limit) {
        staged_.assign(
            std::make_move_iterator(
                current.begin() +
                static_cast<std::ptrdiff_t>(
                    input_limit)),
            std::make_move_iterator(current.end()));
        current.resize(input_limit);
        ++statistics_.input_batch_splits;
      }
      ++statistics_.wavefronts;

      std::vector<EmParticleState> photons;
      std::vector<EmParticleState> leptons;
      photons.reserve(current.size());
      leptons.reserve(current.size());
      for (auto const& particle : current) {
        if (particle.pid ==
            static_cast<std::int32_t>(EmPid::Photon)) {
          photons.push_back(particle);
        } else if (isChargedLeptonPid(particle.pid)) {
          leptons.push_back(particle);
        } else {
          throw std::logic_error(
              "physical CUDA EM router owns an unsupported particle");
        }
      }

      auto returned_to_cpu = std::size_t{0};
      auto const maximum_photon_input =
          backend_.maximumResidentPhotonBatchSize();
      auto const pending_photon_chunk =
          std::min(
              backend_.pendingPhotonCount(),
              maximum_photon_input);
      auto const host_photon_limit =
          maximum_photon_input -
          pending_photon_chunk;
      if (photons.size() > host_photon_limit) {
        staged_.insert(
            staged_.end(),
            std::make_move_iterator(
                photons.begin() +
                static_cast<std::ptrdiff_t>(
                    host_photon_limit)),
            std::make_move_iterator(photons.end()));
        photons.resize(host_photon_limit);
        ++statistics_.input_batch_splits;
      }
      auto const photon_input_count =
          photons.size() +
          pending_photon_chunk;
      if (photon_input_count != 0) {
        statistics_.maximum_input_batch =
            std::max(
                statistics_.maximum_input_batch,
                photon_input_count);
        auto const first_photon_step = step_records_.size();
        constexpr std::size_t
            ResidentPhotonWavefronts = 16;
        auto const first_history =
            reservePhotonSecondaryRange(
                stack, photon_input_count,
                ResidentPhotonWavefronts);
        auto const backend_start =
            std::chrono::steady_clock::now();
        auto result =
            backend_
                .runResidentPhotonCascadeForValidation(
                    photons, first_history,
                    ResidentPhotonWavefronts,
                    backend_.minimumBatchSize());
        auto const backend_stop =
            std::chrono::steady_clock::now();
        statistics_.photon_backend_wall_time_ms +=
            elapsedMilliseconds(backend_start, backend_stop);
        auto const postprocess_start =
            std::chrono::steady_clock::now();
        statistics_.photons_advanced +=
            result.transport_records;
        statistics_.resident_photon_wavefronts +=
            result.wavefronts;
        if (result.below_minimum_batch_checkpoint) {
          ++statistics_.below_minimum_batch_checkpoints;
        }
        if (result.workspace_limit_checkpoint) {
          ++statistics_.workspace_limit_checkpoints;
        }
        if (!result.projected_step_records.empty()) {
          if (retain_records_ ||
              !result.step_records.empty()) {
            throw std::logic_error(
                "projected CUDA photon output is only valid for "
                "non-retaining production routing");
          }
          for (auto& record :
               result.projected_step_records) {
            recordProjectedPhotonStep(record);
          }
          recordProjectedPhotonFinalStateDeposits(
              result.projected_step_records,
              result.final_state_records);
          publishProjectedSteps(
              result.projected_step_records);
        } else {
          for (auto const& record : result.step_records) {
            recordPhotonStep(record);
          }
          recordPhotonFinalStateDeposits(
              result.final_state_records);
          publishPhotonSteps(first_photon_step);
          if (!retain_records_) {
            step_records_.erase(
                step_records_.begin() +
                    static_cast<std::ptrdiff_t>(
                        first_photon_step),
                step_records_.end());
          }
        }
        appendParticles(
            result.electromagnetic_secondaries);
        appendParticles(result.remaining_photons);
        returned_to_cpu += returnMemorySpills(
            stack, result.cpu_spill_particles);
        returned_to_cpu += returnFallbacks(
            stack, result.fallback_events);
        for (auto const& observation : result.observations) {
          recordObservation(observation);
        }
        auto const postprocess_stop =
            std::chrono::steady_clock::now();
        statistics_.photon_host_postprocess_time_ms +=
            elapsedMilliseconds(
                postprocess_start, postprocess_stop);
      }

      auto const lepton_input_count =
          leptons.size() +
          backend_.pendingLeptonCount();
      if (lepton_input_count != 0) {
        auto const first_lepton_step = step_records_.size();
        auto const maximum_lepton_input =
            backend_.maximumResidentLeptonBatchSize();
        auto const pending_leptons =
            std::min(
                backend_.pendingLeptonCount(),
                maximum_lepton_input);
        auto const host_lepton_limit =
            maximum_lepton_input - pending_leptons;
        if (leptons.size() > host_lepton_limit) {
          staged_.insert(
              staged_.end(),
              std::make_move_iterator(
                  leptons.begin() +
                  static_cast<std::ptrdiff_t>(
                      host_lepton_limit)),
              std::make_move_iterator(leptons.end()));
          leptons.resize(host_lepton_limit);
          ++statistics_.input_batch_splits;
        }
        auto const routed_lepton_input_count =
            leptons.size() + pending_leptons;
        statistics_.maximum_input_batch =
            std::max(
                statistics_.maximum_input_batch,
                routed_lepton_input_count);
        auto const history_range =
            reserveResidentSecondaryRange(
                stack, routed_lepton_input_count);
        auto const backend_start =
            std::chrono::steady_clock::now();
        auto result =
            backend_
                .runResidentLeptonCascadeForValidation(
                    leptons, history_range.first,
                    1024, history_range.limit,
                    backend_.minimumBatchSize());
        auto const backend_stop =
            std::chrono::steady_clock::now();
        statistics_.lepton_backend_wall_time_ms +=
            elapsedMilliseconds(backend_start, backend_stop);
        auto const postprocess_start =
            std::chrono::steady_clock::now();
        statistics_.leptons_advanced +=
            result.transport_records;
        statistics_.resident_lepton_wavefronts +=
            result.wavefronts;
        if (result.history_range_exhausted) {
          ++statistics_.history_range_checkpoints;
        }
        if (result.below_minimum_batch_checkpoint) {
          ++statistics_.below_minimum_batch_checkpoints;
        }
        if (result.workspace_limit_checkpoint) {
          ++statistics_.workspace_limit_checkpoints;
        }
        if (!result.projected_step_records.empty()) {
          if (retain_records_ ||
              !result.step_records.empty()) {
            throw std::logic_error(
                "projected CUDA lepton output is only valid for "
                "non-retaining production routing");
          }
          for (auto const& record :
               result.projected_step_records) {
            recordProjectedLeptonStep(record);
          }
        } else {
          for (auto const& record : result.step_records) {
            recordLeptonStep(record);
            statistics_.deposited_energy_GeV +=
                record.continuous_deposited_energy_GeV *
                record.start.weight;
            if (record.limit ==
                LeptonTransportLimit::ParticleCut) {
              statistics_.deposited_energy_GeV +=
                  record.cut_deposited_energy_GeV *
                  record.start.weight;
              ++statistics_.particles_cut;
            }
          }
          annotateLeptonProcessIds(
              first_lepton_step, result.step_records,
              result.final_state_records);
          recordLeptonFinalStateLedger(
              result.step_records,
              result.final_state_records);
        }
        appendParticles(result.generated_photons);
        appendParticles(result.remaining_leptons);
        returned_to_cpu += returnMemorySpills(
            stack, result.cpu_spill_particles);
        returned_to_cpu += returnFallbacks(
            stack, result.fallback_events);
        returned_to_cpu += returnForcedDecays(
            stack, result.decay_candidates);
        for (auto const& observation :
             result.observations) {
          recordObservation(observation);
        }
        auto const postprocess_stop =
            std::chrono::steady_clock::now();
        statistics_.lepton_host_postprocess_time_ms +=
            elapsedMilliseconds(
                postprocess_start, postprocess_stop);
      }
      statistics_.electromagnetic_particles_produced +=
          staged_.size();
      return returned_to_cpu;
    }

    PhysicalCudaEmRouterStatistics const& statistics() const {
      return statistics_;
    }

    std::vector<ObservationRecord> const& observations() const {
      return observations_;
    }

    std::vector<ProposalFallbackEvent> const& fallbackEvents() const {
      return fallback_events_;
    }

    std::vector<EmStepRecord> const& stepRecords() const {
      return step_records_;
    }

    std::vector<RadioTrackRecord> const& radioTracks() const {
      return radio_tracks_;
    }

  private:
    static double elapsedMilliseconds(
        std::chrono::steady_clock::time_point start,
        std::chrono::steady_clock::time_point stop) {
      return std::chrono::duration<double, std::milli>(
                 stop - start)
          .count();
    }

    static bool sameTransportState(
        EmParticleState const& left,
        EmParticleState const& right) {
      if (left.pid != right.pid ||
          left.energy_GeV != right.energy_GeV ||
          left.time_s != right.time_s ||
          left.weight != right.weight) {
        return false;
      }
      for (int axis = 0; axis < 3; ++axis) {
        if (left.position_m[axis] !=
                right.position_m[axis] ||
            left.direction[axis] !=
                right.direction[axis]) {
          return false;
        }
      }
      return true;
    }

    struct ReservedHistoryRange {
      transport::HistoryId first{};
      transport::HistoryId limit{};
      std::uint64_t count{};
    };

    transport::HistoryId reserveSecondaryRange(
        TStack& stack, std::size_t input_count) {
      if (input_count >
          std::numeric_limits<std::uint64_t>::max() / 2) {
        throw std::overflow_error(
            "physical CUDA EM history reservation overflow");
      }
      auto const count =
          static_cast<std::uint64_t>(input_count) * 2;
      auto const first =
          stack.reserveTransportHistoryIds(count);
      statistics_.reserved_history_ids += count;
      return first;
    }

    transport::HistoryId reservePhotonSecondaryRange(
        TStack& stack, std::size_t input_count,
        std::size_t maximum_wavefronts) {
      if (input_count == 0 ||
          maximum_wavefronts == 0 ||
          input_count >
              std::numeric_limits<std::uint64_t>::max() /
                  2 / maximum_wavefronts) {
        throw std::overflow_error(
            "resident CUDA photon history reservation overflow");
      }
      // One resident photon produces at most two final-state children in a
      // wavefront and at most one of them can remain a photon. Therefore the
      // active photon count cannot grow, and this is a strict upper bound for
      // the explicit checkpoint interval.
      auto const count =
          static_cast<std::uint64_t>(input_count) *
          2 * maximum_wavefronts;
      auto const first =
          stack.reserveTransportHistoryIds(count);
      statistics_.reserved_history_ids += count;
      return first;
    }

    ReservedHistoryRange reserveResidentSecondaryRange(
        TStack& stack, std::size_t input_count) {
      if (input_count == 0 ||
          input_count >
              std::numeric_limits<std::uint64_t>::max() /
                  3) {
        throw std::overflow_error(
            "resident CUDA EM history reservation overflow");
      }
      auto const minimum =
          static_cast<std::uint64_t>(input_count) * 3;
      constexpr std::uint64_t ReservationMultiplier = 64;
      constexpr std::uint64_t MaximumChunk = 1U << 20;
      auto count = minimum;
      if (minimum <=
          std::numeric_limits<std::uint64_t>::max() /
              ReservationMultiplier) {
        count = std::min(
            MaximumChunk,
            minimum * ReservationMultiplier);
        count = std::max(count, minimum);
      }
      auto const first =
          stack.reserveTransportHistoryIds(count);
      if (count >
          std::numeric_limits<transport::HistoryId>::max() -
              first) {
        throw std::overflow_error(
            "resident CUDA EM history range limit overflow");
      }
      auto const limit = first + count;
      statistics_.reserved_history_ids += count;
      return {first, limit, count};
    }

    void appendParticles(
        std::vector<EmParticleState> const& particles) {
      staged_.insert(
          staged_.end(), particles.begin(), particles.end());
    }

    void appendNonPhotonParticles(
        std::vector<EmParticleState> const& particles) {
      std::copy_if(
          particles.begin(), particles.end(),
          std::back_inserter(staged_),
          [](EmParticleState const& particle) {
            return particle.pid !=
                   static_cast<std::int32_t>(EmPid::Photon);
          });
    }

    std::size_t returnFallbacks(
        TStack& stack,
        std::vector<ProposalFallbackEvent> const& events) {
      auto returned_to_scalar = std::size_t{0};
      for (auto const& event : events) {
        ++statistics_
              .cpu_fallbacks_by_process[event.process_id];
        ++statistics_.cpu_fallbacks_by_reason[
            static_cast<std::int32_t>(event.reason)];
        if constexpr (
            !std::is_same_v<
                TProposalFallbackHandler,
                ScalarProposalFallback>) {
          if (fallback_handler_ != nullptr &&
              fallback_handler_->canHandle(event)) {
            auto const fallback_start =
                std::chrono::steady_clock::now();
            fallback_handler_->handle(stack, event);
            auto const fallback_stop =
                std::chrono::steady_clock::now();
            statistics_
                .specified_cpu_fallback_time_ms +=
                std::chrono::duration<double, std::milli>(
                    fallback_stop - fallback_start)
                    .count();
            if (retain_records_) {
              fallback_events_.push_back(event);
            }
            ++statistics_.specified_cpu_final_states;
            continue;
          }
        }
        CORSIKA_LOG_WARN(
            "CUDA EM fallback returned to scalar transport: "
            "pid={}, energy_GeV={}, "
            "history={}, step={}, process={}, reason={}, "
            "diagnostic_status={}, diagnostic_values=[{},{},{}], "
            "position_m=[{},{},{}], direction=[{},{},{}]",
            event.particle.pid, event.particle.energy_GeV,
            event.particle.history_id, event.particle.step_id,
            event.process_id,
            proposalFallbackReasonName(event.reason),
            event.diagnostic_status, event.diagnostic_value0,
            event.diagnostic_value1, event.diagnostic_value2,
            event.particle.position_m[0],
            event.particle.position_m[1],
            event.particle.position_m[2],
            event.particle.direction[0],
            event.particle.direction[1],
            event.particle.direction[2]);
        if (fail_on_unexpected_fallback_ &&
            !isPermittedScalarFallbackReason(event.reason)) {
          std::ostringstream message;
          message
              << "unexpected CUDA EM fallback cannot be retried silently: "
              << "pid=" << event.particle.pid
              << ", energy_GeV=" << event.particle.energy_GeV
              << ", history=" << event.particle.history_id
              << ", step=" << event.particle.step_id
              << ", process=" << event.process_id
              << ", reason="
              << proposalFallbackReasonName(event.reason)
              << ", diagnostic_status="
              << event.diagnostic_status
              << ", diagnostic_values=["
              << event.diagnostic_value0 << ","
              << event.diagnostic_value1 << ","
              << event.diagnostic_value2 << "]";
          throw std::runtime_error(message.str());
        }
        auto const key = std::make_pair(
            event.particle.history_id,
            event.particle.step_id);
        if (!cpu_fallback_steps_.insert(key).second) {
          throw std::runtime_error(
              "duplicate CPU fallback for one transport step");
        }
        router_detail::importParticle(
            stack, event.particle, coordinate_system_);
        if (retain_records_) {
          fallback_events_.push_back(event);
        }
        ++returned_to_scalar;
      }
      statistics_.particles_returned_for_cpu_fallback +=
          returned_to_scalar;
      return returned_to_scalar;
    }

    std::size_t returnMemorySpills(
        TStack& stack,
        std::vector<EmParticleState> const& particles) {
      for (auto const& particle : particles) {
        auto const key = std::make_pair(
            particle.history_id,
            particle.step_id);
        if (!cpu_memory_spill_steps_.insert(key).second) {
          throw std::runtime_error(
              "duplicate CPU memory spill for one transport step");
        }
        router_detail::importParticle(
            stack, particle, coordinate_system_);
      }
      statistics_.particles_returned_for_cpu_memory_spill +=
          particles.size();
      return particles.size();
    }

    std::size_t returnForcedDecays(
        TStack& stack,
        std::vector<EmParticleState> const& particles) {
      for (auto const& particle : particles) {
        auto const key = std::make_pair(
            particle.history_id, particle.step_id);
        if (!forced_decay_steps_.insert(key).second) {
          throw std::runtime_error(
              "duplicate forced CPU decay for one transport step");
        }
        router_detail::importParticle(
            stack, particle, coordinate_system_);
      }
      statistics_.particles_returned_for_cpu_decay +=
          particles.size();
      return particles.size();
    }

    void recordObservation(ObservationRecord const& observation) {
      if (retain_records_) {
        observations_.push_back(observation);
      }
      if constexpr (
          !std::is_same_v<TOutputSink, NullCorsikaOutputSink>) {
        if (output_sink_ != nullptr) {
          output_sink_->onObservation(observation);
        }
      }
      if (observation.status ==
          ObservationStatus::ReachedObservationSurface) {
        ++statistics_.particles_observed;
      } else {
        ++statistics_.particles_escaped;
      }
      if (!backend_.gpuProfileEnabled()) {
        auto const weighted_total_energy =
            observation.particle.energy_GeV *
            observation.particle.weight;
        if (!std::isfinite(weighted_total_energy) ||
            weighted_total_energy < 0.) {
          throw std::runtime_error(
              "CUDA EM observation has invalid weighted total energy");
        }
        if (observation.status ==
            ObservationStatus::ReachedObservationSurface) {
          statistics_.observed_total_energy_GeV +=
              weighted_total_energy;
        } else {
          statistics_.escaped_total_energy_GeV +=
              weighted_total_energy;
        }
      }
    }

    static EmStepRecord makeStepRecord(
        EmParticleState const& start,
        EmParticleState const& end,
        std::int32_t process_id,
        double deposited_energy_GeV) {
      EmStepRecord step{};
      step.history_id = start.history_id;
      step.step_id = start.step_id;
      step.pid = start.pid;
      step.process_id = process_id;
      for (int axis = 0; axis < 3; ++axis) {
        step.start_position_m[axis] =
            start.position_m[axis];
        step.end_position_m[axis] =
            end.position_m[axis];
      }
      step.start_time_s = start.time_s;
      step.end_time_s = end.time_s;
      step.start_energy_GeV = start.energy_GeV;
      step.end_energy_GeV = end.energy_GeV;
      step.deposited_energy_GeV =
          deposited_energy_GeV;
      step.weight = start.weight;
      return step;
    }

    static double leptonRestMassGeV(std::int32_t pid) {
      if (isElectronOrPositronPid(pid)) {
        return ElectronMassGeV;
      }
      if (isMuonPid(pid)) {
        return MuonMassGeV;
      }
      throw std::logic_error(
          "CUDA charged-lepton record has an unsupported PID");
    }

    void recordPhotonStep(
        PhotonTransportRecord const& record) {
      auto const process_id =
          record.limit == PhotonTransportLimit::Interaction
              ? record.interaction.process_id
              : 0;
      auto const deposited =
          record.cut_deposited_energy_GeV;
      step_records_.push_back(makeStepRecord(
          record.start, record.end, process_id, deposited));
      if (record.limit ==
          PhotonTransportLimit::ParticleCut) {
        statistics_.deposited_energy_GeV +=
            deposited * record.start.weight;
        ++statistics_.particles_cut;
      }
    }

    void recordPhotonFinalStateDeposits(
        std::vector<PhotonPairFinalStateRecord> const&
            final_state_records) {
      for (auto const& record : final_state_records) {
        if (record.process_id != ComptonProcessId &&
            record.process_id != PhotoelectricProcessId) {
          continue;
        }
        if (!std::isfinite(record.energy_split_fraction) ||
            record.energy_split_fraction < 0. ||
            record.energy_split_fraction > 1.) {
          throw std::runtime_error(
              "photoelectric final state has invalid deposit metadata");
        }
        auto const step = std::find_if(
            step_records_.rbegin(), step_records_.rend(),
            [&](EmStepRecord const& candidate) {
              return candidate.history_id ==
                         record.parent_history_id &&
                     candidate.process_id ==
                         record.process_id;
            });
        if (step == step_records_.rend()) {
          throw std::runtime_error(
              "atomic-electron photon final state has no matching transport step");
        }
        statistics_.medium_rest_mass_input_GeV +=
            ElectronMassGeV * step->weight;
        if (record.process_id != PhotoelectricProcessId) {
          continue;
        }
        auto const deposit =
            step->end_energy_GeV *
            (1. - record.energy_split_fraction);
        statistics_.deposited_energy_GeV +=
            deposit * step->weight;
        step->deposited_energy_GeV += deposit;
      }
    }

    void recordProjectedPhotonStep(
        ProjectedEmStepRecord& record) {
      if (record.transport_limit ==
          static_cast<std::int32_t>(
              PhotonTransportLimit::ParticleCut)) {
        statistics_.deposited_energy_GeV +=
            record.deposited_energy_GeV *
            record.weight;
        ++statistics_.particles_cut;
      }
    }

    void recordProjectedPhotonFinalStateDeposits(
        std::vector<ProjectedEmStepRecord>& steps,
        std::vector<PhotonPairFinalStateRecord> const&
            final_state_records) {
      for (auto const& record : final_state_records) {
        if (record.process_id != ComptonProcessId &&
            record.process_id != PhotoelectricProcessId) {
          continue;
        }
        if (!std::isfinite(record.energy_split_fraction) ||
            record.energy_split_fraction < 0. ||
            record.energy_split_fraction > 1.) {
          throw std::runtime_error(
              "photoelectric final state has invalid projected deposit metadata");
        }
        auto const step = std::find_if(
            steps.rbegin(), steps.rend(),
            [&](ProjectedEmStepRecord const& candidate) {
              return candidate.history_id ==
                         record.parent_history_id &&
                     candidate.process_id ==
                         record.process_id;
            });
        if (step == steps.rend()) {
          throw std::runtime_error(
              "atomic-electron photon final state has no matching projected step");
        }
        statistics_.medium_rest_mass_input_GeV +=
            ElectronMassGeV * step->weight;
        if (record.process_id != PhotoelectricProcessId) {
          continue;
        }
        auto const deposit =
            step->end_energy_GeV *
            (1. - record.energy_split_fraction);
        statistics_.deposited_energy_GeV +=
            deposit * step->weight;
        step->deposited_energy_GeV += deposit;
      }
    }

    void recordLeptonStep(
        LeptonTransportRecord const& record) {
      auto const deposited =
          record.continuous_deposited_energy_GeV +
          record.cut_deposited_energy_GeV;
      auto const process_id =
          record.limit ==
                  LeptonTransportLimit::InteractionCandidate
              ? record.interaction.process_id
              : 0;
      auto const step = makeStepRecord(
          record.start, record.end, process_id, deposited);
      if (retain_records_) {
        step_records_.push_back(step);
      }
      if constexpr (
          !std::is_same_v<TOutputSink, NullCorsikaOutputSink>) {
        if (output_sink_ != nullptr) {
          output_sink_->onStep(step);
        }
      }
      // The scalar CORSIKA radio process deliberately ignores muons. Preserve
      // that observable definition even though the charged transport kernel
      // also advances mu-/mu+.
      if (isElectronOrPositronPid(record.start.pid)) {
        RadioTrackRecord radio{};
        radio.step = step;
        for (int axis = 0; axis < 3; ++axis) {
          radio.start_direction[axis] =
              record.start.direction[axis];
          radio.end_direction[axis] =
              record.end.direction[axis];
        }
        if (retain_records_) {
          radio_tracks_.push_back(radio);
        }
        if constexpr (
            !std::is_same_v<TOutputSink, NullCorsikaOutputSink>) {
          if (output_sink_ != nullptr) {
            output_sink_->onRadioTrack(radio);
          }
        }
      }
      if (!backend_.gpuProfileEnabled() &&
          record.limit ==
              LeptonTransportLimit::ParticleCut) {
        statistics_.cut_rest_mass_energy_GeV +=
            leptonRestMassGeV(record.start.pid) *
            record.start.weight;
      }
    }

    void recordLeptonFinalStateLedger(
        std::vector<LeptonTransportRecord> const& transports,
        std::vector<BremsFinalStateRecord> const& final_states) {
      auto const has_atomic_electron_process =
          std::any_of(
              final_states.begin(), final_states.end(),
              [](BremsFinalStateRecord const& record) {
                return record.process_id ==
                           IonizationProcessId ||
                       record.process_id ==
                           AnnihilationProcessId;
              });
      if (!has_atomic_electron_process) {
        return;
      }
      std::unordered_map<std::uint64_t, double>
          weights_by_history;
      weights_by_history.reserve(transports.size());
      for (auto const& transport : transports) {
        weights_by_history.try_emplace(
            transport.start.history_id,
            transport.start.weight);
      }
      for (auto const& record : final_states) {
        if (record.process_id != IonizationProcessId &&
            record.process_id != AnnihilationProcessId) {
          continue;
        }
        auto const weight = weights_by_history.find(
            record.parent_history_id);
        if (weight == weights_by_history.end()) {
          throw std::runtime_error(
              "atomic-electron lepton final state has no matching transport step");
        }
        statistics_.medium_rest_mass_input_GeV +=
            ElectronMassGeV * weight->second;
      }
    }

    void annotateLeptonProcessIds(
        std::size_t first_step,
        std::vector<LeptonTransportRecord> const& transports,
        std::vector<BremsFinalStateRecord> const& final_states) {
      if (!retain_records_) { return; }
      if (step_records_.size() - first_step != transports.size()) {
        throw std::logic_error(
            "retained lepton transport records lost index alignment");
      }
      std::unordered_map<std::uint64_t, std::vector<std::int32_t>>
          processes_by_history;
      for (auto const& final_state : final_states) {
        processes_by_history[final_state.parent_history_id].push_back(
            final_state.process_id);
      }
      std::unordered_map<std::uint64_t, std::size_t> next_process;
      for (std::size_t index = 0; index < transports.size(); ++index) {
        auto const& transport = transports[index];
        if (transport.limit !=
            LeptonTransportLimit::InteractionCandidate) {
          continue;
        }
        auto const history = transport.start.history_id;
        auto const found = processes_by_history.find(history);
        if (found == processes_by_history.end()) { continue; }
        auto& offset = next_process[history];
        if (offset >= found->second.size()) { continue; }
        step_records_[first_step + index].process_id =
            found->second[offset++];
      }
    }

    void recordProjectedLeptonStep(
        ProjectedEmStepRecord const& record) {
      statistics_.deposited_energy_GeV +=
          record.deposited_energy_GeV *
          record.weight;
      if (record.transport_limit ==
          static_cast<std::int32_t>(
              LeptonTransportLimit::ParticleCut)) {
        ++statistics_.particles_cut;
        statistics_.cut_rest_mass_energy_GeV +=
            leptonRestMassGeV(record.pid) * record.weight;
      }
      if constexpr (
          !std::is_same_v<TOutputSink, NullCorsikaOutputSink>) {
        if (output_sink_ != nullptr) {
          output_sink_->onProjectedStep(record);
        }
      }
    }

    void publishPhotonSteps(std::size_t first) {
      if (first > step_records_.size()) {
        throw std::logic_error(
            "invalid CUDA photon output publication range");
      }
      if constexpr (
          !std::is_same_v<TOutputSink, NullCorsikaOutputSink>) {
        if (output_sink_ != nullptr) {
          for (auto index = first; index < step_records_.size(); ++index) {
            output_sink_->onStep(step_records_[index]);
          }
        }
      }
    }

    void publishProjectedSteps(
        std::vector<ProjectedEmStepRecord> const& records) {
      if constexpr (
          !std::is_same_v<TOutputSink, NullCorsikaOutputSink>) {
        if (output_sink_ != nullptr) {
          for (auto const& record : records) {
            output_sink_->onProjectedStep(record);
          }
        }
      }
    }

    CudaEmBackend& backend_;
    CoordinateSystemPtr coordinate_system_;
    EnvironmentSnapshot environment_{};
    std::vector<EmParticleState> staged_{};
    std::set<std::pair<transport::HistoryId, transport::StepId>>
        cpu_fallback_steps_{};
    std::set<std::pair<transport::HistoryId, transport::StepId>>
        cpu_memory_spill_steps_{};
    std::set<std::pair<transport::HistoryId, transport::StepId>>
        cpu_wavefront_expansion_steps_{};
    std::set<std::pair<transport::HistoryId, transport::StepId>>
        forced_decay_steps_{};
    std::map<transport::HistoryId, EmParticleState>
        last_cpu_expansion_states_{};
    static constexpr std::uint32_t
        MaximumConsecutiveScalarExpansionRounds = 8;
    std::uint32_t consecutive_scalar_expansion_rounds_{};
    bool force_small_batch_to_gpu_{};
    std::vector<ObservationRecord> observations_{};
    std::vector<ProposalFallbackEvent> fallback_events_{};
    std::vector<EmStepRecord> step_records_{};
    std::vector<RadioTrackRecord> radio_tracks_{};
    TProposalFallbackHandler* fallback_handler_{};
    TOutputSink* output_sink_{};
    bool retain_records_{true};
    bool fail_on_unexpected_fallback_{};
    bool radio_finalized_{};
    PhysicalCudaEmRouterStatistics statistics_{};
  };

} // namespace corsika::gpu::em

/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include <cuda_runtime.h>

#include <cub/device/device_scan.cuh>
#include <cub/device/device_select.cuh>
#include <cub/iterator/counting_input_iterator.cuh>
#include <cub/iterator/transform_input_iterator.cuh>

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <vector>

#include <corsika/gpu/em/CudaPhotonSelectionTransport.hpp>
#include <corsika/gpu/em/ProcessCapabilities.hpp>
#include <corsika/gpu/em/SphericalAtmosphere.hpp>
#include <corsika/gpu/em/detail/DeviceBatchStages.hpp>

namespace corsika::gpu::em {

  namespace {

    constexpr unsigned int ThreadsPerBlock = 256;
    constexpr std::size_t PhotonEndpointSummarySize =
        7 + detail::PhotonFinalStateSummaryLayout::Size;

    struct IsChargedEmParticle {
      __host__ __device__ bool operator()(
          EmParticleState const& particle) const {
        return particle.pid ==
                   static_cast<std::int32_t>(
                       EmPid::Electron) ||
               particle.pid ==
                   static_cast<std::int32_t>(
                       EmPid::Positron);
      }
    };

    struct LoadValidPhotonSecondary {
      EmParticleState const* particles{};
      std::uint32_t const* device_count{};

      __host__ __device__ EmParticleState operator()(
          std::size_t index) const {
#ifdef __CUDA_ARCH__
        if (device_count != nullptr &&
            index >= *device_count) {
          return {};
        }
#endif
        return particles[index];
      }
    };

    void checkCuda(cudaError_t status, char const* operation) {
      if (status == cudaSuccess) {
        return;
      }
      std::ostringstream message;
      message << operation << " failed: "
              << cudaGetErrorString(status);
      throw std::runtime_error(message.str());
    }

    __global__ void scatterTransportEndpointsKernel(
        PhotonTransportRecord const* records, std::size_t count,
        std::size_t source_count, EmParticleState* raw_next,
        std::uint32_t* next_flags,
        ObservationRecord* raw_observations,
        std::uint32_t* observation_flags,
        std::uint32_t* error_flag,
        std::uint32_t* particle_cut_count,
        std::uint32_t* observation_before_cut_count) {
      auto const index =
          static_cast<std::size_t>(blockIdx.x) * blockDim.x +
          threadIdx.x;
      if (index >= count) {
        return;
      }
      auto const record = records[index];
      if (record.input_index >= source_count) {
        atomicExch(error_flag, 1U);
        return;
      }
      auto const source =
          static_cast<std::size_t>(record.input_index);
      if (record.limit == PhotonTransportLimit::LayerBoundary) {
        if (atomicCAS(next_flags + source, 0U, 1U) != 0U) {
          atomicExch(error_flag, 2U);
          return;
        }
        raw_next[source] = record.end;
      } else if (
          record.limit ==
              PhotonTransportLimit::ObservationSurface ||
          record.limit ==
              PhotonTransportLimit::EscapedEnvironment) {
        if (atomicCAS(
                observation_flags + source, 0U, 1U) != 0U) {
          atomicExch(error_flag, 3U);
          return;
        }
        ObservationRecord observation{};
        observation.particle = record.end;
        observation.status =
            record.limit ==
                    PhotonTransportLimit::ObservationSurface
                ? ObservationStatus::ReachedObservationSurface
                : ObservationStatus::EscapedEnvironment;
        raw_observations[source] = observation;
      } else if (
          record.limit ==
          PhotonTransportLimit::ParticleCut) {
        if (record.observation_surface_reached_before_cut != 0U) {
          if (atomicCAS(
                  observation_flags + source, 0U, 1U) != 0U) {
            atomicExch(error_flag, 3U);
            return;
          }
          ObservationRecord observation{};
          observation.particle = record.end;
          observation.status =
              ObservationStatus::ReachedObservationSurface;
          raw_observations[source] = observation;
          atomicAdd(observation_before_cut_count, 1U);
        }
        atomicAdd(particle_cut_count, 1U);
      }
    }

    __global__ void scatterSuppressedPhotonsKernel(
        PhotonPairLpmSuppressionRecord const* suppressions,
        std::size_t count,
        std::uint32_t const* device_count,
        std::size_t source_count,
        EmParticleState* raw_next, std::uint32_t* next_flags,
        std::uint32_t* error_flag) {
      auto const index =
          static_cast<std::size_t>(blockIdx.x) * blockDim.x +
          threadIdx.x;
      if (index >= count ||
          (device_count != nullptr &&
           index >= *device_count)) {
        return;
      }
      auto const suppression = suppressions[index];
      if (suppression.input_index >= source_count) {
        atomicExch(error_flag, 4U);
        return;
      }
      auto const source =
          static_cast<std::size_t>(suppression.input_index);
      if (atomicCAS(next_flags + source, 0U, 1U) != 0U) {
        atomicExch(error_flag, 5U);
        return;
      }
      raw_next[source] = suppression.particle;
    }

    __global__ void scatterGeneratedPhotonsKernel(
        PhotonPairFinalStateRecord const* records,
        std::size_t count,
        std::uint32_t const* device_count,
        EmParticleState const* secondaries,
        std::size_t source_count, EmParticleState* raw_next,
        std::uint32_t* next_flags,
        std::uint32_t* error_flag) {
      auto const index =
          static_cast<std::size_t>(blockIdx.x) * blockDim.x +
          threadIdx.x;
      if (index >= count ||
          (device_count != nullptr &&
           index >= *device_count)) {
        return;
      }
      auto const record = records[index];
      if (record.process_id != ComptonProcessId) {
        return;
      }
      if (record.input_index >= source_count ||
          record.secondary_count > 2) {
        atomicExch(error_flag, 6U);
        return;
      }
      EmParticleState photon{};
      bool found_photon = false;
      for (std::uint32_t child = 0;
           child < record.secondary_count; ++child) {
        auto const candidate =
            secondaries[record.secondary_offset + child];
        if (candidate.pid ==
            static_cast<std::int32_t>(EmPid::Photon)) {
          if (found_photon) {
            atomicExch(error_flag, 7U);
            return;
          }
          photon = candidate;
          found_photon = true;
        }
      }
      // Thinning may retain only the recoil electron or discard both
      // children. In that case this source has no resident photon endpoint.
      if (!found_photon) {
        return;
      }
      auto const source =
          static_cast<std::size_t>(record.input_index);
      if (atomicCAS(next_flags + source, 0U, 1U) != 0U) {
        atomicExch(error_flag, 8U);
        return;
      }
      raw_next[source] = photon;
    }

    __global__ void compactPhotonEndpointsKernel(
        EmParticleState const* raw_next,
        std::uint32_t const* next_flags,
        std::uint32_t const* next_offsets,
        ObservationRecord const* raw_observations,
        std::uint32_t const* observation_flags,
        std::uint32_t const* observation_offsets,
        std::size_t source_count,
        EmParticleState* compact_next,
        ObservationRecord* compact_observations) {
      auto const index =
          static_cast<std::size_t>(blockIdx.x) * blockDim.x +
          threadIdx.x;
      if (index >= source_count) {
        return;
      }
      if (next_flags[index] != 0) {
        compact_next[next_offsets[index]] = raw_next[index];
      }
      if (observation_flags[index] != 0) {
        compact_observations[observation_offsets[index]] =
            raw_observations[index];
      }
    }

    __global__ void finalizePhotonEndpointCountsKernel(
        std::uint32_t const* next_flags,
        std::uint32_t const* next_offsets,
        std::uint32_t const* observation_flags,
        std::uint32_t const* observation_offsets,
        std::size_t source_count,
        std::uint32_t const* error_flag,
        std::size_t const* generated_lepton_count,
        bool generated_leptons_compacted,
        std::uint32_t const* final_state_summary,
        std::uint32_t* summary) {
      if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
      }
      auto const last = source_count - 1;
      summary[0] =
          next_offsets[last] + next_flags[last];
      summary[1] =
          observation_offsets[last] +
          observation_flags[last];
      summary[2] = *error_flag;
      if (generated_leptons_compacted) {
        auto const count = *generated_lepton_count;
        if (count >
            static_cast<std::size_t>(0xffffffffU)) {
          summary[2] = 9U;
        } else {
          summary[4] =
              static_cast<std::uint32_t>(count);
          summary[5] = 1U;
        }
      }
      for (std::size_t index = 0;
           index <
           detail::PhotonFinalStateSummaryLayout::Size;
           ++index) {
        summary[7 + index] =
            final_state_summary == nullptr
                ? 0U
                : final_state_summary[index];
      }
    }

  } // namespace

  namespace detail {

    void appendPhotonEndpointWorkspace(
        WorkspaceSize& required, std::size_t source_count) {
      std::size_t scan_bytes = 0;
      checkCuda(cub::DeviceScan::ExclusiveSum(
                    nullptr, scan_bytes,
                    static_cast<std::uint32_t*>(nullptr),
                    static_cast<std::uint32_t*>(nullptr),
                    source_count),
                "query photon endpoint scan storage");
      required.add<EmParticleState>(source_count);
      required.add<std::uint32_t>(source_count);
      required.add<std::uint32_t>(source_count);
      required.add<ObservationRecord>(source_count);
      required.add<std::uint32_t>(source_count);
      required.add<std::uint32_t>(source_count);
      required.add<EmParticleState>(source_count);
      required.add<ObservationRecord>(source_count);
      required.add<std::uint32_t>(1);
      required.add<std::uint32_t>(
          PhotonEndpointSummarySize);
      required.addBytes(scan_bytes);
    }

    DevicePhotonEndpointBatch compactPhotonEndpointsOnDevice(
        PhotonTransportRecord const* transport_records,
        std::size_t transport_count,
        DevicePhotonPairFinalStateBatch& final_state,
        std::size_t source_count,
        DeviceChargedSecondarySink const* charged_secondary_sink,
        DeviceWorkspace& workspace) {
      if (source_count == 0) {
        return {};
      }
      std::size_t scan_bytes = 0;
      checkCuda(cub::DeviceScan::ExclusiveSum(
                    nullptr, scan_bytes,
                    static_cast<std::uint32_t*>(nullptr),
                    static_cast<std::uint32_t*>(nullptr),
                    source_count),
                "query photon endpoint scan storage");
      auto* raw_next =
          workspace.acquire<EmParticleState>(source_count);
      auto* next_flags =
          workspace.acquire<std::uint32_t>(source_count);
      auto* next_offsets =
          workspace.acquire<std::uint32_t>(source_count);
      auto* raw_observations =
          workspace.acquire<ObservationRecord>(source_count);
      auto* observation_flags =
          workspace.acquire<std::uint32_t>(source_count);
      auto* observation_offsets =
          workspace.acquire<std::uint32_t>(source_count);
      auto* compact_next =
          workspace.acquire<EmParticleState>(source_count);
      auto* compact_observations =
          workspace.acquire<ObservationRecord>(source_count);
      auto* error_flag = workspace.acquire<std::uint32_t>(1);
      auto* device_summary =
          workspace.acquire<std::uint32_t>(
              PhotonEndpointSummarySize);
      auto* scan_temporary =
          workspace.acquireBytes(scan_bytes);
      checkCuda(cudaMemset(
                    next_flags, 0,
                    source_count * sizeof(std::uint32_t)),
                "clear next-photon flags");
      checkCuda(cudaMemset(
                    observation_flags, 0,
                    source_count * sizeof(std::uint32_t)),
                "clear observation flags");
      checkCuda(cudaMemset(
                    error_flag, 0, sizeof(std::uint32_t)),
                "clear endpoint error flag");
      checkCuda(cudaMemset(
                    device_summary, 0,
                    PhotonEndpointSummarySize *
                        sizeof(std::uint32_t)),
                "clear photon endpoint summary");
      if (transport_count != 0) {
        auto const blocks = static_cast<unsigned int>(
            (transport_count + ThreadsPerBlock - 1) /
            ThreadsPerBlock);
        scatterTransportEndpointsKernel<<<blocks, ThreadsPerBlock>>>(
            transport_records, transport_count, source_count,
            raw_next, next_flags, raw_observations,
            observation_flags, error_flag,
            device_summary + 3, device_summary + 6);
        checkCuda(cudaGetLastError(),
                  "scatter transport endpoints launch");
      }
      auto const deferred_final_state =
          final_state.counts_deferred;
      auto const final_state_launch_count =
          deferred_final_state
              ? final_state.input_count
              : final_state.gpu_interaction_count;
      auto const suppression_launch_count =
          deferred_final_state
              ? final_state.input_count
              : final_state.suppression_count;
      auto const secondary_selection_count =
          deferred_final_state
              ? 2 * final_state.input_count
              : final_state.secondary_count;
      auto const* final_state_summary =
          final_state.device_summary;
      if (deferred_final_state &&
          final_state_summary == nullptr) {
        throw std::logic_error(
            "deferred photon final state has no device summary");
      }
      auto const* final_state_record_count =
          deferred_final_state
              ? final_state_summary +
                    PhotonFinalStateSummaryLayout::GpuCount
              : nullptr;
      auto const* final_state_secondary_count =
          deferred_final_state
              ? final_state_summary +
                    PhotonFinalStateSummaryLayout::
                        SecondaryCount
              : nullptr;
      auto const* final_state_suppression_count =
          deferred_final_state
              ? final_state_summary +
                    PhotonFinalStateSummaryLayout::
                        SuppressionCount
              : nullptr;
      if (suppression_launch_count != 0) {
        auto const blocks = static_cast<unsigned int>(
            (suppression_launch_count +
             ThreadsPerBlock - 1) /
            ThreadsPerBlock);
        scatterSuppressedPhotonsKernel<<<blocks, ThreadsPerBlock>>>(
            final_state.suppressions,
            suppression_launch_count,
            final_state_suppression_count, source_count,
            raw_next, next_flags, error_flag);
        checkCuda(cudaGetLastError(),
                  "scatter LPM-suppressed photons launch");
      }
      if (final_state_launch_count != 0) {
        if (final_state.records == nullptr ||
            final_state.secondaries == nullptr) {
          throw std::invalid_argument(
              "generated-photon compaction requires final-state storage");
        }
        auto const blocks = static_cast<unsigned int>(
            (final_state_launch_count +
             ThreadsPerBlock - 1) /
            ThreadsPerBlock);
        scatterGeneratedPhotonsKernel<<<blocks, ThreadsPerBlock>>>(
            final_state.records, final_state_launch_count,
            final_state_record_count,
            final_state.secondaries, source_count,
            raw_next, next_flags, error_flag);
        checkCuda(cudaGetLastError(),
                  "scatter generated photons launch");
      }
      checkCuda(cub::DeviceScan::ExclusiveSum(
                    scan_temporary, scan_bytes, next_flags,
                    next_offsets, source_count),
                "scan next-photon flags");
      checkCuda(cub::DeviceScan::ExclusiveSum(
                    scan_temporary, scan_bytes, observation_flags,
                    observation_offsets, source_count),
                "scan observation flags");
      bool generated_leptons_compacted = false;
      if (charged_secondary_sink != nullptr) {
        if (charged_secondary_sink->output == nullptr ||
            charged_secondary_sink->selected_count == nullptr ||
            charged_secondary_sink->temporary_storage == nullptr ||
            charged_secondary_sink->temporary_storage_bytes == 0) {
          throw std::invalid_argument(
              "charged-secondary sink is incomplete");
        }
        if (secondary_selection_count == 0) {
          checkCuda(
              cudaMemset(
                  charged_secondary_sink->selected_count, 0,
                  sizeof(std::size_t)),
              "clear resident charged-secondary count");
          generated_leptons_compacted = true;
        } else if (
            secondary_selection_count <=
            charged_secondary_sink->capacity) {
          if (final_state.secondaries == nullptr) {
            throw std::invalid_argument(
                "charged-secondary compaction requires final-state storage");
          }
          auto selection_temporary_bytes =
              charged_secondary_sink
                  ->temporary_storage_bytes;
          cub::CountingInputIterator<std::size_t>
              secondary_indices(0);
          auto secondary_input =
              cub::TransformInputIterator<
                  EmParticleState,
                  LoadValidPhotonSecondary,
                  cub::CountingInputIterator<std::size_t>>(
                  secondary_indices,
                  LoadValidPhotonSecondary{
                      final_state.secondaries,
                      final_state_secondary_count});
          checkCuda(
              cub::DeviceSelect::If(
                  charged_secondary_sink->temporary_storage,
                  selection_temporary_bytes,
                  secondary_input,
                  charged_secondary_sink->output,
                  charged_secondary_sink->selected_count,
                  secondary_selection_count,
                  IsChargedEmParticle{}),
              "select resident charged photon secondaries");
          if (selection_temporary_bytes >
              charged_secondary_sink
                  ->temporary_storage_bytes) {
            throw std::runtime_error(
                "charged-secondary selection storage is too small");
          }
          generated_leptons_compacted = true;
        }
      }
      finalizePhotonEndpointCountsKernel<<<1, 1>>>(
          next_flags, next_offsets, observation_flags,
          observation_offsets, source_count, error_flag,
          charged_secondary_sink == nullptr
              ? nullptr
              : charged_secondary_sink->selected_count,
          generated_leptons_compacted,
          final_state_summary,
          device_summary);
      checkCuda(
          cudaGetLastError(),
          "finalize photon endpoint counts launch");
      std::array<
          std::uint32_t,
          PhotonEndpointSummarySize>
          host_summary{};
      checkCuda(
          cudaMemcpy(
              host_summary.data(), device_summary,
              sizeof(host_summary), cudaMemcpyDeviceToHost),
          "download photon endpoint summary");
      auto const next_count =
          static_cast<std::size_t>(host_summary[0]);
      auto const observation_count =
          static_cast<std::size_t>(host_summary[1]);
      auto const endpoint_error = host_summary[2];
      auto const particle_cut_count =
          static_cast<std::size_t>(host_summary[3]);
      auto const generated_lepton_count =
          static_cast<std::size_t>(host_summary[4]);
      auto const selection_succeeded =
          host_summary[5] != 0;
      auto const observation_before_cut_count =
          static_cast<std::size_t>(host_summary[6]);
      auto final_count = [&](std::size_t index) {
        return static_cast<std::size_t>(
            host_summary[7 + index]);
      };
      if (deferred_final_state) {
        final_state.input_count =
            final_count(
                PhotonFinalStateSummaryLayout::InputCount);
        final_state.secondary_count =
            final_count(
                PhotonFinalStateSummaryLayout::
                    SecondaryCount);
        final_state.gpu_interaction_count =
            final_count(
                PhotonFinalStateSummaryLayout::GpuCount);
        final_state.fallback_count =
            final_count(
                PhotonFinalStateSummaryLayout::
                    FallbackCount);
        final_state.continuation_count =
            final_count(
                PhotonFinalStateSummaryLayout::
                    ContinuationCount);
        final_state.suppression_count =
            final_count(
                PhotonFinalStateSummaryLayout::
                    SuppressionCount);
        final_state.photon_pair_interaction_count =
            final_count(
                PhotonFinalStateSummaryLayout::
                    PhotonPairCount);
        final_state.compton_interaction_count =
            final_count(
                PhotonFinalStateSummaryLayout::
                    ComptonCount);
        final_state.photoelectric_interaction_count =
            final_count(
                PhotonFinalStateSummaryLayout::
                    PhotoelectricCount);
        final_state.counts_deferred = false;
      }
      auto const final_state_error =
          host_summary[
              7 + PhotonFinalStateSummaryLayout::Error];
      if (final_state_error == 2U) {
        throw std::runtime_error(
            "photon final-state classification lost or duplicated an interaction");
      }
      if (final_state_error == 3U) {
        throw std::overflow_error(
            "photon-pair secondary history ID overflow");
      }
      if (final_state_error != 0U) {
        throw std::runtime_error(
            "photon final-state generation failed");
      }
      if (endpoint_error != 0) {
        throw std::runtime_error(
            "photon endpoint queues contain duplicate or invalid source indices");
      }
      auto const blocks = static_cast<unsigned int>(
          (source_count + ThreadsPerBlock - 1) /
          ThreadsPerBlock);
      compactPhotonEndpointsKernel<<<blocks, ThreadsPerBlock>>>(
          raw_next, next_flags, next_offsets, raw_observations,
          observation_flags, observation_offsets, source_count,
          compact_next, compact_observations);
      checkCuda(cudaGetLastError(),
                "compact photon endpoint queues launch");
      DevicePhotonEndpointBatch result{};
      result.source_count = source_count;
      result.next_photon_count = next_count;
      result.observation_count = observation_count;
      result.particle_cut_count = particle_cut_count;
      result.observation_before_cut_count =
          observation_before_cut_count;
      result.generated_lepton_count =
          generated_lepton_count;
      result.generated_leptons_compacted =
          selection_succeeded;
      result.next_photons = compact_next;
      result.observations = compact_observations;
      return result;
    }

    void appendPhotonDevicePipelineWorkspace(
        WorkspaceSize& required, std::size_t source_count) {
      appendInteractionSelectionWorkspace(required, source_count);
      appendPhotonTransportWorkspace(required, source_count);
      appendTransportInteractionWorkspace(required, source_count);
      appendPhotonPairFinalStateWorkspace(required, source_count);
      appendPhotonEndpointWorkspace(required, source_count);
    }

    DevicePhotonPipelineBatch launchPhotonDevicePipelineOnDevice(
        tables::FlatRateTableView device_table,
        PhotonPairLpmSnapshot const& lpm_snapshot,
        EmThinningConfig const& thinning,
        EnvironmentSnapshot const& environment,
        EmParticleState const* device_particles,
        std::size_t count, std::uint64_t random_seed,
        std::uint64_t shower_id,
        std::uint64_t first_secondary_history_id,
        DeviceWorkspace& workspace,
        DeviceChargedSecondarySink const*
            charged_secondary_sink,
        DeviceFirstInteractionCapture const*
            first_interaction) {
      DevicePhotonPipelineBatch pipeline{};
      pipeline.selection = launchInteractionSelectionOnDevice(
          device_table, device_particles, count, random_seed,
          shower_id, workspace, true);
      pipeline.transport = launchPhotonTransportOnDevice(
          environment, pipeline.selection.interactions,
          count, workspace, &pipeline.selection);
      if (pipeline.transport.record_count == 0) {
        return pipeline;
      }
      if (pipeline.transport.record_count != 0) {
      pipeline.at_interaction =
            extractTransportInteractionsOnDevice(
                pipeline.transport.records,
                pipeline.transport.record_count, workspace,
                true);
      }
      pipeline.final_state =
          launchPhotonPairFinalStateOnDevice(
              device_table, lpm_snapshot, thinning,
              pipeline.at_interaction.interactions,
              pipeline.at_interaction.input_count,
              random_seed, shower_id,
              first_secondary_history_id, workspace,
              true, &pipeline.at_interaction,
              first_interaction);
      pipeline.endpoints = compactPhotonEndpointsOnDevice(
          pipeline.transport.records,
          pipeline.transport.record_count,
          pipeline.final_state, count,
          charged_secondary_sink, workspace);
      pipeline.at_interaction.interaction_count =
          pipeline.final_state.input_count;
      pipeline.at_interaction.count_deferred = false;
      return pipeline;
    }

  } // namespace detail

  PhotonSelectionTransportBatchResult
  selectAndTransportPhotonsForValidation(
      tables::FlatRateTableView device_table,
      EnvironmentSnapshot const& environment,
      std::vector<EmParticleState> const& particles,
      std::uint64_t random_seed, std::uint64_t shower_id,
      int device, detail::DeviceWorkspace& workspace) {
    PhotonSelectionTransportBatchResult result{};
    result.input_particles = particles.size();
    if (particles.empty()) {
      return result;
    }
    if (device < 0) {
      throw std::invalid_argument(
          "selection-transport CUDA device must be non-negative");
    }
    if (!atmosphere_detail::validEnvironment(environment)) {
      throw std::invalid_argument(
          "selection-transport requires a valid spherical environment");
    }
    if (particles.size() >
        std::numeric_limits<std::uint32_t>::max()) {
      throw std::length_error(
          "selection-transport batch exceeds 32-bit scan offsets");
    }
    checkCuda(cudaSetDevice(device),
              "cudaSetDevice(selection-transport)");

    detail::WorkspaceSize required;
    required.add<EmParticleState>(particles.size());
    detail::appendInteractionSelectionWorkspace(
        required, particles.size());
    detail::appendPhotonTransportWorkspace(
        required, particles.size());
    workspace.prepare(required.bytes());
    auto* device_particles =
        workspace.acquire<EmParticleState>(particles.size());
    checkCuda(cudaMemcpy(
                  device_particles, particles.data(),
                  particles.size() * sizeof(EmParticleState),
                  cudaMemcpyHostToDevice),
              "upload selection-transport photons");

    auto const selection =
        detail::launchInteractionSelectionOnDevice(
            device_table, device_particles, particles.size(),
            random_seed, shower_id, workspace);
    result.selection_fallback_events.resize(
        selection.fallback_count);
    if (selection.fallback_count != 0) {
      checkCuda(cudaMemcpy(
                    result.selection_fallback_events.data(),
                    selection.fallbacks,
                    selection.fallback_count *
                        sizeof(ProposalFallbackEvent),
                    cudaMemcpyDeviceToHost),
                "download chained selection fallbacks");
    }

    if (selection.interaction_count == 0) {
      return result;
    }
    auto const transport = detail::launchPhotonTransportOnDevice(
        environment, selection.interactions,
        selection.interaction_count, workspace);
    result.records.resize(transport.record_count);
    result.transport_fallback_events.resize(
        transport.fallback_count);
    if (transport.record_count != 0) {
      checkCuda(cudaMemcpy(
                    result.records.data(), transport.records,
                    transport.record_count *
                        sizeof(PhotonTransportRecord),
                    cudaMemcpyDeviceToHost),
                "download chained transport records");
    }
    if (transport.fallback_count != 0) {
      checkCuda(cudaMemcpy(
                    result.transport_fallback_events.data(),
                    transport.fallbacks,
                    transport.fallback_count *
                        sizeof(ProposalFallbackEvent),
                    cudaMemcpyDeviceToHost),
                "download chained transport fallbacks");
    }
    return result;
  }

  PhotonDevicePipelineBatchResult
  runPhotonDevicePipelineForValidation(
      tables::FlatRateTableView device_table,
      PhotonPairLpmSnapshot const& lpm_snapshot,
      EmThinningConfig const& thinning,
      EnvironmentSnapshot const& environment,
      std::vector<EmParticleState> const& particles,
      std::uint64_t random_seed, std::uint64_t shower_id,
      int device, std::uint64_t first_secondary_history_id,
      detail::DeviceWorkspace& workspace) {
    PhotonDevicePipelineBatchResult result{};
    result.input_particles = particles.size();
    if (particles.empty()) {
      return result;
    }
    if (device < 0 || first_secondary_history_id == 0) {
      throw std::invalid_argument(
          "photon device pipeline has invalid CUDA/history configuration");
    }
    if (!atmosphere_detail::validEnvironment(environment)) {
      throw std::invalid_argument(
          "photon device pipeline requires a valid environment");
    }
    if (particles.size() >
        std::numeric_limits<std::uint32_t>::max() / 2) {
      throw std::length_error(
          "photon device pipeline exceeds 32-bit offsets");
    }
    checkCuda(cudaSetDevice(device),
              "cudaSetDevice(photon device pipeline)");

    auto const count = particles.size();
    detail::WorkspaceSize required;
    required.add<EmParticleState>(count);
    detail::appendPhotonDevicePipelineWorkspace(required, count);
    workspace.prepare(required.bytes());
    auto* device_particles =
        workspace.acquire<EmParticleState>(count);
    checkCuda(cudaMemcpy(
                  device_particles, particles.data(),
                  count * sizeof(EmParticleState),
                  cudaMemcpyHostToDevice),
              "upload photon device pipeline input");

    auto const pipeline =
        detail::launchPhotonDevicePipelineOnDevice(
            device_table, lpm_snapshot, thinning, environment,
            device_particles, count, random_seed, shower_id,
            first_secondary_history_id, workspace);
    auto const& selection = pipeline.selection;
    result.selection_fallback_events.resize(
        selection.fallback_count);
    if (selection.fallback_count != 0) {
      checkCuda(cudaMemcpy(
                    result.selection_fallback_events.data(),
                    selection.fallbacks,
                    selection.fallback_count *
                        sizeof(ProposalFallbackEvent),
                    cudaMemcpyDeviceToHost),
                "download pipeline selection fallbacks");
    }
    auto const& transport = pipeline.transport;
    result.transport_records.resize(transport.record_count);
    result.transport_fallback_events.resize(
        transport.fallback_count);
    if (transport.record_count != 0) {
      checkCuda(cudaMemcpy(
                    result.transport_records.data(),
                    transport.records,
                    transport.record_count *
                        sizeof(PhotonTransportRecord),
                    cudaMemcpyDeviceToHost),
                "download pipeline transport records");
    }
    if (transport.fallback_count != 0) {
      checkCuda(cudaMemcpy(
                    result.transport_fallback_events.data(),
                    transport.fallbacks,
                    transport.fallback_count *
                        sizeof(ProposalFallbackEvent),
                    cudaMemcpyDeviceToHost),
                "download pipeline transport fallbacks");
    }
    auto const& at_interaction = pipeline.at_interaction;
    result.final_states.input_interactions =
        at_interaction.interaction_count;
    auto const& final_state = pipeline.final_state;
    result.next_photons.resize(
        pipeline.endpoints.next_photon_count);
    result.observations.resize(
        pipeline.endpoints.observation_count);
    if (!result.next_photons.empty()) {
      checkCuda(cudaMemcpy(
                    result.next_photons.data(),
                    pipeline.endpoints.next_photons,
                    result.next_photons.size() *
                        sizeof(EmParticleState),
                    cudaMemcpyDeviceToHost),
                "download pipeline next photons");
    }
    if (!result.observations.empty()) {
      checkCuda(cudaMemcpy(
                    result.observations.data(),
                    pipeline.endpoints.observations,
                    result.observations.size() *
                        sizeof(ObservationRecord),
                    cudaMemcpyDeviceToHost),
                "download pipeline observations");
    }
    auto& host = result.final_states;
    host.gpu_interactions =
        final_state.gpu_interaction_count;
    host.photon_pair_interactions =
        final_state.photon_pair_interaction_count;
    host.compton_interactions =
        final_state.compton_interaction_count;
    host.photoelectric_interactions =
        final_state.photoelectric_interaction_count;
    host.final_state_records.resize(
        final_state.gpu_interaction_count);
    host.secondaries.resize(final_state.secondary_count);
    host.fallback_events.resize(final_state.fallback_count);
    host.continuations.resize(final_state.continuation_count);
    host.lpm_suppressed.resize(final_state.suppression_count);
    if (!host.final_state_records.empty()) {
      checkCuda(cudaMemcpy(
                    host.final_state_records.data(),
                    final_state.records,
                    host.final_state_records.size() *
                        sizeof(PhotonPairFinalStateRecord),
                    cudaMemcpyDeviceToHost),
                "download pipeline final-state records");
    }
    if (!host.secondaries.empty()) {
      checkCuda(cudaMemcpy(
                    host.secondaries.data(),
                    final_state.secondaries,
                    host.secondaries.size() *
                        sizeof(EmParticleState),
                    cudaMemcpyDeviceToHost),
                "download pipeline secondaries");
    }
    if (!host.fallback_events.empty()) {
      checkCuda(cudaMemcpy(
                    host.fallback_events.data(),
                    final_state.fallbacks,
                    host.fallback_events.size() *
                        sizeof(ProposalFallbackEvent),
                    cudaMemcpyDeviceToHost),
                "download pipeline final-state fallbacks");
    }
    if (!host.continuations.empty()) {
      checkCuda(cudaMemcpy(
                    host.continuations.data(),
                    final_state.continuations,
                    host.continuations.size() *
                        sizeof(EmInteractionRecord),
                    cudaMemcpyDeviceToHost),
                "download pipeline continuations");
    }
    if (!host.lpm_suppressed.empty()) {
      checkCuda(cudaMemcpy(
                    host.lpm_suppressed.data(),
                    final_state.suppressions,
                    host.lpm_suppressed.size() *
                        sizeof(PhotonPairLpmSuppressionRecord),
                    cudaMemcpyDeviceToHost),
                "download pipeline LPM suppressions");
    }
    return result;
  }

} // namespace corsika::gpu::em

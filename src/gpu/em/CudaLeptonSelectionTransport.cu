/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include <cuda_runtime.h>

#include <cub/device/device_scan.cuh>
#include <cub/iterator/counting_input_iterator.cuh>
#include <cub/iterator/transform_input_iterator.cuh>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <type_traits>

#include <corsika/gpu/em/CudaBremsFinalState.hpp>
#include <corsika/gpu/em/CudaLeptonSelectionTransport.hpp>
#include <corsika/gpu/em/ProcessCapabilities.hpp>
#include <corsika/gpu/em/SphericalAtmosphere.hpp>
#include <corsika/gpu/em/detail/DeviceBatchStages.hpp>

namespace corsika::gpu::em {

  namespace {

    constexpr unsigned int ThreadsPerBlock = 256;
    constexpr std::size_t LeptonEndpointVertexOffset = 5;
    constexpr std::size_t LeptonEndpointFinalStateOffset =
        LeptonEndpointVertexOffset +
        detail::LeptonVertexSummaryLayout::Size;
    constexpr std::size_t LeptonEndpointSummarySize =
        LeptonEndpointFinalStateOffset +
        detail::BremsFinalStateSummaryLayout::Size;

    struct LeptonChildCategoryCounts {
      std::uint32_t leptons{};
      std::uint32_t photons{};
    };

    static_assert(
        sizeof(LeptonChildCategoryCounts) ==
        2 * sizeof(std::uint32_t));
    static_assert(
        std::is_trivially_copyable_v<
            LeptonChildCategoryCounts>);

    struct LeptonEndpointCategoryCounts {
      std::uint32_t leptons{};
      std::uint32_t photons{};
      std::uint32_t observations{};
      std::uint32_t decays{};
    };

    static_assert(
        sizeof(LeptonEndpointCategoryCounts) ==
        4 * sizeof(std::uint32_t));
    static_assert(
        std::is_trivially_copyable_v<
            LeptonEndpointCategoryCounts>);

    struct AddLeptonEndpointCategoryCounts {
      __host__ __device__ __forceinline__
      LeptonEndpointCategoryCounts operator()(
          LeptonEndpointCategoryCounts const& left,
          LeptonEndpointCategoryCounts const& right) const {
        return {
            left.leptons + right.leptons,
            left.photons + right.photons,
            left.observations + right.observations,
            left.decays + right.decays};
      }
    };

    struct LoadLeptonEndpointCategoryCounts {
      LeptonChildCategoryCounts const* children{};
      std::uint32_t const* observation_flags{};
      std::uint32_t const* decay_flags{};
      std::size_t source_count{};

      __host__ __device__ __forceinline__
      LeptonEndpointCategoryCounts operator()(
          std::size_t index) const {
        auto const child = children[index];
        return {
            child.leptons, child.photons,
            index < source_count
                ? observation_flags[index]
                : 0U,
            index < source_count
                ? decay_flags[index]
                : 0U};
      }
    };

    using LeptonEndpointCountInput =
        cub::TransformInputIterator<
            LeptonEndpointCategoryCounts,
            LoadLeptonEndpointCategoryCounts,
            cub::CountingInputIterator<std::size_t>>;

    LeptonEndpointCountInput makeLeptonEndpointCountInput(
        LeptonChildCategoryCounts const* children,
        std::uint32_t const* observation_flags,
        std::uint32_t const* decay_flags,
        std::size_t source_count) {
      return {
          cub::CountingInputIterator<std::size_t>{0},
          LoadLeptonEndpointCategoryCounts{
              children, observation_flags, decay_flags,
              source_count}};
    }

    void checkCuda(cudaError_t status, char const* operation) {
      if (status == cudaSuccess) {
        return;
      }
      std::ostringstream message;
      message << operation << " failed: "
              << cudaGetErrorString(status);
      throw std::runtime_error(message.str());
    }

    __device__ bool claimFlag(
        std::uint32_t* flag,
        std::uint32_t* error_flag, std::uint32_t error) {
      if (atomicCAS(flag, 0U, 1U) != 0U) {
        atomicExch(error_flag, error);
        return false;
      }
      return true;
    }

    __device__ bool claimSlot(
        std::uint32_t* flags, std::size_t slot,
        std::uint32_t* error_flag, std::uint32_t error) {
      return claimFlag(flags + slot, error_flag, error);
    }

    __global__ void scatterLeptonTransportEndpointsKernel(
        LeptonTransportRecord const* records, std::size_t count,
        std::size_t source_count, EmParticleState* raw_next,
        LeptonChildCategoryCounts* child_categories,
        ObservationRecord* raw_observations,
        std::uint32_t* observation_flags,
        EmParticleState* raw_decays,
        std::uint32_t* decay_flags,
        std::uint32_t* error_flag) {
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
      switch (record.limit) {
      case LeptonTransportLimit::ContinuousStep:
      case LeptonTransportLimit::LayerBoundary:
      case LeptonTransportLimit::MagneticStep: {
        auto const slot = 3 * source;
        if (claimFlag(
                &child_categories[slot].leptons,
                error_flag, 2U)) {
          raw_next[slot] = record.end;
        }
        break;
      }
      case LeptonTransportLimit::ObservationSurface:
      case LeptonTransportLimit::EscapedEnvironment:
        if (claimSlot(
                observation_flags, source, error_flag, 3U)) {
          ObservationRecord observation{};
          observation.particle = record.end;
          observation.status =
              record.limit ==
                      LeptonTransportLimit::ObservationSurface
                  ? ObservationStatus::
                        ReachedObservationSurface
                  : ObservationStatus::EscapedEnvironment;
          raw_observations[source] = observation;
        }
        break;
      case LeptonTransportLimit::ParticleCut:
      case LeptonTransportLimit::InteractionCandidate:
        break;
      case LeptonTransportLimit::DecayCandidate:
        if (claimSlot(
                decay_flags, source, error_flag, 4U)) {
          raw_decays[source] = record.end;
        }
        break;
      }
    }

    __global__ void scatterLeptonContinuationsKernel(
        EmInteractionRecord const* continuations,
        std::size_t count,
        std::uint32_t const* device_count,
        std::size_t source_count,
        EmParticleState* raw_next,
        LeptonChildCategoryCounts* child_categories,
        std::uint32_t* error_flag, std::uint32_t error_base) {
      auto const index =
          static_cast<std::size_t>(blockIdx.x) * blockDim.x +
          threadIdx.x;
      auto const active_count =
          device_count == nullptr
              ? count
              : static_cast<std::size_t>(*device_count);
      if (index >= active_count) {
        return;
      }
      auto const continuation = continuations[index];
      if (continuation.input_index >= source_count) {
        atomicExch(error_flag, error_base);
        return;
      }
      auto const slot =
          3 * static_cast<std::size_t>(
                  continuation.input_index);
      if (claimFlag(
              &child_categories[slot].leptons,
              error_flag,
              error_base + 1U)) {
        raw_next[slot] = continuation.particle;
      }
    }

    __global__ void scatterBremsSuppressionsKernel(
        BremsLpmSuppressionRecord const* suppressions,
        std::size_t count,
        std::uint32_t const* device_count,
        std::size_t source_count,
        EmParticleState* raw_next,
        LeptonChildCategoryCounts* child_categories,
        std::uint32_t* error_flag) {
      auto const index =
          static_cast<std::size_t>(blockIdx.x) * blockDim.x +
          threadIdx.x;
      auto const active_count =
          device_count == nullptr
              ? count
              : static_cast<std::size_t>(*device_count);
      if (index >= active_count) {
        return;
      }
      auto const suppression = suppressions[index];
      if (suppression.input_index >= source_count) {
        atomicExch(error_flag, 8U);
        return;
      }
      auto const slot =
          3 * static_cast<std::size_t>(
                  suppression.input_index);
      if (claimFlag(
              &child_categories[slot].leptons,
              error_flag, 9U)) {
        raw_next[slot] = suppression.particle;
      }
    }

    __global__ void scatterLeptonFinalStateChildrenKernel(
        BremsFinalStateRecord const* records, std::size_t count,
        std::uint32_t const* device_count,
        EmParticleState const* secondaries,
        std::size_t source_count, EmParticleState* raw_next,
        LeptonChildCategoryCounts* child_categories,
        EmParticleState* raw_photons,
        std::uint32_t* error_flag) {
      auto const index =
          static_cast<std::size_t>(blockIdx.x) * blockDim.x +
          threadIdx.x;
      auto const active_count =
          device_count == nullptr
              ? count
              : static_cast<std::size_t>(*device_count);
      if (index >= active_count) {
        return;
      }
      auto const record = records[index];
      auto const valid_process =
          ((record.process_id == BremsProcessId ||
            record.process_id == AnnihilationProcessId ||
            record.process_id == IonizationProcessId) &&
           record.secondary_count <= 2) ||
          (record.process_id == ElectronPairProcessId &&
           record.secondary_count == 3);
      if (record.input_index >= source_count ||
          !valid_process) {
        atomicExch(error_flag, 10U);
        return;
      }
      auto const source =
          static_cast<std::size_t>(record.input_index);
      std::size_t lepton_index = 0;
      std::size_t photon_index = 0;
      for (std::size_t child = 0;
           child < record.secondary_count; ++child) {
        auto const particle =
            secondaries[record.secondary_offset + child];
        if (isChargedLeptonPid(particle.pid)) {
          if (lepton_index >= 3) {
            atomicExch(error_flag, 11U);
            return;
          }
          auto const slot = 3 * source + lepton_index++;
          if (!claimFlag(
                  &child_categories[slot].leptons,
                  error_flag, 12U)) {
            return;
          }
          raw_next[slot] = particle;
        } else if (
            particle.pid ==
            static_cast<std::int32_t>(EmPid::Photon)) {
          if (photon_index >= 3) {
            atomicExch(error_flag, 13U);
            return;
          }
          auto const slot =
              3 * source + photon_index++;
          if (!claimFlag(
                  &child_categories[slot].photons,
                  error_flag, 14U)) {
            return;
          }
          raw_photons[slot] = particle;
        } else {
          atomicExch(error_flag, 15U);
          return;
        }
      }
    }

    __global__ void compactLeptonEndpointsKernel(
        EmParticleState const* raw_next,
        EmParticleState const* raw_photons,
        LeptonChildCategoryCounts const* child_categories,
        LeptonEndpointCategoryCounts const* endpoint_offsets,
        std::size_t child_slot_count,
        ObservationRecord const* raw_observations,
        std::uint32_t const* observation_flags,
        EmParticleState const* raw_decays,
        std::uint32_t const* decay_flags,
        std::size_t source_count,
        EmParticleState* compact_next,
        EmParticleState* compact_photons,
        ObservationRecord* compact_observations,
        EmParticleState* compact_decays,
        std::uint32_t const* error_flag,
        std::uint32_t const* vertex_summary,
        std::uint32_t const* final_state_summary,
        std::uint32_t* summary) {
      auto const index =
          static_cast<std::size_t>(blockIdx.x) * blockDim.x +
          threadIdx.x;
      if (index >= child_slot_count) {
        return;
      }
      auto const category = child_categories[index];
      auto const offset = endpoint_offsets[index];
      if (category.leptons != 0) {
        compact_next[offset.leptons] = raw_next[index];
      }
      if (category.photons != 0) {
        compact_photons[offset.photons] =
            raw_photons[index];
      }
      if (index < source_count &&
          observation_flags[index] != 0) {
        compact_observations[offset.observations] =
            raw_observations[index];
      }
      if (index < source_count &&
          decay_flags[index] != 0) {
        compact_decays[offset.decays] =
            raw_decays[index];
      }
      if (index == 0) {
        auto const child_last = child_slot_count - 1;
        summary[0] =
            endpoint_offsets[child_last].leptons +
            child_categories[child_last].leptons;
        summary[1] =
            endpoint_offsets[child_last].photons +
            child_categories[child_last].photons;
        summary[2] =
            endpoint_offsets[child_last].observations;
        summary[3] =
            endpoint_offsets[child_last].decays;
        summary[4] = *error_flag;
        for (std::size_t field = 0;
             field < detail::LeptonVertexSummaryLayout::Size;
             ++field) {
          summary[LeptonEndpointVertexOffset + field] =
              vertex_summary == nullptr
                  ? 0U
                  : vertex_summary[field];
        }
        for (std::size_t field = 0;
             field <
             detail::BremsFinalStateSummaryLayout::Size;
             ++field) {
          summary[LeptonEndpointFinalStateOffset + field] =
              final_state_summary == nullptr
                  ? 0U
                  : final_state_summary[field];
        }
      }
    }

  } // namespace

  namespace detail {

    void appendLeptonEndpointWorkspace(
        WorkspaceSize& required, std::size_t source_count) {
      if (source_count >
          std::numeric_limits<std::size_t>::max() / 3) {
        throw std::length_error(
            "lepton endpoint source count overflow");
      }
      auto const child_slots = 3 * source_count;
      std::size_t scan_bytes = 0;
      checkCuda(cub::DeviceScan::ExclusiveScan(
                    nullptr, scan_bytes,
                    makeLeptonEndpointCountInput(
                        nullptr, nullptr, nullptr,
                        source_count),
                    static_cast<
                        LeptonEndpointCategoryCounts*>(nullptr),
                    AddLeptonEndpointCategoryCounts{},
                    LeptonEndpointCategoryCounts{},
                    child_slots),
                "query lepton endpoint scan storage");
      required.add<EmParticleState>(child_slots);
      required.add<LeptonChildCategoryCounts>(child_slots);
      required.add<LeptonEndpointCategoryCounts>(child_slots);
      required.add<EmParticleState>(child_slots);
      required.add<ObservationRecord>(source_count);
      required.add<std::uint32_t>(source_count);
      required.add<EmParticleState>(source_count);
      required.add<std::uint32_t>(source_count);
      required.add<EmParticleState>(child_slots);
      required.add<EmParticleState>(child_slots);
      required.add<ObservationRecord>(source_count);
      required.add<EmParticleState>(source_count);
      required.add<std::uint32_t>(1);
      required.add<std::uint32_t>(
          LeptonEndpointSummarySize);
      required.addBytes(scan_bytes);
    }

    DeviceLeptonEndpointBatch compactLeptonEndpointsOnDevice(
        LeptonTransportRecord const* transport_records,
        std::size_t transport_count,
        DeviceLeptonVertexSelectionBatch& vertex,
        DeviceBremsFinalStateBatch& final_state,
        std::size_t source_count,
        DeviceWorkspace& workspace) {
      if (source_count == 0) {
        return {};
      }
      if (source_count >
          std::numeric_limits<std::uint32_t>::max() / 3) {
        throw std::length_error(
            "lepton endpoint batch exceeds 32-bit offsets");
      }
      if (vertex.counts_deferred &&
          vertex.device_summary == nullptr) {
        throw std::logic_error(
            "deferred lepton vertex selection has no device summary");
      }
      auto const child_slots = 3 * source_count;
      std::size_t scan_bytes = 0;
      checkCuda(cub::DeviceScan::ExclusiveScan(
                    nullptr, scan_bytes,
                    makeLeptonEndpointCountInput(
                        nullptr, nullptr, nullptr,
                        source_count),
                    static_cast<
                        LeptonEndpointCategoryCounts*>(nullptr),
                    AddLeptonEndpointCategoryCounts{},
                    LeptonEndpointCategoryCounts{},
                    child_slots),
                "query lepton endpoint scan storage");
      auto* raw_next =
          workspace.acquire<EmParticleState>(child_slots);
      auto* child_categories =
          workspace.acquire<LeptonChildCategoryCounts>(
              child_slots);
      auto* endpoint_offsets =
          workspace.acquire<LeptonEndpointCategoryCounts>(
              child_slots);
      auto* raw_photons =
          workspace.acquire<EmParticleState>(child_slots);
      auto* raw_observations =
          workspace.acquire<ObservationRecord>(source_count);
      auto* observation_flags =
          workspace.acquire<std::uint32_t>(source_count);
      auto* raw_decays =
          workspace.acquire<EmParticleState>(source_count);
      auto* decay_flags =
          workspace.acquire<std::uint32_t>(source_count);
      auto* compact_next =
          workspace.acquire<EmParticleState>(child_slots);
      auto* compact_photons =
          workspace.acquire<EmParticleState>(child_slots);
      auto* compact_observations =
          workspace.acquire<ObservationRecord>(source_count);
      auto* compact_decays =
          workspace.acquire<EmParticleState>(source_count);
      auto* error_flag =
          workspace.acquire<std::uint32_t>(1);
      auto* device_summary =
          workspace.acquire<std::uint32_t>(
              LeptonEndpointSummarySize);
      auto* scan_temporary =
          workspace.acquireBytes(scan_bytes);
      checkCuda(cudaMemset(
                    child_categories, 0,
                    child_slots *
                        sizeof(LeptonChildCategoryCounts)),
                "clear lepton child categories");
      checkCuda(cudaMemset(
                    observation_flags, 0,
                    source_count * sizeof(std::uint32_t)),
                "clear lepton observation flags");
      checkCuda(cudaMemset(
                    decay_flags, 0,
                    source_count * sizeof(std::uint32_t)),
                "clear lepton decay flags");
      checkCuda(cudaMemset(
                    error_flag, 0, sizeof(std::uint32_t)),
                "clear lepton endpoint error flag");

      auto launchContinuations =
          [&](EmInteractionRecord const* records,
              std::size_t count,
              std::uint32_t const* device_count,
              std::uint32_t error_base,
              char const* operation) {
            if (count == 0) {
              return;
            }
            auto const blocks = static_cast<unsigned int>(
                (count + ThreadsPerBlock - 1) /
                ThreadsPerBlock);
            scatterLeptonContinuationsKernel
                <<<blocks, ThreadsPerBlock>>>(
                    records, count, device_count,
                    source_count, raw_next,
                    child_categories,
                    error_flag, error_base);
            checkCuda(cudaGetLastError(), operation);
          };
      if (transport_count != 0) {
        auto const blocks = static_cast<unsigned int>(
            (transport_count + ThreadsPerBlock - 1) /
            ThreadsPerBlock);
        scatterLeptonTransportEndpointsKernel
            <<<blocks, ThreadsPerBlock>>>(
                transport_records, transport_count,
                source_count, raw_next, child_categories,
                raw_observations, observation_flags,
                raw_decays, decay_flags,
                error_flag);
        checkCuda(
            cudaGetLastError(),
            "scatter lepton transport endpoints launch");
      }
      launchContinuations(
          vertex.continuations,
          vertex.counts_deferred
              ? vertex.input_count
              : vertex.continuation_count,
          vertex.counts_deferred
              ? vertex.device_summary +
                    LeptonVertexSummaryLayout::
                        ContinuationCount
              : nullptr,
          4U,
          "scatter vertex continuations launch");
      auto const deferred_final_state =
          final_state.counts_deferred;
      auto const final_state_launch_count =
          deferred_final_state
              ? final_state.input_count
              : final_state.gpu_interaction_count;
      auto const continuation_launch_count =
          deferred_final_state
              ? final_state.input_count
              : final_state.continuation_count;
      auto const suppression_launch_count =
          deferred_final_state
              ? final_state.input_count
              : final_state.suppression_count;
      auto const* final_state_summary =
          final_state.device_summary;
      if (deferred_final_state &&
          final_state_summary == nullptr) {
        throw std::logic_error(
            "deferred lepton final state has no device summary");
      }
      auto const* final_state_record_count =
          deferred_final_state
              ? final_state_summary +
                    BremsFinalStateSummaryLayout::GpuCount
              : nullptr;
      auto const* final_state_continuation_count =
          deferred_final_state
              ? final_state_summary +
                    BremsFinalStateSummaryLayout::
                        ContinuationCount
              : nullptr;
      auto const* final_state_suppression_count =
          deferred_final_state
              ? final_state_summary +
                    BremsFinalStateSummaryLayout::
                        SuppressionCount
              : nullptr;
      launchContinuations(
          final_state.continuations,
          continuation_launch_count,
          final_state_continuation_count, 6U,
          "scatter final-state continuations launch");
      if (suppression_launch_count != 0) {
        auto const blocks = static_cast<unsigned int>(
            (suppression_launch_count +
             ThreadsPerBlock - 1) /
            ThreadsPerBlock);
        scatterBremsSuppressionsKernel
            <<<blocks, ThreadsPerBlock>>>(
                final_state.suppressions,
                suppression_launch_count,
                final_state_suppression_count,
                source_count, raw_next, child_categories,
                error_flag);
        checkCuda(
            cudaGetLastError(),
            "scatter bremsstrahlung suppressions launch");
      }
      if (final_state_launch_count != 0) {
        if (final_state.records == nullptr ||
            final_state.secondaries == nullptr) {
          throw std::invalid_argument(
              "lepton endpoint compaction requires final-state storage");
        }
        auto const blocks = static_cast<unsigned int>(
            (final_state_launch_count +
             ThreadsPerBlock - 1) /
            ThreadsPerBlock);
        scatterLeptonFinalStateChildrenKernel
            <<<blocks, ThreadsPerBlock>>>(
                final_state.records,
                final_state_launch_count,
                final_state_record_count,
                final_state.secondaries, source_count,
                raw_next, child_categories, raw_photons,
                error_flag);
        checkCuda(
            cudaGetLastError(),
            "scatter lepton final-state children launch");
      }
      checkCuda(cub::DeviceScan::ExclusiveScan(
                    scan_temporary, scan_bytes,
                    makeLeptonEndpointCountInput(
                        child_categories, observation_flags,
                        decay_flags,
                        source_count),
                    endpoint_offsets,
                    AddLeptonEndpointCategoryCounts{},
                    LeptonEndpointCategoryCounts{},
                    child_slots),
                "scan lepton endpoint categories");
      auto const child_blocks = static_cast<unsigned int>(
          (child_slots + ThreadsPerBlock - 1) /
          ThreadsPerBlock);
      compactLeptonEndpointsKernel
          <<<child_blocks, ThreadsPerBlock>>>(
              raw_next, raw_photons, child_categories,
              endpoint_offsets, child_slots, raw_observations,
              observation_flags, raw_decays, decay_flags,
              source_count,
              compact_next, compact_photons,
              compact_observations, compact_decays,
              error_flag,
              vertex.device_summary, final_state_summary,
              device_summary);
      checkCuda(
          cudaGetLastError(),
          "compact lepton endpoint queues launch");
      std::array<
          std::uint32_t,
          LeptonEndpointSummarySize>
          host_summary{};
      checkCuda(
          cudaMemcpy(
              host_summary.data(), device_summary,
              sizeof(host_summary), cudaMemcpyDeviceToHost),
          "download lepton endpoint summary");
      auto const next_count =
          static_cast<std::size_t>(host_summary[0]);
      auto const photon_count =
          static_cast<std::size_t>(host_summary[1]);
      auto const observation_count =
          static_cast<std::size_t>(host_summary[2]);
      auto const decay_count =
          static_cast<std::size_t>(host_summary[3]);
      auto const endpoint_error = host_summary[4];
      if (endpoint_error != 0) {
        throw std::runtime_error(
            "lepton endpoint queues contain duplicate or invalid source slots");
      }
      auto vertex_count = [&](std::size_t index) {
        return static_cast<std::size_t>(
            host_summary[
                LeptonEndpointVertexOffset + index]);
      };
      if (vertex.counts_deferred) {
        vertex.input_count =
            vertex_count(
                LeptonVertexSummaryLayout::InputCount);
        vertex.interaction_count =
            vertex_count(
                LeptonVertexSummaryLayout::
                    InteractionCount);
        vertex.continuation_count =
            vertex_count(
                LeptonVertexSummaryLayout::
                    ContinuationCount);
        vertex.fallback_count =
            vertex_count(
                LeptonVertexSummaryLayout::FallbackCount);
        vertex.counts_deferred = false;
      }
      auto const vertex_error =
          vertex_count(
              LeptonVertexSummaryLayout::Error);
      if (vertex_error != 0) {
        throw std::runtime_error(
            "lepton vertex classification did not conserve inputs");
      }
      auto final_count = [&](std::size_t index) {
        return static_cast<std::size_t>(
            host_summary[
                LeptonEndpointFinalStateOffset + index]);
      };
      if (deferred_final_state) {
        final_state.input_count =
            vertex.interaction_count;
        final_state.secondary_count =
            final_count(
                BremsFinalStateSummaryLayout::
                    SecondaryCount);
        final_state.gpu_interaction_count =
            final_count(
                BremsFinalStateSummaryLayout::GpuCount);
        final_state.fallback_count =
            final_count(
                BremsFinalStateSummaryLayout::
                    FallbackCount);
        final_state.continuation_count =
            final_count(
                BremsFinalStateSummaryLayout::
                    ContinuationCount);
        final_state.suppression_count =
            final_count(
                BremsFinalStateSummaryLayout::
                    SuppressionCount);
        final_state.brems_interaction_count =
            final_count(
                BremsFinalStateSummaryLayout::BremsCount);
        final_state.annihilation_interaction_count =
            final_count(
                BremsFinalStateSummaryLayout::
                    AnnihilationCount);
        final_state.ionization_interaction_count =
            final_count(
                BremsFinalStateSummaryLayout::
                    IonizationCount);
        final_state.electron_pair_interaction_count =
            final_count(
                BremsFinalStateSummaryLayout::
                    ElectronPairCount);
        final_state.brems_lpm_suppression_count =
            final_count(
                BremsFinalStateSummaryLayout::
                    BremsSuppressionCount);
        final_state.electron_pair_lpm_suppression_count =
            final_count(
                BremsFinalStateSummaryLayout::
                    ElectronPairSuppressionCount);
        final_state.electron_pair_rejection_trials =
            final_count(
                BremsFinalStateSummaryLayout::
                    ElectronPairRejectionTrials);
        final_state.electron_pair_zero_weight_samples =
            final_count(
                BremsFinalStateSummaryLayout::
                    ElectronPairZeroWeightSamples);
        final_state.electron_pair_rejection_fallbacks =
            final_count(
                BremsFinalStateSummaryLayout::
                    ElectronPairRejectionFallbacks);
        final_state.electron_pair_envelope_violations =
            final_count(
                BremsFinalStateSummaryLayout::
                    ElectronPairEnvelopeViolations);
        final_state.brems_lpm_trial_count =
            final_state.brems_interaction_count +
            final_state.brems_lpm_suppression_count;
        final_state.electron_pair_lpm_trial_count =
            final_state.electron_pair_interaction_count +
            final_state
                .electron_pair_lpm_suppression_count;
        final_state.counts_deferred = false;
      }
      auto const final_state_error =
          host_summary[
              LeptonEndpointFinalStateOffset +
              BremsFinalStateSummaryLayout::Error];
      if (final_state_error == 2U) {
        throw std::runtime_error(
            "lepton final-state classification lost or duplicated an interaction");
      }
      if (final_state_error == 3U) {
        throw std::overflow_error(
            "bremsstrahlung secondary history ID overflow");
      }
      if (final_state_error != 0) {
        throw std::runtime_error(
            "lepton final-state direction normalization failed");
      }
      return {
          source_count, next_count, photon_count,
          observation_count, decay_count, compact_next,
          compact_photons, compact_observations,
          compact_decays};
    }

    void appendLeptonDevicePipelineWorkspace(
        WorkspaceSize& required, std::size_t source_count) {
      appendInteractionSelectionWorkspace(
          required, source_count);
      appendLeptonTransportWorkspace(
          required, source_count);
      appendLeptonInteractionWorkspace(
          required, source_count);
      appendLeptonVertexSelectionWorkspace(
          required, source_count);
      appendBremsFinalStateWorkspace(
          required, source_count);
      appendLeptonEndpointWorkspace(
          required, source_count);
    }

    DeviceLeptonPipelineBatch
    launchLeptonDevicePipelineOnDevice(
        tables::FlatRateTableView device_table,
        BremsLpmSnapshot const& lpm_snapshot,
        BremsLpmPreparedSnapshot const& prepared_lpm,
        EmThinningConfig const& thinning,
        MoliereSnapshot const& moliere_snapshot,
        MoliereSnapshot const& muon_moliere_snapshot,
        MoliereInterpolationView const& moliere_interpolation,
        bool apply_moliere, bool muon_moliere_available,
        EnvironmentSnapshot const& environment,
        EmParticleState const* device_particles,
        std::size_t count, std::uint64_t random_seed,
        std::uint64_t shower_id,
        std::uint64_t first_secondary_history_id,
        DeviceWorkspace& workspace,
        LeptonPipelineStageEvents const* stage_events) {
      DeviceLeptonPipelineBatch pipeline{};
      pipeline.selection = launchInteractionSelectionOnDevice(
          device_table, device_particles, count, random_seed,
          shower_id, workspace, true);
      if (stage_events != nullptr) {
        checkCuda(
            cudaEventRecord(stage_events->selection_done),
            "record lepton selection stage");
      }
      pipeline.transport = launchLeptonTransportOnDevice(
          device_table, moliere_snapshot,
          muon_moliere_snapshot,
          moliere_interpolation, apply_moliere,
          muon_moliere_available,
          environment,
          pipeline.selection.interactions,
          count, random_seed, shower_id, workspace,
          &pipeline.selection, stage_events);
      if (stage_events != nullptr) {
        checkCuda(
            cudaEventRecord(stage_events->transport_done),
            "record lepton transport stage");
      }
      if (pipeline.transport.record_count == 0) {
        if (stage_events != nullptr) {
          checkCuda(
              cudaEventRecord(
                  stage_events->interaction_extraction_done),
              "record empty lepton interaction extraction stage");
          checkCuda(
              cudaEventRecord(
                  stage_events->vertex_selection_done),
              "record empty lepton vertex stage");
          checkCuda(
              cudaEventRecord(stage_events->final_state_done),
              "record empty lepton final-state stage");
          checkCuda(
              cudaEventRecord(
                  stage_events->endpoint_compaction_done),
              "record empty lepton endpoint stage");
        }
        return pipeline;
      }
      pipeline.at_interaction =
          extractLeptonInteractionsOnDevice(
              pipeline.transport.records,
              pipeline.transport.record_count, workspace,
              true);
      if (stage_events != nullptr) {
        checkCuda(
            cudaEventRecord(
                stage_events->interaction_extraction_done),
            "record lepton interaction extraction stage");
      }
      pipeline.vertex =
          launchLeptonVertexSelectionOnDevice(
              device_table,
              pipeline.at_interaction.interactions,
              pipeline.at_interaction.input_count,
              random_seed, shower_id, workspace,
              &pipeline.at_interaction, true);
      if (stage_events != nullptr) {
        checkCuda(
            cudaEventRecord(stage_events->vertex_selection_done),
            "record lepton vertex stage");
      }
      pipeline.final_state =
          launchBremsFinalStateOnDevice(
              lpm_snapshot, prepared_lpm, thinning,
              pipeline.vertex.interactions,
              pipeline.vertex.input_count,
              random_seed, shower_id,
              first_secondary_history_id, workspace,
              true, &pipeline.vertex, stage_events);
      if (stage_events != nullptr) {
        checkCuda(
            cudaEventRecord(stage_events->final_state_done),
            "record lepton final-state stage");
      }
      pipeline.endpoints = compactLeptonEndpointsOnDevice(
          pipeline.transport.records,
          pipeline.transport.record_count,
          pipeline.vertex,
          pipeline.final_state, count, workspace);
      if (stage_events != nullptr) {
        checkCuda(
            cudaEventRecord(
                stage_events->endpoint_compaction_done),
            "record lepton endpoint stage");
      }
      pipeline.at_interaction.interaction_count =
          pipeline.vertex.input_count;
      pipeline.at_interaction.count_deferred = false;
      return pipeline;
    }

  } // namespace detail

  LeptonDevicePipelineBatchResult
  runLeptonDevicePipelineForValidation(
      tables::FlatRateTableView device_table,
      BremsLpmSnapshot const& lpm_snapshot,
      EmThinningConfig const& thinning,
      MoliereSnapshot const& moliere_snapshot,
      MoliereSnapshot const& muon_moliere_snapshot,
      MoliereInterpolationView const& moliere_interpolation,
      bool apply_moliere, bool muon_moliere_available,
      EnvironmentSnapshot const& environment,
      std::vector<EmParticleState> const& particles,
      std::uint64_t random_seed, std::uint64_t shower_id,
      int device, std::uint64_t first_secondary_history_id,
      detail::DeviceWorkspace& workspace) {
    LeptonDevicePipelineBatchResult result{};
    result.input_particles = particles.size();
    result.multiple_scattering_enabled = apply_moliere;
    if (particles.empty()) {
      return result;
    }
    if (device < 0 || first_secondary_history_id == 0) {
      throw std::invalid_argument(
          "lepton device pipeline has invalid CUDA/history configuration");
    }
    if (!atmosphere_detail::validEnvironment(environment)) {
      throw std::invalid_argument(
          "lepton device pipeline requires a valid environment");
    }
    for (double component : environment.magnetic_field_T) {
      if (!std::isfinite(component)) {
        throw std::invalid_argument(
            "lepton device pipeline received a non-finite magnetic field");
      }
    }
    if (particles.size() >
        std::numeric_limits<std::uint32_t>::max() / 3) {
      throw std::length_error(
          "lepton device pipeline exceeds 32-bit offsets");
    }
    checkCuda(cudaSetDevice(device),
              "cudaSetDevice(lepton device pipeline)");

    auto const count = particles.size();
    detail::WorkspaceSize required;
    required.add<EmParticleState>(count);
    detail::appendLeptonDevicePipelineWorkspace(
        required, count);
    workspace.prepare(required.bytes());
    auto* device_particles =
        workspace.acquire<EmParticleState>(count);
    checkCuda(cudaMemcpy(
                  device_particles, particles.data(),
                  count * sizeof(EmParticleState),
                  cudaMemcpyHostToDevice),
              "upload lepton device pipeline input");

    auto const pipeline =
        detail::launchLeptonDevicePipelineOnDevice(
            device_table, lpm_snapshot,
            prepareBremsLpmSnapshotForCuda(
                lpm_snapshot, device),
            thinning,
            moliere_snapshot, muon_moliere_snapshot,
            moliere_interpolation, apply_moliere,
            muon_moliere_available, environment,
            device_particles, count, random_seed, shower_id,
            first_secondary_history_id, workspace);
    auto const& selection = pipeline.selection;
    result.selection_fallback_events.resize(
        selection.fallback_count);
    if (!result.selection_fallback_events.empty()) {
      checkCuda(cudaMemcpy(
                    result.selection_fallback_events.data(),
                    selection.fallbacks,
                    result.selection_fallback_events.size() *
                        sizeof(ProposalFallbackEvent),
                    cudaMemcpyDeviceToHost),
                "download lepton pipeline selection fallbacks");
    }

    auto const& transport = pipeline.transport;
    result.transport_records.resize(transport.record_count);
    result.transport_fallback_events.resize(
        transport.fallback_count);
    if (!result.transport_records.empty()) {
      checkCuda(cudaMemcpy(
                    result.transport_records.data(),
                    transport.records,
                    result.transport_records.size() *
                        sizeof(LeptonTransportRecord),
                    cudaMemcpyDeviceToHost),
                "download lepton pipeline transport records");
    }
    if (!result.transport_fallback_events.empty()) {
      checkCuda(cudaMemcpy(
                    result.transport_fallback_events.data(),
                    transport.fallbacks,
                    result.transport_fallback_events.size() *
                        sizeof(ProposalFallbackEvent),
                    cudaMemcpyDeviceToHost),
                "download lepton pipeline transport fallbacks");
    }

    result.interaction_candidates =
        pipeline.at_interaction.interaction_count;
    auto const& vertex = pipeline.vertex;
    result.vertex_interactions = vertex.interaction_count;
    result.vertex_continuations.resize(
        vertex.continuation_count);
    result.vertex_fallback_events.resize(
        vertex.fallback_count);
    if (!result.vertex_continuations.empty()) {
      checkCuda(cudaMemcpy(
                    result.vertex_continuations.data(),
                    vertex.continuations,
                    result.vertex_continuations.size() *
                        sizeof(EmInteractionRecord),
                    cudaMemcpyDeviceToHost),
                "download lepton pipeline vertex continuations");
    }
    if (!result.vertex_fallback_events.empty()) {
      checkCuda(cudaMemcpy(
                    result.vertex_fallback_events.data(),
                    vertex.fallbacks,
                    result.vertex_fallback_events.size() *
                        sizeof(ProposalFallbackEvent),
                    cudaMemcpyDeviceToHost),
                "download lepton pipeline vertex fallbacks");
    }

    auto const& final_state = pipeline.final_state;
    auto& host = result.final_states;
    host.input_interactions = vertex.interaction_count;
    host.gpu_interactions =
        final_state.gpu_interaction_count;
    host.brems_interactions =
        final_state.brems_interaction_count;
    host.annihilation_interactions =
        final_state.annihilation_interaction_count;
    host.ionization_interactions =
        final_state.ionization_interaction_count;
    host.electron_pair_interactions =
        final_state.electron_pair_interaction_count;
    host.brems_lpm_trials =
        final_state.brems_lpm_trial_count;
    host.brems_lpm_suppressions =
        final_state.brems_lpm_suppression_count;
    host.electron_pair_lpm_trials =
        final_state.electron_pair_lpm_trial_count;
    host.electron_pair_lpm_suppressions =
        final_state.electron_pair_lpm_suppression_count;
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
                        sizeof(BremsFinalStateRecord),
                    cudaMemcpyDeviceToHost),
                "download lepton pipeline final-state records");
    }
    if (!host.secondaries.empty()) {
      checkCuda(cudaMemcpy(
                    host.secondaries.data(),
                    final_state.secondaries,
                    host.secondaries.size() *
                        sizeof(EmParticleState),
                    cudaMemcpyDeviceToHost),
                "download lepton pipeline secondaries");
    }
    if (!host.fallback_events.empty()) {
      checkCuda(cudaMemcpy(
                    host.fallback_events.data(),
                    final_state.fallbacks,
                    host.fallback_events.size() *
                        sizeof(ProposalFallbackEvent),
                    cudaMemcpyDeviceToHost),
                "download lepton pipeline final-state fallbacks");
    }
    if (!host.continuations.empty()) {
      checkCuda(cudaMemcpy(
                    host.continuations.data(),
                    final_state.continuations,
                    host.continuations.size() *
                        sizeof(EmInteractionRecord),
                    cudaMemcpyDeviceToHost),
                "download lepton pipeline final-state continuations");
    }
    if (!host.lpm_suppressed.empty()) {
      checkCuda(cudaMemcpy(
                    host.lpm_suppressed.data(),
                    final_state.suppressions,
                    host.lpm_suppressed.size() *
                        sizeof(BremsLpmSuppressionRecord),
                    cudaMemcpyDeviceToHost),
                "download lepton pipeline LPM suppressions");
    }
    auto const& endpoints = pipeline.endpoints;
    result.next_leptons.resize(
        endpoints.next_lepton_count);
    result.generated_photons.resize(
        endpoints.generated_photon_count);
    result.observations.resize(
        endpoints.observation_count);
    result.decay_candidates.resize(
        endpoints.decay_candidate_count);
    if (!result.next_leptons.empty()) {
      checkCuda(cudaMemcpy(
                    result.next_leptons.data(),
                    endpoints.next_leptons,
                    result.next_leptons.size() *
                        sizeof(EmParticleState),
                    cudaMemcpyDeviceToHost),
                "download lepton pipeline next queue");
    }
    if (!result.generated_photons.empty()) {
      checkCuda(cudaMemcpy(
                    result.generated_photons.data(),
                    endpoints.generated_photons,
                    result.generated_photons.size() *
                        sizeof(EmParticleState),
                    cudaMemcpyDeviceToHost),
                "download lepton pipeline generated photons");
    }
    if (!result.observations.empty()) {
      checkCuda(cudaMemcpy(
                    result.observations.data(),
                    endpoints.observations,
                    result.observations.size() *
                        sizeof(ObservationRecord),
                    cudaMemcpyDeviceToHost),
                "download lepton pipeline observations");
    }
    if (!result.decay_candidates.empty()) {
      checkCuda(cudaMemcpy(
                    result.decay_candidates.data(),
                    endpoints.decay_candidates,
                    result.decay_candidates.size() *
                        sizeof(EmParticleState),
                    cudaMemcpyDeviceToHost),
                "download lepton pipeline decay candidates");
    }
    return result;
  }

} // namespace corsika::gpu::em

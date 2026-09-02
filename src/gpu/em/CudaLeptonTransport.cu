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

#include <corsika/accelerator/em/detail/LeptonTransportStep.hpp>
#include <corsika/accelerator/em/detail/MoliereStep.hpp>
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

    void checkCuda(cudaError_t status, char const* operation) {
      if (status == cudaSuccess) {
        return;
      }
      std::ostringstream message;
      message << operation << " failed: "
              << cudaGetErrorString(status);
      throw std::runtime_error(message.str());
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
      auto const state = accelerator::em::detail::transportLepton(
          table, apply_moliere, environment, interaction,
          raw_records[index], raw_fallbacks[index]);
      fallback_flags[index] = state;
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
      auto const state = static_cast<std::uint32_t>(fallback_flags[index]);
      fallback_flags[index] =
          accelerator::em::detail::applyMoliereScatteringStage<
              ComponentCapacity>(
              snapshot, muon_snapshot, interpolation,
              muon_snapshot_available, random_seed, shower_id,
              raw_records[index], raw_fallbacks[index], state);
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

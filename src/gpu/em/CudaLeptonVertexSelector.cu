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
#include <type_traits>

#include <corsika/gpu/em/CudaInteractionSelector.hpp>
#include <corsika/gpu/em/CudaLeptonVertexSelector.hpp>
#include <corsika/gpu/em/Philox.hpp>
#include <corsika/gpu/em/ProposalFallback.hpp>
#include <corsika/gpu/em/detail/DeviceBatchStages.hpp>

namespace corsika::gpu::em {

  namespace {

    constexpr unsigned int ThreadsPerBlock = 256;

    struct VertexCategoryCounts {
      std::uint32_t interactions{};
      std::uint32_t continuations{};
      std::uint32_t fallbacks{};
    };

    static_assert(
        sizeof(VertexCategoryCounts) ==
        3 * sizeof(std::uint32_t));
    static_assert(
        std::is_trivially_copyable_v<VertexCategoryCounts>);

    struct AddVertexCategoryCounts {
      __host__ __device__ __forceinline__
      VertexCategoryCounts operator()(
          VertexCategoryCounts const& left,
          VertexCategoryCounts const& right) const {
        return {
            left.interactions + right.interactions,
            left.continuations + right.continuations,
            left.fallbacks + right.fallbacks};
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

    __device__ ProposalFallbackEvent vertexFallback(
        EmInteractionRecord const& candidate,
        ProposalFallbackReason reason) {
      auto event = makeProcessFallbackEvent(candidate, reason);
      event.input_index = candidate.input_index;
      return event;
    }

    __device__ ProposalFallbackEvent vertexTableFallback(
        EmInteractionRecord const& candidate,
        tables::TableQuery const& query,
        tables::TableQueryResult const& result,
        std::uint64_t draw_id, double loss_quantile = 0.) {
      auto event = makeTableFallbackEvent(
          candidate.particle, query, result, draw_id,
          candidate.input_index);
      event.selection_uniform = candidate.process_uniform;
      event.loss_quantile = loss_quantile;
      return event;
    }

    __global__ void selectLeptonVerticesKernel(
        tables::FlatRateTableView table,
        EmInteractionRecord const* candidates, std::size_t count,
        std::uint32_t const* device_input_count,
        std::uint64_t random_seed, std::uint64_t shower_id,
        EmInteractionRecord* raw_records,
        ProposalFallbackEvent* raw_fallbacks,
        VertexCategoryCounts* categories) {
      auto const index =
          static_cast<std::size_t>(blockIdx.x) * blockDim.x +
          threadIdx.x;
      if (index >= count) {
        return;
      }
      categories[index] = {};
      if (device_input_count != nullptr &&
          index >= *device_input_count) {
        return;
      }
      auto record = candidates[index];
      auto const& particle = record.particle;
      if (!isChargedLeptonPid(particle.pid)) {
        raw_fallbacks[index] = vertexFallback(
            record, ProposalFallbackReason::UnsupportedParticle);
        categories[index].fallbacks = 1;
        return;
      }
      if (record.status !=
              EmInteractionStatus::RequiresReselection ||
          !::isfinite(record.total_rate_cm2_per_g) ||
          !(record.total_rate_cm2_per_g > 0.) ||
          !::isfinite(record.process_uniform) ||
          !(record.process_uniform > 0.) ||
          !(record.process_uniform < 1.)) {
        raw_fallbacks[index] = vertexFallback(
            record, ProposalFallbackReason::InvalidFinalState);
        categories[index].fallbacks = 1;
        return;
      }

      auto const energy_MeV = particle.energy_GeV * 1000.;
      tables::TableQuery const total_query{
          tables::TableQueryKind::TotalRate, particle.pid,
          0, 0, energy_MeV, 0.};
      if (!tables::detail::validView(table)) {
        auto const invalid = tables::TableQueryResult{
            tables::TableLookupStatus::InvalidTableView,
            0, 0.};
        raw_fallbacks[index] = vertexTableFallback(
            record, total_query, invalid,
            record.process_draw_id);
        categories[index].fallbacks = 1;
        return;
      }
      auto const particle_index =
          tables::detail::findParticle(table, particle.pid);
      if (particle_index == table.particle_count) {
        auto const missing = tables::TableQueryResult{
            tables::TableLookupStatus::ParticleNotFound, 0, 0.};
        raw_fallbacks[index] = vertexTableFallback(
            record, total_query, missing, record.process_draw_id);
        categories[index].fallbacks = 1;
        return;
      }
      auto const bracket =
          tables::detail::rateInterpolationBracket(
              table, particle_index, energy_MeV);
      if (bracket.status !=
          tables::TableLookupStatus::Success) {
        auto const failure = tables::TableQueryResult{
            bracket.status, 0, 0.};
        raw_fallbacks[index] = vertexTableFallback(
            record, total_query, failure,
            record.process_draw_id);
        categories[index].fallbacks = 1;
        return;
      }
      auto const column_begin =
          table.particle_column_offsets[particle_index];
      auto const column_end =
          column_begin +
          table.particle_column_counts[particle_index];
      auto const threshold =
          record.process_uniform *
          record.total_rate_cm2_per_g;
      double cumulative = 0.;
      auto selected_column = table.column_count;
      auto last_positive_column = table.column_count;
      for (auto column = column_begin;
           column < column_end; ++column) {
        auto const rate =
            tables::detail::interpolateRateColumnAtBracket(
                table, particle_index, column, bracket);
        if (rate.status != tables::TableLookupStatus::Success) {
          raw_fallbacks[index] = vertexTableFallback(
              record, total_query, rate,
              record.process_draw_id);
          categories[index].fallbacks = 1;
          return;
        }
        if (rate.value > 0.) {
          last_positive_column = column;
        }
        cumulative += rate.value;
        if (selected_column == table.column_count &&
            threshold < cumulative) {
          selected_column = column;
        }
      }
      if (!::isfinite(cumulative)) {
        auto const invalid = tables::TableQueryResult{
            tables::TableLookupStatus::InvalidTableView,
            0, 0.};
        raw_fallbacks[index] = vertexTableFallback(
            record, total_query, invalid,
            record.process_draw_id);
        categories[index].fallbacks = 1;
        return;
      }
      record.vertex_total_rate_cm2_per_g = cumulative;
      if (!(cumulative > 0.) || threshold >= cumulative) {
        if (record.particle.step_id ==
            0xffffffffffffffffULL) {
          raw_fallbacks[index] = vertexFallback(
              record, ProposalFallbackReason::InvalidFinalState);
          categories[index].fallbacks = 1;
          return;
        }
        record.status =
            EmInteractionStatus::NoDiscreteInteraction;
        record.process_id = 0;
        record.component_hash = 0;
        record.energy_fraction = 0.;
        record.loss_quantile = 0.;
        record.particle.step_id++;
        raw_records[index] = record;
        categories[index].continuations = 1;
        return;
      }
      if (selected_column == table.column_count &&
          threshold < cumulative) {
        selected_column = last_positive_column;
      }
      if (selected_column == table.column_count) {
        raw_fallbacks[index] = vertexFallback(
            record, ProposalFallbackReason::ZeroTotalRate);
        categories[index].fallbacks = 1;
        return;
      }

      record.process_id =
          table.column_process_ids[selected_column];
      record.component_hash =
          table.column_component_hashes[selected_column];
      RandomNumberKey const loss_key{
          random_seed, shower_id, particle.history_id,
          particle.step_id,
          static_cast<std::uint32_t>(record.process_id),
          InteractionLossDrawId};
      record.loss_quantile = uniformOpen01(loss_key);
      record.loss_draw_id = InteractionLossDrawId;
      tables::TableQuery const loss_query{
          tables::TableQueryKind::LossFraction, particle.pid,
          record.process_id, record.component_hash,
          energy_MeV, record.loss_quantile};
      auto const loss =
          tables::executeTableQuery(table, loss_query);
      if (loss.status != tables::TableLookupStatus::Success) {
        raw_fallbacks[index] = vertexTableFallback(
            record, loss_query, loss, InteractionLossDrawId,
            record.loss_quantile);
        categories[index].fallbacks = 1;
        return;
      }
      record.status = EmInteractionStatus::Selected;
      record.energy_fraction = loss.value;
      raw_records[index] = record;
      categories[index].interactions = 1;
    }

    __global__ void compactLeptonVerticesKernel(
        EmInteractionRecord const* raw_records,
        ProposalFallbackEvent const* raw_fallbacks,
        VertexCategoryCounts const* categories,
        VertexCategoryCounts const* offsets,
        std::size_t count,
        std::uint32_t const* device_input_count,
        EmInteractionRecord* compact_interactions,
        EmInteractionRecord* compact_continuations,
        ProposalFallbackEvent* compact_fallbacks,
        std::uint32_t* summary) {
      auto const index =
          static_cast<std::size_t>(blockIdx.x) * blockDim.x +
          threadIdx.x;
      if (index >= count) {
        return;
      }
      auto const category = categories[index];
      auto const offset = offsets[index];
      if (category.interactions != 0) {
        compact_interactions[offset.interactions] =
            raw_records[index];
      } else if (category.continuations != 0) {
        compact_continuations[offset.continuations] =
            raw_records[index];
      } else if (category.fallbacks != 0) {
        compact_fallbacks[offset.fallbacks] =
            raw_fallbacks[index];
      }
      if (index == 0) {
        auto const last = count - 1;
        auto const totals = VertexCategoryCounts{
            offsets[last].interactions +
                categories[last].interactions,
            offsets[last].continuations +
                categories[last].continuations,
            offsets[last].fallbacks +
                categories[last].fallbacks};
        auto const input_count =
            device_input_count == nullptr
                ? static_cast<std::uint32_t>(count)
                : *device_input_count;
        summary[
            detail::LeptonVertexSummaryLayout::InputCount] =
            input_count;
        summary[
            detail::LeptonVertexSummaryLayout::
                InteractionCount] = totals.interactions;
        summary[
            detail::LeptonVertexSummaryLayout::
                ContinuationCount] = totals.continuations;
        summary[
            detail::LeptonVertexSummaryLayout::
                FallbackCount] = totals.fallbacks;
        summary[
            detail::LeptonVertexSummaryLayout::Error] =
            totals.interactions + totals.continuations +
                        totals.fallbacks ==
                    input_count
                ? 0U
                : 1U;
      }
    }

  } // namespace

  namespace detail {

    void appendLeptonVertexSelectionWorkspace(
        WorkspaceSize& required, std::size_t count) {
      std::size_t scan_bytes = 0;
      checkCuda(cub::DeviceScan::ExclusiveScan(
                    nullptr, scan_bytes,
                    static_cast<VertexCategoryCounts*>(nullptr),
                    static_cast<VertexCategoryCounts*>(nullptr),
                    AddVertexCategoryCounts{},
                    VertexCategoryCounts{}, count),
                "query lepton vertex scan storage");
      required.add<EmInteractionRecord>(count);
      required.add<ProposalFallbackEvent>(count);
      required.add<VertexCategoryCounts>(count);
      required.add<VertexCategoryCounts>(count);
      required.add<EmInteractionRecord>(count);
      required.add<EmInteractionRecord>(count);
      required.add<ProposalFallbackEvent>(count);
      required.add<std::uint32_t>(
          LeptonVertexSummaryLayout::Size);
      required.addBytes(scan_bytes);
    }

    DeviceLeptonVertexSelectionBatch
    launchLeptonVertexSelectionOnDevice(
        tables::FlatRateTableView device_table,
        EmInteractionRecord const* device_candidates,
        std::size_t count, std::uint64_t random_seed,
        std::uint64_t shower_id, DeviceWorkspace& workspace,
        DeviceTransportInteractionBatch*
            deferred_candidates,
        bool defer_count_download) {
      if (count == 0 || device_candidates == nullptr) {
        throw std::invalid_argument(
            "device lepton vertex selector requires candidates");
      }
      std::uint32_t const* device_input_count = nullptr;
      if (deferred_candidates != nullptr) {
        if (!deferred_candidates->count_deferred ||
            deferred_candidates->device_interaction_count ==
                nullptr ||
            deferred_candidates->interactions !=
                device_candidates ||
            deferred_candidates->input_count != count) {
          throw std::invalid_argument(
              "deferred lepton transport interactions are inconsistent");
        }
        device_input_count =
            deferred_candidates->device_interaction_count;
      }
      std::size_t scan_bytes = 0;
      checkCuda(cub::DeviceScan::ExclusiveScan(
                    nullptr, scan_bytes,
                    static_cast<VertexCategoryCounts*>(nullptr),
                    static_cast<VertexCategoryCounts*>(nullptr),
                    AddVertexCategoryCounts{},
                    VertexCategoryCounts{}, count),
                "query lepton vertex scan storage");
      auto* raw_records =
          workspace.acquire<EmInteractionRecord>(count);
      auto* raw_fallbacks =
          workspace.acquire<ProposalFallbackEvent>(count);
      auto* categories =
          workspace.acquire<VertexCategoryCounts>(count);
      auto* offsets =
          workspace.acquire<VertexCategoryCounts>(count);
      auto* compact_interactions =
          workspace.acquire<EmInteractionRecord>(count);
      auto* compact_continuations =
          workspace.acquire<EmInteractionRecord>(count);
      auto* compact_fallbacks =
          workspace.acquire<ProposalFallbackEvent>(count);
      auto* device_summary =
          workspace.acquire<std::uint32_t>(
              LeptonVertexSummaryLayout::Size);
      auto* scan_temporary = workspace.acquireBytes(scan_bytes);

      auto const block_count =
          (count + ThreadsPerBlock - 1) / ThreadsPerBlock;
      if (block_count >
          std::numeric_limits<unsigned int>::max()) {
        throw std::length_error(
            "lepton vertex selection launch is too large");
      }
      auto const blocks = static_cast<unsigned int>(block_count);
      selectLeptonVerticesKernel<<<blocks, ThreadsPerBlock>>>(
          device_table, device_candidates, count,
          device_input_count, random_seed, shower_id,
          raw_records, raw_fallbacks,
          categories);
      checkCuda(cudaGetLastError(),
                "lepton vertex selection kernel launch");
      checkCuda(cub::DeviceScan::ExclusiveScan(
                    scan_temporary, scan_bytes, categories,
                    offsets, AddVertexCategoryCounts{},
                    VertexCategoryCounts{}, count),
                "scan lepton vertex category counts");
      compactLeptonVerticesKernel<<<blocks, ThreadsPerBlock>>>(
          raw_records, raw_fallbacks, categories, offsets,
          count, device_input_count, compact_interactions,
          compact_continuations, compact_fallbacks,
          device_summary);
      checkCuda(cudaGetLastError(),
                "compact lepton vertex selection launch");
      std::size_t input_count = count;
      std::size_t interaction_count = 0;
      std::size_t continuation_count = 0;
      std::size_t fallback_count = 0;
      if (!defer_count_download) {
        std::array<
            std::uint32_t,
            LeptonVertexSummaryLayout::Size>
            host_summary{};
        checkCuda(
            cudaMemcpy(
                host_summary.data(), device_summary,
                sizeof(host_summary),
                cudaMemcpyDeviceToHost),
            "download lepton vertex summary");
        input_count =
            host_summary[
                LeptonVertexSummaryLayout::InputCount];
        interaction_count =
            host_summary[
                LeptonVertexSummaryLayout::
                    InteractionCount];
        continuation_count =
            host_summary[
                LeptonVertexSummaryLayout::
                    ContinuationCount];
        fallback_count =
            host_summary[
                LeptonVertexSummaryLayout::FallbackCount];
        if (host_summary[
                LeptonVertexSummaryLayout::Error] != 0) {
          throw std::runtime_error(
              "lepton vertex classification did not conserve inputs");
        }
      }
      return {input_count, interaction_count,
              continuation_count, fallback_count,
              compact_interactions,
              compact_continuations, compact_fallbacks,
              device_summary, defer_count_download};
    }

  } // namespace detail

  LeptonVertexSelectionBatchResult
  reselectLeptonInteractionsAtVertexForValidation(
      tables::FlatRateTableView device_table,
      std::vector<EmInteractionRecord> const& candidates,
      std::uint64_t random_seed, std::uint64_t shower_id, int device,
      detail::DeviceWorkspace& workspace) {
    LeptonVertexSelectionBatchResult result{};
    result.input_candidates = candidates.size();
    if (candidates.empty()) {
      return result;
    }
    if (device < 0) {
      throw std::invalid_argument(
          "lepton vertex selector CUDA device must be non-negative");
    }
    if (candidates.size() >
        std::numeric_limits<std::uint32_t>::max()) {
      throw std::length_error(
          "lepton vertex batch exceeds 32-bit scan offsets");
    }
    checkCuda(cudaSetDevice(device),
              "cudaSetDevice(lepton vertex selector)");
    auto const count = candidates.size();
    detail::WorkspaceSize required;
    required.add<EmInteractionRecord>(count);
    detail::appendLeptonVertexSelectionWorkspace(
        required, count);
    workspace.prepare(required.bytes());

    auto* device_candidates =
        workspace.acquire<EmInteractionRecord>(count);
    checkCuda(cudaMemcpy(
                  device_candidates, candidates.data(),
                  count * sizeof(EmInteractionRecord),
                  cudaMemcpyHostToDevice),
              "upload lepton vertex candidates");
    auto const batch =
        detail::launchLeptonVertexSelectionOnDevice(
            device_table, device_candidates, count, random_seed,
            shower_id, workspace);

    result.interactions.resize(batch.interaction_count);
    result.continuations.resize(batch.continuation_count);
    result.fallback_events.resize(batch.fallback_count);
    if (batch.interaction_count != 0) {
      checkCuda(cudaMemcpy(
                    result.interactions.data(), batch.interactions,
                    batch.interaction_count *
                        sizeof(EmInteractionRecord),
                    cudaMemcpyDeviceToHost),
                "download lepton vertex interactions");
    }
    if (batch.continuation_count != 0) {
      checkCuda(cudaMemcpy(
                    result.continuations.data(), batch.continuations,
                    batch.continuation_count *
                        sizeof(EmInteractionRecord),
                    cudaMemcpyDeviceToHost),
                "download lepton vertex continuations");
    }
    if (batch.fallback_count != 0) {
      checkCuda(cudaMemcpy(
                    result.fallback_events.data(), batch.fallbacks,
                    batch.fallback_count *
                        sizeof(ProposalFallbackEvent),
                    cudaMemcpyDeviceToHost),
                "download lepton vertex fallbacks");
    }
    return result;
  }

} // namespace corsika::gpu::em

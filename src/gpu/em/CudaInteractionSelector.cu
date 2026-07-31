/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include <cuda_runtime.h>
#include <math_constants.h>

#include <cub/device/device_scan.cuh>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <vector>

#include <corsika/gpu/em/CudaInteractionSelector.hpp>
#include <corsika/gpu/em/CudaLeptonTransport.hpp>
#include <corsika/gpu/em/Philox.hpp>
#include <corsika/gpu/em/ProposalFallback.hpp>
#include <corsika/gpu/em/detail/DeviceBatchStages.hpp>

namespace corsika::gpu::em {

  namespace {

    constexpr unsigned int ThreadsPerBlock = 256;
    constexpr unsigned int SelectionThreadsPerBlock = 64;

    void checkCuda(cudaError_t status, char const* operation) {
      if (status == cudaSuccess) {
        return;
      }
      std::ostringstream message;
      message << operation << " failed: "
              << cudaGetErrorString(status);
      throw std::runtime_error(message.str());
    }

    __device__ double deviceLog(double value) {
      return ::log(value);
    }

    __device__ ProposalFallbackEvent tableFallback(
        EmParticleState const& particle, std::uint64_t input_index,
        tables::TableQuery const& query,
        tables::TableQueryResult const& result,
        std::uint64_t random_draw_id, double loss_quantile = 0.) {
      auto event = makeTableFallbackEvent(
          particle, query, result, random_draw_id, input_index);
      event.loss_quantile = loss_quantile;
      return event;
    }

    __global__ void selectDiscreteInteractionsKernel(
        tables::FlatRateTableView table,
        EmParticleState const* particles, std::size_t count,
        std::uint64_t random_seed, std::uint64_t shower_id,
        EmInteractionRecord* raw_interactions,
        ProposalFallbackEvent* raw_fallbacks,
        std::uint32_t* fallback_flags) {
      auto const index =
          static_cast<std::size_t>(blockIdx.x) * blockDim.x +
          threadIdx.x;
      if (index >= count) {
        return;
      }

      auto const particle = particles[index];
      auto const energy_MeV = particle.energy_GeV * 1000.;
      EmInteractionRecord record{};
      record.particle = particle;
      record.input_index = index;
      record.distance_draw_id = InteractionDistanceDrawId;
      record.process_draw_id = InteractionColumnDrawId;
      record.loss_draw_id = InteractionLossDrawId;

      auto const charged_lepton =
          isChargedLeptonPid(particle.pid);
      auto const photon =
          particle.pid ==
          static_cast<std::int32_t>(EmPid::Photon);
      if (photon && energy_MeV < table.energy_cut_MeV) {
        record.status = EmInteractionStatus::ParticleCut;
        record.interaction_grammage_g_per_cm2 = CUDART_INF;
        raw_interactions[index] = record;
        fallback_flags[index] = 0;
        return;
      }
      if (charged_lepton) {
        // The continuous table deliberately extends a little below the
        // configured kinetic-energy cut. A secondary can be born anywhere
        // in that narrow safety interval or below the final anchor. Such a
        // particle must reach ParticleCut without querying a stochastic rate
        // outside its table domain.
        auto const mass =
            tables::queryContinuousMass(table, particle.pid);
        if (mass.status == tables::TableLookupStatus::Success) {
          record.particle_mass_GeV = mass.value / 1000.;
          if (isMuonPid(particle.pid) &&
              particle.energy_GeV >
                  record.particle_mass_GeV) {
            RandomNumberKey const decay_key{
                random_seed, shower_id,
                particle.history_id, particle.step_id,
                MuonDecayRandomProcessId, MuonDecayDrawId};
            record.decay_uniform =
                uniformOpen01(decay_key);
            record.decay_draw_id = MuonDecayDrawId;
            auto const momentum_GeV = ::sqrt(
                (particle.energy_GeV -
                 record.particle_mass_GeV) *
                (particle.energy_GeV +
                 record.particle_mass_GeV));
            record.decay_distance_m =
                -deviceLog(record.decay_uniform) *
                MuonDecaySpeedOfLightMPerS *
                MuonMeanLifetimeS *
                momentum_GeV /
                record.particle_mass_GeV;
          }
        }
        auto const minimum_energy =
            tables::queryContinuousMinimumEnergy(
                table, particle.pid);
        auto const below_cut =
            mass.status == tables::TableLookupStatus::Success &&
            energy_MeV - mass.value < table.energy_cut_MeV;
        if (below_cut ||
            (minimum_energy.status ==
                tables::TableLookupStatus::Success &&
             energy_MeV <= minimum_energy.value)) {
          record.status = EmInteractionStatus::ParticleCut;
          record.interaction_grammage_g_per_cm2 = CUDART_INF;
          raw_interactions[index] = record;
          fallback_flags[index] = 0;
          return;
        }
      }

      tables::TableQuery const total_query{
          tables::TableQueryKind::TotalRate, particle.pid,
          0, 0, energy_MeV, 0.};
      auto const total =
          tables::executeTableQuery(table, total_query);
      if (total.status != tables::TableLookupStatus::Success) {
        raw_fallbacks[index] = tableFallback(
            particle, index, total_query, total, 0);
        fallback_flags[index] = 1;
        return;
      }

      record.total_rate_cm2_per_g = total.value;
      record.vertex_total_rate_cm2_per_g = total.value;

      if (!(total.value > 0.)) {
        record.status = EmInteractionStatus::NoDiscreteInteraction;
        record.interaction_grammage_g_per_cm2 =
            CUDART_INF;
        raw_interactions[index] = record;
        fallback_flags[index] = 0;
        return;
      }

      RandomNumberKey const distance_key{
          random_seed, shower_id, particle.history_id,
          particle.step_id, InteractionDistanceRandomProcessId,
          InteractionDistanceDrawId};
      record.distance_uniform = uniformOpen01(distance_key);
      record.interaction_grammage_g_per_cm2 =
          -deviceLog(record.distance_uniform) / total.value;

      RandomNumberKey const column_key{
          random_seed, shower_id, particle.history_id,
          particle.step_id, InteractionColumnRandomProcessId,
          InteractionColumnDrawId};
      record.process_uniform = uniformOpen01(column_key);

      // CORSIKA applies continuous loss before doInteraction(). For a charged
      // particle, selecting process/component/v here would therefore use the
      // wrong (pre-transport) energy. Preserve only the sampled distance and
      // cross-section threshold; a vertex kernel completes the selection.
      if (charged_lepton) {
        record.status = EmInteractionStatus::DistanceSampled;
        raw_interactions[index] = record;
        fallback_flags[index] = 0;
        return;
      }

      auto const particle_index =
          tables::detail::findParticle(table, particle.pid);
      if (particle_index == table.particle_count) {
        auto const result = tables::TableQueryResult{
            tables::TableLookupStatus::ParticleNotFound, 0, 0.};
        raw_fallbacks[index] = tableFallback(
            particle, index, total_query, result, 0);
        fallback_flags[index] = 1;
        return;
      }
      auto const column_begin =
          table.particle_column_offsets[particle_index];
      auto const column_end =
          column_begin +
          table.particle_column_counts[particle_index];
      auto const threshold = record.process_uniform * total.value;
      double cumulative = 0.;
      auto selected_column = table.column_count;
      auto last_positive_column = table.column_count;
      for (auto column = column_begin; column < column_end; ++column) {
        auto const rate = tables::detail::interpolateRateColumn(
            table, particle_index, column, energy_MeV);
        if (rate.status != tables::TableLookupStatus::Success) {
          raw_fallbacks[index] = tableFallback(
              particle, index, total_query, rate, 0);
          fallback_flags[index] = 1;
          return;
        }
        if (rate.value > 0.) {
          last_positive_column = column;
        }
        cumulative += rate.value;
        if (threshold < cumulative) {
          selected_column = column;
          break;
        }
      }
      if (selected_column == table.column_count) {
        selected_column = last_positive_column;
      }
      if (selected_column == table.column_count) {
        auto event = ProposalFallbackEvent{};
        event.particle = particle;
        event.input_index = index;
        event.reason = ProposalFallbackReason::ZeroTotalRate;
        raw_fallbacks[index] = event;
        fallback_flags[index] = 1;
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
      tables::TableQuery const loss_query{
          tables::TableQueryKind::LossFraction, particle.pid,
          record.process_id, record.component_hash, energy_MeV,
          record.loss_quantile};
      auto const loss =
          tables::executeTableQuery(table, loss_query);
      if (loss.status != tables::TableLookupStatus::Success) {
        auto event = tableFallback(
            particle, index, loss_query, loss,
            InteractionLossDrawId, record.loss_quantile);
        event.selection_uniform = record.process_uniform;
        raw_fallbacks[index] = event;
        fallback_flags[index] = 1;
        return;
      }

      record.status = EmInteractionStatus::Selected;
      record.energy_fraction = loss.value;
      raw_interactions[index] = record;
      fallback_flags[index] = 0;
    }

    __global__ void compactInteractionSelectionsKernel(
        EmInteractionRecord const* raw_interactions,
        ProposalFallbackEvent const* raw_fallbacks,
        std::uint32_t const* fallback_flags,
        std::uint32_t const* fallback_offsets,
        std::size_t count,
        EmInteractionRecord* compact_interactions,
        ProposalFallbackEvent* compact_fallbacks,
        std::uint32_t const* fallback_count) {
      auto const index =
          static_cast<std::size_t>(blockIdx.x) * blockDim.x +
          threadIdx.x;
      if (index >= count || *fallback_count == 0) {
        return;
      }
      auto const fallbacks_before = fallback_offsets[index];
      if (fallback_flags[index] != 0) {
        compact_fallbacks[fallbacks_before] =
            raw_fallbacks[index];
      } else {
        compact_interactions[index - fallbacks_before] =
            raw_interactions[index];
      }
    }

    __global__ void finalizeSelectionFallbackCountKernel(
        std::uint32_t const* fallback_flags,
        std::uint32_t const* fallback_offsets,
        std::size_t count,
        std::uint32_t* fallback_count) {
      if (blockIdx.x == 0 && threadIdx.x == 0) {
        auto const last = count - 1;
        *fallback_count =
            fallback_offsets[last] + fallback_flags[last];
      }
    }

  } // namespace

  namespace detail {

    void appendInteractionSelectionWorkspace(
        WorkspaceSize& required, std::size_t count) {
      std::size_t scan_bytes = 0;
      checkCuda(cub::DeviceScan::ExclusiveSum(
                    nullptr, scan_bytes,
                    static_cast<std::uint32_t*>(nullptr),
                    static_cast<std::uint32_t*>(nullptr), count),
                "query interaction fallback scan storage");
      required.add<EmInteractionRecord>(count);
      required.add<ProposalFallbackEvent>(count);
      required.add<std::uint32_t>(count);
      required.add<std::uint32_t>(count);
      required.add<EmInteractionRecord>(count);
      required.add<ProposalFallbackEvent>(count);
      required.add<std::uint32_t>(1);
      required.addBytes(scan_bytes);
    }

    DeviceInteractionSelectionBatch
    launchInteractionSelectionOnDevice(
        tables::FlatRateTableView device_table,
        EmParticleState const* device_particles, std::size_t count,
        std::uint64_t random_seed, std::uint64_t shower_id,
        DeviceWorkspace& workspace,
        bool defer_count_download) {
      if (count == 0 || device_particles == nullptr) {
        throw std::invalid_argument(
            "device interaction selection requires a non-empty input");
      }
      std::size_t scan_bytes = 0;
      checkCuda(cub::DeviceScan::ExclusiveSum(
                    nullptr, scan_bytes,
                    static_cast<std::uint32_t*>(nullptr),
                    static_cast<std::uint32_t*>(nullptr), count),
                "query interaction fallback scan storage");
      auto* raw_interactions =
          workspace.acquire<EmInteractionRecord>(count);
      auto* raw_fallbacks =
          workspace.acquire<ProposalFallbackEvent>(count);
      auto* fallback_flags =
          workspace.acquire<std::uint32_t>(count);
      auto* fallback_offsets =
          workspace.acquire<std::uint32_t>(count);
      auto* compact_interactions =
          workspace.acquire<EmInteractionRecord>(count);
      auto* compact_fallbacks =
          workspace.acquire<ProposalFallbackEvent>(count);
      auto* device_fallback_count =
          workspace.acquire<std::uint32_t>(1);
      auto* scan_temporary =
          workspace.acquireBytes(scan_bytes);

      auto const selection_block_count =
          (count + SelectionThreadsPerBlock - 1) /
          SelectionThreadsPerBlock;
      if (selection_block_count >
          std::numeric_limits<unsigned int>::max()) {
        throw std::length_error(
            "interaction selector launch is too large");
      }
      auto const selection_blocks =
          static_cast<unsigned int>(
              selection_block_count);
      auto const block_count =
          (count + ThreadsPerBlock - 1) / ThreadsPerBlock;
      if (block_count >
          std::numeric_limits<unsigned int>::max()) {
        throw std::length_error(
            "interaction compaction launch is too large");
      }
      auto const blocks = static_cast<unsigned int>(block_count);
      selectDiscreteInteractionsKernel
          <<<selection_blocks, SelectionThreadsPerBlock>>>(
          device_table, device_particles, count, random_seed,
          shower_id, raw_interactions, raw_fallbacks,
          fallback_flags);
      checkCuda(cudaGetLastError(),
                "interaction selection kernel launch");
      checkCuda(cub::DeviceScan::ExclusiveSum(
                    scan_temporary, scan_bytes,
                    fallback_flags, fallback_offsets,
                    count),
                "scan interaction fallback flags");
      finalizeSelectionFallbackCountKernel<<<1, 1>>>(
          fallback_flags, fallback_offsets, count,
          device_fallback_count);
      checkCuda(
          cudaGetLastError(),
          "finalize interaction fallback count launch");
      compactInteractionSelectionsKernel
          <<<blocks, ThreadsPerBlock>>>(
              raw_interactions, raw_fallbacks, fallback_flags,
              fallback_offsets, count, compact_interactions,
              compact_fallbacks, device_fallback_count);
      checkCuda(cudaGetLastError(),
                "interaction compaction kernel launch");

      std::size_t fallback_count = 0;
      std::size_t interaction_count = 0;
      auto* selected_interactions = compact_interactions;
      if (!defer_count_download) {
        std::uint32_t host_fallback_count = 0;
        checkCuda(
            cudaMemcpy(
                &host_fallback_count, device_fallback_count,
                sizeof(host_fallback_count),
                cudaMemcpyDeviceToHost),
            "download interaction fallback count");
        fallback_count =
            static_cast<std::size_t>(
                host_fallback_count);
        interaction_count = count - fallback_count;
        if (fallback_count == 0) {
          selected_interactions = raw_interactions;
        }
      }
      return {count, interaction_count, fallback_count,
              selected_interactions, compact_fallbacks,
              raw_interactions, device_fallback_count,
              defer_count_download};
    }

  } // namespace detail

  EmInteractionBatchResult selectDiscreteInteractionsForValidation(
      tables::FlatRateTableView device_table,
      std::vector<EmParticleState> const& particles,
      std::uint64_t random_seed, std::uint64_t shower_id, int device,
      detail::DeviceWorkspace& workspace) {
    EmInteractionBatchResult result{};
    result.input_particles = particles.size();
    if (particles.empty()) {
      return result;
    }
    if (device < 0) {
      throw std::invalid_argument(
          "interaction selector CUDA device must be non-negative");
    }
    if (particles.size() >
        std::numeric_limits<std::uint32_t>::max()) {
      throw std::length_error(
          "interaction selector batch exceeds 32-bit scan offsets");
    }
    checkCuda(cudaSetDevice(device),
              "cudaSetDevice(interaction selector)");

    detail::WorkspaceSize required;
    required.add<EmParticleState>(particles.size());
    detail::appendInteractionSelectionWorkspace(
        required, particles.size());
    workspace.prepare(required.bytes());

    auto* device_particles =
        workspace.acquire<EmParticleState>(particles.size());
    checkCuda(cudaMemcpy(
                  device_particles, particles.data(),
                  particles.size() * sizeof(EmParticleState),
                  cudaMemcpyHostToDevice),
              "upload interaction-selector particles");

    auto const batch = detail::launchInteractionSelectionOnDevice(
        device_table, device_particles, particles.size(),
        random_seed, shower_id, workspace);
    result.interactions.resize(batch.interaction_count);
    result.fallback_events.resize(batch.fallback_count);
    if (batch.interaction_count > 0) {
      checkCuda(cudaMemcpy(
                    result.interactions.data(), batch.interactions,
                    batch.interaction_count *
                        sizeof(EmInteractionRecord),
                    cudaMemcpyDeviceToHost),
                "download compact interaction records");
    }
    if (batch.fallback_count > 0) {
      checkCuda(cudaMemcpy(
                    result.fallback_events.data(), batch.fallbacks,
                    batch.fallback_count *
                        sizeof(ProposalFallbackEvent),
                    cudaMemcpyDeviceToHost),
                "download compact interaction fallback events");
    }
    return result;
  }

} // namespace corsika::gpu::em

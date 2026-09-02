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
#include <vector>

#include <corsika/accelerator/em/detail/PhotonTransportStep.hpp>
#include <corsika/gpu/em/CudaPhotonTransport.hpp>
#include <corsika/gpu/em/ObservationPlane.hpp>
#include <corsika/gpu/em/SphericalAtmosphere.hpp>
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

    __device__ ProposalFallbackEvent makeTransportFallback(
        EmInteractionRecord const& interaction,
        ProposalFallbackReason reason) {
      ProposalFallbackEvent event{};
      event.particle = interaction.particle;
      event.input_index = interaction.input_index;
      event.process_id = interaction.process_id;
      event.reason = reason;
      event.component_hash = interaction.component_hash;
      event.energy_fraction = interaction.energy_fraction;
      event.loss_quantile = interaction.loss_quantile;
      event.random_draw_id = interaction.loss_draw_id;
      return event;
    }

    __device__ bool closeRadius(double left, double right) {
      auto const scale = left > right ? left : right;
      return ::fabs(left - right) <=
             128. * 2.22044604925031308085e-16 *
                 (scale > 1. ? scale : 1.);
    }

    __global__ void transportPhotonsKernel(
        EnvironmentSnapshot environment,
        EmInteractionRecord const* interactions,
        EmInteractionRecord const* raw_interactions,
        std::uint32_t const* selection_fallback_count,
        std::size_t count,
        PhotonTransportRecord* raw_records,
        ProposalFallbackEvent* raw_fallbacks,
        std::uint32_t* fallback_flags) {
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
      auto interaction =
          selection_fallback_count != nullptr &&
                  selection_fallbacks == 0
              ? raw_interactions[index]
              : interactions[index];
      auto const shared =
          accelerator::em::detail::transportPhoton(
              environment, interaction);
      raw_records[index] = shared.record;
      raw_fallbacks[index] = shared.fallback;
      fallback_flags[index] = shared.fallback_flag;
      return;
      auto const start = interaction.particle;
      if (start.pid != static_cast<std::int32_t>(EmPid::Photon)) {
        raw_fallbacks[index] = makeTransportFallback(
            interaction,
            ProposalFallbackReason::UnsupportedParticle);
        fallback_flags[index] = 1;
        return;
      }
      if (interaction.status != EmInteractionStatus::Selected &&
          interaction.status !=
              EmInteractionStatus::NoDiscreteInteraction &&
          interaction.status !=
              EmInteractionStatus::ParticleCut) {
        raw_fallbacks[index] = makeTransportFallback(
            interaction,
            ProposalFallbackReason::InvalidFinalState);
        fallback_flags[index] = 1;
        return;
      }

      auto const layer = queryAtmosphereLayer(
          environment, start.position_m, start.direction);
      if (layer.status != AtmosphereStatus::Success) {
        raw_fallbacks[index] = makeTransportFallback(
            interaction,
            ProposalFallbackReason::UnsupportedGeometry);
        fallback_flags[index] = 1;
        return;
      }
      if (interaction.status ==
          EmInteractionStatus::ParticleCut) {
        PhotonTransportRecord record{};
        record.interaction = interaction;
        record.start = start;
        record.end = start;
        record.end.step_id++;
        record.input_index = interaction.input_index;
        record.limit = PhotonTransportLimit::ParticleCut;
        record.start_layer_index = layer.layer_index;
        record.end_layer_index = layer.layer_index;
        record.start_density_g_per_cm3 =
            layer.density_g_per_cm3;
        record.end_density_g_per_cm3 =
            layer.density_g_per_cm3;
        record.cut_deposited_energy_GeV =
            start.energy_GeV;
        raw_records[index] = record;
        fallback_flags[index] = 0;
        return;
      }
      auto const atmosphere_boundary = distanceToAtmosphereBoundary(
          environment, start.position_m, start.direction);
      auto const observation = intersectObservationPlaneStraight(
          environment, start.position_m, start.direction);
      auto const has_atmosphere_boundary =
          atmosphere_boundary.status == AtmosphereStatus::Success;
      auto const has_observation =
          observation.status == ObservationPlaneStatus::Success;
      if (!has_atmosphere_boundary && !has_observation) {
        raw_fallbacks[index] = makeTransportFallback(
            interaction,
            ProposalFallbackReason::UnsupportedGeometry);
        fallback_flags[index] = 1;
        return;
      }
      auto const observation_wins =
          has_observation &&
          (!has_atmosphere_boundary ||
           observation.distance_m < atmosphere_boundary.distance_m);
      auto const boundary_distance_m =
          observation_wins ? observation.distance_m
                           : atmosphere_boundary.distance_m;
      auto const limiting_radius_m =
          observation_wins ? 0. : atmosphere_boundary.radius_m;
      auto const boundary_grammage = atmosphereGrammage(
          environment, layer.layer_index, start.position_m,
          start.direction, boundary_distance_m);
      if (boundary_grammage.status != AtmosphereStatus::Success) {
        auto fallback = makeTransportFallback(
            interaction,
            ProposalFallbackReason::AtmosphereGrammageFailed);
        fallback.diagnostic_status =
            static_cast<std::int32_t>(boundary_grammage.status);
        fallback.diagnostic_value0 = boundary_distance_m;
        fallback.diagnostic_value1 = layer.density_g_per_cm3;
        fallback.diagnostic_value2 = limiting_radius_m;
        raw_fallbacks[index] = fallback;
        fallback_flags[index] = 1;
        return;
      }

      PhotonTransportRecord record{};
      record.interaction = interaction;
      record.start = start;
      record.input_index = interaction.input_index;
      record.start_layer_index = layer.layer_index;
      record.start_density_g_per_cm3 =
          layer.density_g_per_cm3;
      record.limiting_radius_m = limiting_radius_m;

      auto const reaches_interaction =
          interaction.status == EmInteractionStatus::Selected &&
          ::isfinite(interaction.interaction_grammage_g_per_cm2) &&
          interaction.interaction_grammage_g_per_cm2 >= 0. &&
          interaction.interaction_grammage_g_per_cm2 <=
              boundary_grammage.value;
      if (reaches_interaction) {
        auto const distance = atmosphereDistanceFromGrammage(
            environment, layer.layer_index, start.position_m,
            start.direction,
            interaction.interaction_grammage_g_per_cm2);
        if (distance.status != AtmosphereStatus::Success ||
            distance.value > boundary_distance_m *
                                 (1. + 1.e-12)) {
          auto fallback = makeTransportFallback(
              interaction,
              ProposalFallbackReason::AtmosphereInverseGrammageFailed);
          fallback.diagnostic_status =
              static_cast<std::int32_t>(distance.status);
          fallback.diagnostic_value0 = distance.value;
          fallback.diagnostic_value1 = boundary_distance_m;
          fallback.diagnostic_value2 =
              interaction.interaction_grammage_g_per_cm2;
          raw_fallbacks[index] = fallback;
          fallback_flags[index] = 1;
          return;
        }
        advancePhotonState(start, distance.value, record.end);
        auto const vertex = queryAtmosphereLayer(
            environment, record.end.position_m,
            record.end.direction);
        if (vertex.status != AtmosphereStatus::Success ||
            !::isfinite(vertex.density_g_per_cm3) ||
            !(vertex.density_g_per_cm3 > 0.)) {
          auto fallback = makeTransportFallback(
              interaction,
              ProposalFallbackReason::AtmosphereVertexLookupFailed);
          fallback.diagnostic_status =
              static_cast<std::int32_t>(vertex.status);
          fallback.diagnostic_value0 = vertex.radius_m;
          fallback.diagnostic_value1 = vertex.density_g_per_cm3;
          fallback.diagnostic_value2 = distance.value;
          raw_fallbacks[index] = fallback;
          fallback_flags[index] = 1;
          return;
        }
        record.limit = PhotonTransportLimit::Interaction;
        record.end_layer_index = vertex.layer_index;
        record.distance_m = distance.value;
        record.traversed_grammage_g_per_cm2 =
            interaction.interaction_grammage_g_per_cm2;
        record.end_density_g_per_cm3 =
            vertex.density_g_per_cm3;
        record.interaction.particle = record.end;
        record.interaction.mass_density_g_per_cm3 =
            vertex.density_g_per_cm3;
        // Scalar ParticleCut runs after every completed continuous step and
        // before the selected discrete interaction is generated.  Preserve
        // the full step, but suppress the interaction when its post-step
        // event time is beyond 10 ms.
        if (exceedsParticleCutTime(record.end.time_s)) {
          record.end.step_id++;
          record.limit = PhotonTransportLimit::ParticleCut;
          record.cut_deposited_energy_GeV = start.energy_GeV;
        }
        raw_records[index] = record;
        fallback_flags[index] = 0;
        return;
      }

      advancePhotonState(start, boundary_distance_m, record.end);
      record.end.step_id++;
      record.distance_m = boundary_distance_m;
      record.traversed_grammage_g_per_cm2 =
          boundary_grammage.value;
      if (observation_wins) {
        double radial[3]{};
        record.limiting_radius_m =
            atmosphere_detail::radiusVector(
                environment, record.end.position_m, radial);
        record.limit =
            PhotonTransportLimit::ObservationSurface;
        record.end_layer_index = -1;
        record.end_density_g_per_cm3 =
            layer.density_g_per_cm3;
      } else {
        auto const outermost =
            environment.atmosphere_layers[
                environment.number_of_layers - 1]
                .outer_radius_m;
        if (closeRadius(
                atmosphere_boundary.radius_m, outermost)) {
          record.limit =
              PhotonTransportLimit::EscapedEnvironment;
          record.end_layer_index = -1;
          record.end_density_g_per_cm3 = 0.;
        } else {
          auto const next = queryAtmosphereLayer(
              environment, record.end.position_m,
              record.end.direction);
          if (next.status != AtmosphereStatus::Success ||
              next.layer_index == layer.layer_index) {
            raw_fallbacks[index] = makeTransportFallback(
                interaction,
                ProposalFallbackReason::UnsupportedGeometry);
            fallback_flags[index] = 1;
            return;
          }
          record.limit = PhotonTransportLimit::LayerBoundary;
          record.end_layer_index = next.layer_index;
          record.end.medium_id =
              environment
                  .atmosphere_layers[next.layer_index]
                  .medium_id;
          record.end_density_g_per_cm3 =
              next.density_g_per_cm3;
        }
      }
      if (exceedsParticleCutTime(record.end.time_s)) {
        record.observation_surface_reached_before_cut =
            record.limit ==
                    PhotonTransportLimit::ObservationSurface
                ? 1U
                : 0U;
        record.limit = PhotonTransportLimit::ParticleCut;
        record.cut_deposited_energy_GeV = start.energy_GeV;
      }
      raw_records[index] = record;
      fallback_flags[index] = 0;
    }

    __global__ void compactTransportKernel(
        PhotonTransportRecord const* raw_records,
        ProposalFallbackEvent const* raw_fallbacks,
        std::uint32_t const* fallback_flags,
        std::uint32_t const* fallback_offsets, std::size_t count,
        PhotonTransportRecord* compact_records,
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

    __global__ void countFallbackFlagsKernel(
        std::uint32_t const* fallback_flags,
        std::size_t count,
        std::uint32_t* fallback_count) {
      auto const index =
          static_cast<std::size_t>(blockIdx.x) * blockDim.x +
          threadIdx.x;
      if (index < count && fallback_flags[index] != 0) {
        atomicAdd(fallback_count, 1U);
      }
    }

    __global__ void classifyTransportInteractionsKernel(
        PhotonTransportRecord const* records, std::size_t count,
        std::uint32_t* interaction_flags) {
      auto const index =
          static_cast<std::size_t>(blockIdx.x) * blockDim.x +
          threadIdx.x;
      if (index < count) {
        interaction_flags[index] =
            records[index].limit ==
                    PhotonTransportLimit::Interaction
                ? 1U
                : 0U;
      }
    }

    __global__ void compactTransportInteractionsKernel(
        PhotonTransportRecord const* records,
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

    __global__ void finalizeTransportInteractionCountKernel(
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

    void appendPhotonTransportWorkspace(
        WorkspaceSize& required, std::size_t count) {
      std::size_t scan_bytes = 0;
      checkCuda(cub::DeviceScan::ExclusiveSum(
                    nullptr, scan_bytes,
                    static_cast<std::uint32_t*>(nullptr),
                    static_cast<std::uint32_t*>(nullptr), count),
                "query photon transport scan storage");
      required.add<PhotonTransportRecord>(count);
      required.add<ProposalFallbackEvent>(count);
      required.add<std::uint32_t>(count);
      required.add<std::uint32_t>(count);
      required.add<PhotonTransportRecord>(count);
      required.add<ProposalFallbackEvent>(count);
      required.add<std::uint32_t>(2);
      required.addBytes(scan_bytes);
    }

    DevicePhotonTransportBatch launchPhotonTransportOnDevice(
        EnvironmentSnapshot const& environment,
        EmInteractionRecord const* device_interactions,
        std::size_t count, DeviceWorkspace& workspace,
        DeviceInteractionSelectionBatch*
            deferred_selection) {
      if (count == 0 || device_interactions == nullptr) {
        throw std::invalid_argument(
            "device photon transport requires a non-empty input");
      }
      std::size_t scan_bytes = 0;
      checkCuda(cub::DeviceScan::ExclusiveSum(
                    nullptr, scan_bytes,
                    static_cast<std::uint32_t*>(nullptr),
                    static_cast<std::uint32_t*>(nullptr), count),
                "query photon transport scan storage");
      auto* raw_records =
          workspace.acquire<PhotonTransportRecord>(count);
      auto* raw_fallbacks =
          workspace.acquire<ProposalFallbackEvent>(count);
      auto* fallback_flags =
          workspace.acquire<std::uint32_t>(count);
      auto* fallback_offsets =
          workspace.acquire<std::uint32_t>(count);
      auto* compact_records =
          workspace.acquire<PhotonTransportRecord>(count);
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
            "photon transport launch is too large");
      }
      auto const transport_blocks =
          static_cast<unsigned int>(
              transport_block_count);
      auto const block_count =
          (count + ThreadsPerBlock - 1) / ThreadsPerBlock;
      if (block_count >
          std::numeric_limits<unsigned int>::max()) {
        throw std::length_error(
            "photon transport compaction launch is too large");
      }
      auto const blocks = static_cast<unsigned int>(block_count);
      checkCuda(
          cudaMemset(
              device_summary, 0,
              2 * sizeof(std::uint32_t)),
          "clear photon transport summary");
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
            "photon transport received an invalid deferred selection");
      }
      transportPhotonsKernel
          <<<transport_blocks, TransportThreadsPerBlock>>>(
          environment, device_interactions,
          selection_raw_interactions,
          selection_fallback_count, count, raw_records,
          raw_fallbacks, fallback_flags);
      checkCuda(cudaGetLastError(),
                "photon transport kernel launch");
      countFallbackFlagsKernel<<<blocks, ThreadsPerBlock>>>(
          fallback_flags, count, device_fallback_count);
      checkCuda(
          cudaGetLastError(),
          "count photon transport fallback flags launch");
      if (selection_fallback_count != nullptr) {
        checkCuda(
            cudaMemcpy(
                device_summary, selection_fallback_count,
                sizeof(std::uint32_t),
                cudaMemcpyDeviceToDevice),
            "copy selection count into photon transport summary");
      }
      std::array<std::uint32_t, 2> host_summary{};
      checkCuda(
          cudaMemcpy(
              host_summary.data(), device_summary,
              sizeof(host_summary),
              cudaMemcpyDeviceToHost),
          "download photon transport control summary");
      auto const selection_fallbacks =
          static_cast<std::size_t>(host_summary[0]);
      if (selection_fallbacks > count) {
        throw std::runtime_error(
            "photon selection fallback count exceeds its input");
      }
      auto const selection_interactions =
          count - selection_fallbacks;
      auto const fallback_count =
          static_cast<std::size_t>(host_summary[1]);
      if (fallback_count > selection_interactions) {
        throw std::runtime_error(
            "photon transport fallback count exceeds selected interactions");
      }
      auto const record_count =
          selection_interactions - fallback_count;
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
                  "scan photon transport fallback flags");
        compactTransportKernel<<<blocks, ThreadsPerBlock>>>(
            raw_records, raw_fallbacks, fallback_flags,
            fallback_offsets, selection_interactions,
            compact_records,
            compact_fallbacks);
        checkCuda(cudaGetLastError(),
                  "compact photon transport kernel launch");
        selected_records = compact_records;
      }
      return {selection_interactions, record_count, fallback_count,
              selected_records, compact_fallbacks};
    }

    void appendTransportInteractionWorkspace(
        WorkspaceSize& required, std::size_t count) {
      std::size_t scan_bytes = 0;
      checkCuda(cub::DeviceScan::ExclusiveSum(
                    nullptr, scan_bytes,
                    static_cast<std::uint32_t*>(nullptr),
                    static_cast<std::uint32_t*>(nullptr), count),
                "query transport-interaction scan storage");
      required.add<std::uint32_t>(count);
      required.add<std::uint32_t>(count);
      required.add<EmInteractionRecord>(count);
      required.add<std::uint32_t>(1);
      required.addBytes(scan_bytes);
    }

    DeviceTransportInteractionBatch
    extractTransportInteractionsOnDevice(
        PhotonTransportRecord const* device_records,
        std::size_t count, DeviceWorkspace& workspace,
        bool defer_count_download) {
      if (count == 0 || device_records == nullptr) {
        throw std::invalid_argument(
            "transport interaction extraction requires records");
      }
      std::size_t scan_bytes = 0;
      checkCuda(cub::DeviceScan::ExclusiveSum(
                    nullptr, scan_bytes,
                    static_cast<std::uint32_t*>(nullptr),
                    static_cast<std::uint32_t*>(nullptr), count),
                "query transport-interaction scan storage");
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
      classifyTransportInteractionsKernel<<<blocks, ThreadsPerBlock>>>(
          device_records, count, flags);
      checkCuda(cudaGetLastError(),
                "classify transport interactions launch");
      checkCuda(cub::DeviceScan::ExclusiveSum(
                    scan_temporary, scan_bytes, flags, offsets, count),
                "scan transport interaction flags");
      finalizeTransportInteractionCountKernel<<<1, 1>>>(
          flags, offsets, count, device_interaction_count);
      checkCuda(
          cudaGetLastError(),
          "finalize photon transport interaction count launch");
      compactTransportInteractionsKernel<<<blocks, ThreadsPerBlock>>>(
          device_records, flags, offsets, count, interactions);
      checkCuda(cudaGetLastError(),
                "compact transport interactions launch");
      std::size_t interaction_count = 0;
      if (!defer_count_download) {
        std::uint32_t host_interaction_count = 0;
        checkCuda(
            cudaMemcpy(
                &host_interaction_count,
                device_interaction_count,
                sizeof(host_interaction_count),
                cudaMemcpyDeviceToHost),
            "download photon interaction count");
        interaction_count = host_interaction_count;
      }
      return {count, interaction_count, interactions,
              device_interaction_count,
              defer_count_download};
    }

  } // namespace detail

  PhotonTransportBatchResult transportPhotonsForValidation(
      EnvironmentSnapshot const& environment,
      std::vector<EmInteractionRecord> const& interactions,
      int device, detail::DeviceWorkspace& workspace) {
    PhotonTransportBatchResult result{};
    result.input_interactions = interactions.size();
    if (interactions.empty()) {
      return result;
    }
    if (device < 0) {
      throw std::invalid_argument(
          "photon transport CUDA device must be non-negative");
    }
    if (!atmosphere_detail::validEnvironment(environment)) {
      throw std::invalid_argument(
          "photon transport received an invalid environment snapshot");
    }
    if (interactions.size() >
        std::numeric_limits<std::uint32_t>::max()) {
      throw std::length_error(
          "photon transport batch exceeds 32-bit scan offsets");
    }
    checkCuda(cudaSetDevice(device),
              "cudaSetDevice(photon transport)");

    auto const count = interactions.size();
    detail::WorkspaceSize required;
    required.add<EmInteractionRecord>(count);
    detail::appendPhotonTransportWorkspace(required, count);
    workspace.prepare(required.bytes());

    auto* device_interactions =
        workspace.acquire<EmInteractionRecord>(count);
    checkCuda(cudaMemcpy(
                  device_interactions, interactions.data(),
                  count * sizeof(EmInteractionRecord),
                  cudaMemcpyHostToDevice),
              "upload photon transport interactions");

    auto const batch = detail::launchPhotonTransportOnDevice(
        environment, device_interactions, count, workspace);
    result.records.resize(batch.record_count);
    result.fallback_events.resize(batch.fallback_count);
    if (batch.record_count != 0) {
      checkCuda(cudaMemcpy(
                    result.records.data(), batch.records,
                    batch.record_count *
                        sizeof(PhotonTransportRecord),
                    cudaMemcpyDeviceToHost),
                "download photon transport records");
    }
    if (batch.fallback_count != 0) {
      checkCuda(cudaMemcpy(
                    result.fallback_events.data(),
                    batch.fallbacks,
                    batch.fallback_count *
                        sizeof(ProposalFallbackEvent),
                    cudaMemcpyDeviceToHost),
                "download photon transport fallbacks");
    }
    return result;
  }

} // namespace corsika::gpu::em

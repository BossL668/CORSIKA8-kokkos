/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <vector>

#include <corsika/gpu/em/CudaDecisionReplayVerifier.hpp>

namespace corsika::gpu::em {

  namespace {

    constexpr unsigned int ThreadsPerBlock = 256;
    constexpr double SpeedOfLightMPerS = 299792458.;
    constexpr std::uint64_t FnvOffset =
        14695981039346656037ULL;
    constexpr std::uint64_t FnvPrime =
        1099511628211ULL;

    void checkCuda(cudaError_t status, char const* operation) {
      if (status == cudaSuccess) { return; }
      std::ostringstream message;
      message << operation << " failed: "
              << cudaGetErrorString(status);
      throw std::runtime_error(message.str());
    }

    __host__ __device__ std::uint64_t hashBytes(
        unsigned char const* bytes, std::size_t count) {
      auto hash = FnvOffset;
      for (std::size_t index = 0; index < count; ++index) {
        hash ^= static_cast<std::uint64_t>(bytes[index]);
        hash *= FnvPrime;
      }
      return hash;
    }

    std::uint64_t orderedHash(
        std::vector<std::uint64_t> const& hashes) {
      auto result = FnvOffset;
      for (auto const hash : hashes) {
        for (unsigned int byte = 0; byte < 8; ++byte) {
          result ^= (hash >> (byte * 8U)) & 0xffU;
          result *= FnvPrime;
        }
      }
      return result;
    }

    enum ReplayValidityFlag : std::uint32_t {
      NonFinite = 1U << 0U,
      NegativeEnergy = 1U << 1U,
      NegativeWeight = 1U << 2U,
      BackwardTime = 1U << 3U,
      InvalidDirection = 1U << 4U,
      Superluminal = 1U << 5U,
    };

    constexpr std::uint32_t FatalValidityFlags =
        NonFinite | NegativeEnergy | NegativeWeight |
        BackwardTime | InvalidDirection;

    __global__ void verifyReplayKernel(
        validation::ReplayTransportStep const* records,
        std::size_t count, std::uint64_t* hashes,
        std::uint32_t* flags) {
      auto const index =
          static_cast<std::size_t>(blockIdx.x) * blockDim.x +
          threadIdx.x;
      if (index >= count) { return; }
      auto const& record = records[index];
      hashes[index] = hashBytes(
          reinterpret_cast<unsigned char const*>(&record),
          sizeof(record));

      std::uint32_t status = 0;
      auto finite = isfinite(record.start_time_s) &&
                    isfinite(record.end_time_s) &&
                    isfinite(record.start_total_energy_GeV) &&
                    isfinite(record.end_total_energy_GeV) &&
                    isfinite(record.weight);
      double start_norm2 = 0.;
      double end_norm2 = 0.;
      double distance2 = 0.;
      for (int axis = 0; axis < 3; ++axis) {
        finite = finite &&
                 isfinite(record.start_position_m[axis]) &&
                 isfinite(record.end_position_m[axis]) &&
                 isfinite(record.start_direction[axis]) &&
                 isfinite(record.end_direction[axis]);
        start_norm2 +=
            record.start_direction[axis] *
            record.start_direction[axis];
        end_norm2 +=
            record.end_direction[axis] *
            record.end_direction[axis];
        auto const delta =
            record.end_position_m[axis] -
            record.start_position_m[axis];
        distance2 += delta * delta;
      }
      if (!finite) { status |= NonFinite; }
      if (record.start_total_energy_GeV < 0. ||
          record.end_total_energy_GeV < 0.) {
        status |= NegativeEnergy;
      }
      if (record.weight < 0.) { status |= NegativeWeight; }
      auto const duration =
          record.end_time_s - record.start_time_s;
      if (duration < 0.) { status |= BackwardTime; }
      if (fabs(start_norm2 - 1.) > 1.e-8 ||
          fabs(end_norm2 - 1.) > 1.e-8) {
        status |= InvalidDirection;
      }
      if (duration > 0. && distance2 > 0.) {
        auto const speed = sqrt(distance2) / duration;
        // TrackingLeapFrogCurved advances the second position half-step with
        // the temporarily unnormalised velocity and normalises only the
        // direction returned by the trajectory.  Therefore the legacy scalar
        // Step can have |dx|/dt > c at a coarse magnetic-deflection setting.
        // Preserve this as an explicit diagnostic: exact replay must reproduce
        // the original Step, including this known discretisation property.
        if (speed > SpeedOfLightMPerS * (1. + 1.e-7)) {
          status |= Superluminal;
        }
      }
      flags[index] = status;
    }

  } // namespace

  CudaDecisionReplayVerification verifyDecisionReplayOnCuda(
      std::vector<validation::ReplayTransportStep> const& records,
      int device) {
    CudaDecisionReplayVerification result{};
    result.records = records.size();
    if (records.empty()) { return result; }
    checkCuda(cudaSetDevice(device), "set CUDA replay verifier device");

    auto const record_bytes =
        records.size() * sizeof(records.front());
    auto const hash_bytes =
        records.size() * sizeof(std::uint64_t);
    auto const flag_bytes =
        records.size() * sizeof(std::uint32_t);
    validation::ReplayTransportStep* device_records = nullptr;
    std::uint64_t* device_hashes = nullptr;
    std::uint32_t* device_flags = nullptr;
    checkCuda(
        cudaMalloc(
            reinterpret_cast<void**>(&device_records),
            record_bytes),
        "allocate CUDA replay records");
    try {
      checkCuda(
          cudaMalloc(
              reinterpret_cast<void**>(&device_hashes),
              hash_bytes),
          "allocate CUDA replay hashes");
      checkCuda(
          cudaMalloc(
              reinterpret_cast<void**>(&device_flags),
              flag_bytes),
          "allocate CUDA replay flags");
      checkCuda(
          cudaMemcpy(
              device_records, records.data(), record_bytes,
              cudaMemcpyHostToDevice),
          "upload CUDA replay records");
      auto const blocks = static_cast<unsigned int>(
          (records.size() + ThreadsPerBlock - 1) /
          ThreadsPerBlock);
      verifyReplayKernel<<<blocks, ThreadsPerBlock>>>(
          device_records, records.size(),
          device_hashes, device_flags);
      checkCuda(
          cudaGetLastError(),
          "launch CUDA decision replay verifier");
      std::vector<std::uint64_t> gpu_hashes(records.size());
      std::vector<std::uint32_t> gpu_flags(records.size());
      checkCuda(
          cudaMemcpy(
              gpu_hashes.data(), device_hashes, hash_bytes,
              cudaMemcpyDeviceToHost),
          "download CUDA replay hashes");
      checkCuda(
          cudaMemcpy(
              gpu_flags.data(), device_flags, flag_bytes,
              cudaMemcpyDeviceToHost),
          "download CUDA replay flags");

      std::vector<std::uint64_t> host_hashes;
      host_hashes.reserve(records.size());
      for (std::size_t index = 0;
           index < records.size(); ++index) {
        auto const& record = records[index];
        auto const hash = hashBytes(
            reinterpret_cast<unsigned char const*>(&record),
            sizeof(record));
        host_hashes.push_back(hash);
        if (hash != gpu_hashes[index]) {
          ++result.byte_hash_mismatches;
        }
        if ((gpu_flags[index] & FatalValidityFlags) != 0) {
          ++result.invalid_records;
        }
        if ((gpu_flags[index] & NonFinite) != 0) {
          ++result.nonfinite_records;
        }
        if ((gpu_flags[index] & NegativeEnergy) != 0) {
          ++result.negative_energy_records;
        }
        if ((gpu_flags[index] & NegativeWeight) != 0) {
          ++result.negative_weight_records;
        }
        if ((gpu_flags[index] & BackwardTime) != 0) {
          ++result.backward_time_records;
        }
        if ((gpu_flags[index] & InvalidDirection) != 0) {
          ++result.invalid_direction_records;
        }
        if ((gpu_flags[index] & Superluminal) != 0) {
          ++result.superluminal_records;
        }
        auto const abs_pdg = std::abs(record.pdg);
        if (abs_pdg == 11 || record.pdg == 22) {
          ++result.electromagnetic_records;
        }
        if (abs_pdg == 11) {
          ++result.lepton_records;
        } else if (record.pdg == 22) {
          ++result.photon_records;
        }
        auto distance2 = 0.;
        auto start_norm2 = 0.;
        auto end_norm2 = 0.;
        for (int axis = 0; axis < 3; ++axis) {
          auto const delta =
              record.end_position_m[axis] -
              record.start_position_m[axis];
          distance2 += delta * delta;
          start_norm2 +=
              record.start_direction[axis] *
              record.start_direction[axis];
          end_norm2 +=
              record.end_direction[axis] *
              record.end_direction[axis];
        }
        auto const distance = std::sqrt(distance2);
        result.weighted_track_length_m +=
            record.weight * distance;
        if (abs_pdg == 11) {
          result.weighted_lepton_track_length_m +=
              record.weight * distance;
        }
        result.maximum_direction_norm_error =
            std::max(
                result.maximum_direction_norm_error,
                std::max(
                    std::abs(std::sqrt(start_norm2) - 1.),
                    std::abs(std::sqrt(end_norm2) - 1.)));
        auto const duration =
            record.end_time_s - record.start_time_s;
        if (duration > 0.) {
          result.maximum_speed_over_c =
              std::max(
                  result.maximum_speed_over_c,
                  distance / duration / SpeedOfLightMPerS);
        }
      }
      result.ordered_host_hash = orderedHash(host_hashes);
      result.ordered_device_hash = orderedHash(gpu_hashes);
    } catch (...) {
      cudaFree(device_flags);
      cudaFree(device_hashes);
      cudaFree(device_records);
      throw;
    }
    checkCuda(cudaFree(device_flags),
              "free CUDA replay flags");
    checkCuda(cudaFree(device_hashes),
              "free CUDA replay hashes");
    checkCuda(cudaFree(device_records),
              "free CUDA replay records");
    return result;
  }

} // namespace corsika::gpu::em

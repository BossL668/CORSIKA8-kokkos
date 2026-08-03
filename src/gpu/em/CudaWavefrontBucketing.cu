/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include <cuda_runtime.h>

#include <cub/device/device_radix_sort.cuh>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <sstream>
#include <stdexcept>

#include <corsika/gpu/em/detail/DeviceWavefrontBucketing.hpp>

namespace corsika::gpu::em::detail {

  namespace {

    constexpr unsigned int ThreadsPerBlock = 256;
    constexpr double EnergyBinsPerOctave = 16.;
    constexpr double EnergyBinOffset = 32768.;
    constexpr std::uint16_t InvalidEnergyBin =
        std::numeric_limits<std::uint16_t>::max();
    constexpr std::uint16_t MaximumFiniteEnergyBin =
        InvalidEnergyBin - 1;

    void checkCuda(cudaError_t status, char const* operation) {
      if (status == cudaSuccess) {
        return;
      }
      std::ostringstream message;
      message << operation << " failed: "
              << cudaGetErrorString(status);
      throw std::runtime_error(message.str());
    }

    __device__ std::uint64_t pidBucket(
        std::int32_t pid) {
      if (pid ==
          static_cast<std::int32_t>(EmPid::Photon)) {
        return 0;
      }
      if (pid ==
          static_cast<std::int32_t>(EmPid::Electron)) {
        return 1;
      }
      if (pid ==
          static_cast<std::int32_t>(EmPid::Positron)) {
        return 2;
      }
      return 3;
    }

    __device__ std::uint16_t energyBucket(
        double energy_GeV) {
      if (!(energy_GeV > 0.) || !isfinite(energy_GeV)) {
        return InvalidEnergyBin;
      }
      auto const coordinate =
          floor(log2(energy_GeV) * EnergyBinsPerOctave +
                EnergyBinOffset);
      if (!(coordinate > 0.)) {
        return 0;
      }
      if (coordinate >=
          static_cast<double>(MaximumFiniteEnergyBin)) {
        return MaximumFiniteEnergyBin;
      }
      return static_cast<std::uint16_t>(coordinate);
    }

    __device__ std::uint64_t bucketKey(
        EmParticleState const& particle) {
      auto const ordered_medium =
          static_cast<std::uint32_t>(particle.medium_id) ^
          0x80000000U;
      return
          (pidBucket(particle.pid) << 48U) |
          (static_cast<std::uint64_t>(ordered_medium)
           << 16U) |
          energyBucket(particle.energy_GeV);
    }

    __global__ void makeKeysAndIndices(
        EmParticleState const* particles,
        std::size_t count, std::uint64_t* keys,
        std::uint32_t* indices) {
      auto const index =
          static_cast<std::size_t>(blockIdx.x) *
              blockDim.x +
          threadIdx.x;
      if (index >= count) {
        return;
      }
      keys[index] = bucketKey(particles[index]);
      indices[index] =
          static_cast<std::uint32_t>(index);
    }

    __global__ void gatherSortedParticles(
        EmParticleState const* input,
        std::uint32_t const* sorted_indices,
        std::size_t count, EmParticleState* output) {
      auto const index =
          static_cast<std::size_t>(blockIdx.x) *
              blockDim.x +
          threadIdx.x;
      if (index < count) {
        output[index] = input[sorted_indices[index]];
      }
    }

    std::size_t radixTemporaryBytes(
        std::size_t count) {
      if (count < MinimumWavefrontRadixSortSize) {
        return 0;
      }
      std::size_t temporary_bytes = 0;
      checkCuda(
          cub::DeviceRadixSort::SortPairs(
              nullptr, temporary_bytes,
              static_cast<std::uint64_t const*>(nullptr),
              static_cast<std::uint64_t*>(nullptr),
              static_cast<std::uint32_t const*>(nullptr),
              static_cast<std::uint32_t*>(nullptr),
              count, 0, WavefrontBucketKeyBits),
          "query wavefront radix-sort storage");
      return temporary_bytes;
    }

    unsigned int blockCount(std::size_t count) {
      auto const blocks =
          (count + ThreadsPerBlock - 1) /
          ThreadsPerBlock;
      if (blocks >
          std::numeric_limits<unsigned int>::max()) {
        throw std::length_error(
            "wavefront bucketing launch is too large");
      }
      return static_cast<unsigned int>(blocks);
    }

  } // namespace

  void appendWavefrontBucketingWorkspace(
      WorkspaceSize& required, std::size_t count) {
    if (count < MinimumWavefrontRadixSortSize) {
      return;
    }
    if (count >
        std::numeric_limits<std::uint32_t>::max()) {
      throw std::length_error(
          "wavefront bucketing exceeds 32-bit indices");
    }
    required.add<std::uint64_t>(count);
    required.add<std::uint64_t>(count);
    required.add<std::uint32_t>(count);
    required.add<std::uint32_t>(count);
    required.add<EmParticleState>(count);
    required.addBytes(radixTemporaryBytes(count));
  }

  DeviceWavefrontBucketBatch
  launchWavefrontBucketingOnDevice(
      EmParticleState const* input, std::size_t count,
      DeviceWorkspace& workspace) {
    if (count == 0 || input == nullptr) {
      throw std::invalid_argument(
          "wavefront bucketing requires non-empty input");
    }
    if (count < MinimumWavefrontRadixSortSize) {
      return {input, nullptr, count, false};
    }
    if (count >
        std::numeric_limits<std::uint32_t>::max()) {
      throw std::length_error(
          "wavefront bucketing exceeds 32-bit indices");
    }

    auto* input_keys =
        workspace.acquire<std::uint64_t>(count);
    auto* output_keys =
        workspace.acquire<std::uint64_t>(count);
    auto* input_indices =
        workspace.acquire<std::uint32_t>(count);
    auto* output_indices =
        workspace.acquire<std::uint32_t>(count);
    auto* output_particles =
        workspace.acquire<EmParticleState>(count);
    auto const temporary_bytes =
        radixTemporaryBytes(count);
    auto* temporary =
        workspace.acquireBytes(temporary_bytes);

    auto const blocks = blockCount(count);
    makeKeysAndIndices<<<blocks, ThreadsPerBlock>>>(
        input, count, input_keys, input_indices);
    checkCuda(
        cudaGetLastError(),
        "make wavefront bucket keys launch");
    auto actual_temporary_bytes = temporary_bytes;
    checkCuda(
        cub::DeviceRadixSort::SortPairs(
            temporary, actual_temporary_bytes,
            input_keys, output_keys,
            input_indices, output_indices, count,
            0, WavefrontBucketKeyBits),
        "stable wavefront radix sort");
    gatherSortedParticles<<<blocks, ThreadsPerBlock>>>(
        input, output_indices, count, output_particles);
    checkCuda(
        cudaGetLastError(),
        "gather bucketed wavefront launch");
    return {
        output_particles, output_keys, count, true};
  }

  WavefrontBucketingValidationResult
  bucketWavefrontForValidation(
      std::vector<EmParticleState> const& particles,
      int device, DeviceWorkspace& workspace) {
    WavefrontBucketingValidationResult result{};
    if (particles.empty()) {
      return result;
    }
    if (device < 0) {
      throw std::invalid_argument(
          "wavefront bucketing CUDA device must be non-negative");
    }
    checkCuda(
        cudaSetDevice(device),
        "cudaSetDevice(wavefront bucketing)");

    WorkspaceSize required;
    required.add<EmParticleState>(particles.size());
    appendWavefrontBucketingWorkspace(
        required, particles.size());
    workspace.prepare(required.bytes());
    auto* device_input =
        workspace.acquire<EmParticleState>(
            particles.size());
    checkCuda(
        cudaMemcpy(
            device_input, particles.data(),
            particles.size() * sizeof(EmParticleState),
            cudaMemcpyHostToDevice),
        "upload wavefront bucketing validation input");
    auto const bucketed =
        launchWavefrontBucketingOnDevice(
            device_input, particles.size(), workspace);

    result.particles.resize(particles.size());
    result.keys.resize(particles.size());
    checkCuda(
        cudaMemcpy(
            result.particles.data(), bucketed.particles,
            particles.size() * sizeof(EmParticleState),
            cudaMemcpyDeviceToHost),
        "download bucketed particles");
    if (bucketed.keys != nullptr) {
      checkCuda(
          cudaMemcpy(
              result.keys.data(), bucketed.keys,
              particles.size() * sizeof(std::uint64_t),
              cudaMemcpyDeviceToHost),
          "download wavefront bucket keys");
    } else {
      std::fill(result.keys.begin(), result.keys.end(), 0);
    }
    return result;
  }

} // namespace corsika::gpu::em::detail

/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include <cuda_runtime_api.h>

#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <corsika/gpu/em/CudaEmBackend.hpp>

namespace {

  using namespace corsika::gpu::em;

  std::vector<EmParticleState> runOnce() {
    EnvironmentSnapshot environment{};
    ProposalTableSet tables{};
    GpuEmConfig config{};
    config.min_batch_size = 16;
    config.memory_fraction = 0.01;

    CudaEmBackend backend;
    backend.initialize(environment, tables, config);

    for (std::uint64_t i = 0; i < 16; ++i) {
      EmParticleState particle{};
      particle.pid = static_cast<std::int32_t>(EmPid::Photon);
      particle.energy_GeV = 1000. + static_cast<double>(i);
      particle.position_m[2] = 10000.;
      particle.direction[2] = -1.;
      particle.history_id = i + 1;
      backend.enqueue(particle);
    }

    auto const batch = backend.advanceWavefront();
    if (batch.input_particles != 16) {
      throw std::runtime_error("toy wavefront consumed the wrong number of particles");
    }
    return backend.downloadActiveParticles();
  }

  bool equal(EmParticleState const& a, EmParticleState const& b) {
    if (a.pid != b.pid || a.medium_id != b.medium_id ||
        a.generation != b.generation || a.energy_GeV != b.energy_GeV ||
        a.time_s != b.time_s || a.weight != b.weight ||
        a.history_id != b.history_id ||
        a.parent_history_id != b.parent_history_id || a.step_id != b.step_id) {
      return false;
    }
    for (int i = 0; i < 3; ++i) {
      if (a.position_m[i] != b.position_m[i] ||
          a.direction[i] != b.direction[i]) {
        return false;
      }
    }
    return true;
  }

  void runStress(std::uint64_t target_advanced_particles) {
    constexpr std::size_t InputBatchSize = 262144;
    std::uint64_t total_advanced = 0;
    std::uint64_t total_produced = 0;
    std::uint64_t showers = 0;

    while (total_advanced < target_advanced_particles) {
      EnvironmentSnapshot environment{};
      ProposalTableSet tables{};
      GpuEmConfig config{};
      config.min_batch_size = InputBatchSize;
      config.memory_fraction = 0.30;
      config.random_seed = 0x8c0a5eedULL;
      config.shower_id = showers;

      CudaEmBackend backend;
      backend.initialize(environment, tables, config);
      for (std::uint64_t i = 0; i < InputBatchSize; ++i) {
        EmParticleState particle{};
        particle.pid = static_cast<std::int32_t>(EmPid::Photon);
        particle.energy_GeV = 1000.;
        particle.direction[2] = -1.;
        particle.weight = 1.;
        particle.history_id = i + 1;
        backend.enqueue(particle);
      }
      backend.drain();
      if (!backend.empty()) {
        throw std::runtime_error("stress drain left particles in the GPU queue");
      }
      auto const& statistics = backend.statistics();
      if (statistics.queue_overflows != 0) {
        throw std::runtime_error("stress drain overflowed its GPU queue");
      }
      total_advanced += statistics.particles_advanced;
      total_produced += statistics.particles_produced;
      ++showers;
    }

    std::cout << "CUDA toy stress advanced " << total_advanced
              << " particles and produced " << total_produced << " in "
              << showers << " drained showers\n";
  }

} // namespace

int main(int argc, char** argv) {
  int device_count = 0;
  auto const status = cudaGetDeviceCount(&device_count);
  if (status != cudaSuccess || device_count == 0) {
    std::cerr << "SKIP: no CUDA device is accessible: "
              << cudaGetErrorString(status) << '\n';
    return 77;
  }

  try {
    if (argc == 3 && std::string{argv[1]} == "--stress") {
      runStress(std::stoull(argv[2]));
      return 0;
    }
    if (argc != 1) {
      std::cerr << "usage: testGpuEmCuda [--stress ADVANCED_PARTICLES]\n";
      return 2;
    }

    auto const first = runOnce();
    auto const second = runOnce();
    if (first.size() != second.size()) {
      std::cerr << "deterministic runs produced different queue sizes\n";
      return 1;
    }
    for (std::size_t i = 0; i < first.size(); ++i) {
      if (!equal(first[i], second[i])) {
        std::cerr << "deterministic runs differ at output " << i << '\n';
        return 1;
      }
    }
    std::cout << "CUDA toy wavefront produced " << first.size()
              << " deterministic particles\n";
  } catch (std::exception const& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
  return 0;
}

/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <corsika/gpu/em/detail/DeviceWavefrontBucketing.hpp>

namespace {

  using namespace corsika::gpu::em;

  std::size_t checks = 0;

  void require(bool condition, std::string const& message) {
    ++checks;
    if (!condition) {
      throw std::runtime_error(message);
    }
  }

  EmParticleState particle(
      EmPid pid, std::int32_t medium, double energy,
      std::uint64_t history) {
    EmParticleState result{};
    result.pid = static_cast<std::int32_t>(pid);
    result.medium_id = medium;
    result.energy_GeV = energy;
    result.direction[2] = -1.;
    result.weight = 1.;
    result.history_id = history;
    return result;
  }

  std::vector<std::uint64_t> histories(
      std::vector<EmParticleState> const& particles) {
    std::vector<std::uint64_t> result;
    result.reserve(particles.size());
    for (auto const& value : particles) {
      result.push_back(value.history_id);
    }
    return result;
  }

} // namespace

int main() {
  int devices = 0;
  auto const device_status = cudaGetDeviceCount(&devices);
  if (device_status != cudaSuccess || devices == 0) {
    std::cerr << "CUDA wavefront bucketing test skipped: "
              << cudaGetErrorString(device_status) << '\n';
    return 77;
  }

  try {
    detail::DeviceWorkspace workspace;
    workspace.configure(0, 64ULL * 1024ULL * 1024ULL);

    std::vector<EmParticleState> input{
        particle(EmPid::Positron, 2, 10., 1),
        particle(EmPid::Photon, 2, 100., 2),
        particle(EmPid::Electron, 1, 10., 3),
        particle(EmPid::Photon, 1, 100., 4),
        particle(EmPid::Photon, 1, 1., 5),
        particle(EmPid::Electron, 1, 10., 6),
        particle(EmPid::Electron, 1, 10., 7),
        particle(EmPid::Electron, 0, 1000., 8),
        particle(EmPid::Positron, -1, 0.1, 9)};
    auto next_history = std::uint64_t{10};
    while (
        input.size() <
        detail::MinimumWavefrontRadixSortSize + 37) {
      auto const index = input.size();
      auto const pid =
          index % 3 == 0
              ? EmPid::Photon
              : (index % 3 == 1
                     ? EmPid::Electron
                     : EmPid::Positron);
      input.push_back(
          particle(
              pid,
              static_cast<std::int32_t>(
                  index % 7) -
                  3,
              0.01 *
                  static_cast<double>(
                      1 + (index * 17) % 10000),
              next_history++));
    }

    auto first = detail::bucketWavefrontForValidation(
        input, 0, workspace);
    require(
        first.particles.size() == input.size() &&
            first.keys.size() == input.size(),
        "bucketed output size differs");
    require(
        std::is_sorted(first.keys.begin(), first.keys.end()),
        "PID x medium x energy bucket keys are not sorted");

    auto input_histories = histories(input);
    auto output_histories = histories(first.particles);
    std::sort(input_histories.begin(), input_histories.end());
    auto sorted_output_histories = output_histories;
    std::sort(
        sorted_output_histories.begin(),
        sorted_output_histories.end());
    require(
        input_histories == sorted_output_histories,
        "bucketing lost or duplicated a particle");

    auto position6 = std::find(
        output_histories.begin(), output_histories.end(), 6);
    auto position7 = std::find(
        output_histories.begin(), output_histories.end(), 7);
    require(
        position6 != output_histories.end() &&
            position7 != output_histories.end() &&
            position6 < position7,
        "equal-key source order is not stable");
    require(
        first.keys[
            static_cast<std::size_t>(
                position6 - output_histories.begin())] ==
            first.keys[
                static_cast<std::size_t>(
                    position7 - output_histories.begin())],
        "identical PID/medium/energy particles have different keys");

    auto second = detail::bucketWavefrontForValidation(
        input, 0, workspace);
    require(
        histories(second.particles) == output_histories &&
            second.keys == first.keys,
        "wavefront bucketing is not repeatable");

    std::vector<EmParticleState> singleton{
        particle(EmPid::Photon, 4, 5., 99)};
    auto one = detail::bucketWavefrontForValidation(
        singleton, 0, workspace);
    require(
        one.particles.size() == 1 &&
            one.particles.front().history_id == 99 &&
            one.keys.size() == 1,
        "singleton wavefront bucketing differs");

    std::vector<EmParticleState> empty;
    auto zero = detail::bucketWavefrontForValidation(
        empty, 0, workspace);
    require(
        zero.particles.empty() && zero.keys.empty(),
        "empty wavefront bucketing differs");
  } catch (std::exception const& error) {
    std::cerr
        << "GPU wavefront bucketing validation failed after "
        << checks << " checks: " << error.what() << '\n';
    return EXIT_FAILURE;
  }

  std::cout
      << "GPU wavefront bucketing validation passed: "
      << checks << " checks\n";
  return EXIT_SUCCESS;
}

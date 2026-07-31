/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include <cuda_runtime.h>

#include <corsika/gpu/em/EmThinning.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

  using namespace corsika::gpu::em;

  struct Query {
    EmThinningConfig config{};
    double parent_energy_GeV{};
    double parent_weight{};
    double first_energy_GeV{};
    double second_energy_GeV{};
    double first_uniform{};
    double second_uniform{};
  };

  std::size_t checks = 0;

  void require(bool condition, std::string const& message) {
    ++checks;
    if (!condition) {
      throw std::runtime_error(message);
    }
  }

  void requireNear(
      double actual, double expected, double tolerance,
      std::string const& message) {
    ++checks;
    if (!(std::abs(actual - expected) <= tolerance)) {
      throw std::runtime_error(
          message + ": actual=" + std::to_string(actual) +
          ", expected=" + std::to_string(expected));
    }
  }

  void checkCuda(cudaError_t status, char const* operation) {
    if (status != cudaSuccess) {
      throw std::runtime_error(
          std::string(operation) + ": " +
          cudaGetErrorString(status));
    }
  }

  __global__ void thinningKernel(
      Query const* queries, std::size_t count,
      EmThinningResult* results) {
    auto const index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x +
        threadIdx.x;
    if (index >= count) {
      return;
    }
    auto const query = queries[index];
    results[index] = applyEmThinning(
        query.config, query.parent_energy_GeV,
        query.parent_weight, query.first_energy_GeV,
        query.second_energy_GeV, query.first_uniform,
        query.second_uniform);
  }

} // namespace

int main() {
  int device_count = 0;
  auto const device_status = cudaGetDeviceCount(&device_count);
  if (device_status != cudaSuccess || device_count == 0) {
    std::cerr << "SKIP: no CUDA device is accessible: "
              << cudaGetErrorString(device_status) << '\n';
    return 77;
  }

  try {
    EmThinningConfig const disabled{100., 1000., 0, 1};
    EmThinningConfig const hillas{100., 1000., 1, 1};
    EmThinningConfig const limited{100., 15., 1, 1};

    auto result = applyEmThinning(
        disabled, 10., 2., 7., 3., 0.1, 0.9);
    require(
        result.status == EmThinningStatus::NotApplied &&
            result.keep_mask == 0x3U &&
            result.first_weight == 2. &&
            result.second_weight == 2.,
        "disabled thinning changed a final state");

    result = applyEmThinning(
        hillas, 101., 2., 7., 3., 0.1, 0.9);
    require(
        result.status == EmThinningStatus::NotApplied,
        "above-threshold parent was thinned");
    result = applyEmThinning(
        hillas, 10., 1000., 7., 3., 0.1, 0.9);
    require(
        result.status == EmThinningStatus::NotApplied,
        "maximum-weight parent was thinned");

    result = applyEmThinning(
        hillas, 10., 2., 7., 3., 0.69, 0.9);
    require(
        result.status == EmThinningStatus::Hillas &&
            result.keep_mask == 0x1U,
        "Hillas thinning did not retain child zero");
    requireNear(
        result.first_weight, 2. / 0.7, 1.e-15,
        "Hillas child-zero weight differs");
    result = applyEmThinning(
        hillas, 10., 2., 7., 3., 0.71, 0.1);
    require(
        result.status == EmThinningStatus::Hillas &&
            result.keep_mask == 0x2U,
        "Hillas thinning did not retain child one");
    requireNear(
        result.second_weight, 2. / 0.3, 1.e-15,
        "Hillas child-one weight differs");

    result = applyEmThinning(
        limited, 10., 10., 9., 1., 0.5, 0.5);
    require(
        result.status == EmThinningStatus::Statistical &&
            result.keep_mask == 0x3U,
        "weight-limited statistical thinning branch differs");
    requireNear(
        result.first_weight, 10. / 0.9, 1.e-14,
        "statistical first weight differs");
    requireNear(
        result.second_weight, 15., 1.e-14,
        "statistical second weight differs");

    auto multithin = limited;
    multithin.erase_zero_weight = 0;
    result = applyEmThinning(
        multithin, 10., 10., 9., 1.,
        0.999999, 0.999999);
    require(
        result.status == EmThinningStatus::Statistical &&
            result.keep_mask == 0x3U &&
            result.first_weight == 0. &&
            result.second_weight == 0.,
        "multithin did not retain zero-weight children");

    result = applyEmThinning(
        hillas, 10., 2., 0., 10., 0.5, 0.5);
    require(
        result.status == EmThinningStatus::InvalidInput,
        "zero-energy child was accepted");

    std::vector<Query> queries;
    queries.reserve(8192);
    for (std::size_t index = 0; index < 8192; ++index) {
      auto const first_energy =
          0.01 + 9.98 * ((index % 257) + 0.5) / 257.;
      auto const second_energy = 10. - first_energy;
      auto const first_uniform =
          ((index * 73) % 8192 + 0.5) / 8192.;
      auto const second_uniform =
          ((index * 997) % 8192 + 0.5) / 8192.;
      queries.push_back(
          {index % 2 == 0 ? hillas : limited,
           10., index % 11 + 1., first_energy,
           second_energy, first_uniform, second_uniform});
    }

    Query* device_queries = nullptr;
    EmThinningResult* device_results = nullptr;
    checkCuda(
        cudaMalloc(
            reinterpret_cast<void**>(&device_queries),
            queries.size() * sizeof(Query)),
        "allocate thinning queries");
    checkCuda(
        cudaMalloc(
            reinterpret_cast<void**>(&device_results),
            queries.size() * sizeof(EmThinningResult)),
        "allocate thinning results");
    try {
      checkCuda(
          cudaMemcpy(
              device_queries, queries.data(),
              queries.size() * sizeof(Query),
              cudaMemcpyHostToDevice),
          "upload thinning queries");
      thinningKernel<<<
          static_cast<unsigned int>((queries.size() + 255) / 256),
          256>>>(
          device_queries, queries.size(), device_results);
      checkCuda(
          cudaGetLastError(), "launch thinning kernel");
      std::vector<EmThinningResult> device(queries.size());
      checkCuda(
          cudaMemcpy(
              device.data(), device_results,
              device.size() * sizeof(EmThinningResult),
              cudaMemcpyDeviceToHost),
          "download thinning results");
      for (std::size_t index = 0; index < queries.size();
           ++index) {
        auto const& query = queries[index];
        auto const host = applyEmThinning(
            query.config, query.parent_energy_GeV,
            query.parent_weight, query.first_energy_GeV,
            query.second_energy_GeV, query.first_uniform,
            query.second_uniform);
        require(
            device[index].status == host.status &&
                device[index].keep_mask == host.keep_mask &&
                device[index].first_weight ==
                    host.first_weight &&
                device[index].second_weight ==
                    host.second_weight,
            "host/device thinning result differs");
      }
    } catch (...) {
      cudaFree(device_results);
      cudaFree(device_queries);
      throw;
    }
    cudaFree(device_results);
    cudaFree(device_queries);

    // Deterministic midpoint quadrature checks the two independent
    // statistical acceptances without relying on a noisy PRNG confidence
    // interval.
    constexpr std::size_t Ensemble = 200000;
    double first_weight_sum = 0.;
    double second_weight_sum = 0.;
    for (std::size_t index = 0; index < Ensemble; ++index) {
      auto const first_uniform =
          (index + 0.5) / Ensemble;
      auto const second_uniform =
          ((index * 7919) % Ensemble + 0.5) / Ensemble;
      auto const sample = applyEmThinning(
          limited, 10., 10., 9., 1.,
          first_uniform, second_uniform);
      first_weight_sum += sample.first_weight;
      second_weight_sum += sample.second_weight;
    }
    requireNear(
        first_weight_sum / Ensemble, 10., 2.e-4,
        "statistical first-child mean weight is biased");
    requireNear(
        second_weight_sum / Ensemble, 10., 2.e-4,
        "statistical second-child mean weight is biased");

    std::cout << "CUDA EM thinning primitive passed " << checks
              << " checks including " << queries.size()
              << " host/device cases and " << Ensemble
              << " statistical samples\n";
  } catch (std::exception const& error) {
    std::cerr << "CUDA EM thinning primitive failed after "
              << checks << " checks: " << error.what() << '\n';
    return 1;
  }
  return 0;
}

/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include <cuda_runtime.h>

#include <corsika/gpu/em/UniformMagneticField.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

  using namespace corsika::gpu::em;

  struct Query {
    EmParticleState particle{};
    double mass_GeV{};
    double charge_number{};
    double field_T[3]{};
    double distance_m{};
    double sphere_center_m[3]{};
    double sphere_radius_m{100.};
    double maximum_intersection_distance_m{1000.};
  };

  struct Result {
    MagneticStepLimitResult limit{};
    MagneticAdvanceResult advance{};
    MagneticSphereIntersectionResult intersection{};
  };

  std::size_t checks = 0;

  void require(bool condition, std::string const& message) {
    ++checks;
    if (!condition) {
      throw std::runtime_error(message);
    }
  }

  void requireNear(double actual, double expected,
                   double relative_tolerance,
                   std::string const& message) {
    ++checks;
    auto const scale =
        std::max({1.e-300, std::abs(actual), std::abs(expected)});
    if (std::abs(actual - expected) >
        relative_tolerance * scale) {
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

  __global__ void evaluateMagneticKernel(
      Query const* queries, std::size_t count, Result* results) {
    auto const index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x +
        threadIdx.x;
    if (index < count) {
      auto const query = queries[index];
      results[index].limit = maximumUniformMagneticStep(
          query.particle, query.mass_GeV,
          query.charge_number, query.field_T);
      results[index].advance = advanceUniformMagneticField(
          query.particle, query.mass_GeV,
          query.charge_number, query.field_T,
          query.distance_m);
      results[index].intersection =
          intersectUniformMagneticSphere(
              query.particle, query.mass_GeV,
              query.charge_number, query.field_T,
              query.sphere_center_m, query.sphere_radius_m,
              query.maximum_intersection_distance_m);
    }
  }

} // namespace

int main() {
  int device_count = 0;
  auto const status = cudaGetDeviceCount(&device_count);
  if (status != cudaSuccess || device_count == 0) {
    std::cerr << "SKIP: no CUDA device is accessible\n";
    return 77;
  }
  try {
    std::vector<Query> queries;
    for (double energy : {0.001, 1., 1.e3, 1.e8}) {
      for (double charge : {-1., 1.}) {
        Query query{};
        query.particle.pid =
            charge < 0.
                ? static_cast<std::int32_t>(EmPid::Electron)
                : static_cast<std::int32_t>(EmPid::Positron);
        query.particle.energy_GeV = energy;
        query.particle.position_m[0] = 12.;
        query.particle.position_m[1] = -3.;
        query.particle.position_m[2] = 8.;
        query.particle.direction[0] = 0.6;
        query.particle.direction[1] = 0.;
        query.particle.direction[2] = 0.8;
        query.particle.time_s = 4.e-6;
        query.mass_GeV = 0.0005109989461;
        query.charge_number = charge;
        query.field_T[1] = 5.e-5;
        query.distance_m =
            energy < 1. ? 2. : 200.;
        queries.push_back(query);
      }
    }
    auto parallel = queries.front();
    parallel.particle.direction[0] = 0.;
    parallel.particle.direction[1] = 1.;
    parallel.particle.direction[2] = 0.;
    queries.push_back(parallel);
    auto zero_field = queries.front();
    zero_field.field_T[1] = 0.;
    queries.push_back(zero_field);
    for (double charge : {-1., 1.}) {
      Query sphere{};
      sphere.particle.pid =
          charge < 0.
              ? static_cast<std::int32_t>(EmPid::Electron)
              : static_cast<std::int32_t>(EmPid::Positron);
      sphere.particle.energy_GeV = 1.;
      sphere.particle.position_m[2] = 6371200. + 100.;
      sphere.particle.direction[2] = -1.;
      sphere.mass_GeV = 0.0005109989461;
      sphere.charge_number = charge;
      sphere.field_T[1] = 5.e-5;
      sphere.distance_m = 100.;
      sphere.sphere_radius_m = 6371200.;
      sphere.maximum_intersection_distance_m =
          maximumUniformMagneticStep(
              sphere.particle, sphere.mass_GeV,
              sphere.charge_number, sphere.field_T)
              .distance_m;
      queries.push_back(sphere);
    }

    Query* device_queries = nullptr;
    Result* device_results = nullptr;
    checkCuda(
        cudaMalloc(
            reinterpret_cast<void**>(&device_queries),
            queries.size() * sizeof(Query)),
        "allocate magnetic queries");
    checkCuda(
        cudaMalloc(
            reinterpret_cast<void**>(&device_results),
            queries.size() * sizeof(Result)),
        "allocate magnetic results");
    try {
      checkCuda(
          cudaMemcpy(
              device_queries, queries.data(),
              queries.size() * sizeof(Query),
              cudaMemcpyHostToDevice),
          "upload magnetic queries");
      evaluateMagneticKernel<<<1, 32>>>(
          device_queries, queries.size(), device_results);
      checkCuda(cudaGetLastError(),
                "launch magnetic kernel");
      std::vector<Result> results(queries.size());
      checkCuda(
          cudaMemcpy(
              results.data(), device_results,
              results.size() * sizeof(Result),
              cudaMemcpyDeviceToHost),
          "download magnetic results");

      for (std::size_t index = 0; index < queries.size();
           ++index) {
        auto const host_limit =
            maximumUniformMagneticStep(
                queries[index].particle,
                queries[index].mass_GeV,
                queries[index].charge_number,
                queries[index].field_T);
        auto const host_advance =
            advanceUniformMagneticField(
                queries[index].particle,
                queries[index].mass_GeV,
                queries[index].charge_number,
                queries[index].field_T,
                queries[index].distance_m);
        auto const host_intersection =
            intersectUniformMagneticSphere(
                queries[index].particle,
                queries[index].mass_GeV,
                queries[index].charge_number,
                queries[index].field_T,
                queries[index].sphere_center_m,
                queries[index].sphere_radius_m,
                queries[index]
                    .maximum_intersection_distance_m);
        require(results[index].limit.status ==
                    host_limit.status &&
                    results[index].advance.status ==
                        host_advance.status,
                "host/device magnetic status differs");
        require(
            results[index].intersection.status ==
                host_intersection.status,
            "host/device sphere-intersection status differs");
        requireNear(
            results[index].limit.distance_m,
            host_limit.distance_m, 2.e-15,
            "host/device magnetic step limit differs");
        requireNear(
            results[index].advance.chord_length_m,
            host_advance.chord_length_m, 2.e-15,
            "host/device magnetic chord differs");
        if (host_intersection.status ==
            MagneticIntersectionStatus::Success) {
          requireNear(
              results[index].intersection.distance_m,
              host_intersection.distance_m, 2.e-14,
              "host/device sphere-intersection distance differs");
          auto const endpoint = advanceUniformMagneticField(
              queries[index].particle,
              queries[index].mass_GeV,
              queries[index].charge_number,
              queries[index].field_T,
              results[index].intersection.distance_m);
          auto const radius = std::sqrt(
              endpoint.particle.position_m[0] *
                  endpoint.particle.position_m[0] +
              endpoint.particle.position_m[1] *
                  endpoint.particle.position_m[1] +
              endpoint.particle.position_m[2] *
                  endpoint.particle.position_m[2]);
          requireNear(
              radius, queries[index].sphere_radius_m, 2.e-12,
              "magnetic sphere intersection is off the surface");
        }
        for (int axis = 0; axis < 3; ++axis) {
          requireNear(
              results[index].advance.particle.position_m[axis],
              host_advance.particle.position_m[axis], 2.e-15,
              "host/device magnetic position differs");
          requireNear(
              results[index].advance.particle.direction[axis],
              host_advance.particle.direction[axis], 2.e-15,
              "host/device magnetic direction differs");
        }
        auto const direction_norm =
            std::sqrt(
                results[index].advance.particle.direction[0] *
                    results[index].advance.particle.direction[0] +
                results[index].advance.particle.direction[1] *
                    results[index].advance.particle.direction[1] +
                results[index].advance.particle.direction[2] *
                    results[index].advance.particle.direction[2]);
        requireNear(direction_norm, 1., 2.e-15,
                    "magnetic direction is not normalized");
      }
      require(
          results[queries.size() - 4].limit.status ==
                  MagneticStepStatus::Linear &&
              results[queries.size() - 3].limit.status ==
                  MagneticStepStatus::Linear,
          "parallel/zero-field branch is not linear");
      require(
          results[queries.size() - 2].intersection.status ==
                  MagneticIntersectionStatus::Success &&
              results.back().intersection.status ==
                  MagneticIntersectionStatus::Success,
          "curved sphere test did not find both charge crossings");
    } catch (...) {
      cudaFree(device_results);
      cudaFree(device_queries);
      throw;
    }
    cudaFree(device_results);
    cudaFree(device_queries);
    std::cout
        << "GPU uniform magnetic field validation passed: "
        << checks << " checks, " << queries.size()
        << " states\n";
    return 0;
  } catch (std::exception const& error) {
    std::cerr
        << "GPU uniform magnetic field validation failed after "
        << checks << " checks: " << error.what() << '\n';
    return 1;
  }
}

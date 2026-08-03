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
    // Production regression from proton seed 10200251, shower 22.  The
    // electron starts 0.91 m below the USStdBK 11.4 km layer boundary. Its
    // first root lies about 43 micrometres beyond the 0.2 rad magnetic step,
    // so the endpoint is still inside the old layer but falls within the
    // direction-aware ownership guard. This primitive must keep reporting
    // no bounded root; the transport layer handles the guarded transition.
    auto const production_boundary_index = queries.size();
    Query production_boundary{};
    production_boundary.particle.pid =
        static_cast<std::int32_t>(EmPid::Electron);
    production_boundary.particle.energy_GeV =
        0.0011808222196131866;
    production_boundary.particle.position_m[0] =
        -32.82159051860207;
    production_boundary.particle.position_m[1] =
        9.385832804803458;
    production_boundary.particle.position_m[2] =
        6382399.0942817135;
    production_boundary.particle.direction[0] =
        -0.4188050119591029;
    production_boundary.particle.direction[1] =
        0.05284253648793534;
    production_boundary.particle.direction[2] =
        0.9065373838377857;
    production_boundary.mass_GeV = 0.0005109989461;
    production_boundary.charge_number = -1.;
    production_boundary.field_T[0] =
        2.5067327193094893e-05;
    production_boundary.field_T[1] =
        -1.10186120465756e-06;
    production_boundary.field_T[2] =
        -5.1031311292104371e-05;
    production_boundary.sphere_radius_m = 6382400.;
    production_boundary.maximum_intersection_distance_m =
        maximumUniformMagneticStep(
            production_boundary.particle,
            production_boundary.mass_GeV,
            production_boundary.charge_number,
            production_boundary.field_T, 0.2)
            .distance_m;
    production_boundary.distance_m =
        production_boundary.maximum_intersection_distance_m;
    queries.push_back(production_boundary);
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
        auto const host_device_tolerance =
            index == production_boundary_index ? 2.e-13 : 2.e-15;
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
            host_limit.distance_m, host_device_tolerance,
            "host/device magnetic step limit differs");
        requireNear(
            results[index].advance.chord_length_m,
            host_advance.chord_length_m, host_device_tolerance,
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
              host_advance.particle.position_m[axis],
              host_device_tolerance,
              "host/device magnetic position differs");
          requireNear(
              results[index].advance.particle.direction[axis],
              host_advance.particle.direction[axis],
              host_device_tolerance,
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
      auto const& production_endpoint =
          results[production_boundary_index].advance.particle;
      auto const production_endpoint_radius = std::sqrt(
          production_endpoint.position_m[0] *
                  production_endpoint.position_m[0] +
              production_endpoint.position_m[1] *
                  production_endpoint.position_m[1] +
              production_endpoint.position_m[2] *
                  production_endpoint.position_m[2]);
      require(
          results[production_boundary_index]
                  .intersection.status ==
              MagneticIntersectionStatus::NoForwardIntersection &&
              production_boundary.sphere_radius_m >=
                  production_endpoint_radius &&
              production_boundary.sphere_radius_m -
                      production_endpoint_radius <=
                  1.e-4,
          "production regression no longer ends inside the boundary guard");
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

/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include <cuda_runtime.h>

#include <PROPOSAL/Constants.h>
#include <PROPOSAL/crosssection/parametrization/EpairProduction.h>
#include <PROPOSAL/medium/Medium.h>
#include <PROPOSAL/particle/Particle.h>

#include <corsika/gpu/em/EpairLpm.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <iomanip>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

  using namespace corsika::gpu::em;

  struct LpmQuery {
    double energy_MeV{};
    double v{};
    double rho_squared{};
    double density_g_per_cm3{};
  };

  std::size_t checks = 0;

  void require(bool condition, std::string const& message) {
    ++checks;
    if (!condition) {
      throw std::runtime_error(message);
    }
  }

  void checkCuda(cudaError_t status, char const* operation) {
    if (status != cudaSuccess) {
      throw std::runtime_error(
          std::string(operation) + ": " +
          cudaGetErrorString(status));
    }
  }

  BremsLpmSnapshot makeSnapshot(
      PROPOSAL::Medium const& medium) {
    BremsLpmSnapshot snapshot{};
    snapshot.baseline_mass_density_g_per_cm3 =
        medium.GetMassDensity();
    snapshot.molecular_density_per_cm3 =
        medium.GetMolDensity();
    snapshot.sum_charge = medium.GetSumCharge();
    // EpairLPM reconstructs its own characteristic energy, but the shared
    // snapshot invariant deliberately still requires the Brems field.
    snapshot.e_lpm_MeV = 1.;
    snapshot.lepton_mass_MeV = PROPOSAL::EMinusDef().mass;
    snapshot.electron_mass_MeV = PROPOSAL::ME;
    snapshot.muon_mass_MeV = PROPOSAL::MMU;
    snapshot.classical_electron_radius_cm = PROPOSAL::RE;
    snapshot.fine_structure_constant = PROPOSAL::ALPHA;
    auto const components = medium.GetComponents();
    snapshot.component_count =
        static_cast<std::uint32_t>(components.size());
    for (std::size_t index = 0; index < components.size();
         ++index) {
      snapshot.components[index] = {
          static_cast<std::uint64_t>(
              components[index].GetHash()),
          components[index].GetNucCharge(),
          components[index].GetAtomicNum(),
          components[index].GetLogConstant()};
    }
    return snapshot;
  }

  __global__ void evaluateLpmKernel(
      BremsLpmSnapshot snapshot, LpmQuery const* queries,
      std::size_t count, EpairLpmResult* results) {
    auto const index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x +
        threadIdx.x;
    if (index < count) {
      auto const query = queries[index];
      results[index] = epairLpmSuppressionFactor(
          snapshot, query.energy_MeV, query.v,
          query.rho_squared, query.density_g_per_cm3);
    }
  }

} // namespace

int main() {
  int device_count = 0;
  auto const status = cudaGetDeviceCount(&device_count);
  if (status != cudaSuccess || device_count == 0) {
    std::cerr << "SKIP: no CUDA device is accessible: "
              << cudaGetErrorString(status) << '\n';
    return 77;
  }

  try {
    PROPOSAL::Air const medium;
    auto const snapshot = makeSnapshot(medium);
    PROPOSAL::crosssection::EpairLPM reference(
        PROPOSAL::EMinusDef(), medium);

    std::vector<LpmQuery> queries;
    for (double energy :
         {10., 1.e3, 1.e6, 1.e9, 1.e12, 1.e14}) {
      for (double v :
           {1.e-4, 1.e-2, 0.1, 0.5, 0.9}) {
        for (double rho_squared :
             {0., 1.e-4, 0.01, 0.25, 0.81}) {
          for (double density_ratio :
               {1.e-7, 1.e-4, 1.e-2, 1., 10.}) {
            queries.push_back(
                {energy, v, rho_squared,
                 medium.GetMassDensity() * density_ratio});
          }
        }
      }
    }

    LpmQuery* device_queries = nullptr;
    EpairLpmResult* device_results = nullptr;
    checkCuda(
        cudaMalloc(
            reinterpret_cast<void**>(&device_queries),
            queries.size() * sizeof(LpmQuery)),
        "allocate Epair LPM queries");
    checkCuda(
        cudaMalloc(
            reinterpret_cast<void**>(&device_results),
            queries.size() * sizeof(EpairLpmResult)),
        "allocate Epair LPM results");
    try {
      checkCuda(
          cudaMemcpy(
              device_queries, queries.data(),
              queries.size() * sizeof(LpmQuery),
              cudaMemcpyHostToDevice),
          "upload Epair LPM queries");
      auto const blocks = static_cast<unsigned int>(
          (queries.size() + 255) / 256);
      evaluateLpmKernel<<<blocks, 256>>>(
          snapshot, device_queries, queries.size(),
          device_results);
      checkCuda(
          cudaGetLastError(), "launch Epair LPM oracle kernel");
      std::vector<EpairLpmResult> device(queries.size());
      checkCuda(
          cudaMemcpy(
              device.data(), device_results,
              device.size() * sizeof(EpairLpmResult),
              cudaMemcpyDeviceToHost),
          "download Epair LPM results");

      double maximum_relative_error = 0.;
      for (std::size_t index = 0; index < queries.size();
           ++index) {
        auto const& query = queries[index];
        auto const density_correction =
            query.density_g_per_cm3 /
            medium.GetMassDensity();
        auto const beta =
            query.v * query.v / (2. * (1. - query.v));
        auto const xi =
            PROPOSAL::EMinusDef().mass *
            PROPOSAL::EMinusDef().mass /
            (PROPOSAL::ME * PROPOSAL::ME) *
            query.v * query.v / 4. *
            (1. - query.rho_squared) /
            (1. - query.v);
        auto expected = reference.suppression_factor(
            query.energy_MeV, query.v, query.rho_squared,
            beta, xi, density_correction);
        if (expected > 1. - 1.e-6) {
          expected = 1.;
        }
        auto const host = epairLpmSuppressionFactor(
            snapshot, query.energy_MeV, query.v,
            query.rho_squared, query.density_g_per_cm3);
        require(
            host.status == EpairLpmStatus::Success &&
                device[index].status == EpairLpmStatus::Success,
            "valid Epair LPM query was rejected");
        auto const scale =
            std::max(1.e-300, std::abs(expected));
        auto const host_error =
            std::abs(host.survival_probability - expected) /
            scale;
        auto const device_error =
            std::abs(
                device[index].survival_probability - expected) /
            scale;
        maximum_relative_error =
            std::max(
                {maximum_relative_error, host_error,
                 device_error});
        if (!(host_error < 3.e-13) ||
            !(device_error < 5.e-7)) {
          std::cerr << std::setprecision(17)
                    << "Epair LPM diagnostic expected=" << expected
                    << ", host=" << host.survival_probability
                    << ", device="
                    << device[index].survival_probability
                    << ", host_error=" << host_error
                    << ", device_error=" << device_error << '\n';
          throw std::runtime_error(
              "Epair LPM formula differs from PROPOSAL at E=" +
              std::to_string(query.energy_MeV) +
              ", v=" + std::to_string(query.v) +
              ", rho2=" +
              std::to_string(query.rho_squared) +
              ", density=" +
              std::to_string(query.density_g_per_cm3) +
              ": expected=" + std::to_string(expected) +
              ", host=" +
              std::to_string(host.survival_probability) +
              ", device=" +
              std::to_string(
                  device[index].survival_probability) +
              ", host_error=" +
              std::to_string(host_error) +
              ", device_error=" +
              std::to_string(device_error));
        }
        checks += 2;
      }

      auto const invalid_density = epairLpmSuppressionFactor(
          snapshot, 1.e9, 0.1, 0.25, 0.);
      require(
          invalid_density.status ==
              EpairLpmStatus::InvalidInput,
          "zero-density Epair LPM query was accepted");
      auto const invalid_rho = epairLpmSuppressionFactor(
          snapshot, 1.e9, 0.1, 1., medium.GetMassDensity());
      require(
          invalid_rho.status == EpairLpmStatus::InvalidInput,
          "unit-rho Epair LPM query was accepted");

      std::cout
          << "CUDA Epair LPM oracle passed " << checks
          << " checks for " << queries.size()
          << " energy/v/rho/density points; max relative error="
          << maximum_relative_error << '\n';
    } catch (...) {
      cudaFree(device_results);
      cudaFree(device_queries);
      throw;
    }
    cudaFree(device_results);
    cudaFree(device_queries);
  } catch (std::exception const& error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
  }
  return 0;
}

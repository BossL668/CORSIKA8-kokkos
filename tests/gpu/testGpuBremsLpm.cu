/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include <cuda_runtime.h>

#include <PROPOSAL/Constants.h>
#include <PROPOSAL/crosssection/parametrization/Bremsstrahlung.h>
#include <PROPOSAL/math/Integral.h>
#include <PROPOSAL/medium/Medium.h>
#include <PROPOSAL/particle/Particle.h>

#include <corsika/gpu/em/BremsLpm.hpp>

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

  struct LpmQuery {
    std::uint64_t component_hash{};
    double energy_MeV{};
    double v{};
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
    PROPOSAL::crosssection::BremsElectronScreening parametrization;
    PROPOSAL::EMinusDef const electron;
    constexpr double UpperEnergyMeV = 1.e14;
    PROPOSAL::Integral integral(
        PROPOSAL::IROMB, PROPOSAL::IMAXS, PROPOSAL::IPREC);
    double sum = 0.;
    auto const components = medium.GetComponents();
    for (auto const& component : components) {
      auto const limits = parametrization.GetKinematicLimits(
          electron, component, UpperEnergyMeV);
      auto const contribution = integral.Integrate(
          limits.v_min, limits.v_max,
          [&](double value) {
            return parametrization.FunctionToDEdxIntegral(
                electron, component, UpperEnergyMeV, value);
          },
          2.);
      auto const weight =
          medium.GetSumNucleons() /
          (component.GetAtomInMolecule() *
           component.GetAtomicNum());
      sum += contribution / weight;
    }
    sum *= medium.GetMassDensity();
    auto e_lpm = PROPOSAL::ALPHA * electron.mass;
    e_lpm *=
        2. * e_lpm /
        (PROPOSAL::PI * PROPOSAL::ME * PROPOSAL::RE * sum);

    BremsLpmSnapshot snapshot{};
    snapshot.baseline_mass_density_g_per_cm3 =
        medium.GetMassDensity();
    snapshot.molecular_density_per_cm3 =
        medium.GetMolDensity();
    snapshot.sum_charge = medium.GetSumCharge();
    snapshot.e_lpm_MeV = e_lpm;
    snapshot.lepton_mass_MeV = electron.mass;
    snapshot.electron_mass_MeV = PROPOSAL::ME;
    snapshot.muon_mass_MeV = PROPOSAL::MMU;
    snapshot.classical_electron_radius_cm = PROPOSAL::RE;
    snapshot.fine_structure_constant = PROPOSAL::ALPHA;
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
      std::size_t count, BremsLpmResult* results) {
    auto const index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x +
        threadIdx.x;
    if (index < count) {
      auto const query = queries[index];
      results[index] = bremsLpmSuppressionFactor(
          snapshot, query.component_hash, query.energy_MeV,
          query.v, query.density_g_per_cm3);
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
    PROPOSAL::crosssection::BremsElectronScreening parametrization;
    PROPOSAL::crosssection::BremsLPM reference(
        PROPOSAL::EMinusDef(), medium, parametrization);
    auto const components = medium.GetComponents();

    std::vector<LpmQuery> queries;
    for (auto const& component : components) {
      for (double energy :
           {1., 1.e3, 1.e6, 1.e9, 1.e12, 1.e14}) {
        for (double v :
             {1.e-6, 1.e-4, 1.e-2, 0.1, 0.5, 0.9}) {
          for (double density_ratio :
               {1.e-7, 1.e-4, 1.e-2, 1., 10.}) {
            queries.push_back(
                {static_cast<std::uint64_t>(
                     component.GetHash()),
                 energy, v,
                 medium.GetMassDensity() * density_ratio});
          }
        }
      }
    }

    LpmQuery* device_queries = nullptr;
    BremsLpmResult* device_results = nullptr;
    checkCuda(
        cudaMalloc(
            reinterpret_cast<void**>(&device_queries),
            queries.size() * sizeof(LpmQuery)),
        "allocate bremsstrahlung LPM queries");
    checkCuda(
        cudaMalloc(
            reinterpret_cast<void**>(&device_results),
            queries.size() * sizeof(BremsLpmResult)),
        "allocate bremsstrahlung LPM results");
    try {
      checkCuda(
          cudaMemcpy(
              device_queries, queries.data(),
              queries.size() * sizeof(LpmQuery),
              cudaMemcpyHostToDevice),
          "upload bremsstrahlung LPM queries");
      auto const blocks = static_cast<unsigned int>(
          (queries.size() + 255) / 256);
      evaluateLpmKernel<<<blocks, 256>>>(
          snapshot, device_queries, queries.size(),
          device_results);
      checkCuda(
          cudaGetLastError(),
          "launch bremsstrahlung LPM oracle kernel");
      std::vector<BremsLpmResult> device(queries.size());
      checkCuda(
          cudaMemcpy(
              device.data(), device_results,
              device.size() * sizeof(BremsLpmResult),
              cudaMemcpyDeviceToHost),
          "download bremsstrahlung LPM results");

      double maximum_relative_error = 0.;
      for (std::size_t index = 0; index < queries.size();
           ++index) {
        auto const& query = queries[index];
        auto const component = std::find_if(
            components.begin(), components.end(),
            [&](PROPOSAL::Component const& candidate) {
              return static_cast<std::uint64_t>(
                         candidate.GetHash()) ==
                     query.component_hash;
            });
        require(component != components.end(),
                "reference bremsstrahlung component is missing");
        auto const density_correction =
            query.density_g_per_cm3 /
            medium.GetMassDensity();
        auto const expected = reference.suppression_factor(
            query.energy_MeV, query.v, *component,
            density_correction);
        auto const host = bremsLpmSuppressionFactor(
            snapshot, query.component_hash, query.energy_MeV,
            query.v, query.density_g_per_cm3);
        require(
            host.status == BremsLpmStatus::Success &&
                device[index].status ==
                    BremsLpmStatus::Success,
            "valid bremsstrahlung LPM query was rejected");
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
        require(
            host_error < 2.e-13,
            "host bremsstrahlung LPM formula differs from PROPOSAL");
        require(
            device_error < 3.e-12,
            "CUDA bremsstrahlung LPM formula differs from PROPOSAL");
      }

      auto const missing = bremsLpmSuppressionFactor(
          snapshot, 0xdeadbeefULL, 1.e12, 0.1,
          medium.GetMassDensity());
      require(
          missing.status ==
              BremsLpmStatus::ComponentNotFound,
          "unknown bremsstrahlung LPM component was accepted");
      auto const invalid_v = bremsLpmSuppressionFactor(
          snapshot, snapshot.components[0].component_hash,
          1.e12, 1., medium.GetMassDensity());
      require(
          invalid_v.status == BremsLpmStatus::InvalidInput,
          "unit bremsstrahlung loss fraction was accepted");

      std::cout
          << "CUDA bremsstrahlung LPM oracle passed "
          << checks << " checks for " << queries.size()
          << " energy/v/density/component points; max relative "
             "error="
          << maximum_relative_error
          << ", eLPM=" << snapshot.e_lpm_MeV << " MeV\n";
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

/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include <cuda_runtime.h>

#include <PROPOSAL/Constants.h>
#include <PROPOSAL/medium/Medium.h>
#include <PROPOSAL/particle/Particle.h>
#include <PROPOSAL/scattering/ScatteringFactory.h>
#include <PROPOSAL/scattering/multiple_scattering/Coefficients.h>

#include <corsika/gpu/em/MoliereScattering.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

  using namespace corsika::gpu::em;

  struct Query {
    double grammage_g_per_cm2{};
    double initial_energy_MeV{};
    double final_energy_MeV{};
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

  void checkCuda(cudaError_t status, char const* operation) {
    if (status != cudaSuccess) {
      throw std::runtime_error(
          std::string(operation) + ": " +
          cudaGetErrorString(status));
    }
  }

  MoliereSnapshot makeSnapshot(
      PROPOSAL::ParticleDef const& particle,
      PROPOSAL::Medium const& medium) {
    auto const components = medium.GetComponents();
    if (components.empty() ||
        components.size() > MaxMoliereComponents) {
      throw std::runtime_error(
          "test medium has unsupported Moliere composition");
    }
    MoliereSnapshot snapshot{};
    snapshot.particle_mass_MeV = particle.mass;
    snapshot.electron_mass_MeV = PROPOSAL::ME;
    snapshot.fine_structure_constant = PROPOSAL::ALPHA;
    snapshot.avogadro_per_mol = PROPOSAL::NA;
    snapshot.hbar_MeV_s = PROPOSAL::HBAR;
    snapshot.speed_of_light_cm_per_s = PROPOSAL::SPEED;
    snapshot.euler_mascheroni =
        PROPOSAL::EULER_MASCHERONI;
    snapshot.component_count =
        static_cast<std::uint32_t>(components.size());

    std::vector<double> mass_weights(components.size());
    double atomic_mass_sum = 0.;
    for (auto const& component : components) {
      atomic_mass_sum +=
          component.GetAtomInMolecule() *
          component.GetAtomicNum();
    }
    double average_atomic_mass = 0.;
    double average_charge_squared = 0.;
    double weight_zz_sum = 0.;
    for (std::size_t index = 0; index < components.size();
         ++index) {
      auto const& component = components[index];
      auto const weight =
          component.GetAtomInMolecule() *
          component.GetAtomicNum() / atomic_mass_sum;
      mass_weights[index] = weight;
      average_atomic_mass +=
          weight * component.GetAtomicNum();
      auto const charge = component.GetNucCharge();
      auto const weight_zz = weight * charge * charge;
      weight_zz_sum += weight_zz;
      average_charge_squared +=
          particle.mass == PROPOSAL::ME
              ? weight * charge * (charge + 1.)
              : weight_zz;
      auto const chi_0 =
          PROPOSAL::ME * PROPOSAL::ALPHA *
          std::pow(
              charge * 128. /
                  (9. * PROPOSAL::PI * PROPOSAL::PI),
              1. / 3.);
      snapshot.components[index] = {
          charge,
          weight_zz,
          chi_0 * chi_0,
          3.76 * PROPOSAL::ALPHA * PROPOSAL::ALPHA *
              charge * charge};
      if (index > 0 &&
          mass_weights[index] > mass_weights[index - 1]) {
        snapshot.maximum_weight_index =
            static_cast<std::uint32_t>(index);
      }
    }
    snapshot.z_squared_over_a_average =
        average_charge_squared / average_atomic_mass;
    snapshot.inverse_weight_zz_sum = 1. / weight_zz_sum;
    std::copy_n(
        PROPOSAL::c1, MoliereSeriesCoefficientCount,
        snapshot.c1);
    std::copy_n(
        PROPOSAL::c2, MoliereSeriesCoefficientCount,
        snapshot.c2);
    std::copy_n(
        PROPOSAL::c2large,
        MoliereLargeSeriesCoefficientCount,
        snapshot.c2_large);
    std::copy_n(
        PROPOSAL::s2large,
        MoliereLargeSeriesCoefficientCount,
        snapshot.s2_large);
    std::copy_n(
        PROPOSAL::C1large,
        MoliereLargeIntegralCoefficientCount,
        snapshot.C1_large);
    return snapshot;
  }

  __global__ void evaluateMoliereKernel(
      MoliereSnapshot snapshot,
      MoliereInterpolationView interpolation,
      Query const* queries,
      std::size_t count, MoliereAngleResult* results) {
    auto const index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x +
        threadIdx.x;
    if (index < count) {
      auto const query = queries[index];
      results[index] = sampleMoliereScatteringAngle2D(
          snapshot, interpolation,
          query.grammage_g_per_cm2,
          query.initial_energy_MeV,
          query.final_energy_MeV,
          query.first_uniform, query.second_uniform);
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
    PROPOSAL::EMinusDef const electron;
    auto const snapshot = makeSnapshot(electron, medium);
    auto const cache_path =
        std::filesystem::temp_directory_path() /
        ("c8_moliere_cache_test_" +
         std::to_string(
             reinterpret_cast<std::uintptr_t>(
                 &snapshot)) +
         ".bin");
    std::error_code cache_error;
    std::filesystem::remove(cache_path, cache_error);
    auto const interpolation =
        loadOrMakeMoliereInterpolationTable(
            snapshot, cache_path);
    require(
        std::filesystem::is_regular_file(cache_path),
        "Moliere cold cache was not created");
    auto const cached_interpolation =
        loadOrMakeMoliereInterpolationTable(
            snapshot, cache_path);
    require(
        std::memcmp(
            &interpolation, &cached_interpolation,
            sizeof(interpolation)) == 0,
        "Moliere hot cache changed interpolation data");
    {
      std::fstream cache(
          cache_path,
          std::ios::binary | std::ios::in |
              std::ios::out);
      require(
          static_cast<bool>(cache),
          "could not open Moliere cache for corruption test");
      cache.seekg(-1, std::ios::end);
      char byte = 0;
      cache.read(&byte, 1);
      byte ^= static_cast<char>(0x5a);
      cache.seekp(-1, std::ios::end);
      cache.write(&byte, 1);
    }
    auto const repaired_interpolation =
        loadOrMakeMoliereInterpolationTable(
            snapshot, cache_path);
    require(
        std::memcmp(
            &interpolation, &repaired_interpolation,
            sizeof(interpolation)) == 0,
        "Moliere corrupt cache was not rebuilt exactly");
    std::filesystem::remove(cache_path, cache_error);
    auto const host_interpolation =
        makeMoliereInterpolationView(
            interpolation.polynomials.data(),
            interpolation.initial_guess_delta.data());
    moliere_detail::DistributionState median_state{};
    require(
        moliere_detail::buildDistribution(
            snapshot, 1., 1.e3, median_state) ==
            MoliereStatus::Success,
        "could not build the Moliere median test distribution");
    auto const median =
        moliere_detail::sampleOneDimensional(
            snapshot, host_interpolation, median_state, 0.5);
    require(
        median.status == MoliereStatus::Success &&
            median.iterations == 0 &&
            median.angle_rad == 0.,
        "the exact Moliere median did not return zero");
    auto direct = PROPOSAL::make_multiple_scattering(
        PROPOSAL::MultipleScatteringType::Moliere,
        electron, medium);
    auto interpolated = PROPOSAL::make_multiple_scattering(
        PROPOSAL::MultipleScatteringType::MoliereInterpol,
        electron, medium);

    std::vector<Query> queries;
    std::array<std::array<double, 2>, 15> const
        uniform_pairs{{
            {1.e-10, 1. - 1.e-10},
            {1.e-8, 0.999999},
            {1.e-6, 0.001},
            {1.e-4, 0.01},
            {0.001, 0.1},
            {0.01, 0.9927},
            {0.1, 0.927},
            {0.3, 0.781},
            {0.500001, 0.635},
            {0.7, 0.489},
            {0.9, 0.343},
            {0.99, 0.197},
            {0.999, 0.073},
            {0.999999, 1.e-6},
            {1. - 1.e-10, 1.e-10},
        }};
    for (double energy :
         {1.1, 10., 1.e3, 1.e6, 1.e9, 1.e12}) {
      for (double grammage :
           {1.e-8, 1.e-5, 1.e-2, 1., 100.}) {
        for (auto const& uniforms : uniform_pairs) {
          queries.push_back(
              {grammage, energy,
               std::max(electron.mass, energy * 0.9),
               uniforms[0], uniforms[1]});
        }
      }
    }
    // A deterministic distribution-level sample complements the adversarial
    // tail grid above. Pointwise agreement on thousands of random quantiles
    // is stronger than comparing only shower means, which can decorrelate
    // chaotically after an otherwise harmless last-bit direction change.
    std::array<double, 4> const statistical_energies{
        1.1, 1.e3, 1.e6, 1.e9};
    std::array<double, 4> const statistical_grammages{
        1.e-5, 1.e-2, 1., 100.};
    std::uint64_t random_state =
        0x9e3779b97f4a7c15ULL;
    auto next_uniform = [&]() {
      random_state =
          random_state * 6364136223846793005ULL +
          1442695040888963407ULL;
      return (
          static_cast<double>(random_state >> 11) + 0.5) *
          0x1.0p-53;
    };
    for (std::size_t index = 0; index < 4096;
         ++index) {
      auto const energy =
          statistical_energies[
              index % statistical_energies.size()];
      auto const grammage =
          statistical_grammages[
              (index / statistical_energies.size()) %
              statistical_grammages.size()];
      queries.push_back(
          {grammage, energy,
           std::max(electron.mass, energy * 0.9),
           next_uniform(), next_uniform()});
    }

    Query* device_queries = nullptr;
    MoliereAngleResult* device_results = nullptr;
    MoliereCubicPolynomial* device_interpolation = nullptr;
    checkCuda(
        cudaMalloc(
            reinterpret_cast<void**>(&device_queries),
            queries.size() * sizeof(Query)),
        "allocate Moliere queries");
    checkCuda(
        cudaMalloc(
            reinterpret_cast<void**>(&device_results),
            queries.size() * sizeof(MoliereAngleResult)),
        "allocate Moliere results");
      checkCuda(
          cudaMalloc(
              reinterpret_cast<void**>(&device_interpolation),
              sizeof(MoliereInterpolationTable)),
        "allocate Moliere interpolation");
    try {
      checkCuda(
          cudaMemcpy(
              device_queries, queries.data(),
              queries.size() * sizeof(Query),
              cudaMemcpyHostToDevice),
          "upload Moliere queries");
      checkCuda(
          cudaMemcpy(
              device_interpolation,
              &interpolation,
              sizeof(MoliereInterpolationTable),
              cudaMemcpyHostToDevice),
          "upload Moliere interpolation");
      auto const blocks = static_cast<unsigned int>(
          (queries.size() + 255) / 256);
      evaluateMoliereKernel<<<blocks, 256>>>(
          snapshot,
          makeMoliereInterpolationView(
              device_interpolation,
              reinterpret_cast<double const*>(
                  reinterpret_cast<unsigned char const*>(
                      device_interpolation) +
                  offsetof(
                      MoliereInterpolationTable,
                      initial_guess_delta))),
          device_queries, queries.size(),
          device_results);
      checkCuda(cudaGetLastError(),
                "launch Moliere kernel");
      std::vector<MoliereAngleResult> results(
          queries.size());
      checkCuda(
          cudaMemcpy(
              results.data(), device_results,
              results.size() * sizeof(MoliereAngleResult),
              cudaMemcpyDeviceToHost),
          "download Moliere results");

      double maximum_direct_relative_error = 0.;
      double maximum_interpolated_relative_error = 0.;
      std::size_t maximum_interpolated_error_index = 0;
      double maximum_interpolated_gpu_angle = 0.;
      double maximum_interpolated_cpu_angle = 0.;
      std::size_t deflected = 0;
      std::uint64_t total_newton_iterations = 0;
      std::uint32_t maximum_newton_iterations = 0;
      for (std::size_t index = 0; index < queries.size();
           ++index) {
        auto const& query = queries[index];
        auto const reference =
            direct->CalculateScatteringAngle2D(
                query.grammage_g_per_cm2,
                query.initial_energy_MeV,
                query.final_energy_MeV,
                query.first_uniform,
                query.second_uniform);
        auto const production =
            interpolated->CalculateScatteringAngle2D(
                query.grammage_g_per_cm2,
                query.initial_energy_MeV,
                query.final_energy_MeV,
                query.first_uniform,
                query.second_uniform);
        auto const& result = results[index];
        if (reference == 0.) {
          require(
              result.status == MoliereStatus::NoDeflection &&
                  result.angle_rad == 0.,
              "Moliere zero-deflection branch differs");
          continue;
        }
        ++deflected;
        require(
            result.status == MoliereStatus::Success,
            "device Moliere sampling did not converge for query " +
                std::to_string(index) + " (E=" +
                std::to_string(query.initial_energy_MeV) +
                " MeV, X=" +
                std::to_string(query.grammage_g_per_cm2) +
                " g/cm2, u1=" +
                std::to_string(query.first_uniform) +
                ", u2=" +
                std::to_string(query.second_uniform) + ")");
        total_newton_iterations += result.iterations;
        maximum_newton_iterations = std::max(
            maximum_newton_iterations, result.iterations);
        auto const scale =
            std::max(std::abs(reference), 1.e-300);
        maximum_direct_relative_error = std::max(
            maximum_direct_relative_error,
            std::abs(result.angle_rad - reference) / scale);
        auto const production_scale =
            std::max(std::abs(production), 1.e-300);
        auto const interpolated_relative_error =
            std::abs(result.angle_rad - production) /
            production_scale;
        if (interpolated_relative_error >
            maximum_interpolated_relative_error) {
          maximum_interpolated_relative_error =
              interpolated_relative_error;
          maximum_interpolated_error_index = index;
          maximum_interpolated_gpu_angle =
              result.angle_rad;
          maximum_interpolated_cpu_angle = production;
        }
      }
      require(deflected > 0,
              "Moliere test did not exercise a deflection");
      auto const mean_newton_iterations =
          static_cast<double>(total_newton_iterations) /
          static_cast<double>(deflected);
      require(
          mean_newton_iterations <= 3.6,
          "mixture-aware Moliere initial guess regressed its mean iteration budget");
      require(
          maximum_newton_iterations <= 6,
          "mixture-aware Moliere initial guess regressed its maximum iteration budget");
      std::cout
          << "Moliere diagnostic: max direct relative error="
          << maximum_direct_relative_error
          << ", max GPU/CPU interpolated relative error="
          << maximum_interpolated_relative_error
          << ", mean two-axis Newton iterations="
          << mean_newton_iterations
          << ", max two-axis Newton iterations="
          << maximum_newton_iterations << '\n';
      auto const& maximum_error_query =
          queries[maximum_interpolated_error_index];
      std::cout
          << "Moliere worst interpolated query: index="
          << maximum_interpolated_error_index
          << ", E_MeV="
          << maximum_error_query.initial_energy_MeV
          << ", X_g_cm2="
          << maximum_error_query.grammage_g_per_cm2
          << ", u1=" << maximum_error_query.first_uniform
          << ", u2=" << maximum_error_query.second_uniform
          << ", gpu_angle="
          << maximum_interpolated_gpu_angle
          << ", cpu_angle="
          << maximum_interpolated_cpu_angle
          << '\n';
      require(
          maximum_interpolated_relative_error <= 1.e-5,
          "optimized device Moliere exceeds its pointwise PROPOSAL budget");
      // The expanded 1e-10 tail grid reaches a roughly 3.5% difference
      // between PROPOSAL's own analytical and production interpolated
      // parametrizations. This loose cross-parametrization guard is separate
      // from the 1e-5 GPU/CPU MoliereInterpol requirement above.
      require(
          maximum_direct_relative_error <= 4.e-2,
          "interpolated Moliere is inconsistent with analytical Moliere");
      std::cout
          << "GPU Moliere validation passed: " << checks
          << " checks, " << deflected
          << " deflected samples, mean two-axis Newton iterations="
          << mean_newton_iterations
          << ", max two-axis Newton iterations="
          << maximum_newton_iterations
          << ", max direct relative error="
          << maximum_direct_relative_error
          << ", max GPU/CPU interpolated relative error="
          << maximum_interpolated_relative_error << '\n';
    } catch (...) {
      cudaFree(device_interpolation);
      cudaFree(device_results);
      cudaFree(device_queries);
      throw;
    }
    cudaFree(device_interpolation);
    cudaFree(device_results);
    cudaFree(device_queries);
    return 0;
  } catch (std::exception const& error) {
    std::cerr << "GPU Moliere validation failed after "
              << checks << " checks: " << error.what() << '\n';
    return 1;
  }
}

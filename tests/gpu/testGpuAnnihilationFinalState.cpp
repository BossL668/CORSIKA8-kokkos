/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include <cuda_runtime_api.h>

#include <PROPOSAL/medium/Medium.h>
#include <PROPOSAL/particle/Particle.h>
#include <PROPOSAL/secondaries/parametrization/annihilation/HeitlerAnnihilation.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <GpuEmFlatRateTableFixture.hpp>

#include <corsika/gpu/em/CudaBremsFinalState.hpp>
#include <corsika/gpu/em/CudaEmBackend.hpp>
#include <corsika/gpu/em/CudaInteractionSelector.hpp>
#include <corsika/gpu/em/Philox.hpp>
#include <corsika/gpu/em/ProcessCapabilities.hpp>
#include <corsika/gpu/em/tables/RateTable.hpp>

namespace {

  using namespace corsika::gpu::em;
  using namespace corsika::gpu::em::tables;

  std::size_t checks = 0;

  void require(bool condition, std::string const& message) {
    ++checks;
    if (!condition) {
      throw std::runtime_error(message);
    }
  }

  void requireNear(
      double actual, double expected, double relative_tolerance,
      std::string const& message) {
    ++checks;
    auto const scale =
        std::max({1.e-300, std::abs(actual), std::abs(expected)});
    if (std::abs(actual - expected) >
        relative_tolerance * scale) {
      std::ostringstream details;
      details << std::setprecision(17) << message
              << ": actual=" << actual
              << ", expected=" << expected
              << ", relative error="
              << std::abs(actual - expected) / scale;
      throw std::runtime_error(details.str());
    }
  }

  void requireAbsoluteNear(
      double actual, double expected, double tolerance,
      std::string const& message) {
    ++checks;
    if (std::abs(actual - expected) > tolerance) {
      std::ostringstream details;
      details << std::setprecision(17) << message
              << ": actual=" << actual
              << ", expected=" << expected
              << ", absolute error="
              << std::abs(actual - expected);
      throw std::runtime_error(details.str());
    }
  }

  std::filesystem::path temporaryPath() {
    auto const stamp =
        std::chrono::high_resolution_clock::now()
            .time_since_epoch()
            .count();
    return std::filesystem::temp_directory_path() /
           ("c8_gpu_annihilation_final_state_" +
            std::to_string(stamp) + ".c8emrt");
  }

  std::size_t processCount(RateTableSet const& table) {
    std::size_t count = 0;
    for (auto const& particle : table.particles) {
      count += particle.columns.size();
    }
    return count;
  }

  EmInteractionRecord makeAnnihilation(
      std::size_t index, double energy_GeV) {
    EmInteractionRecord interaction{};
    auto& particle = interaction.particle;
    particle.pid =
        static_cast<std::int32_t>(EmPid::Positron);
    particle.medium_id = 17;
    particle.generation =
        static_cast<std::uint32_t>(index % 31);
    particle.energy_GeV = energy_GeV;
    particle.position_m[0] = 1.25;
    particle.position_m[1] = -2.5;
    particle.position_m[2] = 3.75;
    if (index % 3 == 0) {
      particle.direction[2] = 1.;
    } else if (index % 3 == 1) {
      particle.direction[0] = 0.6;
      particle.direction[2] = 0.8;
    } else {
      particle.direction[0] = -0.48;
      particle.direction[1] = 0.64;
      particle.direction[2] = 0.6;
    }
    particle.time_s = 4.e-6;
    particle.weight = 1. + 0.01 * (index % 7);
    particle.history_id = index + 1;
    particle.parent_history_id = index / 3;
    particle.step_id = index % 13;

    interaction.input_index = index;
    interaction.process_id = AnnihilationProcessId;
    interaction.status = EmInteractionStatus::Selected;
    interaction.component_hash = 101;
    interaction.mass_density_g_per_cm3 = 1.2048e-3;
    interaction.energy_fraction = 1.;
    interaction.loss_quantile =
        (static_cast<double>(index % 997) + 0.5) / 997.;
    interaction.loss_draw_id = InteractionLossDrawId;
    return interaction;
  }

} // namespace

int main() {
  int device_count = 0;
  auto const cuda_status = cudaGetDeviceCount(&device_count);
  if (cuda_status != cudaSuccess || device_count == 0) {
    std::cerr << "SKIP: no CUDA device is accessible\n";
    return 77;
  }

  auto const temporary = temporaryPath();
  try {
    auto source = testing::makeFlatRateTableFixture();
    auto const digest = writeRateTable(temporary, source);
    ProposalTableSet descriptor{};
    descriptor.process_count =
        static_cast<std::uint32_t>(processCount(source));
    descriptor.content_hash = digest;

    GpuEmConfig config{};
    config.device = 0;
    config.min_batch_size = 4096;
    config.memory_fraction = 0.10;
    config.table_tolerance = 1.e-3;
    config.random_seed = 0x414e4e4948494c41ULL;
    config.shower_id = 73;
    config.table_cache = temporary;
    EnvironmentSnapshot environment{};

    double constexpr ElectronMassGeV = 0.0005109989461;
    std::size_t constexpr ValidCount = 4096;
    std::vector<EmInteractionRecord> interactions;
    interactions.reserve(ValidCount + 4);
    for (std::size_t index = 0; index < ValidCount; ++index) {
      auto const fraction =
          (static_cast<double>(index) + 0.5) / ValidCount;
      // The production backend applies a 0.5 MeV kinetic-energy cut. Exercise
      // that complete supported range up to 100 GeV without asking the GPU
      // and CPU libm implementations to agree below the configured cut.
      auto const kinetic_GeV =
          5.e-4 * std::pow(2.e5, fraction);
      interactions.push_back(makeAnnihilation(
          index, ElectronMassGeV + kinetic_GeV));
    }

    auto continuation = makeAnnihilation(
        interactions.size(), 0.01);
    continuation.status =
        EmInteractionStatus::NoDiscreteInteraction;
    continuation.process_id = 0;
    interactions.push_back(continuation);

    auto wrong_pid = makeAnnihilation(
        interactions.size(), 0.01);
    wrong_pid.particle.pid =
        static_cast<std::int32_t>(EmPid::Electron);
    interactions.push_back(wrong_pid);

    auto invalid_fraction = makeAnnihilation(
        interactions.size(), 0.01);
    invalid_fraction.energy_fraction = 0.999;
    interactions.push_back(invalid_fraction);

    auto at_rest = makeAnnihilation(
        interactions.size(), 0.999 * ElectronMassGeV);
    interactions.push_back(at_rest);

    CudaEmBackend backend;
    backend.initialize(environment, descriptor, config);
    std::uint64_t constexpr FirstHistory = 12000000;
    auto const result =
        backend.generateBremsFinalStatesForValidation(
            interactions, FirstHistory);

    require(result.input_interactions == interactions.size(),
            "annihilation input count differs");
    {
      std::ostringstream message;
      message << "annihilation process counts differ: gpu="
              << result.gpu_interactions
              << ", annihilation="
              << result.annihilation_interactions
              << ", brems=" << result.brems_interactions
              << ", fallback="
              << result.fallback_events.size();
      require(result.gpu_interactions == ValidCount &&
                  result.annihilation_interactions == ValidCount &&
                  result.brems_interactions == 0,
              message.str());
    }
    require(result.final_state_records.size() == ValidCount,
            "annihilation record count differs");
    require(result.secondaries.size() == 2 * ValidCount,
            "annihilation secondary count differs");
    require(result.lpm_suppressed.empty(),
            "annihilation unexpectedly entered LPM suppression");
    require(result.continuations.size() == 1,
            "annihilation continuation count differs");
    require(result.fallback_events.size() == 3,
            "annihilation fallback count differs");
    require(
        result.fallback_events[0].reason ==
                ProposalFallbackReason::
                    GpuProcessNotImplemented &&
            result.fallback_events[1].reason ==
                ProposalFallbackReason::InvalidFinalState &&
            result.fallback_events[2].reason ==
                ProposalFallbackReason::InvalidFinalState,
        "annihilation fallback reasons differ");

    PROPOSAL::Air const medium;
    auto const component = medium.GetComponents().front();
    PROPOSAL::secondaries::HeitlerAnnihilation reference_generator(
        PROPOSAL::EPlusDef(), medium);
    for (std::size_t index = 0; index < ValidCount; ++index) {
      auto const& interaction = interactions[index];
      auto const& parent = interaction.particle;
      auto const& record = result.final_state_records[index];
      auto const& first = result.secondaries[2 * index];
      auto const& second = result.secondaries[2 * index + 1];
      RandomNumberKey const rho_key{
          config.random_seed, config.shower_id,
          parent.history_id, parent.step_id,
          static_cast<std::uint32_t>(AnnihilationProcessId),
          AnnihilationRhoDrawId};
      RandomNumberKey const azimuth_key{
          config.random_seed, config.shower_id,
          parent.history_id, parent.step_id,
          static_cast<std::uint32_t>(AnnihilationProcessId),
          AnnihilationAzimuthDrawId};
      auto const rho_uniform = uniformOpen01(rho_key);
      auto const azimuth_uniform = uniformOpen01(azimuth_key);

      require(
          record.input_index == interaction.input_index &&
              record.parent_history_id == parent.history_id &&
              record.secondary_offset == 2 * index &&
              record.secondary_count == 2 &&
              record.process_id == AnnihilationProcessId,
          "annihilation record identity differs");
      require(
          record.final_state_draw_id ==
                  AnnihilationRhoDrawId &&
              record.azimuth_draw_id ==
                  AnnihilationAzimuthDrawId &&
              record.lpm_draw_id == 0,
          "annihilation draw IDs differ");
      requireNear(
          record.final_state_uniform, rho_uniform, 0.,
          "annihilation rho uniform differs");
      requireNear(
          record.azimuth_uniform, azimuth_uniform, 0.,
          "annihilation azimuth uniform differs");
      require(record.photon_energy_fraction > 0. &&
                  record.photon_energy_fraction < 1.,
              "annihilation rho is outside (0,1)");
      require(
          first.pid ==
                  static_cast<std::int32_t>(EmPid::Photon) &&
              second.pid ==
                  static_cast<std::int32_t>(EmPid::Photon),
          "annihilation secondary PID differs");
      require(
          first.history_id == FirstHistory + 2 * index &&
              second.history_id ==
                  FirstHistory + 2 * index + 1 &&
              first.parent_history_id == parent.history_id &&
              second.parent_history_id == parent.history_id,
          "annihilation secondary identity differs");
      requireNear(
          first.energy_GeV + second.energy_GeV,
          parent.energy_GeV + ElectronMassGeV, 3.e-15,
          "annihilation energy does not close");

      PROPOSAL::Cartesian3D const position(
          parent.position_m[0] * 100.,
          parent.position_m[1] * 100.,
          parent.position_m[2] * 100.);
      PROPOSAL::Cartesian3D const direction(
          parent.direction[0], parent.direction[1],
          parent.direction[2]);
      PROPOSAL::StochasticLoss loss(
          AnnihilationProcessId,
          parent.energy_GeV * 1000., position, direction,
          parent.time_s, 0., parent.energy_GeV * 1000.,
          component.GetHash());
      std::vector<double> random_numbers{
          rho_uniform, azimuth_uniform};
      auto const reference =
          reference_generator.CalculateSecondaries(
              loss, component, random_numbers);
      require(reference.size() == 2,
              "PROPOSAL annihilation multiplicity differs");
      requireNear(
          first.energy_GeV * 1000., reference[0].energy,
          2.e-11,
          "first annihilation photon energy differs from PROPOSAL");
      requireNear(
          second.energy_GeV * 1000., reference[1].energy,
          2.e-11,
          "second annihilation photon energy differs from PROPOSAL");
      auto const reference_first =
          reference[0].direction.GetCartesianCoordinates();
      auto const reference_second =
          reference[1].direction.GetCartesianCoordinates();
      for (std::size_t axis = 0; axis < 3; ++axis) {
        requireAbsoluteNear(
            first.direction[axis], reference_first[axis],
            1.e-7,
            "first annihilation photon direction differs from PROPOSAL");
        requireAbsoluteNear(
            second.direction[axis], reference_second[axis],
            1.e-7,
            "second annihilation photon direction differs from PROPOSAL");
      }
    }

    auto const repeat =
        backend.generateBremsFinalStatesForValidation(
            interactions, FirstHistory);
    require(
        repeat.final_state_records.size() ==
                result.final_state_records.size() &&
            repeat.secondaries.size() ==
                result.secondaries.size(),
        "annihilation repeat counts differ");
    for (std::size_t index = 0;
         index < result.secondaries.size(); ++index) {
      auto const& left = repeat.secondaries[index];
      auto const& right = result.secondaries[index];
      require(
          left.history_id == right.history_id &&
              left.energy_GeV == right.energy_GeV &&
              left.direction[0] == right.direction[0] &&
              left.direction[1] == right.direction[1] &&
              left.direction[2] == right.direction[2],
          "annihilation kernel is not deterministic");
    }
    auto const& statistics = backend.statistics();
    require(
        statistics.annihilation_final_states ==
                2 * ValidCount &&
            statistics.brems_lpm_trials == 0 &&
            statistics.brems_lpm_suppressions == 0,
        "annihilation backend statistics differ");

    std::error_code ignored;
    std::filesystem::remove(temporary, ignored);
    std::cout
        << "GPU positron-annihilation PROPOSAL oracle passed "
        << checks << " checks across " << ValidCount
        << " events\n";
    return 0;
  } catch (std::exception const& error) {
    std::error_code ignored;
    std::filesystem::remove(temporary, ignored);
    std::cerr
        << "GPU positron-annihilation PROPOSAL oracle failed after "
        << checks << " checks: " << error.what() << '\n';
    return 1;
  }
}

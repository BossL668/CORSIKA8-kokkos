/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include <cuda_runtime_api.h>

#include <PROPOSAL/medium/Medium.h>
#include <PROPOSAL/particle/Particle.h>
#include <PROPOSAL/secondaries/parametrization/ionization/NaivIonization.h>

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
           ("c8_gpu_ionization_final_state_" +
            std::to_string(stamp) + ".c8emrt");
  }

  std::size_t processCount(RateTableSet const& table) {
    std::size_t count = 0;
    for (auto const& particle : table.particles) {
      count += particle.columns.size();
    }
    return count;
  }

  EmInteractionRecord makeIonization(
      std::size_t index, double energy_GeV,
      double energy_fraction,
      std::int32_t pid = 0,
      double particle_mass_GeV = 0.) {
    EmInteractionRecord interaction{};
    auto& particle = interaction.particle;
    particle.pid =
        pid != 0
            ? pid
            : (index % 2 == 0
                   ? static_cast<std::int32_t>(EmPid::Electron)
                   : static_cast<std::int32_t>(EmPid::Positron));
    particle.medium_id = 17;
    particle.generation =
        static_cast<std::uint32_t>(index % 29);
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
    interaction.process_id = IonizationProcessId;
    interaction.status = EmInteractionStatus::Selected;
    interaction.component_hash = 102;
    interaction.mass_density_g_per_cm3 = 1.2048e-3;
    interaction.particle_mass_GeV = particle_mass_GeV;
    interaction.energy_fraction = energy_fraction;
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
    config.random_seed = 0x494f4e495a415449ULL;
    config.shower_id = 79;
    config.table_cache = temporary;
    EnvironmentSnapshot environment{};

    double constexpr ElectronMassGeV = 0.0005109989461;
    std::size_t constexpr ValidCount = 4096;
    std::vector<EmInteractionRecord> interactions;
    interactions.reserve(ValidCount + 4);
    for (std::size_t index = 0; index < ValidCount; ++index) {
      auto const energy_coordinate =
          (static_cast<double>(index) + 0.5) / ValidCount;
      auto const energy_GeV =
          (ElectronMassGeV + 5.e-4) *
          std::pow(
              100. / (ElectronMassGeV + 5.e-4),
              energy_coordinate);
      auto const maximum_fraction =
          1. - ElectronMassGeV / energy_GeV;
      auto const split_coordinate =
          (static_cast<double>(index % 991) + 0.5) / 991.;
      auto const v =
          maximum_fraction *
          (1.e-4 + 0.9998 * split_coordinate);
      interactions.push_back(
          makeIonization(index, energy_GeV, v));
    }

    auto continuation =
        makeIonization(interactions.size(), 0.01, 0.1);
    continuation.status =
        EmInteractionStatus::NoDiscreteInteraction;
    continuation.process_id = 0;
    interactions.push_back(continuation);

    auto wrong_pid =
        makeIonization(interactions.size(), 0.01, 0.1);
    wrong_pid.particle.pid =
        static_cast<std::int32_t>(EmPid::Photon);
    interactions.push_back(wrong_pid);

    auto zero_loss =
        makeIonization(interactions.size(), 0.01, 0.);
    interactions.push_back(zero_loss);

    auto excessive_loss =
        makeIonization(interactions.size(), 0.01, 1.);
    interactions.push_back(excessive_loss);

    CudaEmBackend backend;
    backend.initialize(environment, descriptor, config);
    std::uint64_t constexpr FirstHistory = 14000000;
    auto const result =
        backend.generateBremsFinalStatesForValidation(
            interactions, FirstHistory);

    require(result.input_interactions == interactions.size(),
            "ionization input count differs");
    require(result.gpu_interactions == ValidCount &&
                result.ionization_interactions == ValidCount &&
                result.brems_interactions == 0 &&
                result.annihilation_interactions == 0,
            "ionization process counts differ");
    require(result.final_state_records.size() == ValidCount,
            "ionization record count differs");
    require(result.secondaries.size() == 2 * ValidCount,
            "ionization secondary count differs");
    require(result.lpm_suppressed.empty(),
            "ionization unexpectedly entered LPM suppression");
    require(result.continuations.size() == 1,
            "ionization continuation count differs");
    require(result.fallback_events.size() == 3,
            "ionization fallback count differs");
    require(
        result.fallback_events[0].reason ==
                ProposalFallbackReason::
                    GpuProcessNotImplemented &&
            result.fallback_events[1].reason ==
                ProposalFallbackReason::InvalidFinalState &&
            result.fallback_events[2].reason ==
                ProposalFallbackReason::InvalidFinalState,
        "ionization fallback reasons differ");

    PROPOSAL::Air const medium;
    auto const component = medium.GetComponents().front();
    PROPOSAL::secondaries::NaivIonization electron_reference(
        PROPOSAL::EMinusDef(), medium);
    PROPOSAL::secondaries::NaivIonization positron_reference(
        PROPOSAL::EPlusDef(), medium);
    for (std::size_t index = 0; index < ValidCount; ++index) {
      auto const& interaction = interactions[index];
      auto const& parent = interaction.particle;
      auto const& record = result.final_state_records[index];
      auto const& outgoing = result.secondaries[2 * index];
      auto const& delta = result.secondaries[2 * index + 1];
      RandomNumberKey const azimuth_key{
          config.random_seed, config.shower_id,
          parent.history_id, parent.step_id,
          static_cast<std::uint32_t>(IonizationProcessId),
          IonizationAzimuthDrawId};
      auto const azimuth_uniform =
          uniformOpen01(azimuth_key);

      require(
          record.input_index == interaction.input_index &&
              record.parent_history_id == parent.history_id &&
              record.secondary_offset == 2 * index &&
              record.secondary_count == 2 &&
              record.process_id == IonizationProcessId,
          "ionization record identity differs");
      require(
          record.final_state_draw_id == 0 &&
              record.azimuth_draw_id ==
                  IonizationAzimuthDrawId &&
              record.lpm_draw_id == 0,
          "ionization draw IDs differ");
      requireNear(
          record.azimuth_uniform, azimuth_uniform, 0.,
          "ionization azimuth uniform differs");
      requireNear(
          record.photon_energy_fraction,
          interaction.energy_fraction, 0.,
          "ionization record loss fraction differs");
      require(
          outgoing.pid == parent.pid &&
              delta.pid ==
                  static_cast<std::int32_t>(EmPid::Electron),
          "ionization secondary PID differs");
      require(
          outgoing.history_id ==
                  FirstHistory + 2 * index &&
              delta.history_id ==
                  FirstHistory + 2 * index + 1 &&
              outgoing.parent_history_id ==
                  parent.history_id &&
              delta.parent_history_id == parent.history_id,
          "ionization secondary identity differs");
      requireNear(
          outgoing.energy_GeV + delta.energy_GeV,
          parent.energy_GeV + ElectronMassGeV, 3.e-15,
          "ionization energy does not close");

      PROPOSAL::Cartesian3D const position(
          parent.position_m[0] * 100.,
          parent.position_m[1] * 100.,
          parent.position_m[2] * 100.);
      PROPOSAL::Cartesian3D const direction(
          parent.direction[0], parent.direction[1],
          parent.direction[2]);
      PROPOSAL::StochasticLoss loss(
          IonizationProcessId,
          interaction.energy_fraction *
              (parent.energy_GeV * 1000.),
          position, direction, parent.time_s, 0.,
          parent.energy_GeV * 1000.,
          component.GetHash());
      std::vector<double> random_numbers{
          azimuth_uniform};
      auto const reference =
          parent.pid ==
                  static_cast<std::int32_t>(EmPid::Electron)
              ? electron_reference.CalculateSecondaries(
                    loss, component, random_numbers)
              : positron_reference.CalculateSecondaries(
                    loss, component, random_numbers);
      require(reference.size() == 2,
              "PROPOSAL ionization multiplicity differs");
      requireNear(
          outgoing.energy_GeV * 1000.,
          reference[0].energy, 3.e-14,
          "outgoing lepton energy differs from PROPOSAL");
      requireNear(
          delta.energy_GeV * 1000.,
          reference[1].energy, 3.e-14,
          "delta-electron energy differs from PROPOSAL");
      auto const reference_outgoing =
          reference[0].direction.GetCartesianCoordinates();
      auto const reference_delta =
          reference[1].direction.GetCartesianCoordinates();
      for (std::size_t axis = 0; axis < 3; ++axis) {
        requireAbsoluteNear(
            outgoing.direction[axis],
            reference_outgoing[axis], 1.e-7,
            "outgoing ionization direction differs from PROPOSAL");
        requireAbsoluteNear(
            delta.direction[axis],
            reference_delta[axis], 1.e-7,
            "delta-electron direction differs from PROPOSAL");
      }
    }

    double constexpr MuonMassGeV = 0.1056583745;
    std::size_t constexpr MuonValidCount = 4096;
    std::vector<EmInteractionRecord> muon_interactions;
    muon_interactions.reserve(MuonValidCount);
    for (std::size_t index = 0; index < MuonValidCount; ++index) {
      auto const energy_coordinate =
          (static_cast<double>(index) + 0.5) /
          MuonValidCount;
      auto const energy_GeV =
          std::pow(1.e6, energy_coordinate);
      auto const gamma = energy_GeV / MuonMassGeV;
      auto const beta_squared =
          1. - 1. / (gamma * gamma);
      auto const mass_ratio =
          ElectronMassGeV / MuonMassGeV;
      auto const maximum_transfer_GeV =
          2. * ElectronMassGeV * beta_squared *
          gamma * gamma /
          (1. + 2. * gamma * mass_ratio +
           mass_ratio * mass_ratio);
      auto const split_coordinate =
          (static_cast<double>(index % 991) + 0.5) / 991.;
      auto const v =
          maximum_transfer_GeV / energy_GeV *
          (1.e-4 + 0.9998 * split_coordinate);
      muon_interactions.push_back(makeIonization(
          100000 + index, energy_GeV, v,
          index % 2 == 0
              ? static_cast<std::int32_t>(EmPid::MuonMinus)
              : static_cast<std::int32_t>(EmPid::MuonPlus),
          MuonMassGeV));
      muon_interactions.back().input_index = index;
    }

    std::uint64_t constexpr FirstMuonHistory = 15000000;
    auto const muon_result =
        backend.generateBremsFinalStatesForValidation(
            muon_interactions, FirstMuonHistory);
    require(
        muon_result.input_interactions == MuonValidCount &&
            muon_result.gpu_interactions == MuonValidCount &&
            muon_result.ionization_interactions ==
                MuonValidCount &&
            muon_result.final_state_records.size() ==
                MuonValidCount &&
            muon_result.secondaries.size() ==
                2 * MuonValidCount &&
            muon_result.fallback_events.empty(),
        "muon ionization process counts differ");

    PROPOSAL::secondaries::NaivIonization muon_minus_reference(
        PROPOSAL::MuMinusDef(), medium);
    PROPOSAL::secondaries::NaivIonization muon_plus_reference(
        PROPOSAL::MuPlusDef(), medium);
    for (std::size_t index = 0; index < MuonValidCount; ++index) {
      auto const& interaction = muon_interactions[index];
      auto const& parent = interaction.particle;
      auto const& outgoing = muon_result.secondaries[2 * index];
      auto const& delta =
          muon_result.secondaries[2 * index + 1];
      RandomNumberKey const azimuth_key{
          config.random_seed, config.shower_id,
          parent.history_id, parent.step_id,
          static_cast<std::uint32_t>(IonizationProcessId),
          IonizationAzimuthDrawId};
      auto const azimuth_uniform =
          uniformOpen01(azimuth_key);
      require(
          outgoing.pid == parent.pid &&
              delta.pid ==
                  static_cast<std::int32_t>(EmPid::Electron) &&
              outgoing.history_id ==
                  FirstMuonHistory + 2 * index &&
              delta.history_id ==
                  FirstMuonHistory + 2 * index + 1,
          "muon ionization secondary identity differs");
      requireNear(
          outgoing.energy_GeV + delta.energy_GeV,
          parent.energy_GeV + ElectronMassGeV, 5.e-15,
          "muon ionization energy does not close");

      PROPOSAL::Cartesian3D const position(
          parent.position_m[0] * 100.,
          parent.position_m[1] * 100.,
          parent.position_m[2] * 100.);
      PROPOSAL::Cartesian3D const direction(
          parent.direction[0], parent.direction[1],
          parent.direction[2]);
      PROPOSAL::StochasticLoss loss(
          IonizationProcessId,
          interaction.energy_fraction *
              (parent.energy_GeV * 1000.),
          position, direction, parent.time_s, 0.,
          parent.energy_GeV * 1000.,
          component.GetHash());
      std::vector<double> random_numbers{azimuth_uniform};
      auto const reference =
          parent.pid ==
                  static_cast<std::int32_t>(EmPid::MuonMinus)
              ? muon_minus_reference.CalculateSecondaries(
                    loss, component, random_numbers)
              : muon_plus_reference.CalculateSecondaries(
                    loss, component, random_numbers);
      require(reference.size() == 2,
              "PROPOSAL muon ionization multiplicity differs");
      requireNear(
          outgoing.energy_GeV * 1000.,
          reference[0].energy, 5.e-14,
          "outgoing muon energy differs from PROPOSAL");
      requireNear(
          delta.energy_GeV * 1000.,
          reference[1].energy, 5.e-14,
          "muon delta-electron energy differs from PROPOSAL");
      auto const reference_outgoing =
          reference[0].direction.GetCartesianCoordinates();
      auto const reference_delta =
          reference[1].direction.GetCartesianCoordinates();
      for (std::size_t axis = 0; axis < 3; ++axis) {
        requireAbsoluteNear(
            outgoing.direction[axis],
            reference_outgoing[axis], 1.e-7,
            "outgoing muon direction differs from PROPOSAL");
        requireAbsoluteNear(
            delta.direction[axis],
            reference_delta[axis], 1.e-7,
            "muon delta-electron direction differs from PROPOSAL");
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
        "ionization repeat counts differ");
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
          "ionization kernel is not deterministic");
    }
    auto const& statistics = backend.statistics();
    require(
        statistics.ionization_final_states ==
                2 * ValidCount + MuonValidCount &&
            statistics.brems_lpm_trials == 0 &&
            statistics.annihilation_final_states == 0,
        "ionization backend statistics differ");

    std::error_code ignored;
    std::filesystem::remove(temporary, ignored);
    std::cout
        << "GPU discrete-ionization PROPOSAL oracle passed "
        << checks << " checks across "
        << ValidCount + MuonValidCount
        << " events\n";
    return 0;
  } catch (std::exception const& error) {
    std::error_code ignored;
    std::filesystem::remove(temporary, ignored);
    std::cerr
        << "GPU discrete-ionization PROPOSAL oracle failed after "
        << checks << " checks: " << error.what() << '\n';
    return 1;
  }
}

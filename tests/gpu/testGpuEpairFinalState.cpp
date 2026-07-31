/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include <cuda_runtime_api.h>

#include <PROPOSAL/crosssection/parametrization/EpairProduction.h>
#include <PROPOSAL/medium/Medium.h>
#include <PROPOSAL/particle/Particle.h>
#include <PROPOSAL/secondaries/parametrization/epairproduction/KelnerKokoulinPetrukhinEpairProduction.h>

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
#include <corsika/gpu/em/EpairFinalState.hpp>
#include <corsika/gpu/em/EpairLpm.hpp>
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
           ("c8_gpu_epair_final_state_" +
            std::to_string(stamp) + ".c8emrt");
  }

  std::size_t processCount(RateTableSet const& table) {
    std::size_t count = 0;
    for (auto const& particle : table.particles) {
      count += particle.columns.size();
    }
    return count;
  }

  EmInteractionRecord makeInteraction(
      std::size_t index, double energy_MeV, double v,
      std::uint64_t component_hash, double density) {
    EmInteractionRecord interaction{};
    auto& particle = interaction.particle;
    particle.pid =
        index % 2 == 0
            ? static_cast<std::int32_t>(EmPid::Electron)
            : static_cast<std::int32_t>(EmPid::Positron);
    particle.medium_id = 17;
    particle.generation =
        static_cast<std::uint32_t>(index % 31);
    particle.energy_GeV = energy_MeV / 1000.;
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
    particle.step_id = index % 17;

    interaction.input_index = index;
    interaction.process_id = ElectronPairProcessId;
    interaction.status = EmInteractionStatus::Selected;
    interaction.component_hash = component_hash;
    interaction.mass_density_g_per_cm3 = density;
    interaction.energy_fraction = v;
    interaction.loss_quantile =
        (static_cast<double>(index % 997) + 0.5) / 997.;
    interaction.loss_draw_id = 0;
    return interaction;
  }

  struct Expected {
    std::vector<std::size_t> accepted;
    std::vector<std::size_t> suppressed;
    std::vector<EpairFinalStateSample> samples;
    std::vector<double> direction_uniforms;
    std::vector<double> lpm_uniforms;
    std::vector<double> lpm_probabilities;
    std::size_t rejection_trials{};
    std::size_t zero_weight_samples{};
  };

  Expected classify(
      std::vector<EmInteractionRecord> const& interactions,
      BremsLpmSnapshot const& snapshot, std::uint64_t seed,
      std::uint64_t shower_id) {
    Expected result;
    result.samples.resize(interactions.size());
    result.direction_uniforms.resize(interactions.size());
    result.lpm_uniforms.resize(interactions.size());
    result.lpm_probabilities.resize(interactions.size());
    for (std::size_t index = 0; index < interactions.size();
         ++index) {
      auto const& interaction = interactions[index];
      auto const& parent = interaction.particle;
      RandomNumberKey const rho_key{
          seed, shower_id, parent.history_id, parent.step_id,
          static_cast<std::uint32_t>(ElectronPairProcessId),
          EpairRhoDrawId};
      RandomNumberKey const sign_key{
          seed, shower_id, parent.history_id, parent.step_id,
          static_cast<std::uint32_t>(ElectronPairProcessId),
          EpairSignDrawId};
      RandomNumberKey const direction_key{
          seed, shower_id, parent.history_id, parent.step_id,
          static_cast<std::uint32_t>(ElectronPairProcessId),
          EpairDirectionDrawId};
      RandomNumberKey const lpm_key{
          seed, shower_id, parent.history_id, parent.step_id,
          static_cast<std::uint32_t>(ElectronPairProcessId),
          EpairLpmDrawId};
      auto const sign_uniform = uniformOpen01(sign_key);
      auto const rejection = sampleEpairRhoRejection(
          snapshot, interaction.component_hash,
          parent.energy_GeV * 1000.,
          interaction.energy_fraction, sign_uniform,
          rho_key);
      result.samples[index] = rejection.sample;
      result.rejection_trials += rejection.trial_count;
      if (rejection.trial_count == 0 &&
          rejection.sample.status ==
              EpairFinalStateStatus::Success &&
          rejection.sample.rho == 0.) {
        ++result.zero_weight_samples;
      }
      if (result.samples[index].status !=
          EpairFinalStateStatus::Success) {
        std::ostringstream message;
        message
            << "valid host Epair rejection sample failed"
            << ": index=" << index
            << ", status="
            << static_cast<std::uint32_t>(
                   result.samples[index].status)
            << ", E_MeV="
            << parent.energy_GeV * 1000.
            << ", v=" << interaction.energy_fraction;
        throw std::runtime_error(message.str());
      }
      result.direction_uniforms[index] =
          uniformOpen01(direction_key);
      auto const lpm = epairLpmSuppressionFactor(
          snapshot, parent.energy_GeV * 1000.,
          interaction.energy_fraction,
          result.samples[index].rho *
              result.samples[index].rho,
          interaction.mass_density_g_per_cm3);
      require(lpm.status == EpairLpmStatus::Success,
              "valid host Epair LPM sample failed");
      result.lpm_probabilities[index] =
          lpm.survival_probability;
      result.lpm_uniforms[index] = uniformOpen01(lpm_key);
      if (result.lpm_uniforms[index] >
          result.lpm_probabilities[index]) {
        result.suppressed.push_back(index);
      } else {
        result.accepted.push_back(index);
      }
    }
    return result;
  }

} // namespace

int main() {
  int device_count = 0;
  auto const status = cudaGetDeviceCount(&device_count);
  if (status != cudaSuccess || device_count == 0) {
    std::cerr << "SKIP: no CUDA device is accessible\n";
    return 77;
  }

  auto const temporary = temporaryPath();
  try {
    auto source = testing::makeFlatRateTableFixture();
    PROPOSAL::Air const medium;
    auto const component = medium.GetComponents().front();
    source.metadata.brems_lpm.lepton_mass_MeV =
        PROPOSAL::EMinusDef().mass;
    source.metadata.brems_lpm.electron_mass_MeV =
        PROPOSAL::ME;
    source.metadata.brems_lpm.muon_mass_MeV =
        PROPOSAL::MMU;
    source.metadata.brems_lpm.components[0].nuclear_charge =
        component.GetNucCharge();
    source.metadata.brems_lpm.components[0]
        .atomic_mass_number = component.GetAtomicNum();
    source.metadata.brems_lpm.components[0]
        .radiation_log_constant = component.GetLogConstant();
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
    config.random_seed = 0x455041495246494eULL;
    config.shower_id = 97;
    config.table_cache = temporary;
    EnvironmentSnapshot environment{};

    auto const snapshot =
        makeBremsLpmSnapshot(source.metadata.brems_lpm);
    PROPOSAL::crosssection::EpairForElectronPositron rate_model(
        false);
    std::size_t constexpr ValidCount = 1024;
    std::vector<EmInteractionRecord> valid;
    valid.reserve(ValidCount);
    for (std::size_t index = 0; index < ValidCount; ++index) {
      auto const energy_coordinate =
          (static_cast<double>(index) + 0.5) / ValidCount;
      auto const energy_MeV =
          20. * std::pow(1.e14 / 20., energy_coordinate);
      auto const limits = rate_model.GetKinematicLimits(
          PROPOSAL::EMinusDef(), component, energy_MeV);
      auto const quantile =
          0.05 +
          0.90 *
              (static_cast<double>(index % 991) + 0.5) /
              991.;
      auto const v =
          limits.v_min +
          quantile * (limits.v_max - limits.v_min);
      auto const density_ratio =
          std::pow(
              1.e-7,
              static_cast<double>(index % 101) / 100.);
      valid.push_back(makeInteraction(
          index, energy_MeV, v,
          snapshot.components[0].component_hash,
          snapshot.baseline_mass_density_g_per_cm3 *
              density_ratio));
    }

    auto const expected = classify(
        valid, snapshot, config.random_seed, config.shower_id);
    require(!expected.accepted.empty(),
            "Epair fixture produced no accepted interactions");
    require(!expected.suppressed.empty(),
            "Epair fixture produced no LPM suppressions");

    auto interactions = valid;
    auto continuation = valid.front();
    continuation.input_index = interactions.size();
    continuation.status =
        EmInteractionStatus::NoDiscreteInteraction;
    continuation.process_id = 0;
    interactions.push_back(continuation);
    auto wrong_pid = valid.front();
    wrong_pid.input_index = interactions.size();
    wrong_pid.particle.pid =
        static_cast<std::int32_t>(EmPid::Photon);
    interactions.push_back(wrong_pid);
    auto invalid_loss = valid.front();
    invalid_loss.input_index = interactions.size();
    invalid_loss.energy_fraction = 1.;
    interactions.push_back(invalid_loss);
    auto missing_component = valid.front();
    missing_component.input_index = interactions.size();
    missing_component.component_hash = 0xdeadbeefULL;
    interactions.push_back(missing_component);

    CudaEmBackend backend;
    backend.initialize(environment, descriptor, config);
    std::uint64_t constexpr FirstHistory = 18000000;
    auto const result =
        backend.generateBremsFinalStatesForValidation(
            interactions, FirstHistory);
    require(
        result.input_interactions == interactions.size() &&
            result.gpu_interactions ==
                expected.accepted.size() &&
            result.electron_pair_interactions ==
                expected.accepted.size() &&
            result.brems_interactions == 0 &&
            result.annihilation_interactions == 0 &&
            result.ionization_interactions == 0,
        "Epair final-state process counts differ");
    require(
        result.secondaries.size() ==
                3 * expected.accepted.size() &&
            result.final_state_records.size() ==
                expected.accepted.size(),
        "Epair final-state multiplicity differs");
    require(
        result.lpm_suppressed.size() ==
                expected.suppressed.size() &&
            result.electron_pair_lpm_trials == ValidCount &&
            result.electron_pair_lpm_suppressions ==
                expected.suppressed.size() &&
            result.brems_lpm_trials == 0 &&
            result.brems_lpm_suppressions == 0,
        "Epair LPM classification counts differ");
    require(result.continuations.size() == 1,
            "Epair continuation count differs");
    require(result.fallback_events.size() == 3,
            "Epair fallback count differs");

    PROPOSAL::secondaries::
        KelnerKokoulinPetrukhinEpairProduction electron_reference(
            PROPOSAL::EMinusDef(), medium);
    PROPOSAL::secondaries::
        KelnerKokoulinPetrukhinEpairProduction positron_reference(
            PROPOSAL::EPlusDef(), medium);
    double maximum_normalized_rho_error = 0.;
    double sampled_maximum_envelope_ratio = 0.;
    double sampled_minimum_acceptance = 1.;
    for (std::size_t output_index = 0;
         output_index < expected.accepted.size();
         ++output_index) {
      auto const input_index = expected.accepted[output_index];
      auto const& interaction = valid[input_index];
      auto const& parent = interaction.particle;
      auto const& sample = expected.samples[input_index];
      auto const& record =
          result.final_state_records[output_index];
      require(
          record.input_index == interaction.input_index &&
              record.parent_history_id == parent.history_id &&
              record.secondary_offset == 3 * output_index &&
              record.secondary_count == 3 &&
              record.process_id == ElectronPairProcessId,
          "Epair record identity differs");
      require(
          record.final_state_draw_id == EpairRhoDrawId &&
              record.azimuth_draw_id == EpairSignDrawId &&
              record.auxiliary_draw_id ==
                  EpairDirectionDrawId &&
              record.lpm_draw_id == EpairLpmDrawId,
          "Epair record draw IDs differ");
      requireNear(
          record.auxiliary_uniform,
          expected.direction_uniforms[input_index], 0.,
          "Epair direction uniform differs");
      requireNear(
          record.lpm_uniform,
          expected.lpm_uniforms[input_index], 0.,
          "Epair LPM uniform differs");

      auto const& surviving =
          result.secondaries[3 * output_index];
      auto const& electron =
          result.secondaries[3 * output_index + 1];
      auto const& positron =
          result.secondaries[3 * output_index + 2];
      require(
          surviving.pid == parent.pid &&
              electron.pid ==
                  static_cast<std::int32_t>(EmPid::Electron) &&
              positron.pid ==
                  static_cast<std::int32_t>(EmPid::Positron),
          "Epair secondary PID order differs");
      requireNear(
          surviving.energy_GeV + electron.energy_GeV +
              positron.energy_GeV,
          parent.energy_GeV, 2.e-15 * parent.energy_GeV,
          "Epair energy does not close");
      for (int axis = 0; axis < 3; ++axis) {
        requireNear(
            surviving.direction[axis], parent.direction[axis],
            0., "Epair surviving direction differs");
        requireNear(
            electron.direction[axis], parent.direction[axis],
            0., "Epair electron direction differs");
        requireNear(
            positron.direction[axis], parent.direction[axis],
            0., "Epair positron direction differs");
      }

      RandomNumberKey const rho_key{
          config.random_seed, config.shower_id,
          parent.history_id, parent.step_id,
          static_cast<std::uint32_t>(ElectronPairProcessId),
          EpairRhoDrawId};
      RandomNumberKey const sign_key{
          config.random_seed, config.shower_id,
          parent.history_id, parent.step_id,
          static_cast<std::uint32_t>(ElectronPairProcessId),
          EpairSignDrawId};
      auto const rho_uniform = uniformOpen01(rho_key);
      auto const sign_uniform = uniformOpen01(sign_key);
      auto const reference_rho =
          (parent.pid ==
                   static_cast<std::int32_t>(EmPid::Electron)
               ? electron_reference
               : positron_reference)
              .CalculateRho(
                  parent.energy_GeV * 1000.,
                  interaction.energy_fraction, component,
                  rho_uniform, sign_uniform);
      auto const exact_sample = sampleEpairRho(
          snapshot, interaction.component_hash,
          parent.energy_GeV * 1000.,
          interaction.energy_fraction, rho_uniform,
          sign_uniform);
      require(
          exact_sample.status ==
              EpairFinalStateStatus::Success,
          "exact Epair rho sample failed");
      auto const normalized_error =
          std::abs(exact_sample.rho - reference_rho) /
          std::max(1.e-12, exact_sample.rho_max);
      maximum_normalized_rho_error =
          std::max(maximum_normalized_rho_error,
                   normalized_error);
      auto const* snapshot_component =
          static_cast<BremsLpmComponentSnapshot const*>(
              nullptr);
      for (std::uint32_t component_index = 0;
           component_index < snapshot.component_count;
           ++component_index) {
        if (snapshot.components[component_index].component_hash ==
            interaction.component_hash) {
          snapshot_component =
              &snapshot.components[component_index];
          break;
        }
      }
      require(snapshot_component != nullptr,
              "Epair envelope component is unavailable");
      auto transformed_weight = [&](double uniform) {
        auto const angle =
            0.5 * 3.14159265358979323846 * uniform;
        auto const sine = std::sin(angle);
        return epair_final_state_detail::kkpRhoWeight(
                   *snapshot_component,
                   snapshot.lepton_mass_MeV,
                   snapshot.electron_mass_MeV,
                   parent.energy_GeV * 1000.,
                   interaction.energy_fraction,
                   sample.rho_max * sine * sine) *
               std::sin(
                   3.14159265358979323846 * uniform);
      };
      auto const coarse_maximum =
          epair_final_state_detail::
              estimateTransformedKkpMaximum(
                  *snapshot_component,
                  snapshot.lepton_mass_MeV,
                  snapshot.electron_mass_MeV,
                  parent.energy_GeV * 1000.,
                  interaction.energy_fraction,
                  sample.rho_max);
      double dense_maximum = 0.;
      double dense_sum = 0.;
      for (std::size_t point = 0; point < 4096; ++point) {
        auto const weight = transformed_weight(
            (static_cast<double>(point) + 0.5) / 4096.);
        dense_maximum = std::max(dense_maximum, weight);
        dense_sum += weight;
      }
      if (dense_maximum > 0.) {
        sampled_maximum_envelope_ratio = std::max(
            sampled_maximum_envelope_ratio,
            dense_maximum / coarse_maximum);
        sampled_minimum_acceptance = std::min(
            sampled_minimum_acceptance,
            dense_sum / (4096. * dense_maximum));
      }
      if (!(normalized_error < 1.e-3)) {
        PROPOSAL::crosssection::
            EpairKelnerKokoulinPetrukhin integrand_reference(
                false);
        std::ostringstream details;
        details << std::setprecision(17)
                << "GPU Epair rho inverse CDF differs from PROPOSAL"
                << ": input=" << input_index
                << ", E_MeV=" << parent.energy_GeV * 1000.
                << ", v=" << interaction.energy_fraction
                << ", u=" << rho_uniform
                << ", sign_u=" << sign_uniform
                << ", actual=" << exact_sample.rho
                << ", expected=" << reference_rho
                << ", rho_max=" << exact_sample.rho_max
                << ", normalized_error=" << normalized_error;
        for (double probe : {0.1, 0.5, 0.9}) {
          auto const reference_weight =
              integrand_reference.FunctionToIntegral(
                  PROPOSAL::EMinusDef(), component,
                  parent.energy_GeV * 1000.,
                  interaction.energy_fraction, probe);
          auto const device_weight =
              epair_final_state_detail::kkpRhoWeight(
                  snapshot.components[0],
                  snapshot.lepton_mass_MeV,
                  snapshot.electron_mass_MeV,
                  parent.energy_GeV * 1000.,
                  interaction.energy_fraction, probe);
          details << ", ratio(" << probe << ")="
                  << reference_weight / device_weight;
        }
        throw std::runtime_error(details.str());
      }
      ++checks;

      auto const loss_energy_MeV =
          interaction.energy_fraction *
          parent.energy_GeV * 1000.;
      auto const expected_electron_energy_MeV =
          0.5 * loss_energy_MeV * (1. + sample.rho);
      auto const expected_positron_energy_MeV =
          0.5 * loss_energy_MeV * (1. - sample.rho);
      auto const pair_energy_tolerance_MeV =
          5.e-13 * loss_energy_MeV;
      requireNear(
          electron.energy_GeV * 1000.,
          expected_electron_energy_MeV,
          pair_energy_tolerance_MeV,
          "GPU Epair electron rejection-sample energy differs");
      requireNear(
          positron.energy_GeV * 1000.,
          expected_positron_energy_MeV,
          pair_energy_tolerance_MeV,
          "GPU Epair positron rejection-sample energy differs");
    }
    double sweep_maximum_envelope_ratio = 0.;
    double sweep_minimum_acceptance = 1.;
    std::size_t sweep_maximum_local_maxima = 0;
    std::array<double, 11> const loss_coordinates{
        1.e-6, 1.e-4, 1.e-3, 1.e-2, 0.05, 0.25,
        0.5, 0.75, 0.95, 0.999, 1. - 1.e-6};
    for (std::uint32_t component_index = 0;
         component_index < snapshot.component_count;
         ++component_index) {
      auto const& sweep_component =
          snapshot.components[component_index];
      for (std::size_t energy_index = 0;
           energy_index < 64; ++energy_index) {
        auto const energy_coordinate =
            (static_cast<double>(energy_index) + 0.5) / 64.;
        auto const energy_MeV =
            20. * std::pow(1.e14 / 20., energy_coordinate);
        auto const limits = rate_model.GetKinematicLimits(
            PROPOSAL::EMinusDef(), component, energy_MeV);
        for (auto const loss_coordinate :
             loss_coordinates) {
          auto const v =
              limits.v_min +
              loss_coordinate *
                  (limits.v_max - limits.v_min);
          auto const threshold_auxiliary =
              1. -
              4. * snapshot.electron_mass_MeV /
                  (energy_MeV * v);
          auto const recoil_auxiliary =
              1. -
              6. * snapshot.lepton_mass_MeV *
                  snapshot.lepton_mass_MeV /
                  (energy_MeV * energy_MeV *
                   (1. - v));
          if (!(threshold_auxiliary > 0.) ||
              !(recoil_auxiliary > 0.)) {
            continue;
          }
          auto const rho_max =
              std::sqrt(threshold_auxiliary) *
              recoil_auxiliary;
          auto transformed_weight =
              [&](double uniform) {
                auto const angle =
                    0.5 * 3.14159265358979323846 *
                    uniform;
                auto const sine = std::sin(angle);
                return epair_final_state_detail::
                           kkpRhoWeight(
                               sweep_component,
                               snapshot.lepton_mass_MeV,
                               snapshot.electron_mass_MeV,
                               energy_MeV, v,
                               rho_max * sine * sine) *
                       std::sin(
                           3.14159265358979323846 *
                           uniform);
              };
          auto const coarse_maximum =
              epair_final_state_detail::
                  estimateTransformedKkpMaximum(
                      sweep_component,
                      snapshot.lepton_mass_MeV,
                      snapshot.electron_mass_MeV,
                      energy_MeV, v, rho_max);
          double dense_maximum = 0.;
          double dense_sum = 0.;
          double previous = 0.;
          bool rising = false;
          std::size_t local_maxima = 0;
          for (std::size_t point = 0; point < 8192;
               ++point) {
            auto const value = transformed_weight(
                (static_cast<double>(point) + 0.5) /
                8192.);
            dense_maximum = std::max(
                dense_maximum, value);
            dense_sum += value;
            if (point != 0) {
              auto const now_rising = value > previous;
              if (rising && !now_rising) {
                ++local_maxima;
              }
              rising = now_rising;
            }
            previous = value;
          }
          if (dense_maximum > 0.) {
            sweep_maximum_envelope_ratio =
                std::max(
                    sweep_maximum_envelope_ratio,
                    dense_maximum / coarse_maximum);
            sweep_minimum_acceptance = std::min(
                sweep_minimum_acceptance,
                dense_sum /
                    (8192. * dense_maximum));
            sweep_maximum_local_maxima = std::max(
                sweep_maximum_local_maxima,
                local_maxima);
          }
        }
      }
    }
    require(
        sampled_maximum_envelope_ratio <
            EpairRejectionEnvelopeSafety,
        "sampled Epair rejection envelope exceeds its safety "
        "factor");
    require(
        sweep_maximum_envelope_ratio <
            EpairRejectionEnvelopeSafety,
        "systematic Epair rejection envelope exceeds its safety "
        "factor");
    require(
        sampled_minimum_acceptance > 0.05 &&
            sweep_minimum_acceptance > 0.05,
        "Epair rejection proposal has insufficient acceptance");
    require(
        sweep_maximum_local_maxima <= 1,
        "transformed Epair rejection density is not unimodal");

    std::size_t constexpr DistributionSamples = 16384;
    std::size_t constexpr DistributionGrid = 16384;
    std::array<std::pair<double, double>, 3> const
        distribution_configurations{{
            {1.e3, 0.25},
            {1.e8, 0.50},
            {1.e12, 0.95},
        }};
    double maximum_distribution_ks = 0.;
    std::uint64_t distribution_rejection_trials = 0;
    for (std::size_t configuration_index = 0;
         configuration_index <
         distribution_configurations.size();
         ++configuration_index) {
      auto const [energy_MeV, loss_coordinate] =
          distribution_configurations[configuration_index];
      auto const limits = rate_model.GetKinematicLimits(
          PROPOSAL::EMinusDef(), component, energy_MeV);
      auto const v =
          limits.v_min +
          loss_coordinate *
              (limits.v_max - limits.v_min);
      std::vector<double> transformed_samples;
      transformed_samples.reserve(DistributionSamples);
      double rho_max = 0.;
      for (std::size_t sample_index = 0;
           sample_index < DistributionSamples;
           ++sample_index) {
        RandomNumberKey key{
            0x4550414952444953ULL,
            200 + configuration_index,
            1000000 * (configuration_index + 1) +
                sample_index,
            1,
            static_cast<std::uint32_t>(
                ElectronPairProcessId),
            EpairRhoDrawId};
        auto const rejection = sampleEpairRhoRejection(
            snapshot, snapshot.components[0].component_hash,
            energy_MeV, v, 0.75, key);
        require(
            rejection.sample.status ==
                EpairFinalStateStatus::Success,
            "Epair distribution rejection sample failed");
        require(rejection.sample.rho_max > 0.,
                "Epair distribution rho_max is invalid");
        rho_max = rejection.sample.rho_max;
        distribution_rejection_trials +=
            rejection.trial_count;
        auto const normalized_rho =
            std::min(
                1.,
                std::abs(rejection.sample.rho) / rho_max);
        transformed_samples.push_back(
            2. / 3.14159265358979323846 *
            std::asin(std::sqrt(normalized_rho)));
      }
      std::sort(
          transformed_samples.begin(),
          transformed_samples.end());

      std::vector<double> target_weights(
          DistributionGrid);
      double target_total = 0.;
      for (std::size_t point = 0;
           point < DistributionGrid; ++point) {
        auto const uniform =
            (static_cast<double>(point) + 0.5) /
            DistributionGrid;
        target_weights[point] =
            epair_final_state_detail::
                transformedKkpRhoWeight(
                    snapshot.components[0],
                    snapshot.lepton_mass_MeV,
                    snapshot.electron_mass_MeV,
                    energy_MeV, v, rho_max, uniform);
        target_total += target_weights[point];
      }
      require(target_total > 0.,
              "Epair distribution target is empty");
      double target_cumulative = 0.;
      std::size_t empirical_count = 0;
      for (std::size_t point = 0;
           point < DistributionGrid; ++point) {
        auto const uniform =
            (static_cast<double>(point) + 0.5) /
            DistributionGrid;
        target_cumulative += target_weights[point];
        while (
            empirical_count < transformed_samples.size() &&
            transformed_samples[empirical_count] <= uniform) {
          ++empirical_count;
        }
        auto const target_cdf =
            target_cumulative / target_total;
        auto const empirical_cdf =
            static_cast<double>(empirical_count) /
            DistributionSamples;
        maximum_distribution_ks = std::max(
            maximum_distribution_ks,
            std::abs(empirical_cdf - target_cdf));
      }
    }
    require(
        maximum_distribution_ks < 0.025,
        "Epair rejection samples fail the transformed-density "
        "Kolmogorov-Smirnov requirement");

    auto const repeat =
        backend.generateBremsFinalStatesForValidation(
            interactions, FirstHistory);
    require(
        repeat.final_state_records.size() ==
                result.final_state_records.size() &&
            repeat.secondaries.size() ==
                result.secondaries.size() &&
            repeat.lpm_suppressed.size() ==
                result.lpm_suppressed.size(),
        "Epair repeat counts differ");
    for (std::size_t index = 0;
         index < result.secondaries.size(); ++index) {
      require(
          repeat.secondaries[index].energy_GeV ==
                  result.secondaries[index].energy_GeV &&
              repeat.secondaries[index].history_id ==
                  result.secondaries[index].history_id,
          "Epair final state is not deterministic");
    }

    auto const& statistics = backend.statistics();
    if (!(statistics.electron_pair_final_states ==
                  2 * expected.accepted.size() &&
              statistics.electron_pair_lpm_trials ==
                  2 * ValidCount &&
              statistics.electron_pair_lpm_suppressions ==
                  2 * expected.suppressed.size() &&
              statistics.electron_pair_rejection_trials ==
                  2 * expected.rejection_trials &&
              statistics.electron_pair_zero_weight_samples ==
                  2 * expected.zero_weight_samples &&
              // The deliberately missing component exercises one protected
              // rejection-sampler fallback in each of the two runs.
              statistics.electron_pair_rejection_fallbacks == 2 &&
              statistics.electron_pair_envelope_violations == 0)) {
      std::ostringstream details;
      details
          << "Epair backend statistics differ: final states="
          << statistics.electron_pair_final_states
          << ", LPM trials/suppressions="
          << statistics.electron_pair_lpm_trials << "/"
          << statistics.electron_pair_lpm_suppressions
          << ", rejection trials="
          << statistics.electron_pair_rejection_trials
          << " expected=" << 2 * expected.rejection_trials
          << ", zero weight="
          << statistics.electron_pair_zero_weight_samples
          << " expected=" << 2 * expected.zero_weight_samples
          << ", rejection fallback/envelope="
          << statistics.electron_pair_rejection_fallbacks
          << "/"
          << statistics.electron_pair_envelope_violations;
      throw std::runtime_error(details.str());
    }

    std::filesystem::remove(temporary);
    std::cout
        << "GPU Epair final-state validation passed " << checks
        << " checks for " << ValidCount
        << " PROPOSAL interactions; accepted="
        << expected.accepted.size()
        << ", suppressed=" << expected.suppressed.size()
        << ", rejection trials=" << expected.rejection_trials
        << ", zero-weight=" << expected.zero_weight_samples
        << ", max normalized rho error="
        << maximum_normalized_rho_error
        << ", sampled envelope ratio/min acceptance="
        << sampled_maximum_envelope_ratio << "/"
        << sampled_minimum_acceptance
        << ", systematic envelope ratio/min acceptance/maxima="
        << sweep_maximum_envelope_ratio << "/"
        << sweep_minimum_acceptance << "/"
        << sweep_maximum_local_maxima
        << ", distribution KS/trials="
        << maximum_distribution_ks << "/"
        << distribution_rejection_trials
        << '\n';
  } catch (std::exception const& error) {
    std::error_code ignored;
    std::filesystem::remove(temporary, ignored);
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
  }
  return 0;
}

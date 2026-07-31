/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include <cuda_runtime_api.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <GpuEmFlatRateTableFixture.hpp>

#include <PROPOSAL/Constants.h>
#include <PROPOSAL/crosssection/parametrization/PhotoPairProduction.h>
#include <PROPOSAL/medium/Components.h>
#include <PROPOSAL/medium/Medium.h>
#include <PROPOSAL/particle/ParticleDef.h>
#include <PROPOSAL/secondaries/parametrization/compton/NaivCompton.h>
#include <PROPOSAL/secondaries/parametrization/photoeffect/PhotoeffectNoDeflection.h>

#include <corsika/gpu/em/CudaEmBackend.hpp>
#include <corsika/gpu/em/CudaPhotonPairFinalState.hpp>
#include <corsika/gpu/em/Philox.hpp>
#include <corsika/gpu/em/ProcessCapabilities.hpp>
#include <corsika/gpu/em/ProposalFallback.hpp>
#include <corsika/gpu/em/tables/FlatRateTable.hpp>

namespace {

  using namespace corsika::gpu::em;
  using namespace corsika::gpu::em::tables;

  constexpr double Pi = 3.1415926535897932384626433832795;
  std::size_t checks = 0;

  void require(bool condition, std::string const& message) {
    ++checks;
    if (!condition) {
      throw std::runtime_error(message);
    }
  }

  template <typename Function>
  void requireThrows(Function function, std::string const& message) {
    ++checks;
    try {
      function();
    } catch (std::exception const&) {
      return;
    }
    throw std::runtime_error(message);
  }

  void requireClose(double actual, double expected, double tolerance,
                    std::string const& message) {
    auto const scale =
        std::max({1., std::abs(actual), std::abs(expected)});
    ++checks;
    if (std::abs(actual - expected) > tolerance * scale) {
      std::ostringstream detail;
      detail << message << ": actual=" << std::setprecision(17)
             << actual << ", expected=" << expected
             << ", difference=" << actual - expected;
      throw std::runtime_error(detail.str());
    }
  }

  std::size_t processCount(RateTableSet const& source) {
    std::size_t count = 0;
    for (auto const& particle : source.particles) {
      count += particle.columns.size();
    }
    return count;
  }

  std::filesystem::path temporaryPath() {
    auto const stamp =
        std::chrono::high_resolution_clock::now()
            .time_since_epoch()
            .count();
    return std::filesystem::temp_directory_path() /
           ("c8_gpu_photon_pair_" + std::to_string(stamp) +
            ".c8emrt");
  }

  std::vector<EmParticleState> makeParticles(
      RateTableSet const& source, std::size_t count) {
    std::vector<EmParticleState> particles;
    particles.reserve(count);
    auto const log_min =
        std::log(source.metadata.energy_min_MeV);
    auto const log_max =
        std::log(source.metadata.energy_max_MeV);
    for (std::size_t i = 0; i < count; ++i) {
      auto const energy_fraction =
          std::fmod(
              (static_cast<double>(i) + 0.5) *
                  0.6180339887498948482,
              1.);
      EmParticleState particle{};
      particle.pid =
          source.particles[i % source.particles.size()].pdg_id;
      particle.energy_GeV =
          std::exp(log_min +
                   energy_fraction * (log_max - log_min)) /
          1000.;
      particle.position_m[0] = 17.;
      particle.position_m[1] = -9.;
      particle.position_m[2] = 3100.;
      particle.direction[0] = 0.36;
      particle.direction[1] = 0.48;
      particle.direction[2] = -0.8;
      particle.time_s = 2.5e-6;
      particle.weight = 0.75;
      particle.history_id = i + 1;
      particle.step_id = i % 11;
      particles.push_back(particle);
    }
    return particles;
  }

  double boundedCosine(double value) {
    return std::max(-1., std::min(1., value));
  }

  double sauterCosine(double energy_GeV, double uniform) {
    auto const mass_squared =
        ElectronMassGeV * ElectronMassGeV;
    auto const momentum =
        std::sqrt(std::max(
            0., energy_GeV * energy_GeV -
                    mass_squared));
    auto const coordinate = 2. * uniform - 1.;
    auto const energy_minus_momentum =
        mass_squared / (energy_GeV + momentum);
    auto const one_minus_cosine =
        energy_minus_momentum * (1. - coordinate) /
        (momentum * coordinate + energy_GeV);
    return boundedCosine(1. - one_minus_cosine);
  }

  bool comptonCosines(
      double energy_GeV, double v,
      double& photon_cosine,
      double& electron_cosine) {
    auto const scattered_energy =
        energy_GeV * (1. - v);
    auto const electron_momentum_squared =
        2. * v * energy_GeV * ElectronMassGeV +
        v * v * energy_GeV * energy_GeV;
    if (!(energy_GeV > 0.) || !(v > 0.) ||
        !(v < 1.) || !(scattered_energy > 0.) ||
        !(electron_momentum_squared > 0.)) {
      return false;
    }
    photon_cosine =
        1. - v * ElectronMassGeV / scattered_energy;
    electron_cosine =
        v * (energy_GeV + ElectronMassGeV) /
        std::sqrt(electron_momentum_squared);
    if (!std::isfinite(photon_cosine) ||
        !std::isfinite(electron_cosine) ||
        photon_cosine < -1. - 1.e-12 ||
        photon_cosine > 1. + 1.e-12 ||
        electron_cosine < -1. - 1.e-12 ||
        electron_cosine > 1. + 1.e-12) {
      return false;
    }
    photon_cosine = boundedCosine(photon_cosine);
    electron_cosine = boundedCosine(electron_cosine);
    return true;
  }

  bool photoelectricEnergy(
      PhotonPairLpmSnapshot const& snapshot,
      std::uint64_t component_hash,
      double parent_energy_GeV,
      double& electron_total_energy_GeV,
      double& kinetic_fraction) {
    if (!(parent_energy_GeV > 0.) ||
        !(snapshot.fine_structure_constant > 0.)) {
      return false;
    }
    auto const found = std::find_if(
        snapshot.components,
        snapshot.components + snapshot.component_count,
        [&](PhotonPairLpmComponentSnapshot const& component) {
          return component.component_hash == component_hash;
        });
    if (found ==
            snapshot.components + snapshot.component_count ||
        !(found->nuclear_charge > 0.)) {
      return false;
    }
    auto const z_alpha =
        found->nuclear_charge *
        snapshot.fine_structure_constant;
    auto const binding_energy_GeV =
        z_alpha * z_alpha * ElectronMassGeV / 2.;
    auto const kinetic_energy_GeV =
        parent_energy_GeV - binding_energy_GeV;
    if (!(kinetic_energy_GeV >= 0.)) {
      return false;
    }
    electron_total_energy_GeV =
        ElectronMassGeV + kinetic_energy_GeV;
    kinetic_fraction =
        kinetic_energy_GeV / parent_energy_GeV;
    return std::isfinite(electron_total_energy_GeV) &&
           std::isfinite(kinetic_fraction);
  }

  void deflect(double const input[3], double cosine,
               double azimuth, double output[3]) {
    auto const transverse =
        std::sqrt(std::max(
            0., input[0] * input[0] + input[1] * input[1]));
    double cosine_phi = 1.;
    double sine_phi = 0.;
    if (transverse > 0.) {
      cosine_phi = input[0] / transverse;
      sine_phi = input[1] / transverse;
    }
    auto const cosine_theta = input[2];
    auto const sine_theta = transverse;
    double const rotation_x[3]{
        cosine_theta * cosine_phi,
        cosine_theta * sine_phi,
        -sine_theta};
    double const rotation_y[3]{-sine_phi, cosine_phi, 0.};
    auto const sine =
        std::sqrt(std::max(
            0., (1. - cosine) * (1. + cosine)));
    auto const local_x = sine * std::cos(azimuth);
    auto const local_y = sine * std::sin(azimuth);
    auto local_z = std::sqrt(std::max(
        0., 1. - local_x * local_x - local_y * local_y));
    if (cosine < 0.) {
      local_z = -local_z;
    }
    for (std::size_t axis = 0; axis < 3; ++axis) {
      output[axis] = local_z * input[axis] +
                     local_x * rotation_x[axis] +
                     local_y * rotation_y[axis];
    }
  }

  struct HostFinalStates {
    EmFinalStateBatchResult result;
  };

  HostFinalStates generateOnHost(
      FlatRateTableView table,
      PhotonPairLpmSnapshot const& lpm_snapshot,
      std::vector<EmInteractionRecord> const& interactions,
      std::uint64_t seed, std::uint64_t shower_id,
      std::uint64_t first_history_id) {
    HostFinalStates expected;
    expected.result.input_interactions = interactions.size();
    for (auto const& interaction : interactions) {
      if (interaction.status ==
          EmInteractionStatus::NoDiscreteInteraction) {
        expected.result.continuations.push_back(interaction);
        continue;
      }
      auto const capability = gpuProcessCapability(
          interaction.particle.pid, interaction.process_id);
      if (capability != GpuProcessCapability::PhotonPair &&
          capability != GpuProcessCapability::Compton &&
          capability != GpuProcessCapability::Photoelectric) {
        expected.result.fallback_events.push_back(
            makeProcessFallbackEvent(
                interaction, processFallbackReason(capability)));
        continue;
      }

      auto const& parent = interaction.particle;
      if (capability == GpuProcessCapability::Compton) {
        double photon_cosine = 0.;
        double electron_cosine = 0.;
        auto const v = interaction.energy_fraction;
        if (!comptonCosines(
                parent.energy_GeV, v, photon_cosine,
                electron_cosine)) {
          expected.result.fallback_events.push_back(
              makeProcessFallbackEvent(
                  interaction,
                  ProposalFallbackReason::InvalidFinalState));
          continue;
        }
        RandomNumberKey const key{
            seed, shower_id, parent.history_id,
            parent.step_id,
            static_cast<std::uint32_t>(ComptonProcessId),
            ComptonAzimuthDrawId};
        auto const azimuth_uniform = uniformOpen01(key);
        auto const azimuth = 2. * Pi * azimuth_uniform;
        auto const child_offset =
            expected.result.secondaries.size();
        auto photon = parent;
        photon.energy_GeV =
            parent.energy_GeV * (1. - v);
        photon.parent_history_id = parent.history_id;
        photon.history_id = first_history_id + child_offset;
        photon.generation = parent.generation + 1;
        photon.step_id = 0;
        photon.reserved = 0;
        deflect(parent.direction, photon_cosine, azimuth,
                photon.direction);
        auto electron = parent;
        electron.pid =
            static_cast<std::int32_t>(EmPid::Electron);
        electron.energy_GeV =
            ElectronMassGeV + parent.energy_GeV * v;
        electron.parent_history_id = parent.history_id;
        electron.history_id =
            first_history_id + child_offset + 1;
        electron.generation = parent.generation + 1;
        electron.step_id = 0;
        electron.reserved = 0;
        deflect(
            parent.direction, electron_cosine,
            std::fmod(azimuth + Pi, 2. * Pi),
            electron.direction);
        expected.result.final_state_records.push_back(
            PhotonPairFinalStateRecord{
                interaction.input_index,
                parent.history_id,
                child_offset,
                2,
                ComptonProcessId,
                v,
                interaction.loss_quantile,
                azimuth_uniform,
                0.,
                0.,
                0.,
                0.,
                interaction.loss_draw_id,
                ComptonAzimuthDrawId,
                0,
                0,
                0});
        expected.result.secondaries.push_back(photon);
        expected.result.secondaries.push_back(electron);
        ++expected.result.compton_interactions;
        continue;
      }

      if (capability ==
          GpuProcessCapability::Photoelectric) {
        double electron_energy_GeV = 0.;
        double kinetic_fraction = 0.;
        if (std::abs(interaction.energy_fraction - 1.) >
                1.e-12 ||
            !photoelectricEnergy(
                lpm_snapshot, interaction.component_hash,
                parent.energy_GeV, electron_energy_GeV,
                kinetic_fraction)) {
          expected.result.fallback_events.push_back(
              makeProcessFallbackEvent(
                  interaction,
                  ProposalFallbackReason::InvalidFinalState));
          continue;
        }
        auto const child_offset =
            expected.result.secondaries.size();
        auto electron = parent;
        electron.pid =
            static_cast<std::int32_t>(EmPid::Electron);
        electron.energy_GeV = electron_energy_GeV;
        electron.parent_history_id = parent.history_id;
        electron.history_id =
            first_history_id + child_offset;
        electron.generation = parent.generation + 1;
        electron.step_id = 0;
        electron.reserved = 0;
        expected.result.final_state_records.push_back(
            PhotonPairFinalStateRecord{
                interaction.input_index,
                parent.history_id,
                child_offset,
                1,
                PhotoelectricProcessId,
                kinetic_fraction,
                interaction.loss_quantile,
                0.,
                0.,
                0.,
                0.,
                0.,
                interaction.loss_draw_id,
                0,
                0,
                0,
                0});
        expected.result.secondaries.push_back(electron);
        ++expected.result.photoelectric_interactions;
        continue;
      }

      if (parent.energy_GeV < 2. * ElectronMassGeV ||
          parent.generation ==
              std::numeric_limits<std::uint32_t>::max()) {
        expected.result.fallback_events.push_back(
            makeProcessFallbackEvent(
                interaction,
                ProposalFallbackReason::InvalidFinalState));
        continue;
      }
      RandomNumberKey key{
          seed, shower_id, parent.history_id, parent.step_id,
          static_cast<std::uint32_t>(PhotonPairProcessId),
          PhotonPairSplitDrawId};
      auto const split_uniform = uniformOpen01(key);
      TableQuery const split_query{
          TableQueryKind::LossFraction, parent.pid,
          PhotonPairFinalStateProcessId,
          interaction.component_hash,
          parent.energy_GeV * 1000., split_uniform};
      auto const split = executeTableQuery(table, split_query);
      double split_fraction = 0.;
      std::uint64_t split_draw_id = PhotonPairSplitDrawId;
      double final_state_uniform = split_uniform;
      bool split_ready = false;
      if (split.status == TableLookupStatus::Success) {
        split_ready = decodePhotonPairNormalizedSplit(
            parent.energy_GeV, split.value,
            split_fraction);
        if (!split_ready) {
          auto event = makeProcessFallbackEvent(
              interaction,
              ProposalFallbackReason::InvalidFinalState);
          event.final_state_uniform = split_uniform;
          event.final_state_draw_id = split_draw_id;
          expected.result.fallback_events.push_back(event);
          continue;
        }
      } else if (
          split.status ==
          TableLookupStatus::LossEnergyOutOfRange) {
        for (std::uint32_t attempt = 0;
             attempt < PhotonPairAnalyticMaximumAttempts; ++attempt) {
          key.draw_id =
              PhotonPairAnalyticCandidateDrawIdBase + attempt;
          auto const candidate_uniform = uniformOpen01(key);
          final_state_uniform = candidate_uniform;
          auto acceptance_key = key;
          acceptance_key.draw_id =
              PhotonPairAnalyticAcceptanceDrawIdBase + attempt;
          auto const trial = photonPairFinalStateTrial(
              lpm_snapshot, interaction.component_hash,
              parent.energy_GeV * 1000., candidate_uniform,
              uniformOpen01(acceptance_key));
          split_draw_id = key.draw_id;
          if (trial.status !=
              PhotonPairFinalStateStatus::Success) {
            auto event = makeProcessFallbackEvent(
                interaction,
                ProposalFallbackReason::InvalidFinalState);
            event.final_state_uniform = final_state_uniform;
            event.final_state_draw_id = split_draw_id;
            expected.result.fallback_events.push_back(event);
            break;
          }
          if (trial.accepted != 0) {
            split_fraction = trial.split_fraction;
            split_ready = true;
            break;
          }
        }
        if (!split_ready) {
          if (expected.result.fallback_events.empty() ||
              expected.result.fallback_events.back()
                      .particle.history_id !=
                  parent.history_id) {
            auto event = makeProcessFallbackEvent(
                interaction,
                ProposalFallbackReason::InvalidFinalState);
            event.final_state_uniform = final_state_uniform;
            event.final_state_draw_id = split_draw_id;
            expected.result.fallback_events.push_back(event);
          }
          continue;
        }
      } else {
        auto event = makeTableFallbackEvent(
            parent, split_query, split, PhotonPairSplitDrawId,
            interaction.input_index);
        event.energy_fraction = interaction.energy_fraction;
        event.process_id = interaction.process_id;
        event.loss_quantile = interaction.loss_quantile;
        event.random_draw_id = interaction.loss_draw_id;
        event.final_state_uniform = split_uniform;
        event.final_state_draw_id = PhotonPairSplitDrawId;
        expected.result.fallback_events.push_back(event);
        continue;
      }

      auto const electron_energy =
          parent.energy_GeV * split_fraction;
      auto const positron_energy =
          parent.energy_GeV - electron_energy;
      if (!(split_fraction > 0.) ||
          !(split_fraction < 1.) ||
          electron_energy < ElectronMassGeV ||
          positron_energy < ElectronMassGeV) {
        auto event = makeProcessFallbackEvent(
            interaction,
            ProposalFallbackReason::InvalidFinalState);
        event.final_state_uniform = final_state_uniform;
        event.final_state_draw_id = split_draw_id;
        expected.result.fallback_events.push_back(event);
        continue;
      }

      key.draw_id = PhotonPairAzimuthDrawId;
      auto const azimuth_uniform = uniformOpen01(key);
      key.draw_id = PhotonPairElectronPolarDrawId;
      auto const electron_polar_uniform = uniformOpen01(key);
      key.draw_id = PhotonPairPositronPolarDrawId;
      auto const positron_polar_uniform = uniformOpen01(key);
      auto const lpm = photonPairLpmSuppressionFactor(
          lpm_snapshot, interaction.component_hash,
          parent.energy_GeV * 1000., split_fraction,
          interaction.mass_density_g_per_cm3);
      if (lpm.status != PhotonPairLpmStatus::Success) {
        expected.result.fallback_events.push_back(
            makeProcessFallbackEvent(
                interaction,
                lpm.status == PhotonPairLpmStatus::InvalidInput
                    ? ProposalFallbackReason::InvalidMassDensity
                    : ProposalFallbackReason::
                          LpmParametersUnavailable));
        continue;
      }
      key.draw_id = PhotonPairLpmDrawId;
      auto const lpm_uniform = uniformOpen01(key);
      if (lpm_uniform > lpm.survival_probability) {
        auto suppressed_parent = parent;
        ++suppressed_parent.step_id;
        expected.result.lpm_suppressed.push_back(
            {suppressed_parent,
             interaction.input_index,
             interaction.component_hash,
             lpm.survival_probability,
             lpm_uniform,
             PhotonPairLpmDrawId});
        continue;
      }
      auto const child_offset =
          expected.result.secondaries.size();
      auto electron = parent;
      electron.pid = static_cast<std::int32_t>(EmPid::Electron);
      electron.energy_GeV = electron_energy;
      electron.parent_history_id = parent.history_id;
      electron.history_id = first_history_id + child_offset;
      electron.generation = parent.generation + 1;
      electron.step_id = 0;
      electron.reserved = 0;
      auto const azimuth = 2. * Pi * azimuth_uniform;
      deflect(parent.direction,
              sauterCosine(
                  electron_energy, electron_polar_uniform),
              azimuth, electron.direction);

      auto positron = parent;
      positron.pid = static_cast<std::int32_t>(EmPid::Positron);
      positron.energy_GeV = positron_energy;
      positron.parent_history_id = parent.history_id;
      positron.history_id = first_history_id + child_offset + 1;
      positron.generation = parent.generation + 1;
      positron.step_id = 0;
      positron.reserved = 0;
      deflect(parent.direction,
              sauterCosine(
                  positron_energy, positron_polar_uniform),
              std::fmod(azimuth + Pi, 2. * Pi),
              positron.direction);

      expected.result.final_state_records.push_back(
          PhotonPairFinalStateRecord{
              interaction.input_index,
              parent.history_id,
              child_offset,
              2,
              PhotonPairProcessId,
              split_fraction,
              final_state_uniform,
              azimuth_uniform,
              electron_polar_uniform,
              positron_polar_uniform,
              lpm.survival_probability,
              lpm_uniform,
              split_draw_id,
              PhotonPairAzimuthDrawId,
              PhotonPairElectronPolarDrawId,
              PhotonPairPositronPolarDrawId,
              PhotonPairLpmDrawId});
      expected.result.secondaries.push_back(electron);
      expected.result.secondaries.push_back(positron);
      ++expected.result.photon_pair_interactions;
    }
    expected.result.gpu_interactions =
        expected.result.final_state_records.size();
    return expected;
  }

  void compareParticle(EmParticleState const& actual,
                       EmParticleState const& expected) {
    require(actual.pid == expected.pid,
            "secondary PID differs");
    require(actual.medium_id == expected.medium_id &&
                actual.generation == expected.generation,
            "secondary transport metadata differs");
    require(actual.history_id == expected.history_id &&
                actual.parent_history_id ==
                    expected.parent_history_id &&
                actual.step_id == expected.step_id,
            "secondary history metadata differs");
    require(actual.time_s == expected.time_s &&
                actual.weight == expected.weight,
            "secondary time or weight differs");
    requireClose(actual.energy_GeV, expected.energy_GeV,
                 3.e-14, "secondary energy differs");
    for (std::size_t axis = 0; axis < 3; ++axis) {
      require(actual.position_m[axis] ==
                  expected.position_m[axis],
              "secondary vertex differs");
      requireClose(actual.direction[axis],
                   expected.direction[axis], 1.e-12,
                   "secondary direction differs");
    }
  }

  void compareRecord(PhotonPairFinalStateRecord const& actual,
                     PhotonPairFinalStateRecord const& expected) {
    require(actual.input_index == expected.input_index &&
                actual.parent_history_id ==
                    expected.parent_history_id,
            "final-state record identity differs");
    require(actual.secondary_offset ==
                expected.secondary_offset &&
                actual.secondary_count ==
                    expected.secondary_count &&
                actual.process_id == expected.process_id,
            "final-state child range differs");
    requireClose(actual.energy_split_fraction,
                 expected.energy_split_fraction, 3.e-14,
                 "photon final-state energy split differs");
    require(actual.split_uniform == expected.split_uniform &&
                actual.azimuth_uniform ==
                    expected.azimuth_uniform &&
                actual.electron_polar_uniform ==
                    expected.electron_polar_uniform &&
                actual.positron_polar_uniform ==
                    expected.positron_polar_uniform &&
                actual.lpm_uniform ==
                    expected.lpm_uniform,
            "photon final-state random values differ");
    requireClose(actual.lpm_survival_probability,
                 expected.lpm_survival_probability, 3.e-14,
                 "photon final-state LPM probability differs");
    require(
        actual.split_draw_id == expected.split_draw_id &&
            actual.azimuth_draw_id ==
                expected.azimuth_draw_id &&
            actual.electron_polar_draw_id ==
                expected.electron_polar_draw_id &&
            actual.positron_polar_draw_id ==
                expected.positron_polar_draw_id &&
            actual.lpm_draw_id == expected.lpm_draw_id,
        "photon final-state draw identities differ");
  }

  void compareFallback(ProposalFallbackEvent const& actual,
                       ProposalFallbackEvent const& expected) {
    require(actual.input_index == expected.input_index &&
                actual.particle.history_id ==
                    expected.particle.history_id,
            "final-state fallback identity differs");
    require(actual.reason == expected.reason &&
                actual.process_id == expected.process_id &&
                actual.component_hash ==
                    expected.component_hash,
            "final-state fallback process differs");
    if (!(actual.energy_fraction ==
              expected.energy_fraction &&
          actual.loss_quantile == expected.loss_quantile &&
          actual.final_state_uniform ==
              expected.final_state_uniform)) {
      std::ostringstream message;
      message << "final-state fallback random payload differs for history "
              << actual.particle.history_id
              << ": actual(Efrac,u_loss,u_final)="
              << std::setprecision(17) << actual.energy_fraction << ","
              << actual.loss_quantile << ","
              << actual.final_state_uniform
              << ", expected=" << expected.energy_fraction << ","
              << expected.loss_quantile << ","
              << expected.final_state_uniform
              << ", draw(actual/expected)="
              << actual.final_state_draw_id << "/"
              << expected.final_state_draw_id;
      require(false, message.str());
    }
    require(actual.random_draw_id ==
                expected.random_draw_id &&
                actual.final_state_draw_id ==
                    expected.final_state_draw_id,
            "final-state fallback draw identity differs");
  }

  void compareBatches(EmFinalStateBatchResult const& actual,
                      EmFinalStateBatchResult const& expected) {
    require(actual.input_interactions ==
                expected.input_interactions &&
                actual.gpu_interactions ==
                    expected.gpu_interactions &&
                actual.photon_pair_interactions ==
                    expected.photon_pair_interactions &&
                actual.compton_interactions ==
                    expected.compton_interactions &&
                actual.photoelectric_interactions ==
                    expected.photoelectric_interactions,
            "final-state batch counts differ");
    require(actual.final_state_records.size() ==
                expected.final_state_records.size() &&
                actual.secondaries.size() ==
                    expected.secondaries.size() &&
                actual.fallback_events.size() ==
                    expected.fallback_events.size() &&
                actual.lpm_suppressed.size() ==
                    expected.lpm_suppressed.size() &&
                actual.continuations.size() ==
                    expected.continuations.size(),
            "final-state compact queue sizes differ");
    for (std::size_t i = 0;
         i < actual.final_state_records.size(); ++i) {
      compareRecord(actual.final_state_records[i],
                    expected.final_state_records[i]);
    }
    for (std::size_t i = 0; i < actual.secondaries.size(); ++i) {
      compareParticle(actual.secondaries[i],
                      expected.secondaries[i]);
    }
    for (std::size_t i = 0;
         i < actual.fallback_events.size(); ++i) {
      compareFallback(actual.fallback_events[i],
                      expected.fallback_events[i]);
    }
    for (std::size_t i = 0;
         i < actual.continuations.size(); ++i) {
      require(actual.continuations[i].particle.history_id ==
                  expected.continuations[i].particle.history_id &&
                  actual.continuations[i].input_index ==
                      expected.continuations[i].input_index,
              "no-interaction continuation order differs");
    }
    for (std::size_t i = 0;
         i < actual.lpm_suppressed.size(); ++i) {
      auto const& left = actual.lpm_suppressed[i];
      auto const& right = expected.lpm_suppressed[i];
      require(left.input_index == right.input_index &&
                  left.component_hash == right.component_hash &&
                  left.particle.history_id ==
                      right.particle.history_id &&
                  left.particle.step_id ==
                      right.particle.step_id &&
                  left.uniform == right.uniform &&
                  left.draw_id == PhotonPairLpmDrawId,
              "LPM-suppressed parent identity differs");
      requireClose(left.survival_probability,
                   right.survival_probability, 3.e-14,
                   "LPM-suppressed probability differs");
      require(left.particle.energy_GeV ==
                  right.particle.energy_GeV &&
                  left.particle.pid == right.particle.pid &&
                  left.particle.weight ==
                      right.particle.weight &&
                  left.particle.generation ==
                      right.particle.generation,
              "LPM suppression changed the parent state");
      for (std::size_t axis = 0; axis < 3; ++axis) {
        require(left.particle.position_m[axis] ==
                    right.particle.position_m[axis] &&
                    left.particle.direction[axis] ==
                        right.particle.direction[axis],
                "LPM suppression changed parent geometry");
      }
    }
  }

  void checkPhysics(
      EmFinalStateBatchResult const& batch,
      std::vector<EmInteractionRecord> const& interactions,
      PhotonPairLpmSnapshot const& lpm_snapshot) {
    std::map<std::uint64_t, EmParticleState const*> parents;
    for (auto const& interaction : interactions) {
      parents.emplace(
          interaction.particle.history_id,
          &interaction.particle);
    }
    std::size_t expected_secondaries = 0;
    for (auto const& record : batch.final_state_records) {
      expected_secondaries += record.secondary_count;
    }
    require(batch.secondaries.size() ==
                expected_secondaries,
            "GPU photon final-state child count does not close");
    for (auto const& record : batch.final_state_records) {
      auto const& first =
          batch.secondaries.at(record.secondary_offset);
      auto const stored_parent =
          parents.find(record.parent_history_id);
      require(stored_parent != parents.end(),
              "photon final-state parent energy is unavailable");
      auto const& parent = *stored_parent->second;
      if (record.process_id == ComptonProcessId) {
        require(record.secondary_count == 2,
                "Compton child count differs");
        auto const& second =
            batch.secondaries.at(record.secondary_offset + 1);
        require(
            first.pid ==
                    static_cast<std::int32_t>(EmPid::Photon) &&
                second.pid ==
                    static_cast<std::int32_t>(EmPid::Electron),
            "Compton child ordering differs");
        requireClose(
            first.energy_GeV +
                second.energy_GeV - ElectronMassGeV,
            parent.energy_GeV, 2.e-15,
            "Compton kinetic energy does not close");
        require(second.energy_GeV >= ElectronMassGeV,
                "Compton electron is below rest mass");
        PROPOSAL::secondaries::NaivCompton reference;
        auto const reference_energies =
            reference.CalculateEnergy(
                parent.energy_GeV * 1000.,
                record.energy_split_fraction);
        requireClose(
            first.energy_GeV,
            std::get<0>(reference_energies) / 1000.,
            3.e-14,
            "GPU Compton photon energy differs from PROPOSAL");
        requireClose(
            second.energy_GeV,
            std::get<1>(reference_energies) / 1000.,
            3.e-14,
            "GPU Compton electron energy differs from PROPOSAL");
        auto const reference_directions =
            reference.CalculateDirections(
                PROPOSAL::Cartesian3D{
                    parent.direction[0],
                    parent.direction[1],
                    parent.direction[2]},
                parent.energy_GeV * 1000.,
                record.energy_split_fraction,
                record.azimuth_uniform);
        auto const gamma_direction =
            std::get<0>(reference_directions)
                .GetCartesianCoordinates();
        auto const electron_direction =
            std::get<1>(reference_directions)
                .GetCartesianCoordinates();
        for (std::size_t axis = 0; axis < 3; ++axis) {
          requireClose(
              first.direction[axis], gamma_direction[axis],
              5.e-12,
              "GPU Compton photon direction differs from PROPOSAL");
          requireClose(
              second.direction[axis],
              electron_direction[axis], 5.e-12,
              "GPU Compton electron direction differs from PROPOSAL");
        }
      } else if (
          record.process_id == PhotoelectricProcessId) {
        require(record.secondary_count == 1 &&
                    first.pid ==
                        static_cast<std::int32_t>(
                            EmPid::Electron),
                "photoelectric child identity differs");
        auto const binding_energy =
            parent.energy_GeV *
            (1. - record.energy_split_fraction);
        requireClose(
            first.energy_GeV - ElectronMassGeV +
                binding_energy,
            parent.energy_GeV, 2.e-15,
            "photoelectric energy and binding deposit do not close");
        for (std::size_t axis = 0; axis < 3; ++axis) {
          require(first.direction[axis] ==
                      parent.direction[axis],
                  "photoelectric electron did not inherit photon direction");
        }
        auto const source_interaction = std::find_if(
            interactions.begin(), interactions.end(),
            [&](EmInteractionRecord const& interaction) {
              return interaction.particle.history_id ==
                     record.parent_history_id;
            });
        require(source_interaction != interactions.end(),
                "photoelectric PROPOSAL oracle lost its target component");
        PROPOSAL::Component component;
        if (source_interaction->component_hash == 101) {
          component = PROPOSAL::Components::Nitrogen{};
        } else if (
            source_interaction->component_hash == 102) {
          component = PROPOSAL::Components::Oxygen{};
        } else if (
            source_interaction->component_hash == 103) {
          component = PROPOSAL::Components::Argon{};
        } else {
          auto const metadata_component = std::find_if(
              lpm_snapshot.components,
              lpm_snapshot.components +
                  lpm_snapshot.component_count,
              [&](PhotonPairLpmComponentSnapshot const& candidate) {
                return candidate.component_hash ==
                       source_interaction->component_hash;
              });
          if (metadata_component ==
              lpm_snapshot.components +
                  lpm_snapshot.component_count) {
            throw std::runtime_error(
                "photoelectric PROPOSAL oracle has no target component");
          }
          auto const charge = static_cast<int>(
              metadata_component->nuclear_charge + 0.5);
          if (charge == 7) {
            component = PROPOSAL::Components::Nitrogen{};
          } else if (charge == 8) {
            component = PROPOSAL::Components::Oxygen{};
          } else if (charge == 18) {
            component = PROPOSAL::Components::Argon{};
          } else {
            throw std::runtime_error(
                "photoelectric PROPOSAL oracle has unsupported charge");
          }
        }
        PROPOSAL::Air air;
        PROPOSAL::secondaries::PhotoeffectNoDeflection reference{
            PROPOSAL::GammaDef{}, air};
        PROPOSAL::Cartesian3D const position{
            parent.position_m[0] * 100.,
            parent.position_m[1] * 100.,
            parent.position_m[2] * 100.};
        PROPOSAL::Cartesian3D const direction{
            parent.direction[0],
            parent.direction[1],
            parent.direction[2]};
        PROPOSAL::StochasticLoss loss{
            static_cast<int>(
                PROPOSAL::InteractionType::Photoeffect),
            parent.energy_GeV * 1000., position, direction,
            parent.time_s, 0.,
            parent.energy_GeV * 1000.,
            component.GetHash()};
        std::vector<double> random_numbers;
        auto const reference_secondaries =
            reference.CalculateSecondaries(
                loss, component, random_numbers);
        require(reference_secondaries.size() == 1,
                "PROPOSAL photoelectric oracle multiplicity differs");
        requireClose(
            first.energy_GeV,
            reference_secondaries.front().energy / 1000.,
            3.e-14,
            "GPU photoelectric energy differs from PROPOSAL");
        auto const reference_direction =
            reference_secondaries.front()
                .direction.GetCartesianCoordinates();
        for (std::size_t axis = 0; axis < 3; ++axis) {
          requireClose(
              first.direction[axis],
              reference_direction[axis], 2.e-13,
              "GPU photoelectric direction differs from PROPOSAL");
        }
      } else {
        require(record.process_id == PhotonPairProcessId,
                "unknown GPU photon final-state process");
        require(record.secondary_count == 2,
                "photon-pair child count differs");
        auto const& electron = first;
        auto const& second =
            batch.secondaries.at(record.secondary_offset + 1);
        auto const& positron = second;
        require(electron.pid ==
                    static_cast<std::int32_t>(EmPid::Electron) &&
                    positron.pid ==
                        static_cast<std::int32_t>(EmPid::Positron),
                "photon-pair child ordering differs");
        auto const parent_energy =
            electron.energy_GeV + positron.energy_GeV;
        requireClose(parent_energy, parent.energy_GeV, 2.e-15,
                     "photon-pair energy does not close");
        require(electron.energy_GeV >= ElectronMassGeV &&
                    positron.energy_GeV >= ElectronMassGeV,
                "photon-pair child is below its rest mass");
      }
      for (std::size_t child_index = 0;
           child_index < record.secondary_count;
           ++child_index) {
        auto const& child = batch.secondaries.at(
            record.secondary_offset + child_index);
        auto const direction_norm =
            child.direction[0] * child.direction[0] +
            child.direction[1] * child.direction[1] +
            child.direction[2] * child.direction[2];
        require(std::abs(direction_norm - 1.) < 2.e-13,
                "photon final-state direction is not normalized");
      }
    }
  }

  RateColumn const& photonPairColumn(RateTableSet const& source) {
    auto const& photon = findParticle(
        source, static_cast<std::int32_t>(EmPid::Photon));
    auto const found = std::find_if(
        photon.columns.begin(), photon.columns.end(),
        [](RateColumn const& column) {
          return column.process_id == PhotonPairProcessId;
        });
    if (found == photon.columns.end()) {
      throw std::runtime_error(
          "test table has no photon-pair column");
    }
    return *found;
  }

  void validateAnalyticPhotonPairSampler() {
    auto const medium = PROPOSAL::Air();
    auto const components = medium.GetComponents();
    PhotonPairLpmSnapshot snapshot{};
    snapshot.fine_structure_constant = PROPOSAL::ALPHA;
    snapshot.component_count =
        static_cast<std::uint32_t>(components.size());
    for (std::size_t index = 0; index < components.size();
         ++index) {
      snapshot.components[index] = {
          static_cast<std::uint64_t>(
              components[index].GetHash()),
          components[index].GetNucCharge(),
          components[index].GetLogConstant()};
    }

    PROPOSAL::crosssection::PhotoPairKochMotz reference;
    PROPOSAL::GammaDef const photon;
    std::uint64_t accepted_samples = 0;
    std::uint64_t total_trials = 0;
    double maximum_shape_error = 0.;
    double maximum_variance_error = 0.;
    for (auto const& component : components) {
      auto const component_hash =
          static_cast<std::uint64_t>(component.GetHash());
      for (double const energy_MeV :
           {4., 5., 10., 49.9, 50., 100., 1.e3, 9999.}) {
        auto const lower = ElectronMassMeV / energy_MeV;
        auto const width = 1. - 2. * lower;
        require(width > 0.,
                "analytical photon-pair test has no phase space");

        auto const center_reference =
            reference.DifferentialCrossSection(
                photon, component, energy_MeV, 0.5);
        auto const center_trial = photonPairFinalStateTrial(
            snapshot, component_hash, energy_MeV, 0.5, 0.5);
        require(
            center_trial.status ==
                    PhotonPairFinalStateStatus::Success &&
                center_reference > 0. &&
                center_trial.differential_weight > 0.,
            "valid analytical photon-pair center was rejected");
        for (double const coordinate :
             {0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99}) {
          auto const trial = photonPairFinalStateTrial(
              snapshot, component_hash, energy_MeV,
              coordinate, 0.5);
          require(
              trial.status ==
                  PhotonPairFinalStateStatus::Success,
              "valid Koch--Motz analytical trial failed");
          auto const split = lower + width * coordinate;
          auto const proposal =
              reference.DifferentialCrossSection(
                  photon, component, energy_MeV, split);
          auto const proposal_ratio =
              proposal / center_reference;
          auto const device_ratio =
              trial.differential_weight /
              center_trial.differential_weight;
          auto const relative_error =
              std::abs(device_ratio - proposal_ratio) /
              std::max(1.e-15, std::abs(proposal_ratio));
          maximum_shape_error =
              std::max(maximum_shape_error, relative_error);
          require(
              relative_error < 2.e-7,
              "analytical Koch--Motz shape differs from PROPOSAL");
          require(
              trial.differential_weight <=
                  trial.envelope_weight *
                      (1. + 1.e-13),
              "analytical Koch--Motz envelope was exceeded");
        }

        // Numerically integrate the normalized-coordinate second moment.
        // Symmetry fixes the mean to 1/2, while the variance distinguishes a
        // uniform proposal from the accepted Koch--Motz distribution.
        constexpr std::size_t IntegrationBins = 8192;
        double normalization = 0.;
        double second_moment = 0.;
        for (std::size_t bin = 0; bin < IntegrationBins; ++bin) {
          auto const coordinate =
              (static_cast<double>(bin) + 0.5) /
              IntegrationBins;
          auto const trial = photonPairFinalStateTrial(
              snapshot, component_hash, energy_MeV,
              coordinate, 0.5);
          require(
              trial.status ==
                  PhotonPairFinalStateStatus::Success,
              "Koch--Motz integration point failed");
          auto const displacement = coordinate - 0.5;
          normalization += trial.differential_weight;
          second_moment +=
              trial.differential_weight *
              displacement * displacement;
        }
        require(normalization > 0.,
                "Koch--Motz integral is not positive");
        auto const expected_variance =
            second_moment / normalization;

        constexpr std::size_t SampleCount = 10000;
        double sampled_variance = 0.;
        for (std::size_t sample_index = 0;
             sample_index < SampleCount; ++sample_index) {
          bool accepted = false;
          for (std::uint32_t attempt = 0;
               attempt < PhotonPairAnalyticMaximumAttempts;
               ++attempt) {
            RandomNumberKey candidate_key{
                0x50414952414e414cULL,
                static_cast<std::uint64_t>(energy_MeV * 10.),
                static_cast<std::uint64_t>(sample_index + 1),
                component_hash,
                static_cast<std::uint32_t>(PhotonPairProcessId),
                PhotonPairAnalyticCandidateDrawIdBase + attempt};
            auto acceptance_key = candidate_key;
            acceptance_key.draw_id =
                PhotonPairAnalyticAcceptanceDrawIdBase + attempt;
            auto const trial = photonPairFinalStateTrial(
                snapshot, component_hash, energy_MeV,
                uniformOpen01(candidate_key),
                uniformOpen01(acceptance_key));
            ++total_trials;
            require(
                trial.status ==
                    PhotonPairFinalStateStatus::Success,
                "valid sampled Koch--Motz trial failed");
            if (trial.accepted != 0) {
              double coordinate = 0.;
              require(
                  encodePhotonPairNormalizedSplit(
                      energy_MeV / 1000.,
                      trial.split_fraction, coordinate),
                  "sampled Koch--Motz split is outside phase space");
              auto const displacement = coordinate - 0.5;
              sampled_variance +=
                  displacement * displacement;
              ++accepted_samples;
              accepted = true;
              break;
            }
          }
          require(
              accepted,
              "Koch--Motz sampler exhausted its fixed attempt budget");
        }
        sampled_variance /= SampleCount;
        auto const variance_error =
            std::abs(sampled_variance - expected_variance);
        maximum_variance_error =
            std::max(maximum_variance_error, variance_error);
        require(
            variance_error < 2.5e-3,
            "sampled Koch--Motz variance differs from numerical CDF");
      }
    }

    for (auto const& component : components) {
      auto const component_hash =
          static_cast<std::uint64_t>(component.GetHash());
      for (double const energy_MeV :
           {1.0221, 1.023, 1.05, 1.1, 1.5, 2., 3., 3.9}) {
        for (std::size_t index = 0; index <= 512; ++index) {
          auto const coordinate =
              (static_cast<double>(index) + 0.5) / 513.;
          auto const trial = photonPairFinalStateTrial(
              snapshot, component_hash, energy_MeV,
              coordinate, 0.5);
          require(
              trial.status ==
                      PhotonPairFinalStateStatus::Success &&
                  trial.differential_weight >= 0. &&
                  trial.envelope_weight > 0. &&
                  trial.differential_weight <=
                      trial.envelope_weight *
                          (1. + 1.e-13),
              "endpoint-regularized Koch--Motz trial is invalid");
        }
      }
    }
    auto const threshold = photonPairFinalStateTrial(
        snapshot,
        static_cast<std::uint64_t>(components.front().GetHash()),
        PhotonPairThresholdMeV, 0.5, 0.5);
    require(
        threshold.status ==
            PhotonPairFinalStateStatus::InvalidInput,
        "zero-phase-space pair threshold was not rejected");
    auto const missing_component = photonPairFinalStateTrial(
        snapshot, 0xdeadbeefULL, 100., 0.5, 0.5);
    require(
        missing_component.status ==
            PhotonPairFinalStateStatus::ComponentNotFound,
        "missing Koch--Motz component was not rejected");
    require(
        total_trials < 3 * accepted_samples,
        "Koch--Motz analytical sampler is unexpectedly inefficient");
    std::cout
        << "Analytical Koch--Motz validation: max shape error "
        << maximum_shape_error << ", max variance error "
        << maximum_variance_error << ", mean trials "
        << static_cast<double>(total_trials) /
               static_cast<double>(accepted_samples)
        << '\n';
  }

} // namespace

int main(int argc, char** argv) {
  int device_count = 0;
  auto const cuda_status = cudaGetDeviceCount(&device_count);
  if (cuda_status != cudaSuccess || device_count == 0) {
    std::cerr << "SKIP: no CUDA device is accessible: "
              << cudaGetErrorString(cuda_status) << '\n';
    return 77;
  }
  if (argc > 2) {
    std::cerr
        << "usage: testGpuPhotonPairFinalState [RATE_TABLE]\n";
    return 2;
  }

  std::filesystem::path temporary;
  try {
    validateAnalyticPhotonPairSampler();
    auto source =
        argc == 2 ? readRateTable(argv[1])
                  : testing::makeFlatRateTableFixture();
    std::filesystem::path table_path;
    Sha256Digest digest{};
    if (argc == 2) {
      table_path = argv[1];
      digest = source.content_hash;
    } else {
      temporary = temporaryPath();
      digest = writeRateTable(temporary, source);
      table_path = temporary;
    }

    ProposalTableSet descriptor{};
    descriptor.process_count =
        static_cast<std::uint32_t>(processCount(source));
    descriptor.content_hash = digest;
    GpuEmConfig config{};
    config.min_batch_size = 16;
    config.memory_fraction = 0.10;
    config.table_tolerance = 1.e-3;
    config.random_seed = 0x5041495253544154ULL;
    config.shower_id = 29;
    config.table_cache = table_path;

    EnvironmentSnapshot environment{};
    CudaEmBackend backend;
    backend.initialize(environment, descriptor, config);
    auto const fixture_mode = argc == 1;
    auto particles =
        makeParticles(source, fixture_mode ? 4096 : 8192);
    if (!fixture_mode) {
      for (auto& particle : particles) {
        particle.pid =
            static_cast<std::int32_t>(EmPid::Photon);
      }
    }
    auto selection =
        backend.selectInteractionsForValidation(particles);
    auto interactions = selection.interactions;
    auto const lpm_snapshot = makePhotonPairLpmSnapshot(
        source.metadata.photon_pair_lpm);
    for (std::size_t index = 0; index < interactions.size();
         ++index) {
      // Span sea level to a thin upper-atmosphere layer while retaining a
      // strictly positive density for every selected physical interaction.
      auto const exponent =
          static_cast<double>(index % 101) / 100.;
      interactions[index].mass_density_g_per_cm3 =
          lpm_snapshot.baseline_mass_density_g_per_cm3 *
          std::pow(1.e-5, exponent);
    }
    auto const& pair_column = photonPairColumn(source);
    auto append_record = [&](EmInteractionStatus status,
                             std::int32_t process_id,
                             double energy_GeV) {
      EmInteractionRecord record{};
      record.particle = particles.front();
      record.particle.pid =
          static_cast<std::int32_t>(EmPid::Photon);
      record.particle.energy_GeV = energy_GeV;
      record.particle.history_id =
          particles.size() + interactions.size() + 101;
      record.input_index =
          particles.size() + interactions.size();
      record.process_id = process_id;
      record.status = status;
      record.component_hash = pair_column.component_hash;
      record.mass_density_g_per_cm3 =
          lpm_snapshot.baseline_mass_density_g_per_cm3;
      record.energy_fraction = 0.4;
      record.loss_quantile = 0.25;
      record.loss_draw_id = 0;
      interactions.push_back(record);
      return record.particle.history_id;
    };
    append_record(EmInteractionStatus::NoDiscreteInteraction,
                  0, 10.);
    append_record(EmInteractionStatus::Selected,
                  PhotonMuonPairProcessId, 10.);
    auto const direct_compton_history =
        append_record(EmInteractionStatus::Selected,
                      ComptonProcessId, 10.);
    auto const direct_photoelectric_history =
        append_record(EmInteractionStatus::Selected,
                      PhotoelectricProcessId, 0.01);
    interactions.back().energy_fraction = 1.;
    append_record(EmInteractionStatus::Selected,
                  PhotonPairProcessId, 0.001);
    append_record(
        EmInteractionStatus::Selected, PhotonPairProcessId,
        fixture_mode ? 0.01 : 10.);
    interactions.back().mass_density_g_per_cm3 = 0.;
    if (!fixture_mode) {
      require(pair_column.inverse_cdf.energies_MeV.size() >= 2,
              "full photon-pair table has no inverse-CDF energy range");
      auto const log_pair_min = std::log(
          pair_column.inverse_cdf.energies_MeV.front());
      auto const log_pair_max = std::log(
          pair_column.inverse_cdf.energies_MeV.back());
      for (std::size_t i = 0; i < 2048; ++i) {
        auto const fraction =
            (static_cast<double>(i) + 0.5) / 2048.;
        append_record(
            EmInteractionStatus::Selected, PhotonPairProcessId,
            std::exp(log_pair_min +
                     fraction * (log_pair_max - log_pair_min)) /
                1000.);
      }
    }

    auto const flat = flattenRateTable(source);
    auto const view = makeFlatRateTableView(flat);
    std::uint64_t constexpr FirstHistory = 9000000;
    auto const expected = generateOnHost(
        view, lpm_snapshot, interactions, config.random_seed,
        config.shower_id, FirstHistory);
    auto const initial_fallbacks =
        backend.statistics().proposal_fallbacks;
    auto const first = backend.generateFinalStatesForValidation(
        interactions, FirstHistory);
    auto const second = backend.generateFinalStatesForValidation(
        interactions, FirstHistory);
    compareBatches(first, expected.result);
    compareBatches(second, first);
    checkPhysics(first, interactions, lpm_snapshot);
    require(first.photon_pair_interactions > 100,
            "too few photon-pair final states were tested: " +
                std::to_string(first.photon_pair_interactions) +
                " GPU, " +
                std::to_string(first.fallback_events.size()) +
                " fallback");
    require(first.compton_interactions > 100,
            "too few Compton final states were tested");
    require(
        std::any_of(
            first.final_state_records.begin(),
            first.final_state_records.end(),
            [&](PhotonPairFinalStateRecord const& record) {
              return record.process_id == ComptonProcessId &&
                     record.parent_history_id ==
                         direct_compton_history;
            }),
        "direct Compton fixture was not generated on the GPU");
    require(
        first.photoelectric_interactions >
            (fixture_mode ? 30U : 0U),
        "too few photoelectric final states were tested");
    require(
        std::any_of(
            first.final_state_records.begin(),
            first.final_state_records.end(),
            [&](PhotonPairFinalStateRecord const& record) {
              return record.process_id ==
                         PhotoelectricProcessId &&
                     record.parent_history_id ==
                         direct_photoelectric_history;
            }),
        "direct photoelectric fixture was not generated on the GPU");
    if (fixture_mode) {
      require(!first.lpm_suppressed.empty(),
              "synthetic batch did not exercise LPM suppression");
    }
    require(first.gpu_interactions +
                first.fallback_events.size() +
                first.lpm_suppressed.size() +
                first.continuations.size() ==
                interactions.size(),
            "final-state dispatch lost or duplicated an interaction");

    auto reversed = interactions;
    std::reverse(reversed.begin(), reversed.end());
    auto const reversed_result =
        backend.generateFinalStatesForValidation(
            reversed, FirstHistory);
    std::map<std::uint64_t, PhotonPairFinalStateRecord>
        first_by_parent;
    for (auto const& record : first.final_state_records) {
      first_by_parent.emplace(record.parent_history_id, record);
    }
    for (auto const& record :
         reversed_result.final_state_records) {
      auto const found =
          first_by_parent.find(record.parent_history_id);
      require(found != first_by_parent.end(),
              "queue reversal changed photon final-state membership");
      require(record.process_id == found->second.process_id &&
                  record.energy_split_fraction ==
                  found->second.energy_split_fraction &&
                  record.split_uniform ==
                      found->second.split_uniform &&
                  record.azimuth_uniform ==
                      found->second.azimuth_uniform &&
                  record.electron_polar_uniform ==
                      found->second.electron_polar_uniform &&
                  record.positron_polar_uniform ==
                      found->second.positron_polar_uniform &&
                  record.lpm_uniform ==
                      found->second.lpm_uniform &&
                  record.lpm_survival_probability ==
                      found->second.lpm_survival_probability,
              "queue reversal changed photon final-state physics");
    }
    require(reversed_result.gpu_interactions ==
                first.gpu_interactions,
            "queue reversal changed photon final-state count");
    require(
        reversed_result.photon_pair_interactions ==
                first.photon_pair_interactions &&
            reversed_result.compton_interactions ==
                first.compton_interactions &&
            reversed_result.photoelectric_interactions ==
                first.photoelectric_interactions,
        "queue reversal changed per-process final-state counts");
    std::map<std::uint64_t, PhotonPairLpmSuppressionRecord>
        suppressed_by_parent;
    for (auto const& record : first.lpm_suppressed) {
      suppressed_by_parent.emplace(
          record.particle.history_id, record);
    }
    require(reversed_result.lpm_suppressed.size() ==
                first.lpm_suppressed.size(),
            "queue reversal changed LPM-suppression count");
    for (auto const& record :
         reversed_result.lpm_suppressed) {
      auto const found = suppressed_by_parent.find(
          record.particle.history_id);
      require(found != suppressed_by_parent.end(),
              "queue reversal changed LPM-suppressed membership");
      require(record.uniform == found->second.uniform &&
                  record.survival_probability ==
                      found->second.survival_probability &&
                  record.particle.step_id ==
                      found->second.particle.step_id,
              "queue reversal changed LPM-suppressed physics");
    }

    auto thinned_config = config;
    thinned_config.thinning = {
        1.e20, 1.e20, 1, 1};
    CudaEmBackend thinned_backend;
    thinned_backend.initialize(
        environment, descriptor, thinned_config);
    auto const thinned =
        thinned_backend.generateFinalStatesForValidation(
            interactions, FirstHistory);
    require(
        thinned.gpu_interactions == first.gpu_interactions &&
            thinned.photon_pair_interactions ==
                first.photon_pair_interactions &&
            thinned.compton_interactions ==
                first.compton_interactions &&
            thinned.photoelectric_interactions ==
                first.photoelectric_interactions &&
            thinned.secondaries.size() ==
                thinned.gpu_interactions,
        "Hillas thinning did not compact photon 1->2 vertices "
        "while preserving photoelectric 1->1 vertices");
    for (std::size_t output_index = 0;
         output_index < thinned.final_state_records.size();
         ++output_index) {
      auto const& record =
          thinned.final_state_records[output_index];
      require(
          record.secondary_offset == output_index &&
              record.secondary_count == 1,
          "thinned photon final-state offset/count differs");
      if (record.process_id == PhotoelectricProcessId) {
        require(
            record.thinning_status ==
                    static_cast<std::uint32_t>(
                        EmThinningStatus::NotApplied) &&
                record.thinning_keep_mask == 0x3U,
            "photoelectric 1->1 final state was thinned");
        continue;
      }
      require(
          record.thinning_status ==
                  static_cast<std::uint32_t>(
                      EmThinningStatus::Hillas) &&
              (record.thinning_keep_mask == 0x1U ||
               record.thinning_keep_mask == 0x2U) &&
              record.thinning_first_draw_id ==
                  EmThinningFirstDrawId &&
              record.thinning_second_draw_id ==
                  EmThinningSecondDrawId,
          "photon Hillas metadata differs");
      auto const parent = std::find_if(
          interactions.begin(), interactions.end(),
          [&](EmInteractionRecord const& interaction) {
            return interaction.particle.history_id ==
                   record.parent_history_id;
          });
      require(
          parent != interactions.end(),
          "thinned photon record has no input parent");
      auto const& child = thinned.secondaries[output_index];
      double first_energy = 0.;
      double second_energy = 0.;
      std::int32_t first_pid = 0;
      std::int32_t second_pid = 0;
      if (record.process_id == PhotonPairProcessId) {
        first_energy =
            parent->particle.energy_GeV *
            record.energy_split_fraction;
        second_energy =
            parent->particle.energy_GeV - first_energy;
        first_pid = static_cast<std::int32_t>(EmPid::Electron);
        second_pid =
            static_cast<std::int32_t>(EmPid::Positron);
      } else {
        first_energy =
            parent->particle.energy_GeV *
            (1. - record.energy_split_fraction);
        second_energy =
            ElectronMassGeV +
            parent->particle.energy_GeV *
                record.energy_split_fraction;
        first_pid = static_cast<std::int32_t>(EmPid::Photon);
        second_pid =
            static_cast<std::int32_t>(EmPid::Electron);
      }
      auto const keep_first =
          record.thinning_keep_mask == 0x1U;
      auto const probability =
          (keep_first ? first_energy : second_energy) /
          (first_energy + second_energy);
      require(
          child.pid == (keep_first ? first_pid : second_pid) &&
              child.history_id == FirstHistory + output_index,
          "thinned photon child identity differs");
      require(
          std::abs(
              child.weight -
              parent->particle.weight / probability) <
              3.e-13,
          "thinned photon child weight differs");
    }
    require(
        thinned_backend.statistics().thinning_hillas_vertices ==
                thinned.photon_pair_interactions +
                    thinned.compton_interactions &&
            thinned_backend.statistics()
                    .thinning_statistical_vertices ==
                0 &&
            thinned_backend.statistics()
                    .thinning_particles_discarded ==
                thinned.photon_pair_interactions +
                    thinned.compton_interactions,
        "integrated photon Hillas statistics differ");

    require(backend.statistics().final_state_batches == 3,
            "backend final-state batch statistic differs");
    require(backend.statistics().gpu_final_states ==
                3 * first.gpu_interactions,
            "backend GPU final-state statistic differs");
    require(backend.statistics()
                .physical_secondaries_generated ==
                3 * first.secondaries.size(),
            "backend physical-secondary statistic differs");
    require(backend.statistics().photon_pair_lpm_trials ==
                3 * (first.photon_pair_interactions +
                     first.lpm_suppressed.size()),
            "backend LPM-trial statistic differs");
    require(
        backend.statistics().photon_pair_lpm_suppressions ==
            3 * first.lpm_suppressed.size(),
        "backend LPM-suppression statistic differs");
    require(backend.statistics().compton_final_states ==
                3 * first.compton_interactions,
            "backend Compton final-state statistic differs");
    require(backend.statistics().photoelectric_final_states ==
                3 * first.photoelectric_interactions,
            "backend photoelectric final-state statistic differs");
    require(backend.statistics().proposal_fallbacks ==
                initial_fallbacks +
                    3 * first.fallback_events.size(),
            "backend final-state fallback statistic differs");

    auto valid_pair = std::find_if(
        interactions.begin(), interactions.end(),
        [&](EmInteractionRecord const& record) {
          return record.particle.history_id ==
                 first.final_state_records.front()
                     .parent_history_id;
        });
    require(valid_pair != interactions.end(),
            "test input has no valid GPU photon pair");
    auto const exact_history_limit =
        backend.generateFinalStatesForValidation(
            std::vector<EmInteractionRecord>{*valid_pair},
            std::numeric_limits<std::uint64_t>::max() - 1);
    require(exact_history_limit.secondaries.size() == 2 &&
                exact_history_limit.secondaries[0].history_id ==
                    std::numeric_limits<std::uint64_t>::max() - 1 &&
                exact_history_limit.secondaries[1].history_id ==
                    std::numeric_limits<std::uint64_t>::max(),
            "backend rejected the exact secondary history-ID limit");
    requireThrows(
        [&] {
          backend.generateFinalStatesForValidation(
              interactions,
              std::numeric_limits<std::uint64_t>::max());
        },
        "backend accepted overflowing secondary history IDs");
    auto invalid_direction = interactions;
    auto pair = std::find_if(
        invalid_direction.begin(), invalid_direction.end(),
        [](EmInteractionRecord const& record) {
          return gpuProcessCapability(
                     record.particle.pid, record.process_id) ==
                 GpuProcessCapability::PhotonPair;
        });
    require(pair != invalid_direction.end(),
            "test input has no GPU photon pair");
    pair->particle.direction[2] *= 0.5;
    requireThrows(
        [&] {
          backend.generateFinalStatesForValidation(
              invalid_direction, FirstHistory);
        },
        "backend accepted a non-normalized photon direction");

    std::cout << "CUDA photon final states passed " << checks
              << " checks for " << interactions.size()
              << " interactions: "
              << first.photon_pair_interactions
              << " GPU pairs, " << first.compton_interactions
              << " GPU Compton, "
              << first.photoelectric_interactions
              << " GPU photoelectric, "
              << first.fallback_events.size()
              << " fallback, " << first.lpm_suppressed.size()
              << " LPM suppressed, "
              << first.continuations.size()
              << " continuation\n";
  } catch (std::exception const& error) {
    if (!temporary.empty()) {
      std::error_code ignored;
      std::filesystem::remove(temporary, ignored);
    }
    std::cerr << "CUDA photon final states failed after "
              << checks << " checks: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
  if (!temporary.empty()) {
    std::error_code ignored;
    std::filesystem::remove(temporary, ignored);
  }
  return EXIT_SUCCESS;
}

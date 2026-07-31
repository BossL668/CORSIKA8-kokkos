/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include <cuda_runtime_api.h>

#include <PROPOSAL/medium/Medium.h>
#include <PROPOSAL/particle/Particle.h>
#include <PROPOSAL/secondaries/parametrization/bremsstrahlung/BremsEGS4Approximation.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <GpuEmFlatRateTableFixture.hpp>

#include <corsika/gpu/em/BremsLpm.hpp>
#include <corsika/gpu/em/CudaBremsFinalState.hpp>
#include <corsika/gpu/em/CudaEmBackend.hpp>
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

  std::filesystem::path temporaryPath() {
    auto const stamp =
        std::chrono::high_resolution_clock::now()
            .time_since_epoch()
            .count();
    return std::filesystem::temp_directory_path() /
           ("c8_gpu_brems_final_state_" +
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
      std::size_t index, BremsLpmSnapshot const& snapshot) {
    EmInteractionRecord interaction{};
    auto& particle = interaction.particle;
    particle.pid =
        index % 2 == 0
            ? static_cast<std::int32_t>(EmPid::Electron)
            : static_cast<std::int32_t>(EmPid::Positron);
    particle.medium_id = 17;
    particle.generation =
        static_cast<std::uint32_t>(index % 17);
    auto const energy_fraction =
        (static_cast<double>(index % 997) + 0.5) / 997.;
    particle.energy_GeV =
        (2. + 98. * energy_fraction) / 1000.;
    particle.position_m[0] = 1.;
    particle.position_m[1] = -2.;
    particle.position_m[2] = 3.;
    if (index % 3 == 0) {
      particle.direction[2] = 1.;
    } else {
      particle.direction[0] = 0.6;
      particle.direction[2] = 0.8;
    }
    particle.time_s = 4.e-6;
    particle.weight = 1. + 0.01 * (index % 5);
    particle.history_id = index + 1;
    particle.parent_history_id = index / 3;
    particle.step_id = index % 11;

    interaction.input_index = index;
    interaction.process_id = BremsProcessId;
    interaction.status = EmInteractionStatus::Selected;
    interaction.component_hash =
        snapshot.components[0].component_hash;
    auto const density_exponent =
        static_cast<double>(index % 101) / 100.;
    interaction.mass_density_g_per_cm3 =
        snapshot.baseline_mass_density_g_per_cm3 *
        std::pow(1.e-7, density_exponent);
    interaction.energy_fraction =
        0.01 + 0.48 *
                   (static_cast<double>(index % 991) + 0.5) /
                   991.;
    interaction.loss_quantile =
        (static_cast<double>(index % 983) + 0.5) / 983.;
    interaction.loss_draw_id = 0;
    return interaction;
  }

  struct ExpectedClassification {
    std::vector<std::size_t> accepted;
    std::vector<std::size_t> suppressed;
    std::vector<std::size_t> continuations;
    std::vector<std::size_t> fallbacks;
    std::vector<double> azimuth_uniforms;
    std::vector<double> lpm_uniforms;
    std::vector<double> lpm_probabilities;
  };

  ExpectedClassification classifyOnHost(
      std::vector<EmInteractionRecord> const& interactions,
      BremsLpmSnapshot const& snapshot, std::uint64_t seed,
      std::uint64_t shower_id) {
    ExpectedClassification result;
    result.azimuth_uniforms.resize(interactions.size());
    result.lpm_uniforms.resize(interactions.size());
    result.lpm_probabilities.resize(interactions.size());
    for (std::size_t index = 0; index < interactions.size();
         ++index) {
      auto const& interaction = interactions[index];
      if (interaction.status ==
          EmInteractionStatus::NoDiscreteInteraction) {
        result.continuations.push_back(index);
        continue;
      }
      auto const capability = gpuProcessCapability(
          interaction.particle.pid, interaction.process_id);
      if (capability !=
          GpuProcessCapability::Bremsstrahlung) {
        result.fallbacks.push_back(index);
        continue;
      }
      auto const& parent = interaction.particle;
      auto const mass_GeV =
          snapshot.lepton_mass_MeV / 1000.;
      auto const photon_energy =
          parent.energy_GeV * interaction.energy_fraction;
      auto const lepton_energy =
          parent.energy_GeV - photon_energy;
      if (!(interaction.energy_fraction > 0.) ||
          !(interaction.energy_fraction < 1.) ||
          !(parent.energy_GeV > mass_GeV) ||
          lepton_energy < mass_GeV) {
        result.fallbacks.push_back(index);
        continue;
      }
      RandomNumberKey const azimuth_key{
          seed, shower_id, parent.history_id, parent.step_id,
          static_cast<std::uint32_t>(BremsProcessId),
          BremsAzimuthDrawId};
      auto const azimuth = uniformOpen01(azimuth_key);
      result.azimuth_uniforms[index] = azimuth;
      auto const lpm = bremsLpmSuppressionFactor(
          snapshot, interaction.component_hash,
          parent.energy_GeV * 1000.,
          interaction.energy_fraction,
          interaction.mass_density_g_per_cm3);
      if (lpm.status != BremsLpmStatus::Success) {
        result.fallbacks.push_back(index);
        continue;
      }
      RandomNumberKey const lpm_key{
          seed, shower_id, parent.history_id, parent.step_id,
          static_cast<std::uint32_t>(BremsProcessId),
          BremsLpmDrawId};
      auto const lpm_uniform = uniformOpen01(lpm_key);
      result.lpm_uniforms[index] = lpm_uniform;
      result.lpm_probabilities[index] =
          lpm.survival_probability;
      if (lpm_uniform > lpm.survival_probability) {
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
    config.random_seed = 0x4252454d53544154ULL;
    config.shower_id = 47;
    config.table_cache = temporary;
    EnvironmentSnapshot environment{};

    auto const snapshot =
        makeBremsLpmSnapshot(source.metadata.brems_lpm);
    std::vector<EmInteractionRecord> interactions;
    interactions.reserve(4100);
    for (std::size_t index = 0; index < 4096; ++index) {
      interactions.push_back(
          makeInteraction(index, snapshot));
    }

    auto continuation =
        makeInteraction(interactions.size(), snapshot);
    continuation.status =
        EmInteractionStatus::NoDiscreteInteraction;
    continuation.process_id = 0;
    interactions.push_back(continuation);

    auto unsupported =
        makeInteraction(interactions.size(), snapshot);
    unsupported.process_id = ComptonProcessId;
    interactions.push_back(unsupported);

    auto invalid_density =
        makeInteraction(interactions.size(), snapshot);
    invalid_density.mass_density_g_per_cm3 = 0.;
    interactions.push_back(invalid_density);

    auto invalid_loss =
        makeInteraction(interactions.size(), snapshot);
    invalid_loss.energy_fraction = 1.;
    interactions.push_back(invalid_loss);

    auto const expected = classifyOnHost(
        interactions, snapshot, config.random_seed,
        config.shower_id);
    require(expected.accepted.size() > 500,
            "fixture has too few accepted bremsstrahlung events");
    require(expected.suppressed.size() > 500,
            "fixture has too few LPM-suppressed bremsstrahlung events");
    require(expected.continuations.size() == 1,
            "fixture continuation count differs");
    require(expected.fallbacks.size() == 3,
            "fixture fallback count differs");

    CudaEmBackend backend;
    backend.initialize(environment, descriptor, config);
    std::uint64_t constexpr FirstHistory = 9000000;
    auto const result =
        backend.generateBremsFinalStatesForValidation(
            interactions, FirstHistory);

    require(result.input_interactions == interactions.size(),
            "bremsstrahlung input count differs");
    require(result.gpu_interactions == expected.accepted.size(),
            "accepted bremsstrahlung count differs");
    require(result.final_state_records.size() ==
                expected.accepted.size(),
            "bremsstrahlung record count differs");
    require(result.secondaries.size() ==
                2 * expected.accepted.size(),
            "bremsstrahlung secondary count differs");
    require(result.lpm_suppressed.size() ==
                expected.suppressed.size(),
            "bremsstrahlung suppression count differs");
    require(result.continuations.size() ==
                expected.continuations.size(),
            "bremsstrahlung continuation count differs");
    require(result.fallback_events.size() ==
                expected.fallbacks.size(),
            "bremsstrahlung fallback count differs");

    PROPOSAL::Air const medium;
    auto const component = medium.GetComponents().front();
    PROPOSAL::secondaries::BremsEGS4Approximation electron_generator(
        PROPOSAL::EMinusDef(), medium);
    PROPOSAL::secondaries::BremsEGS4Approximation positron_generator(
        PROPOSAL::EPlusDef(), medium);
    for (std::size_t output_index = 0;
         output_index < expected.accepted.size();
         ++output_index) {
      auto const input_index = expected.accepted[output_index];
      auto const& interaction = interactions[input_index];
      auto const& parent = interaction.particle;
      auto const& record =
          result.final_state_records[output_index];
      require(record.input_index == interaction.input_index &&
                  record.parent_history_id ==
                      parent.history_id &&
                  record.secondary_offset ==
                      2 * output_index &&
                  record.secondary_count == 2,
              "bremsstrahlung record identity differs");
      require(record.azimuth_draw_id ==
                  BremsAzimuthDrawId &&
                  record.lpm_draw_id == BremsLpmDrawId,
              "bremsstrahlung draw IDs differ");
      requireNear(
          record.azimuth_uniform,
          expected.azimuth_uniforms[input_index], 0.,
          "bremsstrahlung azimuth uniform differs");
      requireNear(
          record.lpm_uniform,
          expected.lpm_uniforms[input_index], 0.,
          "bremsstrahlung LPM uniform differs");
      requireNear(
          record.lpm_survival_probability,
          expected.lpm_probabilities[input_index], 3.e-12,
          "bremsstrahlung LPM probability differs");

      auto const& lepton =
          result.secondaries[2 * output_index];
      auto const& photon =
          result.secondaries[2 * output_index + 1];
      require(lepton.pid == parent.pid &&
                  photon.pid ==
                      static_cast<std::int32_t>(EmPid::Photon),
              "bremsstrahlung secondary PID differs");
      require(
          lepton.history_id ==
                  FirstHistory + 2 * output_index &&
              photon.history_id ==
                  FirstHistory + 2 * output_index + 1 &&
              lepton.parent_history_id == parent.history_id &&
              photon.parent_history_id == parent.history_id,
          "bremsstrahlung secondary identity differs");
      requireNear(
          lepton.energy_GeV + photon.energy_GeV,
          parent.energy_GeV, 2.e-15,
          "bremsstrahlung energy does not close");
      requireNear(
          photon.energy_GeV,
          parent.energy_GeV *
              interaction.energy_fraction,
          2.e-15, "bremsstrahlung photon energy differs");

      PROPOSAL::Cartesian3D const position(
          parent.position_m[0] * 100.,
          parent.position_m[1] * 100.,
          parent.position_m[2] * 100.);
      PROPOSAL::Cartesian3D const direction(
          parent.direction[0], parent.direction[1],
          parent.direction[2]);
      PROPOSAL::StochasticLoss loss(
          BremsProcessId, photon.energy_GeV * 1000.,
          position, direction, parent.time_s, 0.,
          parent.energy_GeV * 1000.,
          component.GetHash());
      std::vector<double> random_numbers{
          expected.azimuth_uniforms[input_index]};
      auto reference =
          parent.pid ==
                  static_cast<std::int32_t>(EmPid::Electron)
              ? electron_generator.CalculateSecondaries(
                    loss, component, random_numbers)
              : positron_generator.CalculateSecondaries(
                    loss, component, random_numbers);
      require(reference.size() == 2,
              "PROPOSAL bremsstrahlung multiplicity differs");
      requireNear(
          lepton.energy_GeV * 1000.,
          reference[0].energy, 2.e-15,
          "lepton energy differs from PROPOSAL");
      requireNear(
          photon.energy_GeV * 1000.,
          reference[1].energy, 2.e-15,
          "photon energy differs from PROPOSAL");
      auto const reference_lepton =
          reference[0].direction.GetCartesianCoordinates();
      auto const reference_photon =
          reference[1].direction.GetCartesianCoordinates();
      for (std::size_t axis = 0; axis < 3; ++axis) {
        requireNear(
            lepton.direction[axis],
            reference_lepton[axis], 4.e-12,
            "lepton direction differs from PROPOSAL");
        requireNear(
            photon.direction[axis],
            reference_photon[axis], 4.e-12,
            "photon direction differs from PROPOSAL");
      }
    }

    for (std::size_t index = 0;
         index < expected.suppressed.size(); ++index) {
      auto const input_index = expected.suppressed[index];
      auto const& actual = result.lpm_suppressed[index];
      auto const& source_interaction =
          interactions[input_index];
      require(
          actual.input_index ==
                  source_interaction.input_index &&
              actual.particle.history_id ==
                  source_interaction.particle.history_id &&
              actual.particle.step_id ==
                  source_interaction.particle.step_id + 1 &&
              actual.draw_id == BremsLpmDrawId,
          "bremsstrahlung LPM continuation identity differs");
      requireNear(
          actual.survival_probability,
          expected.lpm_probabilities[input_index], 3.e-12,
          "suppressed bremsstrahlung probability differs");
      requireNear(
          actual.uniform,
          expected.lpm_uniforms[input_index], 0.,
          "suppressed bremsstrahlung uniform differs");
    }

    require(
        result.continuations.front().input_index ==
            interactions[expected.continuations.front()]
                .input_index,
        "no-interaction bremsstrahlung continuation differs");
    require(
        result.fallback_events[0].reason ==
                ProposalFallbackReason::
                    GpuProcessNotImplemented &&
            result.fallback_events[1].reason ==
                ProposalFallbackReason::InvalidMassDensity &&
            result.fallback_events[2].reason ==
                ProposalFallbackReason::InvalidFinalState,
        "bremsstrahlung fallback reasons differ");

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
        "bremsstrahlung repeat counts differ");
    for (std::size_t index = 0;
         index < result.secondaries.size(); ++index) {
      require(
          repeat.secondaries[index].history_id ==
                  result.secondaries[index].history_id &&
              repeat.secondaries[index].energy_GeV ==
                  result.secondaries[index].energy_GeV &&
              repeat.secondaries[index].direction[0] ==
                  result.secondaries[index].direction[0] &&
              repeat.secondaries[index].direction[1] ==
                  result.secondaries[index].direction[1] &&
              repeat.secondaries[index].direction[2] ==
                  result.secondaries[index].direction[2],
          "bremsstrahlung kernel is not deterministic");
    }

    auto thinned_config = config;
    thinned_config.thinning = {
        1.e20, 1.e20, 1, 1};
    CudaEmBackend thinned_backend;
    thinned_backend.initialize(
        environment, descriptor, thinned_config);
    auto const thinned =
        thinned_backend.generateBremsFinalStatesForValidation(
            interactions, FirstHistory);
    require(
        thinned.final_state_records.size() ==
                expected.accepted.size() &&
            thinned.secondaries.size() ==
                expected.accepted.size(),
        "Hillas thinning did not compact every accepted 1->2 "
        "bremsstrahlung vertex to one child");
    for (std::size_t output_index = 0;
         output_index < expected.accepted.size();
         ++output_index) {
      auto const input_index = expected.accepted[output_index];
      auto const& interaction = interactions[input_index];
      auto const& parent = interaction.particle;
      auto const& record =
          thinned.final_state_records[output_index];
      RandomNumberKey const thinning_key{
          config.random_seed, config.shower_id,
          parent.history_id, parent.step_id,
          static_cast<std::uint32_t>(BremsProcessId),
          EmThinningFirstDrawId};
      auto const thinning_uniform =
          uniformOpen01(thinning_key);
      auto const keep_lepton =
          thinning_uniform <=
          1. - interaction.energy_fraction;
      auto const& child = thinned.secondaries[output_index];
      require(
          record.secondary_offset == output_index &&
              record.secondary_count == 1 &&
              record.thinning_status ==
                  static_cast<std::uint32_t>(
                      EmThinningStatus::Hillas) &&
              record.thinning_keep_mask ==
                  (keep_lepton ? 0x1U : 0x2U) &&
              record.thinning_first_draw_id ==
                  EmThinningFirstDrawId &&
              record.thinning_second_draw_id ==
                  EmThinningSecondDrawId &&
              record.thinning_first_uniform ==
                  thinning_uniform &&
              child.history_id ==
                  FirstHistory + output_index &&
              child.pid ==
                  (keep_lepton
                       ? parent.pid
                       : static_cast<std::int32_t>(
                             EmPid::Photon)),
          "integrated Hillas bremsstrahlung decision differs");
      auto const expected_weight =
          parent.weight /
          (keep_lepton
               ? 1. - interaction.energy_fraction
               : interaction.energy_fraction);
      requireNear(
          child.weight, expected_weight, 2.e-13,
          "integrated Hillas bremsstrahlung weight differs");
    }
    require(
        thinned_backend.statistics().thinning_hillas_vertices ==
                expected.accepted.size() &&
            thinned_backend.statistics()
                    .thinning_statistical_vertices ==
                0 &&
            thinned_backend.statistics()
                    .thinning_particles_discarded ==
                expected.accepted.size(),
        "integrated Hillas bremsstrahlung statistics differ");

    {
      auto statistical_config = config;
      statistical_config.thinning = {
          1.e20, 1.5, 1, 1};
      CudaEmBackend statistical_backend;
      statistical_backend.initialize(
          environment, descriptor, statistical_config);
      auto const statistical =
          statistical_backend
              .generateBremsFinalStatesForValidation(
                  interactions, FirstHistory);
      std::size_t expected_offset = 0;
      std::size_t expected_discarded = 0;
      std::size_t zero_child_vertices = 0;
      for (std::size_t output_index = 0;
           output_index < expected.accepted.size();
           ++output_index) {
        auto const input_index =
            expected.accepted[output_index];
        auto const& interaction = interactions[input_index];
        auto const& parent = interaction.particle;
        RandomNumberKey first_key{
            config.random_seed, config.shower_id,
            parent.history_id, parent.step_id,
            static_cast<std::uint32_t>(BremsProcessId),
            EmThinningFirstDrawId};
        auto second_key = first_key;
        second_key.draw_id = EmThinningSecondDrawId;
        auto const decision = applyEmThinning(
            statistical_config.thinning,
            parent.energy_GeV, parent.weight,
            parent.energy_GeV *
                (1. - interaction.energy_fraction),
            parent.energy_GeV *
                interaction.energy_fraction,
            uniformOpen01(first_key),
            uniformOpen01(second_key));
        auto const expected_count =
            (decision.keep_mask & 0x1U ? 1U : 0U) +
            (decision.keep_mask & 0x2U ? 1U : 0U);
        auto const& record =
            statistical.final_state_records[output_index];
        require(
            decision.status ==
                    EmThinningStatus::Statistical &&
                record.thinning_status ==
                    static_cast<std::uint32_t>(
                        decision.status) &&
                record.thinning_keep_mask ==
                    decision.keep_mask &&
                record.secondary_offset == expected_offset &&
                record.secondary_count == expected_count,
            "integrated statistical thinning decision differs");
        if (expected_count == 0) {
          ++zero_child_vertices;
        }
        expected_discarded += 2 - expected_count;
        std::size_t local_child = 0;
        if ((decision.keep_mask & 0x1U) != 0) {
          auto const& child =
              statistical.secondaries[
                  expected_offset + local_child++];
          require(
              child.pid == parent.pid &&
                  child.history_id ==
                      FirstHistory + expected_offset,
              "statistical thinning surviving-lepton child differs");
          requireNear(
              child.weight, decision.first_weight, 3.e-13,
              "statistical thinning surviving-lepton weight differs");
        }
        if ((decision.keep_mask & 0x2U) != 0) {
          auto const& child =
              statistical.secondaries[
                  expected_offset + local_child++];
          require(
              child.pid ==
                      static_cast<std::int32_t>(
                          EmPid::Photon) &&
                  child.history_id ==
                      FirstHistory + expected_offset +
                          local_child - 1,
              "statistical thinning photon child differs");
          requireNear(
              child.weight, decision.second_weight, 3.e-13,
              "statistical thinning photon weight differs");
        }
        expected_offset += expected_count;
      }
      require(
          statistical.secondaries.size() == expected_offset &&
              zero_child_vertices > 0 &&
              statistical_backend.statistics()
                      .thinning_hillas_vertices ==
                  0 &&
              statistical_backend.statistics()
                      .thinning_statistical_vertices ==
                  expected.accepted.size() &&
              statistical_backend.statistics()
                      .thinning_particles_discarded ==
                  expected_discarded,
          "integrated statistical thinning compaction/statistics differ");
    }

    {
      auto multithin_config = config;
      multithin_config.thinning = {
          1.e20, 1.5, 1, 0};
      CudaEmBackend multithin_backend;
      multithin_backend.initialize(
          environment, descriptor, multithin_config);
      auto const multithin =
          multithin_backend
              .generateBremsFinalStatesForValidation(
                  interactions, FirstHistory);
      require(
          multithin.secondaries.size() ==
                  2 * expected.accepted.size() &&
              std::all_of(
                  multithin.final_state_records.begin(),
                  multithin.final_state_records.end(),
                  [](BremsFinalStateRecord const& record) {
                    return record.secondary_count == 2 &&
                           record.thinning_status ==
                               static_cast<std::uint32_t>(
                                   EmThinningStatus::
                                       Statistical) &&
                           record.thinning_keep_mask == 0x3U;
                  }) &&
              std::any_of(
                  multithin.secondaries.begin(),
                  multithin.secondaries.end(),
                  [](EmParticleState const& particle) {
                    return particle.weight == 0.;
                  }),
          "multithin did not retain zero-weight GPU children");
      require(
          multithin_backend.statistics()
                      .thinning_statistical_vertices ==
                  expected.accepted.size() &&
              multithin_backend.statistics()
                      .thinning_particles_discarded ==
                  0,
          "multithin statistics counted retained zero-weight children as "
          "discarded");

      auto zero_weight_interaction =
          interactions[expected.accepted.front()];
      zero_weight_interaction.particle.weight = 0.;
      auto const zero_weight_result =
          multithin_backend
              .generateBremsFinalStatesForValidation(
                  {zero_weight_interaction}, 12000000);
      require(
          zero_weight_result.secondaries.size() == 2 &&
              zero_weight_result.secondaries[0].weight == 0. &&
              zero_weight_result.secondaries[1].weight == 0.,
          "GPU final-state validation rejected a retained zero-weight "
          "multithin parent");
    }

    auto const& statistics = backend.statistics();
    require(
        statistics.brems_lpm_trials ==
                2 * (expected.accepted.size() +
                     expected.suppressed.size()) &&
            statistics.brems_lpm_suppressions ==
                2 * expected.suppressed.size(),
        "bremsstrahlung LPM statistics differ");

    std::error_code ignored;
    std::filesystem::remove(temporary, ignored);
    std::cout
        << "GPU bremsstrahlung final-state oracle passed "
        << checks << " checks: " << expected.accepted.size()
        << " accepted, " << expected.suppressed.size()
        << " LPM-suppressed, "
        << expected.fallbacks.size()
        << " explicit fallbacks\n";
    return 0;
  } catch (std::exception const& error) {
    std::error_code ignored;
    std::filesystem::remove(temporary, ignored);
    std::cerr
        << "GPU bremsstrahlung final-state oracle failed after "
        << checks << " checks: " << error.what() << '\n';
    return 1;
  }
}

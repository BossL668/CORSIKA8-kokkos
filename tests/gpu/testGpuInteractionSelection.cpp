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
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <GpuEmFlatRateTableFixture.hpp>

#include <corsika/gpu/em/CudaEmBackend.hpp>
#include <corsika/gpu/em/CudaInteractionSelector.hpp>
#include <corsika/gpu/em/CudaLeptonTransport.hpp>
#include <corsika/gpu/em/Philox.hpp>
#include <corsika/gpu/em/ProposalFallback.hpp>
#include <corsika/gpu/em/tables/FlatRateTable.hpp>

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

  void requireClose(double actual, double expected,
                    std::string const& message) {
    if (std::isinf(actual) || std::isinf(expected)) {
      require(actual == expected, message);
      return;
    }
    auto const scale =
        std::max({1., std::abs(actual), std::abs(expected)});
    require(std::abs(actual - expected) <= 3.e-13 * scale,
            message);
  }

  std::size_t processCount(RateTableSet const& source) {
    std::size_t count = 0;
    for (auto const& particle : source.particles) {
      count += particle.columns.size();
    }
    return count;
  }

  struct HostSelection {
    bool fallback{};
    EmInteractionRecord interaction{};
    ProposalFallbackEvent fallback_event{};
  };

  HostSelection selectOnHost(
      RateTableSet const& source, FlatRateTableView flat_view,
      EmParticleState const& particle, std::size_t input_index,
      std::uint64_t seed, std::uint64_t shower_id) {
    auto const energy_MeV = particle.energy_GeV * 1000.;
    HostSelection result;
    auto& record = result.interaction;
    record.particle = particle;
    record.input_index = input_index;
    record.distance_draw_id = InteractionDistanceDrawId;
    record.process_draw_id = InteractionColumnDrawId;
    record.loss_draw_id = InteractionLossDrawId;
    if (exceedsParticleCutTime(particle.time_s)) {
      record.status = EmInteractionStatus::ParticleCut;
      record.interaction_grammage_g_per_cm2 =
          std::numeric_limits<double>::infinity();
      return result;
    }
    auto const charged_lepton =
        isChargedLeptonPid(particle.pid);
    auto const photon =
        particle.pid ==
        static_cast<std::int32_t>(EmPid::Photon);
    if (photon &&
        energy_MeV < flat_view.em_transport_cut_MeV) {
      record.status = EmInteractionStatus::ParticleCut;
      record.interaction_grammage_g_per_cm2 =
          std::numeric_limits<double>::infinity();
      return result;
    }
    if (charged_lepton) {
      auto const mass =
          queryContinuousMass(flat_view, particle.pid);
      auto const minimum_energy =
          queryContinuousMinimumEnergy(
              flat_view, particle.pid);
      if (mass.status == TableLookupStatus::Success) {
        record.particle_mass_GeV =
            mass.value / 1000.;
        if (isMuonPid(particle.pid) &&
            particle.energy_GeV >
                record.particle_mass_GeV) {
          RandomNumberKey const decay_key{
              seed, shower_id, particle.history_id,
              particle.step_id, MuonDecayRandomProcessId,
              MuonDecayDrawId};
          record.decay_uniform =
              uniformOpen01(decay_key);
          record.decay_draw_id = MuonDecayDrawId;
          auto const momentum_GeV = std::sqrt(
              (particle.energy_GeV -
               record.particle_mass_GeV) *
              (particle.energy_GeV +
               record.particle_mass_GeV));
          record.decay_distance_m =
              -std::log(record.decay_uniform) *
              MuonDecaySpeedOfLightMPerS *
              MuonMeanLifetimeS * momentum_GeV /
              record.particle_mass_GeV;
        }
      }
      auto const below_cut =
          mass.status == TableLookupStatus::Success &&
          minimum_energy.status ==
              TableLookupStatus::Success &&
          energy_MeV - mass.value <
              (minimum_energy.value - mass.value) /
                  ContinuousCutSafetyFactor;
      if (below_cut ||
          (minimum_energy.status ==
               TableLookupStatus::Success &&
           energy_MeV <= minimum_energy.value)) {
        record.status = EmInteractionStatus::ParticleCut;
        record.interaction_grammage_g_per_cm2 =
            std::numeric_limits<double>::infinity();
        return result;
      }
    }
    TableQuery const total_query{
        TableQueryKind::TotalRate, particle.pid, 0, 0,
        energy_MeV, 0.};
    auto const total = executeTableQuery(flat_view, total_query);
    if (total.status != TableLookupStatus::Success) {
      result.fallback = true;
      result.fallback_event = makeTableFallbackEvent(
          particle, total_query, total, 0, input_index);
      return result;
    }

    record.total_rate_cm2_per_g = total.value;
    record.vertex_total_rate_cm2_per_g = total.value;
    if (!(total.value > 0.)) {
      record.status = EmInteractionStatus::NoDiscreteInteraction;
      record.interaction_grammage_g_per_cm2 =
          std::numeric_limits<double>::infinity();
      return result;
    }

    RandomNumberKey const distance_key{
        seed, shower_id, particle.history_id, particle.step_id,
        InteractionDistanceRandomProcessId,
        InteractionDistanceDrawId};
    record.distance_uniform = uniformOpen01(distance_key);
    record.interaction_grammage_g_per_cm2 =
        -std::log(record.distance_uniform) / total.value;
    RandomNumberKey const process_key{
        seed, shower_id, particle.history_id, particle.step_id,
        InteractionColumnRandomProcessId,
        InteractionColumnDrawId};
    record.process_uniform = uniformOpen01(process_key);

    if (charged_lepton) {
      record.status = EmInteractionStatus::DistanceSampled;
      return result;
    }

    auto const& source_particle =
        findParticle(source, particle.pid);
    auto const threshold = record.process_uniform * total.value;
    double cumulative = 0.;
    RateColumn const* selected = nullptr;
    RateColumn const* last_positive = nullptr;
    for (auto const& column : source_particle.columns) {
      auto const rate = interpolateRate(
          source_particle, column.process_id,
          column.component_hash, energy_MeV);
      if (rate > 0.) {
        last_positive = &column;
      }
      cumulative += rate;
      if (threshold < cumulative) {
        selected = &column;
        break;
      }
    }
    if (selected == nullptr) {
      selected = last_positive;
    }
    if (selected == nullptr) {
      result.fallback = true;
      result.fallback_event.particle = particle;
      result.fallback_event.input_index = input_index;
      result.fallback_event.reason =
          ProposalFallbackReason::ZeroTotalRate;
      return result;
    }

    record.process_id = selected->process_id;
    record.component_hash = selected->component_hash;
    RandomNumberKey const loss_key{
        seed, shower_id, particle.history_id, particle.step_id,
        static_cast<std::uint32_t>(record.process_id),
        InteractionLossDrawId};
    record.loss_quantile = uniformOpen01(loss_key);
    TableQuery const loss_query{
        TableQueryKind::LossFraction, particle.pid,
        record.process_id, record.component_hash, energy_MeV,
        record.loss_quantile};
    auto const loss = executeTableQuery(flat_view, loss_query);
    if (loss.status != TableLookupStatus::Success) {
      result.fallback = true;
      result.fallback_event = makeTableFallbackEvent(
          particle, loss_query, loss, InteractionLossDrawId,
          input_index);
      result.fallback_event.loss_quantile =
          record.loss_quantile;
      return result;
    }
    record.status = EmInteractionStatus::Selected;
    record.energy_fraction = loss.value;
    return result;
  }

  std::vector<EmParticleState> makeParticles(
      RateTableSet const& source, std::size_t count,
      bool add_fixture_fallbacks) {
    std::vector<EmParticleState> particles;
    particles.reserve(count + (add_fixture_fallbacks ? 3 : 0));
    for (std::size_t i = 0; i < count; ++i) {
      auto const energy_fraction =
          std::fmod(
              (static_cast<double>(i) + 0.5) *
                  0.6180339887498948482,
              1.);
      auto const& source_particle =
          source.particles[i % source.particles.size()];
      auto const log_min =
          std::log(source_particle.energies_MeV.front());
      auto const log_max =
          std::log(source_particle.energies_MeV.back());
      EmParticleState particle{};
      particle.pid = source_particle.pdg_id;
      particle.energy_GeV =
          std::exp(log_min +
                   energy_fraction * (log_max - log_min)) /
          1000.;
      particle.direction[2] = -1.;
      particle.weight = 1.;
      particle.history_id = i + 1;
      particle.step_id = i % 7;
      particles.push_back(particle);
    }
    if (add_fixture_fallbacks) {
      auto missing = particles.front();
      missing.pid = 13;
      missing.history_id = count + 1;
      particles.push_back(missing);
      auto outside = particles.front();
      outside.energy_GeV = 0.0005;
      outside.history_id = count + 2;
      particles.push_back(outside);
      auto below_transport_cut = particles.front();
      below_transport_cut.pid = 11;
      below_transport_cut.energy_GeV =
          (0.5109989461 + 0.45) / 1000.;
      below_transport_cut.history_id = count + 3;
      particles.push_back(below_transport_cut);
    }
    return particles;
  }

  void compareInteraction(EmInteractionRecord const& actual,
                          EmInteractionRecord const& expected) {
    require(actual.input_index == expected.input_index,
            "interaction compaction changed input order");
    require(actual.particle.history_id ==
                expected.particle.history_id,
            "interaction record lost particle identity");
    require(actual.process_id == expected.process_id &&
                actual.component_hash == expected.component_hash,
            "selected process/component differs");
    require(actual.status == expected.status,
            "interaction status differs for input " +
                std::to_string(actual.input_index) +
                ", pid=" +
                std::to_string(actual.particle.pid) +
                ": actual=" +
                std::to_string(
                    static_cast<std::int32_t>(actual.status)) +
                ", expected=" +
                std::to_string(
                    static_cast<std::int32_t>(expected.status)));
    requireClose(actual.total_rate_cm2_per_g,
                 expected.total_rate_cm2_per_g,
                 "selected total rate differs");
    requireClose(actual.vertex_total_rate_cm2_per_g,
                 expected.vertex_total_rate_cm2_per_g,
                 "selected vertex total rate differs");
    requireClose(actual.interaction_grammage_g_per_cm2,
                 expected.interaction_grammage_g_per_cm2,
                 "sampled interaction grammage differs");
    requireClose(actual.energy_fraction,
                 expected.energy_fraction,
                 "sampled energy fraction differs");
    require(actual.distance_uniform == expected.distance_uniform &&
                actual.process_uniform == expected.process_uniform &&
                actual.loss_quantile == expected.loss_quantile,
            "Philox interaction uniforms differ");
  }

  void compareFallback(ProposalFallbackEvent const& actual,
                       ProposalFallbackEvent const& expected) {
    require(actual.input_index == expected.input_index,
            "fallback compaction changed input order");
    require(actual.particle.history_id ==
                expected.particle.history_id,
            "fallback lost particle identity");
    require(actual.reason == expected.reason,
            "fallback reason differs");
    require(actual.process_id == expected.process_id &&
                actual.component_hash == expected.component_hash,
            "fallback lost selected process/component");
    require(actual.loss_quantile == expected.loss_quantile &&
                actual.random_draw_id == expected.random_draw_id,
            "fallback lost random-number identity");
  }

  void compareBatches(EmInteractionBatchResult const& actual,
                      EmInteractionBatchResult const& expected) {
    require(actual.input_particles == expected.input_particles,
            "interaction batch input count differs");
    require(actual.interactions.size() ==
                expected.interactions.size(),
            "interaction compact count differs");
    require(actual.fallback_events.size() ==
                expected.fallback_events.size(),
            "fallback compact count differs");
    for (std::size_t i = 0; i < actual.interactions.size(); ++i) {
      compareInteraction(actual.interactions[i],
                         expected.interactions[i]);
    }
    for (std::size_t i = 0;
         i < actual.fallback_events.size(); ++i) {
      compareFallback(actual.fallback_events[i],
                      expected.fallback_events[i]);
    }
  }

  void compareSchedulingInvariant(
      EmInteractionBatchResult first,
      EmInteractionBatchResult reordered) {
    auto interaction_order =
        [](EmInteractionRecord const& left,
           EmInteractionRecord const& right) {
          return left.particle.history_id <
                 right.particle.history_id;
        };
    auto fallback_order =
        [](ProposalFallbackEvent const& left,
           ProposalFallbackEvent const& right) {
          return left.particle.history_id <
                 right.particle.history_id;
        };
    std::sort(first.interactions.begin(), first.interactions.end(),
              interaction_order);
    std::sort(reordered.interactions.begin(),
              reordered.interactions.end(), interaction_order);
    std::sort(first.fallback_events.begin(),
              first.fallback_events.end(), fallback_order);
    std::sort(reordered.fallback_events.begin(),
              reordered.fallback_events.end(), fallback_order);
    require(first.interactions.size() ==
                reordered.interactions.size() &&
                first.fallback_events.size() ==
                    reordered.fallback_events.size(),
            "queue reorder changed selection counts");
    for (std::size_t i = 0; i < first.interactions.size(); ++i) {
      auto const& left = first.interactions[i];
      auto const& right = reordered.interactions[i];
      require(left.particle.history_id ==
                  right.particle.history_id,
              "queue reorder changed interaction identity");
      require(left.process_id == right.process_id &&
                  left.component_hash == right.component_hash &&
                  left.distance_uniform == right.distance_uniform &&
                  left.process_uniform == right.process_uniform &&
                  left.loss_quantile == right.loss_quantile &&
                  left.energy_fraction == right.energy_fraction &&
                  left.interaction_grammage_g_per_cm2 ==
                      right.interaction_grammage_g_per_cm2,
              "queue reorder changed a history's interaction");
    }
    for (std::size_t i = 0;
         i < first.fallback_events.size(); ++i) {
      auto const& left = first.fallback_events[i];
      auto const& right = reordered.fallback_events[i];
      require(left.particle.history_id ==
                  right.particle.history_id,
              "queue reorder changed fallback identity");
      require(left.reason == right.reason &&
                  left.process_id == right.process_id &&
                  left.component_hash == right.component_hash &&
                  left.loss_quantile == right.loss_quantile,
              "queue reorder changed a history's fallback");
    }
  }

  std::filesystem::path temporaryPath() {
    auto const stamp =
        std::chrono::high_resolution_clock::now()
            .time_since_epoch()
            .count();
    return std::filesystem::temp_directory_path() /
           ("c8_gpu_interaction_selection_" +
            std::to_string(stamp) + ".c8emrt");
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
        << "usage: testGpuInteractionSelection [RATE_TABLE]\n";
    return 2;
  }

  std::filesystem::path temporary;
  try {
    auto const fixture_mode = argc == 1;
    auto source =
        argc == 2 ? readRateTable(argv[1])
                  : testing::makeFlatRateTableFixture();
    if (fixture_mode) {
      source.metadata.energy_cut_MeV = 0.4;
      for (auto& continuous : source.continuous_energy_tables) {
        continuous.minimum_total_energy_MeV =
            continuous.mass_MeV +
            0.5 * ContinuousCutSafetyFactor;
        continuous.energies_MeV.front() =
            continuous.minimum_total_energy_MeV;
      }
    }
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
    config.table_tolerance =
        argc == 2
            ? source.metadata.requested_relative_tolerance
            : 1.e-3;
    config.random_seed = 0x53454c454354ULL;
    config.shower_id = 19;
    config.table_cache = table_path;

    EnvironmentSnapshot environment{};
    CudaEmBackend backend;
    backend.initialize(environment, descriptor, config);
    require(backend.hasProposalTable(),
            "backend did not upload its PROPOSAL table");
    require(backend.proposalTableHash() == digest,
            "backend table hash differs");
    require(backend.statistics().table_device_bytes > 0,
            "backend table byte count is empty");
    require(backend.statistics().peak_device_bytes >=
                backend.statistics().table_device_bytes,
            "backend peak memory excludes its physical table");

    auto const particles =
        makeParticles(source, fixture_mode ? 4096 : 8192,
                      fixture_mode);
    auto const flat = flattenRateTable(source);
    auto const view = makeFlatRateTableView(flat);
    EmInteractionBatchResult expected{};
    expected.input_particles = particles.size();
    for (std::size_t i = 0; i < particles.size(); ++i) {
      auto reference = selectOnHost(
          source, view, particles[i], i, config.random_seed,
          config.shower_id);
      if (reference.fallback) {
        expected.fallback_events.push_back(
            reference.fallback_event);
      } else {
        expected.interactions.push_back(
            reference.interaction);
      }
    }

    auto const first =
        backend.selectInteractionsForValidation(particles);
    auto const second =
        backend.selectInteractionsForValidation(particles);
    compareBatches(first, expected);
    compareBatches(second, first);
    {
      auto normal_time = particles.front();
      normal_time.time_s = 0.;
      auto time_boundary = normal_time;
      time_boundary.history_id += 500000000ULL;
      time_boundary.time_s = ParticleCutMaximumTimeS;
      auto old_secondary = normal_time;
      old_secondary.history_id += 1000000000ULL;
      old_secondary.time_s =
          std::nextafter(
              ParticleCutMaximumTimeS,
              std::numeric_limits<double>::infinity());
      auto const time_selection =
          backend.selectInteractionsForValidation(
              {normal_time, time_boundary, old_secondary});
      require(
          time_selection.fallback_events.empty() &&
              time_selection.interactions.size() == 3,
          "ParticleCut time-boundary selection lost a particle");
      require(
          time_selection.interactions[1].status ==
                  time_selection.interactions[0].status &&
              time_selection.interactions[2].status ==
                  EmInteractionStatus::ParticleCut &&
              std::isinf(
                  time_selection.interactions[2]
                      .interaction_grammage_g_per_cm2),
          "CUDA selector does not preserve scalar timePost > 10 ms semantics");
    }
    if (fixture_mode) {
      auto const split_cut_history = particles.size();
      auto const split_cut_record = std::find_if(
          first.interactions.begin(), first.interactions.end(),
          [&](EmInteractionRecord const& record) {
            return record.particle.history_id ==
                   split_cut_history;
          });
      require(
          split_cut_record != first.interactions.end() &&
              split_cut_record->status ==
                  EmInteractionStatus::ParticleCut,
          "charged-lepton transport used the 0.4 MeV stochastic cut "
          "instead of the 0.5 MeV EM transport cut");
    }
    auto reordered_particles = particles;
    std::reverse(reordered_particles.begin(),
                 reordered_particles.end());
    auto const reordered =
        backend.selectInteractionsForValidation(
            reordered_particles);
    compareSchedulingInvariant(first, reordered);
    require(first.interactions.size() +
                first.fallback_events.size() ==
                particles.size(),
            "selector lost or duplicated a particle");
    double standardized_grammage_sum = 0.;
    std::size_t finite_grammage_count = 0;
    for (auto const& interaction : first.interactions) {
      if (interaction.status == EmInteractionStatus::Selected ||
          interaction.status ==
              EmInteractionStatus::DistanceSampled) {
        standardized_grammage_sum +=
            interaction.interaction_grammage_g_per_cm2 *
            interaction.total_rate_cm2_per_g;
        ++finite_grammage_count;
      }
    }
    require(finite_grammage_count > 1000,
            "too few selected interactions for grammage check");
    auto const standardized_grammage_mean =
        standardized_grammage_sum / finite_grammage_count;
    require(std::abs(standardized_grammage_mean - 1.) < 0.08,
            "interaction grammage is not exponentially distributed");
    for (std::size_t i = 1; i < first.interactions.size(); ++i) {
      require(first.interactions[i].input_index >
                  first.interactions[i - 1].input_index,
              "interaction output is not stably ordered");
    }
    for (std::size_t i = 1;
         i < first.fallback_events.size(); ++i) {
      require(first.fallback_events[i].input_index >
                  first.fallback_events[i - 1].input_index,
              "fallback output is not stably ordered");
    }
    require(backend.statistics().interaction_selection_batches == 4,
            "backend interaction batch statistic differs");
    require(backend.statistics().interactions_selected ==
                3 * first.interactions.size() + 3,
            "backend selected-interaction statistic differs");
    require(backend.statistics().proposal_fallbacks ==
                3 * first.fallback_events.size(),
            "backend fallback statistic differs");
    auto const invariant_table_bytes =
        backend.statistics().table_device_bytes;
    auto const invariant_workspace_bytes =
        backend.statistics().physical_workspace_bytes;
    auto const invariant_peak_bytes =
        backend.statistics().peak_device_bytes;
    auto const one_time_initialization_ms =
        backend.statistics().one_time_initialization_ms;
    auto const static_upload_bytes =
        backend.statistics().static_host_to_device_bytes;
    require(
        backend.statistics().shower_ordinal == 1 &&
            !backend.statistics().reused_for_shower &&
            one_time_initialization_ms > 0. &&
            static_upload_bytes > 0,
        "initial backend lifecycle metadata is incomplete");

    backend.beginShower(makeGpuEmShowerConfig(config));
    auto const& reset_statistics = backend.statistics();
    require(
        reset_statistics.shower_ordinal == 2 &&
            reset_statistics.reused_for_shower &&
            reset_statistics.interaction_selection_batches == 0 &&
            reset_statistics.interactions_selected == 0 &&
            reset_statistics.proposal_fallbacks == 0,
        "backend reuse did not reset per-shower counters");
    require(
        reset_statistics.table_device_bytes ==
                invariant_table_bytes &&
            reset_statistics.physical_workspace_bytes ==
                invariant_workspace_bytes &&
            reset_statistics.peak_device_bytes ==
                invariant_peak_bytes &&
            reset_statistics.one_time_initialization_ms ==
                one_time_initialization_ms &&
            reset_statistics.static_host_to_device_bytes ==
                static_upload_bytes,
        "backend reuse changed invariant allocations or lifecycle metadata");
    auto const reused =
        backend.selectInteractionsForValidation(particles);
    compareBatches(reused, first);
    require(
        backend.statistics().interaction_selection_batches == 1 &&
            backend.statistics().interactions_selected ==
                first.interactions.size() &&
            backend.statistics().proposal_fallbacks ==
                first.fallback_events.size(),
        "reused backend statistics include the preceding shower");

    if (fixture_mode) {
      auto wrong_descriptor = descriptor;
      wrong_descriptor.content_hash[0] ^= 1;
      requireThrows(
          [&] {
            CudaEmBackend wrong;
            wrong.initialize(
                environment, wrong_descriptor, config);
          },
          "backend accepted a mismatched table hash");
      auto wrong_count = descriptor;
      wrong_count.process_count++;
      requireThrows(
          [&] {
            CudaEmBackend wrong;
            wrong.initialize(
                environment, wrong_count, config);
          },
          "backend accepted a mismatched process count");
      auto wrong_format = descriptor;
      wrong_format.format_version--;
      requireThrows(
          [&] {
            CudaEmBackend wrong;
            wrong.initialize(
                environment, wrong_format, config);
          },
          "backend accepted a mismatched table format");
      auto strict_config = config;
      strict_config.table_tolerance = 1.e-4;
      requireThrows(
          [&] {
            CudaEmBackend strict;
            strict.initialize(
                environment, descriptor, strict_config);
          },
          "backend accepted an insufficient table tolerance");

      auto toy_config = config;
      toy_config.table_cache.clear();
      CudaEmBackend toy;
      toy.initialize(environment, ProposalTableSet{}, toy_config);
      require(!toy.hasProposalTable(),
              "empty table path unexpectedly uploaded a table");
      requireThrows(
          [&] {
            toy.selectInteractionsForValidation(particles);
          },
          "table-free backend accepted physical selection");
      toy.enqueue(particles.front());
      requireThrows(
          [&] {
            toy.beginShower(
                makeGpuEmShowerConfig(toy_config));
          },
          "backend reuse discarded a non-empty staged wavefront");
    }

    std::cout << "CUDA interaction selection passed " << checks
              << " checks for " << particles.size()
              << " particles: " << first.interactions.size()
              << " selected, " << first.fallback_events.size()
              << " fallback, "
              << backend.statistics().table_device_bytes
              << " table bytes\n";
  } catch (std::exception const& error) {
    if (!temporary.empty()) {
      std::error_code ignored;
      std::filesystem::remove(temporary, ignored);
    }
    std::cerr << "CUDA interaction selection failed after "
              << checks << " checks: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
  if (!temporary.empty()) {
    std::error_code ignored;
    std::filesystem::remove(temporary, ignored);
  }
  return EXIT_SUCCESS;
}

/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

/*
 * Validation-only scalar-PROPOSAL/CUDA random-contract oracle.
 *
 * This program is not part of the production cascade.  It asks the CUDA
 * selector for its actual Philox uniforms, injects the process uniform into
 * scalar PROPOSAL, and reports the first physical decision that differs.
 */

#include <PROPOSAL/PROPOSAL.h>

#include <corsika/framework/core/ParticleProperties.hpp>
#include <corsika/framework/core/PhysicalUnits.hpp>
#include <corsika/gpu/em/CudaEmBackend.hpp>
#include <corsika/gpu/em/tables/MediumConfig.hpp>
#include <corsika/gpu/em/tables/ProposalMedium.hpp>
#include <corsika/gpu/em/tables/RateTable.hpp>
#include <corsika/modules/proposal/ProposalProcessBase.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

  using namespace corsika;
  using namespace corsika::gpu::em;
  using namespace corsika::gpu::em::tables;
  using namespace corsika::units::si;

  struct Options {
    std::filesystem::path table;
    std::filesystem::path medium;
    std::filesystem::path proposal_cache;
    std::filesystem::path output;
    std::size_t samples_per_particle{4096};
    std::uint64_t seed{2026081002};
    std::uint64_t shower_id{1};
    int device{0};
  };

  struct FirstDivergence {
    bool set{};
    std::string kind;
    std::int32_t pid{};
    std::size_t sample{};
    double energy_MeV{};
    double process_uniform{};
    double gpu_loss_uniform{};
    double cpu_conditional_quantile{};
    std::int32_t cpu_process{};
    std::int32_t gpu_process{};
    std::uint64_t cpu_component{};
    std::uint64_t gpu_component{};
    double cpu_v{};
    double gpu_v{};
    double semantic_gpu_v{};
    double semantic_relative_error{};
  };

  struct ParticleSummary {
    std::int32_t pid{};
    std::size_t requested{};
    std::size_t compared{};
    std::size_t fallbacks{};
    std::size_t process_component_mismatches{};
    std::size_t canonical_order_process_mismatches{};
    std::size_t raw_loss_bitwise_mismatches{};
    std::size_t semantic_loss_compared{};
    std::size_t semantic_loss_unavailable{};
    std::size_t semantic_loss_outside_tolerance{};
    double maximum_semantic_relative_error{};
    double maximum_raw_relative_difference{};
    std::optional<FirstDivergence> first;
    std::optional<FirstDivergence> maximum_semantic_record;
  };

  Options parseOptions(int argc, char** argv) {
    if (argc < 5) {
      throw std::invalid_argument(
          "usage: gpu_em_same_random_process_oracle TABLE MEDIUM_YAML "
          "PROPOSAL_CACHE OUTPUT_JSON [SAMPLES_PER_PARTICLE] [SEED] [DEVICE]");
    }
    Options options;
    options.table = argv[1];
    options.medium = argv[2];
    options.proposal_cache = argv[3];
    options.output = argv[4];
    if (argc >= 6) {
      options.samples_per_particle = std::stoull(argv[5]);
    }
    if (argc >= 7) {
      options.seed = std::stoull(argv[6]);
    }
    if (argc >= 8) {
      options.device = std::stoi(argv[7]);
    }
    if (argc > 8 || options.samples_per_particle == 0) {
      throw std::invalid_argument("invalid same-random oracle arguments");
    }
    return options;
  }

  Code corsikaCode(std::int32_t pid) {
    switch (pid) {
      case 11:
        return Code::Electron;
      case -11:
        return Code::Positron;
      case 22:
        return Code::Photon;
      default:
        throw std::invalid_argument("unsupported oracle PID");
    }
  }

  std::size_t processCount(RateTableSet const& table) {
    std::size_t count = 0;
    for (auto const& particle : table.particles) {
      count += particle.columns.size();
    }
    return count;
  }

  double relativeDifference(double left, double right) {
    return std::abs(left - right) /
           std::max({std::abs(left), std::abs(right), 1.e-15});
  }

  std::vector<EmParticleState> makeParticles(
      ParticleRateTable const& table, std::size_t count,
      std::uint64_t first_history) {
    if (table.energies_MeV.size() < 2) {
      throw std::runtime_error("particle table has no usable energy range");
    }
    auto const log_min = std::log(table.energies_MeV.front() * 1.01);
    auto const log_max = std::log(table.energies_MeV.back() / 1.01);
    std::vector<EmParticleState> particles;
    particles.reserve(count);
    for (std::size_t index = 0; index < count; ++index) {
      auto const scrambled =
          std::fmod((static_cast<double>(index) + 0.5) *
                        0.6180339887498948482,
                    1.);
      EmParticleState particle{};
      particle.pid = table.pdg_id;
      particle.energy_GeV =
          std::exp(log_min + scrambled * (log_max - log_min)) / 1000.;
      particle.direction[2] = -1.;
      particle.weight = 1.;
      particle.history_id = first_history + index;
      particle.step_id = index % 17;
      particles.push_back(particle);
    }
    return particles;
  }

  std::vector<EmInteractionRecord> selectOnDevice(
      CudaEmBackend& backend, std::int32_t pid,
      std::vector<EmParticleState> const& particles,
      std::size_t& fallback_count) {
    if (pid == 22) {
      auto result = backend.selectInteractionsForValidation(particles);
      fallback_count = result.fallback_events.size();
      return std::move(result.interactions);
    }

    auto distances = backend.selectInteractionsForValidation(particles);
    fallback_count = distances.fallback_events.size();
    std::vector<EmInteractionRecord> candidates;
    candidates.reserve(distances.interactions.size());
    for (auto record : distances.interactions) {
      if (record.status != EmInteractionStatus::DistanceSampled) {
        continue;
      }
      record.status = EmInteractionStatus::RequiresReselection;
      record.vertex_total_rate_cm2_per_g = record.total_rate_cm2_per_g;
      candidates.push_back(record);
    }
    auto selected =
        backend.reselectLeptonInteractionsAtVertexForValidation(candidates);
    fallback_count += selected.fallback_events.size();
    return std::move(selected.interactions);
  }

  struct CpuSelection {
    PROPOSAL::Interaction::Loss loss;
    double conditional_quantile{};
    std::int32_t canonical_process{};
    std::uint64_t canonical_component{};
  };

  CpuSelection sampleCpu(
      PROPOSAL::Interaction& interaction, double energy_MeV,
      double uniform, ParticleRateTable const& particle_table) {
    auto const rates = interaction.Rates(energy_MeV);
    auto const loss = interaction.SampleLoss(energy_MeV, rates, uniform);
    auto const total = std::accumulate(
        rates.begin(), rates.end(), 0.,
        [](double sum, auto const& rate) { return sum + rate.rate; });
    if (!(total > 0.) ||
        loss.type == PROPOSAL::InteractionType::Undefined) {
      return {loss, 0., 0, 0};
    }

    // Reorder exact scalar-PROPOSAL rates into the serialized CUDA column
    // order.  This isolates ordering differences from interpolation errors.
    auto const canonical_threshold = uniform * total;
    double canonical_cumulative = 0.;
    std::int32_t canonical_process = 0;
    std::uint64_t canonical_component = 0;
    for (auto const& column : particle_table.columns) {
      auto const native = std::find_if(
          rates.begin(), rates.end(), [&](auto const& rate) {
            return rate.crosssection &&
                   static_cast<std::int32_t>(
                       rate.crosssection->GetInteractionType()) ==
                       column.process_id &&
                   static_cast<std::uint64_t>(rate.comp_hash) ==
                       column.component_hash;
          });
      if (native == rates.end()) {
        // Auxiliary final-state tables (for example the normalized photon
        // pair split) share RateColumn serialization but have zero physical
        // rate and are not returned by Interaction::Rates().
        auto const serialized_rate = interpolateRate(
            particle_table, column.process_id, column.component_hash,
            energy_MeV);
        if (serialized_rate != 0.) {
          throw std::runtime_error(
              "nonzero CUDA rate column is absent from scalar PROPOSAL Rates");
        }
        continue;
      }
      canonical_cumulative += native->rate;
      if (canonical_process == 0 &&
          canonical_threshold < canonical_cumulative) {
        canonical_process = column.process_id;
        canonical_component = column.component_hash;
      }
    }
    if (canonical_process == 0) {
      throw std::runtime_error(
          "canonical PROPOSAL rate order did not select a column");
    }

    auto const threshold = uniform * total;
    double cumulative = 0.;
    for (auto const& rate : rates) {
      auto const next = cumulative + rate.rate;
      if (threshold < next) {
        if (!rate.crosssection ||
            rate.crosssection->GetInteractionType() != loss.type ||
            rate.comp_hash != loss.comp_hash || !(rate.rate > 0.)) {
          throw std::runtime_error(
              "PROPOSAL SampleLoss ordering differs from its Rates ordering");
        }
        // Interaction::SampleLoss subtracts columns from u * total and passes
        // -sampled_rate to CalculateStochasticLoss().  The conditional
        // coordinate is therefore measured from the upper edge of the
        // selected column, not from its lower edge.
        auto quantile = (next - threshold) / rate.rate;
        quantile = std::clamp(
            quantile, 0.,
            std::nextafter(1., 0.));
        return {loss, quantile, canonical_process, canonical_component};
      }
      cumulative = next;
    }
    throw std::runtime_error("cannot resolve CPU conditional loss quantile");
  }

  ParticleSummary compareParticle(
      CudaEmBackend& backend, RateTableSet const& table,
      PROPOSAL::Medium& medium, ParticleRateTable const& particle_table,
      Options const& options, std::uint64_t first_history) {
    ParticleSummary summary;
    summary.pid = particle_table.pdg_id;
    summary.requested = options.samples_per_particle;
    auto const particles = makeParticles(
        particle_table, options.samples_per_particle, first_history);
    auto interactions = selectOnDevice(
        backend, particle_table.pdg_id, particles, summary.fallbacks);
    std::sort(interactions.begin(), interactions.end(),
              [](auto const& left, auto const& right) {
                return left.input_index < right.input_index;
              });

    auto cross_sections = proposal::make_cross_sections(
        corsikaCode(particle_table.pdg_id), medium,
        table.metadata.energy_cut_MeV * 1_MeV, true);
    auto cpu_interaction =
        PROPOSAL::make_interaction(cross_sections, true, true);
    auto const tolerance =
        table.metadata.requested_loss_relative_tolerance > 0.
            ? table.metadata.requested_loss_relative_tolerance
            : table.metadata.requested_relative_tolerance;

    for (auto const& gpu : interactions) {
      if (gpu.status != EmInteractionStatus::Selected) {
        continue;
      }
      ++summary.compared;
      auto const energy_MeV = gpu.particle.energy_GeV * 1000.;
      auto const cpu = sampleCpu(
          *cpu_interaction, energy_MeV, gpu.process_uniform,
          particle_table);
      auto const cpu_process = static_cast<std::int32_t>(cpu.loss.type);
      auto const cpu_component =
          static_cast<std::uint64_t>(cpu.loss.comp_hash);
      auto const same_column =
          cpu_process == gpu.process_id &&
          cpu_component == gpu.component_hash;
      if (cpu.canonical_process != gpu.process_id ||
          cpu.canonical_component != gpu.component_hash) {
        ++summary.canonical_order_process_mismatches;
      }

      FirstDivergence divergence;
      divergence.pid = particle_table.pdg_id;
      divergence.sample = gpu.input_index;
      divergence.energy_MeV = energy_MeV;
      divergence.process_uniform = gpu.process_uniform;
      divergence.gpu_loss_uniform = gpu.loss_quantile;
      divergence.cpu_conditional_quantile = cpu.conditional_quantile;
      divergence.cpu_process = cpu_process;
      divergence.gpu_process = gpu.process_id;
      divergence.cpu_component = cpu_component;
      divergence.gpu_component = gpu.component_hash;
      divergence.cpu_v = cpu.loss.v_loss;
      divergence.gpu_v = gpu.energy_fraction;

      if (!same_column) {
        ++summary.process_component_mismatches;
        divergence.kind = "process_or_component";
      } else if (cpu.loss.v_loss != gpu.energy_fraction) {
        ++summary.raw_loss_bitwise_mismatches;
        summary.maximum_raw_relative_difference = std::max(
            summary.maximum_raw_relative_difference,
            relativeDifference(cpu.loss.v_loss, gpu.energy_fraction));
        divergence.kind = "raw_random_contract_loss";
      }

      try {
        auto const semantic_gpu_v = interpolateLossFraction(
            particle_table, cpu_process, cpu_component,
            energy_MeV, cpu.conditional_quantile);
        ++summary.semantic_loss_compared;
        divergence.semantic_gpu_v = semantic_gpu_v;
        auto const semantic_error = relativeDifference(
            cpu.loss.v_loss, semantic_gpu_v);
        divergence.semantic_relative_error = semantic_error;
        if (semantic_error > summary.maximum_semantic_relative_error) {
          summary.maximum_semantic_relative_error = semantic_error;
          auto maximum = divergence;
          maximum.kind = "maximum_semantic_inverse_cdf_error";
          summary.maximum_semantic_record = maximum;
        }
        if (semantic_error > tolerance) {
          ++summary.semantic_loss_outside_tolerance;
          if (divergence.kind.empty()) {
            divergence.kind = "semantic_inverse_cdf";
          }
        }
      } catch (std::out_of_range const&) {
        ++summary.semantic_loss_unavailable;
      } catch (std::invalid_argument const&) {
        ++summary.semantic_loss_unavailable;
      }
      if (!divergence.kind.empty() && !summary.first) {
        summary.first = divergence;
      }
    }
    return summary;
  }

  void writeDivergence(std::ostream& output,
                       std::optional<FirstDivergence> const& value) {
    if (!value) {
      output << "null";
      return;
    }
    auto const& d = *value;
    output << "{\n"
           << "        \"kind\": \"" << d.kind << "\",\n"
           << "        \"pid\": " << d.pid << ",\n"
           << "        \"sample\": " << d.sample << ",\n"
           << "        \"energy_MeV\": " << d.energy_MeV << ",\n"
           << "        \"process_uniform\": " << d.process_uniform << ",\n"
           << "        \"gpu_loss_uniform\": " << d.gpu_loss_uniform << ",\n"
           << "        \"cpu_conditional_quantile\": "
           << d.cpu_conditional_quantile << ",\n"
           << "        \"cpu_process\": " << d.cpu_process << ",\n"
           << "        \"gpu_process\": " << d.gpu_process << ",\n"
           << "        \"cpu_component\": " << d.cpu_component << ",\n"
           << "        \"gpu_component\": " << d.gpu_component << ",\n"
           << "        \"cpu_v\": " << d.cpu_v << ",\n"
           << "        \"gpu_v\": " << d.gpu_v << ",\n"
           << "        \"semantic_gpu_v\": " << d.semantic_gpu_v << ",\n"
           << "        \"semantic_relative_error\": "
           << d.semantic_relative_error << "\n"
           << "      }";
  }

  void writeReport(Options const& options, RateTableSet const& table,
                   std::vector<ParticleSummary> const& summaries) {
    std::filesystem::create_directories(options.output.parent_path());
    std::ofstream output(options.output);
    if (!output) {
      throw std::runtime_error("cannot open oracle JSON output");
    }
    output << std::setprecision(17);
    output << "{\n"
           << "  \"mode\": \"same-process-uniform-first-divergence\",\n"
           << "  \"production_path_modified\": false,\n"
           << "  \"table\": \"" << options.table.string() << "\",\n"
           << "  \"medium\": \"" << options.medium.string() << "\",\n"
           << "  \"seed\": " << options.seed << ",\n"
           << "  \"samples_per_particle\": "
           << options.samples_per_particle << ",\n"
           << "  \"table_rate_tolerance\": "
           << table.metadata.requested_relative_tolerance << ",\n"
           << "  \"table_loss_tolerance\": "
           << table.metadata.requested_loss_relative_tolerance << ",\n"
           << "  \"random_contract\": {\n"
           << "    \"scalar_proposal\": \"one uniform jointly selects "
              "process, component, and within-column loss\",\n"
           << "    \"cuda\": \"one uniform selects process/component and a "
              "second uniform selects within-column loss\"\n"
           << "  },\n"
           << "  \"particles\": [\n";
    for (std::size_t index = 0; index < summaries.size(); ++index) {
      auto const& s = summaries[index];
      output << "    {\n"
             << "      \"pid\": " << s.pid << ",\n"
             << "      \"requested\": " << s.requested << ",\n"
             << "      \"compared\": " << s.compared << ",\n"
             << "      \"fallbacks\": " << s.fallbacks << ",\n"
             << "      \"process_component_mismatches\": "
             << s.process_component_mismatches << ",\n"
             << "      \"canonical_order_process_mismatches\": "
             << s.canonical_order_process_mismatches << ",\n"
             << "      \"raw_loss_bitwise_mismatches\": "
             << s.raw_loss_bitwise_mismatches << ",\n"
             << "      \"semantic_loss_compared\": "
             << s.semantic_loss_compared << ",\n"
             << "      \"semantic_loss_unavailable\": "
             << s.semantic_loss_unavailable << ",\n"
             << "      \"semantic_loss_outside_tolerance\": "
             << s.semantic_loss_outside_tolerance << ",\n"
             << "      \"maximum_raw_relative_difference\": "
             << s.maximum_raw_relative_difference << ",\n"
             << "      \"maximum_semantic_relative_error\": "
             << s.maximum_semantic_relative_error << ",\n"
             << "      \"first_divergence\": ";
      writeDivergence(output, s.first);
      output << ",\n      \"maximum_semantic_record\": ";
      writeDivergence(output, s.maximum_semantic_record);
      output << "\n    }" << (index + 1 == summaries.size() ? "\n" : ",\n");
    }
    output << "  ]\n}\n";
  }

} // namespace

int main(int argc, char** argv) {
  try {
    auto const options = parseOptions(argc, argv);
    std::filesystem::create_directories(options.proposal_cache);
    PROPOSAL::InterpolationSettings::TABLES_PATH =
        options.proposal_cache.string();
    PROPOSAL::Logging::SetGlobalLoglevel(spdlog::level::critical);

    auto const table = readRateTable(options.table);
    auto const medium_config = loadMediumConfig(options.medium);
    auto medium = makeProposalMedium(medium_config);

    ProposalTableSet descriptor{};
    descriptor.process_count =
        static_cast<std::uint32_t>(processCount(table));
    descriptor.content_hash = table.content_hash;
    GpuEmConfig config{};
    config.device = options.device;
    config.min_batch_size = 1;
    config.memory_fraction = 0.01;
    config.table_tolerance =
        table.metadata.requested_relative_tolerance;
    config.random_seed = options.seed;
    config.shower_id = options.shower_id;
    config.table_cache = options.table;
    CudaEmBackend backend;
    backend.initialize(EnvironmentSnapshot{}, descriptor, config);

    std::vector<ParticleSummary> summaries;
    std::uint64_t first_history = 1;
    for (auto const pid : {11, -11, 22}) {
      auto const found = std::find_if(
          table.particles.begin(), table.particles.end(),
          [pid](auto const& particle) { return particle.pdg_id == pid; });
      if (found == table.particles.end()) {
        continue;
      }
      summaries.push_back(compareParticle(
          backend, table, medium, *found, options, first_history));
      first_history += options.samples_per_particle + 1;
    }
    writeReport(options, table, summaries);

    std::cout << "same-random process oracle wrote " << options.output << '\n';
    for (auto const& summary : summaries) {
      std::cout << "pid=" << summary.pid
                << " compared=" << summary.compared
                << " process_mismatch="
                << summary.process_component_mismatches
                << " canonical_order_mismatch="
                << summary.canonical_order_process_mismatches
                << " raw_loss_mismatch="
                << summary.raw_loss_bitwise_mismatches
                << " semantic_outside_tolerance="
                << summary.semantic_loss_outside_tolerance
                << " max_semantic_relative="
                << summary.maximum_semantic_relative_error << '\n';
    }
    return 0;
  } catch (std::exception const& error) {
    std::cerr << "same-random process oracle failed: "
              << error.what() << '\n';
    return 1;
  }
}

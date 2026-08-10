/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

/* Dump beta2-table, cached-PROPOSAL and direct-PROPOSAL electron curves. */

#include <PROPOSAL/PROPOSAL.h>
#include <PROPOSAL/version.h>

#include <corsika/framework/core/ParticleProperties.hpp>
#include <corsika/framework/core/PhysicalUnits.hpp>
#include <corsika/gpu/em/tables/MediumConfig.hpp>
#include <corsika/gpu/em/tables/ProposalMedium.hpp>
#include <corsika/gpu/em/tables/RateTable.hpp>
#include <corsika/modules/proposal/ProposalProcessBase.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

  using namespace corsika;
  using namespace corsika::gpu::em::tables;
  using namespace corsika::units::si;

  std::string lower(std::string text) {
    std::transform(text.begin(), text.end(), text.begin(),
                   [](unsigned char value) {
                     return static_cast<char>(std::tolower(value));
                   });
    return text;
  }

  std::string componentName(RateTableSet const& table,
                            std::uint64_t hash) {
    if (hash == table.metadata.proposal_medium_hash) {
      return table.metadata.medium_name;
    }
    for (auto const& component : table.metadata.components) {
      if (component.proposal_hash == hash) {
        switch (component.corsika_pid) {
          case 1000070140:
            return "Nitrogen-14";
          case 1000080160:
            return "Oxygen-16";
          case 1000180400:
            return "Argon-40";
          default:
            return component.name + "_pid_" +
                   std::to_string(component.corsika_pid);
        }
      }
    }
    return "component_" + std::to_string(hash);
  }

  RateColumn const& findColumn(ParticleRateTable const& particle,
                               RateTableSet const& table,
                               std::string_view process,
                               std::int32_t target_pid) {
    auto const wanted_process = lower(std::string(process));
    auto const component = std::find_if(
        table.metadata.components.begin(), table.metadata.components.end(),
        [target_pid](auto const& candidate) {
          return candidate.corsika_pid == target_pid;
        });
    if (component == table.metadata.components.end()) {
      throw std::runtime_error("missing target PID " +
                               std::to_string(target_pid));
    }
    for (auto const& column : particle.columns) {
      if (lower(column.process_name) == wanted_process &&
          column.component_hash == component->proposal_hash) {
        return column;
      }
    }
    throw std::runtime_error("missing table column " +
                             std::string(process) + "/" +
                             std::to_string(target_pid));
  }

  RateColumn const& findMediumColumn(ParticleRateTable const& particle,
                                     RateTableSet const& table,
                                     std::string_view process) {
    auto const wanted_process = lower(std::string(process));
    for (auto const& column : particle.columns) {
      if (lower(column.process_name) == wanted_process &&
          column.component_hash == table.metadata.proposal_medium_hash) {
        return column;
      }
    }
    throw std::runtime_error("missing medium table column " +
                             std::string(process));
  }

  std::shared_ptr<PROPOSAL::CrossSectionBase> findCrossSection(
      PROPOSAL::crosssection_list_t const& cross_sections,
      std::int32_t process_id) {
    auto const found = std::find_if(
        cross_sections.begin(), cross_sections.end(),
        [process_id](auto const& cross) {
          return static_cast<std::int32_t>(cross->GetInteractionType()) ==
                 process_id;
        });
    if (found == cross_sections.end()) {
      throw std::runtime_error("missing PROPOSAL cross section for process " +
                               std::to_string(process_id));
    }
    return *found;
  }

  double proposalLoss(std::shared_ptr<PROPOSAL::CrossSectionBase> const& cross,
                      std::uint64_t component_hash, double energy_MeV,
                      double quantile) {
    auto const rate = cross->CalculatedNdx(
        energy_MeV, static_cast<std::size_t>(component_hash));
    if (!(rate > 0.)) {
      throw std::runtime_error("non-positive PROPOSAL column rate");
    }
    return cross->CalculateStochasticLoss(
        static_cast<std::size_t>(component_hash), energy_MeV,
        quantile * rate);
  }

  std::vector<double> linearGrid(double minimum, double maximum,
                                 std::size_t count) {
    std::vector<double> values;
    values.reserve(count);
    for (std::size_t index = 0; index < count; ++index) {
      auto const fraction = static_cast<double>(index) /
                            static_cast<double>(count - 1);
      values.push_back(minimum + fraction * (maximum - minimum));
    }
    return values;
  }

  std::vector<double> logitGrid(double minimum, double maximum,
                                std::size_t count) {
    auto const logit = [](double value) {
      return std::log(value / (1. - value));
    };
    auto const logistic = [](double value) {
      return value >= 0. ? 1. / (1. + std::exp(-value))
                        : std::exp(value) / (1. + std::exp(value));
    };
    auto const coordinates = linearGrid(logit(minimum), logit(maximum), count);
    std::vector<double> values;
    values.reserve(count);
    for (auto const coordinate : coordinates) {
      values.push_back(logistic(coordinate));
    }
    return values;
  }

} // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 5) {
      std::cerr << "usage: " << argv[0]
                << " TABLE MEDIUM_YAML PROPOSAL_CACHE OUTPUT_CSV\n";
      return 2;
    }
    auto const table_path = std::filesystem::path(argv[1]);
    auto const medium_path = std::filesystem::path(argv[2]);
    auto const proposal_cache = std::filesystem::path(argv[3]);
    auto const output_path = std::filesystem::path(argv[4]);

    std::filesystem::create_directories(proposal_cache);
    PROPOSAL::InterpolationSettings::TABLES_PATH = proposal_cache.string();
    PROPOSAL::Logging::SetGlobalLoglevel(spdlog::level::critical);

    auto const table = readRateTable(table_path);
    auto const& electron = findParticle(table, 11);
    auto const medium_config = loadMediumConfig(medium_path);
    auto medium = makeProposalMedium(medium_config);
    auto const interpolated = proposal::make_cross_sections(
        Code::Electron, medium, table.metadata.energy_cut_MeV * 1_MeV, true);
    auto const direct = proposal::make_cross_sections(
        Code::Electron, medium, table.metadata.energy_cut_MeV * 1_MeV, false);

    auto const& argon = findColumn(electron, table, "Brems", 1000180400);
    auto const& nitrogen = findColumn(electron, table, "Brems", 1000070140);
    auto const& oxygen = findColumn(electron, table, "Brems", 1000080160);
    auto const& ionization = findMediumColumn(electron, table, "Ioniz");

    std::filesystem::create_directories(output_path.parent_path());
    std::ofstream output(output_path);
    if (!output) {
      throw std::runtime_error("cannot open output CSV");
    }
    output << std::setprecision(17);
    output << "kind,dataset,process,target,energy_MeV,quantile,"
              "beta2_table,proposal_interpolated,proposal_direct\n";

    auto dump_inverse = [&](std::string const& dataset,
                            RateColumn const& column, double energy_MeV,
                            std::vector<double> const& quantiles) {
      auto const cached_cross =
          findCrossSection(interpolated, column.process_id);
      auto const direct_cross = findCrossSection(direct, column.process_id);
      auto const target = componentName(table, column.component_hash);
      for (auto const quantile : quantiles) {
        auto const table_loss = interpolateLossFraction(
            electron, column.process_id, column.component_hash,
            energy_MeV, quantile);
        auto const cached_loss = proposalLoss(
            cached_cross, column.component_hash, energy_MeV, quantile);
        auto const direct_loss = proposalLoss(
            direct_cross, column.component_hash, energy_MeV, quantile);
        output << "inverse_cdf," << dataset << ',' << column.process_name
               << ',' << target << ',' << energy_MeV << ',' << quantile
               << ',' << table_loss << ',' << cached_loss << ','
               << direct_loss << '\n';
      }
    };

    constexpr double DiagnosticEnergyMeV = 623.7318908;
    dump_inverse("argon_brems_full", argon, DiagnosticEnergyMeV,
                 logitGrid(1.e-5, 0.9999, 321));
    dump_inverse("argon_brems_tail", argon, DiagnosticEnergyMeV,
                 linearGrid(0.90, 0.9995, 321));
    dump_inverse("argon_brems_wide", argon, DiagnosticEnergyMeV,
                 linearGrid(0.982, 0.990, 161));
    auto argon_zoom = linearGrid(0.98596, 0.98602, 241);
    // Preserve the two adjacent quantiles recorded in the beta2 numerical
    // diagnostic so the CSV contains the published values exactly.
    argon_zoom.push_back(0.9859885644);
    argon_zoom.push_back(0.9859897698);
    std::sort(argon_zoom.begin(), argon_zoom.end());
    argon_zoom.erase(std::unique(argon_zoom.begin(), argon_zoom.end()),
                     argon_zoom.end());
    dump_inverse("argon_brems_zoom", argon, DiagnosticEnergyMeV, argon_zoom);
    dump_inverse("nitrogen_brems", nitrogen, DiagnosticEnergyMeV,
                 logitGrid(1.e-4, 0.999, 161));
    dump_inverse("oxygen_brems", oxygen, DiagnosticEnergyMeV,
                 logitGrid(1.e-4, 0.999, 161));
    dump_inverse("air_ionization", ionization, DiagnosticEnergyMeV,
                 logitGrid(1.e-4, 0.999, 161));

    auto dump_rates = [&](RateColumn const& column) {
      auto const cached_cross =
          findCrossSection(interpolated, column.process_id);
      auto const direct_cross = findCrossSection(direct, column.process_id);
      auto const target = componentName(table, column.component_hash);
      auto const energies_log = linearGrid(std::log(1.5), std::log(1.e5), 181);
      for (auto const log_energy : energies_log) {
        auto const energy = std::exp(log_energy);
        auto const table_rate = interpolateRate(
            electron, column.process_id, column.component_hash, energy);
        auto const cached_rate = cached_cross->CalculatedNdx(
            energy, static_cast<std::size_t>(column.component_hash));
        auto const direct_rate = direct_cross->CalculatedNdx(
            energy, static_cast<std::size_t>(column.component_hash));
        output << "rate,electron_rates," << column.process_name << ','
               << target << ',' << energy << ",nan," << table_rate << ','
               << cached_rate << ',' << direct_rate << '\n';
      }
    };
    dump_rates(nitrogen);
    dump_rates(oxygen);
    dump_rates(argon);
    dump_rates(ionization);

    std::cerr << "wrote " << output_path << '\n';
    return 0;
  } catch (std::exception const& error) {
    std::cerr << "curve dump failed: " << error.what() << '\n';
    return 1;
  }
}

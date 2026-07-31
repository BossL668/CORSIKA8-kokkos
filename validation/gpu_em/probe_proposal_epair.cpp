/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

/*
 * Diagnostic oracle for the PROPOSAL electron-pair stochastic-loss inverse CDF.
 *
 * This file is intentionally not part of the default build.  It can be compiled
 * against the same PROPOSAL installation as gpu_em_tablegen when investigating
 * interpolation features near the Epair threshold/cut transition.
 */

#include <PROPOSAL/PROPOSAL.h>
#include <PROPOSAL/Constants.h>

#include <corsika/framework/core/ParticleProperties.hpp>
#include <corsika/framework/core/PhysicalUnits.hpp>
#include <corsika/media/CORSIKA7Atmospheres.hpp>
#include <corsika/media/MediumProperties.hpp>
#include <corsika/modules/proposal/ProposalProcessBase.hpp>

#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

  using namespace corsika;
  using namespace corsika::units::si;

  PROPOSAL::Medium makeDryAirMedium() {
    auto const& medium_data = mediumData(Medium::AirDry1Atm);
    std::vector<PROPOSAL::Component> components;
    auto fraction = standardAirComposition.getFractions().begin();
    for (auto const code : standardAirComposition.getComponents()) {
      components.emplace_back(std::string(get_name(code)),
                              get_nucleus_Z(code),
                              get_nucleus_A(code), *fraction++);
    }
    return PROPOSAL::Medium(
        medium_data.getName(), medium_data.getIeff(), -medium_data.getCbar(),
        medium_data.getAA(), medium_data.getSK(), medium_data.getX0(),
        medium_data.getX1(), medium_data.getDlt0(),
        medium_data.getCorrectedDensity(), std::move(components));
  }

  double parse(char const* value, char const* label) {
    char* end = nullptr;
    auto const result = std::strtod(value, &end);
    if (end == value || *end != '\0') {
      throw std::invalid_argument(std::string("invalid ") + label);
    }
    return result;
  }

} // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 7) {
      std::cerr
          << "usage: probe_proposal_epair CACHE CUT_MEV E_MIN E_MAX "
             "E_STEP QUANTILE\n";
      return 2;
    }
    auto const cache = std::filesystem::path(argv[1]);
    auto const cut_MeV = parse(argv[2], "cut");
    auto const energy_min = parse(argv[3], "minimum energy");
    auto const energy_max = parse(argv[4], "maximum energy");
    auto const energy_step = parse(argv[5], "energy step");
    auto const quantile = parse(argv[6], "quantile");
    if (!(cut_MeV > 0.) || !(energy_min > 0.) ||
        !(energy_max >= energy_min) || !(energy_step > 0.) ||
        !(quantile > 0. && quantile < 1.)) {
      throw std::invalid_argument("numeric arguments are out of range");
    }

    std::filesystem::create_directories(cache);
    PROPOSAL::InterpolationSettings::TABLES_PATH = cache.string();
    PROPOSAL::Logging::SetGlobalLoglevel(spdlog::level::critical);
    auto medium = makeDryAirMedium();
    auto cross_sections = proposal::make_cross_sections(
        Code::Electron, medium, cut_MeV * 1_MeV, true);

    std::shared_ptr<PROPOSAL::CrossSectionBase> epair;
    for (auto const& cross : cross_sections) {
      if (cross->GetInteractionType() ==
          PROPOSAL::InteractionType::Epair) {
        epair = cross;
        break;
      }
    }
    if (!epair) {
      throw std::runtime_error("electron Epair cross section is absent");
    }

    auto const targets = epair->CalculatedNdx_PerTarget(energy_min);
    std::cout << std::setprecision(17)
              << "# energy_MeV target_hash rate_per_gcm2 quantile v_loss "
                 "loss_MeV\n";
    for (auto energy = energy_min;
         energy <= energy_max + 0.5 * energy_step;
         energy += energy_step) {
      for (auto const& [target_hash, ignored] : targets) {
        (void)ignored;
        auto const rate =
            epair->CalculatedNdx(energy, target_hash);
        if (!(rate > 0.)) {
          continue;
        }
        auto const v = epair->CalculateStochasticLoss(
            target_hash, energy, quantile * rate);
        std::cout << energy << ' ' << target_hash << ' ' << rate << ' '
                  << quantile << ' ' << v << ' ' << energy * v << '\n';
      }
    }
    return 0;
  } catch (std::exception const& error) {
    std::cerr << "probe_proposal_epair failed: " << error.what() << '\n';
    return 1;
  }
}

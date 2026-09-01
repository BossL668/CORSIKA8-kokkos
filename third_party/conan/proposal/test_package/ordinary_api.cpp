/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include "PROPOSAL/Constants.h"
#include "PROPOSAL/EnergyCutSettings.h"
#include "PROPOSAL/Logging.h"
#include "PROPOSAL/crosssection/CrossSection.h"
#include "PROPOSAL/crosssection/Factories/ComptonFactory.h"
#include "PROPOSAL/medium/Medium.h"
#include "PROPOSAL/particle/ParticleDef.h"

#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

#include <array>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>

int main(int argc, char** argv) {
  using namespace PROPOSAL;

  Logging::SetGlobalLoglevel(spdlog::level::critical);
  InterpolationSettings::TABLES_PATH = argc > 1 ? argv[1] : ".";
  InterpolationSettings::UPPER_ENERGY_LIM = 1.e7;
  InterpolationSettings::NODES_DEDX = 32u;
  InterpolationSettings::NODES_DNDX_E = 24u;
  InterpolationSettings::NODES_DNDX_V = 24u;

  GammaDef const photon;
  Air const air;
  auto const cuts = std::make_shared<EnergyCutSettings>(0.5, 0.1, false);
  nlohmann::json config{{"parametrization", "KleinNishina"}};
  auto direct = make_compton(photon, air, cuts, false, config);
  auto interpolated = make_compton(photon, air, cuts, true, config);

  std::cout << std::hexfloat;
  for (double const energy :
       std::array<double, 4>{10., 100., 1.e4, 1.e6}) {
    std::cout << "energy " << energy << '\n';
    std::cout << "direct " << direct->CalculatedNdx(energy) << ' '
              << direct->CalculatedEdx(energy) << '\n';
    std::cout << "interpolated " << interpolated->CalculatedNdx(energy)
              << ' ' << interpolated->CalculatedEdx(energy) << '\n';
    for (auto const& component : air.GetComponents()) {
      auto const hash = component.GetHash();
      auto const rate = interpolated->CalculatedNdx(energy, hash);
      std::cout << "component " << hash << ' ' << rate;
      if (rate > 0.) {
        auto const loss = interpolated->CalculateStochasticLoss(
            hash, energy, 0.375 * rate);
        auto const cumulative =
            interpolated->CalculateCumulativeCrosssection(
                energy, hash, loss);
        std::cout << ' ' << loss << ' ' << cumulative;
      }
      std::cout << '\n';
    }
  }
  return 0;
}

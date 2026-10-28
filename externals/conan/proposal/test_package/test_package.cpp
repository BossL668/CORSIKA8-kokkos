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
#include "PROPOSAL/propagation_utility/InteractionBuilder.h"

#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

#include <cmath>
#include <cstddef>
#include <iostream>
#include <memory>
#include <vector>

int main() {
  using namespace PROPOSAL;

  Logging::SetGlobalLoglevel(spdlog::level::critical);
  InterpolationSettings::TABLES_PATH = ".";
  InterpolationSettings::UPPER_ENERGY_LIM = 1.e7;
  InterpolationSettings::NODES_DEDX = 32u;
  InterpolationSettings::NODES_DNDX_E = 24u;
  InterpolationSettings::NODES_DNDX_V = 24u;
  InterpolationSettings::NODES_RATE_INTERPOLANT = 32u;

  GammaDef const photon;
  Air const air;
  auto const cuts = std::make_shared<EnergyCutSettings>(0.5, 0.1, false);
  nlohmann::json config{{"parametrization", "KleinNishina"}};
  auto cross = make_compton(photon, air, cuts, true, config);

  auto const energy = 1.e4;
  auto const component_hash = air.GetComponents().front().GetHash();
  auto const rate_before = cross->CalculatedNdx(energy);
  auto const component_rate_before =
      cross->CalculatedNdx(energy, component_hash);
  auto const dedx_before = cross->CalculatedEdx(energy);
  if (!(rate_before > 0.) || !(component_rate_before > 0.) ||
      !(dedx_before >= 0.)) {
    std::cerr << "ordinary PROPOSAL calculation returned invalid values\n";
    return 1;
  }

  auto const sampled_loss_before = cross->CalculateStochasticLoss(
      component_hash, energy, 0.375 * component_rate_before);
  auto const cumulative_before = cross->CalculateCumulativeCrosssection(
      energy, component_hash, sampled_loss_before);

  auto const dndx_tables = cross->ExportDndxInterpolation();
  auto const dedx_tables = cross->ExportDedxInterpolation();
  if (dndx_tables.empty() || dedx_tables.empty()) {
    std::cerr << "native export did not expose interpolated tables\n";
    return 2;
  }
  for (auto const& table : dndx_tables) {
    auto const expected = table.spline.dimensions[0] * table.spline.dimensions[1];
    if (table.spline.values.size() != static_cast<std::size_t>(expected) ||
        table.spline.derivative_axis0.size() != table.spline.values.size() ||
        table.spline.derivative_axis1.size() != table.spline.values.size() ||
        table.spline.mixed_derivative.size() != table.spline.values.size()) {
      std::cerr << "dN/dX export has inconsistent coefficient dimensions\n";
      return 3;
    }
  }
  for (std::size_t index = 0; index < dedx_tables.size(); ++index) {
    if (dedx_tables[index].component_index != index) {
      std::cerr << "dE/dX export lost its stable component ordinal\n";
      return 6;
    }
  }

  std::vector<std::shared_ptr<CrossSectionBase>> cross_sections;
  cross_sections.emplace_back(std::move(cross));
  auto interaction = make_interaction(cross_sections, true, true);
  auto const total_rate = interaction->ExportTotalRateInterpolation();
  if (!total_rate.available || total_rate.spline.values.size() < 2 ||
      total_rate.spline.values.size() !=
          total_rate.spline.node_derivatives.size() ||
      !(total_rate.lower_energy_limit_MeV > 0.)) {
    std::cerr << "native export did not expose the Interaction rate spline\n";
    return 7;
  }

  auto& shared_cross = cross_sections.front();

  auto const rate_after = shared_cross->CalculatedNdx(energy);
  auto const component_rate_after =
      shared_cross->CalculatedNdx(energy, component_hash);
  auto const dedx_after = shared_cross->CalculatedEdx(energy);
  auto const sampled_loss_after = shared_cross->CalculateStochasticLoss(
      component_hash, energy, 0.375 * component_rate_after);
  auto const cumulative_after = shared_cross->CalculateCumulativeCrosssection(
      energy, component_hash, sampled_loss_after);

  if (rate_before != rate_after ||
      component_rate_before != component_rate_after ||
      dedx_before != dedx_after ||
      sampled_loss_before != sampled_loss_after ||
      cumulative_before != cumulative_after) {
    std::cerr << "read-only export changed ordinary PROPOSAL evaluation\n";
    return 4;
  }
  if (!std::isfinite(sampled_loss_after) ||
      !std::isfinite(cumulative_after)) {
    std::cerr << "ordinary stochastic API returned non-finite values\n";
    return 5;
  }

  std::cout.precision(17);
  std::cout << "rate=" << rate_after << '\n'
            << "dedx=" << dedx_after << '\n'
            << "sampled_loss=" << sampled_loss_after << '\n'
            << "dndx_tables=" << dndx_tables.size() << '\n'
            << "dedx_tables=" << dedx_tables.size() << '\n';
  return 0;
}

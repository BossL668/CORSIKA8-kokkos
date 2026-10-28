/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include <PROPOSAL/EnergyCutSettings.h>
#include <PROPOSAL/crosssection/CrossSection.h>
#include <PROPOSAL/crosssection/Factories/ComptonFactory.h>
#include <PROPOSAL/medium/Medium.h>
#include <PROPOSAL/particle/ParticleDef.h>
#include <PROPOSAL/propagation_utility/DisplacementBuilder.h>
#include <PROPOSAL/propagation_utility/InteractionBuilder.h>
#include <PROPOSAL/propagation_utility/PropagationUtilityInterpolant.h>
#include <CubicInterpolation/CubicSplines.h>
#include <nlohmann/json.hpp>

#include <memory>
#include <vector>

// Configuration only compiles and links this program; it does not build tables.
int main() {
  // Virtual calls alone can link against an incompatible old vtable. These
  // volatile addresses require new non-inline symbols from both patched libs,
  // even when the configure probe is optimized and is never executed.
  auto volatile requiredUtilityExport =
      &PROPOSAL::UtilityInterpolant::ExportInterpolation;
  auto volatile requiredCubicExport = &cubic_splines::CubicSplines<double>::Export;
  PROPOSAL::GammaDef const photon;
  PROPOSAL::Air const air;
  auto const cuts = std::make_shared<PROPOSAL::EnergyCutSettings>(0.5, 0.1, false);
  auto cross = PROPOSAL::make_compton(
      photon, air, cuts, true, nlohmann::json{{"parametrization", "KleinNishina"}});
  auto const rates = cross->ExportDndxInterpolation();
  auto const losses = cross->ExportDedxInterpolation();
  auto const validation =
      cross->EvaluateDndxInterpolation(air.GetComponents().front().GetHash(), 1.e4, 0.5);
  std::vector<std::shared_ptr<PROPOSAL::CrossSectionBase>> crossSections;
  crossSections.emplace_back(std::move(cross));
  auto displacement = PROPOSAL::make_displacement(crossSections, true);
  auto interaction = PROPOSAL::make_interaction(crossSections, true, true);
  auto const range = displacement->ExportInterpolation();
  auto const total = interaction->ExportTotalRateInterpolation();
  return requiredUtilityExport == nullptr || requiredCubicExport == nullptr ||
         rates.empty() || losses.empty() || !range.available || !total.available ||
         validation < 0.;
}

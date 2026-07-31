/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include <corsika/gpu/em/tables/ProposalMedium.hpp>

#include <PROPOSAL/medium/Components.h>

#include <stdexcept>
#include <utility>
#include <vector>

namespace corsika::gpu::em::tables {

  PROPOSAL::Medium makeProposalMedium(MediumConfig const& input) {
    auto const config = normalizeMediumConfig(input);
    std::vector<PROPOSAL::Component> components;
    components.reserve(config.components.size());
    for (auto const& component : config.components) {
      components.emplace_back(
          component.name, component.nuclear_charge,
          component.atomic_mass_g_per_mol, component.number_fraction);
    }
    return PROPOSAL::Medium(
        config.name, config.mean_excitation_energy_eV,
        config.density_correction_C, config.density_correction_a,
        config.density_correction_m, config.density_correction_x0,
        config.density_correction_x1, config.density_correction_delta0,
        config.reference_mass_density_g_per_cm3, std::move(components));
  }

  std::vector<MediumComponent> makeRateTableMediumComponents(
      MediumConfig const& input, PROPOSAL::Medium const& medium) {
    auto const config = normalizeMediumConfig(input);
    auto const proposal_components = medium.GetComponents();
    if (proposal_components.size() != config.components.size()) {
      throw std::runtime_error(
          "PROPOSAL medium component count differs from normalized YAML");
    }
    std::vector<MediumComponent> result;
    result.reserve(config.components.size());
    for (std::size_t index = 0; index < config.components.size(); ++index) {
      auto const& configured = config.components[index];
      auto const& proposal_component = proposal_components[index];
      if (proposal_component.GetName() != configured.name ||
          proposal_component.GetNucCharge() != configured.nuclear_charge ||
          proposal_component.GetAtomicNum() !=
              configured.atomic_mass_g_per_mol ||
          proposal_component.GetAtomInMolecule() !=
              configured.number_fraction) {
        throw std::runtime_error(
            "PROPOSAL medium component differs from normalized YAML");
      }
      result.push_back(
          {configured.corsika_pid,
           static_cast<std::uint64_t>(proposal_component.GetHash()),
           configured.number_fraction, configured.name});
    }
    return result;
  }

} // namespace corsika::gpu::em::tables

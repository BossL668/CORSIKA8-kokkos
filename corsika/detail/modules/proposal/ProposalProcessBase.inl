/*
 * (c) Copyright 2020 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include <corsika/media/IMediumModel.hpp>
#include <corsika/media/NuclearComposition.hpp>
#include <corsika/modules/proposal/ProposalProcessBase.hpp>
#include <corsika/framework/core/PhysicalUnits.hpp>
#include <corsika/framework/utility/COMBoost.hpp>
#include <corsika/framework/utility/CorsikaData.hpp>

#include <cstdlib>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <tuple>

namespace corsika::proposal {

  inline bool ProposalProcessBase::canInteract(Code pcode) const {
    return std::find(begin(tracked), end(tracked), pcode) != end(tracked);
  }

  inline HEPEnergyType ProposalProcessBase::getOptimizedEmCut(Code code) const {
    // get energy above which energy losses need to be considered
    auto const production_threshold = get_energy_production_threshold(code);
    return optimized_proposal_energy_cut(production_threshold);
  }

  template <typename TEnvironment>
  inline ProposalProcessBase::ProposalProcessBase(TEnvironment const& _env) {
    _env.getUniverse()->walk([&](auto& vtn) {
      if (vtn.hasModelProperties()) {
        const auto& prop = vtn.getModelProperties();
        const auto& medium = mediumData(prop.getMedium());

        auto comp_vec = std::vector<PROPOSAL::Component>();
        const auto& comp = prop.getNuclearComposition();
        auto frac_iter = comp.getFractions().cbegin();
        for (auto const pcode : comp.getComponents()) {
          comp_vec.emplace_back(std::string(get_name(pcode)), get_nucleus_Z(pcode),
                                get_nucleus_A(pcode), *frac_iter);
          ++frac_iter;
        }

        media[comp.getHash()] = PROPOSAL::Medium(
            medium.getName(), medium.getIeff(), -medium.getCbar(), medium.getAA(),
            medium.getSK(), medium.getX0(), medium.getX1(), medium.getDlt0(),
            medium.getCorrectedDensity(), comp_vec);
      }
    });

    //! If corsika data exist store interpolation tables to the corresponding
    //! path, otherwise interpolation tables would only stored in main memory if
    //! no explicit intrpolation def is specified.
    PROPOSAL::InterpolationSettings::TABLES_PATH = corsika_data("PROPOSAL").c_str();

    //! Initialize EnergyCutSettings
    for (auto const particle_code : tracked) {
      if (particle_code == Code::Photon) {
        // no EnergyCut for photon, only-stochastic propagation
        continue;
      }
      // use optimized emcut for which tables should be available
      auto optimized_ecut = getOptimizedEmCut(particle_code);
      CORSIKA_LOG_INFO("PROPOSAL: Use tables with EnergyCut of {} for particle {}.",
                       optimized_ecut, particle_code);
      proposal_energycutsettings[particle_code] = optimized_ecut;
    }

    //! Initialize PROPOSAL tables for all media and all particles
    for (auto const& medium : media) {
      for (auto const particle_code : tracked) {
        buildTables(medium.second, particle_code,
                    proposal_energycutsettings[particle_code]);
      }
    }
  }

  void ProposalProcessBase::buildTables(PROPOSAL::Medium medium, Code particle_code,
                                        HEPEnergyType emcut) {
    CORSIKA_LOG_DEBUG("PROPOSAL: Initialize tables for particle {} in {}", particle_code,
                      medium.GetName());
    cross[particle_code](medium, emcut);
  }

  inline size_t ProposalProcessBase::hash::operator()(const calc_key_t& p) const
      noexcept {
    return p.first ^ std::hash<Code>{}(p.second);
  }

} // namespace corsika::proposal

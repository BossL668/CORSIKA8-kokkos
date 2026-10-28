/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include <corsika/kokkos/em/proposal/NativeRequirements.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#include <corsika/framework/core/ParticleProperties.hpp>
#include <corsika/framework/core/PhysicalUnits.hpp>
#include <corsika/kokkos/em/tables/ContinuousPhysicsQueries.hpp>

namespace corsika::kokkos {

  void validate_proposal_native_requirements(ProposalNativeTableSet const& nativeTables,
                                             PhysicsRequirements const& requirements) {
    using namespace corsika::units::si;
    auto const domainEpsilon = 64. * std::numeric_limits<double>::epsilon();
    for (auto const pdg : {22, 11, -11, 13, -13}) {
      auto const total = std::find_if(
          nativeTables.totalRateColumns_.begin(), nativeTables.totalRateColumns_.end(),
          [pdg](auto const& column) { return column.pdgId_ == pdg; });
      if (total == nativeTables.totalRateColumns_.end()) {
        throw std::runtime_error(
            "PROPOSAL native table has no total-rate spline for a routed particle");
      }
      double upperEnergyMeV = total->spline_.axis_.high_;
      double lowerEnergyMeV = total->spline_.axis_.low_;
      for (auto const& column : nativeTables.dndxColumns_) {
        if (column.pdgId_ == pdg && column.rateModel_ == NativeRateModel::BicubicSpline) {
          upperEnergyMeV = std::min(upperEnergyMeV, column.spline_.energyAxis_.high_);
        }
      }
      if (pdg != 22) {
        auto const utility = std::find_if(
            nativeTables.utilityColumns_.begin(), nativeTables.utilityColumns_.end(),
            [pdg](auto const& column) { return column.pdgId_ == pdg; });
        if (utility == nativeTables.utilityColumns_.end()) {
          throw std::runtime_error(
              "PROPOSAL native table has no continuous range spline for a routed charged "
              "lepton");
        }
        lowerEnergyMeV = std::max(lowerEnergyMeV, std::max(utility->lowerEnergyLimitMeV_,
                                                           utility->spline_.axis_.low_));
        upperEnergyMeV = std::min(upperEnergyMeV, utility->spline_.axis_.high_);
        for (auto const& column : nativeTables.dedxColumns_) {
          if (column.pdgId_ == pdg) {
            upperEnergyMeV = std::min(upperEnergyMeV, column.spline_.axis_.high_);
          }
        }
      }
      auto const code = convert_from_PDG(static_cast<PDGCode>(pdg));
      auto const minimumTransportEnergyMeV =
          pdg == 22 ? requirements.emTransportCutMeV_
                    : get_mass(code) / 1_MeV +
                          ContinuousCutSafetyFactor *
                              (std::abs(pdg) == 13 ? requirements.muonTransportCutMeV_
                                                   : requirements.emTransportCutMeV_);
      auto const lowerScale =
          std::max({1., std::abs(lowerEnergyMeV), std::abs(minimumTransportEnergyMeV)});
      if (minimumTransportEnergyMeV + domainEpsilon * lowerScale < lowerEnergyMeV) {
        throw std::runtime_error(
            "PROPOSAL native interpolation domain does not cover the configured "
            "transport cut");
      }
      auto const upperScale = std::max({1., std::abs(upperEnergyMeV),
                                        std::abs(requirements.maximumPrimaryEnergyMeV_)});
      if (requirements.maximumPrimaryEnergyMeV_ >
          upperEnergyMeV + domainEpsilon * upperScale) {
        throw std::runtime_error(
            "configured maximum primary energy exceeds the common PROPOSAL native "
            "interpolation domain");
      }
    }
  }

} // namespace corsika::kokkos

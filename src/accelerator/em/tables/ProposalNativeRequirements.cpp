/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include <corsika/accelerator/em/ProposalNativeRequirements.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#include <corsika/framework/core/ParticleProperties.hpp>
#include <corsika/framework/core/PhysicalUnits.hpp>
#include <corsika/accelerator/em/common/tables/PhysicsConstants.hpp>

namespace corsika::accelerator::em {

  void validateProposalNativeRequirements(
      gpu::em::tables::ProposalNativeTableSet const& native_tables,
      AcceleratedPhysicsRequirements const& requirements) {
    using namespace corsika::units::si;
    using namespace corsika::gpu::em::tables;
    auto const domain_epsilon = 64. * std::numeric_limits<double>::epsilon();
    for (auto const pdg : {22, 11, -11, 13, -13}) {
      auto const total = std::find_if(
          native_tables.total_rate_columns.begin(),
          native_tables.total_rate_columns.end(),
          [pdg](auto const& column) { return column.pdg_id == pdg; });
      if (total == native_tables.total_rate_columns.end()) {
        throw std::runtime_error(
            "PROPOSAL native table has no total-rate spline for a routed particle");
      }
      double upper_energy_MeV = total->spline.axis.high;
      double lower_energy_MeV = total->spline.axis.low;
      for (auto const& column : native_tables.dndx_columns) {
        if (column.pdg_id == pdg &&
            column.rate_model == NativeRateModel::BicubicSpline) {
          upper_energy_MeV =
              std::min(upper_energy_MeV, column.spline.energy_axis.high);
        }
      }
      if (pdg != 22) {
        auto const utility = std::find_if(
            native_tables.utility_columns.begin(),
            native_tables.utility_columns.end(),
            [pdg](auto const& column) { return column.pdg_id == pdg; });
        if (utility == native_tables.utility_columns.end()) {
          throw std::runtime_error(
              "PROPOSAL native table has no continuous range spline for a routed charged lepton");
        }
        lower_energy_MeV = std::max(
            lower_energy_MeV,
            std::max(utility->lower_energy_limit_MeV,
                     utility->spline.axis.low));
        upper_energy_MeV =
            std::min(upper_energy_MeV, utility->spline.axis.high);
        for (auto const& column : native_tables.dedx_columns) {
          if (column.pdg_id == pdg) {
            upper_energy_MeV =
                std::min(upper_energy_MeV, column.spline.axis.high);
          }
        }
      }
      auto const code = convert_from_PDG(static_cast<PDGCode>(pdg));
      auto const minimum_transport_energy_MeV =
          pdg == 22
              ? requirements.em_transport_cut_MeV
              : get_mass(code) / 1_MeV +
                    gpu::em::tables::ContinuousCutSafetyFactor *
                    (std::abs(pdg) == 13
                         ? requirements.muon_transport_cut_MeV
                         : requirements.em_transport_cut_MeV);
      auto const lower_scale = std::max(
          {1., std::abs(lower_energy_MeV),
           std::abs(minimum_transport_energy_MeV)});
      if (minimum_transport_energy_MeV + domain_epsilon * lower_scale <
          lower_energy_MeV) {
        throw std::runtime_error(
            "PROPOSAL native interpolation domain does not cover the configured transport cut");
      }
      auto const upper_scale = std::max(
          {1., std::abs(upper_energy_MeV),
           std::abs(requirements.maximum_primary_energy_MeV)});
      if (requirements.maximum_primary_energy_MeV >
          upper_energy_MeV + domain_epsilon * upper_scale) {
        throw std::runtime_error(
            "configured maximum primary energy exceeds the common PROPOSAL native interpolation domain");
      }
    }
  }

} // namespace corsika::accelerator::em

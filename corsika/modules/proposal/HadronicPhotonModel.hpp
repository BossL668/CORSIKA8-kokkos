/*
 * (c) Copyright 2022 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/framework/core/ParticleProperties.hpp>
#include <corsika/framework/geometry/FourVector.hpp>
#include <corsika/framework/process/ProcessReturn.hpp>

#include <cstdint>
#include <vector>

namespace corsika::proposal {

  struct HadronicPhotonEnergyLedgerStatistics {
    std::uint64_t interactions{};
    std::uint64_t low_energy_interactions{};
    std::uint64_t high_energy_interactions{};
    double weighted_projectile_total_energy_GeV{};
    double weighted_target_total_energy_GeV{};
    double weighted_secondary_total_energy_GeV{};
    double weighted_energy_residual_GeV{};
    double weighted_absolute_energy_residual_GeV{};
    double maximum_absolute_residual_GeV{};
    double maximum_target_total_energy_GeV{};
    double weighted_momentum_residual_norm_GeV{};
    double maximum_momentum_residual_norm_GeV{};
    std::uint64_t negative_energy_residual_interactions{};
    std::uint64_t spacelike_residual_interactions{};
    double maximum_negative_energy_residual_GeV{};
    int maximum_negative_projectile_pdg{};
    int maximum_negative_target_pdg{};
    double maximum_negative_projectile_total_energy_GeV{};
    double maximum_negative_target_total_energy_GeV{};
    double maximum_negative_secondary_total_energy_GeV{};
    std::uint64_t maximum_negative_secondary_count{};
    double maximum_negative_baryon_number_residual{};
    double maximum_negative_charge_number_residual{};
    std::vector<int> maximum_negative_secondary_pdgs{};
    double maximum_spacelike_excess_GeV{};
    double minimum_residual_mass_squared_GeV2{};
    double maximum_timelike_residual_mass_GeV{};
    double maximum_positive_residual_beta{};
  };

  //! Implements the production of secondary hadrons for the hadronic interaction of real
  //! and virtual photons for PROPOSAL. The model distinguishes between resonance
  //! production (LE) and continuum (HE). HE production replaces the photon with a rho0.
  //! External models are needed that implement the hadronic particle production via the
  //! doInteraction(TSecondaries& view, Code const projectile, Code const target,
  //! FourMomentum const& projectileP4, FourMomentum const& targetP4) routine. The
  //! threshold between LE and HE interactions is defined in lab energy.
  //! @tparam THadronicModel

  template <class THadronicLEModel, class THadronicHEModel>
  class HadronicPhotonModel {
  public:
    HadronicPhotonModel(THadronicLEModel&, THadronicHEModel&, HEPEnergyType const&);
    //!
    //! Calculate produce the hadronic secondaries in a hadronic photon interaction and
    //! store them on the particle stack.
    //!
    template <typename TSecondaryView>
    ProcessReturn doHadronicPhotonInteraction(TSecondaryView&, CoordinateSystemPtr const&,
                                              FourMomentum const&, Code const&);

    HadronicPhotonEnergyLedgerStatistics const&
    energyLedgerStatistics() const noexcept {
      return energy_ledger_statistics_;
    }

  private:
    inline static auto logger_{get_logger("corsika_proposal_HadronicPhotonModel")};
    THadronicLEModel& leHadronicInteraction_;
    THadronicHEModel& heHadronicInteraction_;
    //! threshold for high energy hadronic interaction model. Lab. energy per nucleon
    HEPEnergyType const heHadronicModelThresholdLabNN_;
    HadronicPhotonEnergyLedgerStatistics energy_ledger_statistics_{};
  };
} // namespace corsika::proposal

#include <corsika/detail/modules/proposal/HadronicPhotonModel.inl>

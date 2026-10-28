/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include "SessionFixture.hpp"
#include <PROPOSAL/Constants.h>
#include <PROPOSAL/PROPOSAL.h>
#include <PROPOSAL/crosssection/parametrization/Bremsstrahlung.h>
#include <PROPOSAL/crosssection/parametrization/PhotoPairProduction.h>
#include <PROPOSAL/propagation_utility/DisplacementBuilder.h>
#include <PROPOSAL/propagation_utility/InteractionBuilder.h>
#include <corsika/modules/proposal/ProposalProcessBase.hpp>

namespace session_test {
  SmallTableSettings::SmallTableSettings(std::filesystem::path const& directory)
      : path_(PROPOSAL::InterpolationSettings::TABLES_PATH)
      , upper_(PROPOSAL::InterpolationSettings::UPPER_ENERGY_LIM)
      , dndxEnergy_(PROPOSAL::InterpolationSettings::NODES_DNDX_E)
      , dndxLoss_(PROPOSAL::InterpolationSettings::NODES_DNDX_V)
      , dedx_(PROPOSAL::InterpolationSettings::NODES_DEDX)
      , utility_(PROPOSAL::InterpolationSettings::NODES_UTILITY)
      , rate_(PROPOSAL::InterpolationSettings::NODES_RATE_INTERPOLANT) {
    std::filesystem::create_directories(directory);
    PROPOSAL::InterpolationSettings::TABLES_PATH = directory.string();
    PROPOSAL::InterpolationSettings::UPPER_ENERGY_LIM = 1.e8;
    PROPOSAL::InterpolationSettings::NODES_DNDX_E = 32;
    PROPOSAL::InterpolationSettings::NODES_DNDX_V = 32;
    PROPOSAL::InterpolationSettings::NODES_DEDX = 64;
    PROPOSAL::InterpolationSettings::NODES_UTILITY = 64;
    PROPOSAL::InterpolationSettings::NODES_RATE_INTERPOLANT = 64;
  }

  SmallTableSettings::~SmallTableSettings() {
    PROPOSAL::InterpolationSettings::TABLES_PATH = path_;
    PROPOSAL::InterpolationSettings::UPPER_ENERGY_LIM = upper_;
    PROPOSAL::InterpolationSettings::NODES_DNDX_E = dndxEnergy_;
    PROPOSAL::InterpolationSettings::NODES_DNDX_V = dndxLoss_;
    PROPOSAL::InterpolationSettings::NODES_DEDX = dedx_;
    PROPOSAL::InterpolationSettings::NODES_UTILITY = utility_;
    PROPOSAL::InterpolationSettings::NODES_RATE_INTERPOLANT = rate_;
  }

  Calculator::~Calculator() = default;
  Fixture::~Fixture() = default;

  Calculator::Calculator(Code code, PROPOSAL::Medium& medium) {
    auto const crosses = corsika::proposal::cross.at(code)(medium, .4_MeV);
    interaction_ = PROPOSAL::make_interaction(crosses, true, true);
    auto const& particle = corsika::proposal::particle.at(code);
    if (code == Code::Photon) {
      photon_ = std::make_unique<PROPOSAL::crosssection::PhotoPairLPM>(
          particle, medium, PROPOSAL::crosssection::PhotoPairKochMotz());
    } else {
      displacement_ = PROPOSAL::make_displacement(crosses, true);
      brems_ = std::make_unique<PROPOSAL::crosssection::BremsLPM>(
          particle, medium, PROPOSAL::crosssection::BremsElectronScreening());
    }
  }

  Fixture::Fixture(std::filesystem::path const& directory)
      : settings_(directory / "tables") {
    for (Code code :
         {Code::Photon, Code::Electron, Code::Positron, Code::MuMinus, Code::MuPlus}) {
      calculators_.push_back(std::make_unique<Calculator>(code, medium_));
      auto const& owner = *calculators_.back();
      proposal::NativeInteractionCalculatorView::photon_pair_reference_type photon;
      proposal::NativeInteractionCalculatorView::brems_reference_type brems;
      if (owner.photon_) photon = std::cref(*owner.photon_);
      if (owner.brems_) brems = std::cref(*owner.brems_);
      interactions_.emplace_back(code, medium_.GetHash(), std::cref(medium_),
                                 std::cref(*owner.interaction_), photon, brems,
                                 code == Code::Photon ? 0_MeV : .4_MeV,
                                 proposal::particle.at(code).mass);
      if (owner.displacement_)
        continuous_.emplace_back(code, medium_.GetHash(), std::cref(medium_),
                                 std::cref(*owner.displacement_), .4_MeV,
                                 proposal::particle.at(code).mass);
    }
    environment_ = std::make_unique<environment_type>();
    coordinates_ = environment_->getCoordinateSystem();
    auto world =
        environment_type::createNode<Sphere>(Point{coordinates_, 0_m, 0_m, 0_m}, 100_km);
    node_ = world.get();
    using medium_type = UniformRefractiveIndex<MediumPropertyModel<
        UniformMagneticField<HomogeneousMedium<environment_interface_type>>>>;
    node_->setModelProperties<medium_type>(
        1.000312, Medium::AirDry1Atm, MagneticFieldVector(coordinates_, {0_T, 0_T, 0_T}),
        1.225_kg / cube(1_m),
        NuclearComposition({Code::Nitrogen, Code::Oxygen, Code::Argon}, {.78, .21, .01}));
    environment_->getUniverse()->addChild(std::move(world));
    snapshot_ = std::make_unique<cp::EnvironmentSnapshot<environment_interface_type>>(
        *environment_->getUniverse(), coordinates_);
    geometry_.observation_.pointM_[2] = -100.;
    geometry_.observation_.normal_[2] = 1.;
    config_.minBatchSize_ = 8;
    config_.residentBatchLimit_ = 32;
    config_.memoryFraction_ = .10;
    config_.emTransportCutMeV_ = requirements_.emTransportCutMeV_;
    config_.muonTransportCutMeV_ = requirements_.muonTransportCutMeV_;
    config_.randomSeed_ = 54217;
    config_.showerId_ = 23;
    config_.diagnosticInteractionRecords_ = true;
  }
} // namespace session_test

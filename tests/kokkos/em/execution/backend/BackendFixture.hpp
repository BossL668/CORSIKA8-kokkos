/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once
#include <corsika/kokkos/em/execution/EmBackend.hpp>
#include <corsika/kokkos/em/proposal/ProposalNativeAux.hpp>
#include <corsika/kokkos/environment/EnvironmentSnapshotData.hpp>
#include <corsika/kokkos/environment/TransportGeometryConfig.hpp>
#include <tests/common/SetupTestEnvironment.hpp>
#include <corsika/media/UniformRefractiveIndex.hpp>
#include <tests/kokkos/em/models/AuxiliaryModelFixture.hpp>
#include <tests/kokkos/em/tables/NativeTableFixture.hpp>
#include <filesystem>
#include <stdexcept>
#include <string>

namespace backend_test {
  namespace cp = corsika::kokkos;
  using namespace corsika;
  using model_interface_type =
      IRefractiveIndexModel<IMediumPropertyModel<IMagneticFieldModel<IMediumModel>>>;
  using environment_type = Environment<model_interface_type>;
  inline void require(bool value, std::string const& label) {
    if (!value) throw std::runtime_error(label);
  }
  template <typename TFunction>
  bool throws(TFunction&& action) {
    try {
      action();
    } catch (std::exception const&) { return true; }
    return false;
  }
  struct Fixture {
    std::unique_ptr<environment_type> environment_;
    CoordinateSystemPtr coordinates_;
    environment_type::BaseNodeType* node_{};
    std::unique_ptr<cp::EnvironmentSnapshot<model_interface_type>> snapshot_;
    cp::TransportGeometryConfig geometry_{};
    cp::ProposalNativeTableSet tables_;
    cp::ProposalNativeAuxData auxiliary_;
    cp::BackendConfig config_;
    explicit Fixture(std::filesystem::path const& directory) {
      environment_ = std::make_unique<environment_type>();
      coordinates_ = environment_->getCoordinateSystem();
      auto world = environment_type::createNode<Sphere>(
          Point(coordinates_, 0_m, 0_m, 0_m), 100_km);
      node_ = world.get();
      using medium_type = UniformRefractiveIndex<MediumPropertyModel<
          UniformMagneticField<HomogeneousMedium<model_interface_type>>>>;
      // The native atmosphere composition used by C8's dry-air model.
      node_->setModelProperties<medium_type>(
          1.000312, Medium::AirDry1Atm,
          MagneticFieldVector(coordinates_, {0_T, 0_T, 0_T}), 1.225_kg / cube(1_m),
          NuclearComposition({Code::Nitrogen, Code::Oxygen, Code::Argon},
                             {.78, .21, .01}));
      environment_->getUniverse()->addChild(std::move(world));
      snapshot_ = std::make_unique<cp::EnvironmentSnapshot<model_interface_type>>(
          *environment_->getUniverse(), coordinates_);
      geometry_.observation_.pointM_[2] = -100.;
      geometry_.observation_.normal_[2] = 1.;
      tables_ = cp::test::make_native_tables((directory / "tables").string());
      auto const real = cp::test::make_auxiliary_dataset();
      require(!real.snapshots_.empty(), "real PROPOSAL auxiliary snapshots");
      auto const& air = real.snapshots_.front();
      auxiliary_.photonPairLpm_ = air.photon_;
      auxiliary_.bremsLpm_ = air.electron_;
      auxiliary_.electronMoliere_ = air.electronMoliere_;
      auxiliary_.muonMoliere_ = air.muonMoliere_;
      auxiliary_.hasMuonMoliere_ = 1;
      auxiliary_.cacheFile_ = directory / "real-air-aux";
      config_.minBatchSize_ = 8;
      config_.residentBatchLimit_ = 32;
      config_.memoryFraction_ = .10;
      config_.emTransportCutMeV_ = .4;
      config_.muonTransportCutMeV_ = .4;
      config_.randomSeed_ = 54217;
      config_.showerId_ = 23;
      config_.diagnosticInteractionRecords_ = true;
    }
    cp::ParticleState particle(int pid, double energy, std::uint64_t history) const {
      cp::ParticleState p{};
      p.pid_ = pid;
      p.energyGeV_ = energy;
      p.environmentNode_ = snapshot_->getTopology().getNodeIndex(*(node_));
      p.mediumId_ = static_cast<int>(Medium::AirDry1Atm);
      p.direction_[2] = -1.;
      p.weight_ = 1.;
      p.historyId_ = history;
      p.parentHistoryId_ = 1;
      p.generation_ = 1;
      p.stepId_ = 7;
      return p;
    }
    void initialize(cp::EmBackend& backend) const {
      backend.initialize(cp::bind_environment_data(*snapshot_), geometry_, tables_,
                         auxiliary_, config_);
      require(backend.hasProposalTable(), "backend owns real native table");
    }
  };
} // namespace backend_test

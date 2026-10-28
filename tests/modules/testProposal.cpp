/*
 * (c) Copyright 2022 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */
#include <corsika/modules/PROPOSAL.hpp>
#include <corsika/framework/random/RNGManager.hpp>

#include <SetupTestEnvironment.hpp>
#include <SetupTestStack.hpp>
#include <catch2/catch_all.hpp>
#include <tuple>
#include "corsika/framework/core/PhysicalUnits.hpp"

using namespace corsika;
using namespace corsika::proposal;

using DummyEnvironmentInterface = IMediumPropertyModel<IMagneticFieldModel<IMediumModel>>;
using DummyEnvironment = Environment<DummyEnvironmentInterface>;

#include <corsika/media/Environment.hpp>
#include <corsika/media/HomogeneousMedium.hpp>
#include <corsika/media/NuclearComposition.hpp>
#include <corsika/media/UniformMagneticField.hpp>

class DummyHadronicModel {
public:
  DummyHadronicModel(HEPEnergyType thr)
      : threshold_(thr){};

  template <typename TSecondaryView>
  void doInteraction(TSecondaryView& view, Code const, Code const,
                     FourMomentum const& projectileP4, FourMomentum const& targetP4) {
    auto const E = projectileP4.getTimeLikeComponent();
    // add 5 pions
    auto const& csPrime = view.getProjectile().getMomentum().getCoordinateSystem();
    [[maybe_unused]] auto const sqs = (projectileP4 + targetP4).getNorm();
    for (int i = 0; i < 5; ++i) {
      view.addSecondary(
          std::make_tuple(Code::PiPlus, E / 5,
                          MomentumVector(csPrime, {0_GeV, 0_GeV, 0_GeV}).normalized()));
    }
  }
  bool constexpr isValid(Code const, Code const, HEPEnergyType const sqrsNN) const {
    return (sqrsNN >= threshold_);
  };

private:
  HEPEnergyType threshold_;
};

TEST_CASE("ProposalInterface", "modules") {

  logging::set_level(logging::level::info);

  // the test environment
  auto [env, csPtr, nodePtr] = setup::testing::setup_environment(Code::Oxygen);
  auto const& cs = *csPtr;

  auto [stackPtr, viewPtr] = setup::testing::setup_stack(
      Code::Electron, 10_GeV, (DummyEnvironment::BaseNodeType* const)nodePtr, cs);
  test::StackView& view = *viewPtr;

  RNGManager<>::getInstance().registerRandomStream("proposal");

  SECTION("InteractionInterface - hadronic photon model threshold") {
    DummyHadronicModel hadModelLE(100_MeV);
    DummyHadronicModel hadModelHE(10_GeV);
    HEPEnergyType heThresholdLab1 = 12_GeV;
    CHECK_THROWS(corsika::proposal::InteractionModel(*env, hadModelLE, hadModelHE,
                                                     heThresholdLab1));
  }

  DummyHadronicModel hadModelLE(100_MeV);
  DummyHadronicModel hadModelHE(10_GeV);
  HEPEnergyType heThresholdLab = 80_GeV;
  corsika::proposal::InteractionModel emModel(*env, hadModelLE, hadModelHE,
                                              heThresholdLab);

  SECTION("InteractionInterface - cross section") {
    auto& stack = *stackPtr;
    auto particle = stack.first();
    FourMomentum P4(
        100_MeV,
        {cs, {sqrt(static_pow<2>(100_MeV) - static_pow<2>(Proton::mass)), 0_eV, 0_eV}});
    CHECK(emModel.getCrossSection(particle, Code::Proton, P4) == 0_mb);

    FourMomentum eleP4(
        100_MeV,
        {cs, {sqrt(static_pow<2>(100_MeV) - static_pow<2>(Electron::mass)), 0_eV, 0_eV}});
    CHECK(emModel.getCrossSection(particle, Code::Electron, eleP4) > 0_mb);
  }

  SECTION("InteractionInterface - LE hadronic photon interaction") {
    auto& stack = *stackPtr;
    // auto particle = stack.first();
    FourMomentum P4(10_GeV, {cs, {10_GeV, 0_eV, 0_eV}});
    // finish successfully
    CHECK(emModel.doHadronicPhotonInteraction(view, cs, P4, Code::Oxygen) ==
          ProcessReturn::Ok);
    CHECK(stack.getEntries() == 6);
    CORSIKA_LOG_INFO("Number of particles produced in hadronic photon interaction: {}",
                     stack.getEntries() - 1);
  }

  SECTION("InteractionInterface - HE hadronic photon interaction") {
    auto& stack = *stackPtr;
    // auto particle = stack.first();
    FourMomentum P4(100_TeV, {cs, {100_TeV, 0_eV, 0_eV}});
    // finish successfully
    CHECK(emModel.doHadronicPhotonInteraction(view, cs, P4, Code::Oxygen) ==
          ProcessReturn::Ok);
    CHECK(stack.getEntries() > 1);
    CORSIKA_LOG_INFO("Number of particles produced in hadronic photon interaction: {}",
                     stack.getEntries() - 1);
  }
}
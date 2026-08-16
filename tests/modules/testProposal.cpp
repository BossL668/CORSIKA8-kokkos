/*
 * (c) Copyright 2022 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */
#include <corsika/modules/PROPOSAL.hpp>
#include <corsika/modules/proposal/ThresholdPhotoproductionModel.hpp>
#include <corsika/framework/random/RNGManager.hpp>

#include <SetupTestEnvironment.hpp>
#include <SetupTestStack.hpp>
#include <catch2/catch_all.hpp>
#include <cmath>
#include <numeric>
#include <optional>
#include <tuple>
#include <vector>
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
  DummyHadronicModel(
      HEPEnergyType thr = 0_GeV,
      std::optional<Code> rejectedTarget = std::nullopt,
      std::optional<Code> secondRejectedTarget = std::nullopt)
      : threshold_(thr)
      , rejectedTarget_(rejectedTarget)
      , secondRejectedTarget_(secondRejectedTarget){};

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
  bool constexpr isValid(
      Code const, Code const target,
      HEPEnergyType const sqrsNN) const {
    return sqrsNN >= threshold_ &&
           (!rejectedTarget_ || target != *rejectedTarget_) &&
           (!secondRejectedTarget_ || target != *secondRejectedTarget_);
  };

private:
  HEPEnergyType threshold_;
  std::optional<Code> rejectedTarget_;
  std::optional<Code> secondRejectedTarget_;
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

  SECTION("InteractionInterface - split rate selection and specified final state") {
    auto& stack = *stackPtr;
    auto particle = stack.first();

    auto const rates = emModel.getRateTable(particle, Code::Electron);
    REQUIRE_FALSE(rates.entries().empty());
    REQUIRE(rates.entries().size() == rates.nativeRates().size());
    CHECK(rates.energyMeV() ==
          Catch::Approx(particle.getEnergy() / 1_MeV));
    CHECK(rates.totalRate() > 0.);

    for (std::size_t i = 0; i < rates.entries().size(); ++i) {
      auto const& entry = rates.entries()[i];
      auto const& native = rates.nativeRates()[i];
      REQUIRE(native.crosssection);
      CHECK(entry.type == native.crosssection->GetInteractionType());
      CHECK(entry.component_hash == native.comp_hash);
      CHECK(entry.rate == native.rate);
      CHECK(std::isfinite(entry.rate));
      CHECK(entry.rate >= 0.);
    }

    double constexpr selection_uniform = 0.375;
    auto sampled_rate = selection_uniform * rates.totalRate();
    PROPOSAL::InteractionType expected_type =
        PROPOSAL::InteractionType::Undefined;
    std::size_t expected_component_hash = 0;
    double expected_v = 0.;
    for (auto const& native : rates.nativeRates()) {
      sampled_rate -= native.rate;
      if (sampled_rate < 0.) {
        expected_type = native.crosssection->GetInteractionType();
        expected_component_hash = native.comp_hash;
        expected_v = native.crosssection->CalculateStochasticLoss(
            native.comp_hash, rates.energyMeV(), -sampled_rate);
        break;
      }
    }
    REQUIRE(expected_type != PROPOSAL::InteractionType::Undefined);

    ProposalRandomKey const key{17, 23, 42, 3, 11, 5};
    auto const record = emModel.sampleInteraction(
        particle, Code::Electron, rates, selection_uniform, key);
    CHECK(record.type == expected_type);
    CHECK(record.interaction_hash == rates.interactionHash());
    CHECK(record.component_hash == expected_component_hash);
    CHECK(record.v_loss ==
          Catch::Approx(expected_v).epsilon(1e-13));
    CHECK(record.selection_uniform == selection_uniform);
    REQUIRE(record.random_key);
    CHECK(record.random_key->history_id == 42);
    CHECK(record.random_key->step_id == 3);
    CHECK(record.context.projectile_id == Code::Electron);
    CHECK(record.context.medium_hash ==
          particle.getNode()
              ->getModelProperties()
              .getNuclearComposition()
              .getHash());
    CHECK(record.context.projectile_energy_MeV ==
          Catch::Approx(particle.getEnergy() / 1_MeV));
    CHECK(record.context.position_cm ==
          std::array<double, 3>{0., 0., 0.});
    CHECK(record.context.direction ==
          std::array<double, 3>{1., 0., 0.});
    CHECK(record.context.time_s == 0.);

    auto const random_count =
        emModel.requiredFinalStateRandomNumbers(record);
    std::vector<double> final_state_randoms(random_count);
    for (std::size_t i = 0; i < final_state_randoms.size(); ++i) {
      final_state_randoms[i] =
          (static_cast<double>(i) + 0.5) /
          (static_cast<double>(final_state_randoms.size()) + 1.);
    }

    auto const first =
        emModel.generateFinalState(record, final_state_randoms);
    auto const second =
        emModel.generateFinalState(record, final_state_randoms);
    REQUIRE_FALSE(first.secondaries.empty());
    REQUIRE(first.secondaries.size() == second.secondaries.size());
    for (std::size_t i = 0; i < first.secondaries.size(); ++i) {
      CHECK(first.secondaries[i] == second.secondaries[i]);
      CHECK(std::isfinite(first.secondaries[i].energy));
      CHECK(first.secondaries[i].energy >= 0.);
    }
    if (record.type != PROPOSAL::InteractionType::Ioniz) {
      CHECK(first.target.GetHash() == record.component_hash);
      CHECK(second.target.GetHash() == record.component_hash);
    }

    auto const secondaries_before = view.getSize();
    CHECK(emModel.doSpecifiedInteraction(
              view, record, final_state_randoms) ==
          ProcessReturn::Ok);
    CHECK(view.getSize() > secondaries_before);

    auto wrong_random_count = final_state_randoms;
    if (wrong_random_count.empty()) {
      wrong_random_count.push_back(0.5);
    } else {
      wrong_random_count.pop_back();
    }
    CHECK_THROWS(emModel.generateFinalState(record, wrong_random_count));
    CHECK_THROWS(emModel.sampleInteraction(
        particle, Code::Electron, rates, 1.));

    auto undefined_record = record;
    undefined_record.type = PROPOSAL::InteractionType::Undefined;
    CHECK_THROWS(
        emModel.requiredFinalStateRandomNumbers(undefined_record));
    CHECK_THROWS(
        emModel.generateFinalState(undefined_record, {}));

    auto wrong_calculator_record = record;
    ++wrong_calculator_record.interaction_hash;
    CHECK_THROWS(emModel.requiredFinalStateRandomNumbers(
        wrong_calculator_record));
  }

  SECTION("InteractionInterface - split doInteraction integration") {
    auto& stack = *stackPtr;

    auto const projectile = view.getProjectile();
    FourMomentum const projectileP4(
        projectile.getEnergy(), projectile.getMomentum());
    CHECK(emModel.doInteraction(
              view, Code::Electron, projectileP4) ==
          ProcessReturn::Ok);
    CHECK(stack.getEntries() > 1);
  }

  SECTION("InteractionInterface - LE hadronic photon interaction") {
    auto& stack = *stackPtr;
    // auto particle = stack.first();
    FourMomentum P4(10_GeV, {cs, {10_GeV, 0_eV, 0_eV}});
    // finish successfully
    CHECK(emModel.doHadronicPhotonInteraction(view, cs, P4, Code::Oxygen) ==
          ProcessReturn::Ok);
    CHECK(stack.getEntries() == 6);
    auto const& photonLedger =
        emModel.energyLedgerStatistics();
    CHECK(photonLedger.interactions == 1);
    CHECK(photonLedger.low_energy_interactions == 1);
    CHECK(photonLedger.high_energy_interactions == 0);
    CHECK(photonLedger.weighted_target_total_energy_GeV > 0.);
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

  SECTION("InteractionInterface - HE hadronic photon model fallback") {
    auto& stack = *stackPtr;
    DummyHadronicModel preferred(10_GeV, Code::Argon);
    LazyHadronicInteractionModel<DummyHadronicModel> fallback;
    HadronicInteractionModelFallback composite{
        preferred, fallback};
    CHECK_FALSE(fallback.initialized());
    CHECK(composite.isValid(
        Code::Rho0, Code::Oxygen, 100_GeV));
    CHECK_FALSE(fallback.initialized());
    CHECK(composite.isValid(
        Code::Rho0, Code::Argon, 100_GeV));
    CHECK(fallback.initialized());
    corsika::proposal::Interaction fallbackModel(
        *env, hadModelLE, composite, heThresholdLab);
    FourMomentum P4(100_TeV, {cs, {100_TeV, 0_eV, 0_eV}});
    CHECK(fallbackModel.doHadronicPhotonInteraction(
              view, cs, P4, Code::Argon) ==
          ProcessReturn::Ok);
    CHECK(stack.getEntries() == 6);
    CHECK(composite.statistics().preferred_interactions == 0);
    CHECK(composite.statistics().fallback_interactions == 1);
  }

  SECTION("InteractionInterface - unsupported HE final state is fatal") {
    DummyHadronicModel preferred(10_GeV, Code::Argon);
    DummyHadronicModel fallback(10_GeV, Code::Argon);
    HadronicInteractionModelFallback composite{preferred, fallback};
    corsika::proposal::Interaction rejectingModel(
        *env, hadModelLE, composite, heThresholdLab);
    FourMomentum P4(100_TeV, {cs, {100_TeV, 0_eV, 0_eV}});
    CHECK_THROWS_WITH(
        rejectingModel.doHadronicPhotonInteraction(
            view, cs, P4, Code::Argon),
        Catch::Matchers::ContainsSubstring(
            "high-energy photo-hadronic final-state model rejected"));
    CHECK(composite.statistics().preferred_interactions == 0);
    CHECK(composite.statistics().fallback_interactions == 0);
  }

  SECTION("InteractionInterface - unsupported LE final state is fatal") {
    // The low-energy path first samples an on-shell nucleon from the target
    // nucleus and then presents that proton/neutron to the event generator.
    DummyHadronicModel rejectingLE(
        100_MeV, Code::Proton, Code::Neutron);
    corsika::proposal::Interaction rejectingModel(
        *env, rejectingLE, hadModelHE, heThresholdLab);
    FourMomentum P4(10_GeV, {cs, {10_GeV, 0_eV, 0_eV}});
    CHECK_THROWS_WITH(
        rejectingModel.doHadronicPhotonInteraction(
            view, cs, P4, Code::Oxygen),
        Catch::Matchers::ContainsSubstring(
            "low-energy photo-hadronic final-state model rejected"));
  }

  SECTION("Near-threshold photoproduction fallback conserves four-momentum") {
    ThresholdPhotoproductionModel thresholdModel;
    auto const target = Code::Proton;
    auto const targetMass = get_mass(target);
    auto const physicalThreshold =
        targetMass + get_mass(Code::Pi0);
    auto const sophiaThreshold =
        corsika::sophia::minimumCorsikaComEnergy(
            target);
    REQUIRE(physicalThreshold < sophiaThreshold);
    auto const sqrtS =
        0.5 * (physicalThreshold + sophiaThreshold);
    auto const photonEnergy =
        (sqrtS * sqrtS - targetMass * targetMass) /
        (2. * targetMass);
    FourMomentum const photonP4(
        photonEnergy,
        MomentumVector(cs, {photonEnergy, 0_eV, 0_eV}));
    FourMomentum const targetP4(
        targetMass,
        MomentumVector(cs, {0_eV, 0_eV, 0_eV}));

    CHECK(thresholdModel.isValid(
        Code::Photon, target, sqrtS));
    CHECK_FALSE(thresholdModel.isValid(
        Code::Photon, target,
        physicalThreshold - 1_eV));
    CHECK_FALSE(thresholdModel.isValid(
        Code::Photon, target, sophiaThreshold));
    CHECK_FALSE(thresholdModel.isValid(
        Code::Electron, target, sqrtS));

    auto& stack = *stackPtr;
    CHECK(thresholdModel.statistics().interactions == 0);
    thresholdModel.doInteraction(
        view, Code::Photon, target,
        photonP4, targetP4);
    CHECK(thresholdModel.statistics().interactions == 1);
    CHECK(stack.getEntries() == 3);

    HEPEnergyType secondaryEnergy = 0_eV;
    MomentumVector secondaryMomentum(
        cs, {0_eV, 0_eV, 0_eV});
    std::vector<Code> secondaryCodes;
    for (auto const& secondary : view) {
      secondaryEnergy += secondary.getEnergy();
      secondaryMomentum += secondary.getMomentum();
      secondaryCodes.push_back(secondary.getPID());
    }
    REQUIRE(secondaryCodes.size() == 2);
    CHECK(std::find(
              secondaryCodes.begin(), secondaryCodes.end(),
              Code::Pi0) != secondaryCodes.end());
    CHECK(std::find(
              secondaryCodes.begin(), secondaryCodes.end(),
              target) != secondaryCodes.end());
    CHECK(secondaryEnergy / 1_GeV ==
          Catch::Approx(
              (photonP4.getTimeLikeComponent() +
               targetP4.getTimeLikeComponent()) /
              1_GeV)
              .epsilon(2e-13));
    auto const momentumResidual =
        secondaryMomentum -
        photonP4.getSpaceLikeComponents() -
        targetP4.getSpaceLikeComponents();
    CHECK(momentumResidual.getNorm() / 1_GeV <
          1.e-12);
  }
}

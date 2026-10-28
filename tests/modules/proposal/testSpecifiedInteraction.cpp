/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include <corsika/modules/proposal/InteractionModel.hpp>
#include <tests/common/SetupTestEnvironment.hpp>
#include <tests/common/SetupTestStack.hpp>
#include <catch2/catch_test_macros.hpp>
#include <PROPOSAL/medium/Medium.h>
#include <array>
#include <filesystem>
#include <limits>
#include <set>
#ifdef CORSIKA_LEGACY_PROPOSAL_REFERENCE
#include "LegacyInteractionModel.hpp"
#include "ProductionInteractionModel.hpp"
#include <fstream>
#include <iomanip>
#endif

using namespace corsika;
using namespace corsika::proposal;

namespace {
  // Small real tables: this verifies API equivalence, not shower accuracy.
  struct TableSettings {
    double upper_ = PROPOSAL::InterpolationSettings::UPPER_ENERGY_LIM;
    unsigned energy_ = PROPOSAL::InterpolationSettings::NODES_DNDX_E;
    unsigned loss_ = PROPOSAL::InterpolationSettings::NODES_DNDX_V;
    unsigned dedx_ = PROPOSAL::InterpolationSettings::NODES_DEDX;
    unsigned utility_ = PROPOSAL::InterpolationSettings::NODES_UTILITY;
    unsigned rate_ = PROPOSAL::InterpolationSettings::NODES_RATE_INTERPOLANT;
    std::string path_ = PROPOSAL::InterpolationSettings::TABLES_PATH;
    TableSettings() {
      PROPOSAL::InterpolationSettings::UPPER_ENERGY_LIM = 1.e8;
      PROPOSAL::InterpolationSettings::NODES_DNDX_E = 32;
      PROPOSAL::InterpolationSettings::NODES_DNDX_V = 32;
      PROPOSAL::InterpolationSettings::NODES_DEDX = 64;
      PROPOSAL::InterpolationSettings::NODES_UTILITY = 64;
      PROPOSAL::InterpolationSettings::NODES_RATE_INTERPOLANT = 64;
      auto path = std::filesystem::path(corsika_data("PROPOSAL").string());
      std::filesystem::create_directories(path);
      PROPOSAL::InterpolationSettings::TABLES_PATH = path.string();
    }
    ~TableSettings() {
      PROPOSAL::InterpolationSettings::UPPER_ENERGY_LIM = upper_;
      PROPOSAL::InterpolationSettings::NODES_DNDX_E = energy_;
      PROPOSAL::InterpolationSettings::NODES_DNDX_V = loss_;
      PROPOSAL::InterpolationSettings::NODES_DEDX = dedx_;
      PROPOSAL::InterpolationSettings::NODES_UTILITY = utility_;
      PROPOSAL::InterpolationSettings::NODES_RATE_INTERPOLANT = rate_;
      PROPOSAL::InterpolationSettings::TABLES_PATH = path_;
    }
  };

  class HadronicFixture {
  public:
    bool isValid(Code, Code, HEPEnergyType) const { return true; }
    template <typename TView>
    void doInteraction(TView& view, Code, Code, FourMomentum const& momentum,
                       FourMomentum const&) {
      view.addSecondary(std::make_tuple(Code::PiPlus, momentum.getTimeLikeComponent() / 2,
                                        view.getProjectile().getDirection()));
    }
  };

  void compare(PROPOSAL::ParticleState const& a, PROPOSAL::ParticleState const& b) {
    CHECK(a.type == b.type);
    CHECK(a.energy == b.energy);
    CHECK(a.time == b.time);
    CHECK(a.position.GetX() == b.position.GetX());
    CHECK(a.position.GetY() == b.position.GetY());
    CHECK(a.position.GetZ() == b.position.GetZ());
    CHECK(a.direction.GetX() == b.direction.GetX());
    CHECK(a.direction.GetY() == b.direction.GetY());
    CHECK(a.direction.GetZ() == b.direction.GetZ());
  }

  auto& randomManager() {
    auto& rng = RNGManager<>::getInstance();
    rng.setSeed(192837);
    if (!rng.isRegistered("proposal")) rng.registerRandomStream("proposal");
    rng.getRandomStream("proposal").reset();
    return rng;
  }
} // namespace

TEST_CASE("Specified PROPOSAL losses and final states match native calculators",
          "[proposal_completion]") {
  TableSettings settings;
  auto& rng = randomManager();
  auto const before = rng.dumpState().str();
  ProposalRateProvider provider;
  ProposalFinalStateGenerator generator;
  std::vector<std::unique_ptr<PROPOSAL::Medium>> media;
  media.push_back(std::make_unique<PROPOSAL::Air>());
  media.push_back(std::make_unique<PROPOSAL::Ice>());
  media.push_back(std::make_unique<PROPOSAL::StandardRock>());
  std::set<Code> pairProjectiles;
  std::set<PROPOSAL::InteractionType> observed;
  for (auto& medium : media) {
    for (auto code :
         {Code::Photon, Code::Electron, Code::Positron, Code::MuMinus, Code::MuPlus}) {
      CAPTURE(medium->GetName(), code);
      auto crosses = cross.at(code)(*medium, .4_MeV);
      auto types = PROPOSAL::CrossSectionVector::GetInteractionTypes(crosses);
      auto interaction = PROPOSAL::make_interaction(crosses, true, true);
      auto secondary = PROPOSAL::make_secondaries(
          types, corsika::proposal::particle.at(code), *medium);
      for (double energy : {1.e4, 1.e7}) {
        auto table = provider.rates(*interaction, energy);
        auto native = interaction->Rates(energy);
        REQUIRE(table.entries().size() == native.size());
        ProposalInteractionContext context{code,          medium->GetHash(), energy,
                                           {1., -2., 3.}, {0., 0., 1.},      2.e-9};
        for (std::size_t i = 0; i < native.size(); ++i) {
          auto const& column = native[i];
          auto const type = column.crosssection->GetInteractionType();
          CHECK(table.entries()[i].rate_ == column.rate);
          CHECK(table.entries()[i].componentHash_ == column.comp_hash);
          CHECK(table.entries()[i].type_ == type);
          if (!(column.rate > 0.)) continue;
          observed.insert(type);
          for (double quantile : {.13, .5, .91}) {
            CAPTURE(energy, type, quantile);
            double expected = column.crosssection->CalculateStochasticLoss(
                column.comp_hash, energy, quantile * column.rate);
            double actual =
                provider.sampleSelectedLoss(table, type, column.comp_hash, quantile);
            REQUIRE(actual == expected);
            ProposalInteractionRecord record{
                context,     interaction->GetHash(), type, column.comp_hash, actual, .3,
                std::nullopt};
            std::vector<double> draws(
                generator.requiredRandomNumbers(*secondary, record));
            for (std::size_t k = 0; k < draws.size(); ++k)
              draws[k] = (double(k) + .5) / (draws.size() + 1.);
            auto result = generator.generate(*secondary, record, draws);
            PROPOSAL::Component target;
            if (type != PROPOSAL::InteractionType::Ioniz)
              target = PROPOSAL::Component::GetComponentForHash(column.comp_hash);
            PROPOSAL::StochasticLoss loss(static_cast<int>(type), expected * energy,
                                          PROPOSAL::Cartesian3D(1., -2., 3.),
                                          PROPOSAL::Cartesian3D(0., 0., 1.), 2.e-9, 0.,
                                          energy);
            auto reference = secondary->CalculateSecondaries(loss, target, draws);
            REQUIRE(result.secondaries_.size() == reference.size());
            for (std::size_t k = 0; k < reference.size(); ++k)
              compare(result.secondaries_[k], reference[k]);
            if (type == PROPOSAL::InteractionType::Epair) pairProjectiles.insert(code);
            draws.push_back(.5);
            CHECK_THROWS_AS(generator.generate(*secondary, record, draws),
                            std::invalid_argument);
          }
        }
        for (double u : {.0, .37, .999}) {
          auto record = provider.sample(*interaction, table, context, u);
          auto reference = interaction->SampleLoss(energy, native, u);
          CHECK(record.type_ == reference.type);
          CHECK(record.componentHash_ == reference.comp_hash);
          CHECK(record.vLoss_ == reference.v_loss);
        }
        CHECK_THROWS_AS(
            provider.sampleSelectedLoss(table, PROPOSAL::InteractionType::Epair, 0, 1.),
            std::invalid_argument);
        auto invalid = context;
        invalid.projectileEnergyMeV_ *= 2.;
        CHECK_THROWS_AS(provider.sample(*interaction, table, invalid, .5),
                        std::invalid_argument);
      }
    }
  }
  CHECK(pairProjectiles ==
        std::set<Code>{Code::Electron, Code::Positron, Code::MuMinus, Code::MuPlus});
  CHECK(observed.count(PROPOSAL::InteractionType::Photopair) == 1);
  CHECK(observed.count(PROPOSAL::InteractionType::PhotoMuPair) == 1);
  CHECK(observed.count(PROPOSAL::InteractionType::Photonuclear) == 1);
  CHECK(before == rng.dumpState().str());
}

TEST_CASE("InteractionModel completes the retained identity and exact cut",
          "[proposal_completion]") {
  TableSettings settings;
  auto& rng = randomManager();
  auto [environment, cs, node] = setup::testing::setup_environment(Code::Oxygen);
  HadronicFixture low, high;
  InteractionModel model(*environment, low, high, 80_GeV);
  auto const before = rng.dumpState().str();
  for (auto code :
       {Code::Photon, Code::Electron, Code::Positron, Code::MuMinus, Code::MuPlus}) {
    CAPTURE(code);
    auto [stack, view] = setup::testing::setup_stack(code, 100_GeV, node, *cs);
    auto projectile = view->getProjectile();
    auto rates = model.getRateTable(projectile, code);
    auto record = model.sampleInteraction(projectile, code, rates, .43);
    auto replay = record;
    replay.vLoss_ = std::numeric_limits<double>::quiet_NaN();
    model.completeNativeSelectionReplay(replay);
    CHECK(replay.vLoss_ == record.vLoss_);
    CHECK(replay.type_ == record.type_);
    CHECK(replay.componentHash_ == record.componentHash_);
    auto wrong = replay;
    wrong.componentHash_ ^= 123;
    CHECK_THROWS(model.completeNativeSelectionReplay(wrong));
    wrong = replay;
    wrong.interactionHash_ ^= 123;
    CHECK_THROWS(model.requiredFinalStateRandomNumbers(wrong));
    for (auto const& rate : rates.nativeRates()) {
      if (!(rate.rate > 0.)) continue;
      auto selected = record;
      selected.type_ = rate.crosssection->GetInteractionType();
      selected.componentHash_ = rate.comp_hash;
      model.completeSelectedLoss(selected, .37);
      CHECK(selected.vLoss_ == rate.crosssection->CalculateStochasticLoss(
                                   rate.comp_hash, rates.energyMeV(), .37 * rate.rate));
      CHECK(selected.type_ == rate.crosssection->GetInteractionType());
      CHECK(selected.componentHash_ == rate.comp_hash);
      auto draws =
          std::vector<double>(model.requiredFinalStateRandomNumbers(selected), .42);
      CHECK_FALSE(model.generateFinalState(selected, draws).secondaries_.empty());
    }
    if (code == Code::Photon) continue;
    for (auto const& native : model.nativeCalculatorViews()) {
      if (native.getProjectile() != code) continue;
      auto medium = native.getMedium()->get();
      auto crosses = cross.at(code)(medium, .7_MeV);
      auto exact = PROPOSAL::make_interaction(crosses, true, true);
      ProposalRateProvider provider;
      auto exactRates = provider.rates(*exact, rates.energyMeV());
      auto exactRecord = provider.sample(*exact, exactRates, record.context_, .43);
      CHECK_THROWS(model.requiredFinalStateRandomNumbers(exactRecord));
      CHECK_THROWS(model.prepareSpecifiedInteractionCalculator(exactRecord, .8_MeV));
      model.prepareSpecifiedInteractionCalculator(exactRecord, .7_MeV);
      auto completed = exactRecord;
      completed.vLoss_ = 0.;
      model.completeNativeSelectionReplay(completed);
      CHECK(completed.vLoss_ == exactRecord.vLoss_);
      CHECK(model.getRateTable(projectile, code).interactionHash() ==
            rates.interactionHash());
      CHECK(model.getRateTable(projectile, code).totalRate() == rates.totalRate());
    }
  }
  CHECK(before == rng.dumpState().str());
}

TEST_CASE("Specified atomic-electron interactions keep the medium energy ledger",
          "[proposal_completion]") {
  TableSettings settings;
  auto& rng = randomManager();
  auto [environment, cs, node] = setup::testing::setup_environment(Code::Oxygen);
  HadronicFixture low, high;
  InteractionModel model(*environment, low, high, 80_GeV);
  auto [stack, view] = setup::testing::setup_stack(Code::Electron, 100_GeV, node, *cs);
  auto projectile = view->getProjectile();
  projectile.setWeight(3.);
  auto rates = model.getRateTable(projectile, Code::Electron);
  auto record = model.sampleInteraction(projectile, Code::Electron, rates, .43);
  bool found = false;
  for (auto const& rate : rates.nativeRates()) {
    if (rate.crosssection->GetInteractionType() != PROPOSAL::InteractionType::Ioniz ||
        !(rate.rate > 0.))
      continue;
    record.type_ = PROPOSAL::InteractionType::Ioniz;
    record.componentHash_ = rate.comp_hash;
    model.completeSelectedLoss(record, .37);
    found = true;
    break;
  }
  REQUIRE(found);
  auto before = rng.dumpState().str();
  std::vector<double> draws(model.requiredFinalStateRandomNumbers(record), .42);
  REQUIRE(model.doSpecifiedInteraction(*view, record, draws) == ProcessReturn::Ok);
  CHECK(model.atomicElectronTargetInteractions() == 1);
  CHECK(model.weightedAtomicElectronRestMassInputGeV() == 3. * PROPOSAL::ME / 1000.);
  CHECK(stack->getEntries() > 1);
  CHECK(rng.dumpState().str() == before);
}

#ifdef CORSIKA_LEGACY_PROPOSAL_REFERENCE
TEST_CASE("Ordinary interactions preserve production and bound upstream roundoff",
          "[proposal_legacy]") {
  TableSettings settings;
  auto& rng = randomManager();
  auto [environment, cs, node] = setup::testing::setup_environment(Code::Oxygen);
  HadronicFixture low, high;
  InteractionModel current(*environment, low, high, 80_GeV);
  LegacyInteractionModel legacy(*environment, low, high, 80_GeV);
  corsika::production_reference::InteractionModel production(*environment, low, high,
                                                             80_GeV);
  double maxEnergyRelative = 0., maxDirectionAbsolute = 0.;
  std::ofstream differences("native-roundoff.csv");
  differences << "pid,momentum_GeV,seed,energy_relative,direction_absolute\n"
              << std::setprecision(17);
  for (auto code : {Code::Photon, Code::Electron, Code::Positron, Code::MuMinus,
                    Code::MuPlus, Code::TauMinus, Code::TauPlus}) {
    for (auto momentum : {1_GeV, 100_GeV, 10000_GeV}) {
      for (unsigned seed = 1; seed <= 16; ++seed) {
        CAPTURE(code, momentum, seed);
        auto [a, av] = setup::testing::setup_stack(code, momentum, node, *cs);
        auto [b, bv] = setup::testing::setup_stack(code, momentum, node, *cs);
        auto [c, cv] = setup::testing::setup_stack(code, momentum, node, *cs);
        auto p = av->getProjectile();
        FourMomentum p4(p.getEnergy(), p.getMomentum());
        rng.setSeed(seed);
        rng.getRandomStream("proposal").reset();
        auto const initial = rng.getRandomStream("proposal");
        auto const initialState = rng.dumpState().str();
        auto expected = legacy.doInteraction(*av, code, p4);
        auto state = rng.dumpState().str();
        rng.getRandomStream("proposal") = initial;
        REQUIRE(rng.dumpState().str() == initialState);
        CHECK(current.doInteraction(*bv, code, p4) == expected);
        CHECK(rng.dumpState().str() == state);
        REQUIRE(a->getEntries() == b->getEntries());
        rng.getRandomStream("proposal") = initial;
        CHECK(production.doInteraction(*cv, code, p4) == expected);
        CHECK(rng.dumpState().str() == state);
        REQUIRE(c->getEntries() == b->getEntries());
        auto productionParticle = c->begin();
        auto j = b->begin();
        for (auto i = a->begin(); i != a->end(); ++i, ++j, ++productionParticle) {
          CHECK(productionParticle.getPID() == j.getPID());
          CHECK(productionParticle.getEnergy() == j.getEnergy());
          CHECK(productionParticle.getWeight() == j.getWeight());
          CHECK(productionParticle.getTime() == j.getTime());
          CHECK(productionParticle.getNode() == j.getNode());
          CHECK(i.getPID() == j.getPID());
          double energyRelative =
              std::abs((i.getEnergy() - j.getEnergy()) / i.getEnergy());
          maxEnergyRelative = std::max(maxEnergyRelative, energyRelative);
          CHECK(energyRelative <= 64. * std::numeric_limits<double>::epsilon());
          CHECK(i.getWeight() == j.getWeight());
          CHECK(i.getTime() == j.getTime());
          CHECK(i.getNode() == j.getNode());
          auto n = i.getDirection().getComponents(*cs);
          auto m = j.getDirection().getComponents(*cs);
          auto original = productionParticle.getDirection().getComponents(*cs);
          CHECK(original.getX() == m.getX());
          CHECK(original.getY() == m.getY());
          CHECK(original.getZ() == m.getZ());
          double directionAbsolute =
              std::max({std::abs((n.getX() - m.getX()).magnitude()),
                        std::abs((n.getY() - m.getY()).magnitude()),
                        std::abs((n.getZ() - m.getZ()).magnitude())});
          maxDirectionAbsolute = std::max(maxDirectionAbsolute, directionAbsolute);
          CHECK(directionAbsolute <= 1.e-10);
          differences << int(code) << ',' << momentum / 1_GeV << ',' << seed << ','
                      << energyRelative << ',' << directionAbsolute << '\n';
        }
      }
    }
  }
}
#endif

/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include <corsika/kokkos/em/proposal/ProposalCpuFallbackHandler.hpp>
#include <corsika/modules/proposal/InteractionModel.hpp>
#include <corsika/modules/ParticleCut.hpp>
#include <tests/common/TransportTestStack.hpp>
#include <tests/common/SetupTestEnvironment.hpp>
#include <catch2/catch_test_macros.hpp>
#include <filesystem>
#include <set>
#ifdef CORSIKA_PRODUCTION_HANDLER_REFERENCE
#include "ProductionCpuHandler.hpp"
#endif
using namespace corsika;
namespace cp = corsika::kokkos;
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

  struct Thresholds {
    std::vector<std::pair<Code, HEPEnergyType>> old_;
    Thresholds() {
      for (auto p : get_all_particles())
        old_.emplace_back(p, get_kinetic_energy_propagation_threshold(p));
    }
    ~Thresholds() {
      for (auto p : old_) set_kinetic_energy_propagation_threshold(p.first, p.second);
    }
  };
  struct Postprocess {
    ParticleCut<> cut_{.4_MeV, .4_MeV, .4_MeV, .4_MeV, .4_MeV, true};
    unsigned calls_ = 0;
    template <typename TView>
    void doSecondaries(TView& view) {
      ++calls_;
      for (auto& p : view) {
        CHECK(p.getParentHistoryId() == 42);
        CHECK(p.getGeneration() == 3);
        CHECK(p.getNode() == view.getProjectile().getNode());
      }
      cut_.doSecondaries(view);
    }
  };
  using TransportStackType = test::TransportStack<DummyEnvironment>::type;
#ifdef CORSIKA_PRODUCTION_HANDLER_REFERENCE
  void compare(TransportStackType& a, TransportStackType& b, CoordinateSystemPtr cs) {
    REQUIRE(a.getEntries() == b.getEntries());
    auto j = b.begin();
    for (auto i = a.begin(); i != a.end(); ++i, ++j) {
      CHECK(i.getPID() == j.getPID());
      CHECK(i.getEnergy() == j.getEnergy());
      CHECK(i.getWeight() == j.getWeight());
      CHECK(i.getTime() == j.getTime());
      CHECK(i.getNode() == j.getNode());
      CHECK(i.getHistoryId() == j.getHistoryId());
      CHECK(i.getParentHistoryId() == j.getParentHistoryId());
      CHECK(i.getGeneration() == j.getGeneration());
      CHECK(i.getStepId() == j.getStepId());
      auto ni = i.getDirection().getComponents(cs),
           nj = j.getDirection().getComponents(cs);
      auto pi = i.getPosition().getCoordinates(cs),
           pj = j.getPosition().getCoordinates(cs);
      for (int k = 0; k < 3; ++k) {
        CHECK(ni[k] == nj[k]);
        CHECK(pi[k] == pj[k]);
      }
    }
  }
#endif
} // namespace

TEST_CASE("CPU handler retains real specified reactions and native replay",
          "[cpu_handler_physics]") {
  TableSettings settings;
  Thresholds thresholds;
  auto& rng = RNGManager<>::getInstance();
  rng.setSeed(192837);
  if (!rng.isRegistered("proposal")) rng.registerRandomStream("proposal");
  rng.getRandomStream("proposal").reset();
  auto [env, cs, node] = setup::testing::setup_environment(Code::Oxygen);
  cp::EnvironmentTopology<DummyEnvironmentInterface> tree(*env->getUniverse());
  Postprocess post, referencePost;
  HadronicFixture low, high;
  proposal::InteractionModel model(*env, low, high, 80_GeV);
#ifdef CORSIKA_PRODUCTION_HANDLER_REFERENCE
  proposal::InteractionModel referenceModel(*env, low, high, 80_GeV);
  cp::production_reference::environment_tree::HostMap<DummyEnvironmentInterface>
      nativeMap(tree);
#endif
  std::set<Code> pairs;
  std::set<PROPOSAL::InteractionType> types;
  unsigned completed = 0;
  auto execute = [&](cp::FallbackRequest const& e) {
    TransportStackType output;
    cp::ProposalCpuFallbackHandler<TransportStackType, decltype(model), Postprocess,
                                   DummyEnvironment>
        handler(model, post, *env, tree, *cs, 17, 19);
#ifdef CORSIKA_PRODUCTION_HANDLER_REFERENCE
    auto initial = rng.getRandomStream("proposal");
#endif
    REQUIRE(handler.canHandle(e));
    handler.handle(output, e);
#ifdef CORSIKA_PRODUCTION_HANDLER_REFERENCE
    auto expectedState = rng.dumpState().str();
#endif
    CHECK(handler.statistics().specifiedInteractions_ == 1);
    CHECK(handler.statistics().generatedSecondaries_ > 0);
    CHECK(output.getEntries() > 0);
    for (auto& p : output) {
      CHECK(p.getHistoryId() != 42);
      CHECK(p.getKineticEnergy() >= 0_GeV);
    }
#ifdef CORSIKA_PRODUCTION_HANDLER_REFERENCE
    TransportStackType original;
    cp::production_reference::ProposalCpuFallbackHandler<
        TransportStackType, decltype(referenceModel), Postprocess, DummyEnvironment>
        old(referenceModel, referencePost, *env, *cs, 17, 19);
    old.setEnvironmentTree(nativeMap.view());
    // Production LPM acceptance may draw from the sequential stream.
    // Compare identical starting streams, not an incorrect no-draw assumption.
    rng.getRandomStream("proposal") = initial;
    old.handle(original, e);
    compare(output, original, *cs);
    CHECK(rng.dumpState().str() == expectedState);
    CHECK(handler.statistics().generatedSecondaries_ ==
          old.statistics().generatedSecondaries_);
    CHECK(handler.statistics().completedSelectedLosses_ ==
          old.statistics().completedSelectedLosses_);
    CHECK(handler.statistics().completedNativeSelectionReplays_ ==
          old.statistics().completedNativeSelectionReplays_);
#endif
    ++completed;
  };
  for (auto code :
       {Code::Photon, Code::Electron, Code::Positron, Code::MuMinus, Code::MuPlus}) {
    for (auto energy : {100_GeV, 10000_GeV}) {
      CAPTURE(code, energy);
      TransportStackType source;
      auto p = source.addParticle(std::make_tuple(code, energy,
                                                  DirectionVector(*cs, {0., 0., 1.}),
                                                  Point(*cs, 0_m, 0_m, 0_m), 2_ns));
      p.setNode(node);
      p.setWeight(3.);
      p.setTransportIdentity({42, 3, 2, 7});
      auto rates = model.getRateTable(p, code);
      auto record = model.sampleInteraction(p, code, rates, .43);
      cp::FallbackRequest e{};
      e.particle_ = cp::particle_transfer::to_device_state(p, *cs, 42, 3, 2, 7, tree);
      e.mediumHash_ = 7654; // Deliberately different hash domain; handler must use the
                            // node's composition.
      e.interactionHash_ = record.interactionHash_;
      e.interactionVertexReached_ = 1;
      e.selectionUniform_ = .43;
      e.lossQuantile_ = .37;
      for (auto const& column : rates.nativeRates()) {
        if (!(column.rate > 0.)) continue;
        auto type = column.crosssection->GetInteractionType();
        types.insert(type);
        if (type == PROPOSAL::InteractionType::Epair) pairs.insert(code);
        e.processId_ = static_cast<int>(type);
        e.componentHash_ = column.comp_hash;
        e.energyFraction_ = column.crosssection->CalculateStochasticLoss(
            column.comp_hash, rates.energyMeV(), .37 * column.rate);
        e.reason_ = cp::process_capability(e.particle_.pid_, e.processId_) ==
                            cp::ProcessCapability::CpuOnlyFinalState
                        ? cp::FallbackReason::CpuOnlyProcess
                        : cp::FallbackReason::InverseCdfUnavailable;
        CAPTURE(type);
        execute(e);
      }
      e.processId_ = static_cast<int>(record.type_);
      e.componentHash_ = record.componentHash_;
      e.energyFraction_ = 0.;
      e.lossQuantile_ = std::numeric_limits<double>::quiet_NaN();
      e.reason_ = cp::FallbackReason::NativeSelectionReplay;
      execute(e);
    }
  }
  CHECK(pairs ==
        std::set<Code>{Code::Electron, Code::Positron, Code::MuMinus, Code::MuPlus});
  CHECK(types.count(PROPOSAL::InteractionType::PhotoMuPair) == 1);
  CHECK(types.count(PROPOSAL::InteractionType::Photopair) == 1);
  CHECK(types.count(PROPOSAL::InteractionType::Photonuclear) == 1);
  CHECK(types.count(PROPOSAL::InteractionType::Brems) == 1);
  CHECK(post.calls_ == completed);
}

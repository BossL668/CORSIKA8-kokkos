/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include <corsika/kokkos/scheduling/HadronicInteractionDeferral.hpp>
#include <corsika/framework/geometry/RootCoordinateSystem.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cstring>
#include <future>
#include <limits>
#ifdef CORSIKA_PRODUCTION_HADRONIC_REFERENCE
#include "ProductionHadronicInteractionDeferral.hpp"
namespace reference = corsika::production_hadronic_reference;
#endif
using namespace corsika;
namespace cp = corsika::kokkos;
namespace {
  struct WorkerProcess {
    static constexpr auto hadronic_worker_model = cp::HadronicWorkerModel::Sibyll;
  };
  struct HostProcess {};
  static_assert(cp::has_hadronic_worker_model_v<WorkerProcess>);
  static_assert(!cp::has_hadronic_worker_model_v<HostProcess>);
  static_assert(!std::is_copy_constructible_v<cp::HadronicInteractionDeferralContext>);
  static_assert(!std::is_copy_constructible_v<cp::ScopedHadronicInteractionDeferral>);
} // namespace

TEST_CASE("Hadronic capture preserves the selected request and can be consumed once",
          "[hadronic_deferral]") {
  auto const& cs = get_root_CoordinateSystem();
  FourMomentum projectile{100_GeV, MomentumVector(cs, {3_GeV, 4_GeV, 90_GeV})};
  FourMomentum target{get_mass(Code::Oxygen), MomentumVector(cs, {0_GeV, 0_GeV, 0_GeV})};
  cp::HadronicRandomKey key{11, 22, 33, 44, 55, 66};
  for (auto model : {cp::HadronicWorkerModel::Fluka, cp::HadronicWorkerModel::Sibyll,
                     cp::HadronicWorkerModel::QgsJetII, cp::HadronicWorkerModel::EposLhc,
                     cp::HadronicWorkerModel::Pythia8}) {
    cp::HadronicInteractionDeferralContext context(key, 1234);
    CHECK_FALSE(context.hasPrepared());
    CHECK_THROWS_AS(context.takePrepared(), std::logic_error);
    CHECK(context.tryPrepare(model, Code::Proton, Code::Oxygen, projectile, target));
    REQUIRE(context.hasPrepared());
    CHECK_THROWS_AS(
        context.tryPrepare(model, Code::Proton, Code::Oxygen, projectile, target),
        std::logic_error);
    auto result = context.takePrepared();
    CHECK_FALSE(context.hasPrepared());
    CHECK_THROWS_AS(context.takePrepared(), std::logic_error);
    auto const& request = result.request_;
    CHECK(request.sequenceId_ == 1234);
    CHECK(request.protocolVersion_ == cp::HadronicBatchProtocolVersion);
    CHECK(request.model_ == model);
    CHECK(request.projectilePdg_ == static_cast<std::int32_t>(get_PDG(Code::Proton)));
    CHECK(request.targetPdg_ == static_cast<std::int32_t>(get_PDG(Code::Oxygen)));
    CHECK(request.projectileFourMomentumGeV_ == std::array<double, 4>{100., 3., 4., 90.});
    CHECK(request.targetFourMomentumGeV_ ==
          std::array<double, 4>{get_mass(Code::Oxygen) / 1_GeV, 0., 0., 0.});
    CHECK(cp::derive_hadronic_worker_seed(request.randomKey_) ==
          cp::derive_hadronic_worker_seed(key));
#ifdef CORSIKA_PRODUCTION_HADRONIC_REFERENCE
    reference::HadronicRandomKey oldKey{};
    std::memcpy(&oldKey, &key, sizeof(key));
    reference::HadronicInteractionDeferralContext old(oldKey, 1234);
    CHECK(old.tryPrepare(static_cast<reference::HadronicWorkerModel>(model), Code::Proton,
                         Code::Oxygen, projectile, target));
    auto original = old.takePrepared().request;
    reference::HadronicInteractionRequest wire{};
    std::memcpy(&wire, &request, sizeof(wire));
    CHECK(wire.protocol_version == original.protocol_version);
    CHECK(wire.model == original.model);
    CHECK(wire.reserved == original.reserved);
    CHECK(wire.sequence_id == original.sequence_id);
    CHECK(wire.projectile_pdg == original.projectile_pdg);
    CHECK(wire.target_pdg == original.target_pdg);
    CHECK(wire.projectile_four_momentum_GeV == original.projectile_four_momentum_GeV);
    CHECK(wire.target_four_momentum_GeV == original.target_four_momentum_GeV);
    CHECK(reference::deriveHadronicWorkerSeed(wire.random_key) ==
          reference::deriveHadronicWorkerSeed(original.random_key));
#endif
    // A rejected request must not leave a pending interaction behind.
    FourMomentum invalid{0_GeV, MomentumVector(cs, {0_GeV, 0_GeV, 0_GeV})};
    CHECK_THROWS_AS(
        context.tryPrepare(model, Code::Proton, Code::Oxygen, invalid, target),
        std::invalid_argument);
    CHECK_FALSE(context.hasPrepared());
    CHECK(context.tryPrepare(model, Code::Proton, Code::Oxygen, projectile, target));
    CHECK(context.takePrepared().request_.sequenceId_ == 1234);
  }
}

TEST_CASE("Hadronic deferral scopes restore nesting, exceptions and thread isolation",
          "[hadronic_deferral]") {
  CHECK(cp::HadronicInteractionDeferralContext::current() == nullptr);
  cp::HadronicInteractionDeferralContext outer({}, 1), inner({}, 2);
  {
    cp::ScopedHadronicInteractionDeferral active(outer);
    CHECK(cp::HadronicInteractionDeferralContext::current() == &outer);
    CHECK_THROWS_AS(([&]() {
                      cp::ScopedHadronicInteractionDeferral nested(inner);
                      CHECK(cp::HadronicInteractionDeferralContext::current() == &inner);
                      throw std::runtime_error("scope restoration test");
                    })(),
                    std::runtime_error);
    CHECK(cp::HadronicInteractionDeferralContext::current() == &outer);
    auto isolated = std::async(std::launch::async, []() {
      if (cp::HadronicInteractionDeferralContext::current() != nullptr) return false;
      cp::HadronicInteractionDeferralContext local({}, 3);
      {
        cp::ScopedHadronicInteractionDeferral scope(local);
        if (cp::HadronicInteractionDeferralContext::current() != &local) return false;
      }
      return cp::HadronicInteractionDeferralContext::current() == nullptr;
    });
    CHECK(isolated.get());
    CHECK(cp::HadronicInteractionDeferralContext::current() == &outer);
  }
  CHECK(cp::HadronicInteractionDeferralContext::current() == nullptr);
}

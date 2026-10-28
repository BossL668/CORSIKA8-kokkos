/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */
#include <corsika/framework/process/InteractionCounter.hpp>
#include <corsika/framework/geometry/RootCoordinateSystem.hpp>
#include <catch2/catch_test_macros.hpp>
#include <numeric>

namespace {
  struct View {};
  struct Model {
    int calls_ = 0;
    void doInteraction(View&, corsika::Code, corsika::Code, corsika::FourMomentum const&,
                       corsika::FourMomentum const&) {
      ++calls_;
    }
  };
} // namespace

TEST_CASE("An ordinary counter needs no execution adapter", "[interaction_deferral]") {
  using namespace corsika;
  Model model;
  View view;
  InteractionCounter counter(model);
  auto const cs = get_root_CoordinateSystem();
  FourMomentum projectile(100_GeV, MomentumVector(cs, {0_GeV, 0_GeV, 99_GeV}));
  FourMomentum target(get_mass(Code::Oxygen), MomentumVector(cs, {0_GeV, 0_GeV, 0_GeV}));
  counter.doInteraction(view, Code::Proton, Code::Oxygen, projectile, target);
  CHECK(model.calls_ == 1);
  CHECK(counter.getCount() == 1);
  CHECK(counter.getEnergyLedgerStatistics().auditedInteractions_ == 0);
  CHECK(counter.getEnergyLedgerStatistics().deferredInteractions_ == 0);
  auto const& histogram = counter.getHistogram().labHist();
  CHECK(std::accumulate(histogram.begin(), histogram.end(), 0) == 1);
}

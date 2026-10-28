/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include "BackendFixture.hpp"
#include <corsika/kokkos/scheduling/EmRouter.hpp>
#include <corsika/kokkos/em/algorithms/TransportMass.hpp>
#include <tests/common/TransportTestStack.hpp>
#include <chrono>
#include <cmath>
#include <iostream>
#include <thread>

int main(int argc, char** argv) {
  using namespace backend_test;
  try {
    require(argc == 3, "backend and cache directory arguments");
    Fixture fixture(argv[2]);
    // This case exercises resident terminal cuts. Keep its three particles
    // above dispatch threshold; small-batch scalar expansion is a distinct
    // original router policy, not a failed or lost GPU cut.
    fixture.config_.minBatchSize_ = 1;
    cp::RuntimeConfig runtime;
    runtime.executionBackend_ = argv[1];
    runtime.threads_ =
        (runtime.executionBackend_ == "cuda" || runtime.executionBackend_ == "hip" ||
         runtime.executionBackend_ == "sycl")
            ? 1
            : 2;
    cp::EmBackend backend(runtime);
    fixture.initialize(backend);
    using stack_type = test::TransportStack<environment_type>::type;
    stack_type stack;
    auto const data = cp::bind_environment_data(*fixture.snapshot_);
    cp::EnvironmentBorrowedView const view(data);
    cp::physical_em_router_type<stack_type, model_interface_type, cp::EmBackend> router(
        backend, fixture.coordinates_, view, fixture.geometry_,
        fixture.snapshot_->getTopology());
    router.setFailOnUnexpectedFallback(true);
    for (auto code : {Code::Photon, Code::Electron, Code::MuMinus}) {
      auto const mass = code == Code::Photon
                            ? 0.
                            : cp::transport_mass_gev(code == Code::Electron ? 11 : 13);
      // The native C8 stack constructor takes kinetic energy, not total energy.
      auto particle = stack.addParticle(std::make_tuple(
          code, .0001 * 1_GeV, DirectionVector(fixture.coordinates_, {0., 0., -1.}),
          Point(fixture.coordinates_, 0_m, 0_m, 0_m), 0_s));
      particle.setNode(fixture.node_);
      particle.setWeight(1.);
      particle.setTransportIdentity({100 + stack.getEntries(), 1, 1, 7});
      require(std::abs(particle.getEnergy() / 1_GeV - (mass + .0001)) < 1.e-15,
              "native kinetic input retains the intended total energy");
      require(router.canRoute(particle, 7),
              "native node and real backend admit particle");
      router.stage(particle, particle.getHistoryId(), particle.getParentHistoryId(),
                   particle.getGeneration(), particle.getStepId());
    }
    stack.clear();
    unsigned calls = 0;
    auto const deadline = std::chrono::steady_clock::now() + std::chrono::seconds(60);
    while (router.pending() && std::chrono::steady_clock::now() < deadline) {
      ++calls;
      require(router.advanceOneWavefrontAndReturn(stack) == 0,
              "terminal cut has no scalar retry");
      std::this_thread::yield();
    }
    require(!router.pending(), "actual resident queues and router drain");
    require(router.statistics().particlesStaged_ == 3, "three species staged");
    require(router.statistics().particlesCut_ == 3, "three terminal cuts recorded");
    require(stack.getEntries() == 0, "no lost CPU continuation hidden on stack");
    require(router.statistics().depositedEnergyGeV_ > 0.,
            "terminal kinetic deposit retained");
    router.endOfShower();
    require(throws([&] { router.endOfShower(); }),
            "duplicate finalization still rejected");
    std::cout << "real router: cuts=" << router.statistics().particlesCut_
              << " calls=" << calls
              << " deposit=" << router.statistics().depositedEnergyGeV_ << '\n';
    return 0;
  } catch (std::exception const& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}

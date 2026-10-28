/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include <corsika/kokkos/data/RouterParticleConversion.hpp>
#include <corsika/kokkos/data/ParticleConversion.hpp>
#include <corsika/media/ExponentialFirn.hpp>
#include <corsika/media/IRefractiveIndexModel.hpp>
#include <corsika/framework/geometry/SeparationPlane.hpp>
#include <tests/common/TransportTestStack.hpp>
#include <tests/common/SetupTestEnvironment.hpp>
#include <catch2/catch_test_macros.hpp>
#ifdef CORSIKA_PRODUCTION_HANDLER_REFERENCE
#include "ProductionCpuHandler.hpp"
#endif

using namespace corsika;
namespace cp = corsika::kokkos;

TEST_CASE("Complete particle transfer preserves logical node, identity and values",
          "[particle_transfer]") {
  auto [env, cs, node] = setup::testing::setup_environment(Code::Oxygen);
  // A different logical child at the same position is deliberately not selected.
  auto child = DummyEnvironment::createNode<Sphere>(Point(*cs, 0_m, 0_m, 0_m), 1_m);
  auto* other = child.get();
  node->addChild(std::move(child));
  cp::EnvironmentTopology<DummyEnvironmentInterface> tree(*env->getUniverse());
  using stack_type = test::TransportStack<DummyEnvironment>::type;
  stack_type source, destination;
  for (auto code :
       {Code::Photon, Code::Electron, Code::Positron, Code::MuMinus, Code::MuPlus}) {
    auto p = source.addParticle(std::make_tuple(code, 10_GeV,
                                                DirectionVector(*cs, {.6, 0., .8}),
                                                Point(*cs, 0_m, 0_m, 0_m), 2_ns));
    p.setNode(node);
    p.setWeight(3.5);
    p.setTransportIdentity({1024, 3, 4, 12});
    auto state = cp::particle_transfer::to_device_state(
        p, *cs, p.getHistoryId(), p.getParentHistoryId(), p.getGeneration(),
        p.getStepId(), tree);

#ifdef CORSIKA_PRODUCTION_HANDLER_REFERENCE
    cp::production_reference::environment_tree::HostMap<DummyEnvironmentInterface>
        nativeMap(tree);
    auto original = cp::production_reference::router_detail::toDeviceState(
        p, *cs, p.getHistoryId(), p.getParentHistoryId(), p.getGeneration(),
        p.getStepId(), nativeMap.view());
    CHECK(state.pid_ == original.pid_);
    CHECK(state.energyGeV_ == original.energyGeV_);
    CHECK(state.environmentNode_ == original.environmentNode_);
    CHECK(state.timeS_ == original.timeS_);
    CHECK(state.weight_ == original.weight_);
    for (int k = 0; k < 3; ++k) {
      CHECK(state.positionM_[k] == original.positionM_[k]);
      CHECK(state.direction_[k] == original.direction_[k]);
    }
#endif
    cp::ParticleState values;
    cp::export_particle_values(p, *cs, values);
    CHECK(state.pid_ == values.pid_);
    CHECK(state.energyGeV_ == values.energyGeV_);
    CHECK(state.environmentNode_ == tree.getNodeIndex(*(node)));
    CHECK(state.environmentNode_ != tree.getNodeIndex(*(other)));
    for (int k = 0; k < 3; ++k) {
      CHECK(state.positionM_[k] == values.positionM_[k]);
      CHECK(state.direction_[k] == values.direction_[k]);
    }
    auto q = cp::particle_transfer::import_particle(destination, state, *cs, tree);
    CHECK(q.getNode() == node);
    CHECK(q.getHistoryId() == 1024);
    CHECK(q.getParentHistoryId() == 3);
    CHECK(q.getGeneration() == 4);
    CHECK(q.getStepId() == 12);
    CHECK(q.getWeight() == 3.5);
    CHECK(q.getEnergy() == p.getEnergy());
    CHECK(q.getTime() == p.getTime());
    state.energyGeV_ = 0.;
    state.pid_ = 13;
    auto size = destination.getEntries();
    CHECK_THROWS_AS(cp::particle_transfer::import_particle(destination, state, *cs, tree),
                    std::runtime_error);
    CHECK(destination.getEntries() == size);
    state.environmentNode_ = -3;
    CHECK_THROWS(cp::particle_transfer::import_logical_node(q, state, tree));
    state.environmentNode_ = -1;
    CHECK_THROWS(cp::particle_transfer::import_logical_node(q, state, tree));
    state.environmentNode_ = 999;
    CHECK_THROWS(cp::particle_transfer::import_logical_node(q, state, tree));
    CHECK(q.getNode() == node);
  }
}

TEST_CASE("Particle node transfers support separate air ice rock and firn environments",
          "[particle_transfer]") {
  using stack_type = test::TransportStack<DummyEnvironment>::type;
  using firn_type = UniformMagneticField<MediumPropertyModel<
      ExponentialFirn<IRefractiveIndexModel<DummyEnvironmentInterface>>>>;
  for (int scenario = 0; scenario < 4; ++scenario) {
    CAPTURE(scenario);
    auto [env, cs, node] = setup::testing::setup_environment(
        scenario == 2 ? get_nucleus_code(28, 14) : Code::Oxygen);
    if (scenario == 3) {
      auto firn = DummyEnvironment::createNode<SeparationPlane>(
          Plane(Point(*cs, 0_m, 0_m, 0_m), DirectionVector(*cs, {0., 0., 1.})));
      firn->setModelProperties<firn_type>(
          MagneticFieldVector(*cs, {0_T, 0_T, 0_T}), Medium::WaterIce, *cs,
          NuclearComposition({Code::Hydrogen, Code::Oxygen}, {2. / 3., 1. / 3.}),
          ExponentialFirnProfile{});
      node = firn.get();
      env->getUniverse()->addChild(std::move(firn));
    } else if (scenario == 1 || scenario == 2) {
      using medium_type = MediumPropertyModel<
          UniformMagneticField<HomogeneousMedium<DummyEnvironmentInterface>>>;
      node->setModelProperties<medium_type>(
          scenario == 1 ? Medium::WaterIce : Medium::StandardRock,
          MagneticFieldVector(*cs, {0_T, 0_T, 0_T}),
          (scenario == 1 ? 917. : 2650.) * 1_kg / cube(1_m),
          scenario == 1
              ? NuclearComposition({Code::Hydrogen, Code::Oxygen}, {2. / 3., 1. / 3.})
              : NuclearComposition({get_nucleus_code(28, 14), Code::Oxygen},
                                   {1. / 3., 2. / 3.}));
    }
    cp::EnvironmentTopology<DummyEnvironmentInterface> tree(*env->getUniverse());
    stack_type a, b;
    auto p = a.addParticle(std::make_tuple(Code::MuMinus, 1_GeV,
                                           DirectionVector(*cs, {0., 0., -1.}),
                                           Point(*cs, 0_m, 0_m, -10_m), 0_s));
    p.setNode(node);
    auto state = cp::particle_transfer::to_device_state(p, *cs, 1, 0, 0, 0, tree);
    auto q = cp::particle_transfer::import_particle(b, state, *cs, tree);
    CHECK(q.getNode() == node);
    CHECK(q.getNode()->getModelProperties().getMassDensity(q.getPosition()) ==
          node->getModelProperties().getMassDensity(p.getPosition()));
  }
}

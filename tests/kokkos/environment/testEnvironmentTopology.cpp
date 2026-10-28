/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include <corsika/kokkos/environment/EnvironmentTopology.hpp>
#include <corsika/kokkos/data/ParticleConversion.hpp>
#include <corsika/framework/stack/CombinedStack.hpp>
#include <corsika/media/Environment.hpp>
#include <corsika/media/HomogeneousMedium.hpp>
#include <corsika/media/LayeredSphericalAtmosphereBuilder.hpp>
#include <corsika/stack/GeometryNodeStackExtension.hpp>
#include <corsika/stack/VectorStack.hpp>

#include <catch2/catch_all.hpp>

#include <array>
#include <limits>
#include <memory>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

namespace {

  using namespace corsika;
  using environment_type = Environment<IMediumModel>;
  using topology_type = kokkos::EnvironmentTopology<IMediumModel>;

  auto sphere(LengthType const radius = 1_m) {
    return environment_type::createNode<Sphere>(
        Point(get_root_CoordinateSystem(), 0_m, 0_m, 0_m), radius);
  }

  void check_against_native(topology_type const& topology,
                            environment_type::BaseNodeType const& root) {
    std::vector<environment_type::BaseNodeType const*> preorder;
    root.walk([&](auto const& node) { preorder.push_back(&node); });
    REQUIRE(topology.getNodes().size() == preorder.size());
    std::size_t childOffset = 0;
    std::size_t excludedOffset = 0;
    for (std::size_t i = 0; i < preorder.size(); ++i) {
      CAPTURE(i);
      auto const* native = preorder[i];
      auto const& node = topology.getNodes()[i];
      REQUIRE(topology.getNodeIndex(*(native)) == static_cast<std::int32_t>(i));
      REQUIRE(topology.getNativeNode(static_cast<std::int32_t>(i)).get_ptr() == native);
      REQUIRE(topology.getNativeNode(node.parentIndex_).get_ptr() == native->getParent());
      REQUIRE(node.childOffset_ == childOffset);
      REQUIRE(node.excludedOffset_ == excludedOffset);
      REQUIRE(node.childCount_ == native->getChildNodes().size());
      REQUIRE(node.excludedCount_ == native->getExcludedNodes().size());
      for (auto const& child : native->getChildNodes()) {
        REQUIRE(
            topology.getNativeNode(topology.getChildren().at(childOffset++)).get_ptr() ==
            child.get());
      }
      for (auto const* excluded : native->getExcludedNodes()) {
        REQUIRE(topology.getNativeNode(topology.getExclusions().at(excludedOffset++))
                    .get_ptr() == excluded);
      }
      REQUIRE(topology.getNativeModel(node.modelIndex_).get_ptr() ==
              (native->hasModelProperties() ? &native->getModelProperties() : nullptr));
    }
    REQUIRE(topology.getChildren().size() == childOffset);
    REQUIRE(topology.getExclusions().size() == excludedOffset);
  }

  template <typename TStackIterator>
  using geometry_interface_type =
      typename node::MakeGeometryDataInterface<TStackIterator, environment_type>::type;
  template <typename TStackIterator>
  using particle_interface_type =
      CombinedParticleInterface<VectorStack::pi_type, geometry_interface_type,
                                TStackIterator>;
  using stack_type =
      CombinedStack<VectorStack::stack_data_type, node::GeometryData<environment_type>,
                    particle_interface_type>;

} // namespace

TEST_CASE("Environment topology retains root and rejects unknown indices",
          "[kokkos_environment]") {
  environment_type environment;
  topology_type topology(*environment.getUniverse());
  check_against_native(topology, *environment.getUniverse());
  REQUIRE(topology.getNodes().size() == 1);
  REQUIRE(topology.getNodes()[0].parentIndex_ == -1);
  REQUIRE(topology.getNodes()[0].modelIndex_ == -1);
  REQUIRE(topology.getNodeIndex(boost::none) == -1);
  REQUIRE(topology.getNativeNode(-1).get_ptr() == nullptr);
  REQUIRE(topology.getModelIndex(boost::none) == -1);
  REQUIRE(topology.getNativeModel(-1).get_ptr() == nullptr);
  for (auto const index : {-3, -2, 1, std::numeric_limits<std::int32_t>::max()}) {
    REQUIRE_THROWS_AS(topology.getNativeNode(index).get_ptr(), std::out_of_range);
    REQUIRE_THROWS_AS(topology.getNativeModel(index).get_ptr(), std::out_of_range);
  }
  REQUIRE_THROWS_AS(topology.getNativeModel(0).get_ptr(), std::out_of_range);
  environment_type otherEnvironment;
  REQUIRE_THROWS_AS(topology.getNodeIndex(*(otherEnvironment.getUniverse().get())),
                    std::out_of_range);
}

TEST_CASE("Environment topology preserves directed ordered exclusions and ownership",
          "[kokkos_environment]") {
  environment_type environment;
  auto outer = sphere(100_m);
  auto first = sphere(10_m);
  auto nested = sphere(1_m);
  auto second = sphere(20_m);
  auto third = sphere(30_m);
  auto const* firstPointer = first.get();
  auto const* secondPointer = second.get();
  auto const* thirdPointer = third.get();
  // Preserve the original lists even when references precede ownership insertion.
  first->excludeOverlapWith(third);
  first->excludeOverlapWith(second);
  first->excludeOverlapWith(third);
  second->excludeOverlapWith(first);
  first->addChild(std::move(nested));
  outer->addChild(std::move(first));
  outer->addChild(std::move(second));
  outer->addChild(std::move(third));
  environment.getUniverse()->addChild(std::move(outer));
  topology_type topology(*environment.getUniverse());
  check_against_native(topology, *environment.getUniverse());
  REQUIRE(topology.getNodeIndex(*(firstPointer)) == 2);
  REQUIRE(topology.getNodeIndex(*(secondPointer)) == 4);
  REQUIRE(topology.getNodeIndex(*(thirdPointer)) == 5);
  REQUIRE(topology.getChildren() == std::vector<std::int32_t>{1, 2, 4, 5, 3});
  REQUIRE(topology.getExclusions() == std::vector<std::int32_t>{5, 4, 5, 2});
  REQUIRE(topology.getNodes()[5].excludedCount_ == 0); // no invented reverse edge
}

TEST_CASE("Environment models retain pointer identity without implicit inheritance",
          "[kokkos_environment]") {
  environment_type environment;
  NuclearComposition const composition({Code::Nitrogen, Code::Oxygen}, {0.8, 0.2});
  auto a = sphere();
  auto b = sphere();
  auto c = sphere();
  auto childWithoutModel = sphere();
  auto const* childPointer = childWithoutModel.get();
  auto shared = a->setModelProperties<HomogeneousMedium<IMediumModel>>(
      1_kg / (1_m * 1_m * 1_m), composition);
  b->setModelProperties(shared);
  auto equalButDistinct = c->setModelProperties<HomogeneousMedium<IMediumModel>>(
      1_kg / (1_m * 1_m * 1_m), composition);
  a->addChild(std::move(childWithoutModel));
  environment.getUniverse()->addChild(std::move(a));
  environment.getUniverse()->addChild(std::move(b));
  environment.getUniverse()->addChild(std::move(c));
  topology_type topology(*environment.getUniverse());
  check_against_native(topology, *environment.getUniverse());
  REQUIRE(topology.getModelIndex(*(shared.get())) == 0);
  REQUIRE(topology.getModelIndex(*(equalButDistinct.get())) == 1);
  REQUIRE(topology.getNodes()[1].modelIndex_ == topology.getNodes()[3].modelIndex_);
  REQUIRE(topology.getNodes()[topology.getNodeIndex(*(childPointer))].modelIndex_ == -1);
  auto foreign = std::make_shared<HomogeneousMedium<IMediumModel>>(
      1_kg / (1_m * 1_m * 1_m), composition);
  REQUIRE_THROWS_AS(topology.getModelIndex(*(foreign.get())), std::out_of_range);
}

TEST_CASE("Invalid topology exports fail rather than drop references",
          "[kokkos_environment]") {
  environment_type environment;
  SECTION("external exclusion") {
    auto external = sphere();
    environment.getUniverse()->excludeOverlapWith(external);
    REQUIRE_THROWS_AS(topology_type(*environment.getUniverse()), std::invalid_argument);
  }
  SECTION("null exclusion") {
    environment_type::BaseNodeType::VTNUPtr nullNode;
    environment.getUniverse()->excludeOverlapWith(nullNode);
    REQUIRE_THROWS_AS(topology_type(*environment.getUniverse()), std::invalid_argument);
  }
  SECTION("partial subtree") {
    auto child = sphere();
    auto const* childPointer = child.get();
    environment.getUniverse()->addChild(std::move(child));
    REQUIRE_THROWS_AS(topology_type(*childPointer), std::invalid_argument);
  }
}

TEST_CASE("Large topology exports preserve C8 preorder and are repeatable",
          "[kokkos_environment]") {
  environment_type environment;
  for (int i = 0; i < 64; ++i) {
    auto branch = sphere();
    auto* current = branch.get();
    for (int depth = 0; depth < 8; ++depth) {
      auto child = sphere();
      auto* next = child.get();
      current->addChild(std::move(child));
      current = next;
    }
    environment.getUniverse()->addChild(std::move(branch));
  }
  topology_type first(*environment.getUniverse());
  topology_type second(*environment.getUniverse());
  auto copied = first;
  auto moved = std::move(copied);
  REQUIRE(first.getNodes().size() == 577);
  for (auto const* topology : {&first, &second, &moved}) {
    check_against_native(*topology, *environment.getUniverse());
    REQUIRE(topology->getChildren() == first.getChildren());
    REQUIRE(topology->getExclusions() == first.getExclusions());
  }
}

TEST_CASE("Actual layered atmosphere keeps every native node and model",
          "[kokkos_environment]") {
  auto const frame = get_root_CoordinateSystem();
  auto builder = make_layered_spherical_atmosphere_builder<>::create(
      Point(frame, 0_m, 0_m, 0_m), constants::EarthRadius::Mean);
  builder.setNuclearComposition({{Code::Nitrogen, Code::Oxygen}, {0.8, 0.2}});
  for (int layer = 1; layer <= 4; ++layer) {
    builder.addExponentialLayer(540.1778_g / (1_cm * 1_cm), 772170.16_cm, layer * 10_km);
  }
  builder.addLinearLayer(1_g / (1_cm * 1_cm), 1_km, 50_km);
  auto const environment = builder.assemble();
  topology_type topology(*environment.getUniverse());
  check_against_native(topology, *environment.getUniverse());
  REQUIRE(topology.getNodes().size() == 6);
  for (int i = 0; i < 6; ++i) {
    REQUIRE(topology.getNodes()[i].parentIndex_ == i - 1);
    REQUIRE(topology.getNodes()[i].modelIndex_ == i - 1);
  }
  // This is native location + identity mapping, NOT a new device location test.
  for (int altitude = 5; altitude < 60; altitude += 10) {
    auto const* native = environment.getUniverse()->getContainingNode(
        Point(frame, 0_m, 0_m, constants::EarthRadius::Mean + altitude * 1_km));
    REQUIRE(topology.getNativeNode(topology.getNodeIndex(*(native))).get_ptr() == native);
  }
}

TEST_CASE("Particle node round trip preserves values and unrelated metadata",
          "[kokkos_environment]") {
  environment_type environment;
  environment.getUniverse()->addChild(sphere());
  topology_type topology(*environment.getUniverse());
  auto const frame = get_root_CoordinateSystem();
  stack_type stack;
  auto particle = stack.addParticle(std::make_tuple(Code::MuPlus, 2_GeV,
                                                    DirectionVector(frame, {0., 0., 1.}),
                                                    Point(frame, 0_m, 0_m, 0_m), 3_ns));
  for (std::int32_t index : {-1, 0, 1}) {
    auto const* native = topology.getNativeNode(index).get_ptr();
    particle.setNode(native);
    kokkos::ParticleState state{};
    state.historyId_ = 193;
    state.parentHistoryId_ = 29;
    state.stepId_ = 17;
    state.mediumId_ = 13; // model indices are deliberately NOT assigned as media
    state.environmentNode_ = topology.getNodeIndex(
        (particle.getNode() ? topology_type::node_reference_type(*particle.getNode())
                            : boost::none));
    kokkos::export_particle_values(particle, frame, state);
    REQUIRE(state.environmentNode_ == index);
    particle.setNode(nullptr);
    kokkos::import_particle_values(state, frame, particle);
    REQUIRE(particle.getNode() == nullptr); // value conversion never does mapping
    particle.setNode(topology.getNativeNode(state.environmentNode_).get_ptr());
    REQUIRE(particle.getNode() == native);
    REQUIRE(particle.getPID() == Code::MuPlus);
    REQUIRE(particle.getKineticEnergy() / 1_GeV == Catch::Approx(2.));
    REQUIRE(state.historyId_ == 193);
    REQUIRE(state.parentHistoryId_ == 29);
    REQUIRE(state.stepId_ == 17);
    REQUIRE(state.mediumId_ == 13);
    REQUIRE(stack.getEntries() == 1);
  }
  particle.setNode(topology.getNativeNode(1).get_ptr());
  REQUIRE_THROWS_AS(particle.setNode(topology.getNativeNode(-3).get_ptr()),
                    std::out_of_range);
  REQUIRE(particle.getNode() == topology.getNativeNode(1).get_ptr());
}

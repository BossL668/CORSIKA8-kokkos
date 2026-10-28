/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include <corsika/kokkos/environment/EnvironmentSnapshotData.hpp>
#include <corsika/media/Environment.hpp>
#include <corsika/media/HomogeneousMedium.hpp>
#include <corsika/media/IMagneticFieldModel.hpp>
#include <corsika/media/IMediumPropertyModel.hpp>
#include <corsika/media/IRefractiveIndexModel.hpp>
#include <corsika/media/MediumPropertyModel.hpp>
#include <corsika/media/UniformMagneticField.hpp>
#include <corsika/media/UniformRefractiveIndex.hpp>

#include <catch2/catch_test_macros.hpp>
#include <type_traits>
#include <utility>

namespace {
  using namespace corsika;
  using model_interface_type =
      IRefractiveIndexModel<IMediumPropertyModel<IMagneticFieldModel<IMediumModel>>>;
  using environment_type = Environment<model_interface_type>;
  using snapshot_type = kokkos::EnvironmentSnapshot<model_interface_type>;
  using model_type = UniformRefractiveIndex<
      MediumPropertyModel<UniformMagneticField<HomogeneousMedium<model_interface_type>>>>;

  template <typename T, typename = void>
  struct CanBorrow : std::false_type {};
  template <typename T>
  struct CanBorrow<
      T, std::void_t<decltype(kokkos::bind_environment_data(std::declval<T>()))>>
      : std::true_type {};
  static_assert(CanBorrow<snapshot_type&>::value);
  static_assert(CanBorrow<snapshot_type const&>::value);
  static_assert(!CanBorrow<snapshot_type&&>::value);
  static_assert(!CanBorrow<snapshot_type const&&>::value);

  void check(snapshot_type const& snapshot) {
    auto const borrowed = kokkos::bind_environment_data(snapshot);
    auto const& topology = snapshot.getTopology();
    REQUIRE(borrowed.nodes_ == topology.getNodes().data());
    REQUIRE(borrowed.shapes_ == snapshot.getShapes().data());
    REQUIRE(borrowed.children_ == topology.getChildren().data());
    REQUIRE(borrowed.exclusions_ == topology.getExclusions().data());
    REQUIRE(borrowed.models_ == snapshot.getModels().data());
    REQUIRE(borrowed.composition_ == snapshot.getComposition().data());
    REQUIRE(borrowed.nodeCount_ == topology.getNodes().size());
    REQUIRE(borrowed.childCount_ == topology.getChildren().size());
    REQUIRE(borrowed.exclusionCount_ == topology.getExclusions().size());
    REQUIRE(borrowed.modelCount_ == snapshot.getModels().size());
    REQUIRE(borrowed.componentCount_ == snapshot.getComposition().size());
  }
} // namespace

TEST_CASE("Borrowed native snapshot retains the empty universe",
          "[environment][storage]") {
  environment_type environment;
  snapshot_type const snapshot(*environment.getUniverse(), get_root_CoordinateSystem());
  check(snapshot);
  auto const data = kokkos::bind_environment_data(snapshot);
  REQUIRE(data.nodeCount_ == 1);
  REQUIRE(data.modelCount_ == 0);
  REQUIRE(data.componentCount_ == 0);
  REQUIRE(data.shapes_[0].kind_ == kokkos::EnvironmentShape::Kind::Universe);
}

TEST_CASE("Native snapshot binding preserves shared models and ordered exclusions",
          "[environment][storage]") {
  environment_type environment;
  auto const frame = get_root_CoordinateSystem();
  auto first = environment_type::createNode<Sphere>(Point(frame, 0_m, 0_m, 0_m), 10_m);
  auto second = environment_type::createNode<Sphere>(Point(frame, 3_m, 0_m, 0_m), 2_m);
  auto const shared = first->setModelProperties<model_type>(
      1.000312, Medium::AirDry1Atm, MagneticFieldVector(frame, {11_uT, -22_uT, 33_uT}),
      1.25_kg / cube(1_m),
      NuclearComposition({Code::Nitrogen, Code::Oxygen}, {0.8, 0.2}));
  second->setModelProperties(shared);
  first->excludeOverlapWith(second);
  first->excludeOverlapWith(second);
  second->excludeOverlapWith(first);
  environment.getUniverse()->addChild(std::move(first));
  environment.getUniverse()->addChild(std::move(second));
  snapshot_type const snapshot(*environment.getUniverse(), frame);
  check(snapshot);
  auto const data = kokkos::bind_environment_data(snapshot);
  REQUIRE(data.nodeCount_ == 3);
  REQUIRE(data.modelCount_ == 1);
  REQUIRE(data.componentCount_ == 2);
  REQUIRE(data.exclusionCount_ == 3);
  REQUIRE(data.exclusions_[0] == 2);
  REQUIRE(data.exclusions_[1] == 2);
  REQUIRE(data.exclusions_[2] == 1);
  REQUIRE(data.nodes_[1].modelIndex_ == data.nodes_[2].modelIndex_);
  // Copying the host snapshot creates another owner; binding must use that owner's
  // arrays.
  snapshot_type const copy(snapshot);
  check(copy);
  REQUIRE(kokkos::bind_environment_data(copy).nodes_ != data.nodes_);
}

/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include "HostEnvironmentTree.hpp"
#include <corsika/kokkos/environment/EnvironmentSearch.hpp>
#include <corsika/kokkos/environment/EnvironmentTopology.hpp>
#include <corsika/kokkos/environment/ShapeConversion.hpp>
#include <corsika/kokkos/runtime/Runtime.hpp>
#include <corsika/framework/core/PhysicalConstants.hpp>
#include <corsika/framework/geometry/Box.hpp>
#include <corsika/framework/geometry/Plane.hpp>
#include <corsika/framework/geometry/SeparationPlane.hpp>
#include <corsika/framework/geometry/Sphere.hpp>
#include <corsika/media/CORSIKA7Atmospheres.hpp>
#include <corsika/media/Environment.hpp>
#include <corsika/media/HomogeneousMedium.hpp>
#include <corsika/media/Universe.hpp>

#include "EnvironmentSearchReference.hpp"

#include <catch2/catch_all.hpp>

#include <array>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

namespace {

  using namespace corsika;
  using tree_type = environment_test::HostEnvironmentTree;
  using dataset_type = environment_search_reference::Dataset;
  using environment_type = Environment<IMediumModel>;
  using topology_type = kokkos::EnvironmentTopology<IMediumModel>;

  tree_type make_view(dataset_type const& input) {
    return tree_type(input.nodes_, input.shapes_, input.children_, input.exclusions_);
  }

  void check_raw_datasets() {
    INFO("Independent fixed-production search and shape oracle");
    for (auto const& input : environment_search_reference::datasets()) {
      CAPTURE(input.name_);
      auto const view = make_view(input);
      auto const copy = view;
      REQUIRE(view.getNodeCount() == input.nodes_.size());
      for (std::size_t index = 0; index < input.nodes_.size(); ++index) {
        auto const node = static_cast<std::int32_t>(index);
        auto const& expected = input.nodes_[index];
        REQUIRE(view.getParent(node) == expected.parentIndex_);
        REQUIRE(view.getChildCount(node) == expected.childCount_);
        REQUIRE(view.getExcludedCount(node) == expected.excludedCount_);
        REQUIRE(&copy.getShape(node) == &view.getShape(node));
        for (std::uint32_t child = 0; child < expected.childCount_; ++child) {
          REQUIRE(view.getChild(node, child) ==
                  input.children_[expected.childOffset_ + child]);
        }
        for (std::uint32_t excluded = 0; excluded < expected.excludedCount_; ++excluded) {
          REQUIRE(view.getExcluded(node, excluded) ==
                  input.exclusions_[expected.excludedOffset_ + excluded]);
        }
      }
      for (auto const& query : input.queries_) {
        CAPTURE(query.start_, query.positionM_[0], query.positionM_[1],
                query.positionM_[2]);
        REQUIRE(kokkos::locate_node(view, query.positionM_, query.start_) ==
                environment_search_reference::locate(input.tree(), query.positionM_,
                                                     query.start_));
        // Non-finite primitive inputs are compared only with production. In
        // particular native Plane uses !(dot > 0), unlike production dot <= 0.
        for (auto const& shape : input.shapes_) {
          REQUIRE(kokkos::contains(shape, query.positionM_) ==
                  environment_search_reference::contains(shape, query.positionM_));
        }
      }
    }
    double const point[3]{};
    REQUIRE(kokkos::locate_node(tree_type{}, point) == -2);
    auto const fixtures = environment_search_reference::datasets();
    REQUIRE(kokkos::locate_node(make_view(fixtures[1]), point) == 601);
    REQUIRE(kokkos::locate_node(make_view(fixtures[2]), point) == -2);
    REQUIRE(kokkos::locate_node(make_view(fixtures[3]), point) == -2);
  }

  dataset_type export_tree(topology_type const& topology,
                           CoordinateSystemPtr const& frame) {
    dataset_type result;
    result.nodes_ = topology.getNodes();
    result.children_ = topology.getChildren();
    result.exclusions_ = topology.getExclusions();
    for (std::size_t index = 0; index < result.nodes_.size(); ++index) {
      result.shapes_.push_back(kokkos::export_shape(
          topology.getNativeNode(static_cast<std::int32_t>(index)).get_ptr()->getVolume(),
          frame));
    }
    return result;
  }

  void check_native_point(topology_type const& topology, tree_type const& view,
                          dataset_type const& input, Point const& point,
                          CoordinateSystemPtr const& frame, std::int32_t start) {
    auto const coordinates = point.getCoordinates(frame);
    double const position[3]{coordinates[0] / 1_m, coordinates[1] / 1_m,
                             coordinates[2] / 1_m};
    auto const* expected =
        topology.getNativeNode(start).get_ptr()->getContainingNode(point);
    CAPTURE(start, position[0], position[1], position[2]);
    REQUIRE(kokkos::locate_node(view, position, start) ==
            topology.getNodeIndex((
                expected ? topology_type::node_reference_type(*expected) : boost::none)));
    REQUIRE(kokkos::locate_node(view, position, start) ==
            environment_search_reference::locate(input.tree(), position, start));
  }

  void check_native_tree(CoordinateSystemPtr const& shapeFrame,
                         CoordinateSystemPtr const& commonFrame) {
    INFO(
        "Native recursion, child priority, directed exclusion order and model "
        "independence");
    environment_type environment;
    auto sphere = [&](LengthType radius) {
      return environment_type::createNode<Sphere>(Point(shapeFrame, 0_m, 0_m, 0_m),
                                                  radius);
    };
    auto outer = sphere(100_m);
    auto first = environment_type::createNode<Box>(shapeFrame, 10_m, 10_m, 10_m);
    auto nested = sphere(2_m);
    auto second = sphere(20_m);
    auto excludedFirst = sphere(6_m);
    auto excludedSecond = sphere(9_m);
    auto shared = outer->setModelProperties<HomogeneousMedium<IMediumModel>>(
        1_kg / (1_m * 1_m * 1_m), NuclearComposition({Code::Nitrogen}, {1.}));
    excludedFirst->setModelProperties(shared);
    first->excludeOverlapWith(excludedFirst);
    first->excludeOverlapWith(excludedSecond);
    first->excludeOverlapWith(excludedFirst);
    excludedSecond->excludeOverlapWith(excludedFirst);
    first->addChild(std::move(nested));
    outer->addChild(std::move(first));
    outer->addChild(std::move(second));
    environment.getUniverse()->addChild(std::move(outer));
    environment.getUniverse()->addChild(std::move(excludedFirst));
    environment.getUniverse()->addChild(std::move(excludedSecond));
    topology_type const topology(*environment.getUniverse());
    auto const input = export_tree(topology, commonFrame);
    auto const view = make_view(input);
    REQUIRE(input.children_ == std::vector<std::int32_t>{1, 5, 6, 2, 4, 3});
    REQUIRE(input.exclusions_ == std::vector<std::int32_t>{5, 6, 5, 5});
    REQUIRE(input.nodes_[1].modelIndex_ == 0);
    REQUIRE(input.nodes_[2].modelIndex_ == -1);
    REQUIRE(input.nodes_[5].modelIndex_ == 0);

    std::array<double, 7> const positions{0., 5., 8., 9.5, 15., 50., 120.};
    std::array<std::int32_t, 7> const expected{3, 5, 6, 2, 4, 1, 0};
    for (std::size_t index = 0; index < positions.size(); ++index) {
      Point const point(shapeFrame, positions[index] * 1_m, 0_m, 0_m);
      REQUIRE(topology.getNodeIndex(*(environment.getUniverse()->getContainingNode(
                  point))) == expected[index]);
      check_native_point(topology, view, input, point, commonFrame, 0);
    }
    // Avoid exact surfaces after frame transformations; their strict predicates
    // are independently tested in the axis-aligned primitive fixtures.
    for (int x = -20; x <= 20; ++x) {
      for (double y : {-8.37, .13, 8.63}) {
        Point const point(shapeFrame, (1.23 * x + .137) * 1_m, y * 1_m, .21_m);
        for (int start : {0, 1, 2, 5, 6}) {
          check_native_point(topology, view, input, point, commonFrame, start);
        }
      }
    }
    auto const after = export_tree(topology, commonFrame);
    REQUIRE(after.children_ == input.children_);
    REQUIRE(after.exclusions_ == input.exclusions_);
    REQUIRE_FALSE(topology.getNativeNode(2).get_ptr()->hasModelProperties());
    REQUIRE(&topology.getNativeNode(1).get_ptr()->getModelProperties() == shared.get());
  }

  void check_native_primitives(CoordinateSystemPtr const& shapeFrame,
                               CoordinateSystemPtr const& commonFrame) {
    INFO("Finite native shape predicates in translated and rotated frames");
    std::vector<std::unique_ptr<IVolume>> shapes;
    shapes.push_back(std::make_unique<Universe>(shapeFrame));
    shapes.push_back(std::make_unique<Sphere>(Point(shapeFrame, 0_m, 0_m, 0_m), 3_m));
    shapes.push_back(std::make_unique<Box>(shapeFrame, 2_m, 3_m, 4_m));
    shapes.push_back(std::make_unique<SeparationPlane>(Plane(
        Point(shapeFrame, 0_m, 0_m, 0_m), DirectionVector(shapeFrame, {1., 2., 3.}))));
    for (auto const& shape : shapes) {
      auto const numeric = kokkos::export_shape(*shape, commonFrame);
      for (double x : {-5.13, -.37, .27, 5.41}) {
        for (double y : {-2.7, .31, 2.83}) {
          Point const point(shapeFrame, x * 1_m, y * 1_m, .17_m);
          auto const coordinates = point.getCoordinates(commonFrame);
          double const position[3]{coordinates[0] / 1_m, coordinates[1] / 1_m,
                                   coordinates[2] / 1_m};
          REQUIRE(kokkos::contains(numeric, position) == shape->contains(point));
        }
      }
    }
  }

  void check_atmosphere(CoordinateSystemPtr const& shapeFrame,
                        CoordinateSystemPtr const& commonFrame) {
    INFO("Real Linsley five-layer hierarchy without material inheritance");
    environment_type environment;
    create_5layer_atmosphere<IMediumModel>(environment, AtmosphereId::LinsleyUSStd,
                                           Point(shapeFrame, 0_m, 0_m, 0_m));
    topology_type const topology(*environment.getUniverse());
    auto const input = export_tree(topology, commonFrame);
    auto const view = make_view(input);
    REQUIRE(input.nodes_.size() == 6);
    REQUIRE(input.exclusions_.empty());
    REQUIRE(input.nodes_[0].modelIndex_ == -1);
    for (double altitude : {2., 7., 25., 70., 106.4, 115.}) {
      auto const radius = constants::EarthRadius::Mean + altitude * 1_km;
      for (auto const& axis : std::array<std::array<double, 3>, 4>{
               {{1., 0., 0.}, {0., -1., 0.}, {0., 0., 1.}, {.6, 0., .8}}}) {
        Point const point(shapeFrame, axis[0] * radius, axis[1] * radius,
                          axis[2] * radius);
        check_native_point(topology, view, input, point, commonFrame, 0);
        auto const expected =
            altitude > 112.8
                ? 0
                : altitude > 100.
                      ? 1
                      : altitude > 40. ? 2 : altitude > 10. ? 3 : altitude > 4. ? 4 : 5;
        REQUIRE(topology.getNodeIndex(
                    *(environment.getUniverse()->getContainingNode(point))) == expected);
      }
    }
  }

} // namespace

TEST_CASE("Environment node search preserves production and native finite geometry",
          "[kokkos_environment_search]") {
  REQUIRE_FALSE(environment_test::is_initialized());
  {
    kokkos::Runtime runtime;
    check_raw_datasets();
    auto const root = get_root_CoordinateSystem();
    auto const shifted = make_translation(root, {1_km, -2_km, 3_km});
    auto const rotated =
        make_rotation(shifted, QuantityVector<dimensionless_d>{1., 2., 3.}, .7);
    check_native_tree(root, root);
    check_native_tree(rotated, root);
    check_native_tree(root, rotated);
    check_native_primitives(rotated, root);
    check_native_primitives(root, rotated);
    check_atmosphere(root, root);
    check_atmosphere(rotated, root);
    check_atmosphere(root, rotated);
  }
  REQUIRE_FALSE(environment_test::is_initialized());
  REQUIRE(environment_test::is_finalized());
}

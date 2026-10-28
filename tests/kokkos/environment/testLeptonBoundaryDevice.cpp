/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include <corsika/kokkos/environment/EnvironmentTreeView.hpp>
#include <corsika/kokkos/ScalarPhysicalConstants.hpp>
#include <corsika/kokkos/environment/EnvironmentLeptonBoundary.hpp>
#include <corsika/kokkos/runtime/Runtime.hpp>

#include "LeptonBoundaryTestData.hpp"

#include <Kokkos_Core.hpp>
#include <catch2/catch_all.hpp>

#include <cmath>
#include <limits>
#include <type_traits>
#include <vector>

namespace {

  namespace fixture = lepton_boundary_test;
  using execution_space_type = Kokkos::DefaultExecutionSpace;
  using memory_space_type = typename execution_space_type::memory_space;
  using tree_type = corsika::kokkos::EnvironmentTreeView<memory_space_type>;
#if defined(CORSIKA_EXPECT_CUDA)
  static_assert(std::is_same_v<execution_space_type, Kokkos::Cuda>);
#elif defined(CORSIKA_EXPECT_OPENMP)
  static_assert(std::is_same_v<execution_space_type, Kokkos::OpenMP>);
#endif

  void require_exact(double actual, double expected) {
    if (std::isnan(expected)) {
      REQUIRE(std::isnan(actual));
    } else {
      REQUIRE(actual == expected);
      if (expected == 0.) { REQUIRE(std::signbit(actual) == std::signbit(expected)); }
    }
  }

  void require_cross_device(double actual, double expected) {
    if (!std::isfinite(expected) || expected == 0.) {
      require_exact(actual, expected);
      return;
    }
    REQUIRE(std::isfinite(actual));
    REQUIRE(actual != 0.);
    REQUIRE(std::signbit(actual) == std::signbit(expected));
    REQUIRE(std::abs(actual - expected) <=
            512 * std::numeric_limits<double>::epsilon() * std::abs(expected) +
                32 * std::numeric_limits<double>::denorm_min());
  }

  void require_state(fixture::Outcome const& actual, fixture::Outcome const& expected) {
    REQUIRE(actual.nextNode_ == expected.nextNode_);
    REQUIRE(actual.failed_ == expected.failed_);
    REQUIRE(actual.unsupported_ == expected.unsupported_);
    REQUIRE(actual.incomplete_ == expected.incomplete_);
    REQUIRE(actual.freshFlagsClear_);
    REQUIRE(expected.freshFlagsClear_);
    REQUIRE(actual.wrapperAgrees_);
    REQUIRE(expected.wrapperAgrees_);
  }

  template <typename TValue>
  auto copy_to_device(std::vector<TValue> const& input) {
    Kokkos::View<TValue*, Kokkos::HostSpace> host("lepton_host_input", input.size());
    for (std::size_t index = 0; index < input.size(); ++index) {
      host(index) = input[index];
    }
    Kokkos::View<TValue*, memory_space_type> device("lepton_device_input", input.size());
    Kokkos::deep_copy(device, host);
    return device;
  }

  void check_queries(fixture::TreeKind kind) {
    auto const data = fixture::tree(kind);
    auto const nodes = copy_to_device(data.nodes_);
    auto const shapes = copy_to_device(data.shapes_);
    auto const children = copy_to_device(data.children_);
    auto const exclusions = copy_to_device(data.exclusions_);
    tree_type const tree(nodes, shapes, children, exclusions);
    auto const inputs = fixture::queries(kind);
    auto const queries = copy_to_device(inputs);
    Kokkos::View<fixture::Outcome * [2], Kokkos::LayoutRight, memory_space_type> outcomes(
        "particle_boundary_outcomes", inputs.size());
    Kokkos::View<unsigned int*, memory_space_type> unchanged("unchanged_inputs",
                                                             inputs.size());
    Kokkos::View<double*, memory_space_type> straightObservation(
        "straight_observation_witness", inputs.size());
    Kokkos::View<double[5], memory_space_type> constants("scalar_constants");
    Kokkos::parallel_for(
        "per_particle_lepton_boundaries",
        Kokkos::RangePolicy<execution_space_type>(0, inputs.size()),
        KOKKOS_LAMBDA(int index) {
          auto const query = queries(index);
          straight_boundary_reference::Tree reference;
          reference.nodes_ = nodes.data();
          reference.shapes_ = shapes.data();
          reference.children_ = children.data();
          reference.exclusions_ = exclusions.data();
          // Both accessors are created from this lane's particle inside the kernel.
          outcomes(index, 0) = fixture::actual(tree, query);
          outcomes(index, 1) = fixture::expected(reference, query);
          unchanged(index) = fixture::same_input(query, queries(index));
          if (query.category_ == fixture::Category::ForcedCurved) {
            auto const access = lepton_boundary_reference::lepton_access(
                reference, query.particle_, query.massGeV_, query.chargeNumber_,
                query.fieldT_, query.limit_);
            fixture::shape_type plane;
            plane.kind_ = fixture::shape_type::Kind::SeparationPlane;
            for (int axis = 0; axis < 3; ++axis) {
              plane.centerM_[axis] = query.pointM_[axis];
              plane.normal_[axis] = query.normal_[axis];
            }
            straightObservation(index) =
                access.intersectShape(plane).getEntry() * access.speed_;
          }
          if (index == 0) {
            constants(0) = corsika::kokkos::scalar_constants::Version;
            constants(1) = corsika::kokkos::scalar_constants::SpeedOfLightMPerS;
            constants(2) = corsika::kokkos::scalar_constants::ElementaryChargeC;
            constants(3) = corsika::kokkos::scalar_constants::VacuumPermittivityFPerM;
            constants(4) =
                corsika::kokkos::scalar_constants::MagneticRigidityGeVPerTeslaMeter;
          }
        });
    auto const host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, outcomes);
    auto const hostUnchanged =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, unchanged);
    auto const hostInputs =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, queries);
    auto const hostStraight =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, straightObservation);
    auto const hostConstants =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, constants);
    require_exact(hostConstants(0), 1.);
    require_exact(hostConstants(1), 299792458.);
    require_exact(hostConstants(2), 1.6021766208e-19);
    require_exact(hostConstants(3), 8.8541878128e-12);
    require_exact(hostConstants(4), .2997925146834389);
    straight_boundary_reference::Tree reference;
    reference.nodes_ = data.nodes_.data();
    reference.shapes_ = data.shapes_.data();
    reference.children_ = data.children_.data();
    reference.exclusions_ = data.exclusions_.data();
    for (std::size_t index = 0; index < inputs.size(); ++index) {
      auto const& query = inputs[index];
      CAPTURE(kind, index, query.category_, query.particle_.energyGeV_,
              query.chargeNumber_);
      auto const& actual = host(index, 0);
      {
        INFO("Same-device fixed-production comparison");
        auto const& expected = host(index, 1);
        require_state(actual, expected);
        require_exact(actual.speedMPerS_, expected.speedMPerS_);
        require_exact(actual.distanceM_, expected.distanceM_);
        require_exact(actual.observationM_, expected.observationM_);
        require_exact(actual.emptyObservationM_, expected.emptyObservationM_);
        require_exact(actual.emptyObservationM_, actual.observationM_);
      }
      {
        INFO("Cross-device fixed-production comparison");
        auto const expected = fixture::expected(reference, query);
        require_state(actual, expected);
        require_cross_device(actual.speedMPerS_, expected.speedMPerS_);
        require_cross_device(actual.distanceM_, expected.distanceM_);
        require_cross_device(actual.observationM_, expected.observationM_);
        require_cross_device(actual.emptyObservationM_, expected.emptyObservationM_);
      }
      REQUIRE(hostUnchanged(index) == 1);
      REQUIRE(fixture::same_input(query, hostInputs(index)));
      if (kind == fixture::TreeKind::Ordered || kind == fixture::TreeKind::Spatial) {
        REQUIRE_FALSE(actual.failed_);
      } else if (query.limit_.status_ == fixture::status_type::Success) {
        REQUIRE(actual.failed_);
        REQUIRE(actual.unsupported_ == (kind != fixture::TreeKind::Incomplete));
        REQUIRE(actual.incomplete_ == (kind != fixture::TreeKind::Unsupported));
      } else {
        REQUIRE_FALSE(actual.failed_);
        REQUIRE_FALSE(actual.unsupported_);
        REQUIRE_FALSE(actual.incomplete_);
      }
      if (query.category_ == fixture::Category::Spatial) {
        REQUIRE(actual.nextNode_ == 0);
        REQUIRE(actual.distanceM_ > 0.);
        REQUIRE(actual.distanceM_ < query.limit_.distanceM_);
        REQUIRE(std::isfinite(actual.observationM_));
      } else if (query.category_ == fixture::Category::ForcedCurved) {
        REQUIRE(std::isfinite(actual.observationM_));
        REQUIRE(std::isfinite(hostStraight(index)));
        REQUIRE(std::abs(actual.observationM_ - hostStraight(index)) >
                1024 * std::numeric_limits<double>::epsilon() *
                    std::abs(hostStraight(index)));
      }
    }
    REQUIRE(inputs.size() == (kind == fixture::TreeKind::Ordered
                                  ? 91
                                  : kind == fixture::TreeKind::Spatial ? 3 : 2));
  }

} // namespace

TEST_CASE("Device particle boundary adapters preserve production results and inputs",
          "[lepton_boundary_device]") {
  REQUIRE_FALSE(Kokkos::is_initialized());
  {
    corsika::kokkos::Runtime runtime;
    INFO("Actual execution space: " << execution_space_type::name());
    for (auto kind : {fixture::TreeKind::Ordered, fixture::TreeKind::Spatial,
                      fixture::TreeKind::Unsupported, fixture::TreeKind::Incomplete,
                      fixture::TreeKind::Both}) {
      check_queries(kind);
    }
  }
  REQUIRE_FALSE(Kokkos::is_initialized());
  REQUIRE(Kokkos::is_finalized());
}

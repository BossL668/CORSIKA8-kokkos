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
#include <corsika/framework/core/PhysicalUnits.hpp>

#include "LeptonBoundaryTestData.hpp"

#include <Kokkos_Core.hpp>
#include <catch2/catch_all.hpp>

#include <array>
#include <cmath>
#include <limits>
#include <type_traits>
#include <vector>

namespace {

  namespace fixture = lepton_boundary_test;
  using tree_type = corsika::kokkos::EnvironmentTreeView<Kokkos::HostSpace>;

  void require_exact(double actual, double expected) {
    if (std::isnan(expected)) {
      REQUIRE(std::isnan(actual));
    } else {
      REQUIRE(actual == expected);
      if (expected == 0.) { REQUIRE(std::signbit(actual) == std::signbit(expected)); }
    }
  }

  template <typename TValue>
  auto host_view(std::vector<TValue> const& input) {
    Kokkos::View<TValue*, Kokkos::HostSpace> output("lepton_host_input", input.size());
    for (std::size_t index = 0; index < input.size(); ++index) {
      output(index) = input[index];
    }
    return output;
  }

  void check_constants_and_records() {
    namespace constants = corsika::kokkos::scalar_constants;
    using namespace corsika::units::si;
    constexpr auto momentum =
        corsika::constants::c * convert_HEP_to_SI<MassType::dimension_type>(1_GeV);
    constexpr double rigidity = corsika::constants::e * tesla / momentum * meter;
    REQUIRE(constants::Version == 1);
    require_exact(constants::SpeedOfLightMPerS, corsika::constants::c / (meter / second));
    require_exact(constants::ElementaryChargeC, corsika::constants::e / coulomb);
    require_exact(constants::VacuumPermittivityFPerM,
                  corsika::constants::epsilonZero / (farad / meter));
    REQUIRE(constants::MagneticRigidityGeVPerTeslaMeter / rigidity >
            1. - 8. * std::numeric_limits<double>::epsilon());
    REQUIRE(constants::MagneticRigidityGeVPerTeslaMeter / rigidity <
            1. + 8. * std::numeric_limits<double>::epsilon());
    require_exact(constants::MagneticRigidityGeVPerTeslaMeter, .2997925146834389);
    REQUIRE(static_cast<std::uint32_t>(fixture::status_type::Success) == 0);
    REQUIRE(static_cast<std::uint32_t>(fixture::status_type::Linear) == 1);
    REQUIRE(static_cast<std::uint32_t>(fixture::status_type::InvalidInput) == 2);
    REQUIRE(static_cast<std::uint32_t>(fixture::status_type::NonFiniteResult) == 3);
    fixture::limit_type const limit;
    REQUIRE(limit.status_ == fixture::status_type::InvalidInput);
    REQUIRE(limit.reserved_ == 0);
    require_exact(limit.distanceM_, 0.);
    require_exact(limit.gyroradiusM_, 0.);
    corsika::kokkos::LeptonBoundary const boundary;
    require_exact(boundary.distanceM_, 0.);
    REQUIRE(boundary.nextNode_ == -3);
    REQUIRE_FALSE(boundary.failed_);
  }

  void require_same(fixture::Outcome const& actual, fixture::Outcome const& expected) {
    require_exact(actual.speedMPerS_, expected.speedMPerS_);
    require_exact(actual.distanceM_, expected.distanceM_);
    require_exact(actual.observationM_, expected.observationM_);
    require_exact(actual.emptyObservationM_, expected.emptyObservationM_);
    require_exact(actual.emptyObservationM_, actual.observationM_);
    REQUIRE(actual.nextNode_ == expected.nextNode_);
    REQUIRE(actual.failed_ == expected.failed_);
    REQUIRE(actual.unsupported_ == expected.unsupported_);
    REQUIRE(actual.incomplete_ == expected.incomplete_);
    REQUIRE(actual.freshFlagsClear_);
    REQUIRE(expected.freshFlagsClear_);
    REQUIRE(actual.wrapperAgrees_);
    REQUIRE(expected.wrapperAgrees_);
  }

  void check_queries(fixture::TreeKind kind, std::array<unsigned int, 18>& covered) {
    auto const data = fixture::tree(kind);
    auto const nodes = host_view(data.nodes_);
    auto const shapes = host_view(data.shapes_);
    auto const children = host_view(data.children_);
    auto const exclusions = host_view(data.exclusions_);
    tree_type const tree(nodes, shapes, children, exclusions);
    straight_boundary_reference::Tree reference;
    reference.nodes_ = data.nodes_.data();
    reference.shapes_ = data.shapes_.data();
    reference.children_ = data.children_.data();
    reference.exclusions_ = data.exclusions_.data();
    auto const queries = fixture::queries(kind);
    for (std::size_t index = 0; index < queries.size(); ++index) {
      auto const& query = queries[index];
      auto const before = query;
      CAPTURE(kind, index, query.category_, query.particle_.energyGeV_,
              query.chargeNumber_);
      auto const actual = fixture::actual(tree, query);
      auto const expected = fixture::expected(reference, query);
      require_same(actual, expected);
      REQUIRE(fixture::same_input(query, before));
      REQUIRE(query.particle_.mediumId_ == 7);
      REQUIRE(query.particle_.mediumId_ != query.particle_.environmentNode_);
      REQUIRE(std::isfinite(actual.speedMPerS_));
      REQUIRE(actual.speedMPerS_ > 0.);
      REQUIRE(actual.speedMPerS_ <= lepton_boundary_reference::SpeedOfLightMPerS);
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
        for (int axis = 0; axis < 3; ++axis) {
          REQUIRE(query.particle_.positionM_[axis] != 0.);
          REQUIRE(query.particle_.direction_[axis] != 0.);
          REQUIRE(query.fieldT_[axis] != 0.);
          REQUIRE(query.normal_[axis] != 0.);
        }
      } else if (query.category_ == fixture::Category::TimeTie) {
        auto const access = lepton_boundary_reference::lepton_access(
            reference, query.particle_, query.massGeV_, query.chargeNumber_,
            query.fieldT_, query.limit_);
        auto const child = access.intersect(2);
        auto const excluded = access.intersect(3);
        require_exact(child.getEntry(), query.limit_.distanceM_ / access.speed_);
        require_exact(excluded.getEntry(), child.getEntry());
        REQUIRE(actual.nextNode_ == 3);
        REQUIRE(actual.nextNode_ != query.particle_.mediumId_);
      } else if (query.category_ == fixture::Category::Outside) {
        REQUIRE(actual.nextNode_ == 1);
      } else if (query.category_ == fixture::Category::NearRest) {
        REQUIRE(query.particle_.energyGeV_ > query.massGeV_);
        REQUIRE(actual.speedMPerS_ < 10.);
      } else if (query.category_ == fixture::Category::Mass) {
        double const momentum = query.massGeV_ == 3. ? 4. : 3.;
        require_exact(actual.speedMPerS_,
                      lepton_boundary_reference::SpeedOfLightMPerS * momentum / 5.);
        // The two masses change speed but not the geometrical limiting distance;
        // retain the rounding from the original division and multiplication.
        REQUIRE(std::abs(actual.distanceM_ - query.limit_.distanceM_) <=
                2 * std::numeric_limits<double>::epsilon() * query.limit_.distanceM_);
      } else if (query.category_ == fixture::Category::ObservationZero ||
                 query.category_ == fixture::Category::ObservationBehind ||
                 query.category_ == fixture::Category::ObservationParallel ||
                 query.category_ == fixture::Category::NoRealPlaneRoots) {
        REQUIRE(std::isinf(actual.observationM_));
        REQUIRE(actual.observationM_ > 0.);
      } else if (query.category_ == fixture::Category::ObservationBeyond) {
        REQUIRE(actual.observationM_ > query.limit_.distanceM_);
      } else if (query.category_ == fixture::Category::ObservationEqual) {
        require_exact(actual.observationM_, query.limit_.distanceM_);
      } else if (query.category_ == fixture::Category::ForcedCurved) {
        REQUIRE(query.limit_.status_ == fixture::status_type::Linear);
        REQUIRE(std::isfinite(query.limit_.gyroradiusM_));
        REQUIRE(query.limit_.gyroradiusM_ > 1.e9);
        auto const access = lepton_boundary_reference::lepton_access(
            reference, query.particle_, query.massGeV_, query.chargeNumber_,
            query.fieldT_, query.limit_);
        fixture::shape_type plane;
        plane.kind_ = fixture::shape_type::Kind::SeparationPlane;
        for (int axis = 0; axis < 3; ++axis) {
          plane.centerM_[axis] = query.pointM_[axis];
          plane.normal_[axis] = query.normal_[axis];
        }
        auto const linear = access.intersectShape(plane);
        double const linearDistance = linear.getEntry() * access.speed_;
        REQUIRE(std::isfinite(actual.observationM_));
        REQUIRE(std::isfinite(linearDistance));
        REQUIRE(std::abs(actual.observationM_ - linearDistance) >
                1024 * std::numeric_limits<double>::epsilon() * std::abs(linearDistance));
      }
      ++covered[static_cast<unsigned int>(query.category_)];
    }
    REQUIRE(queries.size() == (kind == fixture::TreeKind::Ordered
                                   ? 91
                                   : kind == fixture::TreeKind::Spatial ? 3 : 2));
  }

} // namespace

TEST_CASE(
    "Particle boundary adaptation preserves production units and observation semantics",
    "[lepton_boundary]") {
  REQUIRE_FALSE(Kokkos::is_initialized());
  check_constants_and_records();
  {
    corsika::kokkos::Runtime runtime;
    std::array<unsigned int, 18> covered{};
    for (auto kind : {fixture::TreeKind::Ordered, fixture::TreeKind::Spatial,
                      fixture::TreeKind::Unsupported, fixture::TreeKind::Incomplete,
                      fixture::TreeKind::Both}) {
      check_queries(kind, covered);
    }
    REQUIRE(covered[static_cast<unsigned int>(fixture::Category::Scan)] == 72);
    REQUIRE(covered[static_cast<unsigned int>(fixture::Category::Diagnostics)] == 6);
    REQUIRE(covered[static_cast<unsigned int>(fixture::Category::Spatial)] == 3);
    for (auto count : covered) { REQUIRE(count > 0); }
  }
  REQUIRE_FALSE(Kokkos::is_initialized());
  REQUIRE(Kokkos::is_finalized());
}

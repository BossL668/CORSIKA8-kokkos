/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include <corsika/kokkos/environment/EnvironmentTreeView.hpp>
#include <corsika/kokkos/environment/CurvedBoundaryAccess.hpp>
#include <corsika/kokkos/runtime/Runtime.hpp>

#include "CurvedBoundaryTestData.hpp"

#include <Kokkos_Core.hpp>
#include <catch2/catch_all.hpp>

#include <array>
#include <cmath>
#include <limits>
#include <vector>

namespace {

  namespace fixture = curved_boundary_test;
  using tree_type = corsika::kokkos::EnvironmentTreeView<Kokkos::HostSpace>;
  using access_type = corsika::kokkos::CurvedBoundaryAccess<
      corsika::kokkos::EnvironmentTreeView<Kokkos::HostSpace>>;

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
    Kokkos::View<TValue*, Kokkos::HostSpace> result("curved_host_input", input.size());
    for (std::size_t index = 0; index < input.size(); ++index) {
      result(index) = input[index];
    }
    return result;
  }

  access_type bind(tree_type const& tree, fixture::Ray const& ray) {
    return access_type(tree, ray.positionM_, ray.velocityMPerS_, ray.direction_,
                       ray.fieldT_, ray.speedMPerS_, ray.momentumKgMPerS_, ray.chargeC_,
                       ray.gyroradiusM_, ray.linear_);
  }

  void require_flags(access_type const& actual,
                     curved_boundary_reference::Access const& reference) {
    REQUIRE(actual.hasUnsupportedShape() == reference.unsupportedShape_);
    REQUIRE(actual.hasIncompleteRoots() == reference.incompleteRoots_);
  }

  void check_shapes() {
    auto const queries = fixture::queries();
    tree_type const empty;
    std::array<unsigned int, 11> covered{};
    for (std::size_t index = 0; index < queries.size(); ++index) {
      auto const& query = queries[index];
      CAPTURE(index, query.category_, query.shape_.kind_);
      auto actual = bind(empty, query.ray_);
      auto expected = fixture::reference(query.ray_);
      REQUIRE_FALSE(actual.hasUnsupportedShape());
      REQUIRE_FALSE(actual.hasIncompleteRoots());
      auto const interval = actual.intersectShape(query.shape_);
      auto const comparison = expected.intersectShape(query.shape_);
      REQUIRE(interval.hasIntersections() == comparison.hasIntersections());
      require_exact(interval.getEntry(), comparison.getEntry());
      require_exact(interval.getExit(), comparison.getExit());
      require_flags(actual, expected);
      if (query.valid_ >= 0) {
        REQUIRE(interval.hasIntersections() == (query.valid_ != 0));
      }
      if (query.checkFlags_) {
        REQUIRE(actual.hasUnsupportedShape() == query.unsupported_);
        REQUIRE(actual.hasIncompleteRoots() == query.incomplete_);
      }
      if (query.knownEntry_) {
        if (!std::isfinite(query.entry_) || query.entry_ == 0.) {
          require_exact(interval.getEntry(), query.entry_);
        } else {
          REQUIRE(std::abs(interval.getEntry() - query.entry_) <=
                  16 * std::numeric_limits<double>::epsilon() * std::abs(query.entry_));
        }
      }
      // A harmless second intersection must not erase the accumulated failure.
      auto const later = actual.intersectShape(fixture::universe());
      auto const laterExpected = expected.intersectShape(fixture::universe());
      REQUIRE(later.hasIntersections() == laterExpected.hasIntersections());
      require_exact(later.getEntry(), laterExpected.getEntry());
      require_exact(later.getExit(), laterExpected.getExit());
      require_flags(actual, expected);
      auto const fresh = bind(empty, query.ray_);
      REQUIRE_FALSE(fresh.hasUnsupportedShape());
      REQUIRE_FALSE(fresh.hasIncompleteRoots());
      if (query.category_ == fixture::Category::EmptyRoots ||
          query.category_ == fixture::Category::IncompleteRoots ||
          query.category_ == fixture::Category::NonfiniteRoots) {
        auto const coefficients = fixture::sphere_polynomial(query);
        REQUIRE(std::isfinite(coefficients.quadratic_));
        REQUIRE(std::isfinite(coefficients.linear_));
        REQUIRE(std::isfinite(coefficients.constant_));
        auto const& ray = query.ray_;
        REQUIRE(std::isfinite(ray.speedMPerS_));
        REQUIRE(std::isfinite(ray.momentumKgMPerS_));
        REQUIRE(std::isfinite(ray.chargeC_));
        REQUIRE(std::isfinite(ray.gyroradiusM_));
        for (int axis = 0; axis < 3; ++axis) {
          REQUIRE(std::isfinite(ray.positionM_[axis]));
          REQUIRE(std::isfinite(ray.direction_[axis]));
          REQUIRE(std::isfinite(ray.fieldT_[axis]));
        }
        require_exact(curved_boundary_reference::dot(ray.direction_, ray.direction_), 1.);
        require_exact(ray.momentumKgMPerS_ / std::abs(ray.chargeC_), ray.gyroradiusM_);
        // All three directed fixtures have |B_perp|=1 tesla.
        require_exact(curved_boundary_reference::dot(ray.fieldT_, ray.fieldT_), 1.);
        require_exact(curved_boundary_reference::dot(ray.fieldT_, ray.direction_), 0.);
        double const distance = std::abs(
            std::sqrt(curved_boundary_reference::dot(ray.positionM_, ray.positionM_)) -
            query.shape_.radiusM_);
        REQUIRE(distance <= ray.gyroradiusM_);
        REQUIRE(ray.gyroradiusM_ <= 1000 * distance);
        auto const roots = polynomial_roots_reference::stableQuartic(
            coefficients.quadratic_, coefficients.linear_, coefficients.constant_);
        if (query.category_ == fixture::Category::EmptyRoots) {
          REQUIRE(roots.size == 0);
        } else if (query.category_ == fixture::Category::IncompleteRoots) {
          REQUIRE(roots.size == 2);
          REQUIRE(roots.values[0] < 0.);
          REQUIRE(roots.values[1] > 0.);
        } else {
          REQUIRE(coefficients.linear_ != 0.);
          REQUIRE(roots.size == 1);
          REQUIRE(std::isinf(roots.values[0]));
          REQUIRE(roots.values[0] > 0.);
        }
      }
      ++covered[static_cast<unsigned int>(query.category_)];
    }
    REQUIRE(covered[static_cast<unsigned int>(fixture::Category::Linear)] == 520);
    REQUIRE(covered[static_cast<unsigned int>(fixture::Category::Earth)] == 16);
    for (auto count : covered) { REQUIRE(count > 0); }
    REQUIRE(queries.size() == 558);

    // Independent algebraic evidence for the three distinct failure outcomes.
    auto const emptyRoots = polynomial_roots_reference::stableQuartic(-.4, 0., .84);
    REQUIRE(emptyRoots.size == 0);
    auto const tangent = polynomial_roots_reference::stableQuartic(-1.5, 0., .5625);
    REQUIRE(tangent.size == 2);
    REQUIRE(tangent.values[0] < 0.);
    REQUIRE(tangent.values[1] > 0.);
    auto const nonfinite =
        polynomial_roots_reference::stableQuartic(2.e150, 8.e-110, -3.e300);
    REQUIRE(nonfinite.size == 1);
    REQUIRE(std::isinf(nonfinite.values[0]));
    REQUIRE(nonfinite.values[0] > 0.);
  }

  void check_tree(bool failures) {
    auto const data = failures ? fixture::failing_tree() : fixture::ordered_tree();
    auto const nodes = host_view(data.nodes_);
    auto const shapes = host_view(data.shapes_);
    auto const children = host_view(data.children_);
    auto const exclusions = host_view(data.exclusions_);
    tree_type const tree(nodes, shapes, children, exclusions);
    auto const queries = curved_boundary_test::search_queries(failures);
    unsigned int stickyFailures = 0;
    for (std::size_t index = 0; index < queries.size(); ++index) {
      auto const& query = queries[index];
      CAPTURE(failures, index, query.ray_.linear_, query.firstNode_, query.secondNode_);
      auto actual = bind(tree, query.ray_);
      auto expected = curved_boundary_test::reference(query.ray_);
      expected.nodes_ = data.nodes_.data();
      expected.shapes_ = data.shapes_.data();
      expected.children_ = data.children_.data();
      expected.exclusions_ = data.exclusions_.data();
      for (int search = 0; search < 2; ++search) {
        auto const current = search ? query.secondNode_ : query.firstNode_;
        auto const limit = search ? query.secondLimit_ : query.firstLimit_;
        auto const result =
            corsika::kokkos::next_environment_boundary(actual, current, limit);
        auto const reference = boundary_reference::next(expected, current, limit);
        REQUIRE(result.getNode() == reference.node_);
        require_exact(result.getParameter(), reference.parameter_);
        require_flags(actual, expected);
        if (failures && !query.ray_.linear_) {
          REQUIRE(actual.hasUnsupportedShape());
          REQUIRE(actual.hasIncompleteRoots());
          ++stickyFailures;
        }
      }
      auto const fresh = bind(tree, query.ray_);
      REQUIRE_FALSE(fresh.hasUnsupportedShape());
      REQUIRE_FALSE(fresh.hasIncompleteRoots());
    }
    REQUIRE(queries.size() == (failures ? 64 : 300));
    REQUIRE(stickyFailures == (failures ? 64 : 0));
    if (!failures) {
      curved_boundary_test::Ray ray;
      ray.linear_ = true;
      auto access = bind(tree, ray);
      auto const selected =
          corsika::kokkos::next_environment_boundary(access, std::int32_t{1}, 100.);
      REQUIRE(selected.getNode() ==
              4); // Last repeated exclusion wins identical entry times.
      require_exact(selected.getParameter(), 5.);
    }
  }

} // namespace

TEST_CASE("Curved boundaries preserve production intersections and sticky diagnostics",
          "[curved_boundary]") {
  REQUIRE_FALSE(Kokkos::is_initialized());
  {
    corsika::kokkos::Runtime runtime;
    check_shapes();
    check_tree(false);
    check_tree(true);
  }
  REQUIRE_FALSE(Kokkos::is_initialized());
  REQUIRE(Kokkos::is_finalized());
}

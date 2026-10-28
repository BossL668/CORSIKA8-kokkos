/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include "MoliereQueryTestData.hpp"
#include "MoliereTestUtilities.hpp"
#include <catch2/catch_test_macros.hpp>
#include <type_traits>

TEST_CASE("Moliere records and interpolation views retain their contracts",
          "[kokkos][moliere][contracts]") {
  using namespace corsika::kokkos;
  namespace fixture = corsika::kokkos::test;
  namespace old = fixture::moliere_reference;
  static_assert(std::is_standard_layout_v<MoliereAngleResult>);
  static_assert(std::is_trivially_copyable_v<MoliereDirectionResult>);
  static_assert(sizeof(MoliereAngleResult) == sizeof(old::MoliereAngleResult));
  static_assert(offsetof(MoliereAngleResult, status_) ==
                offsetof(old::MoliereAngleResult, status));
  static_assert(offsetof(MoliereAngleResult, iterations_) ==
                offsetof(old::MoliereAngleResult, iterations));
  static_assert(offsetof(MoliereAngleResult, angleRad_) ==
                offsetof(old::MoliereAngleResult, angle_rad));
  static_assert(sizeof(MoliereDirectionResult) == sizeof(old::MoliereDirectionResult));
  static_assert(offsetof(MoliereDirectionResult, direction_) ==
                offsetof(old::MoliereDirectionResult, direction));
  static_assert(sizeof(MoliereInterpolationView) ==
                sizeof(old::MoliereInterpolationView));
  REQUIRE(MoliereAngleResult{}.status_ == MoliereStatus::InvalidSnapshot);
  REQUIRE(MoliereDirectionResult{}.status_ == MoliereStatus::InvalidInput);
  REQUIRE(MoliereAngleResult{}.iterations_ == 0);
  REQUIRE(MoliereAngleResult{}.angleRad_ == 0.);
  MoliereInterpolationView empty;
  REQUIRE_FALSE(empty.getF1().hasStorage());
  REQUIRE_FALSE(empty.getF2().hasStorage());
  REQUIRE_FALSE(empty.getIntegralF1().hasStorage());
  REQUIRE_FALSE(empty.getIntegralF2().hasStorage());
  REQUIRE_FALSE(empty.getInitialGuessDelta().hasStorage());
  REQUIRE(empty.getIntervalCount() == 0);
  REQUIRE_FALSE(moliere_detail::has_interpolation_view(empty));
  REQUIRE_FALSE(moliere_detail::valid_interpolation_view(empty));
  MoliereInterpolationTable table;
  auto const current = make_moliere_interpolation_view(
      NativeArrayView<MoliereCubicPolynomial>{table.polynomials_},
      NativeArrayView<double>{table.initialGuessDelta_});
  REQUIRE(moliere_detail::has_interpolation_view(current));
  REQUIRE(moliere_detail::valid_interpolation_view(current));
  REQUIRE(current.getIntervalCount() == MoliereInterpolationIntervalCount);
  REQUIRE(current.getReserved() == 0);
  REQUIRE(current.getInitialGuessBNodeCount() == MoliereInitialGuessBNodeCount);
  REQUIRE(current.getInitialGuessBetaSquaredNodeCount() ==
          MoliereInitialGuessBetaSquaredNodeCount);
  REQUIRE(current.getInitialGuessCoordinateNodeCount() ==
          MoliereInitialGuessCoordinateNodeCount);
  REQUIRE(current.getF1().getSize() == MoliereInterpolationIntervalCount);
  REQUIRE(current.getF2().getSize() == MoliereInterpolationIntervalCount);
  REQUIRE(current.getInitialGuessDelta().getSize() == table.initialGuessDelta_.size());
  for (unsigned index = 0; index < 7; ++index) {
    auto const view = fixture::moliere_view_variant(current, index);
    REQUIRE(moliere_detail::valid_interpolation_view(view) == (index == 1 || index == 2));
  }
  auto const snapshot = fixture::small_moliere_snapshot();
  REQUIRE(valid_moliere_snapshot(snapshot));
  REQUIRE(
      old::moliere_detail::validSnapshot(fixture::moliere_reference_snapshot(snapshot)));
  REQUIRE(moliere_detail::solve_moliere_b(
              std::nextafter(MoliereMinimumBEquationRightHandSide, 0.)) == 0.);
  // The retained Halley result rounds below 4.5 at the exact nominal threshold;
  // the final production >=4.5 guard decides whether it is accepted.
  REQUIRE(moliere_detail::solve_moliere_b(MoliereMinimumBEquationRightHandSide) ==
          old::moliere_detail::solveMoliereB(MoliereMinimumBEquationRightHandSide));
  REQUIRE(moliere_detail::solve_moliere_b(MoliereMinimumBEquationRightHandSide + 1.e-9) >=
          MoliereMinimumB);
  REQUIRE(moliere_detail::normal_ppf(.5) == 0.);
  double const negativeAxis[3]{-1., 0., 0.};
  REQUIRE(apply_moliere_direction(negativeAxis, .1, .37).status_ ==
          MoliereStatus::InvalidInput);
}

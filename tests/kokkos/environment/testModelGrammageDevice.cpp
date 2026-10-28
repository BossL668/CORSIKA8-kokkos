/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include <corsika/kokkos/environment/ModelGrammage.hpp>
#include <corsika/kokkos/runtime/Runtime.hpp>

#include "ModelGrammageReference.hpp"

#include <Kokkos_Core.hpp>
#include <catch2/catch_all.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <type_traits>

namespace {

  using execution_space_type = Kokkos::DefaultExecutionSpace;
  using memory_space_type = typename execution_space_type::memory_space;
  using query_type = model_grammage_reference::Query;
#if defined(CORSIKA_EXPECT_CUDA)
  static_assert(std::is_same_v<execution_space_type, Kokkos::Cuda>);
#elif defined(CORSIKA_EXPECT_OPENMP)
  static_assert(std::is_same_v<execution_space_type, Kokkos::OpenMP>);
#endif

  void require_exact(double const actual, double const expected) {
    if (std::isnan(expected)) {
      REQUIRE(std::isnan(actual));
    } else {
      REQUIRE(actual == expected);
      if (expected == 0.) { REQUIRE(std::signbit(actual) == std::signbit(expected)); }
    }
  }

  void require_cross_device(double const actual, double const expected) {
    if (!std::isfinite(expected)) {
      require_exact(actual, expected);
      return;
    }
    REQUIRE(std::isfinite(actual));
    double const relative = 64 * std::numeric_limits<double>::epsilon();
    double const absolute = 32 * std::numeric_limits<double>::denorm_min();
    REQUIRE(std::abs(actual - expected) <=
            absolute + relative * std::max(std::abs(actual), std::abs(expected)));
    if (actual == 0. && expected == 0.) {
      REQUIRE(std::signbit(actual) == std::signbit(expected));
    }
  }

} // namespace

TEST_CASE("Device path laws preserve fixed production and match host within roundoff",
          "[model_grammage_device]") {
  REQUIRE_FALSE(Kokkos::is_initialized());
  {
    corsika::kokkos::Runtime runtime;
    INFO("Actual execution space: " << execution_space_type::name());
    auto const inputs = model_grammage_reference::queries();
    int const count = static_cast<int>(inputs.size());
    Kokkos::View<query_type*, Kokkos::HostSpace> hostInputs("path_inputs", count);
    for (int index = 0; index < count; ++index) { hostInputs(index) = inputs[index]; }
    Kokkos::View<query_type*, memory_space_type> queries("path_queries", count);
    Kokkos::View<double * [7], Kokkos::LayoutRight, memory_space_type> results(
        "path_values", count);
    Kokkos::deep_copy(queries, hostInputs);
    Kokkos::parallel_for(
        "model_path_values", Kokkos::RangePolicy<execution_space_type>(0, count),
        KOKKOS_LAMBDA(int const index) {
          auto const query = queries(index);
          results(index, 0) = corsika::kokkos::grammage(query.model_, query.positionM_,
                                                        query.direction_, query.lengthM_);
          results(index, 1) = model_grammage_reference::grammage(
              query.model_, query.positionM_, query.direction_, query.lengthM_);
          results(index, 2) = corsika::kokkos::arclength(
              query.model_, query.positionM_, query.direction_, query.grammageGPerCm2_);
          results(index, 3) = model_grammage_reference::arclength(
              query.model_, query.positionM_, query.direction_, query.grammageGPerCm2_);
          double const rho =
              model_evaluation_reference::density(query.model_, query.positionM_);
          double const projection = model_grammage_reference::projected_axis(
              query.model_, query.positionM_, query.direction_);
          results(index, 4) = rho;
          results(index, 5) = projection;
          results(index, 6) = query.grammageGPerCm2_ * .01 *
                              query.model_.inverseScaleLengthPerM_ * projection / rho;
        });
    Kokkos::fence("path evaluation complete");
    auto const hostResults =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, results);
    for (int index = 0; index < count; ++index) {
      auto const& query = inputs[index];
      double const hostRho =
          model_evaluation_reference::density(query.model_, query.positionM_);
      double const hostProjection = model_grammage_reference::projected_axis(
          query.model_, query.positionM_, query.direction_);
      double const hostLogArgument = query.grammageGPerCm2_ * .01 *
                                     query.model_.inverseScaleLengthPerM_ *
                                     hostProjection / hostRho;
      CAPTURE(index, query.lengthM_, query.grammageGPerCm2_, hostRho, hostProjection,
              hostLogArgument, hostResults(index, 4), hostResults(index, 5),
              hostResults(index, 6));
      // Exact within each backend is the migration gate. Diagnostics expose
      // log1p conditioning rather than concealing a boundary discrepancy.
      require_exact(hostResults(index, 0), hostResults(index, 1));
      require_exact(hostResults(index, 2), hostResults(index, 3));
      require_cross_device(hostResults(index, 0), model_grammage_reference::grammage(
                                                      query.model_, query.positionM_,
                                                      query.direction_, query.lengthM_));
      require_cross_device(
          hostResults(index, 2),
          model_grammage_reference::arclength(query.model_, query.positionM_,
                                              query.direction_, query.grammageGPerCm2_));
    }
  }
  REQUIRE_FALSE(Kokkos::is_initialized());
  REQUIRE(Kokkos::is_finalized());
}

/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include <corsika/kokkos/environment/ModelEvaluation.hpp>
#include <corsika/kokkos/runtime/Runtime.hpp>

#include "ModelEvaluationReference.hpp"

#include <Kokkos_Core.hpp>
#include <catch2/catch_all.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <type_traits>

namespace {

  using execution_space_type = Kokkos::DefaultExecutionSpace;
  using memory_space_type = typename execution_space_type::memory_space;
  using model_type = corsika::kokkos::EnvironmentModel;
  using query_type = model_evaluation_reference::Query;
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

TEST_CASE("Device point model evaluation preserves production and agrees across devices",
          "[model_evaluation_device]") {
  REQUIRE_FALSE(Kokkos::is_initialized());
  {
    corsika::kokkos::Runtime runtime;
    INFO("Actual execution space: " << execution_space_type::name());
    auto const inputs = model_evaluation_reference::queries();
    auto const count = static_cast<int>(inputs.size());
    Kokkos::View<query_type*, Kokkos::HostSpace> hostInputs("point_model_inputs", count);
    for (int index = 0; index < count; ++index) { hostInputs(index) = inputs[index]; }
    Kokkos::View<query_type*, memory_space_type> queries("point_models", count);
    Kokkos::View<double * [7], Kokkos::LayoutRight, memory_space_type> results(
        "point_values", count);
    Kokkos::View<double * [2], Kokkos::LayoutRight, memory_space_type> shortCircuits(
        "constant_index", 1);
    Kokkos::deep_copy(queries, hostInputs);
    Kokkos::parallel_for(
        "model_point_values", Kokkos::RangePolicy<execution_space_type>(0, count),
        KOKKOS_LAMBDA(int const index) {
          auto const query = queries(index);
          results(index, 0) = corsika::kokkos::density(query.model_, query.positionM_);
          results(index, 1) =
              model_evaluation_reference::density(query.model_, query.positionM_);
          results(index, 2) =
              corsika::kokkos::refractive_index(query.model_, query.positionM_);
          results(index, 3) = model_evaluation_reference::refractive_index(
              query.model_, query.positionM_);
          for (int component = 0; component < 3; ++component) {
            results(index, 4 + component) = query.model_.magneticFieldT_[component];
          }
          if (index == 0) {
            model_type constant;
            double const nan = model_evaluation_reference::not_a_number();
            double const position[3]{nan, nan, nan};
            // No valid density is needed for these two index branches.
            constant.refractivityKind_ = model_type::Refractivity::Uniform;
            constant.refractivityValue_ = -2.;
            shortCircuits(0, 0) = corsika::kokkos::refractive_index(constant, position);
            constant.refractivityKind_ = model_type::Refractivity::Unavailable;
            shortCircuits(0, 1) = corsika::kokkos::refractive_index(constant, position);
          }
        });
    Kokkos::fence("point evaluation complete");
    auto const hostResults =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, results);
    auto const hostShortCircuits =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, shortCircuits);
    require_exact(hostShortCircuits(0, 0), -2.);
    REQUIRE(std::isnan(hostShortCircuits(0, 1)));
    for (int index = 0; index < count; ++index) {
      CAPTURE(index);
      // New and frozen implementations execute on the SAME device: exact
      // finite values, signed zeros and exceptional-value classification.
      require_exact(hostResults(index, 0), hostResults(index, 1));
      require_exact(hostResults(index, 2), hostResults(index, 3));
      auto const& query = inputs[index];
      // Host versus device libm may round differently. Only this comparison
      // allows a tolerance; it does not weaken the same-device production test.
      require_cross_device(hostResults(index, 0), model_evaluation_reference::density(
                                                      query.model_, query.positionM_));
      require_cross_device(
          hostResults(index, 2),
          model_evaluation_reference::refractive_index(query.model_, query.positionM_));
      for (int component = 0; component < 3; ++component) {
        require_exact(hostResults(index, 4 + component),
                      query.model_.magneticFieldT_[component]);
      }
    }
  }
  REQUIRE_FALSE(Kokkos::is_initialized());
  REQUIRE(Kokkos::is_finalized());
}

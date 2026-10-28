/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */
#ifdef CORSIKA_PRODUCTION_STEPPER_REFERENCE
#include "ProductionScalarCascadeStepper.hpp"
#endif
#include "ScalarStepFixture.hpp"
#include <catch2/catch_test_macros.hpp>
namespace fixture = scalar_fixture;
namespace cp = corsika::kokkos;
#ifdef CORSIKA_PRODUCTION_STEPPER_REFERENCE
namespace reference = corsika::production_stepper_reference;
#endif

TEST_CASE("Scalar stepping matches production and native Cascade branches",
          "[scalar_stepper]") {
  corsika::logging::set_level(corsika::logging::level::off);
  for (int mode = 0; mode <= static_cast<int>(fixture::Mode::TiedContinuousGeometry);
       ++mode) {
    for (std::uint64_t seed = 1; seed <= 16; ++seed) {
      CAPTURE(mode, seed);
      auto kind = static_cast<fixture::Mode>(mode);
      auto current = fixture::run_stepper<cp::ScalarCascadeStepper>(kind, seed);
      CHECK(current.threw_ == (kind == fixture::Mode::SelfDecay));
      auto count = [&](char const* name) {
        return std::count(current.calls_.begin(), current.calls_.end(), name);
      };
      CHECK(count("interaction") == ((mode == 3 || mode == 4 || mode == 7) ? 1 : 0));
      CHECK(count("decay") ==
            ((mode == 5 || mode == 6 || mode == 8 || mode == 9) ? 1 : 0));
      CHECK(count("secondaries") ==
            ((mode == 3 || mode == 5 || mode == 7 || mode == 8) ? 1 : 0));
      CHECK(count("boundary") == ((mode == 2) ? 1 : 0));
      CHECK(count("continuous") == ((mode == 7 || mode == 8) ? 0 : 1));
#ifdef CORSIKA_PRODUCTION_STEPPER_REFERENCE
      auto production = fixture::run_stepper<reference::ScalarCascadeStepper>(kind, seed);
#endif
      auto native = fixture::run_native(kind, seed);
      auto timed = fixture::run_stepper<cp::ScalarCascadeStepper>(kind, seed, true);
      std::vector<fixture::Trace const*> comparisons{&native, &timed};
#ifdef CORSIKA_PRODUCTION_STEPPER_REFERENCE
      comparisons.push_back(&production);
#endif
      for (auto const* other : comparisons) {
        CHECK(current.calls_ == other->calls_);
        CHECK(current.arguments_ == other->arguments_);
        CHECK(current.particles_ == other->particles_);
        CHECK(current.rng_ == other->rng_);
        CHECK(current.threw_ == other->threw_);
      }
    }
  }
}

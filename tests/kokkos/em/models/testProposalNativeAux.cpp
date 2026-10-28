/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include "AuxiliaryModelFixture.hpp"
#include <catch2/catch_test_macros.hpp>

TEST_CASE("Live helpers preserve auxiliary export and version-four cache semantics",
          "[kokkos][proposal][auxiliary]") {
  auto const audit = corsika::kokkos::test::audit_auxiliary_models();
  REQUIRE(audit.checks_.size() >= 30);
  REQUIRE(audit.corruptionCases_ >= 10);
  for (auto const& check : audit.checks_) {
    INFO(check.first);
    REQUIRE(check.second);
  }
}

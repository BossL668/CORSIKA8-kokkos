/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include <corsika/kokkos/environment/UniformMagneticField.hpp>
#include <corsika/kokkos/runtime/Runtime.hpp>
#include "UniformMagneticFieldTestData.hpp"
#include <Kokkos_Core.hpp>
#include <catch2/catch_all.hpp>
#include <array>
#include <cmath>
#include <limits>

namespace {
  namespace fixture = uniform_magnetic_test;

  void require_exact(double actual, double expected) {
    if (std::isnan(expected)) {
      REQUIRE(std::isnan(actual));
    } else {
      REQUIRE(actual == expected);
      if (expected == 0.) { REQUIRE(std::signbit(actual) == std::signbit(expected)); }
    }
  }

  void require_particle_exact(fixture::particle_type const& actual,
                              fixture::particle_type const& expected) {
    REQUIRE(actual.pid_ == expected.pid_);
    REQUIRE(actual.mediumId_ == expected.mediumId_);
    REQUIRE(actual.generation_ == expected.generation_);
    REQUIRE(actual.reserved_ == expected.reserved_);
    require_exact(actual.energyGeV_, expected.energyGeV_);
    require_exact(actual.timeS_, expected.timeS_);
    require_exact(actual.weight_, expected.weight_);
    REQUIRE(actual.historyId_ == expected.historyId_);
    REQUIRE(actual.parentHistoryId_ == expected.parentHistoryId_);
    REQUIRE(actual.stepId_ == expected.stepId_);
    REQUIRE(actual.environmentNode_ == expected.environmentNode_);
    for (int axis = 0; axis < 3; ++axis) {
      CAPTURE(axis);
      require_exact(actual.positionM_[axis], expected.positionM_[axis]);
      require_exact(actual.direction_[axis], expected.direction_[axis]);
    }
  }

  void require_advance_exact(fixture::Advance const& actual,
                             fixture::Advance const& expected) {
    REQUIRE(actual.status_ == expected.status_);
    REQUIRE(actual.reserved_ == expected.reserved_);
    require_exact(actual.bendParameter_, expected.bendParameter_);
    require_exact(actual.chordLengthM_, expected.chordLengthM_);
    require_particle_exact(actual.particle_, expected.particle_);
  }

  void require_same(fixture::Outcome const& actual, fixture::Outcome const& expected) {
    REQUIRE(actual.limit_.status_ == expected.limit_.status_);
    REQUIRE(actual.limit_.reserved_ == expected.limit_.reserved_);
    require_exact(actual.limit_.distanceM_, expected.limit_.distanceM_);
    require_exact(actual.limit_.gyroradiusM_, expected.limit_.gyroradiusM_);
    require_advance_exact(actual.advance_, expected.advance_);
    REQUIRE(actual.callerRan_ == expected.callerRan_);
    REQUIRE(actual.callerUsedZeroField_ == expected.callerUsedZeroField_);
    require_advance_exact(actual.callerAdvance_, expected.callerAdvance_);
  }

  void require_input_unchanged(fixture::Query const& actual,
                               fixture::Query const& expected) {
    require_particle_exact(actual.particle_, expected.particle_);
    require_exact(actual.massGeV_, expected.massGeV_);
    require_exact(actual.chargeNumber_, expected.chargeNumber_);
    require_exact(actual.distanceM_, expected.distanceM_);
    require_exact(actual.maximumDeflection_, expected.maximumDeflection_);
    REQUIRE(actual.category_ == expected.category_);
    REQUIRE(actual.crossDevice_ == expected.crossDevice_);
    REQUIRE(actual.side_ == expected.side_);
    for (int axis = 0; axis < 3; ++axis) {
      require_exact(actual.fieldT_[axis], expected.fieldT_[axis]);
    }
  }

  void require_branch(fixture::Query const& query, fixture::Outcome const& actual,
                      fixture::Witness const& witness) {
    using category = fixture::Category;
    auto const success = static_cast<std::uint32_t>(fixture::status_type::Success);
    auto const linear = static_cast<std::uint32_t>(fixture::status_type::Linear);
    auto const invalid = static_cast<std::uint32_t>(fixture::status_type::InvalidInput);
    auto const nonfinite =
        static_cast<std::uint32_t>(fixture::status_type::NonFiniteResult);
    if (query.category_ == category::PerpendicularThreshold) {
      if (query.side_ < 0) { REQUIRE(witness.perpendicularMomentumGeV_ < 1.e-9); }
      if (query.side_ > 0) { REQUIRE(witness.perpendicularMomentumGeV_ > 1.e-9); }
      REQUIRE(actual.limit_.status_ ==
              (witness.perpendicularMomentumGeV_ < 1.e-9 ? linear : success));
      if (witness.perpendicularMomentumGeV_ < 1.e-9) {
        REQUIRE(actual.advance_.status_ == success);
        REQUIRE(actual.callerAdvance_.status_ == linear);
      }
    } else if (query.category_ == category::RadiusThreshold) {
      if (query.side_ < 0) { REQUIRE(witness.gyroradiusM_ < 1.e9); }
      if (query.side_ > 0) { REQUIRE(witness.gyroradiusM_ > 1.e9); }
      REQUIRE(actual.limit_.status_ == (witness.gyroradiusM_ > 1.e9 ? linear : success));
      if (witness.gyroradiusM_ > 1.e9) {
        REQUIRE(actual.advance_.status_ == success);
        REQUIRE(actual.callerAdvance_.status_ == linear);
      }
    } else if (query.category_ == category::Neutral ||
               query.category_ == category::ZeroField ||
               query.category_ == category::Parallel) {
      REQUIRE(actual.limit_.status_ == linear);
      REQUIRE(actual.advance_.status_ == linear);
    } else if (query.category_ == category::InvalidParticle ||
               query.category_ == category::InvalidField) {
      auto const expected = query.side_ == 1 ? success : invalid;
      REQUIRE(actual.limit_.status_ == expected);
      REQUIRE(actual.advance_.status_ == expected);
    } else if (query.category_ == category::InvalidDistance) {
      REQUIRE(actual.limit_.status_ == success);
      REQUIRE(actual.advance_.status_ == invalid);
    } else if (query.category_ == category::ZeroDistance) {
      REQUIRE(actual.advance_.status_ == success);
      require_particle_exact(actual.advance_.particle_, query.particle_);
      require_exact(actual.advance_.bendParameter_, 0.);
      require_exact(actual.advance_.chordLengthM_, 0.);
    } else if (query.category_ == category::SpecialMetadata) {
      REQUIRE(actual.limit_.status_ == success);
      REQUIRE(actual.advance_.status_ == success);
      require_exact(actual.advance_.particle_.weight_, query.particle_.weight_);
    } else if (query.category_ == category::NonFiniteMomentum ||
               query.category_ == category::UnderflowMomentum ||
               query.category_ == category::OverflowField ||
               query.category_ == category::OverflowChord) {
      REQUIRE(actual.advance_.status_ == nonfinite);
      REQUIRE(actual.limit_.status_ ==
              (query.category_ == category::OverflowField ? nonfinite : linear));
      require_exact(actual.advance_.particle_.timeS_, query.particle_.timeS_);
      require_exact(actual.advance_.bendParameter_, 0.);
      require_exact(actual.advance_.chordLengthM_, 0.);
      bool changedPosition = false;
      for (int axis = 0; axis < 3; ++axis) {
        require_exact(actual.advance_.particle_.direction_[axis],
                      query.particle_.direction_[axis]);
        changedPosition = changedPosition || actual.advance_.particle_.positionM_[axis] !=
                                                 query.particle_.positionM_[axis];
      }
      REQUIRE(changedPosition);
    } else if (query.category_ == category::Angle) {
      bool const valid = std::isfinite(query.maximumDeflection_) &&
                         query.maximumDeflection_ > 0. &&
                         query.maximumDeflection_ < .5 * 3.141592653589793;
      REQUIRE(actual.limit_.status_ == (valid ? success : invalid));
      REQUIRE(actual.advance_.status_ == success);
    }
  }

  void require_analytic(fixture::Query const& query, fixture::Advance const& result) {
    double const length = query.distanceM_;
    double const tolerance = 32. * std::numeric_limits<double>::epsilon();
    double const expectedNorm = std::sqrt(2.);
    REQUIRE(result.status_ == static_cast<std::uint32_t>(fixture::status_type::Success));
    REQUIRE(std::abs(result.bendParameter_ - 1.) <= tolerance);
    REQUIRE(std::abs(result.particle_.positionM_[0] - query.particle_.positionM_[0] -
                     length) <= tolerance * length);
    REQUIRE(std::abs(result.particle_.positionM_[1] - query.particle_.positionM_[1] +
                     .5 * length) <= tolerance * length);
    require_exact(result.particle_.positionM_[2], query.particle_.positionM_[2]);
    REQUIRE(std::abs(result.particle_.direction_[0] - 1. / expectedNorm) <= tolerance);
    REQUIRE(std::abs(result.particle_.direction_[1] + 1. / expectedNorm) <= tolerance);
    require_exact(result.particle_.direction_[2], 0.);
    REQUIRE(std::abs(result.chordLengthM_ - length * std::sqrt(1.25)) <=
            tolerance * length);
    double const flightTime =
        length / (.8 * uniform_magnetic_reference::SpeedOfLightMPerS);
    REQUIRE(std::abs(result.particle_.timeS_ - query.particle_.timeS_ - flightTime) <=
            tolerance * flightTime);
    double const chordTime =
        result.chordLengthM_ / (.8 * uniform_magnetic_reference::SpeedOfLightMPerS);
    REQUIRE(std::abs(result.particle_.timeS_ - query.particle_.timeS_ - chordTime) >
            .1 * flightTime);
    // Normalizing before the second half displacement gives a different endpoint.
    double const normalizedHalfY = -.5 * length / expectedNorm;
    REQUIRE(std::abs(result.particle_.positionM_[1] - query.particle_.positionM_[1] -
                     normalizedHalfY) > .1 * length);
  }
} // namespace

TEST_CASE(
    "Uniform magnetic primitives preserve production results and leapfrog semantics",
    "[uniform_magnetic_field]") {
  REQUIRE_FALSE(Kokkos::is_initialized());
  {
    corsika::kokkos::Runtime runtime;
    auto const queries = fixture::queries();
    REQUIRE(queries.size() == 104);
    std::array<unsigned int, 19> coverage{};
    unsigned int crossDevice = 0;
    for (std::size_t index = 0; index < queries.size(); ++index) {
      auto const query = queries[index];
      auto const original = query;
      CAPTURE(index, query.category_, query.particle_.energyGeV_, query.chargeNumber_);
      auto const result = fixture::actual(query);
      auto const expected = fixture::expected(query);
      require_same(result, expected);
      require_input_unchanged(query, original);
      require_branch(query, result, fixture::witness(query));
      if (query.category_ == fixture::Category::Analytic) {
        require_analytic(query, result.advance_);
      }
      if (query.maximumDeflection_ == .2) {
        auto const withDefault =
            fixture::convert(corsika::kokkos::maximum_uniform_magnetic_step(
                query.particle_, query.massGeV_, query.chargeNumber_, query.fieldT_));
        REQUIRE(withDefault.status_ == result.limit_.status_);
        require_exact(withDefault.distanceM_, result.limit_.distanceM_);
        require_exact(withDefault.gyroradiusM_, result.limit_.gyroradiusM_);
      }
      ++coverage[static_cast<unsigned int>(query.category_)];
      crossDevice += query.crossDevice_;
    }
    for (auto count : coverage) { REQUIRE(count > 0); }
    REQUIRE(coverage[static_cast<unsigned int>(fixture::Category::Scan)] == 48);
    REQUIRE(crossDevice == 55);
    corsika::kokkos::MagneticAdvanceResult const empty;
    REQUIRE(empty.status_ == fixture::status_type::InvalidInput);
    REQUIRE(empty.reserved_ == 0);
    require_particle_exact(empty.particle_, fixture::particle_type{});
    require_exact(empty.bendParameter_, 0.);
    require_exact(empty.chordLengthM_, 0.);
    require_exact(corsika::kokkos::DefaultMaximumMagneticDeflection, .2);
  }
  REQUIRE_FALSE(Kokkos::is_initialized());
  REQUIRE(Kokkos::is_finalized());
}

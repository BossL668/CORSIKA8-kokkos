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
#include <algorithm>
#include <cmath>
#include <limits>
#include <type_traits>

namespace {
  namespace fixture = uniform_magnetic_test;
  using execution_space_type = Kokkos::DefaultExecutionSpace;
  using memory_space_type = typename execution_space_type::memory_space;
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

  void require_close(double actual, double expected, double scale) {
    if (!std::isfinite(expected)) {
      require_exact(actual, expected);
      return;
    }
    REQUIRE(std::isfinite(actual));
    REQUIRE(std::abs(actual - expected) <=
            512 * std::numeric_limits<double>::epsilon() * scale +
                32 * std::numeric_limits<double>::denorm_min());
  }

  void require_advance_cross(fixture::Query const& query, fixture::Advance const& actual,
                             fixture::Advance const& expected) {
    REQUIRE(actual.status_ == expected.status_);
    REQUIRE(actual.reserved_ == expected.reserved_);
    require_close(actual.bendParameter_, expected.bendParameter_,
                  std::abs(expected.bendParameter_));
    require_close(actual.chordLengthM_, expected.chordLengthM_,
                  std::abs(expected.chordLengthM_));
    auto metadata = actual.particle_;
    for (int axis = 0; axis < 3; ++axis) {
      require_close(
          actual.particle_.positionM_[axis], expected.particle_.positionM_[axis],
          std::abs(query.particle_.positionM_[axis]) + std::abs(query.distanceM_) +
              std::abs(expected.particle_.positionM_[axis]));
      require_close(actual.particle_.direction_[axis],
                    expected.particle_.direction_[axis], 1.);
      metadata.positionM_[axis] = expected.particle_.positionM_[axis];
      metadata.direction_[axis] = expected.particle_.direction_[axis];
    }
    require_close(actual.particle_.timeS_, expected.particle_.timeS_,
                  std::abs(expected.particle_.timeS_) + std::abs(query.particle_.timeS_));
    metadata.timeS_ = expected.particle_.timeS_;
    require_particle_exact(metadata, expected.particle_);
  }

  void require_cross(fixture::Query const& query, fixture::Outcome const& actual,
                     fixture::Outcome const& expected) {
    REQUIRE(actual.limit_.status_ == expected.limit_.status_);
    REQUIRE(actual.limit_.reserved_ == expected.limit_.reserved_);
    require_close(actual.limit_.distanceM_, expected.limit_.distanceM_,
                  std::abs(expected.limit_.distanceM_));
    require_close(actual.limit_.gyroradiusM_, expected.limit_.gyroradiusM_,
                  std::abs(expected.limit_.gyroradiusM_));
    require_advance_cross(query, actual.advance_, expected.advance_);
    REQUIRE(actual.callerRan_ == expected.callerRan_);
    REQUIRE(actual.callerUsedZeroField_ == expected.callerUsedZeroField_);
    require_advance_cross(query, actual.callerAdvance_, expected.callerAdvance_);
  }
} // namespace

TEST_CASE("Device uniform magnetic primitives match fixed production in every lane",
          "[uniform_magnetic_field_device]") {
  REQUIRE_FALSE(Kokkos::is_initialized());
  {
    corsika::kokkos::Runtime runtime;
    INFO("Actual execution space: " << execution_space_type::name());
    auto const inputs = fixture::queries();
    REQUIRE(inputs.size() == 104);
    Kokkos::View<fixture::Query*, Kokkos::HostSpace> hostInputs("host_queries",
                                                                inputs.size());
    for (std::size_t index = 0; index < inputs.size(); ++index) {
      hostInputs(index) = inputs[index];
    }
    Kokkos::View<fixture::Query*, memory_space_type> queries("queries", inputs.size());
    Kokkos::deep_copy(queries, hostInputs);
    Kokkos::View<fixture::Query*, memory_space_type> after("after_queries",
                                                           inputs.size());
    Kokkos::View<fixture::Outcome * [2], Kokkos::LayoutRight, memory_space_type> outcomes(
        "magnetic_outcomes", inputs.size());
    Kokkos::View<fixture::Witness*, memory_space_type> witnesses("threshold_witnesses",
                                                                 inputs.size());
    Kokkos::parallel_for(
        "per_particle_uniform_magnetic_field",
        Kokkos::RangePolicy<execution_space_type>(0, inputs.size()),
        KOKKOS_LAMBDA(int index) {
          auto const query = queries(index);
          outcomes(index, 0) = fixture::actual(query);
          outcomes(index, 1) = fixture::expected(query);
          witnesses(index) = fixture::witness(query);
          after(index) = query;
        });
    auto const host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, outcomes);
    auto const hostWitness =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, witnesses);
    auto const hostAfter =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, after);
    auto const sourceAfter =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, queries);
    unsigned int crossDevice = 0;
    unsigned int sameDevice = 0;
    for (std::size_t index = 0; index < inputs.size(); ++index) {
      auto const& query = inputs[index];
      CAPTURE(index, query.category_, query.particle_.energyGeV_, query.chargeNumber_,
              query.crossDevice_);
      {
        INFO("Same-device immutable production oracle; every query and output field");
        require_same(host(index, 0), host(index, 1));
        require_branch(query, host(index, 0), hostWitness(index));
        ++sameDevice;
      }
      require_input_unchanged(hostAfter(index), query);
      require_input_unchanged(sourceAfter(index), query);
      if (query.crossDevice_) {
        INFO(
            "Cross-device well-conditioned subset; threshold and IEEE cases are "
            "intentionally not in this subset");
        require_cross(query, host(index, 0), fixture::expected(query));
        ++crossDevice;
      }
    }
    REQUIRE(sameDevice == 104);
    REQUIRE(crossDevice == 55);
  }
  REQUIRE_FALSE(Kokkos::is_initialized());
  REQUIRE(Kokkos::is_finalized());
}

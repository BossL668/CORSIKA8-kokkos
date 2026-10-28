/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/kokkos/data/ParticleState.hpp>
#include <corsika/kokkos/data/StepRecord.hpp>

#include "FrozenCp8Data.hpp"

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>

// Explicit mapping from the new API to the independently frozen declarations.
// A scalar/array distinction keeps declared array types and all elements checked.
#define CORSIKA_TEST_PARTICLE_FIELDS(SCALAR, ARRAY) \
  SCALAR(pid_, pid)                                 \
  SCALAR(mediumId_, medium_id)                      \
  SCALAR(generation_, generation)                   \
  SCALAR(reserved_, reserved)                       \
  SCALAR(energyGeV_, energy_GeV)                    \
  ARRAY(positionM_, position_m)                     \
  ARRAY(direction_, direction)                      \
  SCALAR(timeS_, time_s)                            \
  SCALAR(weight_, weight)                           \
  SCALAR(historyId_, history_id)                    \
  SCALAR(parentHistoryId_, parent_history_id)       \
  SCALAR(stepId_, step_id)                          \
  SCALAR(environmentNode_, environment_node)

#define CORSIKA_TEST_STEP_FIELDS(SCALAR, ARRAY)            \
  SCALAR(historyId_, history_id)                           \
  SCALAR(stepId_, step_id)                                 \
  SCALAR(pid_, pid)                                        \
  SCALAR(processId_, process_id)                           \
  ARRAY(startPositionM_, start_position_m)                 \
  ARRAY(endPositionM_, end_position_m)                     \
  SCALAR(startTimeS_, start_time_s)                        \
  SCALAR(endTimeS_, end_time_s)                            \
  SCALAR(startEnergyGeV_, start_energy_GeV)                \
  SCALAR(endEnergyGeV_, end_energy_GeV)                    \
  SCALAR(depositedEnergyGeV_, deposited_energy_GeV)        \
  SCALAR(weight_, weight)                                  \
  SCALAR(cutDepositedEnergyGeV_, cut_deposited_energy_GeV) \
  SCALAR(nativeMuonPairCut_, native_muon_pair_cut)

namespace data_test {

  static_assert(sizeof(double) == sizeof(std::uint64_t));
  static_assert(std::numeric_limits<double>::is_iec559);
  static_assert(std::numeric_limits<double>::digits == 53);

  // Exercise storage representations only. These are deliberately not physical
  // input states. No arithmetic is performed on floating-point test patterns.
  inline std::array<std::uint64_t, 14> constexpr floatingPatterns{
      0x0000000000000000ULL, // positive zero
      0x8000000000000000ULL, // negative zero
      0x3ff0000000000000ULL, // one
      0xbff0000000000000ULL, // minus one
      0x0000000000000001ULL, // smallest positive subnormal
      0x8000000000000001ULL, // negative subnormal
      0x0010000000000000ULL, // smallest positive normal
      0x7fefffffffffffffULL, // largest finite positive value
      0xffefffffffffffffULL, // largest finite negative magnitude
      0x7ff0000000000000ULL, // positive infinity
      0xfff0000000000000ULL, // negative infinity
      0x7ff8000000000001ULL, // quiet NaN with payload
      0x7ff80000deadbeefULL, // a different quiet NaN payload
      0xfff80000abcdef12ULL  // negative quiet NaN with payload
  };

  inline std::uint64_t floating_bits(double const& value) {
    std::uint64_t bits{};
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
  }

  template <std::size_t TCount>
  void fill_floating_patterns(double* const (&fields)[TCount], std::size_t const index) {
    for (std::size_t fieldIndex = 0; fieldIndex < TCount; ++fieldIndex) {
      auto const bits = floatingPatterns[(index + fieldIndex) % floatingPatterns.size()];
      std::memcpy(fields[fieldIndex], &bits, sizeof(bits));
    }
  }

  inline cp8_frozen::EmParticleState particle_pattern(std::size_t const index) {
    cp8_frozen::EmParticleState reference{};
    std::array<std::int32_t, 7> constexpr identifiers{
        11,
        -11,
        13,
        -13,
        22,
        std::numeric_limits<std::int32_t>::min(),
        std::numeric_limits<std::int32_t>::max()};
    std::array<std::int32_t, 7> constexpr nodes{-3,
                                                -1,
                                                0,
                                                1,
                                                17,
                                                std::numeric_limits<std::int32_t>::min(),
                                                std::numeric_limits<std::int32_t>::max()};
    reference.pid = identifiers[index % identifiers.size()];
    // Adjacent records share a medium while retaining independent logical nodes.
    reference.medium_id = identifiers[(index / 2) % identifiers.size()];
    reference.generation =
        std::numeric_limits<std::uint32_t>::max() - static_cast<std::uint32_t>(index);
    reference.reserved = index % 3 == 0 ? 0U : index % 3 == 1 ? 1U : 0x80000000U;
    reference.history_id = 0xfedcba9876543210ULL + index;
    reference.parent_history_id = std::numeric_limits<std::uint64_t>::max() - index;
    reference.step_id = 0x0020000000000001ULL + index;
    reference.environment_node = nodes[index % nodes.size()];
    double* const fields[]{
        &reference.energy_GeV,    &reference.position_m[0], &reference.position_m[1],
        &reference.position_m[2], &reference.direction[0],  &reference.direction[1],
        &reference.direction[2],  &reference.time_s,        &reference.weight};
    fill_floating_patterns(fields, index);
    return reference;
  }

  inline cp8_frozen::EmStepRecord step_pattern(std::size_t const index) {
    cp8_frozen::EmStepRecord reference{};
    reference.history_id = std::numeric_limits<std::uint64_t>::max() - index;
    reference.step_id = 0x0020000000000001ULL + index;
    reference.pid = index % 2 == 0 ? -13 : 22;
    reference.process_id = index % 2 == 0 ? std::numeric_limits<std::int32_t>::min()
                                          : std::numeric_limits<std::int32_t>::max();
    reference.native_muon_pair_cut =
        index % 3 == 0 ? 0U
                       : index % 3 == 1 ? 1U : std::numeric_limits<std::uint32_t>::max();
    double* const fields[]{&reference.start_position_m[0],
                           &reference.start_position_m[1],
                           &reference.start_position_m[2],
                           &reference.end_position_m[0],
                           &reference.end_position_m[1],
                           &reference.end_position_m[2],
                           &reference.start_time_s,
                           &reference.end_time_s,
                           &reference.start_energy_GeV,
                           &reference.end_energy_GeV,
                           &reference.deposited_energy_GeV,
                           &reference.weight,
                           &reference.cut_deposited_energy_GeV};
    fill_floating_patterns(fields, index);
    return reference;
  }

  inline corsika::kokkos::ParticleState make_particle(
      cp8_frozen::EmParticleState const& reference) {
    corsika::kokkos::ParticleState result{};
#define CORSIKA_TEST_COPY_SCALAR(newField, oldField) result.newField = reference.oldField;
#define CORSIKA_TEST_COPY_ARRAY(newField, oldField)   \
  for (std::size_t axis = 0; axis < 3; ++axis) {      \
    result.newField[axis] = reference.oldField[axis]; \
  }
    CORSIKA_TEST_PARTICLE_FIELDS(CORSIKA_TEST_COPY_SCALAR, CORSIKA_TEST_COPY_ARRAY)
#undef CORSIKA_TEST_COPY_SCALAR
#undef CORSIKA_TEST_COPY_ARRAY
    return result;
  }

  inline corsika::kokkos::StepRecord make_step(
      cp8_frozen::EmStepRecord const& reference) {
    corsika::kokkos::StepRecord result{};
#define CORSIKA_TEST_COPY_SCALAR(newField, oldField) result.newField = reference.oldField;
#define CORSIKA_TEST_COPY_ARRAY(newField, oldField)   \
  for (std::size_t axis = 0; axis < 3; ++axis) {      \
    result.newField[axis] = reference.oldField[axis]; \
  }
    CORSIKA_TEST_STEP_FIELDS(CORSIKA_TEST_COPY_SCALAR, CORSIKA_TEST_COPY_ARRAY)
#undef CORSIKA_TEST_COPY_SCALAR
#undef CORSIKA_TEST_COPY_ARRAY
    return result;
  }

  template <typename TValue>
  void require_field(TValue const actual, TValue const expected) {
    REQUIRE(actual == expected);
  }

  inline void require_field(double const& actual, double const& expected) {
    REQUIRE(floating_bits(actual) == floating_bits(expected));
  }

  template <typename TValue, std::size_t TCount>
  void require_field(TValue const (&actual)[TCount], TValue const (&expected)[TCount]) {
    for (std::size_t axis = 0; axis < TCount; ++axis) {
      CAPTURE(axis);
      require_field(actual[axis], expected[axis]);
    }
  }

  inline void require_particle(corsika::kokkos::ParticleState const& actual,
                               cp8_frozen::EmParticleState const& expected) {
#define CORSIKA_TEST_CHECK_FIELD(newField, oldField)   \
  do {                                                 \
    INFO("ParticleState." #newField);                  \
    require_field(actual.newField, expected.oldField); \
  } while (false);
    CORSIKA_TEST_PARTICLE_FIELDS(CORSIKA_TEST_CHECK_FIELD, CORSIKA_TEST_CHECK_FIELD)
#undef CORSIKA_TEST_CHECK_FIELD
  }

  inline void require_step(corsika::kokkos::StepRecord const& actual,
                           cp8_frozen::EmStepRecord const& expected) {
#define CORSIKA_TEST_CHECK_FIELD(newField, oldField)   \
  do {                                                 \
    INFO("StepRecord." #newField);                     \
    require_field(actual.newField, expected.oldField); \
  } while (false);
    CORSIKA_TEST_STEP_FIELDS(CORSIKA_TEST_CHECK_FIELD, CORSIKA_TEST_CHECK_FIELD)
#undef CORSIKA_TEST_CHECK_FIELD
  }

} // namespace data_test

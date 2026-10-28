/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include "DataTestHelpers.hpp"

#include <cstddef>
#include <type_traits>

namespace {

  using particle_type = corsika::kokkos::ParticleState;
  using step_type = corsika::kokkos::StepRecord;
  using reference_particle_type = cp8_frozen::EmParticleState;
  using reference_step_type = cp8_frozen::EmStepRecord;

  static_assert(std::is_standard_layout_v<particle_type>);
  static_assert(std::is_trivially_copyable_v<particle_type>);
  static_assert(std::is_aggregate_v<particle_type>);
  static_assert(std::is_standard_layout_v<step_type>);
  static_assert(std::is_trivially_copyable_v<step_type>);
  static_assert(std::is_aggregate_v<step_type>);
  static_assert(sizeof(particle_type) == sizeof(reference_particle_type));
  static_assert(alignof(particle_type) == alignof(reference_particle_type));
  static_assert(alignof(particle_type) == 16);
  static_assert(sizeof(step_type) == sizeof(reference_step_type));
  static_assert(alignof(step_type) == alignof(reference_step_type));

#define CORSIKA_TEST_PARTICLE_LAYOUT(newField, oldField)                      \
  static_assert(std::is_same_v<decltype(particle_type::newField),             \
                               decltype(reference_particle_type::oldField)>); \
  static_assert(offsetof(particle_type, newField) ==                          \
                offsetof(reference_particle_type, oldField));
  CORSIKA_TEST_PARTICLE_FIELDS(CORSIKA_TEST_PARTICLE_LAYOUT, CORSIKA_TEST_PARTICLE_LAYOUT)
#undef CORSIKA_TEST_PARTICLE_LAYOUT

#define CORSIKA_TEST_STEP_LAYOUT(newField, oldField)                      \
  static_assert(std::is_same_v<decltype(step_type::newField),             \
                               decltype(reference_step_type::oldField)>); \
  static_assert(offsetof(step_type, newField) == offsetof(reference_step_type, oldField));
  CORSIKA_TEST_STEP_FIELDS(CORSIKA_TEST_STEP_LAYOUT, CORSIKA_TEST_STEP_LAYOUT)
#undef CORSIKA_TEST_STEP_LAYOUT

} // namespace

TEST_CASE("Data records preserve the frozen Cp8 member layout", "[data_layout]") {
  REQUIRE(sizeof(particle_type) == sizeof(reference_particle_type));
  REQUIRE(alignof(particle_type) == alignof(reference_particle_type));
  REQUIRE(sizeof(step_type) == sizeof(reference_step_type));
  REQUIRE(alignof(step_type) == alignof(reference_step_type));
}

TEST_CASE("Data records preserve every frozen default", "[data_defaults]") {
  particle_type const particle{};
  step_type const step{};
  data_test::require_particle(particle, reference_particle_type{});
  data_test::require_step(step, reference_step_type{});
  REQUIRE(particle.weight_ == 1.);
  REQUIRE(particle.environmentNode_ == -3);
  REQUIRE(step.weight_ == 0.);

  // Both forms must execute the preserved in-class initializers.
  particle_type const defaultInitializedParticle;
  step_type const defaultInitializedStep;
  data_test::require_particle(defaultInitializedParticle, reference_particle_type{});
  data_test::require_step(defaultInitializedStep, reference_step_type{});
}

TEST_CASE("Data records preserve independent field representations", "[data_patterns]") {
  for (std::size_t index = 0; index < 257; ++index) {
    CAPTURE(index);
    auto const referenceParticle = data_test::particle_pattern(index);
    auto const referenceStep = data_test::step_pattern(index);
    auto const particle = data_test::make_particle(referenceParticle);
    auto const step = data_test::make_step(referenceStep);
    data_test::require_particle(particle, referenceParticle);
    data_test::require_step(step, referenceStep);
    particle_type const particleCopy = particle;
    step_type const stepCopy = step;
    data_test::require_particle(particleCopy, referenceParticle);
    data_test::require_step(stepCopy, referenceStep);
  }
  auto const first = data_test::make_particle(data_test::particle_pattern(0));
  auto const second = data_test::make_particle(data_test::particle_pattern(1));
  REQUIRE(first.mediumId_ == second.mediumId_);
  REQUIRE(first.environmentNode_ == -3);
  REQUIRE(second.environmentNode_ == -1);
  REQUIRE(first.historyId_ > (std::uint64_t{1} << 53));
  REQUIRE(first.stepId_ > (std::uint64_t{1} << 53));
}

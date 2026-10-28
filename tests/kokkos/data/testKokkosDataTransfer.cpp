/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include <corsika/kokkos/runtime/Runtime.hpp>

#include "DataTestHelpers.hpp"

#include <Kokkos_Core.hpp>

#include <cstdint>
#include <type_traits>

#if defined(CORSIKA_EXPECT_CUDA) && defined(CORSIKA_EXPECT_OPENMP)
#error "Select only one expected default execution space."
#endif

namespace {

  using execution_space_type = Kokkos::DefaultExecutionSpace;
  using memory_space_type = typename execution_space_type::memory_space;
  using particle_type = corsika::kokkos::ParticleState;
  using step_type = corsika::kokkos::StepRecord;

#if defined(CORSIKA_EXPECT_CUDA)
#ifndef KOKKOS_ENABLE_CUDA
#error "The CUDA data audit requires a CUDA-enabled Kokkos build."
#else
  static_assert(std::is_same_v<execution_space_type, Kokkos::Cuda>,
                "The CUDA data audit must execute in Kokkos::Cuda.");
#endif
#elif defined(CORSIKA_EXPECT_OPENMP)
#ifndef KOKKOS_ENABLE_OPENMP
#error "The OpenMP data audit requires an OpenMP-enabled Kokkos build."
#else
  static_assert(std::is_same_v<execution_space_type, Kokkos::OpenMP>,
                "The OpenMP data audit must execute in Kokkos::OpenMP.");
#endif
#endif

  int constexpr recordCount = 257;

  void require_fresh_runtime() {
    REQUIRE_FALSE(Kokkos::is_initialized());
    REQUIRE_FALSE(Kokkos::is_finalized());
  }

} // namespace

TEST_CASE("Data records survive a real execution-space field transfer",
          "[data_transfer]") {
  require_fresh_runtime();
  {
    corsika::kokkos::Runtime runtime;
    INFO("Actual execution space: " << execution_space_type::name());
    REQUIRE(Kokkos::is_initialized());
    REQUIRE(runtime.getExecutionSpaceName() == execution_space_type::name());

    Kokkos::View<particle_type*, memory_space_type> inputParticles("input_particles",
                                                                   recordCount);
    Kokkos::View<particle_type*, memory_space_type> outputParticles("output_particles",
                                                                    recordCount);
    Kokkos::View<step_type*, memory_space_type> inputSteps("input_steps", recordCount);
    Kokkos::View<step_type*, memory_space_type> outputSteps("output_steps", recordCount);
    Kokkos::View<std::uint32_t*, memory_space_type> markers("transfer_markers",
                                                            recordCount);

    // Explicit HostSpace allocations cannot alias the execution-space outputs,
    // including when the selected backend itself uses host memory.
    Kokkos::View<particle_type*, Kokkos::HostSpace> hostInputParticles(
        "host_input_particles", recordCount);
    Kokkos::View<step_type*, Kokkos::HostSpace> hostInputSteps("host_input_steps",
                                                               recordCount);
    Kokkos::View<particle_type*, Kokkos::HostSpace> hostOutputParticles(
        "host_output_particles", recordCount);
    Kokkos::View<step_type*, Kokkos::HostSpace> hostOutputSteps("host_output_steps",
                                                                recordCount);
    Kokkos::View<std::uint32_t*, Kokkos::HostSpace> hostMarkers("host_transfer_markers",
                                                                recordCount);
    for (int index = 0; index < recordCount; ++index) {
      hostInputParticles(index) =
          data_test::make_particle(data_test::particle_pattern(index));
      hostInputSteps(index) = data_test::make_step(data_test::step_pattern(index));
    }
    Kokkos::deep_copy(inputParticles, hostInputParticles);
    Kokkos::deep_copy(inputSteps, hostInputSteps);
    Kokkos::deep_copy(markers, std::uint32_t{0});

    Kokkos::parallel_for(
        "data_field_transfer", Kokkos::RangePolicy<execution_space_type>(0, recordCount),
        KOKKOS_LAMBDA(int const index) {
          int const sourceIndex = recordCount - 1 - index;
          auto const& sourceParticle = inputParticles(sourceIndex);
          auto& destinationParticle = outputParticles(index);
#define CORSIKA_TEST_DEVICE_PARTICLE_SCALAR(newField, oldField) \
  destinationParticle.newField = sourceParticle.newField;
#define CORSIKA_TEST_DEVICE_PARTICLE_ARRAY(newField, oldField)          \
  for (int axis = 0; axis < 3; ++axis) {                                \
    destinationParticle.newField[axis] = sourceParticle.newField[axis]; \
  }
          CORSIKA_TEST_PARTICLE_FIELDS(CORSIKA_TEST_DEVICE_PARTICLE_SCALAR,
                                       CORSIKA_TEST_DEVICE_PARTICLE_ARRAY)
#undef CORSIKA_TEST_DEVICE_PARTICLE_SCALAR
#undef CORSIKA_TEST_DEVICE_PARTICLE_ARRAY

          auto const& sourceStep = inputSteps(sourceIndex);
          auto& destinationStep = outputSteps(index);
#define CORSIKA_TEST_DEVICE_STEP_SCALAR(newField, oldField) \
  destinationStep.newField = sourceStep.newField;
#define CORSIKA_TEST_DEVICE_STEP_ARRAY(newField, oldField)      \
  for (int axis = 0; axis < 3; ++axis) {                        \
    destinationStep.newField[axis] = sourceStep.newField[axis]; \
  }
          CORSIKA_TEST_STEP_FIELDS(CORSIKA_TEST_DEVICE_STEP_SCALAR,
                                   CORSIKA_TEST_DEVICE_STEP_ARRAY)
#undef CORSIKA_TEST_DEVICE_STEP_SCALAR
#undef CORSIKA_TEST_DEVICE_STEP_ARRAY
          markers(index) = static_cast<std::uint32_t>(sourceIndex + 1);
        });
    Kokkos::fence("data field transfer complete");
    Kokkos::deep_copy(hostOutputParticles, outputParticles);
    Kokkos::deep_copy(hostOutputSteps, outputSteps);
    Kokkos::deep_copy(hostMarkers, markers);
    for (int index = 0; index < recordCount; ++index) {
      CAPTURE(index);
      auto const sourceIndex = recordCount - 1 - index;
      REQUIRE(hostMarkers(index) == static_cast<std::uint32_t>(sourceIndex + 1));
      data_test::require_particle(hostOutputParticles(index),
                                  data_test::particle_pattern(sourceIndex));
      data_test::require_step(hostOutputSteps(index),
                              data_test::step_pattern(sourceIndex));
    }
  }
  REQUIRE_FALSE(Kokkos::is_initialized());
  REQUIRE(Kokkos::is_finalized());
}

TEST_CASE("Data defaults are constructed inside the execution space",
          "[data_device_defaults]") {
  require_fresh_runtime();
  {
    corsika::kokkos::Runtime runtime;
    INFO("Actual execution space: " << execution_space_type::name());
    Kokkos::View<particle_type*, memory_space_type> particles("default_particles",
                                                              recordCount);
    Kokkos::View<step_type*, memory_space_type> steps("default_steps", recordCount);
    Kokkos::View<std::uint32_t*, memory_space_type> markers("default_markers",
                                                            recordCount);
    Kokkos::deep_copy(markers, std::uint32_t{0});

    Kokkos::parallel_for(
        "data_construct_defaults",
        Kokkos::RangePolicy<execution_space_type>(0, recordCount),
        KOKKOS_LAMBDA(int const index) {
          particle_type const particle{};
          step_type const step{};
          particles(index) = particle;
          steps(index) = step;
          markers(index) = static_cast<std::uint32_t>(index + 1);
        });
    Kokkos::fence("data default construction complete");

    Kokkos::View<particle_type*, Kokkos::HostSpace> hostParticles(
        "host_default_particles", recordCount);
    Kokkos::View<step_type*, Kokkos::HostSpace> hostSteps("host_default_steps",
                                                          recordCount);
    Kokkos::View<std::uint32_t*, Kokkos::HostSpace> hostMarkers("host_default_markers",
                                                                recordCount);
    Kokkos::deep_copy(hostParticles, particles);
    Kokkos::deep_copy(hostSteps, steps);
    Kokkos::deep_copy(hostMarkers, markers);
    for (int index = 0; index < recordCount; ++index) {
      CAPTURE(index);
      REQUIRE(hostMarkers(index) == static_cast<std::uint32_t>(index + 1));
      data_test::require_particle(hostParticles(index), cp8_frozen::EmParticleState{});
      data_test::require_step(hostSteps(index), cp8_frozen::EmStepRecord{});
    }
  }
  REQUIRE_FALSE(Kokkos::is_initialized());
  REQUIRE(Kokkos::is_finalized());
}

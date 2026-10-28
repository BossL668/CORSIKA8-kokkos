/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include <corsika/kokkos/runtime/Runtime.hpp>

#include <Kokkos_Core.hpp>
#include <catch2/catch_test_macros.hpp>

#include <exception>
#include <memory>
#include <stdexcept>
#include <thread>
#include <type_traits>

namespace {

  using runtime_type = corsika::kokkos::Runtime;
  using execution_space_type = Kokkos::DefaultExecutionSpace;
  using host_execution_space_type = Kokkos::DefaultHostExecutionSpace;

  static_assert(!std::is_copy_constructible_v<runtime_type>);
  static_assert(!std::is_copy_assignable_v<runtime_type>);
  static_assert(!std::is_move_constructible_v<runtime_type>);
  static_assert(!std::is_move_assignable_v<runtime_type>);

  void require_fresh_process() {
    REQUIRE_FALSE(Kokkos::is_initialized());
    REQUIRE_FALSE(Kokkos::is_finalized());
  }

  void require_finalized() {
    REQUIRE_FALSE(Kokkos::is_initialized());
    REQUIRE(Kokkos::is_finalized());
  }

  // Test real execution and data movement, not merely the backend name.
  // All Views are destroyed before this helper returns to its Runtime owner.
  void exercise_execution() {
    int constexpr valueCount = 257;
    Kokkos::View<int*, typename execution_space_type::memory_space> values(
        "runtime_test_values", valueCount);
    auto hostValues = Kokkos::create_mirror_view(values);
    for (int index = 0; index < valueCount; ++index) { hostValues(index) = index; }
    Kokkos::deep_copy(values, hostValues);

    Kokkos::parallel_for(
        "runtime_test_update", Kokkos::RangePolicy<execution_space_type>(0, valueCount),
        KOKKOS_LAMBDA(int const index) { values(index) = 2 * values(index) + 3; });

    int sum = 0;
    Kokkos::parallel_reduce(
        "runtime_test_sum", Kokkos::RangePolicy<execution_space_type>(0, valueCount),
        KOKKOS_LAMBDA(int const index, int& partialSum) { partialSum += values(index); },
        sum);
    Kokkos::deep_copy(hostValues, values);

    REQUIRE(sum == valueCount * (valueCount + 2));
    for (int index = 0; index < valueCount; ++index) {
      REQUIRE(hostValues(index) == 2 * index + 3);
    }
  }

} // namespace

TEST_CASE("Runtime owns initialization", "[runtime_owned]") {
  require_fresh_process();
  {
    runtime_type runtime;
    REQUIRE(Kokkos::is_initialized());
    REQUIRE_FALSE(Kokkos::is_finalized());
    REQUIRE(runtime.ownsInitialization());
    REQUIRE(runtime.getNumThreads() == Kokkos::num_threads());
    REQUIRE(runtime.getDeviceId() == Kokkos::device_id());
    REQUIRE(runtime.getExecutionSpaceName() == execution_space_type::name());
    exercise_execution();
  }
  require_finalized();
}

TEST_CASE("Runtime shares initialization until the last owner", "[runtime_shared]") {
  require_fresh_process();
  auto first = std::make_unique<runtime_type>();
  auto second = std::make_unique<runtime_type>();
  REQUIRE(first->ownsInitialization());
  REQUIRE(second->ownsInitialization());
  REQUIRE(first->getNumThreads() == second->getNumThreads());
  REQUIRE(first->getDeviceId() == second->getDeviceId());
  first.reset();
  REQUIRE(Kokkos::is_initialized());
  REQUIRE_FALSE(Kokkos::is_finalized());
  exercise_execution();
  second.reset();
  require_finalized();
}

TEST_CASE("Runtime accepts the current explicit configuration",
          "[runtime_explicit_configuration]") {
  require_fresh_process();
  {
    runtime_type first;
    runtime_type second(first.getNumThreads(), first.getDeviceId());
    REQUIRE(second.getNumThreads() == first.getNumThreads());
    REQUIRE(second.getDeviceId() == first.getDeviceId());
    exercise_execution();
  }
  require_finalized();
}

TEST_CASE("Runtime rejects a conflicting configuration without ending its owner",
          "[runtime_conflicting_configuration]") {
  require_fresh_process();
  {
    runtime_type runtime(1);
    REQUIRE_THROWS_AS(runtime_type(runtime.getNumThreads() + 1), std::exception);
    int const differentDevice = runtime.getDeviceId() < 0 ? 0 : runtime.getDeviceId() + 1;
    REQUIRE_THROWS_AS(runtime_type(0, differentDevice), std::exception);
    REQUIRE(Kokkos::is_initialized());
    REQUIRE_FALSE(Kokkos::is_finalized());
    exercise_execution();
  }
  require_finalized();
}

TEST_CASE("Runtime borrows external initialization without finalizing it",
          "[runtime_borrowed]") {
  require_fresh_process();
  {
    Kokkos::ScopeGuard external(Kokkos::InitializationSettings().set_num_threads(1));
    {
      runtime_type first;
      runtime_type second(first.getNumThreads(), first.getDeviceId());
      REQUIRE_FALSE(first.ownsInitialization());
      REQUIRE_FALSE(second.ownsInitialization());
      exercise_execution();
    }
    REQUIRE(Kokkos::is_initialized());
    REQUIRE_FALSE(Kokkos::is_finalized());
    exercise_execution();
  }
  require_finalized();
}

TEST_CASE("Invalid Runtime options do not initialize Kokkos",
          "[runtime_invalid_options]") {
  require_fresh_process();
  REQUIRE_THROWS_AS(runtime_type(-1), std::exception);
  require_fresh_process();
  REQUIRE_THROWS_AS(runtime_type(0, -2), std::exception);
  require_fresh_process();

  if constexpr (std::is_same_v<execution_space_type, host_execution_space_type>) {
    REQUIRE_THROWS_AS(runtime_type(0, 0), std::exception);
    require_fresh_process();
  }
#ifdef KOKKOS_ENABLE_SERIAL
  if constexpr (std::is_same_v<host_execution_space_type, Kokkos::Serial>) {
    REQUIRE_THROWS_AS(runtime_type(2), std::exception);
    require_fresh_process();
  }
#endif

  {
    runtime_type runtime(1);
    exercise_execution();
  }
  require_finalized();
}

TEST_CASE("Runtime cannot restart Kokkos after finalization", "[runtime_finalized]") {
  require_fresh_process();
  {
    runtime_type runtime(1);
    exercise_execution();
  }
  require_finalized();
  REQUIRE_THROWS_AS(runtime_type(), std::exception);
  require_finalized();
}

TEST_CASE("Runtime creation stays on its coordinator thread",
          "[runtime_coordinator_thread]") {
  require_fresh_process();
  {
    runtime_type runtime(1);
    bool rejected = false;
    std::thread otherThread([&rejected] {
      try {
        runtime_type differentThreadRuntime;
      } catch (std::exception const&) { rejected = true; }
    });
    otherThread.join();
    REQUIRE(rejected);
    REQUIRE(Kokkos::is_initialized());
    exercise_execution();
  }
  require_finalized();
}

TEST_CASE("Runtime releases initialization during exception cleanup",
          "[runtime_exception_cleanup]") {
  require_fresh_process();
  bool caught = false;
  bool executed = false;
  try {
    runtime_type runtime(1);
    exercise_execution();
    executed = true;
    throw std::runtime_error("intentional runtime test exception");
  } catch (std::runtime_error const&) { caught = true; }
  REQUIRE(executed);
  REQUIRE(caught);
  require_finalized();
}

TEST_CASE("Runtime validates external configuration without changing its lifetime",
          "[runtime_external_configuration]") {
  require_fresh_process();
  {
    Kokkos::ScopeGuard external(Kokkos::InitializationSettings().set_num_threads(1));
    REQUIRE_THROWS_AS(runtime_type(Kokkos::num_threads() + 1), std::exception);
    REQUIRE(Kokkos::is_initialized());
    REQUIRE_FALSE(Kokkos::is_finalized());
    {
      runtime_type runtime(Kokkos::num_threads(), Kokkos::device_id());
      REQUIRE_FALSE(runtime.ownsInitialization());
      exercise_execution();
    }
    REQUIRE(Kokkos::is_initialized());
  }
  require_finalized();
}

TEST_CASE("Runtime executes with explicit host and device settings",
          "[runtime_explicit_execution]") {
  require_fresh_process();
  int requestedThreads = 2;
#ifdef KOKKOS_ENABLE_SERIAL
  if constexpr (std::is_same_v<host_execution_space_type, Kokkos::Serial>) {
    requestedThreads = 1;
  }
#endif
  int const requestedDevice =
      std::is_same_v<execution_space_type, host_execution_space_type> ? -1 : 0;
  {
    runtime_type runtime(requestedThreads, requestedDevice);
    REQUIRE(runtime.getNumThreads() == requestedThreads);
    REQUIRE(runtime.getDeviceId() == requestedDevice);
    exercise_execution();
  }
  require_finalized();
}

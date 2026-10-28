/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */
#include <corsika/kokkos/scheduling/PendingParticleQueue.hpp>
#include <corsika/kokkos/scheduling/WavefrontExecutionGuard.hpp>
#include <corsika/kokkos/runtime/Runtime.hpp>
#include <tests/kokkos/runtime/QueueReference.hpp>
#include <tests/kokkos/runtime/QueueChecks.hpp>
#include <catch2/catch_test_macros.hpp>

namespace {
  namespace cp = corsika::kokkos;
  namespace ref = queue_reference;
  using execution_space_type = Kokkos::DefaultExecutionSpace;
  using memory_space_type = typename execution_space_type::memory_space;
  void ensure_runtime() { static cp::Runtime runtime; }

  std::vector<cp::ParticleState> make_particles(std::size_t count) {
    std::vector<cp::ParticleState> result(count);
    for (std::size_t i = 0; i < count; ++i) {
      auto& p = result[i];
      p.pid_ = (i % 3 == 0 ? 22 : i % 3 == 1 ? 11 : -13);
      p.mediumId_ = static_cast<int>(i % 5);
      p.environmentNode_ = static_cast<int>(i % 7) - 3;
      p.generation_ = i % 11;
      p.reserved_ = i % 13;
      p.energyGeV_ = 0.125 * (i + 1);
      p.timeS_ = 0.0625 * i;
      p.weight_ = i % 3 == 0 ? 0. : 0.5 * (i + 1);
      p.historyId_ = (std::uint64_t{1} << 61) + i;
      p.parentHistoryId_ = (std::uint64_t{1} << 57) + (i / 2);
      p.stepId_ = (std::uint64_t{1} << 33) + i;
      for (int k = 0; k < 3; ++k) {
        p.positionM_[k] = -100. + i * 4. + k;
        p.direction_[k] = 0.25 * (k + 1);
      }
    }
    return result;
  }

  template <class TParticle>
  auto upload_array(std::vector<TParticle> const& particles,
                    execution_space_type const& execution) {
    Kokkos::View<TParticle*, memory_space_type> data("queue fixture", particles.size());
    auto host = Kokkos::create_mirror_view(data);
    for (std::size_t i = 0; i < particles.size(); ++i) host(i) = particles[i];
    Kokkos::deep_copy(execution, data, host);
    execution.fence("fixture input lifetime");
    return data;
  }
} // namespace

TEST_CASE("Production stable wavefront and count scans", "[kokkos][queue]") {
  ensure_runtime();
  execution_space_type execution;
  cp::WavefrontQueue<execution_space_type> current(512, execution);
  ref::KokkosWavefrontQueue<execution_space_type> original(512, execution);
  for (std::size_t count : {0, 1, 7, 33, 257}) {
    auto input = make_particles(count);
    auto oldInput = queue_checks::original(input);
    current.upload(input, execution);
    original.upload(oldInput, execution);
    if (count) {
      REQUIRE_THROWS_AS(current.clear(), std::logic_error);
      REQUIRE_THROWS_AS(original.clear(), std::logic_error);
      REQUIRE_THROWS_AS(current.ensureCapacity(513), std::logic_error);
      REQUIRE_THROWS_AS(original.ensureCapacity(513), std::logic_error);
    }
    queue_checks::equal(current.download(), original.download(), input);
    REQUIRE(current.deviceBytes() == original.deviceBytes());
    REQUIRE(current.hostPinnedBytes() == original.hostPinnedBytes());
    Kokkos::View<std::uint32_t*, memory_space_type> flags("flags", count);
    auto host = Kokkos::create_mirror_view(flags);
    std::vector<cp::ParticleState> selected;
    for (std::size_t i = 0; i < count; ++i) {
      host(i) = i % 3 == 1 ? 0U : 2U;
      if (host(i)) selected.push_back(input[i]);
    }
    Kokkos::deep_copy(execution, flags, host);
    REQUIRE(current.compact(flags) == selected.size());
    REQUIRE(original.compact(flags) == selected.size());
    queue_checks::equal(current.download(), original.download(), selected);
    for (std::size_t i = 0; i < count; ++i)
      host(i) = std::numeric_limits<std::uint32_t>::max() - i;
    Kokkos::deep_copy(execution, flags, host);
    std::uint64_t sum = 0;
    auto got = current.secondaryOffsets(flags);
    REQUIRE(got == original.secondaryOffsets(flags));
    auto prefix =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, current.offsets());
    auto oldPrefix =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, original.offsets());
    for (std::size_t i = 0; i < count; ++i) {
      REQUIRE(prefix(i) == sum);
      REQUIRE(oldPrefix(i) == sum);
      sum += host(i);
    }
    REQUIRE(got == sum);
    auto src = current.current().rawDeviceView();
    auto dst = current.next().rawDeviceView();
    auto oldSrc = original.current().rawDeviceView();
    auto oldDst = original.next().rawDeviceView();
    auto size = selected.size();
    Kokkos::parallel_for(
        "reverse next front",
        Kokkos::RangePolicy<execution_space_type>(execution, 0, size),
        KOKKOS_LAMBDA(std::size_t i) {
          dst.store(i, src.load(size - 1 - i));
          oldDst.store(i, oldSrc.load(size - 1 - i));
        });
    current.commitNext(size);
    original.commitNext(size);
    std::reverse(selected.begin(), selected.end());
    queue_checks::equal(current.download(), original.download(), selected);
  }
  REQUIRE_THROWS_AS(current.commitNext(513), std::length_error);
  REQUIRE_THROWS_AS(original.commitNext(513), std::length_error);
  auto projection = current.projectedCapacity(513);
  auto oldProjection = original.projectedCapacity(513);
  REQUIRE(projection.retainedBytes_ == oldProjection.retained_bytes);
  REQUIRE(projection.transientPeakBytes_ == oldProjection.transient_peak_bytes);
  current.ensureCapacity(513);
  original.ensureCapacity(513);
  REQUIRE(current.capacity() == original.capacity());
  REQUIRE(current.empty());
  REQUIRE(original.empty());
  REQUIRE_THROWS_AS(current.ensureCapacity(0), std::invalid_argument);
  REQUIRE_THROWS_AS(original.ensureCapacity(0), std::invalid_argument);
}

TEST_CASE("Production pending queue growth and prefix order", "[kokkos][queue]") {
  ensure_runtime();
  execution_space_type execution;
  auto input = make_particles(12);
  auto oldInput = queue_checks::original(input);
  auto data = upload_array(input, execution);
  auto oldData = upload_array(oldInput, execution);
  cp::PendingParticleQueue<execution_space_type> current("new");
  ref::KokkosPendingParticleQueue<execution_space_type> original("old");
  REQUIRE(current.tryAppend(data, 5, 32, execution));
  REQUIRE(original.tryAppend(oldData, 5, 32, execution));
  current.consume(3);
  original.consume(3);
  REQUIRE(current.tryAppend(data, 7, 32, execution));
  REQUIRE(original.tryAppend(oldData, 7, 32, execution));
  std::vector<cp::ParticleState> expected(input.begin() + 3, input.begin() + 5);
  expected.insert(expected.end(), input.begin(), input.begin() + 7);
  queue_checks::equal(current.downloadPrefix(9, execution),
                      original.downloadPrefix(9, execution), expected);
  REQUIRE(current.head() == original.head());
  REQUIRE(current.capacity() == original.capacity());
  REQUIRE(current.deviceBytes() == original.deviceBytes());
  REQUIRE_FALSE(current.tryAppend(data, 12, 10, execution));
  REQUIRE_FALSE(original.tryAppend(oldData, 12, 10, execution));
  REQUIRE_THROWS_AS(current.tryAppend(data, 13, 100, execution), std::length_error);
  REQUIRE_THROWS_AS(original.tryAppend(oldData, 13, 100, execution), std::length_error);
  REQUIRE_THROWS_AS(current.consume(10), std::length_error);
  REQUIRE_THROWS_AS(original.consume(10), std::length_error);
  std::size_t vetoCalls = 0, oldVetoCalls = 0;
  cp::ResidentMemoryBudgetGate veto{
      &vetoCalls, [](void* count, std::size_t, cp::MemoryProjection const&, char const*) {
        ++*static_cast<std::size_t*>(count);
        return false;
      }};
  ref::KokkosResidentMemoryBudgetGate oldVeto{
      &oldVetoCalls,
      [](void* count, std::size_t, ref::KokkosMemoryProjection const&, char const*) {
        ++*static_cast<std::size_t*>(count);
        return false;
      }};
  auto const capacity = current.capacity();
  REQUIRE_FALSE(current.tryAppend(data, 12, 32, execution, veto));
  REQUIRE_FALSE(original.tryAppend(oldData, 12, 32, execution, oldVeto));
  REQUIRE(vetoCalls == 1);
  REQUIRE(oldVetoCalls == 1);
  REQUIRE(current.capacity() == capacity);
  REQUIRE(original.capacity() == capacity);
  REQUIRE(current.tryAppend(data, 0, 32, execution, veto));
  REQUIRE(original.tryAppend(oldData, 0, 32, execution, oldVeto));
  REQUIRE(vetoCalls == 1);
  REQUIRE(oldVetoCalls == 1);
  queue_checks::equal(current.downloadPrefix(9, execution),
                      original.downloadPrefix(9, execution), expected);
  cp::WavefrontQueue<execution_space_type> front(32, execution);
  ref::KokkosWavefrontQueue<execution_space_type> oldFront(32, execution);
  auto suffix = make_particles(3);
  front.upload(suffix, current.view(), current.head(), current.size(), execution);
  oldFront.upload(queue_checks::original(suffix), original.view(), original.head(),
                  original.size(), execution);
  expected.insert(expected.end(), suffix.begin(), suffix.end());
  queue_checks::equal(front.download(), oldFront.download(), expected);
  front.upload({}, current.view(), current.head(), current.size(), execution);
  oldFront.upload({}, original.view(), original.head(), original.size(), execution);
  REQUIRE_THROWS_AS(front.clear(), std::logic_error);
  REQUIRE_THROWS_AS(oldFront.clear(), std::logic_error);
  expected.resize(9);
  queue_checks::equal(front.download(), oldFront.download(), expected);
  current.consume(9);
  original.consume(9);
  REQUIRE(current.empty());
  REQUIRE(original.empty());
  REQUIRE(current.head() == 0);
  REQUIRE(original.head() == 0);
}

TEST_CASE("Production queue guard and growth limits", "[kokkos][queue]") {
  ensure_runtime();
  execution_space_type execution;
  cp::WavefrontQueue<execution_space_type> current(8, execution);
  ref::KokkosWavefrontQueue<execution_space_type> original(8, execution);
  auto input = make_particles(4);
  try {
    cp::WavefrontExecutionGuard<execution_space_type> guard(current, execution);
    current.upload(input, execution);
    throw std::runtime_error("intentional queue unwind");
  } catch (std::runtime_error const&) {}
  try {
    ref::KokkosWavefrontExecutionGuard<execution_space_type> guard(original, execution);
    original.upload(queue_checks::original(input), execution);
    throw std::runtime_error("intentional queue unwind");
  } catch (std::runtime_error const&) {}
  queue_checks::equal(current.download(), original.download(), input);
  REQUIRE_NOTHROW(current.clear());
  REQUIRE_NOTHROW(original.clear());
  {
    cp::WavefrontExecutionGuard<execution_space_type> guard(current, execution);
    current.upload(input, execution);
    guard.release();
  }
  {
    ref::KokkosWavefrontExecutionGuard<execution_space_type> guard(original, execution);
    original.upload(queue_checks::original(input), execution);
    guard.release();
  }
  REQUIRE_THROWS_AS(current.clear(), std::logic_error);
  REQUIRE_THROWS_AS(original.clear(), std::logic_error);
  execution.fence("caller finishes released guards");
  current.markExecutionSynchronized();
  original.markExecutionSynchronized();
  queue_checks::equal(current.download(), original.download(), input);
  for (std::size_t capacity : {0, 1, 7, 64, 100})
    for (std::size_t requested : {0, 1, 7, 64, 100})
      REQUIRE(cp::checked_geometric_capacity(capacity, requested, 100) ==
              ref::checkedGeometricCapacity(capacity, requested, 100));
  REQUIRE_THROWS_AS(cp::checked_geometric_capacity(32, 101, 100), std::length_error);
  REQUIRE_THROWS_AS(ref::checkedGeometricCapacity(32, 101, 100), std::length_error);
}

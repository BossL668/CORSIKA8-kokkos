/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */
#include <corsika/kokkos/scheduling/WavefrontBucketing.hpp>
#include <corsika/kokkos/scheduling/WavefrontQueue.hpp>
#include <corsika/kokkos/runtime/Runtime.hpp>
#include <tests/kokkos/scheduling/BucketingReference.hpp>
#include <tests/kokkos/runtime/QueueChecks.hpp>
#include <catch2/catch_test_macros.hpp>
#include <algorithm>
#include <array>
#include <numeric>

namespace {
  namespace cp = corsika::kokkos;
  namespace ref = bucket_reference;
  using execution_space_type = Kokkos::DefaultExecutionSpace;
  using memory_space_type = typename execution_space_type::memory_space;
#if defined(CORSIKA_EXPECT_CUDA)
  static_assert(std::is_same_v<execution_space_type, Kokkos::Cuda>);
#elif defined(CORSIKA_EXPECT_OPENMP)
  static_assert(std::is_same_v<execution_space_type, Kokkos::OpenMP>);
#else
  static_assert(Kokkos::is_execution_space<execution_space_type>::value);
#endif

  void ensure_runtime() { static cp::Runtime runtime; }

  std::vector<cp::ParticleState> make_particles(std::size_t count) {
    std::vector<cp::ParticleState> result(count);
    for (std::size_t index = 0; index < count; ++index) {
      auto& particle = result[index];
      particle.pid_ = std::array<int, 5>{22, 11, -11, 13, -13}[index % 5];
      particle.mediumId_ = static_cast<int>((index / 5) % 7) - 3;
      particle.environmentNode_ = index % 13;
      particle.energyGeV_ = std::ldexp(1., static_cast<int>((index / 35) % 9) - 4);
      particle.generation_ = index % 17;
      particle.reserved_ = index % 19;
      particle.timeS_ = .125 * index;
      particle.weight_ = 1. + index % 3;
      particle.historyId_ = (std::uint64_t{1} << 63) + index;
      particle.parentHistoryId_ = (std::uint64_t{1} << 61) + index / 2;
      particle.stepId_ = (std::uint64_t{1} << 35) + index;
      for (unsigned int component = 0; component < 3; ++component) {
        particle.positionM_[component] = -100. + 4. * index + component;
        particle.direction_[component] = .25 * (component + 1);
      }
    }
    return result;
  }

  template <class TParticle, class TSoA>
  std::vector<TParticle> download(TSoA const& particles, std::size_t count,
                                  execution_space_type const& execution) {
    Kokkos::View<TParticle*, memory_space_type> values("bucketing readback", count);
    auto const raw = particles.rawDeviceView();
    Kokkos::parallel_for(
        "read bucketing", Kokkos::RangePolicy<execution_space_type>(execution, 0, count),
        KOKKOS_LAMBDA(std::size_t const index) { values(index) = raw.load(index); });
    execution.fence("read bucketing before host access");
    auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, values);
    std::vector<TParticle> result(count);
    for (std::size_t index = 0; index < count; ++index) result[index] = host(index);
    return result;
  }
} // namespace

TEST_CASE("Production bucket keys retain signed media and energy limits",
          "[kokkos][bucketing]") {
  ensure_runtime();
  execution_space_type execution;
  auto particles = make_particles(512);
  auto const extremes = std::array<double, 8>{0.,
                                              -1.,
                                              std::numeric_limits<double>::infinity(),
                                              std::numeric_limits<double>::quiet_NaN(),
                                              std::numeric_limits<double>::denorm_min(),
                                              std::numeric_limits<double>::max(),
                                              .5,
                                              1.};
  for (std::size_t index = 0; index < particles.size(); ++index) {
    particles[index].energyGeV_ = extremes[index % extremes.size()];
    if (index % 5 == 0)
      particles[index].mediumId_ = std::numeric_limits<std::int32_t>::min();
    if (index % 5 == 1)
      particles[index].mediumId_ = std::numeric_limits<std::int32_t>::max();
  }
  cp::WavefrontQueue<execution_space_type> current(512, execution);
  ref::KokkosWavefrontQueue<execution_space_type> original(512, execution);
  current.upload(particles, execution);
  original.upload(queue_checks::original(particles), execution);
  auto input = current.current().rawDeviceView();
  auto oldInput = original.current().rawDeviceView();
  Kokkos::View<std::uint64_t * [2], memory_space_type> keys("bucket keys", 512);
  Kokkos::parallel_for(
      "compare production keys",
      Kokkos::RangePolicy<execution_space_type>(execution, 0, 512),
      KOKKOS_LAMBDA(std::size_t const index) {
        keys(index, 0) = cp::wavefront_bucket_key(input.load(index));
        keys(index, 1) = ref::kokkosWavefrontBucketKey(oldInput.load(index));
      });
  execution.fence("finish key comparison");
  current.markExecutionSynchronized();
  original.markExecutionSynchronized();
  auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, keys);
  for (std::size_t index = 0; index < 512; ++index)
    REQUIRE(host(index, 0) == host(index, 1));
}

TEST_CASE("Production stable bucketing preserves every particle field and capacity",
          "[kokkos][bucketing]") {
  ensure_runtime();
  execution_space_type execution;
  cp::WavefrontQueue<execution_space_type> current(1025, execution);
  ref::KokkosWavefrontQueue<execution_space_type> original(1025, execution);
  cp::WavefrontBucketingWorkspace<execution_space_type> workspace;
  ref::KokkosWavefrontBucketingWorkspace<execution_space_type> oldWorkspace;
  for (std::size_t count : {0, 1, 255, 256, 257, 512, 1025, 256}) {
    auto particles = make_particles(count);
    current.upload(particles, execution);
    auto oldParticles = queue_checks::original(particles);
    original.upload(oldParticles, execution);
    auto projection = workspace.projectedCapacity(count, execution);
    auto oldProjection = oldWorkspace.projectedCapacity(count, execution);
    REQUIRE(projection.retainedBytes_ == oldProjection.retained_bytes);
    REQUIRE(projection.transientPeakBytes_ == oldProjection.transient_peak_bytes);
    auto batch = cp::bucket_wavefront(current.current(), count, workspace, execution, 7);
    auto oldBatch =
        ref::bucketKokkosWavefront(original.current(), count, oldWorkspace, execution, 7);
    execution.fence("finish both bucket sorts");
    current.markExecutionSynchronized();
    original.markExecutionSynchronized();
    REQUIRE(batch.count_ == oldBatch.count);
    REQUIRE(batch.sorted_ == oldBatch.sorted);
    REQUIRE(batch.sorted_ == (count >= 256));
    REQUIRE(workspace.deviceBytes() == oldWorkspace.deviceBytes());
    REQUIRE(workspace.deviceBytes() == projection.retainedBytes_);
    auto expected = particles;
    if (count >= 256) {
      std::vector<std::size_t> order(count);
      std::iota(order.begin(), order.end(), 0);
      std::stable_sort(order.begin(), order.end(), [&](auto left, auto right) {
        return ref::kokkosWavefrontBucketKey(oldParticles[left]) <
               ref::kokkosWavefrontBucketKey(oldParticles[right]);
      });
      for (std::size_t index = 0; index < count; ++index)
        expected[index] = particles[order[index]];
      REQUIRE_THROWS_AS(
          cp::bucket_wavefront(batch.particles_, count, workspace, execution),
          std::invalid_argument);
      REQUIRE_THROWS_AS(
          ref::bucketKokkosWavefront(oldBatch.particles, count, oldWorkspace, execution),
          std::invalid_argument);
    } else {
      REQUIRE(batch.particles_.historyId_.data() == current.current().historyId_.data());
      REQUIRE(oldBatch.particles.history_id.data() ==
              original.current().history_id.data());
    }
    queue_checks::equal(
        download<cp::ParticleState>(batch.particles_, count, execution),
        download<ref::EmParticleState>(oldBatch.particles, count, execution), expected);
    queue_checks::equal(
        download<cp::ParticleState>(current.current(), count, execution),
        download<ref::EmParticleState>(original.current(), count, execution), particles);
    REQUIRE(workspace.cudaRadixSortAvailable() == oldWorkspace.cudaRadixSortAvailable());
    REQUIRE(workspace.portableCapacity() == oldWorkspace.portableCapacity());
    REQUIRE(workspace.radixCapacity() == oldWorkspace.radixCapacity());
  }
  REQUIRE_THROWS_AS(cp::bucket_wavefront(current.current(), 1026, workspace, execution),
                    std::length_error);
  REQUIRE_THROWS_AS(
      ref::bucketKokkosWavefront(original.current(), 1026, oldWorkspace, execution),
      std::length_error);
  auto projection = workspace.projectedCountRange(2051, execution);
  auto oldProjection = oldWorkspace.projectedCountRange(2051, execution);
  REQUIRE(projection.retainedBytes_ == oldProjection.retained_bytes);
  REQUIRE(projection.transientPeakBytes_ == oldProjection.transient_peak_bytes);
  workspace.ensureCountRange(2051, execution);
  oldWorkspace.ensureCountRange(2051, execution);
  execution.fence("complete reserved range");
  REQUIRE(workspace.deviceBytes() == oldWorkspace.deviceBytes());
  REQUIRE(workspace.deviceBytes() == projection.retainedBytes_);
}

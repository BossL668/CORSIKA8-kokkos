/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include <Kokkos_Core.hpp>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <corsika/accelerator/em/kokkos/KokkosWavefrontQueue.hpp>
#include <corsika/accelerator/em/common/Types.hpp>

namespace {

  using Particle = corsika::gpu::em::EmParticleState;
  using ExecutionSpace = Kokkos::DefaultExecutionSpace;
  using PendingQueue = corsika::accelerator::em::kokkos_detail::
      KokkosPendingParticleQueue<ExecutionSpace>;
  using WavefrontQueue = corsika::accelerator::em::kokkos_detail::
      KokkosWavefrontQueue<ExecutionSpace>;

  static_assert(std::is_trivially_copyable_v<Particle>);
  static_assert(std::is_same_v<WavefrontQueue::host_staging_space,
                               Kokkos::SharedHostPinnedSpace>);

  double doubleFromBits(std::uint64_t const bits) {
    double value{};
    std::memcpy(&value, &bits, sizeof(value));
    return value;
  }

  bool equalDoubleBits(double const left, double const right) {
    std::uint64_t left_bits{};
    std::uint64_t right_bits{};
    std::memcpy(&left_bits, &left, sizeof(left));
    std::memcpy(&right_bits, &right, sizeof(right));
    return left_bits == right_bits;
  }

  Particle makeParticle(std::uint64_t const history) {
    Particle particle{};
    particle.pid = history % 2 == 0 ? 22 : 11;
    particle.medium_id = static_cast<std::int32_t>(history % 3);
    particle.generation = static_cast<std::uint32_t>(history + 10);
    particle.reserved = static_cast<std::uint32_t>(0xa5000000U + history);
    particle.energy_GeV = 0.25 * static_cast<double>(history);
    particle.time_s = 1.e-9 * static_cast<double>(history);
    particle.weight = 1. + 0.125 * static_cast<double>(history);
    particle.history_id = history;
    particle.parent_history_id = history + 1000;
    particle.step_id = history + 2000;
    for (int axis = 0; axis < 3; ++axis) {
      particle.position_m[axis] =
          static_cast<double>(10 * history + axis);
      particle.direction[axis] =
          static_cast<double>(100 * history + axis);
    }
    return particle;
  }

  Particle makeBitPatternParticle(std::uint64_t const history) {
    auto particle = makeParticle(history);
    particle.pid = std::numeric_limits<std::int32_t>::min() + 17;
    particle.medium_id = std::numeric_limits<std::int32_t>::max() - 23;
    particle.generation = 0xfedcba98U;
    particle.reserved = 0x89abcdefU;
    particle.energy_GeV = doubleFromBits(0x8000000000000000ULL);
    particle.position_m[0] = doubleFromBits(0x0000000000000001ULL);
    particle.position_m[1] = doubleFromBits(0x7fefffffffffffffULL);
    particle.position_m[2] = doubleFromBits(0xffefffffffffffffULL);
    particle.direction[0] = doubleFromBits(0x3ff0000000000001ULL);
    particle.direction[1] = doubleFromBits(0xbff0000000000001ULL);
    particle.direction[2] = doubleFromBits(0x0010000000000000ULL);
    particle.time_s = doubleFromBits(0x8000000000000001ULL);
    particle.weight = doubleFromBits(0x7fd23456789abcdeULL);
    particle.history_id = 0xfedcba9876543210ULL;
    particle.parent_history_id = 0x0123456789abcdefULL;
    particle.step_id = 0x8877665544332211ULL;
    return particle;
  }

  bool equalParticle(Particle const& left, Particle const& right) {
    if (left.pid != right.pid || left.medium_id != right.medium_id ||
        left.generation != right.generation ||
        left.reserved != right.reserved ||
        !equalDoubleBits(left.energy_GeV, right.energy_GeV) ||
        !equalDoubleBits(left.time_s, right.time_s) ||
        !equalDoubleBits(left.weight, right.weight) ||
        left.history_id != right.history_id ||
        left.parent_history_id != right.parent_history_id ||
        left.step_id != right.step_id)
      return false;
    for (int axis = 0; axis < 3; ++axis) {
      if (!equalDoubleBits(left.position_m[axis], right.position_m[axis]) ||
          !equalDoubleBits(left.direction[axis], right.direction[axis]))
        return false;
    }
    return true;
  }

  PendingQueue::View uploadParticles(
      char const* label, std::vector<Particle> const& particles,
      ExecutionSpace const& execution) {
    PendingQueue::View device(
        Kokkos::view_alloc(execution, Kokkos::WithoutInitializing,
                           std::string{label}),
        particles.size());
    auto host = Kokkos::create_mirror_view(device);
    for (std::size_t index = 0; index < particles.size(); ++index)
      host(index) = particles[index];
    Kokkos::deep_copy(execution, device, host);
    execution.fence("upload pending-queue test particles");
    return device;
  }

  std::vector<Particle> downloadPending(PendingQueue const& queue) {
    if (queue.empty()) return {};
    auto active = Kokkos::subview(
        queue.view(), std::make_pair(queue.head(), queue.head() + queue.size()));
    auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), active);
    std::vector<Particle> result(queue.size());
    for (std::size_t index = 0; index < result.size(); ++index)
      result[index] = host(index);
    return result;
  }

  void requireSequence(std::vector<Particle> const& actual,
                       std::vector<Particle> const& expected,
                       char const* message) {
    if (actual.size() != expected.size()) throw std::runtime_error(message);
    for (std::size_t index = 0; index < actual.size(); ++index) {
      if (!equalParticle(actual[index], expected[index]))
        throw std::runtime_error(message);
    }
  }

  bool denyResidentGrowth(
      void*, std::size_t,
      corsika::accelerator::em::kokkos_detail::KokkosMemoryProjection const&,
      char const*) {
    return false;
  }

} // namespace

int main(int argc, char** argv) {
  try {
    Kokkos::ScopeGuard guard(argc, argv);
    ExecutionSpace execution{};

    using corsika::accelerator::em::kokkos_detail::checkedGeometricCapacity;
    auto const maximum = std::numeric_limits<std::size_t>::max();
    if (checkedGeometricCapacity(0, 3) != 3 ||
        checkedGeometricCapacity(3, 4) != 6 ||
        checkedGeometricCapacity(6, 5) != 6 ||
        checkedGeometricCapacity(maximum - 4, maximum - 2) != maximum)
      throw std::runtime_error(
          "overflow-safe geometric capacity calculation is incorrect");
    bool bounded_growth_rejected = false;
    try {
      (void)checkedGeometricCapacity(4, 7, 6);
    } catch (std::length_error const&) {
      bounded_growth_rejected = true;
    }
    if (!bounded_growth_rejected)
      throw std::runtime_error(
          "geometric capacity accepted a request above its hard limit");

    std::vector<Particle> first{
        makeParticle(1), makeParticle(2), makeParticle(3), makeParticle(4)};
    std::vector<Particle> second{makeParticle(5), makeParticle(6)};
    auto first_device = uploadParticles("pending_first", first, execution);
    auto second_device = uploadParticles("pending_second", second, execution);

    PendingQueue pending("pending_contract");
    auto const pending_initial_projection =
        pending.projectedTailCapacity(first.size(), 6);
    if (pending_initial_projection.retained_bytes !=
            first.size() * sizeof(Particle) ||
        pending_initial_projection.transient_peak_bytes <
            pending_initial_projection.retained_bytes)
      throw std::runtime_error(
          "initial pending allocation projection is incorrect");
    if (!pending.tryAppend(first_device, first.size(), 6, execution))
      throw std::runtime_error("initial pending append was rejected");
    if (pending.deviceBytes() != pending_initial_projection.retained_bytes)
      throw std::runtime_error(
          "pending allocation projection differs from retained storage");
    requireSequence(
        pending.downloadPrefix(2, execution), {first[0], first[1]},
        "pending prefix download changed FIFO order");
    pending.consume(2);
    if (!pending.tryAppend(second_device, second.size(), 6, execution))
      throw std::runtime_error("compact-and-append was rejected");
    requireSequence(downloadPending(pending),
                    {first[2], first[3], second[0], second[1]},
                    "pending FIFO order changed during compaction");

    auto const capacity_before_rejection = pending.capacity();
    if (pending.tryAppend(first_device, 3, 6, execution))
      throw std::runtime_error("over-limit pending append was accepted");
    if (pending.capacity() != capacity_before_rejection || pending.size() != 4)
      throw std::runtime_error("rejected append modified pending state");
    requireSequence(downloadPending(pending),
                    {first[2], first[3], second[0], second[1]},
                    "rejected append modified pending contents");

    bool consume_rejected = false;
    try {
      pending.consume(5);
    } catch (std::length_error const&) {
      consume_rejected = true;
    }
    if (!consume_rejected)
      throw std::runtime_error("over-consume was not rejected");

    PendingQueue bounded("pending_bounded_growth");
    if (!bounded.tryAppend(first_device, first.size(), 6, execution) ||
        !bounded.tryAppend(second_device, second.size(), 6, execution))
      throw std::runtime_error("bounded pending growth was rejected");
    if (bounded.capacity() != 6 || bounded.size() != 6)
      throw std::runtime_error(
          "pending allocation overshot its configured capacity");

    PendingQueue geometric_pending("pending_geometric_growth");
    if (!geometric_pending.tryAppend(first_device, 3, 100, execution) ||
        geometric_pending.capacity() != 3 ||
        !geometric_pending.tryAppend(second_device, 1, 100, execution) ||
        geometric_pending.capacity() != 6)
      throw std::runtime_error(
          "pending queue did not use bounded geometric growth");
    requireSequence(downloadPending(geometric_pending),
                    {first[0], first[1], first[2], second[0]},
                    "pending geometric growth changed FIFO contents");

    // A predictive memory-budget rejection must occur before the queue View
    // is replaced and must leave both the logical FIFO and its allocation
    // unchanged. This is the contract used by the resident-cascade spill
    // path to hand the same source order back to the scalar CPU router.
    PendingQueue budget_rejected("pending_budget_rejected");
    auto const rejected_projection =
        budget_rejected.projectedTailCapacity(first.size(), 8);
    corsika::accelerator::em::kokkos_detail::
        KokkosResidentMemoryBudgetGate deny_gate{
            nullptr, &denyResidentGrowth};
    if (budget_rejected.tryAppend(
            first_device, first.size(), 8, execution, deny_gate,
            "pending queue unit-test denial"))
      throw std::runtime_error(
          "memory-budget gate accepted a denied pending allocation");
    if (!budget_rejected.empty() || budget_rejected.capacity() != 0 ||
        budget_rejected.deviceBytes() != 0 ||
        rejected_projection.retained_bytes == 0)
      throw std::runtime_error(
          "memory-budget rejection modified pending queue state");

    std::vector<Particle> host_suffix{makeParticle(7), makeParticle(8)};
    WavefrontQueue mixed(8, execution);
    mixed.upload(host_suffix, pending.view(), pending.head(), pending.size(),
                 execution);
    requireSequence(mixed.download(),
                    {first[2], first[3], second[0], second[1],
                     host_suffix[0], host_suffix[1]},
                    "pending-prefix/host-suffix upload order changed");

    WavefrontQueue pending_only(8, execution);
    pending_only.upload({}, pending.view(), pending.head(), pending.size(),
                        execution);
    bool rejected_prefix_only_rebind = false;
    try {
      pending_only.ensureCapacity(8, execution);
    } catch (std::logic_error const&) {
      rejected_prefix_only_rebind = true;
    }
    if (!rejected_prefix_only_rebind)
      throw std::runtime_error(
          "prefix-only upload did not retain execution ownership");
    requireSequence(pending_only.download(),
                    {first[2], first[3], second[0], second[1]},
                    "pending-only upload changed particle state");

    // The AoS staging transfer must preserve every field bit-for-bit,
    // including signed zero, subnormal values and extreme finite values.
    auto const bit_particle = makeBitPatternParticle(9);
    WavefrontQueue bit_exact(2, execution);
    bit_exact.upload({bit_particle});
    requireSequence(bit_exact.download(), {bit_particle},
                    "AoS staging changed particle field bits");

    // Device storage is grow-only and is reused by smaller later fronts.
    WavefrontQueue projected_queue;
    auto const queue_projection = projected_queue.projectedCapacity(2);
    projected_queue.ensureCapacity(2, execution);
    if (queue_projection.retained_bytes != projected_queue.deviceBytes() ||
        queue_projection.transient_peak_bytes <
            queue_projection.retained_bytes)
      throw std::runtime_error(
          "wavefront allocation projection differs from retained storage");
    WavefrontQueue reusable(2, execution);
    auto const initial_bytes = reusable.deviceBytes();
    auto const initial_host_pinned_bytes = reusable.hostPinnedBytes();
    if (initial_host_pinned_bytes != 2 * sizeof(Particle))
      throw std::runtime_error(
          "initial pinned AoS staging capacity is incorrect");
    reusable.upload({makeParticle(10), makeParticle(11)});
    requireSequence(reusable.download(),
                    {makeParticle(10), makeParticle(11)},
                    "initial reusable wavefront changed particle state");
    reusable.ensureCapacity(16, execution);
    auto const grown_bytes = reusable.deviceBytes();
    auto const grown_host_pinned_bytes = reusable.hostPinnedBytes();
    if (reusable.capacity() != 16 || grown_bytes <= initial_bytes ||
        grown_host_pinned_bytes <= initial_host_pinned_bytes ||
        grown_host_pinned_bytes != 16 * sizeof(Particle))
      throw std::runtime_error("AoS staging storage did not grow with queue");
    auto const unchanged_projection = reusable.projectedCapacity(4);
    if (unchanged_projection.retained_bytes != grown_bytes ||
        unchanged_projection.transient_peak_bytes != grown_bytes)
      throw std::runtime_error(
          "sub-capacity wavefront projection predicted an allocation");
    std::vector<Particle> large_front;
    for (std::uint64_t history = 20; history < 32; ++history)
      large_front.push_back(makeParticle(history));
    reusable.upload(large_front);
    requireSequence(reusable.download(), large_front,
                    "grown AoS staging changed particle order or state");
    reusable.ensureCapacity(4, execution);
    if (reusable.capacity() != 16 || reusable.deviceBytes() != grown_bytes ||
        reusable.hostPinnedBytes() != grown_host_pinned_bytes)
      throw std::runtime_error("wavefront staging unexpectedly shrank");
    reusable.upload({bit_particle});
    requireSequence(reusable.download(), {bit_particle},
                    "reused AoS staging changed particle field bits");

    // Empty upload/download must remain a pure host-side no-op while keeping
    // all grow-only allocations available for the next non-empty front.
    auto const bytes_before_empty = reusable.deviceBytes();
    auto const host_pinned_bytes_before_empty = reusable.hostPinnedBytes();
    reusable.upload({});
    if (!reusable.empty() || !reusable.download().empty() ||
        reusable.deviceBytes() != bytes_before_empty ||
        reusable.hostPinnedBytes() != host_pinned_bytes_before_empty)
      throw std::runtime_error("empty wavefront path changed queue storage");

    WavefrontQueue geometric_wavefront(3, execution);
    geometric_wavefront.upload(
        {makeParticle(40), makeParticle(41), makeParticle(42)}, execution);
    bool rejected_in_flight_growth = false;
    try {
      geometric_wavefront.ensureCapacity(4, execution);
    } catch (std::logic_error const&) {
      rejected_in_flight_growth = true;
    }
    if (!rejected_in_flight_growth)
      throw std::runtime_error(
          "wavefront queue allowed pinned staging growth while H2D was in flight");
    execution.fence("complete geometric wavefront test upload");
    geometric_wavefront.markExecutionSynchronized();
    geometric_wavefront.ensureCapacity(4, execution);
    if (geometric_wavefront.capacity() != 6 ||
        !geometric_wavefront.empty())
      throw std::runtime_error(
          "wavefront queue did not geometrically grow and reset");
    geometric_wavefront.upload(
        {makeParticle(43), makeParticle(44), makeParticle(45),
         makeParticle(46)},
        execution);
    auto const geometric_bytes = geometric_wavefront.deviceBytes();
    bool rejected_in_flight_rebind = false;
    try {
      geometric_wavefront.ensureCapacity(5, execution);
    } catch (std::logic_error const&) {
      rejected_in_flight_rebind = true;
    }
    if (!rejected_in_flight_rebind)
      throw std::runtime_error(
          "wavefront queue allowed execution rebinding while H2D was in flight");
    execution.fence("complete sub-capacity wavefront test upload");
    geometric_wavefront.markExecutionSynchronized();
    geometric_wavefront.ensureCapacity(5, execution);
    if (geometric_wavefront.capacity() != 6 ||
        geometric_wavefront.deviceBytes() != geometric_bytes)
      throw std::runtime_error(
          "sub-capacity request changed geometric wavefront storage");
    requireSequence(
        geometric_wavefront.download(execution),
        {makeParticle(43), makeParticle(44), makeParticle(45),
         makeParticle(46)},
        "sub-capacity request changed wavefront logical contents");

    // Exception unwinding must drain the owning execution and make a
    // persistent queue reusable rather than leaving it permanently gated.
    WavefrontQueue guarded(4, execution);
    {
      corsika::accelerator::em::kokkos_detail::
          KokkosWavefrontExecutionGuard<ExecutionSpace> completion_guard(
              guarded, execution);
      guarded.upload({makeParticle(50)}, execution);
    }
    guarded.ensureCapacity(4, execution);
    requireSequence(guarded.download(execution), {makeParticle(50)},
                    "exception guard changed or lost queued particle state");

    pending.clear();
    if (!pending.empty() || pending.head() != 0 ||
        pending.capacity() != capacity_before_rejection)
      throw std::runtime_error("clear did not retain pending allocation");
  } catch (std::exception const& error) {
    std::cerr << "Kokkos pending-particle queue failed: " << error.what()
              << '\n';
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}

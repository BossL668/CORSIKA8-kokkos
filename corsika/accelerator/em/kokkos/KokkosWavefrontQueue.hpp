/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <corsika/accelerator/em/common/Types.hpp>
#include <corsika/accelerator/em/kokkos/KokkosPhysicsContext.hpp>
#include <corsika/accelerator/em/kokkos/KokkosResidentMemoryBudget.hpp>

namespace corsika::accelerator::em::kokkos_detail {

  /**
   * Return a grow-only geometric allocation size without overflowing.
   *
   * `requested` is the logical capacity required by the current operation;
   * the returned value is only an allocation reserve.  Keeping those two
   * quantities separate lets resident cascades retain their existing
   * per-wavefront overflow/checkpoint tests while avoiding a complete queue
   * reallocation for every small increase in the front size.
   */
  inline std::size_t checkedGeometricCapacity(
      std::size_t const current, std::size_t const requested,
      std::size_t const maximum = std::numeric_limits<std::size_t>::max()) {
    if (requested > maximum)
      throw std::length_error(
          "Kokkos grow-only allocation exceeds configured capacity");
    if (requested <= current) return current;
    if (current == 0) return requested;
    if (current > maximum / 2) return maximum;
    return std::min(maximum, std::max(requested, current * 2));
  }

  struct ParticleSoARawView {
    std::int32_t* pid{};
    std::int32_t* medium_id{};
    std::uint32_t* generation{};
    std::uint32_t* reserved{};
    double* energy_GeV{};
    RawView2D<double> position_m{};
    RawView2D<double> direction{};
    double* time_s{};
    double* weight{};
    std::uint64_t* history_id{};
    std::uint64_t* parent_history_id{};
    std::uint64_t* step_id{};

    KOKKOS_INLINE_FUNCTION gpu::em::EmParticleState load(
        std::size_t const index) const {
      gpu::em::EmParticleState state{};
      state.pid = pid[index];
      state.medium_id = medium_id[index];
      state.generation = generation[index];
      state.reserved = reserved[index];
      state.energy_GeV = energy_GeV[index];
      state.time_s = time_s[index];
      state.weight = weight[index];
      state.history_id = history_id[index];
      state.parent_history_id = parent_history_id[index];
      state.step_id = step_id[index];
      for (int axis = 0; axis < 3; ++axis) {
        state.position_m[axis] = position_m(index, axis);
        state.direction[axis] = direction(index, axis);
      }
      return state;
    }

    KOKKOS_INLINE_FUNCTION void store(
        std::size_t const index,
        gpu::em::EmParticleState const& state) const {
      pid[index] = state.pid;
      medium_id[index] = state.medium_id;
      generation[index] = state.generation;
      reserved[index] = state.reserved;
      energy_GeV[index] = state.energy_GeV;
      time_s[index] = state.time_s;
      weight[index] = state.weight;
      history_id[index] = state.history_id;
      parent_history_id[index] = state.parent_history_id;
      step_id[index] = state.step_id;
      for (int axis = 0; axis < 3; ++axis) {
        position_m(index, axis) = state.position_m[axis];
        direction(index, axis) = state.direction[axis];
      }
    }
  };

  static_assert(std::is_standard_layout_v<ParticleSoARawView>);
  static_assert(std::is_trivially_copyable_v<ParticleSoARawView>);
  static_assert(sizeof(ParticleSoARawView) <= 160);

  template <class MemorySpace>
  struct ParticleSoA {
    using execution_space = typename MemorySpace::execution_space;

    Kokkos::View<std::int32_t*, MemorySpace> pid;
    Kokkos::View<std::int32_t*, MemorySpace> medium_id;
    Kokkos::View<std::uint32_t*, MemorySpace> generation;
    Kokkos::View<std::uint32_t*, MemorySpace> reserved;
    Kokkos::View<double*, MemorySpace> energy_GeV;
    Kokkos::View<double*[3], MemorySpace> position_m;
    Kokkos::View<double*[3], MemorySpace> direction;
    Kokkos::View<double*, MemorySpace> time_s;
    Kokkos::View<double*, MemorySpace> weight;
    Kokkos::View<std::uint64_t*, MemorySpace> history_id;
    Kokkos::View<std::uint64_t*, MemorySpace> parent_history_id;
    Kokkos::View<std::uint64_t*, MemorySpace> step_id;

    ParticleSoA() = default;

    static std::size_t deviceBytesForCapacity(
        std::size_t const capacity) {
      constexpr auto bytes_per_particle =
          2 * sizeof(std::int32_t) + 2 * sizeof(std::uint32_t) +
          9 * sizeof(double) + 3 * sizeof(std::uint64_t);
      return checkedMemoryMultiply(capacity, bytes_per_particle);
    }

    ParticleSoA(std::string const& label, std::size_t const capacity)
        : ParticleSoA(label, capacity, execution_space{}) {}

    ParticleSoA(std::string const& label, std::size_t const capacity,
                execution_space const& execution)
        : pid(Kokkos::view_alloc(
                  execution, Kokkos::WithoutInitializing, label + "_pid"),
              capacity)
        , medium_id(Kokkos::view_alloc(
                        execution, Kokkos::WithoutInitializing,
                        label + "_medium"),
                    capacity)
        , generation(Kokkos::view_alloc(
                         execution, Kokkos::WithoutInitializing,
                         label + "_generation"),
                     capacity)
        , reserved(Kokkos::view_alloc(
                       execution, Kokkos::WithoutInitializing,
                       label + "_reserved"),
                   capacity)
        , energy_GeV(Kokkos::view_alloc(
                         execution, Kokkos::WithoutInitializing,
                         label + "_energy"),
                     capacity)
        , position_m(Kokkos::view_alloc(
                         execution, Kokkos::WithoutInitializing,
                         label + "_position"),
                     capacity)
        , direction(Kokkos::view_alloc(
                        execution, Kokkos::WithoutInitializing,
                        label + "_direction"),
                    capacity)
        , time_s(Kokkos::view_alloc(
                     execution, Kokkos::WithoutInitializing, label + "_time"),
                 capacity)
        , weight(Kokkos::view_alloc(
                     execution, Kokkos::WithoutInitializing,
                     label + "_weight"),
                 capacity)
        , history_id(Kokkos::view_alloc(
                         execution, Kokkos::WithoutInitializing,
                         label + "_history"),
                     capacity)
        , parent_history_id(
              Kokkos::view_alloc(
                  execution, Kokkos::WithoutInitializing,
                  label + "_parent_history"),
              capacity)
        , step_id(Kokkos::view_alloc(
                      execution, Kokkos::WithoutInitializing, label + "_step"),
                  capacity) {}

    KOKKOS_INLINE_FUNCTION gpu::em::EmParticleState load(
        std::size_t const index) const {
      gpu::em::EmParticleState state{};
      state.pid = pid(index);
      state.medium_id = medium_id(index);
      state.generation = generation(index);
      state.reserved = reserved(index);
      state.energy_GeV = energy_GeV(index);
      state.time_s = time_s(index);
      state.weight = weight(index);
      state.history_id = history_id(index);
      state.parent_history_id = parent_history_id(index);
      state.step_id = step_id(index);
      for (int axis = 0; axis < 3; ++axis) {
        state.position_m[axis] = position_m(index, axis);
        state.direction[axis] = direction(index, axis);
      }
      return state;
    }

    KOKKOS_INLINE_FUNCTION void store(
        std::size_t const index,
        gpu::em::EmParticleState const& state) const {
      pid(index) = state.pid;
      medium_id(index) = state.medium_id;
      generation(index) = state.generation;
      reserved(index) = state.reserved;
      energy_GeV(index) = state.energy_GeV;
      time_s(index) = state.time_s;
      weight(index) = state.weight;
      history_id(index) = state.history_id;
      parent_history_id(index) = state.parent_history_id;
      step_id(index) = state.step_id;
      for (int axis = 0; axis < 3; ++axis) {
        position_m(index, axis) = state.position_m[axis];
        direction(index, axis) = state.direction[axis];
      }
    }

    ParticleSoARawView rawDeviceView() const noexcept {
      return {
          pid.data(),
          medium_id.data(),
          generation.data(),
          reserved.data(),
          energy_GeV.data(),
          kokkos_detail::rawDeviceView(position_m),
          kokkos_detail::rawDeviceView(direction),
          time_s.data(),
          weight.data(),
          history_id.data(),
          parent_history_id.data(),
          step_id.data()};
    }

    std::size_t deviceBytes() const noexcept {
      auto const view_bytes = [](auto const& view) {
        return view.span() *
               sizeof(typename std::decay_t<decltype(view)>::value_type);
      };
      return view_bytes(pid) + view_bytes(medium_id) +
             view_bytes(generation) + view_bytes(reserved) +
             view_bytes(energy_GeV) + view_bytes(position_m) +
             view_bytes(direction) + view_bytes(time_s) + view_bytes(weight) +
             view_bytes(history_id) + view_bytes(parent_history_id) +
             view_bytes(step_id);
    }
  };

  template <class ExecutionSpace>
  struct QueueScanFunctor {
    using memory_space = typename ExecutionSpace::memory_space;
    Kokkos::View<std::uint32_t const*, memory_space> flags;
    Kokkos::View<std::uint64_t*, memory_space> offsets;

    KOKKOS_INLINE_FUNCTION void operator()(
        std::size_t const index, std::uint64_t& update,
        bool const final) const {
      if (final) offsets(index) = update;
      update += flags(index) != 0U ? 1U : 0U;
    }
  };

  template <class ExecutionSpace>
  struct QueueScatterFunctor {
    ParticleSoARawView input;
    ParticleSoARawView output;
    std::uint32_t const* flags{};
    std::uint64_t const* offsets{};

    KOKKOS_INLINE_FUNCTION void operator()(std::size_t const index) const {
      if (flags[index] != 0U) output.store(offsets[index], input.load(index));
    }
  };

  template <class ExecutionSpace>
  struct CountScanFunctor {
    using memory_space = typename ExecutionSpace::memory_space;
    Kokkos::View<std::uint32_t const*, memory_space> counts;
    Kokkos::View<std::uint64_t*, memory_space> offsets;

    KOKKOS_INLINE_FUNCTION void operator()(
        std::size_t const index, std::uint64_t& update,
        bool const final) const {
      if (final) offsets(index) = update;
      update += counts(index);
    }
  };

  /**
   * Grow-only device queue used to retain photon/lepton secondaries between
   * host-visible resident-cascade calls.
   *
   * The logical order is always the stable materialization order.  No atomic
   * append is used: the resident cascade has already assigned deterministic
   * offsets with an exclusive scan, so a contiguous device-to-device copy is
   * sufficient.  A non-zero head is compacted only when tail capacity is
   * exhausted.
   */
  template <class ExecutionSpace>
  class KokkosPendingParticleQueue {
  public:
    using memory_space = typename ExecutionSpace::memory_space;
    using View = Kokkos::View<gpu::em::EmParticleState*, memory_space>;

    explicit KokkosPendingParticleQueue(std::string label = {})
        : label_(std::move(label)) {}

    std::size_t size() const noexcept { return size_; }
    std::size_t head() const noexcept { return head_; }
    std::size_t capacity() const noexcept { return capacity_; }
    bool empty() const noexcept { return size_ == 0; }
    std::size_t deviceBytes() const noexcept {
      return capacity_ * sizeof(gpu::em::EmParticleState);
    }
    KokkosMemoryProjection projectedTailCapacity(
        std::size_t const appended,
        std::size_t const maximum_capacity) const {
      auto const current = deviceBytes();
      if (appended == 0) return {current, current};
      if (appended > std::numeric_limits<std::size_t>::max() - size_)
        throw std::overflow_error(
            "Kokkos pending-particle queue size overflow");
      auto const required = size_ + appended;
      if (required > maximum_capacity)
        throw std::length_error(
            "Kokkos pending-particle queue exceeds configured capacity");
      if (head_ <= capacity_ && required <= capacity_ - head_)
        return {current, current};
      auto const grown = checkedGeometricCapacity(
          capacity_, required, maximum_capacity);
      auto const replacement = checkedMemoryMultiply(
          grown, sizeof(gpu::em::EmParticleState));
      KokkosMemoryProjectionBuilder projection(current);
      projection.replace(current, replacement);
      return projection.result();
    }
    View const& view() const noexcept { return particles_; }

    void clear() noexcept {
      head_ = 0;
      size_ = 0;
    }

    void ensureTailCapacity(std::size_t const appended,
                            std::size_t const maximum_capacity,
                            ExecutionSpace const& execution) {
      if (appended == 0) return;
      if (appended > std::numeric_limits<std::size_t>::max() - size_)
        throw std::overflow_error("Kokkos pending-particle queue size overflow");
      auto const required = size_ + appended;
      if (required > maximum_capacity)
        throw std::length_error(
            "Kokkos pending-particle queue exceeds configured capacity");
      if (head_ <= capacity_ && required <= capacity_ - head_) return;

      auto const grown = checkedGeometricCapacity(
          capacity_, required, maximum_capacity);
      View replacement(
          Kokkos::view_alloc(
              execution, Kokkos::WithoutInitializing,
              label_.empty() ? "c8_kokkos_pending_particles" : label_),
          grown);
      if (size_ != 0) {
        Kokkos::deep_copy(
            execution,
            Kokkos::subview(
                replacement, std::make_pair<std::size_t>(0, size_)),
            Kokkos::subview(
                particles_, std::make_pair(head_, head_ + size_)));
        execution.fence("compact Kokkos cross-species queue");
      }
      particles_ = std::move(replacement);
      capacity_ = grown;
      head_ = 0;
    }

    bool tryAppend(View const& source, std::size_t const count,
                   std::size_t const maximum_size,
                   ExecutionSpace const& execution,
                   KokkosResidentMemoryBudgetGate const memory_gate = {},
                   char const* const stage =
                       "Kokkos pending-particle queue growth") {
      if (count == 0) return true;
      if (count > source.extent(0))
        throw std::length_error(
            "Kokkos cross-species append exceeds source extent");
      if (size_ > maximum_size || count > maximum_size - size_)
        return false;
      auto const projection = projectedTailCapacity(count, maximum_size);
      if (!memory_gate.permits(deviceBytes(), projection, stage))
        return false;
      ensureTailCapacity(count, maximum_size, execution);
      auto const tail = head_ + size_;
      Kokkos::deep_copy(
          execution,
          Kokkos::subview(
              particles_, std::make_pair(tail, tail + count)),
          Kokkos::subview(source, std::make_pair<std::size_t>(0, count)));
      size_ += count;
      return true;
    }

    /** Download a stable logical prefix without consuming it. */
    std::vector<gpu::em::EmParticleState> downloadPrefix(
        std::size_t const count,
        ExecutionSpace const& execution) const {
      if (count > size_)
        throw std::length_error(
            "Kokkos pending-particle prefix exceeds logical queue size");
      if (count == 0) return {};
      auto const active = Kokkos::subview(
          particles_, std::make_pair(head_, head_ + count));
      auto host = Kokkos::create_mirror_view(active);
      Kokkos::deep_copy(execution, host, active);
      execution.fence("download Kokkos pending-particle prefix");
      std::vector<gpu::em::EmParticleState> output(count);
      for (std::size_t index = 0; index < count; ++index)
        output[index] = host(index);
      return output;
    }

    void consume(std::size_t const count) {
      if (count > size_)
        throw std::length_error(
            "Kokkos cross-species consume exceeds pending count");
      head_ += count;
      size_ -= count;
      if (size_ == 0) head_ = 0;
    }

  private:
    std::string label_{};
    View particles_{};
    std::size_t capacity_{};
    std::size_t head_{};
    std::size_t size_{};
  };

  /** Grow-only, deterministic double-buffered SoA wavefront queue. */
  template <class ExecutionSpace>
  class KokkosWavefrontQueue {
  public:
    using execution_space = ExecutionSpace;
    using memory_space = typename ExecutionSpace::memory_space;
    using FlagView = Kokkos::View<std::uint32_t*, memory_space>;
    using OffsetView = Kokkos::View<std::uint64_t*, memory_space>;
    using ParticleView =
        Kokkos::View<gpu::em::EmParticleState*, memory_space>;
    using host_staging_space = Kokkos::SharedHostPinnedSpace;
    using HostStagingView =
        Kokkos::View<gpu::em::EmParticleState*, host_staging_space>;

    static_assert(
        std::is_trivially_copyable_v<gpu::em::EmParticleState>,
        "Kokkos wavefront AoS staging requires a trivially-copyable particle");

    KokkosWavefrontQueue() = default;

    explicit KokkosWavefrontQueue(std::size_t const capacity) {
      ensureCapacity(capacity, execution_);
    }

    KokkosWavefrontQueue(std::size_t const capacity,
                         ExecutionSpace const& execution)
        : execution_(execution) {
      ensureCapacity(capacity, execution_);
    }

    /**
     * Grow the device allocation and reset the current logical queue.
     *
     * Resident cascade workspaces call this at a synchronized host wavefront
     * boundary and then upload a new queue. Growing discards any current
     * contents and sets size() to zero. The allocation grows geometrically and
     * may therefore exceed the requested logical checkpoint; physics-side
     * overflow tests continue to use that unchanged request. Capacity never
     * shrinks, so calls at or below the retained capacity leave both the
     * allocation and logical queue unchanged.
     */
    void ensureCapacity(std::size_t const requested) {
      ensureCapacity(requested, execution_);
    }

    void ensureCapacity(std::size_t const requested,
                        ExecutionSpace const& execution) {
      if (requested == 0) {
        throw std::invalid_argument("Kokkos wavefront capacity must be positive");
      }
      if (!execution_synchronized_)
        throw std::logic_error(
            "Kokkos wavefront queue cannot be rebound while asynchronous work is in flight");
      execution_ = execution;
      if (requested <= capacity_) return;
      auto const grown = checkedGeometricCapacity(capacity_, requested);

      // Construct a complete replacement first.  A failed allocation leaves
      // the old queue and its logical contents untouched.
      auto replacement_current = ParticleSoA<memory_space>(
          "c8_wavefront_current", grown, execution);
      auto replacement_next = ParticleSoA<memory_space>(
          "c8_wavefront_next", grown, execution);
      auto replacement_offsets = OffsetView(
          Kokkos::view_alloc(
              execution, Kokkos::WithoutInitializing,
              "c8_wavefront_offsets"),
          grown);
      auto replacement_aos_staging = ParticleView(
          Kokkos::view_alloc(
              execution, Kokkos::WithoutInitializing,
              "c8_wavefront_aos_staging"),
          grown);
      auto replacement_host_staging = HostStagingView(
          Kokkos::view_alloc(
              Kokkos::WithoutInitializing,
              "c8_wavefront_host_pinned_aos_staging"),
          grown);

      current_ = std::move(replacement_current);
      next_ = std::move(replacement_next);
      offsets_ = std::move(replacement_offsets);
      aos_staging_ = std::move(replacement_aos_staging);
      host_staging_ = std::move(replacement_host_staging);
      capacity_ = grown;
      size_ = 0;
    }

    std::size_t size() const noexcept { return size_; }
    std::size_t capacity() const noexcept { return capacity_; }
    bool empty() const noexcept { return size_ == 0; }
    std::size_t deviceBytes() const noexcept {
      auto const view_bytes = [](auto const& view) {
        return view.span() *
               sizeof(typename std::decay_t<decltype(view)>::value_type);
      };
      return current_.deviceBytes() + next_.deviceBytes() +
             view_bytes(offsets_) + view_bytes(aos_staging_);
    }
    KokkosMemoryProjection projectedCapacity(
        std::size_t const requested) const {
      auto const current = deviceBytes();
      if (requested == 0)
        throw std::invalid_argument(
            "Kokkos wavefront capacity must be positive");
      if (requested <= capacity_) return {current, current};
      auto const grown = checkedGeometricCapacity(capacity_, requested);
      auto replacement = checkedMemoryMultiply(
          2, ParticleSoA<memory_space>::deviceBytesForCapacity(grown));
      replacement = checkedMemoryAdd(
          replacement,
          checkedMemoryMultiply(grown, sizeof(std::uint64_t)));
      replacement = checkedMemoryAdd(
          replacement,
          checkedMemoryMultiply(
              grown, sizeof(gpu::em::EmParticleState)));
      KokkosMemoryProjectionBuilder projection(current);
      projection.replace(current, replacement);
      return projection.result();
    }
    std::size_t hostPinnedBytes() const noexcept {
      return host_staging_.span() *
             sizeof(typename HostStagingView::value_type);
    }
    ParticleSoA<memory_space> const& current() const noexcept { return current_; }
    ParticleSoA<memory_space>& next() noexcept { return next_; }

    void commitNext(std::size_t const next_size) {
      validateNextSize(next_size);
      execution_.fence("commit Kokkos EM wavefront");
      markExecutionSynchronized();
      commitNextAlreadySynchronized(next_size);
    }

    /**
     * Swap fronts after the caller has already established execution-space
     * ordering.  Resident cascades fence immediately after materialization
     * and submit all later profile/queue work to the same execution-space
     * instance, so another unconditional device-wide fence at the host-only
     * pointer swap only serializes the next wavefront.
     */
    void commitNextAlreadySynchronized(std::size_t const next_size) {
      validateNextSize(next_size);
      std::swap(current_, next_);
      size_ = next_size;
    }

    void clear() {
      if (!execution_synchronized_)
        throw std::logic_error(
            "Kokkos wavefront queue cannot be cleared while asynchronous work is in flight");
      size_ = 0;
    }

  private:
    void validateNextSize(std::size_t const next_size) const {
      if (next_size > capacity_) {
        throw std::length_error(
            "Kokkos next wavefront exceeds queue capacity");
      }
    }

  public:

    /** Mark all queue work on the bound execution instance complete. */
    void markExecutionSynchronized() noexcept {
      execution_synchronized_ = true;
    }

    void upload(std::vector<gpu::em::EmParticleState> const& particles) {
      upload(particles, ParticleView{}, 0, 0, execution_);
    }

    void upload(std::vector<gpu::em::EmParticleState> const& particles,
                ExecutionSpace const& execution) {
      upload(particles, ParticleView{}, 0, 0, execution);
    }

    void upload(std::vector<gpu::em::EmParticleState> const& particles,
                ParticleView const& prefix, std::size_t const prefix_head,
                std::size_t const prefix_count,
                ExecutionSpace const& execution) {
      if (!execution_synchronized_)
        throw std::logic_error(
            "Kokkos wavefront upload requires preceding queue work to be synchronized");
      execution_ = execution;
      if (prefix_count > prefix.extent(0) ||
          prefix_head > prefix.extent(0) - prefix_count)
        throw std::length_error(
            "Kokkos wavefront prefix exceeds pending queue extent");
      if (prefix_count > std::numeric_limits<std::size_t>::max() -
                             particles.size())
        throw std::overflow_error("Kokkos wavefront upload size overflow");
      auto const total = prefix_count + particles.size();
      if (total > capacity_) {
        throw std::length_error("Kokkos wavefront upload exceeds queue capacity");
      }
      if (total == 0) {
        size_ = 0;
        return;
      }

      // From this point onward the queue owns work submitted to execution_.
      // Set the state before the first asynchronous operation so that a
      // launch/copy exception cannot leave the queue falsely reusable.  This
      // also covers prefix-only uploads: they do not consume pinned host
      // memory, but their unpack kernel still writes current_ asynchronously.
      execution_synchronized_ = false;

      // Copy pageable application storage into a persistent pinned AoS view,
      // then submit exactly one host-to-device transfer.  SharedHostPinnedSpace
      // is CUDA/HIP/SYCL pinned host memory and aliases HostSpace for a
      // host-only Kokkos build.  One device kernel then writes both the
      // pending-device prefix and staged host suffix into the canonical SoA
      // queue in stable order.
      if (!particles.empty()) {
        std::memcpy(host_staging_.data(), particles.data(),
                    particles.size() *
                        sizeof(gpu::em::EmParticleState));
        Kokkos::deep_copy(
            execution,
            Kokkos::subview(
                aos_staging_,
                std::make_pair<std::size_t>(0, particles.size())),
            Kokkos::subview(
                host_staging_,
                std::make_pair<std::size_t>(0, particles.size())));
      }
      auto const current = current_.rawDeviceView();
      auto const staging = aos_staging_;
      Kokkos::parallel_for(
          "c8_kokkos_unpack_wavefront_aos",
          Kokkos::RangePolicy<ExecutionSpace>(execution, 0, total),
          KOKKOS_LAMBDA(std::size_t const index) {
            current.store(
                index,
                index < prefix_count
                    ? prefix(prefix_head + index)
                    : staging(index - prefix_count));
          });
      // The pinned staging allocation is persistent and all consumers use the
      // same execution-space stream, so stream ordering is sufficient here.
      // A host fence would serialize every small resident-cascade hand-off.
      size_ = total;
    }

    std::vector<gpu::em::EmParticleState> download() const {
      return download(execution_);
    }

    std::vector<gpu::em::EmParticleState> download(
        ExecutionSpace const& execution) const {
      if (size_ == 0) return {};
      // upload() records the execution-space instance that owns all pending
      // queue work.  Always enqueue the pack/copy behind that work instead of
      // accepting a caller-provided stream that could race it.  The explicit
      // argument is retained for source compatibility with existing callers.
      (void)execution;
      auto const& ordered_execution = execution_;
      execution_synchronized_ = false;
      auto const current = current_.rawDeviceView();
      auto const staging = aos_staging_;
      Kokkos::parallel_for(
          "c8_kokkos_pack_wavefront_aos",
          Kokkos::RangePolicy<ExecutionSpace>(ordered_execution, 0, size_),
          KOKKOS_LAMBDA(std::size_t const index) {
            staging(index) = current.load(index);
          });

      Kokkos::deep_copy(
          ordered_execution,
          Kokkos::subview(
              host_staging_, std::make_pair<std::size_t>(0, size_)),
          Kokkos::subview(
              aos_staging_, std::make_pair<std::size_t>(0, size_)));
      ordered_execution.fence("download Kokkos EM wavefront");
      execution_synchronized_ = true;

      std::vector<gpu::em::EmParticleState> output(size_);
      std::memcpy(output.data(), host_staging_.data(),
                  output.size() * sizeof(gpu::em::EmParticleState));
      return output;
    }

    std::size_t compact(Kokkos::View<std::uint32_t const*, memory_space> flags) {
      if (flags.extent(0) < size_) {
        throw std::invalid_argument("Kokkos compaction flags are shorter than the queue");
      }
      execution_synchronized_ = false;
      std::uint64_t selected = 0;
      Kokkos::parallel_scan(
          "c8_kokkos_wavefront_scan",
          Kokkos::RangePolicy<ExecutionSpace>(execution_, 0, size_),
          QueueScanFunctor<ExecutionSpace>{flags, offsets_}, selected);
      Kokkos::parallel_for(
          "c8_kokkos_wavefront_scatter",
          Kokkos::RangePolicy<ExecutionSpace>(execution_, 0, size_),
          QueueScatterFunctor<ExecutionSpace>{current_.rawDeviceView(),
                                               next_.rawDeviceView(),
                                               flags.data(), offsets_.data()});
      static_assert(sizeof(QueueScatterFunctor<ExecutionSpace>) < 512);
      execution_.fence("compact Kokkos EM wavefront");
      execution_synchronized_ = true;
      std::swap(current_, next_);
      size_ = static_cast<std::size_t>(selected);
      return size_;
    }

    std::uint64_t secondaryOffsets(
        Kokkos::View<std::uint32_t const*, memory_space> counts) {
      if (counts.extent(0) > capacity_) {
        throw std::length_error("Kokkos secondary counts exceed queue capacity");
      }
      execution_synchronized_ = false;
      std::uint64_t total = 0;
      Kokkos::parallel_scan(
          "c8_kokkos_secondary_scan",
          Kokkos::RangePolicy<ExecutionSpace>(execution_, 0, counts.extent(0)),
          CountScanFunctor<ExecutionSpace>{counts, offsets_}, total);
      execution_.fence("allocate deterministic secondary offsets");
      execution_synchronized_ = true;
      return total;
    }

    OffsetView const& offsets() const noexcept { return offsets_; }

  private:
    ExecutionSpace execution_{};
    std::size_t capacity_{};
    std::size_t size_{};
    ParticleSoA<memory_space> current_{};
    ParticleSoA<memory_space> next_{};
    OffsetView offsets_{};
    ParticleView aos_staging_{};
    HostStagingView host_staging_{};
    // False means work touching current_, staging, or the pinned host buffer
    // is still ordered on execution_.  Rebinding/reallocation is forbidden
    // until the owning execution instance has been fenced.
    mutable bool execution_synchronized_{true};
  };

  /**
   * Exception-path owner for work submitted through a wavefront queue.
   *
   * Resident cascades normally fence and release this guard once their final
   * diagnostics have reached the host.  If a host-side validation throws at
   * an earlier wavefront boundary, the guard drains the queue's owning
   * execution instance before persistent Views can be reused or local Views
   * can be destroyed.  Device/runtime failures remain fatal to the shower;
   * the destructor deliberately never masks the original exception.
   */
  template <class ExecutionSpace>
  class KokkosWavefrontExecutionGuard {
  public:
    KokkosWavefrontExecutionGuard(
        KokkosWavefrontQueue<ExecutionSpace>& queue,
        ExecutionSpace const& execution) noexcept
        : queue_(&queue), execution_(&execution) {}

    KokkosWavefrontExecutionGuard(
        KokkosWavefrontExecutionGuard const&) = delete;
    KokkosWavefrontExecutionGuard& operator=(
        KokkosWavefrontExecutionGuard const&) = delete;

    ~KokkosWavefrontExecutionGuard() noexcept {
      if (!active_) return;
      try {
        execution_->fence("unwind resident Kokkos EM wavefront");
        queue_->markExecutionSynchronized();
      } catch (...) {
        // The caller is already unwinding.  A CUDA/HIP/SYCL runtime failure
        // invalidates the current shower and must not replace that exception.
      }
    }

    void release() noexcept { active_ = false; }

  private:
    KokkosWavefrontQueue<ExecutionSpace>* queue_{};
    ExecutionSpace const* execution_{};
    bool active_{true};
  };

} // namespace corsika::accelerator::em::kokkos_detail

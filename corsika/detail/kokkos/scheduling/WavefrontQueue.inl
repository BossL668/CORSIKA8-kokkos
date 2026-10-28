/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

namespace corsika::kokkos {

  template <class TExecutionSpace>
  KOKKOS_INLINE_FUNCTION void QueueScanFunctor<TExecutionSpace>::operator()(
      std::size_t const index, std::uint64_t& update, bool const final) const {
    if (final) offsets_(index) = update;
    update += flags_(index) != 0U ? 1U : 0U;
  }

  template <class TExecutionSpace>
  KOKKOS_INLINE_FUNCTION void QueueScatterFunctor<TExecutionSpace>::operator()(
      std::size_t const index) const {
    if (flags_[index] != 0U) output_.store(offsets_[index], input_.load(index));
  }

  template <class TExecutionSpace>
  KOKKOS_INLINE_FUNCTION void CountScanFunctor<TExecutionSpace>::operator()(
      std::size_t const index, std::uint64_t& update, bool const final) const {
    if (final) offsets_(index) = update;
    update += counts_(index);
  }
  template <class TExecutionSpace>
  WavefrontQueue<TExecutionSpace>::WavefrontQueue(std::size_t const capacity) {
    ensureCapacity(capacity, execution_);
  }

  template <class TExecutionSpace>
  WavefrontQueue<TExecutionSpace>::WavefrontQueue(std::size_t const capacity,
                                                  TExecutionSpace const& execution)
      : execution_(execution) {
    ensureCapacity(capacity, execution_);
  }

  template <class TExecutionSpace>
  void WavefrontQueue<TExecutionSpace>::ensureCapacity(std::size_t const requested) {
    ensureCapacity(requested, execution_);
  }

  template <class TExecutionSpace>
  void WavefrontQueue<TExecutionSpace>::ensureCapacity(std::size_t const requested,
                                                       TExecutionSpace const& execution) {
    if (requested == 0) {
      throw std::invalid_argument("Kokkos wavefront capacity must be positive");
    }
    if (!executionSynchronized_)
      throw std::logic_error(
          "Kokkos wavefront queue cannot be rebound while asynchronous work is in "
          "flight");
    execution_ = execution;
    if (requested <= capacity_) return;
    auto const grown = checked_geometric_capacity(capacity_, requested);

    // Construct a complete replacement first.  A failed allocation leaves
    // the old queue and its logical contents untouched.
    auto replacement_current =
        ParticleSoA<memory_space_type>("c8_wavefront_current", grown, execution);
    auto replacement_next =
        ParticleSoA<memory_space_type>("c8_wavefront_next", grown, execution);
    auto replacement_offsets =
        offset_view_type(Kokkos::view_alloc(execution, Kokkos::WithoutInitializing,
                                            "c8_wavefront_offsets"),
                         grown);
    auto replacement_aos_staging =
        particle_view_type(Kokkos::view_alloc(execution, Kokkos::WithoutInitializing,
                                              "c8_wavefront_aos_staging"),
                           grown);
    auto replacement_host_staging =
        host_staging_view_type(Kokkos::view_alloc(Kokkos::WithoutInitializing,
                                                  "c8_wavefront_host_pinned_aos_staging"),
                               grown);

    current_ = std::move(replacement_current);
    next_ = std::move(replacement_next);
    offsets_ = std::move(replacement_offsets);
    aosStaging_ = std::move(replacement_aos_staging);
    hostStaging_ = std::move(replacement_host_staging);
    capacity_ = grown;
    size_ = 0;
  }

  template <class TExecutionSpace>
  std::size_t WavefrontQueue<TExecutionSpace>::size() const noexcept {
    return size_;
  }

  template <class TExecutionSpace>
  std::size_t WavefrontQueue<TExecutionSpace>::capacity() const noexcept {
    return capacity_;
  }

  template <class TExecutionSpace>
  bool WavefrontQueue<TExecutionSpace>::empty() const noexcept {
    return size_ == 0;
  }

  template <class TExecutionSpace>
  std::size_t WavefrontQueue<TExecutionSpace>::deviceBytes() const noexcept {
    auto const view_bytes = [](auto const& view) {
      return view.span() * sizeof(typename std::decay_t<decltype(view)>::value_type);
    };
    return current_.deviceBytes() + next_.deviceBytes() + view_bytes(offsets_) +
           view_bytes(aosStaging_);
  }

  template <class TExecutionSpace>
  MemoryProjection WavefrontQueue<TExecutionSpace>::projectedCapacity(
      std::size_t const requested) const {
    auto const current = deviceBytes();
    if (requested == 0)
      throw std::invalid_argument("Kokkos wavefront capacity must be positive");
    if (requested <= capacity_) return {current, current};
    auto const grown = checked_geometric_capacity(capacity_, requested);
    auto replacement = checked_memory_multiply(
        2, ParticleSoA<memory_space_type>::deviceBytesForCapacity(grown));
    replacement = checked_memory_add(
        replacement, checked_memory_multiply(grown, sizeof(std::uint64_t)));
    replacement = checked_memory_add(
        replacement, checked_memory_multiply(grown, sizeof(ParticleState)));
    MemoryProjectionBuilder projection(current);
    projection.replace(current, replacement);
    return projection.result();
  }

  template <class TExecutionSpace>
  std::size_t WavefrontQueue<TExecutionSpace>::hostPinnedBytes() const noexcept {
    return hostStaging_.span() * sizeof(typename host_staging_view_type::value_type);
  }

  template <class TExecutionSpace>
  ParticleSoA<typename TExecutionSpace::memory_space> const&
  WavefrontQueue<TExecutionSpace>::current() const noexcept {
    return current_;
  }

  template <class TExecutionSpace>
  ParticleSoA<typename TExecutionSpace::memory_space>&
  WavefrontQueue<TExecutionSpace>::next() noexcept {
    return next_;
  }

  template <class TExecutionSpace>
  ParticleSoA<typename TExecutionSpace::memory_space> const&
  WavefrontQueue<TExecutionSpace>::next() const noexcept {
    return next_;
  }

  template <class TExecutionSpace>
  void WavefrontQueue<TExecutionSpace>::commitNext(std::size_t const next_size) {
    validateNextSize(next_size);
    execution_.fence("commit Kokkos EM wavefront");
    markExecutionSynchronized();
    commitNextAlreadySynchronized(next_size);
  }

  template <class TExecutionSpace>
  void WavefrontQueue<TExecutionSpace>::commitNextAlreadySynchronized(
      std::size_t const next_size) {
    validateNextSize(next_size);
    std::swap(current_, next_);
    size_ = next_size;
  }

  template <class TExecutionSpace>
  void WavefrontQueue<TExecutionSpace>::clear() {
    if (!executionSynchronized_)
      throw std::logic_error(
          "Kokkos wavefront queue cannot be cleared while asynchronous work is in "
          "flight");
    size_ = 0;
  }

  template <class TExecutionSpace>
  void WavefrontQueue<TExecutionSpace>::validateNextSize(
      std::size_t const next_size) const {
    if (next_size > capacity_) {
      throw std::length_error("Kokkos next wavefront exceeds queue capacity");
    }
  }

  template <class TExecutionSpace>
  void WavefrontQueue<TExecutionSpace>::markExecutionSynchronized() noexcept {
    executionSynchronized_ = true;
  }

  template <class TExecutionSpace>
  void WavefrontQueue<TExecutionSpace>::upload(
      std::vector<ParticleState> const& particles) {
    upload(particles, particle_view_type{}, 0, 0, execution_);
  }

  template <class TExecutionSpace>
  void WavefrontQueue<TExecutionSpace>::upload(
      std::vector<ParticleState> const& particles, TExecutionSpace const& execution) {
    upload(particles, particle_view_type{}, 0, 0, execution);
  }

  template <class TExecutionSpace>
  void WavefrontQueue<TExecutionSpace>::upload(
      std::vector<ParticleState> const& particles, particle_view_type const& prefix,
      std::size_t const prefix_head, std::size_t const prefix_count,
      TExecutionSpace const& execution) {
    if (!executionSynchronized_)
      throw std::logic_error(
          "Kokkos wavefront upload requires preceding queue work to be synchronized");
    execution_ = execution;
    if (prefix_count > prefix.extent(0) || prefix_head > prefix.extent(0) - prefix_count)
      throw std::length_error("Kokkos wavefront prefix exceeds pending queue extent");
    if (prefix_count > std::numeric_limits<std::size_t>::max() - particles.size())
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
    executionSynchronized_ = false;

    // Copy pageable application storage into a persistent pinned AoS view,
    // then submit exactly one host-to-device transfer.  SharedHostPinnedSpace
    // is CUDA/HIP/SYCL pinned host memory and aliases HostSpace for a
    // host-only Kokkos build.  One device kernel then writes both the
    // pending-device prefix and staged host suffix into the canonical SoA
    // queue in stable order.
    if (!particles.empty()) {
      std::memcpy(hostStaging_.data(), particles.data(),
                  particles.size() * sizeof(ParticleState));
      Kokkos::deep_copy(
          execution,
          Kokkos::subview(aosStaging_, std::make_pair<std::size_t>(0, particles.size())),
          Kokkos::subview(hostStaging_,
                          std::make_pair<std::size_t>(0, particles.size())));
    }
    auto const current = current_.rawDeviceView();
    auto const staging = aosStaging_;
    Kokkos::parallel_for(
        "c8_kokkos_unpack_wavefront_aos",
        Kokkos::RangePolicy<TExecutionSpace>(execution, 0, total),
        KOKKOS_LAMBDA(std::size_t const index) {
          current.store(index, index < prefix_count ? prefix(prefix_head + index)
                                                    : staging(index - prefix_count));
        });
    // The pinned staging allocation is persistent and all consumers use the
    // same execution-space stream, so stream ordering is sufficient here.
    // A host fence would serialize every small resident-cascade hand-off.
    size_ = total;
  }

  template <class TExecutionSpace>
  std::vector<ParticleState> WavefrontQueue<TExecutionSpace>::download() const {
    return download(execution_);
  }

  template <class TExecutionSpace>
  std::vector<ParticleState> WavefrontQueue<TExecutionSpace>::download(
      TExecutionSpace const& execution,
      ResidentExecutionWait<TExecutionSpace>* const blocking_wait) const {
    if (size_ == 0) return {};
    // upload() records the execution-space instance that owns all pending
    // queue work.  Always enqueue the pack/copy behind that work instead of
    // accepting a caller-provided stream that could race it.  The explicit
    // argument is retained for source compatibility with existing callers.
    (void)execution;
    auto const& ordered_execution = execution_;
    executionSynchronized_ = false;
    auto const current = current_.rawDeviceView();
    auto const staging = aosStaging_;
    Kokkos::parallel_for(
        "c8_kokkos_pack_wavefront_aos",
        Kokkos::RangePolicy<TExecutionSpace>(ordered_execution, 0, size_),
        KOKKOS_LAMBDA(std::size_t const index) { staging(index) = current.load(index); });

    Kokkos::deep_copy(
        ordered_execution,
        Kokkos::subview(hostStaging_, std::make_pair<std::size_t>(0, size_)),
        Kokkos::subview(aosStaging_, std::make_pair<std::size_t>(0, size_)));
    // Only CPU-primary's explicit blocking mode changes this boundary.
    // In particular, never introduce a progress callback/reentrant host-work
    // point here for GPU-primary or other existing callers. The queue's
    // owning stream, not the compatibility argument, orders pack and copy.
    auto* const waiter =
        blocking_wait && blocking_wait->blockingEnabled() ? blocking_wait : nullptr;
    wait_resident_execution(ordered_execution, "download Kokkos EM wavefront", waiter);
    executionSynchronized_ = true;

    std::vector<ParticleState> output(size_);
    std::memcpy(output.data(), hostStaging_.data(),
                output.size() * sizeof(ParticleState));
    return output;
  }

  template <class TExecutionSpace>
  std::size_t WavefrontQueue<TExecutionSpace>::compact(
      Kokkos::View<std::uint32_t const*, memory_space_type> flags) {
    if (flags.extent(0) < size_) {
      throw std::invalid_argument("Kokkos compaction flags are shorter than the queue");
    }
    executionSynchronized_ = false;
    std::uint64_t selected = 0;
    Kokkos::parallel_scan("c8_kokkos_wavefront_scan",
                          Kokkos::RangePolicy<TExecutionSpace>(execution_, 0, size_),
                          QueueScanFunctor<TExecutionSpace>{flags, offsets_}, selected);
    Kokkos::parallel_for("c8_kokkos_wavefront_scatter",
                         Kokkos::RangePolicy<TExecutionSpace>(execution_, 0, size_),
                         QueueScatterFunctor<TExecutionSpace>{
                             current_.rawDeviceView(), next_.rawDeviceView(),
                             flags.data(), offsets_.data()});
    static_assert(sizeof(QueueScatterFunctor<TExecutionSpace>) < 512);
    execution_.fence("compact Kokkos EM wavefront");
    executionSynchronized_ = true;
    std::swap(current_, next_);
    size_ = static_cast<std::size_t>(selected);
    return size_;
  }

  template <class TExecutionSpace>
  std::uint64_t WavefrontQueue<TExecutionSpace>::secondaryOffsets(
      Kokkos::View<std::uint32_t const*, memory_space_type> counts) {
    if (counts.extent(0) > capacity_) {
      throw std::length_error("Kokkos secondary counts exceed queue capacity");
    }
    executionSynchronized_ = false;
    std::uint64_t total = 0;
    Kokkos::parallel_scan(
        "c8_kokkos_secondary_scan",
        Kokkos::RangePolicy<TExecutionSpace>(execution_, 0, counts.extent(0)),
        CountScanFunctor<TExecutionSpace>{counts, offsets_}, total);
    execution_.fence("allocate deterministic secondary offsets");
    executionSynchronized_ = true;
    return total;
  }

  template <class TExecutionSpace>
  typename WavefrontQueue<TExecutionSpace>::offset_view_type const&
  WavefrontQueue<TExecutionSpace>::offsets() const noexcept {
    return offsets_;
  }

} // namespace corsika::kokkos

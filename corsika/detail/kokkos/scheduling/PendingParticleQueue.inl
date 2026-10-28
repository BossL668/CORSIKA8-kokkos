/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

namespace corsika::kokkos {

  template <class TExecutionSpace>
  PendingParticleQueue<TExecutionSpace>::PendingParticleQueue(std::string label)
      : label_(std::move(label)) {}

  template <class TExecutionSpace>
  std::size_t PendingParticleQueue<TExecutionSpace>::size() const noexcept {
    return size_;
  }

  template <class TExecutionSpace>
  std::size_t PendingParticleQueue<TExecutionSpace>::head() const noexcept {
    return head_;
  }

  template <class TExecutionSpace>
  std::size_t PendingParticleQueue<TExecutionSpace>::capacity() const noexcept {
    return capacity_;
  }

  template <class TExecutionSpace>
  bool PendingParticleQueue<TExecutionSpace>::empty() const noexcept {
    return size_ == 0;
  }

  template <class TExecutionSpace>
  std::size_t PendingParticleQueue<TExecutionSpace>::deviceBytes() const noexcept {
    return capacity_ * sizeof(ParticleState);
  }

  template <class TExecutionSpace>
  MemoryProjection PendingParticleQueue<TExecutionSpace>::projectedTailCapacity(
      std::size_t const appended, std::size_t const maximum_capacity) const {
    auto const current = deviceBytes();
    if (appended == 0) return {current, current};
    if (appended > std::numeric_limits<std::size_t>::max() - size_)
      throw std::overflow_error("Kokkos pending-particle queue size overflow");
    auto const required = size_ + appended;
    if (required > maximum_capacity)
      throw std::length_error(
          "Kokkos pending-particle queue exceeds configured capacity");
    if (head_ <= capacity_ && required <= capacity_ - head_) return {current, current};
    auto const grown = checked_geometric_capacity(capacity_, required, maximum_capacity);
    auto const replacement = checked_memory_multiply(grown, sizeof(ParticleState));
    MemoryProjectionBuilder projection(current);
    projection.replace(current, replacement);
    return projection.result();
  }

  template <class TExecutionSpace>
  typename PendingParticleQueue<TExecutionSpace>::view_type const&
  PendingParticleQueue<TExecutionSpace>::view() const noexcept {
    return particles_;
  }

  template <class TExecutionSpace>
  void PendingParticleQueue<TExecutionSpace>::clear() noexcept {
    head_ = 0;
    size_ = 0;
  }

  template <class TExecutionSpace>
  void PendingParticleQueue<TExecutionSpace>::ensureTailCapacity(
      std::size_t const appended, std::size_t const maximum_capacity,
      TExecutionSpace const& execution) {
    if (appended == 0) return;
    if (appended > std::numeric_limits<std::size_t>::max() - size_)
      throw std::overflow_error("Kokkos pending-particle queue size overflow");
    auto const required = size_ + appended;
    if (required > maximum_capacity)
      throw std::length_error(
          "Kokkos pending-particle queue exceeds configured capacity");
    if (head_ <= capacity_ && required <= capacity_ - head_) return;

    auto const grown = checked_geometric_capacity(capacity_, required, maximum_capacity);
    view_type replacement(
        Kokkos::view_alloc(execution, Kokkos::WithoutInitializing,
                           label_.empty() ? "c8_kokkos_pending_particles" : label_),
        grown);
    if (size_ != 0) {
      Kokkos::deep_copy(
          execution, Kokkos::subview(replacement, std::make_pair<std::size_t>(0, size_)),
          Kokkos::subview(particles_, std::make_pair(head_, head_ + size_)));
      execution.fence("compact Kokkos cross-species queue");
    }
    particles_ = std::move(replacement);
    capacity_ = grown;
    head_ = 0;
  }

  template <class TExecutionSpace>
  bool PendingParticleQueue<TExecutionSpace>::tryAppend(
      view_type const& source, std::size_t const count, std::size_t const maximum_size,
      TExecutionSpace const& execution, ResidentMemoryBudgetGate const memory_gate,
      char const* const stage) {
    if (count == 0) return true;
    if (count > source.extent(0))
      throw std::length_error("Kokkos cross-species append exceeds source extent");
    if (size_ > maximum_size || count > maximum_size - size_) return false;
    auto const projection = projectedTailCapacity(count, maximum_size);
    if (!memory_gate.permits(deviceBytes(), projection, stage)) return false;
    ensureTailCapacity(count, maximum_size, execution);
    auto const tail = head_ + size_;
    Kokkos::deep_copy(execution,
                      Kokkos::subview(particles_, std::make_pair(tail, tail + count)),
                      Kokkos::subview(source, std::make_pair<std::size_t>(0, count)));
    size_ += count;
    return true;
  }

  template <class TExecutionSpace>
  std::vector<ParticleState> PendingParticleQueue<TExecutionSpace>::downloadPrefix(
      std::size_t const count, TExecutionSpace const& execution) const {
    if (count > size_)
      throw std::length_error(
          "Kokkos pending-particle prefix exceeds logical queue size");
    if (count == 0) return {};
    auto const active = Kokkos::subview(particles_, std::make_pair(head_, head_ + count));
    auto host = Kokkos::create_mirror_view(active);
    Kokkos::deep_copy(execution, host, active);
    execution.fence("download Kokkos pending-particle prefix");
    std::vector<ParticleState> output(count);
    for (std::size_t index = 0; index < count; ++index) output[index] = host(index);
    return output;
  }

  template <class TExecutionSpace>
  void PendingParticleQueue<TExecutionSpace>::consume(std::size_t const count) {
    if (count > size_)
      throw std::length_error("Kokkos cross-species consume exceeds pending count");
    head_ += count;
    size_ -= count;
    if (size_ == 0) head_ = 0;
  }

} // namespace corsika::kokkos

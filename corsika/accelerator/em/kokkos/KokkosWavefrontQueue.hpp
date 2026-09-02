/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <Kokkos_Core.hpp>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <corsika/gpu/em/Types.hpp>

namespace corsika::accelerator::em::kokkos_detail {

  template <class MemorySpace>
  struct ParticleSoA {
    Kokkos::View<std::int32_t*, MemorySpace> pid;
    Kokkos::View<std::int32_t*, MemorySpace> medium_id;
    Kokkos::View<std::uint32_t*, MemorySpace> generation;
    Kokkos::View<double*, MemorySpace> energy_GeV;
    Kokkos::View<double*[3], MemorySpace> position_m;
    Kokkos::View<double*[3], MemorySpace> direction;
    Kokkos::View<double*, MemorySpace> time_s;
    Kokkos::View<double*, MemorySpace> weight;
    Kokkos::View<std::uint64_t*, MemorySpace> history_id;
    Kokkos::View<std::uint64_t*, MemorySpace> parent_history_id;
    Kokkos::View<std::uint64_t*, MemorySpace> step_id;

    ParticleSoA() = default;

    ParticleSoA(std::string const& label, std::size_t const capacity)
        : pid(label + "_pid", capacity)
        , medium_id(label + "_medium", capacity)
        , generation(label + "_generation", capacity)
        , energy_GeV(label + "_energy", capacity)
        , position_m(label + "_position", capacity)
        , direction(label + "_direction", capacity)
        , time_s(label + "_time", capacity)
        , weight(label + "_weight", capacity)
        , history_id(label + "_history", capacity)
        , parent_history_id(label + "_parent_history", capacity)
        , step_id(label + "_step", capacity) {}

    KOKKOS_INLINE_FUNCTION gpu::em::EmParticleState load(
        std::size_t const index) const {
      gpu::em::EmParticleState state{};
      state.pid = pid(index);
      state.medium_id = medium_id(index);
      state.generation = generation(index);
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
    using memory_space = typename ExecutionSpace::memory_space;
    ParticleSoA<memory_space> input;
    ParticleSoA<memory_space> output;
    Kokkos::View<std::uint32_t const*, memory_space> flags;
    Kokkos::View<std::uint64_t const*, memory_space> offsets;

    KOKKOS_INLINE_FUNCTION void operator()(std::size_t const index) const {
      if (flags(index) != 0U) output.store(offsets(index), input.load(index));
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

  /** Fixed-capacity, deterministic double-buffered SoA wavefront queue. */
  template <class ExecutionSpace>
  class KokkosWavefrontQueue {
  public:
    using execution_space = ExecutionSpace;
    using memory_space = typename ExecutionSpace::memory_space;
    using FlagView = Kokkos::View<std::uint32_t*, memory_space>;
    using OffsetView = Kokkos::View<std::uint64_t*, memory_space>;

    explicit KokkosWavefrontQueue(std::size_t const capacity)
        : capacity_(capacity)
        , current_("c8_wavefront_current", capacity)
        , next_("c8_wavefront_next", capacity)
        , offsets_("c8_wavefront_offsets", capacity) {
      if (capacity == 0) {
        throw std::invalid_argument("Kokkos wavefront capacity must be positive");
      }
    }

    std::size_t size() const noexcept { return size_; }
    std::size_t capacity() const noexcept { return capacity_; }
    bool empty() const noexcept { return size_ == 0; }
    ParticleSoA<memory_space> const& current() const noexcept { return current_; }
    ParticleSoA<memory_space>& next() noexcept { return next_; }

    void commitNext(std::size_t const next_size) {
      if (next_size > capacity_) {
        throw std::length_error(
            "Kokkos next wavefront exceeds queue capacity");
      }
      execution_.fence("commit Kokkos EM wavefront");
      std::swap(current_, next_);
      size_ = next_size;
    }

    void upload(std::vector<gpu::em::EmParticleState> const& particles) {
      if (particles.size() > capacity_) {
        throw std::length_error("Kokkos wavefront upload exceeds queue capacity");
      }
      auto pid = Kokkos::create_mirror_view(current_.pid);
      auto medium = Kokkos::create_mirror_view(current_.medium_id);
      auto generation = Kokkos::create_mirror_view(current_.generation);
      auto energy = Kokkos::create_mirror_view(current_.energy_GeV);
      auto position = Kokkos::create_mirror_view(current_.position_m);
      auto direction = Kokkos::create_mirror_view(current_.direction);
      auto time = Kokkos::create_mirror_view(current_.time_s);
      auto weight = Kokkos::create_mirror_view(current_.weight);
      auto history = Kokkos::create_mirror_view(current_.history_id);
      auto parent = Kokkos::create_mirror_view(current_.parent_history_id);
      auto step = Kokkos::create_mirror_view(current_.step_id);
      for (std::size_t i = 0; i < particles.size(); ++i) {
        auto const& state = particles[i];
        pid(i) = state.pid;
        medium(i) = state.medium_id;
        generation(i) = state.generation;
        energy(i) = state.energy_GeV;
        time(i) = state.time_s;
        weight(i) = state.weight;
        history(i) = state.history_id;
        parent(i) = state.parent_history_id;
        step(i) = state.step_id;
        for (int axis = 0; axis < 3; ++axis) {
          position(i, axis) = state.position_m[axis];
          direction(i, axis) = state.direction[axis];
        }
      }
      Kokkos::deep_copy(execution_, current_.pid, pid);
      Kokkos::deep_copy(execution_, current_.medium_id, medium);
      Kokkos::deep_copy(execution_, current_.generation, generation);
      Kokkos::deep_copy(execution_, current_.energy_GeV, energy);
      Kokkos::deep_copy(execution_, current_.position_m, position);
      Kokkos::deep_copy(execution_, current_.direction, direction);
      Kokkos::deep_copy(execution_, current_.time_s, time);
      Kokkos::deep_copy(execution_, current_.weight, weight);
      Kokkos::deep_copy(execution_, current_.history_id, history);
      Kokkos::deep_copy(execution_, current_.parent_history_id, parent);
      Kokkos::deep_copy(execution_, current_.step_id, step);
      execution_.fence("upload Kokkos EM wavefront");
      size_ = particles.size();
    }

    std::vector<gpu::em::EmParticleState> download() const {
      auto pid = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), current_.pid);
      auto medium = Kokkos::create_mirror_view_and_copy(
          Kokkos::HostSpace(), current_.medium_id);
      auto generation = Kokkos::create_mirror_view_and_copy(
          Kokkos::HostSpace(), current_.generation);
      auto energy = Kokkos::create_mirror_view_and_copy(
          Kokkos::HostSpace(), current_.energy_GeV);
      auto position = Kokkos::create_mirror_view_and_copy(
          Kokkos::HostSpace(), current_.position_m);
      auto direction = Kokkos::create_mirror_view_and_copy(
          Kokkos::HostSpace(), current_.direction);
      auto time = Kokkos::create_mirror_view_and_copy(
          Kokkos::HostSpace(), current_.time_s);
      auto weight = Kokkos::create_mirror_view_and_copy(
          Kokkos::HostSpace(), current_.weight);
      auto history = Kokkos::create_mirror_view_and_copy(
          Kokkos::HostSpace(), current_.history_id);
      auto parent = Kokkos::create_mirror_view_and_copy(
          Kokkos::HostSpace(), current_.parent_history_id);
      auto step = Kokkos::create_mirror_view_and_copy(
          Kokkos::HostSpace(), current_.step_id);
      std::vector<gpu::em::EmParticleState> output(size_);
      for (std::size_t i = 0; i < size_; ++i) {
        auto& state = output[i];
        state.pid = pid(i);
        state.medium_id = medium(i);
        state.generation = generation(i);
        state.energy_GeV = energy(i);
        state.time_s = time(i);
        state.weight = weight(i);
        state.history_id = history(i);
        state.parent_history_id = parent(i);
        state.step_id = step(i);
        for (int axis = 0; axis < 3; ++axis) {
          state.position_m[axis] = position(i, axis);
          state.direction[axis] = direction(i, axis);
        }
      }
      return output;
    }

    std::size_t compact(Kokkos::View<std::uint32_t const*, memory_space> flags) {
      if (flags.extent(0) < size_) {
        throw std::invalid_argument("Kokkos compaction flags are shorter than the queue");
      }
      std::uint64_t selected = 0;
      Kokkos::parallel_scan(
          "c8_kokkos_wavefront_scan",
          Kokkos::RangePolicy<ExecutionSpace>(execution_, 0, size_),
          QueueScanFunctor<ExecutionSpace>{flags, offsets_}, selected);
      Kokkos::parallel_for(
          "c8_kokkos_wavefront_scatter",
          Kokkos::RangePolicy<ExecutionSpace>(execution_, 0, size_),
          QueueScatterFunctor<ExecutionSpace>{
              current_, next_, flags, offsets_});
      execution_.fence("compact Kokkos EM wavefront");
      std::swap(current_, next_);
      size_ = static_cast<std::size_t>(selected);
      return size_;
    }

    std::uint64_t secondaryOffsets(
        Kokkos::View<std::uint32_t const*, memory_space> counts) {
      if (counts.extent(0) > capacity_) {
        throw std::length_error("Kokkos secondary counts exceed queue capacity");
      }
      std::uint64_t total = 0;
      Kokkos::parallel_scan(
          "c8_kokkos_secondary_scan",
          Kokkos::RangePolicy<ExecutionSpace>(execution_, 0, counts.extent(0)),
          CountScanFunctor<ExecutionSpace>{counts, offsets_}, total);
      execution_.fence("allocate deterministic secondary offsets");
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
  };

} // namespace corsika::accelerator::em::kokkos_detail

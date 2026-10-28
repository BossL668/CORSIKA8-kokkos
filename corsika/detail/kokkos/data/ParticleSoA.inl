/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

namespace corsika::kokkos {

  KOKKOS_INLINE_FUNCTION ParticleState
  ParticleSoARawView::load(std::size_t const index) const {
    ParticleState state{};
    state.pid_ = pid_[index];
    state.mediumId_ = mediumId_[index];
    state.environmentNode_ = environmentNode_[index];
    state.generation_ = generation_[index];
    state.reserved_ = reserved_[index];
    state.energyGeV_ = energyGeV_[index];
    state.timeS_ = timeS_[index];
    state.weight_ = weight_[index];
    state.historyId_ = historyId_[index];
    state.parentHistoryId_ = parentHistoryId_[index];
    state.stepId_ = stepId_[index];
    for (int axis = 0; axis < 3; ++axis) {
      state.positionM_[axis] = positionM_(index, axis);
      state.direction_[axis] = direction_(index, axis);
    }
    return state;
  }

  KOKKOS_INLINE_FUNCTION void ParticleSoARawView::store(
      std::size_t const index, ParticleState const& state) const {
    pid_[index] = state.pid_;
    mediumId_[index] = state.mediumId_;
    environmentNode_[index] = state.environmentNode_;
    generation_[index] = state.generation_;
    reserved_[index] = state.reserved_;
    energyGeV_[index] = state.energyGeV_;
    timeS_[index] = state.timeS_;
    weight_[index] = state.weight_;
    historyId_[index] = state.historyId_;
    parentHistoryId_[index] = state.parentHistoryId_;
    stepId_[index] = state.stepId_;
    for (int axis = 0; axis < 3; ++axis) {
      positionM_(index, axis) = state.positionM_[axis];
      direction_(index, axis) = state.direction_[axis];
    }
  }
  template <class TMemorySpace>
  std::size_t ParticleSoA<TMemorySpace>::deviceBytesForCapacity(
      std::size_t const capacity) {
    constexpr auto bytes_per_particle = 3 * sizeof(std::int32_t) +
                                        2 * sizeof(std::uint32_t) + 9 * sizeof(double) +
                                        3 * sizeof(std::uint64_t);
    return checked_memory_multiply(capacity, bytes_per_particle);
  }

  template <class TMemorySpace>
  ParticleSoA<TMemorySpace>::ParticleSoA(std::string const& label,
                                         std::size_t const capacity)
      : ParticleSoA(label, capacity, execution_space_type{}) {}

  template <class TMemorySpace>
  ParticleSoA<TMemorySpace>::ParticleSoA(std::string const& label,
                                         std::size_t const capacity,
                                         execution_space_type const& execution)
      : pid_(Kokkos::view_alloc(execution, Kokkos::WithoutInitializing, label + "_pid"),
             capacity)
      , mediumId_(
            Kokkos::view_alloc(execution, Kokkos::WithoutInitializing, label + "_medium"),
            capacity)
      , environmentNode_(Kokkos::view_alloc(execution, Kokkos::WithoutInitializing,
                                            label + "_environment_node"),
                         capacity)
      , generation_(Kokkos::view_alloc(execution, Kokkos::WithoutInitializing,
                                       label + "_generation"),
                    capacity)
      , reserved_(Kokkos::view_alloc(execution, Kokkos::WithoutInitializing,
                                     label + "_reserved"),
                  capacity)
      , energyGeV_(
            Kokkos::view_alloc(execution, Kokkos::WithoutInitializing, label + "_energy"),
            capacity)
      , positionM_(Kokkos::view_alloc(execution, Kokkos::WithoutInitializing,
                                      label + "_position"),
                   capacity)
      , direction_(Kokkos::view_alloc(execution, Kokkos::WithoutInitializing,
                                      label + "_direction"),
                   capacity)
      , timeS_(
            Kokkos::view_alloc(execution, Kokkos::WithoutInitializing, label + "_time"),
            capacity)
      , weight_(
            Kokkos::view_alloc(execution, Kokkos::WithoutInitializing, label + "_weight"),
            capacity)
      , historyId_(Kokkos::view_alloc(execution, Kokkos::WithoutInitializing,
                                      label + "_history"),
                   capacity)
      , parentHistoryId_(Kokkos::view_alloc(execution, Kokkos::WithoutInitializing,
                                            label + "_parent_history"),
                         capacity)
      , stepId_(
            Kokkos::view_alloc(execution, Kokkos::WithoutInitializing, label + "_step"),
            capacity) {}

  template <class TMemorySpace>
  KOKKOS_INLINE_FUNCTION ParticleState
  ParticleSoA<TMemorySpace>::load(std::size_t const index) const {
    ParticleState state{};
    state.pid_ = pid_(index);
    state.mediumId_ = mediumId_(index);
    state.environmentNode_ = environmentNode_(index);
    state.generation_ = generation_(index);
    state.reserved_ = reserved_(index);
    state.energyGeV_ = energyGeV_(index);
    state.timeS_ = timeS_(index);
    state.weight_ = weight_(index);
    state.historyId_ = historyId_(index);
    state.parentHistoryId_ = parentHistoryId_(index);
    state.stepId_ = stepId_(index);
    for (int axis = 0; axis < 3; ++axis) {
      state.positionM_[axis] = positionM_(index, axis);
      state.direction_[axis] = direction_(index, axis);
    }
    return state;
  }

  template <class TMemorySpace>
  KOKKOS_INLINE_FUNCTION void ParticleSoA<TMemorySpace>::store(
      std::size_t const index, ParticleState const& state) const {
    pid_(index) = state.pid_;
    mediumId_(index) = state.mediumId_;
    environmentNode_(index) = state.environmentNode_;
    generation_(index) = state.generation_;
    reserved_(index) = state.reserved_;
    energyGeV_(index) = state.energyGeV_;
    timeS_(index) = state.timeS_;
    weight_(index) = state.weight_;
    historyId_(index) = state.historyId_;
    parentHistoryId_(index) = state.parentHistoryId_;
    stepId_(index) = state.stepId_;
    for (int axis = 0; axis < 3; ++axis) {
      positionM_(index, axis) = state.positionM_[axis];
      direction_(index, axis) = state.direction_[axis];
    }
  }

  template <class TMemorySpace>
  ParticleSoARawView ParticleSoA<TMemorySpace>::rawDeviceView() const noexcept {
    return {pid_.data(),
            mediumId_.data(),
            environmentNode_.data(),
            generation_.data(),
            reserved_.data(),
            energyGeV_.data(),
            corsika::kokkos::raw_device_view(positionM_),
            corsika::kokkos::raw_device_view(direction_),
            timeS_.data(),
            weight_.data(),
            historyId_.data(),
            parentHistoryId_.data(),
            stepId_.data()};
  }

  template <class TMemorySpace>
  std::size_t ParticleSoA<TMemorySpace>::deviceBytes() const noexcept {
    auto const view_bytes = [](auto const& view) {
      return view.span() * sizeof(typename std::decay_t<decltype(view)>::value_type);
    };
    return view_bytes(pid_) + view_bytes(mediumId_) + view_bytes(environmentNode_) +
           view_bytes(generation_) + view_bytes(reserved_) + view_bytes(energyGeV_) +
           view_bytes(positionM_) + view_bytes(direction_) + view_bytes(timeS_) +
           view_bytes(weight_) + view_bytes(historyId_) + view_bytes(parentHistoryId_) +
           view_bytes(stepId_);
  }

} // namespace corsika::kokkos

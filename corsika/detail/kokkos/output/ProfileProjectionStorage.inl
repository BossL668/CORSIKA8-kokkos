/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

namespace corsika::kokkos {

  template <typename TExecutionSpace>
  std::size_t ProfileProjectionStorage<TExecutionSpace>::projectedDeviceBytes(
      ProfileProjectionConfig const& source) {
    if (!source.enabled_) return 0;
    if (!(source.axisStepLengthM_ > 0.) || source.axisGrammageGPerCm2_.size() < 2)
      throw std::invalid_argument(
          "Kokkos profile projection requires a valid ShowerAxis support");
    return checked_memory_multiply(source.axisGrammageGPerCm2_.size(), sizeof(double));
  }

  template <typename TExecutionSpace>
  void ProfileProjectionStorage<TExecutionSpace>::initialize(
      ProfileProjectionConfig const& source, TExecutionSpace const& execution) {
    if (!source.enabled_) return;
    (void)projectedDeviceBytes(source);
    axis_ = Kokkos::View<double*, memory_space_type>("c8_kokkos_profile_axis",
                                                     source.axisGrammageGPerCm2_.size());
    auto host = Kokkos::create_mirror_view(axis_);
    for (std::size_t i = 0; i < source.axisGrammageGPerCm2_.size(); ++i)
      host(i) = source.axisGrammageGPerCm2_[i];
    Kokkos::deep_copy(execution, axis_, host);
    for (int component = 0; component < 3; ++component) {
      view_.axisStartPositionM_[component] = source.axisStartPositionM_[component];
      view_.axisDirection_[component] = source.axisDirection_[component];
    }
    view_.axisStepLengthM_ = source.axisStepLengthM_;
    view_.axisGrammageGPerCm2_ = axis_.data();
    view_.axisSupportCount_ = source.axisGrammageGPerCm2_.size();
    enabled_ = true;
  }

  template <typename TExecutionSpace>
  bool ProfileProjectionStorage<TExecutionSpace>::enabled() const noexcept {
    return enabled_;
  }

  template <typename TExecutionSpace>
  ProfileProjectionView ProfileProjectionStorage<TExecutionSpace>::deviceView() const
      noexcept {
    return view_;
  }

  template <typename TExecutionSpace>
  std::size_t ProfileProjectionStorage<TExecutionSpace>::deviceBytes() const noexcept {
    return axis_.extent(0) * sizeof(double);
  }

} // namespace corsika::kokkos

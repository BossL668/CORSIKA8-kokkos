/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

namespace corsika::kokkos {

  template <class T>
  KOKKOS_INLINE_FUNCTION T& RawView1D<T>::operator()(std::size_t const index) const {
    return data_[index];
  }

  template <class T>
  KOKKOS_INLINE_FUNCTION T& RawView2D<T>::operator()(std::size_t const first,
                                                     std::size_t const second) const {
    return data_[first * stride0_ + second * stride1_];
  }

  template <class T, class... TProperties>
  RawView1D<T> raw_device_view(Kokkos::View<T*, TProperties...> const& view) noexcept {
    return {view.data()};
  }

  template <class T, std::size_t NExtent, class... TProperties>
  RawView2D<T> raw_device_view(
      Kokkos::View<T * [NExtent], TProperties...> const& view) noexcept {
    return {view.data(), view.stride_0(), view.stride_1()};
  }

} // namespace corsika::kokkos

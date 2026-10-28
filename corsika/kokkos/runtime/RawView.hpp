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

namespace corsika::kokkos {

  /** Borrowed device arrays; valid only while the owning allocation is unchanged. */
  template <class T>
  struct RawView1D {
    T* data_{};

    KOKKOS_INLINE_FUNCTION T& operator()(std::size_t const index) const;
  };
  template <class T>
  struct RawView2D {
    T* data_{};
    std::size_t stride0_{};
    std::size_t stride1_{};

    KOKKOS_INLINE_FUNCTION T& operator()(std::size_t const first,
                                         std::size_t const second) const;
  };
  template <class T, class... TProperties>
  RawView1D<T> raw_device_view(Kokkos::View<T*, TProperties...> const& view) noexcept;
  template <class T, std::size_t NExtent, class... TProperties>
  RawView2D<T> raw_device_view(
      Kokkos::View<T * [NExtent], TProperties...> const& view) noexcept;

} // namespace corsika::kokkos

#include <corsika/detail/kokkos/runtime/RawView.inl>

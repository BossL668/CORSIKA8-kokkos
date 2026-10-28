/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/kokkos/environment/EnvironmentDataView.hpp>
#include <corsika/kokkos/environment/EnvironmentSnapshot.hpp>

namespace corsika::kokkos {

  /**
   * Borrow the host numerical arrays of an already exported native environment.
   *
   * No allocation, re-export, coordinate conversion or topology search occurs.
   * The snapshot guarantees valid counts and same-index node/shape arrays.
   * Native node/model pointers and coordinate-system handles are not included.
   * Keep the snapshot alive and unchanged until the synchronous storage upload
   * finishes. The returned pointers are host addresses, not device addresses.
   */
  template <typename TModelProperties>
  EnvironmentDataView bind_environment_data(
      EnvironmentSnapshot<TModelProperties> const& snapshot);

  /// Prevent borrowing from a snapshot destroyed at the end of this expression.
  template <typename TModelProperties>
  EnvironmentDataView bind_environment_data(
      EnvironmentSnapshot<TModelProperties>&& snapshot) = delete;

  template <typename TModelProperties>
  EnvironmentDataView bind_environment_data(
      EnvironmentSnapshot<TModelProperties> const&& snapshot) = delete;

} // namespace corsika::kokkos

#include <corsika/detail/kokkos/environment/EnvironmentSnapshotData.inl>

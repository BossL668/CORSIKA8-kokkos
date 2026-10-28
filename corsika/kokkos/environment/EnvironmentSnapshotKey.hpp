/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once
#include <corsika/kokkos/environment/EnvironmentDataView.hpp>
#include <cstdint>
#include <vector>
namespace corsika::kokkos {
  /** Host-side named-field key for session reuse; never compare padding or pointers.
   * Valid production shapes/models retain their original numeric key encoding.
   * Firn adds its two exported deficit parameters; no density is evaluated here.
   * Arrays must describe one valid host-accessible snapshot and remain alive.
   */
  inline std::vector<double> make_environment_snapshot_key(EnvironmentDataView const&);
  /// Original FNV compatibility tag, not an archive-integrity checksum.
  inline std::uint64_t environment_snapshot_fingerprint(EnvironmentDataView const&);
} // namespace corsika::kokkos
#include <corsika/detail/kokkos/environment/EnvironmentSnapshotKey.inl>

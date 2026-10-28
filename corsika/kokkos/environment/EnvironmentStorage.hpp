/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/kokkos/environment/EnvironmentDataView.hpp>
#include <corsika/kokkos/environment/EnvironmentView.hpp>

#include <Kokkos_Core.hpp>
#include <cstddef>
#include <stdexcept>
#include <string>

namespace corsika::kokkos {

  /**
   * Own the device arrays uploaded from an already exported host environment.
   *
   * Retains production FullEnvironmentOwner's allocation, mirror and deep-copy
   * sequence. Shapes are a separate same-index array in the receiving layout.
   * No topology search, model evaluation or environment re-export happens here.
   * Copies share Kokkos allocations; borrowed data views do not own them. Drain
   * the caller's execution before replacing or releasing the last owner, and
   * release all owners before Kokkos finalization.
   */
  template <typename TExecutionSpace>
  class EnvironmentStorage {
  public:
    using memory_space_type = typename TExecutionSpace::memory_space;
    using view_type = EnvironmentView<memory_space_type>;

    EnvironmentStorage() = default;

    /** Upload host-accessible arrays without changing values or index order.
     * Each nonempty array requires a nonnull pointer; the exported snapshot is
     * responsible for valid topology, shapes, model indices and array lengths.
     */
    explicit EnvironmentStorage(EnvironmentDataView const& host);

    /// Managed read-only view; shares ownership of the existing arrays.
    view_type getView() const;
    /// Borrowed pointer/count record for a device context; does not own arrays.
    EnvironmentDataView getDataView() const;
    /// Allocated numerical payload bytes, including the separate shapes.
    std::size_t getDeviceBytes() const;

  private:
    Kokkos::View<EnvironmentNode*, TExecutionSpace> nodes_;
    Kokkos::View<EnvironmentShape*, TExecutionSpace> shapes_;
    Kokkos::View<std::int32_t*, TExecutionSpace> children_, exclusions_;
    Kokkos::View<EnvironmentModel*, TExecutionSpace> models_;
    Kokkos::View<EnvironmentComponent*, TExecutionSpace> composition_;

    template <typename TView, typename T>
    static void upload(TView& out, char const* name, T const* data, std::uint32_t count);
  };

} // namespace corsika::kokkos

#include <corsika/detail/kokkos/environment/EnvironmentStorage.inl>

/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/kokkos/em/tables/MoliereInterpolation.hpp>

#include <Kokkos_Core.hpp>

#include <cstddef>
#include <filesystem>

namespace corsika::kokkos {

  /**
   * Own an immutable copy of the shared Moliere interpolation in device memory.
   *
   * Initialize the Kokkos runtime before constructing this owner and keep it
   * initialized until the owner is destroyed. Copies share immutable Kokkos
   * allocations; a returned view borrows them. Finish all kernels using a view
   * before releasing its last owner.
   *
   * The production caller supplies the electron snapshot for the table shared
   * by electron and muon queries. This owner does not select a particle model.
   * Publication follows both uploads and their completion fence. An exception
   * leaves initialization incomplete and may retain partial allocations, as in
   * the production implementation; getDeviceBytes() includes those allocations.
   */
  template <typename TExecutionSpace>
  class MoliereInterpolationStorage {
  public:
    using execution_space_type = TExecutionSpace;
    using memory_space_type = typename TExecutionSpace::memory_space;

    /** Return the original allocation size of the two interpolation arrays. */
    [[nodiscard]] static constexpr std::size_t projectedDeviceBytes() noexcept;

    /**
     * Load or generate the host table and upload its two arrays exactly once.
     *
     * The cache retains its independent version-one failure semantics. Failed
     * initialization may be retried; an initialized owner rejects another call.
     * No additional memory budget, rollback or reset policy is imposed here.
     */
    void initialize(MoliereSnapshot const& snapshot,
                    std::filesystem::path const& cachePath = {});

    /** Return a non-owning query view; reject access before initialization. */
    [[nodiscard]] MoliereInterpolationView getDeviceView() const;
    /** Return actual array bytes, including allocations from an incomplete upload. */
    [[nodiscard]] std::size_t getDeviceBytes() const noexcept;
    /** Report whether both arrays and the query view have been published. */
    [[nodiscard]] bool isInitialized() const noexcept;

  private:
    using polynomial_view_type = Kokkos::View<MoliereCubicPolynomial*, memory_space_type>;
    using double_view_type = Kokkos::View<double*, memory_space_type>;

    execution_space_type execution_{};
    polynomial_view_type polynomials_{};
    double_view_type initialGuess_{};
    MoliereInterpolationView view_{};
    bool initialized_{};
  };

} // namespace corsika::kokkos

#include <corsika/detail/kokkos/em/execution/MoliereInterpolationStorage.inl>

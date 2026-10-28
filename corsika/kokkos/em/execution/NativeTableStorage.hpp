/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/kokkos/em/tables/ProposalNativeTable.hpp>

#include <Kokkos_Core.hpp>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace corsika::kokkos {

  /**
   * Own an immutable native-table copy in one execution space's memory.
   *
   * Initialization validates the canonical export and its content hash before
   * checking the memory budget or allocating arrays. It publishes a numerical
   * view only after the original upload order and completion fence. Allocation
   * failure resets partial storage. No PROPOSAL object enters device memory.
   *
   * Initialize the Kokkos runtime before constructing this storage, and keep it
   * initialized until the storage is destroyed. A copied storage object shares
   * immutable allocations; a returned view does not own them.
   * Finish all kernels using a view before resetting or releasing its last owner.
   */
  template <typename TExecutionSpace>
  class NativeTableStorage {
  public:
    using execution_space_type = TExecutionSpace;
    using memory_space_type = typename TExecutionSpace::memory_space;

    /** Return the production allocation budget for all exported arrays. */
    [[nodiscard]] static std::size_t projectedDeviceBytes(
        ProposalNativeTableSet const& source);

    /**
     * Validate and upload exactly once, until reset() is called.
     *
     * \param source Canonical format-v7 export with a matching content hash.
     * \param maximumBytes Upper bound on the original array-byte accounting.
     * \throws std::logic_error If this owner is already initialized.
     * \throws std::invalid_argument For invalid exported data or a hash mismatch.
     * \throws std::runtime_error If the configured budget is insufficient.
     */
    void initialize(ProposalNativeTableSet const& source,
                    std::size_t maximumBytes = std::numeric_limits<std::size_t>::max());

    /** Release owned allocations; the caller must have completed their queries. */
    void reset() noexcept;
    [[nodiscard]] bool isInitialized() const noexcept;
    [[nodiscard]] std::size_t getDeviceBytes() const noexcept;
    /** Return the uploaded hash; reject access before initialization. */
    [[nodiscard]] sha256_digest_type const& getSourceContentHash() const;
    /** Return a non-owning view; reject access before initialization. */
    [[nodiscard]] ProposalNativeDeviceView getDeviceView() const;

  private:
    template <typename TValue>
    using array_view_type = Kokkos::View<TValue*, memory_space_type>;

    template <typename TValue>
    array_view_type<TValue> upload(std::vector<TValue> const& source,
                                   std::string const& label);
    void requireInitialized() const;

    execution_space_type execution_{};
    array_view_type<NativeDndxColumn> dndx_{};
    array_view_type<NativeTotalRateColumn> totalRate_{};
    array_view_type<std::uint32_t> selectionIndices_{};
    array_view_type<NativeDedxColumn> dedx_{};
    array_view_type<std::uint32_t> dedxAccumulationIndices_{};
    array_view_type<NativeUtilityColumn> utility_{};
    array_view_type<double> bicubicValues_{};
    array_view_type<double> bicubicDerivativeEnergy_{};
    array_view_type<double> bicubicDerivativeLoss_{};
    array_view_type<double> bicubicMixedDerivative_{};
    array_view_type<double> bicubicPolynomial_{};
    array_view_type<double> cubicValues_{};
    array_view_type<double> cubicDerivatives_{};
    ProposalNativeDeviceView view_{};
    sha256_digest_type hash_{};
    std::size_t bytes_{};
    bool initialized_{};
  };

} // namespace corsika::kokkos

#include <corsika/detail/kokkos/em/execution/NativeTableStorage.inl>

/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/kokkos/em/execution/PhysicsContextData.hpp>
#include <Kokkos_Core.hpp>
#include <cstddef>
#include <stdexcept>

namespace corsika::kokkos {

  /** Own one rank-zero device record and its persistent host staging record.
   * Initialize after all borrowed arrays have been uploaded. The caller must
   * finish previous kernels before reinitialization or beginShower(), and keep
   * Kokkos and array owners alive until all users of getDeviceView() finish.
   * Both publication operations retain production's copy and completion fence;
   * no per-wavefront synchronization is added. Copies share both allocations,
   * as in production; use one owner to update a shared context.
   */
  template <typename TExecutionSpace>
  class PhysicsContext {
  public:
    using memory_space_type = typename TExecutionSpace::memory_space;
    using device_view_type = Kokkos::View<PhysicsContextData const, memory_space_type>;

    /** Bytes owned by the device context record, excluding borrowed arrays. */
    static constexpr std::size_t projectedDeviceBytes() noexcept;
    /** Publish numerical models, environment views and shower state.
     * All supplied views borrow already uploaded arrays. Their owners must
     * remain valid until the last kernel using this context has completed.
     * Reinitialization reuses the context allocation after prior work ends.
     */
    void initialize(
        InteractionPhysicsView const& physics, EnvironmentBorrowedView const& environment,
        TransportGeometryConfig const& geometry, MoliereSnapshot const& electronMoliere,
        MoliereSnapshot const& muonMoliere,
        MoliereInterpolationView const& moliereInterpolation,
        bool const muonMoliereAvailable, PhotonPairLpmSnapshot const& photonPairLpm,
        BremsLpmSnapshot const& bremsLpm, EmThinningConfig const& thinning,
        ProfileProjectionView const& profileProjection, std::uint64_t const randomSeed,
        std::uint64_t const showerId, TExecutionSpace const& execution);
    /** Replace only shower keys and thinning; reject calls before initialization. */
    void beginShower(EmThinningConfig const& thinning, std::uint64_t const randomSeed,
                     std::uint64_t const showerId, TExecutionSpace const& execution);
    /** Whether the initial context publication has completed. */
    bool isInitialized() const noexcept;
    /** Read-only device record; retaining it does not retain the borrowed arrays. */
    device_view_type getDeviceView() const noexcept;
    /** Owned device bytes, or zero before allocation; excludes borrowed arrays. */
    std::size_t getDeviceBytes() const noexcept;

  private:
    using device_storage_type = Kokkos::View<PhysicsContextData, memory_space_type>;
    using host_storage_type = Kokkos::View<PhysicsContextData, Kokkos::HostSpace>;
    device_storage_type device_{};
    host_storage_type host_{};
    bool initialized_{};
  };

} // namespace corsika::kokkos

#include <corsika/detail/kokkos/em/execution/PhysicsContext.inl>

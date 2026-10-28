/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>
#include <Kokkos_Core.hpp>
#include <corsika/kokkos/runtime/HostStagingSpace.hpp>
#include <corsika/kokkos/runtime/CheckedMemory.hpp>
#include <corsika/kokkos/output/ProfileAtomicOperations.hpp>
#include <corsika/kokkos/output/HostProfileShards.hpp>
#include <corsika/kokkos/output/algorithms/ProfileAccumulation.hpp>

namespace corsika::kokkos {

  /** Own eight fixed-point histograms and their production energy ledger.
   * Kokkos must remain initialized for this owner's lifetime. Borrowed views and
   * step records must remain valid until their execution-space work completes;
   * reset, reinitialization and destruction require an idle owner. Use the same
   * execution-space instance for ordered accumulation and download.
   */
  template <typename TExecutionSpace>
  class ProfileAccumulatorStorage {
  public:
    using memory_space_type = typename TExecutionSpace::memory_space;
    using host_staging_space_type =
        corsika::kokkos::host_staging_space_type<TExecutionSpace>;
    using photon_step_view_type = Kokkos::View<PhotonTransportRecord*, memory_space_type>;
    using photon_final_view_type =
        Kokkos::View<PhotonFinalStateRecord*, memory_space_type>;
    using lepton_step_view_type = Kokkos::View<LeptonTransportRecord*, memory_space_type>;
    using lepton_final_view_type =
        Kokkos::View<LeptonFinalStateRecord*, memory_space_type>;

    static std::size_t projectedDeviceBytes(ProfileProjectionConfig const& config);

    void initialize(ProfileProjectionConfig const& config,
                    TExecutionSpace const& execution = {});

    void reset(double const weight_limit, double const energy_limit_GeV,
               TExecutionSpace const& execution = {});

    bool enabled() const noexcept;
    void enableHostShards([[maybe_unused]] std::size_t workers,
                          [[maybe_unused]] std::size_t budget);
    // Keep the cooperative endpoint's existing interface and opt-in unchanged.
    void enableCooperativeHostShards(std::size_t workers, std::size_t budget);
    HostProfileShards const& hostShards() const noexcept;
    void requireOpenForAccumulation() const;
    /** Borrow the current buffers; the returned view does not own storage. */
    ProfileAccumulatorView deviceView() const noexcept;
    std::size_t deviceBytes() const noexcept;

    void accumulatePhoton(ProfileProjectionView const projection,
                          photon_step_view_type const& steps,
                          std::size_t const step_count,
                          photon_final_view_type const& final_states,
                          std::size_t const final_state_count,
                          TExecutionSpace const& execution = {});

    void accumulateLepton(ProfileProjectionView const projection,
                          lepton_step_view_type const& steps,
                          std::size_t const step_count,
                          lepton_final_view_type const& final_states,
                          std::size_t const final_state_count,
                          TExecutionSpace const& execution = {});

    /** Seal this event and export checked integers, merging host shards first.
     * Invalid records or overflow throw; a failed export stays sealed until reset.
     */
    FixedProfileSnapshot downloadFixed(std::string const& identity,
                                       CooperativeEndpoint endpoint,
                                       TExecutionSpace const& execution = {});

    /** Copy and decode the current event once. Without host shards, diagnostic
     * counts are returned for the caller to check. With shards, downloadFixed
     * validates and merges the integers before decoding. Reset before reuse.
     */
    ProfileResult download(TExecutionSpace const& execution = {});

  private:
    HostProfileShards hostShards_;
    ProfileProjectionConfig fixedSnapshotConfig_{};
    Kokkos::View<long long*, memory_space_type> histograms_{};
    Kokkos::View<ProfileCounters*, memory_space_type> counters_{};
    Kokkos::View<long long*, host_staging_space_type> hostHistograms_{};
    Kokkos::View<ProfileCounters*, host_staging_space_type> hostCounters_{};
    ProfileAccumulatorView view_{};
    std::size_t bins_{};
    bool enabled_{};
    bool downloaded_{};
    bool fixedSnapshotSealed_{};
  };

} // namespace corsika::kokkos

#include <corsika/detail/kokkos/output/ProfileAccumulatorStorage.inl>

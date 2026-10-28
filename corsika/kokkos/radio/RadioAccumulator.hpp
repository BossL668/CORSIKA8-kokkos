/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <Kokkos_Core.hpp>
#include <corsika/kokkos/radio/RadioAccumulatorOperations.hpp>
#include <corsika/kokkos/radio/CooperativeRadioMerge.hpp>
#include <corsika/kokkos/radio/RadioStatistics.hpp>
#include <corsika/kokkos/radio/algorithms/RadioProjection.hpp>
#include <corsika/kokkos/runtime/ResidentMemoryBudget.hpp>
#include <corsika/kokkos/runtime/TuningParameters.hpp>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace corsika::kokkos::radio {
  /** Own production radio storage and submit work on one execution instance.
   * Keep this object and external device inputs alive until queued work finishes.
   * Host enqueue copies records before returning. reset begins a new shower only
   * after pending host input is consumed; download/downloadFixed finalize output.
   * Integer and floating modes, tiling and batching follow the production path.
   * Use the same execution instance for initialization, submission and finalization.
   */
  template <class TExecutionSpace>
  class RadioAccumulator {
  public:
#if defined(KOKKOS_ENABLE_OPENMP)
    static constexpr bool IsOpenMP = std::is_same_v<TExecutionSpace, Kokkos::OpenMP>;
#else
    static constexpr bool IsOpenMP = false;
#endif
    using memory_space_type = typename TExecutionSpace::memory_space;
    using track_input_view_type =
        Kokkos::View<::corsika::kokkos::LeptonTransportRecord*, memory_space_type>;
    using host_track_input_view_type = typename track_input_view_type::HostMirror;
    // Independent of the EM dispatch threshold: even a small particle batch
    // can create substantial radio work when many observers are configured.
    static constexpr std::size_t HostTrackBatchCapacity = 4096;
    using track_view_type = Kokkos::View<RadioTrackKinematics*, memory_space_type>;
    using observer_view_type = Kokkos::View<RadioObserverView*, memory_space_type>;
    using fixed_waveform_view_type = Kokkos::View<long long*, memory_space_type>;
    using floating_waveform_view_type = Kokkos::View<double*, memory_space_type>;
    using thread_fixed_waveform_view_type =
        Kokkos::View<long long**, Kokkos::LayoutRight, memory_space_type>;
    using thread_counter_view_type = Kokkos::View<RadioCounters*, memory_space_type>;
    using counter_view_type = Kokkos::View<RadioCounters*, memory_space_type>;

    static std::size_t projectedDeviceBytes(RadioConfig const& requested);

    static constexpr std::size_t TrackTileSize = IsOpenMP ? 4 : 8;
    static constexpr std::size_t ObserverTileSize = IsOpenMP ? 4 : 32;
    static constexpr std::size_t MaximumTrackTileSize = 16;
    static constexpr std::size_t MaximumObserverTileSize = 64;

    struct RadioTileScratch {
      RadioTrackKinematics tracks_[MaximumTrackTileSize];
      RadioObserverView coreasObservers_[MaximumObserverTileSize];
      RadioObserverView zhsObservers_[MaximumObserverTileSize];
    };

    // A named functor gives host and device passes the same kernel identity.
    // NVCC can number the two extended lambdas inside project()'s constexpr
    // OpenMP/GPU branches differently, leaving the tiled GPU entry unregistered.
    // This is the former GPU lambda body, with the same data and tile order.
    template <bool Coreas, bool Zhs, bool ReduceCounters = false>
    struct RadioTileProjection {
      track_view_type tracks_;
      observer_view_type coreasObservers_;
      observer_view_type zhsObservers_;
      RadioPropagationView propagation_;
      RadioWaveformView coreasWaveforms_;
      RadioWaveformView zhsWaveforms_;
      counter_view_type counters_;
      std::size_t trackCount_;
      std::size_t observerCount_;
      std::size_t observerTiles_;
      std::size_t trackTileSize_;
      std::size_t observerTileSize_;
      std::size_t tileLanes_;

      using member_type = typename Kokkos::TeamPolicy<TExecutionSpace>::member_type;
      KOKKOS_INLINE_FUNCTION void operator()(member_type const& team) const;
    };

    // Native CUDA falls back to its flat pair kernel after the EM workspace
    // consumes the optional track-cache budget.  Production traces show that
    // this is also the faster launch shape for the many small edge
    // wavefronts: it exposes all track/observer pairs directly and avoids the
    // TeamPolicy overhead of mostly empty tiles.
    static constexpr std::size_t DirectProjectionTrackThreshold = 16384;

    void initialize(RadioConfig const& requested,
                    ::corsika::kokkos::TuningParameters const& tuning = {},
                    TExecutionSpace const& execution = {});

    bool enabled() const noexcept;

    void reset(TExecutionSpace const& execution = {});

    // Copy ownership immediately; scalar transport may reuse its record as
    // soon as this method returns. No allocation or fence occurs per track.
    void enqueueHostLeptonTracks(::corsika::kokkos::LeptonTransportRecord const* records,
                                 std::size_t count,
                                 TExecutionSpace const& execution = {});

    void flushHostLeptonTracks(TExecutionSpace const& execution = {});

    void accumulateLeptonTracks(track_input_view_type const& records,
                                std::size_t const count,
                                TExecutionSpace const& execution = {});

    // Cooperative end-of-shower export. No conversion/differentiation, and
    // no default execution-space copy that might wait on another endpoint.
    FixedRadioSnapshot downloadFixed(std::string const& shower_identity,
                                     CooperativeEndpoint endpoint,
                                     TExecutionSpace const& execution = {});

    RadioWaveforms download(TExecutionSpace const& execution = {});

    RadioStatistics const& statistics() const noexcept;

    std::size_t deviceBytes() const noexcept;

    /**
     * Bytes copied from immutable host configuration during initialize().
     * Waveforms and counters are initialized directly in the execution space,
     * so they are resident bytes but not host-to-device configuration bytes.
     */
    std::size_t staticHostToDeviceBytes() const;

    void reserveTrackCapacity(std::size_t const count, TExecutionSpace const& execution);

    ::corsika::kokkos::MemoryProjection projectedTrackCapacity(
        std::size_t const count) const;

  private:
    template <class THostValue, class TView>
    static void upload(std::vector<THostValue> const& host, TView& device,
                       TExecutionSpace const& execution, char const* label);

    std::vector<RadioObserverView> makeObservers(
        std::vector<RadioObserverSnapshot> const& input, std::size_t& total_bins,
        bool zhs_potential) const;

    static bool observersCanShareTrackKinematics(
        std::vector<RadioObserverView> const& coreas,
        std::vector<RadioObserverView> const& zhs);

    void allocateWaveforms(std::size_t bins, fixed_waveform_view_type& fixed,
                           floating_waveform_view_type& floating, RadioWaveformView& view,
                           char const* label);

    void allocateThreadWaveforms(std::size_t bins, thread_fixed_waveform_view_type& fixed,
                                 std::size_t thread_slots,
                                 TExecutionSpace const& execution, char const* label);

    unsigned long long mergeThreadWaveform(
        thread_fixed_waveform_view_type const& private_waveform,
        fixed_waveform_view_type const& waveform, TExecutionSpace const& execution,
        char const* label);

    void mergeThreadWaveforms(TExecutionSpace const& execution);

    void ensureTrackCapacity(std::size_t count, TExecutionSpace const& execution);

  public:
    // Public only to satisfy NVCC's extended-lambda access restriction.  This
    // function is an implementation entry point; callers should use
    // accumulateLeptonTracks instead.
    template <bool Coreas, bool Zhs>
    void projectDirect(track_input_view_type const& records, std::size_t track_count,
                       observer_view_type const& coreas_observers,
                       observer_view_type const& zhs_observers,
                       std::size_t observer_count, TExecutionSpace const& execution);

    template <bool Coreas, bool Zhs>
    void project(track_view_type const& tracks, std::size_t track_count,
                 observer_view_type const& coreas_observers,
                 observer_view_type const& zhs_observers, std::size_t observer_count,
                 TExecutionSpace const& execution);

  private:
    template <class TFixedView, class TFloatView>
    std::vector<RadioWaveform> downloadSet(
        std::vector<RadioObserverSnapshot> const& observers, std::size_t total_bins,
        TFixedView const& fixed, TFloatView const& floating, bool zhs_potential);

    void copyStatistics(RadioCounters const& counters);

    static void validate(RadioConfig const& config);

    RadioConfig config_{};
    Kokkos::View<double*, memory_space_type> refractivity_{};
    Kokkos::View<double*, memory_space_type> integratedRefractivity_{};
    observer_view_type coreasObservers_{};
    observer_view_type zhsObservers_{};
    std::vector<RadioObserverView> coreasObserversHost_{};
    std::vector<RadioObserverView> zhsObserversHost_{};
    fixed_waveform_view_type coreasFixed_{};
    fixed_waveform_view_type zhsFixed_{};
    floating_waveform_view_type coreasFloating_{};
    floating_waveform_view_type zhsFloating_{};
    thread_fixed_waveform_view_type coreasThreadFixed_{};
    thread_fixed_waveform_view_type zhsThreadFixed_{};
    thread_counter_view_type threadCounters_{};
    std::size_t openmpThreadSlots_{};
    RadioWaveformView coreasWaveforms_{};
    RadioWaveformView zhsWaveforms_{};
    RadioPropagationView propagation_{};
    counter_view_type counters_{};
    track_view_type tracks_{};
    track_input_view_type hostTrackInputDevice_{};
    host_track_input_view_type hostTrackInput_{};
    std::size_t hostTrackCount_{};
    RadioStatistics statistics_{};
    std::size_t coreasBins_{};
    std::size_t zhsBins_{};
    bool fuseCoreasZhs_{};
    bool initialized_{};
    bool downloaded_{};
    bool fixedSnapshotSealed_{};
    std::size_t trackTileSize_{TrackTileSize};
    std::size_t observerTileSize_{ObserverTileSize};
    std::size_t teamSize_{IsOpenMP ? 1U : 256U};
  };
} // namespace corsika::kokkos::radio

#include <corsika/detail/kokkos/radio/RadioAccumulator.inl>

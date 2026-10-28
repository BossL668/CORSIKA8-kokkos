/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

namespace corsika::kokkos::radio {
  template <class TExecutionSpace>
  template <bool Coreas, bool Zhs, bool ReduceCounters>
  KOKKOS_INLINE_FUNCTION void RadioAccumulator<TExecutionSpace>::RadioTileProjection<
      Coreas, Zhs, ReduceCounters>::operator()(member_type const& team) const {
    auto* const scratch = static_cast<RadioTileScratch*>(
        team.team_shmem().get_shmem(sizeof(RadioTileScratch)));
    auto const track_tile = static_cast<std::size_t>(team.league_rank()) / observerTiles_;
    auto const observer_tile =
        static_cast<std::size_t>(team.league_rank()) % observerTiles_;
    auto const track_base = track_tile * trackTileSize_;
    auto const observer_base = observer_tile * observerTileSize_;
    for (std::size_t index = team.team_rank(); index < trackTileSize_;
         index += team.team_size()) {
      auto const global = track_base + index;
      scratch->tracks_[index] =
          global < trackCount_ ? tracks_(global) : RadioTrackKinematics{};
    }
    for (std::size_t index = team.team_rank(); index < observerTileSize_;
         index += team.team_size()) {
      auto const global = observer_base + index;
      if (global < observerCount_) {
        if (Coreas) scratch->coreasObservers_[index] = coreasObservers_(global);
        if (Zhs) scratch->zhsObservers_[index] = zhsObservers_(global);
      }
    }
    team.team_barrier();
    RadioCounters local_counts{};
    using projection_operations_type =
        std::conditional_t<ReduceCounters,
                           accumulator_detail::RadioLocalCounterOperations,
                           RadioAtomicOperations>;
    auto* const projection_counts = ReduceCounters ? &local_counts : counters_.data();
    for (std::size_t lane = team.team_rank(); lane < tileLanes_;
         lane += team.team_size()) {
      auto const track_lane = lane / observerTileSize_;
      auto const observer_lane = lane % observerTileSize_;
      auto const track_index = track_tile * trackTileSize_ + track_lane;
      auto const observer_index = observer_tile * observerTileSize_ + observer_lane;
      if (track_index >= trackCount_ || observer_index >= observerCount_ ||
          scratch->tracks_[track_lane].valid_ == 0)
        continue;
      if (Coreas)
        accumulate_co_reas<projection_operations_type>(
            scratch->tracks_[track_lane], propagation_,
            scratch->coreasObservers_[observer_lane], coreasWaveforms_,
            projection_counts);
      if (Zhs)
        accumulate_zhs<projection_operations_type>(
            scratch->tracks_[track_lane], propagation_,
            scratch->zhsObservers_[observer_lane], zhsWaveforms_, projection_counts);
    }
    if constexpr (ReduceCounters) {
      // All team members participate, including empty tail-tile lanes.
      team.team_reduce(
          Kokkos::Sum<unsigned long long>(local_counts.coreasContributions_));
      if (team.team_rank() == 0 && local_counts.coreasContributions_)
        RadioAtomicOperations::add(&counters_(0).coreasContributions_,
                                   local_counts.coreasContributions_);
      team.team_reduce(Kokkos::Sum<unsigned long long>(local_counts.zhsContributions_));
      if (team.team_rank() == 0 && local_counts.zhsContributions_)
        RadioAtomicOperations::add(&counters_(0).zhsContributions_,
                                   local_counts.zhsContributions_);
      team.team_reduce(Kokkos::Sum<unsigned long long>(local_counts.zhsSubtracks_));
      if (team.team_rank() == 0 && local_counts.zhsSubtracks_)
        RadioAtomicOperations::add(&counters_(0).zhsSubtracks_,
                                   local_counts.zhsSubtracks_);
      team.team_reduce(
          Kokkos::Sum<unsigned long long>(local_counts.fixedPointOverflows_));
      if (team.team_rank() == 0 && local_counts.fixedPointOverflows_)
        RadioAtomicOperations::add(&counters_(0).fixedPointOverflows_,
                                   local_counts.fixedPointOverflows_);
    }
  }

  template <class TExecutionSpace>
  std::size_t RadioAccumulator<TExecutionSpace>::projectedDeviceBytes(
      RadioConfig const& requested) {
    if (!requested.enabled_) return 0;
    validate(requested);
    using ::corsika::kokkos::checked_memory_add;
    using ::corsika::kokkos::checked_memory_multiply;
    auto total = checked_memory_multiply(requested.propagation_.refractivity_.size(),
                                         sizeof(double));
    total = checked_memory_add(
        total,
        checked_memory_multiply(requested.propagation_.integratedRefractivity_.size(),
                                sizeof(double)));
    total = checked_memory_add(total,
                               checked_memory_multiply(requested.coreasObservers_.size(),
                                                       sizeof(RadioObserverView)));
    total =
        checked_memory_add(total, checked_memory_multiply(requested.zhsObservers_.size(),
                                                          sizeof(RadioObserverView)));
    auto append_waveforms = [&](auto const& observers) {
      std::size_t bins{};
      for (auto const& observer : observers)
        bins = checked_memory_add(bins, static_cast<std::size_t>(observer.numberOfBins_));
      auto const values = checked_memory_multiply(bins, std::size_t{3});
      total = checked_memory_add(
          total,
          checked_memory_multiply(
              values, requested.deterministic_ ? sizeof(long long) : sizeof(double)));
      if constexpr (IsOpenMP) {
        if (requested.deterministic_) {
          auto const thread_slots =
              static_cast<std::size_t>(std::max(1, TExecutionSpace().concurrency()));
          total = checked_memory_add(
              total,
              checked_memory_multiply(checked_memory_multiply(values, thread_slots),
                                      sizeof(long long)));
        }
      }
    };
    append_waveforms(requested.coreasObservers_);
    append_waveforms(requested.zhsObservers_);
    total = checked_memory_add(total, sizeof(RadioCounters));
    total = checked_memory_add(
        total, checked_memory_multiply(HostTrackBatchCapacity,
                                       sizeof(::corsika::kokkos::LeptonTransportRecord)));
    if constexpr (IsOpenMP) {
      if (requested.deterministic_) {
        auto const thread_slots =
            static_cast<std::size_t>(std::max(1, TExecutionSpace().concurrency()));
        total = checked_memory_add(
            total, checked_memory_multiply(thread_slots, sizeof(RadioCounters)));
      }
    }
    return total;
  }

  template <class TExecutionSpace>
  void RadioAccumulator<TExecutionSpace>::initialize(
      RadioConfig const& requested, ::corsika::kokkos::TuningParameters const& tuning,
      TExecutionSpace const& execution) {
    if (initialized_)
      throw std::logic_error("Kokkos radio accumulator already initialized");
    config_ = requested;
    if (tuning.trackTileSize_ != 0) trackTileSize_ = tuning.trackTileSize_;
    if (tuning.observerTileSize_ != 0) observerTileSize_ = tuning.observerTileSize_;
    if (tuning.teamSize_ != 0) teamSize_ = tuning.teamSize_;
    if (trackTileSize_ == 0 || observerTileSize_ == 0 || teamSize_ == 0 ||
        teamSize_ > 1024)
      throw std::invalid_argument("Kokkos radio tuning is invalid");
    if constexpr (!IsOpenMP) {
      if (trackTileSize_ > MaximumTrackTileSize ||
          observerTileSize_ > MaximumObserverTileSize)
        throw std::invalid_argument(
            "Kokkos GPU radio tiling exceeds the validated scratch limits");
    }
    if (!requested.enabled_) return;
    validate(requested);
    coreasObserversHost_ = makeObservers(requested.coreasObservers_, coreasBins_, false);
    zhsObserversHost_ = makeObservers(requested.zhsObservers_, zhsBins_, true);
    upload(coreasObserversHost_, coreasObservers_, execution,
           "c8_kokkos_coreas_observers");
    upload(zhsObserversHost_, zhsObservers_, execution, "c8_kokkos_zhs_observers");
    upload(requested.propagation_.refractivity_, refractivity_, execution,
           "c8_kokkos_radio_refractivity");
    upload(requested.propagation_.integratedRefractivity_, integratedRefractivity_,
           execution, "c8_kokkos_radio_integrated_refractivity");
    propagation_.homogeneousRefractiveIndex_ =
        requested.propagation_.homogeneousRefractiveIndex_;
    propagation_.minimumHeightM_ = requested.propagation_.minimumHeightM_;
    propagation_.maximumHeightM_ = requested.propagation_.maximumHeightM_;
    propagation_.stepM_ = requested.propagation_.stepM_;
    propagation_.inverseStepPerM_ = requested.propagation_.inverseStepPerM_;
    propagation_.slopeRefractivityLower_ = requested.propagation_.slopeRefractivityLower_;
    propagation_.slopeIntegratedRefractivityLower_ =
        requested.propagation_.slopeIntegratedRefractivityLower_;
    propagation_.slopeRefractivityUpper_ = requested.propagation_.slopeRefractivityUpper_;
    propagation_.slopeIntegratedRefractivityUpper_ =
        requested.propagation_.slopeIntegratedRefractivityUpper_;
    propagation_.refractivity_ = refractivity_.data();
    propagation_.integratedRefractivity_ = integratedRefractivity_.data();
    propagation_.tableSize_ = refractivity_.extent(0);
    propagation_.zhsSubtrackRefinement_ = requested.zhsSubtrackRefinement_;
    allocateWaveforms(coreasBins_, coreasFixed_, coreasFloating_, coreasWaveforms_,
                      "c8_kokkos_coreas_waveforms");
    allocateWaveforms(zhsBins_, zhsFixed_, zhsFloating_, zhsWaveforms_,
                      "c8_kokkos_zhs_waveforms");
    if constexpr (IsOpenMP) {
      if (requested.deterministic_) {
        Kokkos::Experimental::UniqueToken<TExecutionSpace> tokens(execution);
        openmpThreadSlots_ = std::max<std::size_t>(1, tokens.size());
        allocateThreadWaveforms(coreasBins_, coreasThreadFixed_, openmpThreadSlots_,
                                execution, "c8_kokkos_coreas_thread_waveforms");
        allocateThreadWaveforms(zhsBins_, zhsThreadFixed_, openmpThreadSlots_, execution,
                                "c8_kokkos_zhs_thread_waveforms");
        threadCounters_ = thread_counter_view_type("c8_kokkos_radio_thread_counters",
                                                   openmpThreadSlots_);
      }
    }
    counters_ = counter_view_type("c8_kokkos_radio_counters", 1);
    hostTrackInputDevice_ =
        track_input_view_type(Kokkos::view_alloc(execution, Kokkos::WithoutInitializing,
                                                 "c8_kokkos_host_radio_input"),
                              HostTrackBatchCapacity);
    hostTrackInput_ = Kokkos::create_mirror_view(hostTrackInputDevice_);
    fuseCoreasZhs_ =
        requested.coreasEnabled_ && requested.zhsEnabled_ &&
        observersCanShareTrackKinematics(coreasObserversHost_, zhsObserversHost_);
    initialized_ = true;
    reset(execution);
  }

  template <class TExecutionSpace>
  bool RadioAccumulator<TExecutionSpace>::enabled() const noexcept {
    return initialized_ && config_.enabled_;
  }

  template <class TExecutionSpace>
  void RadioAccumulator<TExecutionSpace>::reset(TExecutionSpace const& execution) {
    if (!enabled()) return;
    if (hostTrackCount_ != 0)
      throw std::logic_error("cannot reset radio with uncommitted host tracks");
    if (coreasFixed_.extent(0)) Kokkos::deep_copy(execution, coreasFixed_, 0LL);
    if (zhsFixed_.extent(0)) Kokkos::deep_copy(execution, zhsFixed_, 0LL);
    if (coreasFloating_.extent(0)) Kokkos::deep_copy(execution, coreasFloating_, 0.);
    if (zhsFloating_.extent(0)) Kokkos::deep_copy(execution, zhsFloating_, 0.);
    if constexpr (IsOpenMP) {
      if (coreasThreadFixed_.extent(0))
        Kokkos::deep_copy(execution, coreasThreadFixed_, 0LL);
      if (zhsThreadFixed_.extent(0)) Kokkos::deep_copy(execution, zhsThreadFixed_, 0LL);
      if (threadCounters_.extent(0))
        Kokkos::deep_copy(execution, threadCounters_, RadioCounters{});
    }
    Kokkos::deep_copy(execution, counters_, RadioCounters{});
    execution.fence("reset Kokkos radio accumulator");
    statistics_ = {};
    statistics_.deviceBytes_ = deviceBytes();
    statistics_.trackPrecomputeEnabled_ = true;
    statistics_.trackTileSize_ = trackTileSize_;
    statistics_.observerTileSize_ = observerTileSize_;
    statistics_.trackDiagnosticsEnabled_ = config_.trackDiagnostics_;
    downloaded_ = false;
    fixedSnapshotSealed_ = false;
  }

  template <class TExecutionSpace>
  void RadioAccumulator<TExecutionSpace>::enqueueHostLeptonTracks(
      ::corsika::kokkos::LeptonTransportRecord const* records, std::size_t count,
      TExecutionSpace const& execution) {
    if (!enabled())
      throw std::logic_error("host radio tracks requested while radio is disabled");
    if (downloaded_ || fixedSnapshotSealed_)
      throw std::logic_error("radio already downloaded; reset before host tracks");
    if (count != 0 && records == nullptr)
      throw std::invalid_argument("null host radio track input");
    while (count != 0) {
      auto const copied = std::min(count, HostTrackBatchCapacity - hostTrackCount_);
      for (std::size_t index = 0; index < copied; ++index)
        hostTrackInput_(hostTrackCount_ + index) = records[index];
      hostTrackCount_ += copied;
      statistics_.hostTracksEnqueued_ += copied;
      records += copied;
      count -= copied;
      if (hostTrackCount_ == HostTrackBatchCapacity) flushHostLeptonTracks(execution);
    }
  }

  template <class TExecutionSpace>
  void RadioAccumulator<TExecutionSpace>::flushHostLeptonTracks(
      TExecutionSpace const& execution) {
    if (hostTrackCount_ == 0) return;
    auto const range = std::make_pair(std::size_t{0}, hostTrackCount_);
    Kokkos::deep_copy(execution, Kokkos::subview(hostTrackInputDevice_, range),
                      Kokkos::subview(hostTrackInput_, range));
    // The reusable host mirror may be filled again while projection runs.
    // Complete its upload first; all projection/device reuse stays ordered
    // on this same execution stream, without a per-track synchronization.
    execution.fence("upload batched host radio tracks");
    accumulateLeptonTracks(hostTrackInputDevice_, hostTrackCount_, execution);
    statistics_.hostToDeviceBytes_ +=
        hostTrackCount_ * sizeof(::corsika::kokkos::LeptonTransportRecord);
    ++statistics_.hostTrackBatches_;
    hostTrackCount_ = 0;
  }

  template <class TExecutionSpace>
  void RadioAccumulator<TExecutionSpace>::accumulateLeptonTracks(
      track_input_view_type const& records, std::size_t const count,
      TExecutionSpace const& execution) {
    if (!enabled() || count == 0) return;
    if (fixedSnapshotSealed_)
      throw std::logic_error("radio endpoint already committed; reset before new tracks");
    if (count > records.extent(0))
      throw std::out_of_range("Kokkos radio track count exceeds input view");
    if constexpr (!IsOpenMP) {
      if (count < DirectProjectionTrackThreshold) {
        auto counters = counters_;
        auto const collect_diagnostics = config_.trackDiagnostics_;
        Kokkos::parallel_for(
            "c8_kokkos_scan_direct_radio_tracks",
            Kokkos::RangePolicy<TExecutionSpace>(execution, 0, count),
            KOKKOS_LAMBDA(std::size_t const index) {
              record_track<RadioAtomicOperations>(records(index), nullptr,
                                                  counters.data(), collect_diagnostics);
            });
        if (fuseCoreasZhs_) {
          projectDirect<true, true>(records, count, coreasObservers_, zhsObservers_,
                                    coreasObservers_.extent(0), execution);
        } else {
          if (config_.coreasEnabled_)
            projectDirect<true, false>(records, count, coreasObservers_, {},
                                       coreasObservers_.extent(0), execution);
          if (config_.zhsEnabled_)
            projectDirect<false, true>(records, count, {}, zhsObservers_,
                                       zhsObservers_.extent(0), execution);
        }
        statistics_.directProjectionBatches_++;
        statistics_.directProjectionRecords_ += count;
        statistics_.maximumTrackBatch_ = std::max(statistics_.maximumTrackBatch_, count);
        downloaded_ = false;
        return;
      }
    }
    ensureTrackCapacity(count, execution);
    auto tracks = tracks_;
    auto counters = counters_;
    auto const collect_diagnostics = config_.trackDiagnostics_;
    Kokkos::parallel_for(
        "c8_kokkos_precompute_radio_tracks",
        Kokkos::RangePolicy<TExecutionSpace>(execution, 0, count),
        KOKKOS_LAMBDA(std::size_t const index) {
          record_track<RadioAtomicOperations>(records(index), &tracks(index),
                                              counters.data(), collect_diagnostics);
        });
    statistics_.trackPrecomputeBatches_++;
    statistics_.trackPrecomputedRecords_ += count;
    statistics_.maximumTrackBatch_ = std::max(statistics_.maximumTrackBatch_, count);

    if (fuseCoreasZhs_) {
      project<true, true>(tracks, count, coreasObservers_, zhsObservers_,
                          coreasObservers_.extent(0), execution);
    } else {
      if (config_.coreasEnabled_)
        project<true, false>(tracks, count, coreasObservers_, {},
                             coreasObservers_.extent(0), execution);
      if (config_.zhsEnabled_)
        project<false, true>(tracks, count, {}, zhsObservers_, zhsObservers_.extent(0),
                             execution);
    }
    downloaded_ = false;
  }

  template <class TExecutionSpace>
  FixedRadioSnapshot RadioAccumulator<TExecutionSpace>::downloadFixed(
      std::string const& shower_identity, CooperativeEndpoint endpoint,
      TExecutionSpace const& execution) {
    if (!enabled() || !config_.deterministic_ || downloaded_ || fixedSnapshotSealed_)
      throw std::logic_error("fixed radio snapshot unavailable/already committed");
    flushHostLeptonTracks(execution);
    FixedRadioSnapshot result;
    result.showerIdentity_ = shower_identity;
    result.endpoint_ = endpoint;
    result.config_ = config_;
    if (shower_identity.empty() || static_cast<unsigned>(endpoint) > 1)
      throw std::invalid_argument("invalid cooperative radio snapshot identity");
    // Poison on a failed export: folding thread counters must never happen
    // twice if a transfer or allocation subsequently throws.
    fixedSnapshotSealed_ = true;
    if constexpr (IsOpenMP) mergeThreadWaveforms(execution);
    auto host_counters = Kokkos::create_mirror(counters_);
    auto host_coreas = Kokkos::create_mirror(coreasFixed_);
    auto host_zhs = Kokkos::create_mirror(zhsFixed_);
    try {
      Kokkos::deep_copy(execution, host_counters, counters_);
      if (coreasFixed_.extent(0)) Kokkos::deep_copy(execution, host_coreas, coreasFixed_);
      if (zhsFixed_.extent(0)) Kokkos::deep_copy(execution, host_zhs, zhsFixed_);
      execution.fence("download cooperative fixed radio endpoint");
    } catch (...) {
      // A later copy can fail with an earlier copy still in flight. Keep
      // these local host destinations alive while draining this endpoint.
      try {
        execution.fence("unwind cooperative radio endpoint copy");
      } catch (...) {}
      throw;
    }
    result.counters_ = host_counters(0);
    copyStatistics(result.counters_);
    result.coreas_.resize(host_coreas.extent(0));
    result.zhs_.resize(host_zhs.extent(0));
    for (std::size_t i = 0; i < result.coreas_.size(); ++i)
      result.coreas_[i] = host_coreas(i);
    for (std::size_t i = 0; i < result.zhs_.size(); ++i) result.zhs_[i] = host_zhs(i);
    validate_fixed_radio(result);
    statistics_.deviceToHostBytes_ +=
        sizeof(RadioCounters) +
        (result.coreas_.size() + result.zhs_.size()) * sizeof(long long);
    downloaded_ = true;
    return result;
  }

  template <class TExecutionSpace>
  RadioWaveforms RadioAccumulator<TExecutionSpace>::download(
      TExecutionSpace const& execution) {
    if (!enabled()) return {};
    if (downloaded_ || fixedSnapshotSealed_)
      throw std::logic_error("Kokkos radio waveforms downloaded twice");
    flushHostLeptonTracks(execution);
    // CUDA/HIP/SYCL parallel dispatch is asynchronous.  Keep projection
    // work queued until the waveform download instead of fencing every
    // lepton wavefront.  All kernels use the same execution-space instance,
    // so overwriting the reusable track workspace remains stream ordered.
    // The explicit fence in download() is the owning completion boundary.
    if constexpr (IsOpenMP) {
      mergeThreadWaveforms(execution);
    } else {
      (void)execution;
    }
    auto host_counters =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), counters_);
    copyStatistics(host_counters(0));
    if (statistics_.fixedPointOverflows_ != 0)
      throw std::runtime_error(
          "Kokkos radio fixed-point accumulator overflowed; increase "
          "fixed_point_field_limit_V_per_m");
    RadioWaveforms result{};
    result.coreas_ = downloadSet(config_.coreasObservers_, coreasBins_, coreasFixed_,
                                 coreasFloating_, false);
    result.zhs_ =
        downloadSet(config_.zhsObservers_, zhsBins_, zhsFixed_, zhsFloating_, true);
    statistics_.deviceToHostBytes_ += sizeof(RadioCounters);
    downloaded_ = true;
    return result;
  }

  template <class TExecutionSpace>
  RadioStatistics const& RadioAccumulator<TExecutionSpace>::statistics() const noexcept {
    return statistics_;
  }

  template <class TExecutionSpace>
  std::size_t RadioAccumulator<TExecutionSpace>::deviceBytes() const noexcept {
    return refractivity_.span() * sizeof(double) +
           integratedRefractivity_.span() * sizeof(double) +
           coreasObservers_.span() * sizeof(RadioObserverView) +
           zhsObservers_.span() * sizeof(RadioObserverView) +
           coreasFixed_.span() * sizeof(long long) +
           zhsFixed_.span() * sizeof(long long) +
           coreasFloating_.span() * sizeof(double) +
           zhsFloating_.span() * sizeof(double) +
           coreasThreadFixed_.span() * sizeof(long long) +
           zhsThreadFixed_.span() * sizeof(long long) +
           threadCounters_.span() * sizeof(RadioCounters) +
           hostTrackInputDevice_.span() *
               sizeof(::corsika::kokkos::LeptonTransportRecord) +
           tracks_.span() * sizeof(RadioTrackKinematics) +
           counters_.span() * sizeof(RadioCounters);
  }

  template <class TExecutionSpace>
  std::size_t RadioAccumulator<TExecutionSpace>::staticHostToDeviceBytes() const {
    using ::corsika::kokkos::checked_memory_add;
    using ::corsika::kokkos::checked_memory_multiply;
    auto bytes = checked_memory_multiply(refractivity_.span(), sizeof(double));
    bytes = checked_memory_add(
        bytes, checked_memory_multiply(integratedRefractivity_.span(), sizeof(double)));
    bytes = checked_memory_add(bytes, checked_memory_multiply(coreasObservers_.span(),
                                                              sizeof(RadioObserverView)));
    return checked_memory_add(
        bytes, checked_memory_multiply(zhsObservers_.span(), sizeof(RadioObserverView)));
  }

  template <class TExecutionSpace>
  void RadioAccumulator<TExecutionSpace>::reserveTrackCapacity(
      std::size_t const count, TExecutionSpace const& execution) {
    if (!enabled() || count == 0) return;
    if constexpr (!IsOpenMP) {
      if (count < DirectProjectionTrackThreshold) return;
    }
    ensureTrackCapacity(count, execution);
  }

  template <class TExecutionSpace>
  ::corsika::kokkos::MemoryProjection
  RadioAccumulator<TExecutionSpace>::projectedTrackCapacity(
      std::size_t const count) const {
    using ::corsika::kokkos::MemoryProjectionBuilder;
    auto const current = deviceBytes();
    if (!enabled() || count == 0) return {current, current};
    if constexpr (!IsOpenMP) {
      if (count < DirectProjectionTrackThreshold) return {current, current};
    }
    auto const old_capacity = tracks_.extent(0);
    if (count <= old_capacity) return {current, current};
    if (old_capacity > std::numeric_limits<std::size_t>::max() / 2)
      throw std::length_error("Kokkos radio track capacity overflow");
    auto const grown = old_capacity == 0 ? count : std::max(count, 2 * old_capacity);
    MemoryProjectionBuilder projection(current);
    projection.replace(
        tracks_.span() * sizeof(RadioTrackKinematics),
        ::corsika::kokkos::checked_memory_multiply(grown, sizeof(RadioTrackKinematics)));
    return projection.result();
  }

  template <class TExecutionSpace>
  template <class THostValue, class TView>
  void RadioAccumulator<TExecutionSpace>::upload(std::vector<THostValue> const& host,
                                                 TView& device,
                                                 TExecutionSpace const& execution,
                                                 char const* label) {
    device = TView(std::string(label), host.size());
    auto mirror = Kokkos::create_mirror_view(device);
    for (std::size_t index = 0; index < host.size(); ++index) mirror(index) = host[index];
    Kokkos::deep_copy(execution, device, mirror);
  }

  template <class TExecutionSpace>
  std::vector<RadioObserverView> RadioAccumulator<TExecutionSpace>::makeObservers(
      std::vector<RadioObserverSnapshot> const& input, std::size_t& total_bins,
      bool zhs_potential) const {
    std::vector<RadioObserverView> result;
    result.reserve(input.size());
    total_bins = 0;
    for (auto const& observer : input) {
      RadioObserverView device{};
      for (int axis = 0; axis < 3; ++axis)
        device.positionM_[axis] = observer.positionM_[axis];
      device.startTimeS_ = observer.startTimeS_;
      device.durationS_ = observer.durationS_;
      device.sampleRateHz_ = observer.sampleRateHz_;
      auto const potential_factor = zhs_potential ? observer.sampleRateHz_ : 1.;
      device.fixedPointScale_ =
          config_.deterministic_
              ? FixedPointHeadroom * potential_factor / config_.fixedPointFieldLimitVPerM_
              : 1.;
      device.inverseFixedPointScale_ = 1. / device.fixedPointScale_;
      device.numberOfBins_ = observer.numberOfBins_;
      device.waveformOffset_ = total_bins;
      if (observer.numberOfBins_ > std::numeric_limits<std::size_t>::max() - total_bins)
        throw std::overflow_error("Kokkos radio waveform bin overflow");
      total_bins += static_cast<std::size_t>(observer.numberOfBins_);
      result.push_back(device);
    }
    return result;
  }

  template <class TExecutionSpace>
  bool RadioAccumulator<TExecutionSpace>::observersCanShareTrackKinematics(
      std::vector<RadioObserverView> const& coreas,
      std::vector<RadioObserverView> const& zhs) {
    if (coreas.size() != zhs.size() || coreas.empty()) return false;
    for (std::size_t index = 0; index < coreas.size(); ++index) {
      auto const& lhs = coreas[index];
      auto const& rhs = zhs[index];
      if (lhs.startTimeS_ != rhs.startTimeS_ || lhs.durationS_ != rhs.durationS_ ||
          lhs.sampleRateHz_ != rhs.sampleRateHz_ ||
          lhs.numberOfBins_ != rhs.numberOfBins_)
        return false;
      for (int axis = 0; axis < 3; ++axis)
        if (lhs.positionM_[axis] != rhs.positionM_[axis]) return false;
    }
    return true;
  }

  template <class TExecutionSpace>
  void RadioAccumulator<TExecutionSpace>::allocateWaveforms(
      std::size_t bins, fixed_waveform_view_type& fixed,
      floating_waveform_view_type& floating, RadioWaveformView& view, char const* label) {
    if (bins == 0) return;
    if (bins > std::numeric_limits<std::size_t>::max() / 3)
      throw std::length_error("Kokkos radio waveform size overflow");
    auto const waveform_values = 3 * bins;
    if (config_.deterministic_) {
      fixed = fixed_waveform_view_type(std::string(label), waveform_values);
      view.fixedX_ = fixed.data();
      view.fixedY_ = fixed.data() + bins;
      view.fixedZ_ = fixed.data() + 2 * bins;
    } else {
      floating = floating_waveform_view_type(std::string(label), waveform_values);
      view.floatingX_ = floating.data();
      view.floatingY_ = floating.data() + bins;
      view.floatingZ_ = floating.data() + 2 * bins;
    }
  }

  template <class TExecutionSpace>
  void RadioAccumulator<TExecutionSpace>::allocateThreadWaveforms(
      std::size_t bins, thread_fixed_waveform_view_type& fixed, std::size_t thread_slots,
      TExecutionSpace const& execution, char const* label) {
    if (bins == 0) return;
    if (bins > std::numeric_limits<std::size_t>::max() / 3)
      throw std::length_error("Kokkos OpenMP private radio waveform size overflow");
    fixed = thread_fixed_waveform_view_type(
        Kokkos::view_alloc(execution, Kokkos::WithoutInitializing, std::string(label)),
        thread_slots, 3 * bins);
  }

  template <class TExecutionSpace>
  unsigned long long RadioAccumulator<TExecutionSpace>::mergeThreadWaveform(
      thread_fixed_waveform_view_type const& private_waveform,
      fixed_waveform_view_type const& waveform, TExecutionSpace const& execution,
      char const* label) {
    if (private_waveform.extent(0) == 0) return 0;
    if (private_waveform.extent(1) != waveform.extent(0))
      throw std::logic_error("Kokkos OpenMP private radio waveform extent mismatch");
    auto const thread_slots = private_waveform.extent(0);
    unsigned long long overflow_count{};
    Kokkos::parallel_reduce(
        label,
        Kokkos::RangePolicy<TExecutionSpace>(execution, 0, private_waveform.extent(1)),
        accumulator_detail::MergeThreadWaveformKernel<thread_fixed_waveform_view_type,
                                                      fixed_waveform_view_type>{
            private_waveform, waveform, thread_slots},
        overflow_count);
    return overflow_count;
  }

  template <class TExecutionSpace>
  void RadioAccumulator<TExecutionSpace>::mergeThreadWaveforms(
      TExecutionSpace const& execution) {
    auto overflow_count = mergeThreadWaveform(coreasThreadFixed_, coreasFixed_, execution,
                                              "c8_kokkos_merge_coreas_thread_waveforms");
    overflow_count += mergeThreadWaveform(zhsThreadFixed_, zhsFixed_, execution,
                                          "c8_kokkos_merge_zhs_thread_waveforms");
    // OpenMP memory space is host accessible.  The reductions above are the
    // completion boundary, so no projection kernel can still modify these
    // private counters while they are folded into the global diagnostics.
    auto& global = counters_(0);
    for (std::size_t slot = 0; slot < threadCounters_.extent(0); ++slot) {
      auto const& local = threadCounters_(slot);
      global.coreasContributions_ += local.coreasContributions_;
      global.zhsContributions_ += local.zhsContributions_;
      global.zhsSubtracks_ += local.zhsSubtracks_;
      global.fixedPointOverflows_ += local.fixedPointOverflows_;
    }
    global.fixedPointOverflows_ += overflow_count;
  }

  template <class TExecutionSpace>
  void RadioAccumulator<TExecutionSpace>::ensureTrackCapacity(
      std::size_t count, TExecutionSpace const& execution) {
    if (tracks_.extent(0) >= count) return;
    if (tracks_.extent(0) > std::numeric_limits<std::size_t>::max() / 2)
      throw std::length_error("Kokkos radio track capacity overflow");
    auto capacity =
        tracks_.extent(0) == 0 ? count : std::max(count, 2 * tracks_.extent(0));
    // A prior asynchronous projection can still reference the old
    // allocation.  Capacity growth is rare; fence only this ownership
    // transition rather than every radio batch.
    if (tracks_.extent(0) != 0) execution.fence("grow Kokkos radio track workspace");
    tracks_ = track_view_type(Kokkos::view_alloc(execution, Kokkos::WithoutInitializing,
                                                 "c8_kokkos_radio_track_kinematics"),
                              capacity);
    statistics_.trackWorkspaceBytes_ = capacity * sizeof(RadioTrackKinematics);
    statistics_.deviceBytes_ = deviceBytes();
  }

  template <class TExecutionSpace>
  template <bool Coreas, bool Zhs>
  void RadioAccumulator<TExecutionSpace>::projectDirect(
      track_input_view_type const& records, std::size_t track_count,
      observer_view_type const& coreas_observers, observer_view_type const& zhs_observers,
      std::size_t observer_count, TExecutionSpace const& execution) {
    if (observer_count == 0) return;
    if (track_count > std::numeric_limits<std::size_t>::max() / observer_count)
      throw std::overflow_error("Kokkos direct radio pair count overflow");
    auto const pairs = track_count * observer_count;
    auto const propagation = propagation_;
    auto const coreas_waveforms = coreasWaveforms_;
    auto const zhs_waveforms = zhsWaveforms_;
    auto counters = counters_;
    Kokkos::parallel_for(
        Coreas && Zhs ? "c8_kokkos_coreas_zhs_direct"
                      : Coreas ? "c8_kokkos_coreas_direct" : "c8_kokkos_zhs_direct",
        Kokkos::RangePolicy<TExecutionSpace>(execution, 0, pairs),
        KOKKOS_LAMBDA(std::size_t const pair_index) {
          auto const track_index = pair_index / observer_count;
          auto const observer_index = pair_index % observer_count;
          RadioTrackKinematics track{};
          if (!make_radio_track_kinematics(records(track_index), track)) return;
          if (Coreas)
            accumulate_co_reas<RadioAtomicOperations>(track, propagation,
                                                      coreas_observers(observer_index),
                                                      coreas_waveforms, counters.data());
          if (Zhs)
            accumulate_zhs<RadioAtomicOperations>(track, propagation,
                                                  zhs_observers(observer_index),
                                                  zhs_waveforms, counters.data());
        });
    statistics_.trackObserverPairs_ += pairs * static_cast<std::size_t>(Coreas + Zhs);
    if constexpr (Coreas && Zhs) statistics_.fusedTrackObserverPairs_ += pairs;
  }

  template <class TExecutionSpace>
  template <bool Coreas, bool Zhs>
  void RadioAccumulator<TExecutionSpace>::project(
      track_view_type const& tracks, std::size_t track_count,
      observer_view_type const& coreas_observers, observer_view_type const& zhs_observers,
      std::size_t observer_count, TExecutionSpace const& execution) {
    if (observer_count == 0) return;
    auto const track_tiles = (track_count + trackTileSize_ - 1) / trackTileSize_;
    auto const observer_tiles =
        (observer_count + observerTileSize_ - 1) / observerTileSize_;
    auto const league_size = track_tiles * observer_tiles;
    auto const propagation = propagation_;
    auto const coreas_waveforms = coreasWaveforms_;
    auto const zhs_waveforms = zhsWaveforms_;
    auto counters = counters_;
    auto const track_tile_size = trackTileSize_;
    auto const observer_tile_size = observerTileSize_;
    using policy_type = Kokkos::TeamPolicy<TExecutionSpace>;
    using member_type = typename policy_type::member_type;
    if constexpr (IsOpenMP) {
      // A Kokkos OpenMP team consumes threads from the process-wide OpenMP
      // pool.  Requiring TrackTileSize * ObserverTileSize members would make
      // the kernel invalid whenever the user selects fewer than 16 threads.
      // Keep one OpenMP worker per tile and evaluate the small 4x4 tile
      // serially inside that worker.  Different tiles still run in parallel,
      // while the GPU execution spaces retain one lane per track/observer
      // pair below.
      auto const coreas_thread_fixed = coreasThreadFixed_;
      auto const zhs_thread_fixed = zhsThreadFixed_;
      auto const thread_counters = threadCounters_;
      auto const coreas_bins = coreasBins_;
      auto const zhs_bins = zhsBins_;
      Kokkos::Experimental::UniqueToken<TExecutionSpace> thread_tokens(execution);
      Kokkos::parallel_for(
          Coreas && Zhs ? "c8_kokkos_coreas_zhs_tiled"
                        : Coreas ? "c8_kokkos_coreas_tiled" : "c8_kokkos_zhs_tiled",
          policy_type(execution, league_size, 1), KOKKOS_LAMBDA(member_type const& team) {
            auto const thread_slot = thread_tokens.acquire();
            auto thread_coreas_waveforms = coreas_waveforms;
            auto thread_zhs_waveforms = zhs_waveforms;
            if (coreas_thread_fixed.extent(0) != 0) {
              auto* const base = &coreas_thread_fixed(thread_slot, 0);
              thread_coreas_waveforms.fixedX_ = base;
              thread_coreas_waveforms.fixedY_ = base + coreas_bins;
              thread_coreas_waveforms.fixedZ_ = base + 2 * coreas_bins;
            }
            if (zhs_thread_fixed.extent(0) != 0) {
              auto* const base = &zhs_thread_fixed(thread_slot, 0);
              thread_zhs_waveforms.fixedX_ = base;
              thread_zhs_waveforms.fixedY_ = base + zhs_bins;
              thread_zhs_waveforms.fixedZ_ = base + 2 * zhs_bins;
            }
            auto* const thread_counter = thread_counters.extent(0) == 0
                                             ? counters.data()
                                             : &thread_counters(thread_slot);
            auto const track_tile =
                static_cast<std::size_t>(team.league_rank()) / observer_tiles;
            auto const observer_tile =
                static_cast<std::size_t>(team.league_rank()) % observer_tiles;
            for (std::size_t track_lane = 0; track_lane < track_tile_size; ++track_lane) {
              auto const track_index = track_tile * track_tile_size + track_lane;
              if (track_index >= track_count || tracks(track_index).valid_ == 0) continue;
              for (std::size_t observer_lane = 0; observer_lane < observer_tile_size;
                   ++observer_lane) {
                auto const observer_index =
                    observer_tile * observer_tile_size + observer_lane;
                if (observer_index >= observer_count) continue;
                if (thread_counters.extent(0) != 0) {
                  if (Coreas)
                    accumulate_co_reas<
                        accumulator_detail::RadioOpenMpThreadLocalOperations>(
                        tracks(track_index), propagation,
                        coreas_observers(observer_index), thread_coreas_waveforms,
                        thread_counter);
                  if (Zhs)
                    accumulate_zhs<accumulator_detail::RadioOpenMpThreadLocalOperations>(
                        tracks(track_index), propagation, zhs_observers(observer_index),
                        thread_zhs_waveforms, thread_counter);
                } else {
                  if (Coreas)
                    accumulate_co_reas<RadioAtomicOperations>(
                        tracks(track_index), propagation,
                        coreas_observers(observer_index), coreas_waveforms,
                        counters.data());
                  if (Zhs)
                    accumulate_zhs<RadioAtomicOperations>(
                        tracks(track_index), propagation, zhs_observers(observer_index),
                        zhs_waveforms, counters.data());
                }
              }
            }
            thread_tokens.release(thread_slot);
          });
    } else {
      auto const tile_lanes = track_tile_size * observer_tile_size;
      auto const team_size = std::min(teamSize_, tile_lanes);
      auto policy = policy_type(execution, league_size, team_size);
      policy.set_scratch_size(0, Kokkos::PerTeam(sizeof(RadioTileScratch)));
      auto const label = Coreas && Zhs
                             ? "c8_kokkos_coreas_zhs_tiled"
                             : Coreas ? "c8_kokkos_coreas_tiled" : "c8_kokkos_zhs_tiled";
      // Sparse observer tiles and CoREAS-only work keep the original kernel:
      // their counters do not repay the extra team-reduction overhead.
      // Small track batches already use the unchanged projectDirect() path.
      if (Zhs && observer_count >= observer_tile_size) {
        Kokkos::parallel_for(
            label, policy,
            RadioTileProjection<Coreas, Zhs, Zhs>{
                tracks, coreas_observers, zhs_observers, propagation, coreas_waveforms,
                zhs_waveforms, counters, track_count, observer_count, observer_tiles,
                track_tile_size, observer_tile_size, tile_lanes});
      } else {
        Kokkos::parallel_for(
            label, policy,
            RadioTileProjection<Coreas, Zhs>{
                tracks, coreas_observers, zhs_observers, propagation, coreas_waveforms,
                zhs_waveforms, counters, track_count, observer_count, observer_tiles,
                track_tile_size, observer_tile_size, tile_lanes});
      }
    }
    auto const pairs = track_count * observer_count;
    statistics_.trackObserverPairs_ += pairs * static_cast<std::size_t>(Coreas + Zhs);
    if constexpr (Coreas && Zhs) statistics_.fusedTrackObserverPairs_ += pairs;
    statistics_.projectionTiles_ += league_size;
  }

  template <class TExecutionSpace>
  template <class TFixedView, class TFloatView>
  std::vector<RadioWaveform> RadioAccumulator<TExecutionSpace>::downloadSet(
      std::vector<RadioObserverSnapshot> const& observers, std::size_t total_bins,
      TFixedView const& fixed, TFloatView const& floating, bool zhs_potential) {
    std::vector<RadioWaveform> result;
    result.reserve(observers.size());
    if (total_bins == 0) return result;
    if (total_bins > std::numeric_limits<std::size_t>::max() / 3)
      throw std::length_error("Kokkos radio download size overflow");
    std::vector<double> host(3 * total_bins);
    if (fixed.extent(0) != 0) {
      auto mirror = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), fixed);
      std::size_t offset = 0;
      for (auto const& observer : observers) {
        auto const count = static_cast<std::size_t>(observer.numberOfBins_);
        auto const potential_factor = zhs_potential ? observer.sampleRateHz_ : 1.;
        auto const inverse_scale =
            config_.fixedPointFieldLimitVPerM_ / (FixedPointHeadroom * potential_factor);
        for (std::size_t bin = 0; bin < count; ++bin) {
          auto const local = offset + bin;
          host[local] = static_cast<double>(mirror(local)) * inverse_scale;
          host[total_bins + local] =
              static_cast<double>(mirror(total_bins + local)) * inverse_scale;
          host[2 * total_bins + local] =
              static_cast<double>(mirror(2 * total_bins + local)) * inverse_scale;
        }
        offset += count;
      }
      statistics_.deviceToHostBytes_ += fixed.extent(0) * sizeof(long long);
    } else {
      auto mirror = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), floating);
      for (std::size_t index = 0; index < host.size(); ++index)
        host[index] = mirror(index);
      statistics_.deviceToHostBytes_ += floating.extent(0) * sizeof(double);
    }
    std::size_t offset = 0;
    for (auto const& observer : observers) {
      auto const count = static_cast<std::size_t>(observer.numberOfBins_);
      RadioWaveform waveform{};
      waveform.x_.assign(host.begin() + offset, host.begin() + offset + count);
      waveform.y_.assign(host.begin() + total_bins + offset,
                         host.begin() + total_bins + offset + count);
      waveform.z_.assign(host.begin() + 2 * total_bins + offset,
                         host.begin() + 2 * total_bins + offset + count);
      result.push_back(std::move(waveform));
      offset += count;
    }
    return result;
  }

  template <class TExecutionSpace>
  void RadioAccumulator<TExecutionSpace>::copyStatistics(RadioCounters const& counters) {
    statistics_.leptonTracks_ = counters.validTracks_;
    statistics_.coreasContributions_ = counters.coreasContributions_;
    statistics_.zhsContributions_ = counters.zhsContributions_;
    statistics_.zhsSubtracks_ = counters.zhsSubtracks_;
    statistics_.fixedPointOverflows_ = counters.fixedPointOverflows_;
    statistics_.weightedSegmentCount_ = counters.weightedSegmentCount_;
    statistics_.trackLengthM_ = counters.trackLengthM_;
    statistics_.weightedTrackLengthM_ = counters.weightedTrackLengthM_;
    statistics_.electronWeightedTrackLengthM_ = counters.electronWeightedTrackLengthM_;
    statistics_.positronWeightedTrackLengthM_ = counters.positronWeightedTrackLengthM_;
    statistics_.signedChargeWeightedTrackLengthM_ =
        counters.signedChargeWeightedTrackLengthM_;
    statistics_.energyWeightedTrackLengthGeVM_ = counters.energyWeightedTrackLengthGeVM_;
    statistics_.maximumSegmentLengthM_ = counters.maximumSegmentLengthM_;
    statistics_.weightedDirectionChangeRad_ = counters.weightedDirectionChangeRad_;
    statistics_.weightedDirectionChangeSquaredRad2_ =
        counters.weightedDirectionChangeSquaredRad2_;
    statistics_.weightedBetaDeficitTrackLengthM_ =
        counters.weightedBetaDeficitTrackLengthM_;
    statistics_.weightedTimeResidualS_ = counters.weightedTimeResidualS_;
    statistics_.maximumDirectionChangeRad_ = counters.maximumDirectionChangeRad_;
    for (std::size_t axis = 0; axis < 3; ++axis)
      statistics_.signedChargeWeightedDirectionChange_[axis] =
          counters.signedChargeWeightedDirectionChange_[axis];
    for (std::size_t index = 0; index < 15; ++index)
      statistics_.weightedTrackLengthByKineticEnergyM_[index] =
          counters.weightedTrackLengthByKineticEnergyM_[index];
  }

  template <class TExecutionSpace>
  void RadioAccumulator<TExecutionSpace>::validate(RadioConfig const& config) {
    auto const& propagation = config.propagation_;
    if (!config.coreasEnabled_ && !config.zhsEnabled_)
      throw std::invalid_argument("enabled Kokkos radio has no algorithm");
    if (!std::isfinite(config.fixedPointFieldLimitVPerM_) ||
        !(config.fixedPointFieldLimitVPerM_ > 0.))
      throw std::invalid_argument("invalid Kokkos radio fixed-point limit");
    if (config.zhsSubtrackRefinement_ < 1 || config.zhsSubtrackRefinement_ > 64)
      throw std::invalid_argument("Kokkos ZHS subtrack refinement must be in [1,64]");
    if (!std::isfinite(propagation.homogeneousRefractiveIndex_) ||
        propagation.homogeneousRefractiveIndex_ < 0.)
      throw std::invalid_argument("invalid Kokkos homogeneous radio refractive index");
    bool const homogeneous = propagation.homogeneousRefractiveIndex_ > 0.;
    if (!homogeneous &&
        (propagation.refractivity_.size() < 11 ||
         propagation.refractivity_.size() != propagation.integratedRefractivity_.size() ||
         !std::isfinite(propagation.minimumHeightM_) ||
         !std::isfinite(propagation.maximumHeightM_) ||
         !(propagation.maximumHeightM_ > propagation.minimumHeightM_) ||
         !std::isfinite(propagation.stepM_) || !(propagation.stepM_ > 0.) ||
         !std::isfinite(propagation.inverseStepPerM_) ||
         !(propagation.inverseStepPerM_ > 0.)))
      throw std::invalid_argument("invalid Kokkos flat-atmosphere radio snapshot");
    auto validate_observers = [&](std::vector<RadioObserverSnapshot> const& observers) {
      for (auto const& observer : observers) {
        if (!std::isfinite(observer.startTimeS_) || !std::isfinite(observer.durationS_) ||
            !(observer.durationS_ >= 0.) || !std::isfinite(observer.sampleRateHz_) ||
            !(observer.sampleRateHz_ > 0.) || observer.numberOfBins_ == 0)
          throw std::invalid_argument("invalid Kokkos radio observer");
        auto const height = (observer.positionM_[2] - propagation.minimumHeightM_) *
                            propagation.inverseStepPerM_;
        if (!homogeneous &&
            (!std::isfinite(height) || height < 0. ||
             height + 0.5 >= static_cast<double>(propagation.refractivity_.size())))
          throw std::invalid_argument("Kokkos radio observer outside propagation table");
        for (double coordinate : observer.positionM_)
          if (!std::isfinite(coordinate))
            throw std::invalid_argument("Kokkos radio observer position is not finite");
      }
    };
    validate_observers(config.coreasObservers_);
    validate_observers(config.zhsObservers_);
  }
} // namespace corsika::kokkos::radio

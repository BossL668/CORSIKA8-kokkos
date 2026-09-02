/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <corsika/accelerator/radio/detail/RadioProjectionStep.hpp>
#include <corsika/accelerator/em/KokkosTuningCache.hpp>
#include <corsika/gpu/radio/Types.hpp>

namespace corsika::accelerator::radio::kokkos_detail {

  struct KokkosRadioAtomicOperations {
    KOKKOS_INLINE_FUNCTION static long long add(long long* address,
                                                long long value) {
      return Kokkos::atomic_fetch_add(address, value);
    }

    KOKKOS_INLINE_FUNCTION static unsigned long long add(
        unsigned long long* address, unsigned long long value) {
      return Kokkos::atomic_fetch_add(address, value);
    }

    KOKKOS_INLINE_FUNCTION static double add(double* address, double value) {
      return Kokkos::atomic_fetch_add(address, value);
    }

    KOKKOS_INLINE_FUNCTION static void maximum(double* address, double value) {
      Kokkos::atomic_max(address, value);
    }
  };

  template <class ExecutionSpace>
  class KokkosRadioAccumulator {
  public:
    using memory_space = typename ExecutionSpace::memory_space;
    using TrackInputView =
        Kokkos::View<gpu::em::LeptonTransportRecord*, memory_space>;
    using TrackView = Kokkos::View<detail::RadioTrackKinematics*, memory_space>;
    using ObserverView = Kokkos::View<detail::DeviceObserver*, memory_space>;
    using FixedWaveformView = Kokkos::View<long long*, memory_space>;
    using FloatingWaveformView = Kokkos::View<double*, memory_space>;
    using CounterView =
        Kokkos::View<detail::DeviceRadioCounters*, memory_space>;

#if defined(CORSIKA8_KOKKOS_BACKEND_OPENMP)
    static constexpr std::size_t TrackTileSize = 4;
    static constexpr std::size_t ObserverTileSize = 4;
#else
    static constexpr std::size_t TrackTileSize = 8;
    static constexpr std::size_t ObserverTileSize = 32;
#endif

    void initialize(
                    gpu::radio::GpuRadioConfig const& requested,
                    corsika::accelerator::em::KokkosTuningParameters const&
                        tuning = {},
                    ExecutionSpace const& execution = {}) {
      if (initialized_)
        throw std::logic_error("Kokkos radio accumulator already initialized");
      config_ = requested;
      if (tuning.track_tile_size != 0)
        track_tile_size_ = tuning.track_tile_size;
      if (tuning.observer_tile_size != 0)
        observer_tile_size_ = tuning.observer_tile_size;
      if (tuning.team_size != 0) team_size_ = tuning.team_size;
      if (track_tile_size_ == 0 || observer_tile_size_ == 0 ||
          team_size_ == 0 || team_size_ > 1024)
        throw std::invalid_argument("Kokkos radio tuning is invalid");
      if (!requested.enabled) return;
      validate(requested);
      coreas_observers_host_ = makeObservers(
          requested.coreas_observers, coreas_bins_, false);
      zhs_observers_host_ = makeObservers(
          requested.zhs_observers, zhs_bins_, true);
      upload(coreas_observers_host_, coreas_observers_, execution,
             "c8_kokkos_coreas_observers");
      upload(zhs_observers_host_, zhs_observers_, execution,
             "c8_kokkos_zhs_observers");
      upload(requested.propagation.refractivity, refractivity_, execution,
             "c8_kokkos_radio_refractivity");
      upload(requested.propagation.integrated_refractivity,
             integrated_refractivity_, execution,
             "c8_kokkos_radio_integrated_refractivity");
      propagation_.minimum_height_m =
          requested.propagation.minimum_height_m;
      propagation_.maximum_height_m =
          requested.propagation.maximum_height_m;
      propagation_.step_m = requested.propagation.step_m;
      propagation_.inverse_step_per_m =
          requested.propagation.inverse_step_per_m;
      propagation_.slope_refractivity_lower =
          requested.propagation.slope_refractivity_lower;
      propagation_.slope_integrated_refractivity_lower =
          requested.propagation.slope_integrated_refractivity_lower;
      propagation_.slope_refractivity_upper =
          requested.propagation.slope_refractivity_upper;
      propagation_.slope_integrated_refractivity_upper =
          requested.propagation.slope_integrated_refractivity_upper;
      propagation_.refractivity = refractivity_.data();
      propagation_.integrated_refractivity =
          integrated_refractivity_.data();
      propagation_.table_size = refractivity_.extent(0);
      propagation_.zhs_subtrack_refinement =
          requested.zhs_subtrack_refinement;
      allocateWaveforms(coreas_bins_, coreas_fixed_, coreas_floating_,
                        coreas_waveforms_, "c8_kokkos_coreas_waveforms");
      allocateWaveforms(zhs_bins_, zhs_fixed_, zhs_floating_, zhs_waveforms_,
                        "c8_kokkos_zhs_waveforms");
      counters_ = CounterView("c8_kokkos_radio_counters", 1);
      fuse_coreas_zhs_ =
          requested.coreas_enabled && requested.zhs_enabled &&
          observersCanShareTrackKinematics(coreas_observers_host_,
                                           zhs_observers_host_);
      initialized_ = true;
      reset(execution);
    }

    bool enabled() const noexcept { return initialized_ && config_.enabled; }

    void reset(ExecutionSpace const& execution = {}) {
      if (!enabled()) return;
      if (coreas_fixed_.extent(0)) Kokkos::deep_copy(execution, coreas_fixed_, 0LL);
      if (zhs_fixed_.extent(0)) Kokkos::deep_copy(execution, zhs_fixed_, 0LL);
      if (coreas_floating_.extent(0))
        Kokkos::deep_copy(execution, coreas_floating_, 0.);
      if (zhs_floating_.extent(0))
        Kokkos::deep_copy(execution, zhs_floating_, 0.);
      Kokkos::deep_copy(execution, counters_, detail::DeviceRadioCounters{});
      execution.fence("reset Kokkos radio accumulator");
      statistics_ = {};
      statistics_.device_bytes = deviceBytes();
      statistics_.track_precompute_enabled = true;
      statistics_.track_tile_size = track_tile_size_;
      statistics_.observer_tile_size = observer_tile_size_;
      statistics_.track_diagnostics_enabled = config_.track_diagnostics;
      downloaded_ = false;
    }

    void accumulateLeptonTracks(
        TrackInputView const& records, std::size_t const count,
        ExecutionSpace const& execution = {}) {
      if (!enabled() || count == 0) return;
      if (count > records.extent(0))
        throw std::out_of_range("Kokkos radio track count exceeds input view");
      ensureTrackCapacity(count);
      auto tracks = tracks_;
      auto counters = counters_;
      auto const collect_diagnostics = config_.track_diagnostics;
      Kokkos::parallel_for(
          "c8_kokkos_precompute_radio_tracks",
          Kokkos::RangePolicy<ExecutionSpace>(execution, 0, count),
          KOKKOS_LAMBDA(std::size_t const index) {
            detail::recordTrack<KokkosRadioAtomicOperations>(
                records(index), &tracks(index), counters.data(),
                collect_diagnostics);
          });
      statistics_.track_precompute_batches++;
      statistics_.track_precomputed_records += count;
      statistics_.maximum_track_batch =
          std::max(statistics_.maximum_track_batch, count);

      if (fuse_coreas_zhs_) {
        project<true, true>(tracks, count, coreas_observers_,
                            zhs_observers_, coreas_observers_.extent(0),
                            execution);
      } else {
        if (config_.coreas_enabled)
          project<true, false>(tracks, count, coreas_observers_, {},
                               coreas_observers_.extent(0), execution);
        if (config_.zhs_enabled)
          project<false, true>(tracks, count, {}, zhs_observers_,
                               zhs_observers_.extent(0), execution);
      }
      downloaded_ = false;
    }

    gpu::radio::GpuRadioWaveforms download(
        ExecutionSpace const& execution = {}) {
      if (!enabled()) return {};
      if (downloaded_)
        throw std::logic_error("Kokkos radio waveforms downloaded twice");
      execution.fence("finish Kokkos radio projection");
      auto host_counters = Kokkos::create_mirror_view_and_copy(
          Kokkos::HostSpace(), counters_);
      copyStatistics(host_counters(0));
      if (statistics_.fixed_point_overflows != 0)
        throw std::runtime_error(
            "Kokkos radio fixed-point accumulator overflowed; increase "
            "fixed_point_field_limit_V_per_m");
      gpu::radio::GpuRadioWaveforms result{};
      result.coreas = downloadSet(
          config_.coreas_observers, coreas_bins_, coreas_fixed_,
          coreas_floating_, false);
      result.zhs = downloadSet(config_.zhs_observers, zhs_bins_, zhs_fixed_,
                               zhs_floating_, true);
      statistics_.device_to_host_bytes += sizeof(detail::DeviceRadioCounters);
      downloaded_ = true;
      return result;
    }

    gpu::radio::GpuRadioStatistics const& statistics() const noexcept {
      return statistics_;
    }

    std::size_t deviceBytes() const noexcept {
      return refractivity_.span() * sizeof(double) +
             integrated_refractivity_.span() * sizeof(double) +
             coreas_observers_.span() * sizeof(detail::DeviceObserver) +
             zhs_observers_.span() * sizeof(detail::DeviceObserver) +
             coreas_fixed_.span() * sizeof(long long) +
             zhs_fixed_.span() * sizeof(long long) +
             coreas_floating_.span() * sizeof(double) +
             zhs_floating_.span() * sizeof(double) +
             tracks_.span() * sizeof(detail::RadioTrackKinematics) +
             counters_.span() * sizeof(detail::DeviceRadioCounters);
    }

  private:
    template <class HostValue, class View>
    static void upload(std::vector<HostValue> const& host, View& device,
                       ExecutionSpace const& execution, char const* label) {
      device = View(std::string(label), host.size());
      auto mirror = Kokkos::create_mirror_view(device);
      for (std::size_t index = 0; index < host.size(); ++index)
        mirror(index) = host[index];
      Kokkos::deep_copy(execution, device, mirror);
    }

    std::vector<detail::DeviceObserver> makeObservers(
        std::vector<gpu::radio::RadioObserverSnapshot> const& input,
        std::size_t& total_bins, bool zhs_potential) const {
      std::vector<detail::DeviceObserver> result;
      result.reserve(input.size());
      total_bins = 0;
      for (auto const& observer : input) {
        detail::DeviceObserver device{};
        for (int axis = 0; axis < 3; ++axis)
          device.position_m[axis] = observer.position_m[axis];
        device.start_time_s = observer.start_time_s;
        device.duration_s = observer.duration_s;
        device.sample_rate_Hz = observer.sample_rate_Hz;
        auto const potential_factor =
            zhs_potential ? observer.sample_rate_Hz : 1.;
        device.fixed_point_scale =
            config_.deterministic
                ? detail::FixedPointHeadroom * potential_factor /
                      config_.fixed_point_field_limit_V_per_m
                : 1.;
        device.inverse_fixed_point_scale = 1. / device.fixed_point_scale;
        device.number_of_bins = observer.number_of_bins;
        device.waveform_offset = total_bins;
        if (observer.number_of_bins >
            std::numeric_limits<std::size_t>::max() - total_bins)
          throw std::overflow_error("Kokkos radio waveform bin overflow");
        total_bins += static_cast<std::size_t>(observer.number_of_bins);
        result.push_back(device);
      }
      return result;
    }

    static bool observersCanShareTrackKinematics(
        std::vector<detail::DeviceObserver> const& coreas,
        std::vector<detail::DeviceObserver> const& zhs) {
      if (coreas.size() != zhs.size() || coreas.empty()) return false;
      for (std::size_t index = 0; index < coreas.size(); ++index) {
        auto const& lhs = coreas[index];
        auto const& rhs = zhs[index];
        if (lhs.start_time_s != rhs.start_time_s ||
            lhs.duration_s != rhs.duration_s ||
            lhs.sample_rate_Hz != rhs.sample_rate_Hz ||
            lhs.number_of_bins != rhs.number_of_bins)
          return false;
        for (int axis = 0; axis < 3; ++axis)
          if (lhs.position_m[axis] != rhs.position_m[axis]) return false;
      }
      return true;
    }

    void allocateWaveforms(std::size_t bins, FixedWaveformView& fixed,
                           FloatingWaveformView& floating,
                           detail::DeviceWaveforms& view, char const* label) {
      if (bins == 0) return;
      if (config_.deterministic) {
        fixed = FixedWaveformView(std::string(label), 3 * bins);
        view.fixed_x = fixed.data();
        view.fixed_y = fixed.data() + bins;
        view.fixed_z = fixed.data() + 2 * bins;
      } else {
        floating = FloatingWaveformView(std::string(label), 3 * bins);
        view.floating_x = floating.data();
        view.floating_y = floating.data() + bins;
        view.floating_z = floating.data() + 2 * bins;
      }
    }

    void ensureTrackCapacity(std::size_t count) {
      if (tracks_.extent(0) >= count) return;
      auto capacity = tracks_.extent(0) == 0
                          ? count
                          : std::max(count, 2 * tracks_.extent(0));
      tracks_ = TrackView("c8_kokkos_radio_track_kinematics", capacity);
      statistics_.track_workspace_bytes =
          capacity * sizeof(detail::RadioTrackKinematics);
      statistics_.device_bytes = deviceBytes();
    }

  public:
    // Public only to satisfy NVCC's extended-lambda access restriction.  The
    // enclosing class remains in an internal detail namespace and this
    // function is not part of the backend interface.
    template <bool Coreas, bool Zhs>
    void project(TrackView const& tracks, std::size_t track_count,
                 ObserverView const& coreas_observers,
                 ObserverView const& zhs_observers,
                 std::size_t observer_count,
                 ExecutionSpace const& execution) {
      if (observer_count == 0) return;
      auto const track_tiles =
          (track_count + track_tile_size_ - 1) / track_tile_size_;
      auto const observer_tiles =
          (observer_count + observer_tile_size_ - 1) /
          observer_tile_size_;
      auto const league_size = track_tiles * observer_tiles;
      auto const propagation = propagation_;
      auto const coreas_waveforms = coreas_waveforms_;
      auto const zhs_waveforms = zhs_waveforms_;
      auto counters = counters_;
      auto const track_tile_size = track_tile_size_;
      auto const observer_tile_size = observer_tile_size_;
      using Policy = Kokkos::TeamPolicy<ExecutionSpace>;
      using Member = typename Policy::member_type;
#if defined(CORSIKA8_KOKKOS_BACKEND_OPENMP)
      // A Kokkos OpenMP team consumes threads from the process-wide OpenMP
      // pool.  Requiring TrackTileSize * ObserverTileSize members would make
      // the kernel invalid whenever the user selects fewer than 16 threads.
      // Keep one OpenMP worker per tile and evaluate the small 4x4 tile
      // serially inside that worker.  Different tiles still run in parallel,
      // while the GPU execution spaces retain one lane per track/observer
      // pair below.
      Kokkos::parallel_for(
          Coreas && Zhs ? "c8_kokkos_coreas_zhs_tiled"
                        : Coreas ? "c8_kokkos_coreas_tiled"
                                 : "c8_kokkos_zhs_tiled",
          Policy(execution, league_size, 1),
          KOKKOS_LAMBDA(Member const& team) {
            auto const track_tile =
                static_cast<std::size_t>(team.league_rank()) /
                observer_tiles;
            auto const observer_tile =
                static_cast<std::size_t>(team.league_rank()) %
                observer_tiles;
            for (std::size_t track_lane = 0;
                 track_lane < track_tile_size;
                 ++track_lane) {
              auto const track_index =
                  track_tile * track_tile_size + track_lane;
              if (track_index >= track_count ||
                  tracks(track_index).valid == 0)
                continue;
              for (std::size_t observer_lane = 0;
                   observer_lane < observer_tile_size; ++observer_lane) {
                auto const observer_index =
                    observer_tile * observer_tile_size + observer_lane;
                if (observer_index >= observer_count) continue;
                if (Coreas)
                  detail::accumulateCoREAS<KokkosRadioAtomicOperations>(
                      tracks(track_index), propagation,
                      coreas_observers(observer_index), coreas_waveforms,
                      counters.data());
                if (Zhs)
                  detail::accumulateZHS<KokkosRadioAtomicOperations>(
                      tracks(track_index), propagation,
                      zhs_observers(observer_index), zhs_waveforms,
                      counters.data());
              }
            }
          });
#else
      auto const tile_lanes = track_tile_size * observer_tile_size;
      auto const team_size = std::min(team_size_, tile_lanes);
      Kokkos::parallel_for(
          Coreas && Zhs ? "c8_kokkos_coreas_zhs_tiled"
                        : Coreas ? "c8_kokkos_coreas_tiled"
                                 : "c8_kokkos_zhs_tiled",
          Policy(execution, league_size, team_size),
          KOKKOS_LAMBDA(Member const& team) {
            auto const track_tile =
                static_cast<std::size_t>(team.league_rank()) /
                observer_tiles;
            auto const observer_tile =
                static_cast<std::size_t>(team.league_rank()) %
                observer_tiles;
            for (std::size_t lane = team.team_rank(); lane < tile_lanes;
                 lane += team.team_size()) {
              auto const track_lane = lane / observer_tile_size;
              auto const observer_lane = lane % observer_tile_size;
              auto const track_index =
                  track_tile * track_tile_size + track_lane;
              auto const observer_index =
                  observer_tile * observer_tile_size + observer_lane;
              if (track_index >= track_count ||
                  observer_index >= observer_count ||
                  tracks(track_index).valid == 0)
                continue;
              if (Coreas)
                detail::accumulateCoREAS<KokkosRadioAtomicOperations>(
                    tracks(track_index), propagation,
                    coreas_observers(observer_index), coreas_waveforms,
                    counters.data());
              if (Zhs)
                detail::accumulateZHS<KokkosRadioAtomicOperations>(
                    tracks(track_index), propagation,
                    zhs_observers(observer_index), zhs_waveforms,
                    counters.data());
            }
          });
#endif
      auto const pairs = track_count * observer_count;
      statistics_.track_observer_pairs +=
          pairs * static_cast<std::size_t>(Coreas + Zhs);
      if constexpr (Coreas && Zhs)
        statistics_.fused_track_observer_pairs += pairs;
      statistics_.projection_tiles += league_size;
    }

  private:
    template <class FixedView, class FloatView>
    std::vector<gpu::radio::RadioWaveform> downloadSet(
        std::vector<gpu::radio::RadioObserverSnapshot> const& observers,
        std::size_t total_bins, FixedView const& fixed,
        FloatView const& floating, bool zhs_potential) {
      std::vector<gpu::radio::RadioWaveform> result;
      result.reserve(observers.size());
      if (total_bins == 0) return result;
      std::vector<double> host(3 * total_bins);
      if (fixed.extent(0) != 0) {
        auto mirror = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),
                                                          fixed);
        std::size_t offset = 0;
        for (auto const& observer : observers) {
          auto const count =
              static_cast<std::size_t>(observer.number_of_bins);
          auto const potential_factor =
              zhs_potential ? observer.sample_rate_Hz : 1.;
          auto const inverse_scale =
              config_.fixed_point_field_limit_V_per_m /
              (detail::FixedPointHeadroom * potential_factor);
          for (std::size_t bin = 0; bin < count; ++bin) {
            auto const local = offset + bin;
            host[local] = static_cast<double>(mirror(local)) * inverse_scale;
            host[total_bins + local] =
                static_cast<double>(mirror(total_bins + local)) *
                inverse_scale;
            host[2 * total_bins + local] =
                static_cast<double>(mirror(2 * total_bins + local)) *
                inverse_scale;
          }
          offset += count;
        }
        statistics_.device_to_host_bytes +=
            fixed.extent(0) * sizeof(long long);
      } else {
        auto mirror = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),
                                                          floating);
        for (std::size_t index = 0; index < host.size(); ++index)
          host[index] = mirror(index);
        statistics_.device_to_host_bytes +=
            floating.extent(0) * sizeof(double);
      }
      std::size_t offset = 0;
      for (auto const& observer : observers) {
        auto const count =
            static_cast<std::size_t>(observer.number_of_bins);
        gpu::radio::RadioWaveform waveform{};
        waveform.x.assign(host.begin() + offset,
                          host.begin() + offset + count);
        waveform.y.assign(host.begin() + total_bins + offset,
                          host.begin() + total_bins + offset + count);
        waveform.z.assign(host.begin() + 2 * total_bins + offset,
                          host.begin() + 2 * total_bins + offset + count);
        result.push_back(std::move(waveform));
        offset += count;
      }
      return result;
    }

    void copyStatistics(detail::DeviceRadioCounters const& counters) {
      statistics_.lepton_tracks = counters.valid_tracks;
      statistics_.coreas_contributions = counters.coreas_contributions;
      statistics_.zhs_contributions = counters.zhs_contributions;
      statistics_.zhs_subtracks = counters.zhs_subtracks;
      statistics_.fixed_point_overflows = counters.fixed_point_overflows;
      statistics_.weighted_segment_count = counters.weighted_segment_count;
      statistics_.track_length_m = counters.track_length_m;
      statistics_.weighted_track_length_m = counters.weighted_track_length_m;
      statistics_.electron_weighted_track_length_m =
          counters.electron_weighted_track_length_m;
      statistics_.positron_weighted_track_length_m =
          counters.positron_weighted_track_length_m;
      statistics_.signed_charge_weighted_track_length_m =
          counters.signed_charge_weighted_track_length_m;
      statistics_.energy_weighted_track_length_GeV_m =
          counters.energy_weighted_track_length_GeV_m;
      statistics_.maximum_segment_length_m =
          counters.maximum_segment_length_m;
      statistics_.weighted_direction_change_rad =
          counters.weighted_direction_change_rad;
      statistics_.weighted_direction_change_squared_rad2 =
          counters.weighted_direction_change_squared_rad2;
      statistics_.weighted_beta_deficit_track_length_m =
          counters.weighted_beta_deficit_track_length_m;
      statistics_.weighted_time_residual_s = counters.weighted_time_residual_s;
      statistics_.maximum_direction_change_rad =
          counters.maximum_direction_change_rad;
      for (std::size_t axis = 0; axis < 3; ++axis)
        statistics_.signed_charge_weighted_direction_change[axis] =
            counters.signed_charge_weighted_direction_change[axis];
      for (std::size_t index = 0; index < 15; ++index)
        statistics_.weighted_track_length_by_kinetic_energy_m[index] =
            counters.weighted_track_length_by_kinetic_energy_m[index];
    }

    static void validate(gpu::radio::GpuRadioConfig const& config) {
      auto const& propagation = config.propagation;
      if (!config.coreas_enabled && !config.zhs_enabled)
        throw std::invalid_argument("enabled Kokkos radio has no algorithm");
      if (!std::isfinite(config.fixed_point_field_limit_V_per_m) ||
          !(config.fixed_point_field_limit_V_per_m > 0.))
        throw std::invalid_argument("invalid Kokkos radio fixed-point limit");
      if (config.zhs_subtrack_refinement < 1 ||
          config.zhs_subtrack_refinement > 64)
        throw std::invalid_argument(
            "Kokkos ZHS subtrack refinement must be in [1,64]");
      if (propagation.refractivity.size() < 11 ||
          propagation.refractivity.size() !=
              propagation.integrated_refractivity.size() ||
          !std::isfinite(propagation.minimum_height_m) ||
          !std::isfinite(propagation.maximum_height_m) ||
          !(propagation.maximum_height_m > propagation.minimum_height_m) ||
          !std::isfinite(propagation.step_m) || !(propagation.step_m > 0.) ||
          !std::isfinite(propagation.inverse_step_per_m) ||
          !(propagation.inverse_step_per_m > 0.))
        throw std::invalid_argument(
            "invalid Kokkos flat-atmosphere radio snapshot");
      auto validate_observers =
          [&](std::vector<gpu::radio::RadioObserverSnapshot> const& observers) {
            for (auto const& observer : observers) {
              if (!std::isfinite(observer.start_time_s) ||
                  !std::isfinite(observer.duration_s) ||
                  !(observer.duration_s >= 0.) ||
                  !std::isfinite(observer.sample_rate_Hz) ||
                  !(observer.sample_rate_Hz > 0.) ||
                  observer.number_of_bins == 0)
                throw std::invalid_argument("invalid Kokkos radio observer");
              auto const height =
                  (observer.position_m[2] - propagation.minimum_height_m) *
                  propagation.inverse_step_per_m;
              if (!std::isfinite(height) || height < 0. ||
                  height + 0.5 >=
                      static_cast<double>(propagation.refractivity.size()))
                throw std::invalid_argument(
                    "Kokkos radio observer outside propagation table");
              for (double coordinate : observer.position_m)
                if (!std::isfinite(coordinate))
                  throw std::invalid_argument(
                      "Kokkos radio observer position is not finite");
            }
          };
      validate_observers(config.coreas_observers);
      validate_observers(config.zhs_observers);
    }

    gpu::radio::GpuRadioConfig config_{};
    Kokkos::View<double*, memory_space> refractivity_{};
    Kokkos::View<double*, memory_space> integrated_refractivity_{};
    ObserverView coreas_observers_{};
    ObserverView zhs_observers_{};
    std::vector<detail::DeviceObserver> coreas_observers_host_{};
    std::vector<detail::DeviceObserver> zhs_observers_host_{};
    FixedWaveformView coreas_fixed_{};
    FixedWaveformView zhs_fixed_{};
    FloatingWaveformView coreas_floating_{};
    FloatingWaveformView zhs_floating_{};
    detail::DeviceWaveforms coreas_waveforms_{};
    detail::DeviceWaveforms zhs_waveforms_{};
    detail::DevicePropagation propagation_{};
    CounterView counters_{};
    TrackView tracks_{};
    gpu::radio::GpuRadioStatistics statistics_{};
    std::size_t coreas_bins_{};
    std::size_t zhs_bins_{};
    bool fuse_coreas_zhs_{};
    bool initialized_{};
    bool downloaded_{};
#if defined(CORSIKA8_KOKKOS_BACKEND_OPENMP)
    std::size_t track_tile_size_{TrackTileSize};
    std::size_t observer_tile_size_{ObserverTileSize};
    std::size_t team_size_{1};
#else
    std::size_t track_tile_size_{TrackTileSize};
    std::size_t observer_tile_size_{ObserverTileSize};
    std::size_t team_size_{256};
#endif
  };

} // namespace corsika::accelerator::radio::kokkos_detail

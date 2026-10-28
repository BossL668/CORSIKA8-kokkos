/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once
// Frozen production expressions: no migrated accumulation or merge function is called.
#include <tests/kokkos/radio/RadioProjectionReference.hpp>
#include <Kokkos_Core.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>
namespace radio_accumulator_reference {
  namespace gpu {
    namespace em = ::radio_reference::gpu::em;
    namespace radio = ::radio_reference;
  } // namespace gpu
  namespace em {
    struct KokkosTuningParameters {
      std::size_t batch_size{};
      std::size_t chunk_size{};
      std::size_t team_size{};
      std::size_t track_tile_size{};
      std::size_t observer_tile_size{};
      std::size_t device_queues{1};
    };
    namespace kokkos_detail {

      inline std::size_t checkedMemoryAdd(std::size_t const left,
                                          std::size_t const right) {
        if (right > std::numeric_limits<std::size_t>::max() - left)
          throw std::overflow_error("Kokkos resident byte count overflow");
        return left + right;
      }

      inline std::size_t checkedMemoryMultiply(std::size_t const count,
                                               std::size_t const element_bytes) {
        if (element_bytes != 0 &&
            count > std::numeric_limits<std::size_t>::max() / element_bytes)
          throw std::overflow_error("Kokkos resident byte count overflow");
        return count * element_bytes;
      }

      /** Retained bytes after growth and the peak while old/new storage overlaps. */
      struct KokkosMemoryProjection {
        std::size_t retained_bytes{};
        std::size_t transient_peak_bytes{};
      };

      /**
       * Compose a projection in allocation order.
       *
       * replace() models `member = newly_allocated_view`: the complete new
       * allocation exists before the old member is released. merge() embeds a
       * projection produced by a nested owner such as a queue or bucket workspace.
       */
      class KokkosMemoryProjectionBuilder {
      public:
        explicit KokkosMemoryProjectionBuilder(std::size_t const current)
            : retained_{current}
            , peak_{current} {}

        void replace(std::size_t const old_bytes, std::size_t const new_bytes) {
          if (old_bytes > retained_)
            throw std::logic_error(
                "Kokkos memory projection exceeds current retained bytes");
          peak_ = std::max(peak_, checkedMemoryAdd(retained_, new_bytes));
          retained_ = checkedMemoryAdd(retained_ - old_bytes, new_bytes);
        }

        void merge(std::size_t const old_bytes, KokkosMemoryProjection const& nested) {
          if (old_bytes > retained_)
            throw std::logic_error(
                "Kokkos nested memory projection exceeds retained bytes");
          auto const base = retained_ - old_bytes;
          peak_ = std::max(peak_, checkedMemoryAdd(base, nested.transient_peak_bytes));
          retained_ = checkedMemoryAdd(base, nested.retained_bytes);
        }

        /** Compose disjoint allocations projected against the same initial owner. */
        void appendDisjointGrowth(std::size_t const initial_owner_bytes,
                                  KokkosMemoryProjection const& projected) {
          if (projected.retained_bytes < initial_owner_bytes ||
              projected.transient_peak_bytes < projected.retained_bytes)
            throw std::logic_error("invalid disjoint Kokkos growth projection");
          peak_ =
              std::max(peak_, checkedMemoryAdd(retained_, projected.transient_peak_bytes -
                                                              initial_owner_bytes));
          retained_ =
              checkedMemoryAdd(retained_, projected.retained_bytes - initial_owner_bytes);
        }

        KokkosMemoryProjection result() const noexcept { return {retained_, peak_}; }

      private:
        std::size_t retained_{};
        std::size_t peak_{};
      };

      /**
       * Host-only callback used immediately before a grow-only device allocation.
       * The callback receives the current byte count of the owner being replaced,
       * so the backend can combine the projection with every other retained owner.
       */
      struct KokkosResidentMemoryBudgetGate {
        using Callback = bool (*)(void*, std::size_t, KokkosMemoryProjection const&,
                                  char const*);

        void* context{};
        Callback callback{};

        bool permits(std::size_t const current_owner_bytes,
                     KokkosMemoryProjection const& projection,
                     char const* const stage) const {
          return callback == nullptr ||
                 callback(context, current_owner_bytes, projection, stage);
        }
      };

    } // namespace kokkos_detail
    namespace detail {
      enum class CooperativeEndpoint : unsigned { Cuda, OpenMP };

      struct FixedAccumulatorLayout {
        // Identity covers algorithm, observers, binning and units, not just length.
        std::string identity;
        double units_per_integer{};
        std::size_t bins{};
      };

      inline void mergeFixedAccumulators(std::vector<std::int64_t>& destination,
                                         FixedAccumulatorLayout const& a,
                                         std::vector<std::int64_t> const& source,
                                         FixedAccumulatorLayout const& b) {
        if (a.identity.empty() || a.identity != b.identity || a.bins != b.bins ||
            !std::isfinite(a.units_per_integer) || a.units_per_integer <= 0. ||
            a.units_per_integer != b.units_per_integer || destination.size() != a.bins ||
            source.size() != b.bins)
          throw std::invalid_argument("incompatible fixed-point accumulators");
        // Preflight every bin before writing: failed merges cannot partially commit.
        for (std::size_t i = 0; i < source.size(); ++i) {
          auto x = destination[i], y = source[i];
          if ((y > 0 && x > std::numeric_limits<std::int64_t>::max() - y) ||
              (y < 0 && x < std::numeric_limits<std::int64_t>::min() - y))
            throw std::overflow_error("cooperative fixed-point accumulation overflow");
        }
        for (std::size_t i = 0; i < source.size(); ++i) destination[i] += source[i];
      }

    } // namespace detail
  }   // namespace em
  namespace detail {
    using namespace ::radio_reference;

    using CooperativeEndpoint = em::detail::CooperativeEndpoint;

    /** Final (not incremental) per-endpoint ledger. ZHS integers are potential,
     * not differentiated fields. Identity must include shower/config identity.
     */
    struct FixedRadioSnapshot {
      std::string shower_identity;
      CooperativeEndpoint endpoint{};
      gpu::radio::GpuRadioConfig config;
      std::vector<std::int64_t> coreas, zhs;
      DeviceRadioCounters counters{};
    };

    inline bool sameObservers(std::vector<gpu::radio::RadioObserverSnapshot> const& a,
                              std::vector<gpu::radio::RadioObserverSnapshot> const& b) {
      if (a.size() != b.size()) return false;
      for (std::size_t i = 0; i < a.size(); ++i) {
        for (int d = 0; d < 3; ++d)
          if (a[i].position_m[d] != b[i].position_m[d]) return false;
        if (a[i].start_time_s != b[i].start_time_s ||
            a[i].duration_s != b[i].duration_s ||
            a[i].sample_rate_Hz != b[i].sample_rate_Hz ||
            a[i].number_of_bins != b[i].number_of_bins)
          return false;
      }
      return true;
    }
    inline bool sameRadioPhysics(gpu::radio::GpuRadioConfig const& a,
                                 gpu::radio::GpuRadioConfig const& b) {
#define SAME(f) \
  if (a.f != b.f) return false
      SAME(enabled);
      SAME(coreas_enabled);
      SAME(zhs_enabled);
      SAME(deterministic);
      SAME(fixed_point_field_limit_V_per_m);
      SAME(zhs_subtrack_refinement);
      SAME(track_diagnostics);
      SAME(propagation.homogeneous_refractive_index);
      SAME(propagation.minimum_height_m);
      SAME(propagation.maximum_height_m);
      SAME(propagation.step_m);
      SAME(propagation.inverse_step_per_m);
      SAME(propagation.slope_refractivity_lower);
      SAME(propagation.slope_refractivity_upper);
      SAME(propagation.slope_integrated_refractivity_lower);
      SAME(propagation.slope_integrated_refractivity_upper);
      SAME(propagation.refractivity);
      SAME(propagation.integrated_refractivity);
#undef SAME
      return sameObservers(a.coreas_observers, b.coreas_observers) &&
             sameObservers(a.zhs_observers, b.zhs_observers);
    }
    inline std::size_t radioBins(
        std::vector<gpu::radio::RadioObserverSnapshot> const& observers) {
      std::size_t n = 0;
      for (auto const& o : observers) {
        if (!o.number_of_bins ||
            o.number_of_bins > std::numeric_limits<std::size_t>::max() / 3 - n ||
            !std::isfinite(o.sample_rate_Hz) || o.sample_rate_Hz <= 0. ||
            !std::isfinite(o.start_time_s) || !std::isfinite(o.duration_s) ||
            o.duration_s < 0.)
          throw std::invalid_argument("invalid cooperative radio observer layout");
        for (double x : o.position_m)
          if (!std::isfinite(x))
            throw std::invalid_argument("nonfinite cooperative radio observer");
        n += o.number_of_bins;
      }
      return n;
    }
    inline void validateFixedRadio(FixedRadioSnapshot const& s) {
      if (s.shower_identity.empty() || !s.config.enabled || !s.config.deterministic ||
          (!s.config.coreas_enabled && !s.config.zhs_enabled) ||
          !std::isfinite(s.config.fixed_point_field_limit_V_per_m) ||
          s.config.fixed_point_field_limit_V_per_m <= 0. ||
          static_cast<unsigned>(s.endpoint) > 1 || s.counters.fixed_point_overflows)
        throw std::invalid_argument(
            "invalid/overflowed cooperative fixed radio snapshot");
      if (s.coreas.size() != 3 * radioBins(s.config.coreas_observers) ||
          s.zhs.size() != 3 * radioBins(s.config.zhs_observers))
        throw std::invalid_argument("cooperative radio snapshot extent mismatch");
    }
    inline DeviceRadioCounters mergeRadioCounters(DeviceRadioCounters a,
                                                  DeviceRadioCounters const& b) {
      auto add = [](unsigned long long x, unsigned long long y) {
        if (y > std::numeric_limits<unsigned long long>::max() - x)
          throw std::overflow_error("cooperative radio counter overflow");
        return x + y;
      };
#define COUNT(f) a.f = add(a.f, b.f)
      COUNT(coreas_contributions);
      COUNT(zhs_contributions);
      COUNT(zhs_subtracks);
      COUNT(valid_tracks);
      COUNT(fixed_point_overflows);
#undef COUNT
      auto sum = [](double x, double y) {
        double z = x + y;
        if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z))
          throw std::overflow_error("nonfinite cooperative radio diagnostics");
        return z;
      };
#define SUM(f) a.f = sum(a.f, b.f)
      SUM(weighted_segment_count);
      SUM(track_length_m);
      SUM(weighted_track_length_m);
      SUM(electron_weighted_track_length_m);
      SUM(positron_weighted_track_length_m);
      SUM(signed_charge_weighted_track_length_m);
      SUM(energy_weighted_track_length_GeV_m);
      SUM(weighted_direction_change_rad);
      SUM(weighted_direction_change_squared_rad2);
      SUM(weighted_beta_deficit_track_length_m);
      SUM(weighted_time_residual_s);
      for (int i = 0; i < 3; ++i) SUM(signed_charge_weighted_direction_change[i]);
      for (int i = 0; i < 15; ++i) SUM(weighted_track_length_by_kinetic_energy_m[i]);
#undef SUM
      // Maxima are not additive. Two finite maxima can have an overflowing sum
      // without their (valid) merged maximum overflowing.
      for (double x : {a.maximum_segment_length_m, b.maximum_segment_length_m,
                       a.maximum_direction_change_rad, b.maximum_direction_change_rad})
        if (!std::isfinite(x))
          throw std::overflow_error("nonfinite cooperative radio maximum");
      a.maximum_segment_length_m =
          std::max(a.maximum_segment_length_m, b.maximum_segment_length_m);
      a.maximum_direction_change_rad =
          std::max(a.maximum_direction_change_rad, b.maximum_direction_change_rad);
      return a;
    }

    /** Convert only after the integer ledgers have been merged. This preserves
     * downloadSet's exact inverse-scale expression and SoA order. The existing
     * RadioProcess::endOfShower, not this function, differentiates ZHS potential.
     */
    inline gpu::radio::GpuRadioWaveforms decodeFixedRadio(FixedRadioSnapshot const& s) {
      validateFixedRadio(s);
      gpu::radio::GpuRadioWaveforms result;
      auto decode = [&](auto const& observers, auto const& integers, bool zhs) {
        std::vector<gpu::radio::RadioWaveform> out;
        auto bins = radioBins(observers);
        std::size_t offset = 0;
        for (auto const& observer : observers) {
          auto count = static_cast<std::size_t>(observer.number_of_bins);
          auto factor = zhs ? observer.sample_rate_Hz : 1.;
          auto inv =
              s.config.fixed_point_field_limit_V_per_m / (FixedPointHeadroom * factor);
          gpu::radio::RadioWaveform w;
          w.x.resize(count);
          w.y.resize(count);
          w.z.resize(count);
          for (std::size_t i = 0; i < count; ++i) {
            w.x[i] = static_cast<double>(integers[offset + i]) * inv;
            w.y[i] = static_cast<double>(integers[bins + offset + i]) * inv;
            w.z[i] = static_cast<double>(integers[2 * bins + offset + i]) * inv;
          }
          offset += count;
          out.push_back(std::move(w));
        }
        return out;
      };
      result.coreas = decode(s.config.coreas_observers, s.coreas, false);
      result.zhs = decode(s.config.zhs_observers, s.zhs, true);
      return result;
    }

    /** Exactly once per endpoint, all-or-nothing across BOTH algorithms/counters.
     * Final snapshots only: accepting incremental snapshots here would double
     * count tracks. This object must be reset/replaced at each shower boundary.
     */
    class CooperativeRadioMerge {
    public:
      CooperativeRadioMerge(std::string identity, gpu::radio::GpuRadioConfig config)
          : identity_(std::move(identity))
          , config_(std::move(config)) {
        if (identity_.empty())
          throw std::invalid_argument("empty cooperative shower identity");
      }
      void commit(FixedRadioSnapshot const& input) {
        if (finished_) throw std::logic_error("cooperative radio output already taken");
        validateFixedRadio(input);
        auto bit = 1u << static_cast<unsigned>(input.endpoint);
        if ((seen_ & bit) || input.shower_identity != identity_ ||
            !sameRadioPhysics(input.config, config_))
          throw std::invalid_argument("duplicate/incompatible cooperative radio commit");
        auto next = seen_ ? merged_ : input;
        if (seen_) {
          // Config equality above covers each observer's sample rate/scale and
          // propagation model. The layout label also disambiguates field/potential.
          auto merge = [&](auto& dst, auto const& src, char const* algorithm) {
            em::detail::FixedAccumulatorLayout layout{
                identity_ + algorithm, config_.fixed_point_field_limit_V_per_m,
                dst.size()};
            em::detail::mergeFixedAccumulators(dst, layout, src, layout);
          };
          merge(next.coreas, input.coreas, "/coreas-field-soa-v1");
          merge(next.zhs, input.zhs, "/zhs-potential-soa-v1");
          next.counters = mergeRadioCounters(next.counters, input.counters);
        }
        merged_ = std::move(next);
        seen_ |= bit;
      }
      FixedRadioSnapshot take() {
        if (seen_ != 3 || finished_)
          throw std::logic_error(
              "cooperative radio endpoints not complete/already taken");
        finished_ = true;
        return std::move(merged_);
      }

    private:
      std::string identity_;
      gpu::radio::GpuRadioConfig config_;
      FixedRadioSnapshot merged_;
      unsigned seen_{};
      bool finished_{};
    };

  } // namespace detail

  struct KokkosRadioAtomicOperations {
    KOKKOS_INLINE_FUNCTION static long long add(long long* address, long long value) {
      // Radio fixed-point waveforms execute this fetch-add billions of times.
      // Kokkos/desul implements the returning 64-bit overload with a CAS loop
      // on CUDA, whereas the production native-CUDA path emits one hardware
      // ATOMG.ADD.64.  Use the backend intrinsic in device compilation and
      // retain Kokkos' portable implementation for OpenMP.  Two's-complement
      // modular addition has the same bit result for signed and unsigned
      // storage, and the caller performs its existing overflow check on the
      // returned bits.
      auto* const storage = reinterpret_cast<unsigned long long*>(address);
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      auto const previous = atomicAdd(storage, static_cast<unsigned long long>(value));
#else
      auto const previous =
          Kokkos::atomic_fetch_add(storage, static_cast<unsigned long long>(value));
#endif
      return static_cast<long long>(previous);
    }

    KOKKOS_INLINE_FUNCTION static unsigned long long add(unsigned long long* address,
                                                         unsigned long long value) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      return atomicAdd(address, value);
#else
      return Kokkos::atomic_fetch_add(address, value);
#endif
    }

    KOKKOS_INLINE_FUNCTION static double add(double* address, double value) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      return atomicAdd(address, value);
#else
      return Kokkos::atomic_fetch_add(address, value);
#endif
    }

    KOKKOS_INLINE_FUNCTION static void maximum(double* address, double value) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      // Diagnostic maxima are finite and non-negative, so IEEE-754 bit order
      // is monotonic and matches the native CUDA implementation.
      auto* const bits = reinterpret_cast<unsigned long long*>(address);
      atomicMax(bits, __double_as_longlong(value));
#else
      Kokkos::atomic_max(address, value);
#endif
    }
  };

  // Projection diagnostics use thread-local unsigned counters. Waveform
  // additions (signed fixed-point or double) keep the global atomic path.
  struct KokkosRadioLocalCounterOperations : KokkosRadioAtomicOperations {
    using KokkosRadioAtomicOperations::add;

    KOKKOS_INLINE_FUNCTION static unsigned long long add(unsigned long long* address,
                                                         unsigned long long value) {
      auto const previous = *address;
      *address += value;
      return previous;
    }
  };

  struct KokkosRadioOpenMpThreadLocalOperations {
    KOKKOS_INLINE_FUNCTION static long long add(long long* address, long long value) {
      // Each OpenMP worker owns a separate waveform row.  Preserve the same
      // two's-complement addition and previous-value overflow contract as the
      // atomic path without paying for a locked instruction on every sample.
      auto* const storage = reinterpret_cast<unsigned long long*>(address);
      auto const previous = *storage;
      *storage = previous + static_cast<unsigned long long>(value);
      return static_cast<long long>(previous);
    }

    KOKKOS_INLINE_FUNCTION static unsigned long long add(unsigned long long* address,
                                                         unsigned long long value) {
      auto const previous = *address;
      *address = previous + value;
      return previous;
    }

    KOKKOS_INLINE_FUNCTION static double add(double* address, double value) {
      auto const previous = *address;
      *address = previous + value;
      return previous;
    }

    KOKKOS_INLINE_FUNCTION static void maximum(double* address, double value) {
      if (value > *address) *address = value;
    }
  };

  // A named functor also compiles in a CUDA-enabled Kokkos package when the
  // caller uses OpenMP. NVCC forbids extended lambdas in private members.
  template <class ThreadView, class OutputView>
  struct MergeThreadWaveformKernel {
    ThreadView private_waveform;
    OutputView waveform;
    std::size_t thread_slots;
    KOKKOS_INLINE_FUNCTION void operator()(std::size_t const index,
                                           unsigned long long& local_overflows) const {
      long long sum{};
      for (std::size_t slot = 0; slot < thread_slots; ++slot) {
        auto const increment = private_waveform(slot, index);
        auto const previous =
            KokkosRadioOpenMpThreadLocalOperations::add(&sum, increment);
        if ((increment > 0 && previous > detail::SignedIntegerMaximum - increment) ||
            (increment < 0 && previous < detail::SignedIntegerMinimum - increment))
          ++local_overflows;
      }
      waveform(index) = sum;
    }
  };

  template <class ExecutionSpace>
  class KokkosRadioAccumulator {
  public:
#if defined(KOKKOS_ENABLE_OPENMP)
    static constexpr bool IsOpenMP = std::is_same_v<ExecutionSpace, Kokkos::OpenMP>;
#else
    static constexpr bool IsOpenMP = false;
#endif
    using memory_space = typename ExecutionSpace::memory_space;
    using TrackInputView = Kokkos::View<gpu::em::LeptonTransportRecord*, memory_space>;
    using HostTrackInputView = typename TrackInputView::HostMirror;
    // Independent of the EM dispatch threshold: even a small particle batch
    // can create substantial radio work when many observers are configured.
    static constexpr std::size_t HostTrackBatchCapacity = 4096;
    using TrackView = Kokkos::View<detail::RadioTrackKinematics*, memory_space>;
    using ObserverView = Kokkos::View<detail::DeviceObserver*, memory_space>;
    using FixedWaveformView = Kokkos::View<long long*, memory_space>;
    using FloatingWaveformView = Kokkos::View<double*, memory_space>;
    using ThreadFixedWaveformView =
        Kokkos::View<long long**, Kokkos::LayoutRight, memory_space>;
    using ThreadCounterView = Kokkos::View<detail::DeviceRadioCounters*, memory_space>;
    using CounterView = Kokkos::View<detail::DeviceRadioCounters*, memory_space>;

    static std::size_t projectedDeviceBytes(gpu::radio::GpuRadioConfig const& requested) {
      if (!requested.enabled) return 0;
      validate(requested);
      using ::radio_accumulator_reference::em::kokkos_detail::checkedMemoryAdd;
      using ::radio_accumulator_reference::em::kokkos_detail::checkedMemoryMultiply;
      auto total = checkedMemoryMultiply(requested.propagation.refractivity.size(),
                                         sizeof(double));
      total = checkedMemoryAdd(
          total,
          checkedMemoryMultiply(requested.propagation.integrated_refractivity.size(),
                                sizeof(double)));
      total =
          checkedMemoryAdd(total, checkedMemoryMultiply(requested.coreas_observers.size(),
                                                        sizeof(detail::DeviceObserver)));
      total =
          checkedMemoryAdd(total, checkedMemoryMultiply(requested.zhs_observers.size(),
                                                        sizeof(detail::DeviceObserver)));
      auto append_waveforms = [&](auto const& observers) {
        std::size_t bins{};
        for (auto const& observer : observers)
          bins =
              checkedMemoryAdd(bins, static_cast<std::size_t>(observer.number_of_bins));
        auto const values = checkedMemoryMultiply(bins, std::size_t{3});
        total =
            checkedMemoryAdd(total, checkedMemoryMultiply(values, requested.deterministic
                                                                      ? sizeof(long long)
                                                                      : sizeof(double)));
        if constexpr (IsOpenMP) {
          if (requested.deterministic) {
            auto const thread_slots =
                static_cast<std::size_t>(std::max(1, ExecutionSpace().concurrency()));
            total = checkedMemoryAdd(
                total, checkedMemoryMultiply(checkedMemoryMultiply(values, thread_slots),
                                             sizeof(long long)));
          }
        }
      };
      append_waveforms(requested.coreas_observers);
      append_waveforms(requested.zhs_observers);
      total = checkedMemoryAdd(total, sizeof(detail::DeviceRadioCounters));
      total = checkedMemoryAdd(
          total, checkedMemoryMultiply(HostTrackBatchCapacity,
                                       sizeof(gpu::em::LeptonTransportRecord)));
      if constexpr (IsOpenMP) {
        if (requested.deterministic) {
          auto const thread_slots =
              static_cast<std::size_t>(std::max(1, ExecutionSpace().concurrency()));
          total = checkedMemoryAdd(
              total,
              checkedMemoryMultiply(thread_slots, sizeof(detail::DeviceRadioCounters)));
        }
      }
      return total;
    }

    static constexpr std::size_t TrackTileSize = IsOpenMP ? 4 : 8;
    static constexpr std::size_t ObserverTileSize = IsOpenMP ? 4 : 32;
    static constexpr std::size_t MaximumTrackTileSize = 16;
    static constexpr std::size_t MaximumObserverTileSize = 64;

    struct RadioTileScratch {
      detail::RadioTrackKinematics tracks[MaximumTrackTileSize];
      detail::DeviceObserver coreas_observers[MaximumObserverTileSize];
      detail::DeviceObserver zhs_observers[MaximumObserverTileSize];
    };

    // A named functor gives host and device passes the same kernel identity.
    // NVCC can number the two extended lambdas inside project()'s constexpr
    // OpenMP/GPU branches differently, leaving the tiled GPU entry unregistered.
    // This is the former GPU lambda body, with the same data and tile order.
    template <bool Coreas, bool Zhs, bool ReduceCounters = false>
    struct RadioTileProjection {
      TrackView tracks;
      ObserverView coreas_observers;
      ObserverView zhs_observers;
      detail::DevicePropagation propagation;
      detail::DeviceWaveforms coreas_waveforms;
      detail::DeviceWaveforms zhs_waveforms;
      CounterView counters;
      std::size_t track_count;
      std::size_t observer_count;
      std::size_t observer_tiles;
      std::size_t track_tile_size;
      std::size_t observer_tile_size;
      std::size_t tile_lanes;

      using Member = typename Kokkos::TeamPolicy<ExecutionSpace>::member_type;
      KOKKOS_INLINE_FUNCTION void operator()(Member const& team) const {
        auto* const scratch = static_cast<RadioTileScratch*>(
            team.team_shmem().get_shmem(sizeof(RadioTileScratch)));
        auto const track_tile =
            static_cast<std::size_t>(team.league_rank()) / observer_tiles;
        auto const observer_tile =
            static_cast<std::size_t>(team.league_rank()) % observer_tiles;
        auto const track_base = track_tile * track_tile_size;
        auto const observer_base = observer_tile * observer_tile_size;
        for (std::size_t index = team.team_rank(); index < track_tile_size;
             index += team.team_size()) {
          auto const global = track_base + index;
          scratch->tracks[index] =
              global < track_count ? tracks(global) : detail::RadioTrackKinematics{};
        }
        for (std::size_t index = team.team_rank(); index < observer_tile_size;
             index += team.team_size()) {
          auto const global = observer_base + index;
          if (global < observer_count) {
            if (Coreas) scratch->coreas_observers[index] = coreas_observers(global);
            if (Zhs) scratch->zhs_observers[index] = zhs_observers(global);
          }
        }
        team.team_barrier();
        detail::DeviceRadioCounters local_counts{};
        using ProjectionOperations =
            std::conditional_t<ReduceCounters, KokkosRadioLocalCounterOperations,
                               KokkosRadioAtomicOperations>;
        auto* const projection_counts = ReduceCounters ? &local_counts : counters.data();
        for (std::size_t lane = team.team_rank(); lane < tile_lanes;
             lane += team.team_size()) {
          auto const track_lane = lane / observer_tile_size;
          auto const observer_lane = lane % observer_tile_size;
          auto const track_index = track_tile * track_tile_size + track_lane;
          auto const observer_index = observer_tile * observer_tile_size + observer_lane;
          if (track_index >= track_count || observer_index >= observer_count ||
              scratch->tracks[track_lane].valid == 0)
            continue;
          if (Coreas)
            detail::accumulateCoREAS<ProjectionOperations>(
                scratch->tracks[track_lane], propagation,
                scratch->coreas_observers[observer_lane], coreas_waveforms,
                projection_counts);
          if (Zhs)
            detail::accumulateZHS<ProjectionOperations>(
                scratch->tracks[track_lane], propagation,
                scratch->zhs_observers[observer_lane], zhs_waveforms, projection_counts);
        }
        if constexpr (ReduceCounters) {
          // All team members participate, including empty tail-tile lanes.
          team.team_reduce(
              Kokkos::Sum<unsigned long long>(local_counts.coreas_contributions));
          if (team.team_rank() == 0 && local_counts.coreas_contributions)
            KokkosRadioAtomicOperations::add(&counters(0).coreas_contributions,
                                             local_counts.coreas_contributions);
          team.team_reduce(
              Kokkos::Sum<unsigned long long>(local_counts.zhs_contributions));
          if (team.team_rank() == 0 && local_counts.zhs_contributions)
            KokkosRadioAtomicOperations::add(&counters(0).zhs_contributions,
                                             local_counts.zhs_contributions);
          team.team_reduce(Kokkos::Sum<unsigned long long>(local_counts.zhs_subtracks));
          if (team.team_rank() == 0 && local_counts.zhs_subtracks)
            KokkosRadioAtomicOperations::add(&counters(0).zhs_subtracks,
                                             local_counts.zhs_subtracks);
          team.team_reduce(
              Kokkos::Sum<unsigned long long>(local_counts.fixed_point_overflows));
          if (team.team_rank() == 0 && local_counts.fixed_point_overflows)
            KokkosRadioAtomicOperations::add(&counters(0).fixed_point_overflows,
                                             local_counts.fixed_point_overflows);
        }
      }
    };

    // Native CUDA falls back to its flat pair kernel after the EM workspace
    // consumes the optional track-cache budget.  Production traces show that
    // this is also the faster launch shape for the many small edge
    // wavefronts: it exposes all track/observer pairs directly and avoids the
    // TeamPolicy overhead of mostly empty tiles.
    static constexpr std::size_t DirectProjectionTrackThreshold = 16384;

    void initialize(
        gpu::radio::GpuRadioConfig const& requested,
        ::radio_accumulator_reference::em::KokkosTuningParameters const& tuning = {},
        ExecutionSpace const& execution = {}) {
      if (initialized_)
        throw std::logic_error("Kokkos radio accumulator already initialized");
      config_ = requested;
      if (tuning.track_tile_size != 0) track_tile_size_ = tuning.track_tile_size;
      if (tuning.observer_tile_size != 0) observer_tile_size_ = tuning.observer_tile_size;
      if (tuning.team_size != 0) team_size_ = tuning.team_size;
      if (track_tile_size_ == 0 || observer_tile_size_ == 0 || team_size_ == 0 ||
          team_size_ > 1024)
        throw std::invalid_argument("Kokkos radio tuning is invalid");
      if constexpr (!IsOpenMP) {
        if (track_tile_size_ > MaximumTrackTileSize ||
            observer_tile_size_ > MaximumObserverTileSize)
          throw std::invalid_argument(
              "Kokkos GPU radio tiling exceeds the validated scratch limits");
      }
      if (!requested.enabled) return;
      validate(requested);
      coreas_observers_host_ =
          makeObservers(requested.coreas_observers, coreas_bins_, false);
      zhs_observers_host_ = makeObservers(requested.zhs_observers, zhs_bins_, true);
      upload(coreas_observers_host_, coreas_observers_, execution,
             "c8_kokkos_coreas_observers");
      upload(zhs_observers_host_, zhs_observers_, execution, "c8_kokkos_zhs_observers");
      upload(requested.propagation.refractivity, refractivity_, execution,
             "c8_kokkos_radio_refractivity");
      upload(requested.propagation.integrated_refractivity, integrated_refractivity_,
             execution, "c8_kokkos_radio_integrated_refractivity");
      propagation_.homogeneous_refractive_index =
          requested.propagation.homogeneous_refractive_index;
      propagation_.minimum_height_m = requested.propagation.minimum_height_m;
      propagation_.maximum_height_m = requested.propagation.maximum_height_m;
      propagation_.step_m = requested.propagation.step_m;
      propagation_.inverse_step_per_m = requested.propagation.inverse_step_per_m;
      propagation_.slope_refractivity_lower =
          requested.propagation.slope_refractivity_lower;
      propagation_.slope_integrated_refractivity_lower =
          requested.propagation.slope_integrated_refractivity_lower;
      propagation_.slope_refractivity_upper =
          requested.propagation.slope_refractivity_upper;
      propagation_.slope_integrated_refractivity_upper =
          requested.propagation.slope_integrated_refractivity_upper;
      propagation_.refractivity = refractivity_.data();
      propagation_.integrated_refractivity = integrated_refractivity_.data();
      propagation_.table_size = refractivity_.extent(0);
      propagation_.zhs_subtrack_refinement = requested.zhs_subtrack_refinement;
      allocateWaveforms(coreas_bins_, coreas_fixed_, coreas_floating_, coreas_waveforms_,
                        "c8_kokkos_coreas_waveforms");
      allocateWaveforms(zhs_bins_, zhs_fixed_, zhs_floating_, zhs_waveforms_,
                        "c8_kokkos_zhs_waveforms");
      if constexpr (IsOpenMP) {
        if (requested.deterministic) {
          Kokkos::Experimental::UniqueToken<ExecutionSpace> tokens(execution);
          openmp_thread_slots_ = std::max<std::size_t>(1, tokens.size());
          allocateThreadWaveforms(coreas_bins_, coreas_thread_fixed_,
                                  openmp_thread_slots_, execution,
                                  "c8_kokkos_coreas_thread_waveforms");
          allocateThreadWaveforms(zhs_bins_, zhs_thread_fixed_, openmp_thread_slots_,
                                  execution, "c8_kokkos_zhs_thread_waveforms");
          thread_counters_ =
              ThreadCounterView("c8_kokkos_radio_thread_counters", openmp_thread_slots_);
        }
      }
      counters_ = CounterView("c8_kokkos_radio_counters", 1);
      host_track_input_device_ =
          TrackInputView(Kokkos::view_alloc(execution, Kokkos::WithoutInitializing,
                                            "c8_kokkos_host_radio_input"),
                         HostTrackBatchCapacity);
      host_track_input_ = Kokkos::create_mirror_view(host_track_input_device_);
      fuse_coreas_zhs_ =
          requested.coreas_enabled && requested.zhs_enabled &&
          observersCanShareTrackKinematics(coreas_observers_host_, zhs_observers_host_);
      initialized_ = true;
      reset(execution);
    }

    bool enabled() const noexcept { return initialized_ && config_.enabled; }

    void reset(ExecutionSpace const& execution = {}) {
      if (!enabled()) return;
      if (host_track_count_ != 0)
        throw std::logic_error("cannot reset radio with uncommitted host tracks");
      if (coreas_fixed_.extent(0)) Kokkos::deep_copy(execution, coreas_fixed_, 0LL);
      if (zhs_fixed_.extent(0)) Kokkos::deep_copy(execution, zhs_fixed_, 0LL);
      if (coreas_floating_.extent(0)) Kokkos::deep_copy(execution, coreas_floating_, 0.);
      if (zhs_floating_.extent(0)) Kokkos::deep_copy(execution, zhs_floating_, 0.);
      if constexpr (IsOpenMP) {
        if (coreas_thread_fixed_.extent(0))
          Kokkos::deep_copy(execution, coreas_thread_fixed_, 0LL);
        if (zhs_thread_fixed_.extent(0))
          Kokkos::deep_copy(execution, zhs_thread_fixed_, 0LL);
        if (thread_counters_.extent(0))
          Kokkos::deep_copy(execution, thread_counters_, detail::DeviceRadioCounters{});
      }
      Kokkos::deep_copy(execution, counters_, detail::DeviceRadioCounters{});
      execution.fence("reset Kokkos radio accumulator");
      statistics_ = {};
      statistics_.device_bytes = deviceBytes();
      statistics_.track_precompute_enabled = true;
      statistics_.track_tile_size = track_tile_size_;
      statistics_.observer_tile_size = observer_tile_size_;
      statistics_.track_diagnostics_enabled = config_.track_diagnostics;
      downloaded_ = false;
      fixed_snapshot_sealed_ = false;
    }

    // Copy ownership immediately; scalar transport may reuse its record as
    // soon as this method returns. No allocation or fence occurs per track.
    void enqueueHostLeptonTracks(gpu::em::LeptonTransportRecord const* records,
                                 std::size_t count,
                                 ExecutionSpace const& execution = {}) {
      if (!enabled())
        throw std::logic_error("host radio tracks requested while radio is disabled");
      if (downloaded_ || fixed_snapshot_sealed_)
        throw std::logic_error("radio already downloaded; reset before host tracks");
      if (count != 0 && records == nullptr)
        throw std::invalid_argument("null host radio track input");
      while (count != 0) {
        auto const copied = std::min(count, HostTrackBatchCapacity - host_track_count_);
        for (std::size_t index = 0; index < copied; ++index)
          host_track_input_(host_track_count_ + index) = records[index];
        host_track_count_ += copied;
        statistics_.host_tracks_enqueued += copied;
        records += copied;
        count -= copied;
        if (host_track_count_ == HostTrackBatchCapacity) flushHostLeptonTracks(execution);
      }
    }

    void flushHostLeptonTracks(ExecutionSpace const& execution = {}) {
      if (host_track_count_ == 0) return;
      auto const range = std::make_pair(std::size_t{0}, host_track_count_);
      Kokkos::deep_copy(execution, Kokkos::subview(host_track_input_device_, range),
                        Kokkos::subview(host_track_input_, range));
      // The reusable host mirror may be filled again while projection runs.
      // Complete its upload first; all projection/device reuse stays ordered
      // on this same execution stream, without a per-track synchronization.
      execution.fence("upload batched host radio tracks");
      accumulateLeptonTracks(host_track_input_device_, host_track_count_, execution);
      statistics_.host_to_device_bytes +=
          host_track_count_ * sizeof(gpu::em::LeptonTransportRecord);
      ++statistics_.host_track_batches;
      host_track_count_ = 0;
    }

    void accumulateLeptonTracks(TrackInputView const& records, std::size_t const count,
                                ExecutionSpace const& execution = {}) {
      if (!enabled() || count == 0) return;
      if (fixed_snapshot_sealed_)
        throw std::logic_error(
            "radio endpoint already committed; reset before new tracks");
      if (count > records.extent(0))
        throw std::out_of_range("Kokkos radio track count exceeds input view");
      if constexpr (!IsOpenMP) {
        if (count < DirectProjectionTrackThreshold) {
          auto counters = counters_;
          auto const collect_diagnostics = config_.track_diagnostics;
          Kokkos::parallel_for(
              "c8_kokkos_scan_direct_radio_tracks",
              Kokkos::RangePolicy<ExecutionSpace>(execution, 0, count),
              KOKKOS_LAMBDA(std::size_t const index) {
                detail::recordTrack<KokkosRadioAtomicOperations>(
                    records(index), nullptr, counters.data(), collect_diagnostics);
              });
          if (fuse_coreas_zhs_) {
            projectDirect<true, true>(records, count, coreas_observers_, zhs_observers_,
                                      coreas_observers_.extent(0), execution);
          } else {
            if (config_.coreas_enabled)
              projectDirect<true, false>(records, count, coreas_observers_, {},
                                         coreas_observers_.extent(0), execution);
            if (config_.zhs_enabled)
              projectDirect<false, true>(records, count, {}, zhs_observers_,
                                         zhs_observers_.extent(0), execution);
          }
          statistics_.direct_projection_batches++;
          statistics_.direct_projection_records += count;
          statistics_.maximum_track_batch =
              std::max(statistics_.maximum_track_batch, count);
          downloaded_ = false;
          return;
        }
      }
      ensureTrackCapacity(count, execution);
      auto tracks = tracks_;
      auto counters = counters_;
      auto const collect_diagnostics = config_.track_diagnostics;
      Kokkos::parallel_for(
          "c8_kokkos_precompute_radio_tracks",
          Kokkos::RangePolicy<ExecutionSpace>(execution, 0, count),
          KOKKOS_LAMBDA(std::size_t const index) {
            detail::recordTrack<KokkosRadioAtomicOperations>(
                records(index), &tracks(index), counters.data(), collect_diagnostics);
          });
      statistics_.track_precompute_batches++;
      statistics_.track_precomputed_records += count;
      statistics_.maximum_track_batch = std::max(statistics_.maximum_track_batch, count);

      if (fuse_coreas_zhs_) {
        project<true, true>(tracks, count, coreas_observers_, zhs_observers_,
                            coreas_observers_.extent(0), execution);
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

    // Cooperative end-of-shower export. No conversion/differentiation, and
    // no default execution-space copy that might wait on another endpoint.
    detail::FixedRadioSnapshot downloadFixed(std::string const& shower_identity,
                                             detail::CooperativeEndpoint endpoint,
                                             ExecutionSpace const& execution = {}) {
      if (!enabled() || !config_.deterministic || downloaded_ || fixed_snapshot_sealed_)
        throw std::logic_error("fixed radio snapshot unavailable/already committed");
      flushHostLeptonTracks(execution);
      detail::FixedRadioSnapshot result;
      result.shower_identity = shower_identity;
      result.endpoint = endpoint;
      result.config = config_;
      if (shower_identity.empty() || static_cast<unsigned>(endpoint) > 1)
        throw std::invalid_argument("invalid cooperative radio snapshot identity");
      // Poison on a failed export: folding thread counters must never happen
      // twice if a transfer or allocation subsequently throws.
      fixed_snapshot_sealed_ = true;
      if constexpr (IsOpenMP) mergeThreadWaveforms(execution);
      auto host_counters = Kokkos::create_mirror(counters_);
      auto host_coreas = Kokkos::create_mirror(coreas_fixed_);
      auto host_zhs = Kokkos::create_mirror(zhs_fixed_);
      try {
        Kokkos::deep_copy(execution, host_counters, counters_);
        if (coreas_fixed_.extent(0))
          Kokkos::deep_copy(execution, host_coreas, coreas_fixed_);
        if (zhs_fixed_.extent(0)) Kokkos::deep_copy(execution, host_zhs, zhs_fixed_);
        execution.fence("download cooperative fixed radio endpoint");
      } catch (...) {
        // A later copy can fail with an earlier copy still in flight. Keep
        // these local host destinations alive while draining this endpoint.
        try {
          execution.fence("unwind cooperative radio endpoint copy");
        } catch (...) {}
        throw;
      }
      result.counters = host_counters(0);
      copyStatistics(result.counters);
      result.coreas.resize(host_coreas.extent(0));
      result.zhs.resize(host_zhs.extent(0));
      for (std::size_t i = 0; i < result.coreas.size(); ++i)
        result.coreas[i] = host_coreas(i);
      for (std::size_t i = 0; i < result.zhs.size(); ++i) result.zhs[i] = host_zhs(i);
      detail::validateFixedRadio(result);
      statistics_.device_to_host_bytes +=
          sizeof(detail::DeviceRadioCounters) +
          (result.coreas.size() + result.zhs.size()) * sizeof(long long);
      downloaded_ = true;
      return result;
    }

    gpu::radio::GpuRadioWaveforms download(ExecutionSpace const& execution = {}) {
      if (!enabled()) return {};
      if (downloaded_ || fixed_snapshot_sealed_)
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
      if (statistics_.fixed_point_overflows != 0)
        throw std::runtime_error(
            "Kokkos radio fixed-point accumulator overflowed; increase "
            "fixed_point_field_limit_V_per_m");
      gpu::radio::GpuRadioWaveforms result{};
      result.coreas = downloadSet(config_.coreas_observers, coreas_bins_, coreas_fixed_,
                                  coreas_floating_, false);
      result.zhs =
          downloadSet(config_.zhs_observers, zhs_bins_, zhs_fixed_, zhs_floating_, true);
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
             coreas_thread_fixed_.span() * sizeof(long long) +
             zhs_thread_fixed_.span() * sizeof(long long) +
             thread_counters_.span() * sizeof(detail::DeviceRadioCounters) +
             host_track_input_device_.span() * sizeof(gpu::em::LeptonTransportRecord) +
             tracks_.span() * sizeof(detail::RadioTrackKinematics) +
             counters_.span() * sizeof(detail::DeviceRadioCounters);
    }

    /**
     * Bytes copied from immutable host configuration during initialize().
     * Waveforms and counters are initialized directly in the execution space,
     * so they are resident bytes but not host-to-device configuration bytes.
     */
    std::size_t staticHostToDeviceBytes() const {
      using ::radio_accumulator_reference::em::kokkos_detail::checkedMemoryAdd;
      using ::radio_accumulator_reference::em::kokkos_detail::checkedMemoryMultiply;
      auto bytes = checkedMemoryMultiply(refractivity_.span(), sizeof(double));
      bytes = checkedMemoryAdd(
          bytes, checkedMemoryMultiply(integrated_refractivity_.span(), sizeof(double)));
      bytes =
          checkedMemoryAdd(bytes, checkedMemoryMultiply(coreas_observers_.span(),
                                                        sizeof(detail::DeviceObserver)));
      return checkedMemoryAdd(
          bytes,
          checkedMemoryMultiply(zhs_observers_.span(), sizeof(detail::DeviceObserver)));
    }

    void reserveTrackCapacity(std::size_t const count, ExecutionSpace const& execution) {
      if (!enabled() || count == 0) return;
      if constexpr (!IsOpenMP) {
        if (count < DirectProjectionTrackThreshold) return;
      }
      ensureTrackCapacity(count, execution);
    }

    ::radio_accumulator_reference::em::kokkos_detail::KokkosMemoryProjection
    projectedTrackCapacity(std::size_t const count) const {
      using ::radio_accumulator_reference::em::kokkos_detail::
          KokkosMemoryProjectionBuilder;
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
      KokkosMemoryProjectionBuilder projection(current);
      projection.replace(
          tracks_.span() * sizeof(detail::RadioTrackKinematics),
          ::radio_accumulator_reference::em::kokkos_detail::checkedMemoryMultiply(
              grown, sizeof(detail::RadioTrackKinematics)));
      return projection.result();
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
        auto const potential_factor = zhs_potential ? observer.sample_rate_Hz : 1.;
        device.fixed_point_scale = config_.deterministic
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
        if (lhs.start_time_s != rhs.start_time_s || lhs.duration_s != rhs.duration_s ||
            lhs.sample_rate_Hz != rhs.sample_rate_Hz ||
            lhs.number_of_bins != rhs.number_of_bins)
          return false;
        for (int axis = 0; axis < 3; ++axis)
          if (lhs.position_m[axis] != rhs.position_m[axis]) return false;
      }
      return true;
    }

    void allocateWaveforms(std::size_t bins, FixedWaveformView& fixed,
                           FloatingWaveformView& floating, detail::DeviceWaveforms& view,
                           char const* label) {
      if (bins == 0) return;
      if (bins > std::numeric_limits<std::size_t>::max() / 3)
        throw std::length_error("Kokkos radio waveform size overflow");
      auto const waveform_values = 3 * bins;
      if (config_.deterministic) {
        fixed = FixedWaveformView(std::string(label), waveform_values);
        view.fixed_x = fixed.data();
        view.fixed_y = fixed.data() + bins;
        view.fixed_z = fixed.data() + 2 * bins;
      } else {
        floating = FloatingWaveformView(std::string(label), waveform_values);
        view.floating_x = floating.data();
        view.floating_y = floating.data() + bins;
        view.floating_z = floating.data() + 2 * bins;
      }
    }

    void allocateThreadWaveforms(std::size_t bins, ThreadFixedWaveformView& fixed,
                                 std::size_t thread_slots,
                                 ExecutionSpace const& execution, char const* label) {
      if (bins == 0) return;
      if (bins > std::numeric_limits<std::size_t>::max() / 3)
        throw std::length_error("Kokkos OpenMP private radio waveform size overflow");
      fixed = ThreadFixedWaveformView(
          Kokkos::view_alloc(execution, Kokkos::WithoutInitializing, std::string(label)),
          thread_slots, 3 * bins);
    }

    unsigned long long mergeThreadWaveform(
        ThreadFixedWaveformView const& private_waveform,
        FixedWaveformView const& waveform, ExecutionSpace const& execution,
        char const* label) {
      if (private_waveform.extent(0) == 0) return 0;
      if (private_waveform.extent(1) != waveform.extent(0))
        throw std::logic_error("Kokkos OpenMP private radio waveform extent mismatch");
      auto const thread_slots = private_waveform.extent(0);
      unsigned long long overflow_count{};
      Kokkos::parallel_reduce(
          label,
          Kokkos::RangePolicy<ExecutionSpace>(execution, 0, private_waveform.extent(1)),
          MergeThreadWaveformKernel<ThreadFixedWaveformView, FixedWaveformView>{
              private_waveform, waveform, thread_slots},
          overflow_count);
      return overflow_count;
    }

    void mergeThreadWaveforms(ExecutionSpace const& execution) {
      auto overflow_count =
          mergeThreadWaveform(coreas_thread_fixed_, coreas_fixed_, execution,
                              "c8_kokkos_merge_coreas_thread_waveforms");
      overflow_count += mergeThreadWaveform(zhs_thread_fixed_, zhs_fixed_, execution,
                                            "c8_kokkos_merge_zhs_thread_waveforms");
      // OpenMP memory space is host accessible.  The reductions above are the
      // completion boundary, so no projection kernel can still modify these
      // private counters while they are folded into the global diagnostics.
      auto& global = counters_(0);
      for (std::size_t slot = 0; slot < thread_counters_.extent(0); ++slot) {
        auto const& local = thread_counters_(slot);
        global.coreas_contributions += local.coreas_contributions;
        global.zhs_contributions += local.zhs_contributions;
        global.zhs_subtracks += local.zhs_subtracks;
        global.fixed_point_overflows += local.fixed_point_overflows;
      }
      global.fixed_point_overflows += overflow_count;
    }

    void ensureTrackCapacity(std::size_t count, ExecutionSpace const& execution) {
      if (tracks_.extent(0) >= count) return;
      if (tracks_.extent(0) > std::numeric_limits<std::size_t>::max() / 2)
        throw std::length_error("Kokkos radio track capacity overflow");
      auto capacity =
          tracks_.extent(0) == 0 ? count : std::max(count, 2 * tracks_.extent(0));
      // A prior asynchronous projection can still reference the old
      // allocation.  Capacity growth is rare; fence only this ownership
      // transition rather than every radio batch.
      if (tracks_.extent(0) != 0) execution.fence("grow Kokkos radio track workspace");
      tracks_ = TrackView(Kokkos::view_alloc(execution, Kokkos::WithoutInitializing,
                                             "c8_kokkos_radio_track_kinematics"),
                          capacity);
      statistics_.track_workspace_bytes = capacity * sizeof(detail::RadioTrackKinematics);
      statistics_.device_bytes = deviceBytes();
    }

  public:
    // Public only to satisfy NVCC's extended-lambda access restriction.  The
    // enclosing class remains in an internal detail namespace and this
    // function is not part of the backend interface.
    template <bool Coreas, bool Zhs>
    void projectDirect(TrackInputView const& records, std::size_t track_count,
                       ObserverView const& coreas_observers,
                       ObserverView const& zhs_observers, std::size_t observer_count,
                       ExecutionSpace const& execution) {
      if (observer_count == 0) return;
      if (track_count > std::numeric_limits<std::size_t>::max() / observer_count)
        throw std::overflow_error("Kokkos direct radio pair count overflow");
      auto const pairs = track_count * observer_count;
      auto const propagation = propagation_;
      auto const coreas_waveforms = coreas_waveforms_;
      auto const zhs_waveforms = zhs_waveforms_;
      auto counters = counters_;
      Kokkos::parallel_for(
          Coreas && Zhs ? "c8_kokkos_coreas_zhs_direct"
                        : Coreas ? "c8_kokkos_coreas_direct" : "c8_kokkos_zhs_direct",
          Kokkos::RangePolicy<ExecutionSpace>(execution, 0, pairs),
          KOKKOS_LAMBDA(std::size_t const pair_index) {
            auto const track_index = pair_index / observer_count;
            auto const observer_index = pair_index % observer_count;
            detail::RadioTrackKinematics track{};
            if (!detail::makeRadioTrackKinematics(records(track_index), track)) return;
            if (Coreas)
              detail::accumulateCoREAS<KokkosRadioAtomicOperations>(
                  track, propagation, coreas_observers(observer_index), coreas_waveforms,
                  counters.data());
            if (Zhs)
              detail::accumulateZHS<KokkosRadioAtomicOperations>(
                  track, propagation, zhs_observers(observer_index), zhs_waveforms,
                  counters.data());
          });
      statistics_.track_observer_pairs += pairs * static_cast<std::size_t>(Coreas + Zhs);
      if constexpr (Coreas && Zhs) statistics_.fused_track_observer_pairs += pairs;
    }

    template <bool Coreas, bool Zhs>
    void project(TrackView const& tracks, std::size_t track_count,
                 ObserverView const& coreas_observers, ObserverView const& zhs_observers,
                 std::size_t observer_count, ExecutionSpace const& execution) {
      if (observer_count == 0) return;
      auto const track_tiles = (track_count + track_tile_size_ - 1) / track_tile_size_;
      auto const observer_tiles =
          (observer_count + observer_tile_size_ - 1) / observer_tile_size_;
      auto const league_size = track_tiles * observer_tiles;
      auto const propagation = propagation_;
      auto const coreas_waveforms = coreas_waveforms_;
      auto const zhs_waveforms = zhs_waveforms_;
      auto counters = counters_;
      auto const track_tile_size = track_tile_size_;
      auto const observer_tile_size = observer_tile_size_;
      using Policy = Kokkos::TeamPolicy<ExecutionSpace>;
      using Member = typename Policy::member_type;
      if constexpr (IsOpenMP) {
        // A Kokkos OpenMP team consumes threads from the process-wide OpenMP
        // pool.  Requiring TrackTileSize * ObserverTileSize members would make
        // the kernel invalid whenever the user selects fewer than 16 threads.
        // Keep one OpenMP worker per tile and evaluate the small 4x4 tile
        // serially inside that worker.  Different tiles still run in parallel,
        // while the GPU execution spaces retain one lane per track/observer
        // pair below.
        auto const coreas_thread_fixed = coreas_thread_fixed_;
        auto const zhs_thread_fixed = zhs_thread_fixed_;
        auto const thread_counters = thread_counters_;
        auto const coreas_bins = coreas_bins_;
        auto const zhs_bins = zhs_bins_;
        Kokkos::Experimental::UniqueToken<ExecutionSpace> thread_tokens(execution);
        Kokkos::parallel_for(
            Coreas && Zhs ? "c8_kokkos_coreas_zhs_tiled"
                          : Coreas ? "c8_kokkos_coreas_tiled" : "c8_kokkos_zhs_tiled",
            Policy(execution, league_size, 1), KOKKOS_LAMBDA(Member const& team) {
              auto const thread_slot = thread_tokens.acquire();
              auto thread_coreas_waveforms = coreas_waveforms;
              auto thread_zhs_waveforms = zhs_waveforms;
              if (coreas_thread_fixed.extent(0) != 0) {
                auto* const base = &coreas_thread_fixed(thread_slot, 0);
                thread_coreas_waveforms.fixed_x = base;
                thread_coreas_waveforms.fixed_y = base + coreas_bins;
                thread_coreas_waveforms.fixed_z = base + 2 * coreas_bins;
              }
              if (zhs_thread_fixed.extent(0) != 0) {
                auto* const base = &zhs_thread_fixed(thread_slot, 0);
                thread_zhs_waveforms.fixed_x = base;
                thread_zhs_waveforms.fixed_y = base + zhs_bins;
                thread_zhs_waveforms.fixed_z = base + 2 * zhs_bins;
              }
              auto* const thread_counter = thread_counters.extent(0) == 0
                                               ? counters.data()
                                               : &thread_counters(thread_slot);
              auto const track_tile =
                  static_cast<std::size_t>(team.league_rank()) / observer_tiles;
              auto const observer_tile =
                  static_cast<std::size_t>(team.league_rank()) % observer_tiles;
              for (std::size_t track_lane = 0; track_lane < track_tile_size;
                   ++track_lane) {
                auto const track_index = track_tile * track_tile_size + track_lane;
                if (track_index >= track_count || tracks(track_index).valid == 0)
                  continue;
                for (std::size_t observer_lane = 0; observer_lane < observer_tile_size;
                     ++observer_lane) {
                  auto const observer_index =
                      observer_tile * observer_tile_size + observer_lane;
                  if (observer_index >= observer_count) continue;
                  if (thread_counters.extent(0) != 0) {
                    if (Coreas)
                      detail::accumulateCoREAS<KokkosRadioOpenMpThreadLocalOperations>(
                          tracks(track_index), propagation,
                          coreas_observers(observer_index), thread_coreas_waveforms,
                          thread_counter);
                    if (Zhs)
                      detail::accumulateZHS<KokkosRadioOpenMpThreadLocalOperations>(
                          tracks(track_index), propagation, zhs_observers(observer_index),
                          thread_zhs_waveforms, thread_counter);
                  } else {
                    if (Coreas)
                      detail::accumulateCoREAS<KokkosRadioAtomicOperations>(
                          tracks(track_index), propagation,
                          coreas_observers(observer_index), coreas_waveforms,
                          counters.data());
                    if (Zhs)
                      detail::accumulateZHS<KokkosRadioAtomicOperations>(
                          tracks(track_index), propagation, zhs_observers(observer_index),
                          zhs_waveforms, counters.data());
                  }
                }
              }
              thread_tokens.release(thread_slot);
            });
      } else {
        auto const tile_lanes = track_tile_size * observer_tile_size;
        auto const team_size = std::min(team_size_, tile_lanes);
        auto policy = Policy(execution, league_size, team_size);
        policy.set_scratch_size(0, Kokkos::PerTeam(sizeof(RadioTileScratch)));
        auto const label =
            Coreas && Zhs ? "c8_kokkos_coreas_zhs_tiled"
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
      statistics_.track_observer_pairs += pairs * static_cast<std::size_t>(Coreas + Zhs);
      if constexpr (Coreas && Zhs) statistics_.fused_track_observer_pairs += pairs;
      statistics_.projection_tiles += league_size;
    }

  private:
    template <class FixedView, class FloatView>
    std::vector<gpu::radio::RadioWaveform> downloadSet(
        std::vector<gpu::radio::RadioObserverSnapshot> const& observers,
        std::size_t total_bins, FixedView const& fixed, FloatView const& floating,
        bool zhs_potential) {
      std::vector<gpu::radio::RadioWaveform> result;
      result.reserve(observers.size());
      if (total_bins == 0) return result;
      if (total_bins > std::numeric_limits<std::size_t>::max() / 3)
        throw std::length_error("Kokkos radio download size overflow");
      std::vector<double> host(3 * total_bins);
      if (fixed.extent(0) != 0) {
        auto mirror = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), fixed);
        std::size_t offset = 0;
        for (auto const& observer : observers) {
          auto const count = static_cast<std::size_t>(observer.number_of_bins);
          auto const potential_factor = zhs_potential ? observer.sample_rate_Hz : 1.;
          auto const inverse_scale = config_.fixed_point_field_limit_V_per_m /
                                     (detail::FixedPointHeadroom * potential_factor);
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
        statistics_.device_to_host_bytes += fixed.extent(0) * sizeof(long long);
      } else {
        auto mirror = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), floating);
        for (std::size_t index = 0; index < host.size(); ++index)
          host[index] = mirror(index);
        statistics_.device_to_host_bytes += floating.extent(0) * sizeof(double);
      }
      std::size_t offset = 0;
      for (auto const& observer : observers) {
        auto const count = static_cast<std::size_t>(observer.number_of_bins);
        gpu::radio::RadioWaveform waveform{};
        waveform.x.assign(host.begin() + offset, host.begin() + offset + count);
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
      statistics_.maximum_segment_length_m = counters.maximum_segment_length_m;
      statistics_.weighted_direction_change_rad = counters.weighted_direction_change_rad;
      statistics_.weighted_direction_change_squared_rad2 =
          counters.weighted_direction_change_squared_rad2;
      statistics_.weighted_beta_deficit_track_length_m =
          counters.weighted_beta_deficit_track_length_m;
      statistics_.weighted_time_residual_s = counters.weighted_time_residual_s;
      statistics_.maximum_direction_change_rad = counters.maximum_direction_change_rad;
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
      if (config.zhs_subtrack_refinement < 1 || config.zhs_subtrack_refinement > 64)
        throw std::invalid_argument("Kokkos ZHS subtrack refinement must be in [1,64]");
      if (!std::isfinite(propagation.homogeneous_refractive_index) ||
          propagation.homogeneous_refractive_index < 0.)
        throw std::invalid_argument("invalid Kokkos homogeneous radio refractive index");
      bool const homogeneous = propagation.homogeneous_refractive_index > 0.;
      if (!homogeneous &&
          (propagation.refractivity.size() < 11 ||
           propagation.refractivity.size() !=
               propagation.integrated_refractivity.size() ||
           !std::isfinite(propagation.minimum_height_m) ||
           !std::isfinite(propagation.maximum_height_m) ||
           !(propagation.maximum_height_m > propagation.minimum_height_m) ||
           !std::isfinite(propagation.step_m) || !(propagation.step_m > 0.) ||
           !std::isfinite(propagation.inverse_step_per_m) ||
           !(propagation.inverse_step_per_m > 0.)))
        throw std::invalid_argument("invalid Kokkos flat-atmosphere radio snapshot");
      auto validate_observers =
          [&](std::vector<gpu::radio::RadioObserverSnapshot> const& observers) {
            for (auto const& observer : observers) {
              if (!std::isfinite(observer.start_time_s) ||
                  !std::isfinite(observer.duration_s) || !(observer.duration_s >= 0.) ||
                  !std::isfinite(observer.sample_rate_Hz) ||
                  !(observer.sample_rate_Hz > 0.) || observer.number_of_bins == 0)
                throw std::invalid_argument("invalid Kokkos radio observer");
              auto const height =
                  (observer.position_m[2] - propagation.minimum_height_m) *
                  propagation.inverse_step_per_m;
              if (!homogeneous &&
                  (!std::isfinite(height) || height < 0. ||
                   height + 0.5 >= static_cast<double>(propagation.refractivity.size())))
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
    ThreadFixedWaveformView coreas_thread_fixed_{};
    ThreadFixedWaveformView zhs_thread_fixed_{};
    ThreadCounterView thread_counters_{};
    std::size_t openmp_thread_slots_{};
    detail::DeviceWaveforms coreas_waveforms_{};
    detail::DeviceWaveforms zhs_waveforms_{};
    detail::DevicePropagation propagation_{};
    CounterView counters_{};
    TrackView tracks_{};
    TrackInputView host_track_input_device_{};
    HostTrackInputView host_track_input_{};
    std::size_t host_track_count_{};
    gpu::radio::GpuRadioStatistics statistics_{};
    std::size_t coreas_bins_{};
    std::size_t zhs_bins_{};
    bool fuse_coreas_zhs_{};
    bool initialized_{};
    bool downloaded_{};
    bool fixed_snapshot_sealed_{};
    std::size_t track_tile_size_{TrackTileSize};
    std::size_t observer_tile_size_{ObserverTileSize};
    std::size_t team_size_{IsOpenMP ? 1U : 256U};
  };

} // namespace radio_accumulator_reference

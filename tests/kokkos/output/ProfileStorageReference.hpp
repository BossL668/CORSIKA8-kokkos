/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <tests/kokkos/output/ProfileAccumulationReference.hpp>
#include <Kokkos_Core.hpp>
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
// Frozen production 41cdc543: original storage and merge method bodies.
namespace profile_storage_reference {
  using namespace profile_accumulation_reference;
  inline constexpr std::size_t DeviceProfileHistogramCount = 8;
  struct ProfileProjection {
    bool enabled{};
    /**
     * Keep the projected longitudinal profile and energy-loss histograms
     * resident on the device. When enabled, per-step projected records are
     * not copied to the host; only the final O(number-of-bins) histograms
     * are downloaded at end of shower.
     */
    bool accumulate_on_device{};
    double axis_start_position_m[3]{};
    double axis_direction[3]{};
    double axis_step_length_m{};
    std::vector<double> axis_grammage_g_per_cm2{};
    std::size_t output_bin_count{};
    double output_bin_width_g_per_cm2{};
    ProfileCrossingMode crossing_mode{ProfileCrossingMode::Both};
    double energy_loss_threshold_g_per_cm2{1.e-4};
    /**
     * Expected positive ranges for checked deterministic fixed-point sums.
     * Two integer headroom bits are retained; exceeding the representable
     * range is a hard shower error rather than a silent wraparound.
     */
    double fixed_point_weight_limit{};
    double fixed_point_energy_limit_GeV{};
  };
  struct GpuProfileResult {
    std::vector<double> photons{};
    std::vector<double> electrons{};
    std::vector<double> positrons{};
    std::vector<double> muons_minus{};
    std::vector<double> muons_plus{};
    /**
     * Muons produced by a muon parent at accepted device-resident discrete
     * interaction vertices.  This is the accelerator counterpart of
     * ProductionProfile::doSecondaries(), not a transported-particle count.
     */
    std::vector<double> muon_parent_productions{};
    std::vector<double> energy_loss_GeV{};
    std::vector<double> muon_energy_loss_GeV{};
    std::uint64_t steps{};
    std::uint64_t deposited_steps{};
    std::uint64_t particle_cuts{};
    std::uint64_t fixed_point_overflows{};
    std::uint64_t invalid_records{};
    double weighted_deposited_energy_GeV{};
    /**
     * Deterministic energy-ledger terms accumulated from the same resident
     * transport/final-state records as the longitudinal profile.
     *
     * Total particle energies include rest mass.  Atomic-electron processes
     * (Compton, photoelectric, ionization and positron annihilation) therefore
     * import one electron rest mass from the medium.  ParticleCut deposits only
     * kinetic energy, so a terminated electron/positron rest mass is retained as
     * an explicit terminal sink instead of being silently hidden.
     */
    double weighted_medium_rest_mass_input_GeV{};
    double weighted_cut_rest_mass_energy_GeV{};
    double weighted_observed_total_energy_GeV{};
    double weighted_escaped_total_energy_GeV{};
    // PROPOSAL's scalar writer does not emit the photoelectric binding loss.
    // Keep it outside legacy dE/dX, but explicitly account for it in closure.
    double weighted_unwritten_photoelectric_binding_energy_GeV{};
    // ObservationPlane precedes ParticleCut; their terminal records can overlap.
    double weighted_observation_cut_overlap_energy_GeV{};
    // Signed numerical convention change from PROPOSAL total energy to the
    // CORSIKA kinetic-energy stack; not a physical interaction with the medium.
    double weighted_mass_convention_correction_GeV{};
    // Signed ParticleCut bookkeeping for tagged sub-rest-mass BKK children.
    double weighted_native_muon_pair_cut_energy_GeV{};
  };
  enum class CooperativeEndpoint : unsigned { Cuda, OpenMP };

  // SharedHostPinnedSpace follows the package's default device. In a dual
  // package OpenMP must still use ordinary RAM, not CUDA-pinned staging memory.
  template <class ExecutionSpace>
  using HostStagingSpace = std::conditional_t<
      std::is_same_v<typename ExecutionSpace::memory_space, Kokkos::HostSpace>,
      Kokkos::HostSpace, Kokkos::SharedHostPinnedSpace>;

  inline std::size_t checkedMemoryAdd(std::size_t const left, std::size_t const right) {
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
        throw std::logic_error("Kokkos memory projection exceeds current retained bytes");
      peak_ = std::max(peak_, checkedMemoryAdd(retained_, new_bytes));
      retained_ = checkedMemoryAdd(retained_ - old_bytes, new_bytes);
    }

    void merge(std::size_t const old_bytes, KokkosMemoryProjection const& nested) {
      if (old_bytes > retained_)
        throw std::logic_error("Kokkos nested memory projection exceeds retained bytes");
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
      peak_ = std::max(peak_, checkedMemoryAdd(retained_, projected.transient_peak_bytes -
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

  struct FixedProfileSnapshot {
    std::string shower_identity;
    CooperativeEndpoint endpoint{};
    ProfileProjection config;
    double weight_units{}, energy_units{};
    std::vector<std::int64_t> histograms;
    DeviceProfileCounters counters{};
  };
  inline bool sameProfileConfig(ProfileProjection const& a, ProfileProjection const& b) {
#define SAME(f) \
  if (a.f != b.f) return false
    SAME(enabled);
    SAME(accumulate_on_device);
    SAME(axis_step_length_m);
    SAME(axis_grammage_g_per_cm2);
    SAME(output_bin_count);
    SAME(output_bin_width_g_per_cm2);
    SAME(energy_loss_threshold_g_per_cm2);
    SAME(fixed_point_weight_limit);
    SAME(fixed_point_energy_limit_GeV);
    SAME(crossing_mode);
    for (int d = 0; d < 3; ++d) {
      SAME(axis_start_position_m[d]);
      SAME(axis_direction[d]);
    }
#undef SAME
    return true;
  }
  inline void validateFixedProfile(FixedProfileSnapshot const& s) {
    if (s.counters.fixed_point_overflows || s.counters.invalid_records)
      throw std::runtime_error(
          "cooperative profile rejected: overflows=" +
          std::to_string(s.counters.fixed_point_overflows) +
          ", invalid_records=" + std::to_string(s.counters.invalid_records));
    if (s.shower_identity.empty() || static_cast<unsigned>(s.endpoint) > 1 ||
        !s.config.enabled || !s.config.accumulate_on_device ||
        !s.config.output_bin_count ||
        s.config.output_bin_count > std::numeric_limits<std::size_t>::max() / 8 ||
        s.histograms.size() != 8 * s.config.output_bin_count ||
        !std::isfinite(s.weight_units) || s.weight_units <= 0. ||
        !std::isfinite(s.energy_units) || s.energy_units <= 0.)
      throw std::invalid_argument("invalid/overflowed cooperative profile snapshot");
    if (s.weight_units != 1. / (0x1p62 / s.config.fixed_point_weight_limit) ||
        s.energy_units != 1. / (0x1p62 / s.config.fixed_point_energy_limit_GeV))
      throw std::invalid_argument("cooperative profile scale mismatch");
  }
  inline DeviceProfileCounters mergeProfileCounters(DeviceProfileCounters a,
                                                    DeviceProfileCounters const& b) {
    auto add = [](auto x, auto y) {
      using T = decltype(x);
      if constexpr (std::is_signed_v<T>) {
        if ((y > 0 && x > std::numeric_limits<T>::max() - y) ||
            (y < 0 && x < std::numeric_limits<T>::min() - y))
          throw std::overflow_error("cooperative profile energy counter overflow");
      } else if (y > std::numeric_limits<T>::max() - x)
        throw std::overflow_error("cooperative profile counter overflow");
      return x + y;
    };
#define ADD(f) a.f = add(a.f, b.f)
    ADD(steps);
    ADD(deposited_steps);
    ADD(photon_cuts);
    for (unsigned i = 0; i < 8; ++i) ADD(lepton_limits[i]);
    ADD(moliere_trials);
    ADD(moliere_deflections);
    ADD(moliere_zero_deflections);
    ADD(moliere_newton_iterations);
    ADD(thinning_hillas_vertices);
    ADD(thinning_statistical_vertices);
    ADD(thinning_particles_discarded);
    ADD(fixed_point_overflows);
    ADD(invalid_records);
    ADD(weighted_medium_rest_mass_input);
    ADD(weighted_cut_rest_mass_energy);
    ADD(weighted_observed_total_energy);
    ADD(weighted_escaped_total_energy);
    ADD(weighted_unwritten_photoelectric_binding_energy);
    ADD(weighted_observation_cut_overlap_energy);
    ADD(weighted_mass_convention_correction);
    ADD(weighted_native_muon_pair_cut_energy);
#undef ADD
    a.moliere_max_newton_iterations =
        std::max(a.moliere_max_newton_iterations, b.moliere_max_newton_iterations);
    return a;
  }
  inline GpuProfileResult decodeFixedProfile(FixedProfileSnapshot const& s) {
    validateFixedProfile(s);
    auto bins_ = s.config.output_bin_count;
    DeviceProfileAccumulator view_{};
    view_.inverse_weight_scale = s.weight_units;
    view_.inverse_energy_scale = s.energy_units;
    auto host_histograms_ = [&](std::size_t i) { return s.histograms.at(i); };
    auto const& counters = s.counters;
    GpuProfileResult result{};
    result.photons.resize(bins_);
    result.electrons.resize(bins_);
    result.positrons.resize(bins_);
    result.muons_minus.resize(bins_);
    result.muons_plus.resize(bins_);
    result.muon_parent_productions.resize(bins_);
    result.energy_loss_GeV.resize(bins_);
    result.muon_energy_loss_GeV.resize(bins_);
    for (std::size_t bin = 0; bin < bins_; ++bin) {
      result.photons[bin] =
          static_cast<double>(host_histograms_(bin)) * view_.inverse_weight_scale;
      result.electrons[bin] =
          static_cast<double>(host_histograms_(bins_ + bin)) * view_.inverse_weight_scale;
      result.positrons[bin] = static_cast<double>(host_histograms_(2 * bins_ + bin)) *
                              view_.inverse_weight_scale;
      result.muons_minus[bin] = static_cast<double>(host_histograms_(3 * bins_ + bin)) *
                                view_.inverse_weight_scale;
      result.muons_plus[bin] = static_cast<double>(host_histograms_(4 * bins_ + bin)) *
                               view_.inverse_weight_scale;
      result.muon_parent_productions[bin] =
          static_cast<double>(host_histograms_(5 * bins_ + bin)) *
          view_.inverse_weight_scale;
      auto const electromagnetic_loss =
          static_cast<double>(host_histograms_(6 * bins_ + bin)) *
          view_.inverse_energy_scale;
      result.muon_energy_loss_GeV[bin] =
          static_cast<double>(host_histograms_(7 * bins_ + bin)) *
          view_.inverse_energy_scale;
      result.energy_loss_GeV[bin] =
          electromagnetic_loss + result.muon_energy_loss_GeV[bin];
      result.weighted_deposited_energy_GeV += result.energy_loss_GeV[bin];
    }
    result.steps = counters.steps;
    result.deposited_steps = counters.deposited_steps;
    if (counters
            .lepton_limits[static_cast<std::size_t>(LeptonTransportLimit::ParticleCut)] >
        std::numeric_limits<unsigned long long>::max() - counters.photon_cuts)
      throw std::overflow_error("cooperative decoded particle cut counter overflow");
    result.particle_cuts =
        counters.photon_cuts +
        counters
            .lepton_limits[static_cast<std::size_t>(LeptonTransportLimit::ParticleCut)];
    result.fixed_point_overflows = counters.fixed_point_overflows;
    result.invalid_records = counters.invalid_records;
    result.weighted_medium_rest_mass_input_GeV =
        static_cast<double>(counters.weighted_medium_rest_mass_input) *
        view_.inverse_energy_scale;
    result.weighted_cut_rest_mass_energy_GeV =
        static_cast<double>(counters.weighted_cut_rest_mass_energy) *
        view_.inverse_energy_scale;
    result.weighted_observed_total_energy_GeV =
        static_cast<double>(counters.weighted_observed_total_energy) *
        view_.inverse_energy_scale;
    result.weighted_escaped_total_energy_GeV =
        static_cast<double>(counters.weighted_escaped_total_energy) *
        view_.inverse_energy_scale;
    result.weighted_unwritten_photoelectric_binding_energy_GeV =
        static_cast<double>(counters.weighted_unwritten_photoelectric_binding_energy) *
        view_.inverse_energy_scale;
    result.weighted_observation_cut_overlap_energy_GeV =
        static_cast<double>(counters.weighted_observation_cut_overlap_energy) *
        view_.inverse_energy_scale;
    result.weighted_mass_convention_correction_GeV =
        static_cast<double>(counters.weighted_mass_convention_correction) *
        view_.inverse_energy_scale;
    result.weighted_native_muon_pair_cut_energy_GeV =
        static_cast<double>(counters.weighted_native_muon_pair_cut_energy) *
        view_.inverse_energy_scale;

    return result;
  }
  class CooperativeProfileMerge {
  public:
    CooperativeProfileMerge(std::string identity, ProfileProjection config)
        : identity_(std::move(identity))
        , config_(std::move(config)) {}
    void commit(FixedProfileSnapshot const& s) {
      if (finished_) throw std::logic_error("cooperative profile already taken");
      validateFixedProfile(s);
      auto bit = 1u << static_cast<unsigned>(s.endpoint);
      if ((seen_ & bit) || s.shower_identity != identity_ ||
          !sameProfileConfig(s.config, config_))
        throw std::invalid_argument("duplicate/incompatible cooperative profile commit");
      auto next = seen_ ? merged_ : s;
      if (seen_) {
        if (next.weight_units != s.weight_units || next.energy_units != s.energy_units)
          throw std::invalid_argument("cooperative profile scales differ");
        FixedAccumulatorLayout layout{identity_ + "/profile-eight-columns-v1",
                                      s.weight_units, s.histograms.size()};
        mergeFixedAccumulators(next.histograms, layout, s.histograms, layout);
        next.counters = mergeProfileCounters(next.counters, s.counters);
      }
      merged_ = std::move(next);
      seen_ |= bit;
    }
    FixedProfileSnapshot take() {
      if (seen_ != 3 || finished_)
        throw std::logic_error("profile endpoints not complete/already taken");
      finished_ = true;
      return std::move(merged_);
    }

  private:
    std::string identity_;
    ProfileProjection config_;
    FixedProfileSnapshot merged_;
    unsigned seen_{};
    bool finished_{};
  };

  /** Bounded host-only replicas of the existing integer profile ledger.
   *
   * No formula, particle ordering or scale changes. OpenMP workers select a
   * replica; the coordinator combines integers exactly once after all work has
   * finished. Replica collisions remain safe because the original checked
   * atomic operations are used. This is deliberately NOT a per-thread copy of
   * transport workspaces, tables or radio buffers.
   */
  class HostProfileShards {
  public:
    using Accumulator = DeviceProfileAccumulator;
    using Counters = DeviceProfileCounters;
    struct alignas(64) Entry {
      Counters counters{};
      Accumulator accumulator{};
    };
    static constexpr std::size_t MaximumBytes = 16 * 1024 * 1024;
    static constexpr std::size_t MaximumShards = 256;

    static std::size_t bytesPerShard(std::size_t bins) {
      constexpr auto columns = DeviceProfileHistogramCount;
      if (bins == 0 || bins > (std::numeric_limits<std::size_t>::max() - sizeof(Entry)) /
                                  columns / sizeof(long long))
        throw std::length_error("host profile shard size overflow/empty bins");
      return bins * columns * sizeof(long long) + sizeof(Entry);
    }

    void configure(Accumulator const& base, std::size_t workers,
                   std::size_t available_bytes) {
      if (configured_) throw std::logic_error("host profile shards already configured");
      configured_ = true;
      auto const per_shard = bytesPerShard(base.bins);
      auto const allowed = std::min(available_bytes, MaximumBytes) / per_shard;
      // Do not round down to a power of two: e.g. 130 workers sharing 128
      // shards leave two contended pairs whose reduction barrier stalls all
      // workers, even though the budget can easily accommodate 130 replicas.
      auto const count = std::min({workers, allowed, MaximumShards});
      // The canonical accumulator already covers the one-shard case.
      if (workers < 2 || count < 2 || allowed < 2) return;
      count_ = count;
      bins_ = base.bins;
      values_per_shard_ = bins_ * DeviceProfileHistogramCount;
      histograms_ = std::make_unique<long long[]>(count_ * values_per_shard_);
      entries_ = std::make_unique<Entry[]>(count_);
      bytes_ = count_ * per_shard;
      reset(base);
    }

    // Called at beginShower, only with an idle endpoint. Allocation is reused;
    // every bin/counter and the event-specific fixed-point scales are reset.
    void reset(Accumulator const& base) {
      if (!enabled()) return;
      if (base.bins != bins_ || !base.counters ||
          !std::isfinite(base.inverse_weight_scale) ||
          !std::isfinite(base.inverse_energy_scale) || base.inverse_weight_scale <= 0. ||
          base.inverse_energy_scale <= 0.)
        throw std::invalid_argument("host profile shard layout/scale mismatch");
      std::fill_n(histograms_.get(), count_ * values_per_shard_, 0LL);
      base_ = base;
      for (std::size_t i = 0; i < count_; ++i) {
        auto& entry = entries_[i];
        entry.counters = {};
        entry.accumulator = base;
        auto& a = entry.accumulator;
        auto* h = histograms_.get() + i * values_per_shard_;
        a.photons = h;
        a.electrons = h + bins_;
        a.positrons = h + 2 * bins_;
        a.muons_minus = h + 3 * bins_;
        a.muons_plus = h + 4 * bins_;
        a.muon_parent_productions = h + 5 * bins_;
        a.energy_loss = h + 6 * bins_;
        a.muon_energy_loss = h + 7 * bins_;
        a.counters = &entry.counters;
      }
      sealed_ = false;
    }

    bool enabled() const noexcept { return count_ != 0; }
    std::size_t count() const noexcept { return count_; }
    std::size_t bytes() const noexcept { return bytes_; }
    Entry const* entries() const noexcept { return entries_.get(); }

    // Transactional: no partial integer output is published on overflow or a
    // malformed shard. A failed finalization is sealed, not retried as new data.
    void mergeInto(FixedProfileSnapshot& canonical) {
      if (!enabled()) return;
      if (sealed_) throw std::logic_error("host profile shards already committed");
      sealed_ = true;
      validateFixedProfile(canonical);
      if (canonical.config.output_bin_count != bins_ ||
          canonical.config.crossing_mode != base_.crossing_mode ||
          canonical.config.output_bin_width_g_per_cm2 != base_.bin_width_g_per_cm2 ||
          canonical.config.energy_loss_threshold_g_per_cm2 !=
              base_.energy_loss_threshold_g_per_cm2 ||
          canonical.weight_units != base_.inverse_weight_scale ||
          canonical.energy_units != base_.inverse_energy_scale)
        throw std::invalid_argument("host profile shard snapshot mismatch");
      auto merged = canonical;
      std::vector<std::int64_t> row(values_per_shard_);
      FixedAccumulatorLayout layout{canonical.shower_identity + "/host-profile-shards",
                                    canonical.weight_units, values_per_shard_};
      for (std::size_t i = 0; i < count_; ++i) {
        auto const& entry = entries_[i];
        if (entry.counters.fixed_point_overflows || entry.counters.invalid_records)
          throw std::runtime_error("host profile shard invalid/overflowed");
        std::copy_n(histograms_.get() + i * values_per_shard_, values_per_shard_,
                    row.begin());
        mergeFixedAccumulators(merged.histograms, layout, row, layout);
        merged.counters = mergeProfileCounters(merged.counters, entry.counters);
      }
      validateFixedProfile(merged);
      canonical = std::move(merged);
    }

  private:
    std::unique_ptr<long long[]> histograms_;
    std::unique_ptr<Entry[]> entries_;
    Accumulator base_{};
    std::size_t count_{}, bins_{}, values_per_shard_{}, bytes_{};
    bool configured_{}, sealed_{};
  };

  /** Own the shower-axis support in the selected Kokkos memory space. */
  template <class ExecutionSpace>
  class KokkosProfileProjection {
  public:
    using memory_space = typename ExecutionSpace::memory_space;

    static std::size_t projectedDeviceBytes(ProfileProjection const& source) {
      if (!source.enabled) return 0;
      if (!(source.axis_step_length_m > 0.) || source.axis_grammage_g_per_cm2.size() < 2)
        throw std::invalid_argument(
            "Kokkos profile projection requires a valid ShowerAxis support");
      return checkedMemoryMultiply(source.axis_grammage_g_per_cm2.size(), sizeof(double));
    }

    void initialize(ProfileProjection const& source,
                    ExecutionSpace const& execution = {}) {
      if (!source.enabled) return;
      (void)projectedDeviceBytes(source);
      axis_ = Kokkos::View<double*, memory_space>("c8_kokkos_profile_axis",
                                                  source.axis_grammage_g_per_cm2.size());
      auto host = Kokkos::create_mirror_view(axis_);
      for (std::size_t i = 0; i < source.axis_grammage_g_per_cm2.size(); ++i)
        host(i) = source.axis_grammage_g_per_cm2[i];
      Kokkos::deep_copy(execution, axis_, host);
      for (int component = 0; component < 3; ++component) {
        view_.axis_start_position_m[component] = source.axis_start_position_m[component];
        view_.axis_direction[component] = source.axis_direction[component];
      }
      view_.axis_step_length_m = source.axis_step_length_m;
      view_.axis_grammage_g_per_cm2 = axis_.data();
      view_.axis_support_count = source.axis_grammage_g_per_cm2.size();
      enabled_ = true;
    }

    bool enabled() const noexcept { return enabled_; }
    DeviceProfileProjection deviceView() const noexcept { return view_; }
    std::size_t deviceBytes() const noexcept { return axis_.extent(0) * sizeof(double); }

  private:
    Kokkos::View<double*, memory_space> axis_{};
    DeviceProfileProjection view_{};
    bool enabled_{};
  };

  template <class ExecutionSpace>
  class KokkosProfileAccumulator {
  public:
    using memory_space = typename ExecutionSpace::memory_space;
    using host_staging_space = HostStagingSpace<ExecutionSpace>;
    using PhotonStepView = Kokkos::View<PhotonTransportRecord*, memory_space>;
    using PhotonFinalView = Kokkos::View<PhotonPairFinalStateRecord*, memory_space>;
    using LeptonStepView = Kokkos::View<LeptonTransportRecord*, memory_space>;
    using LeptonFinalView = Kokkos::View<BremsFinalStateRecord*, memory_space>;

    static std::size_t projectedDeviceBytes(ProfileProjection const& config) {
      if (!config.enabled || !config.accumulate_on_device) return 0;
      if (config.output_bin_count == 0 || !(config.output_bin_width_g_per_cm2 > 0.) ||
          !(config.energy_loss_threshold_g_per_cm2 >= 0.) ||
          !(config.fixed_point_weight_limit > 0.) ||
          !(config.fixed_point_energy_limit_GeV > 0.))
        throw std::invalid_argument("Kokkos resident profile configuration is invalid");
      auto const histograms =
          checkedMemoryMultiply(DeviceProfileHistogramCount, config.output_bin_count);
      return checkedMemoryAdd(checkedMemoryMultiply(histograms, sizeof(long long)),
                              sizeof(DeviceProfileCounters));
    }

    void initialize(ProfileProjection const& config,
                    ExecutionSpace const& execution = {}) {
      if (!config.enabled || !config.accumulate_on_device) return;
      (void)projectedDeviceBytes(config);
      fixed_snapshot_config_ = config;
      bins_ = config.output_bin_count;
      if (bins_ > std::numeric_limits<std::size_t>::max() / DeviceProfileHistogramCount)
        throw std::length_error("Kokkos resident profile histogram size overflow");
      auto const histogram_values = DeviceProfileHistogramCount * bins_;
      histograms_ = Kokkos::View<long long*, memory_space>(
          Kokkos::view_alloc(execution, Kokkos::WithoutInitializing,
                             "c8_kokkos_profile_histograms"),
          histogram_values);
      counters_ = Kokkos::View<DeviceProfileCounters*, memory_space>(
          Kokkos::view_alloc(execution, Kokkos::WithoutInitializing,
                             "c8_kokkos_profile_counters"),
          1);
      host_histograms_ = Kokkos::View<long long*, host_staging_space>(
          Kokkos::view_alloc(Kokkos::WithoutInitializing,
                             "c8_kokkos_profile_host_histograms"),
          histogram_values);
      host_counters_ = Kokkos::View<DeviceProfileCounters*, host_staging_space>(
          Kokkos::view_alloc(Kokkos::WithoutInitializing,
                             "c8_kokkos_profile_host_counters"),
          1);
      view_.photons = histograms_.data();
      view_.electrons = view_.photons + bins_;
      view_.positrons = view_.electrons + bins_;
      view_.muons_minus = view_.positrons + bins_;
      view_.muons_plus = view_.muons_minus + bins_;
      view_.muon_parent_productions = view_.muons_plus + bins_;
      view_.energy_loss = view_.muon_parent_productions + bins_;
      view_.muon_energy_loss = view_.energy_loss + bins_;
      view_.counters = counters_.data();
      view_.bins = bins_;
      view_.bin_width_g_per_cm2 = config.output_bin_width_g_per_cm2;
      view_.crossing_mode = config.crossing_mode;
      view_.energy_loss_threshold_g_per_cm2 = config.energy_loss_threshold_g_per_cm2;
      enabled_ = true;
      reset(config.fixed_point_weight_limit, config.fixed_point_energy_limit_GeV,
            execution);
    }

    void reset(double const weight_limit, double const energy_limit_GeV,
               ExecutionSpace const& execution = {}) {
      if (!enabled_) return;
      if (!(weight_limit > 0.) || !(energy_limit_GeV > 0.))
        throw std::invalid_argument(
            "Kokkos resident profile fixed-point limits must be positive");
      constexpr double FixedPointHeadroom = 0x1p62;
      view_.weight_scale = FixedPointHeadroom / weight_limit;
      view_.inverse_weight_scale = 1. / view_.weight_scale;
      view_.energy_scale = FixedPointHeadroom / energy_limit_GeV;
      view_.inverse_energy_scale = 1. / view_.energy_scale;
      fixed_snapshot_config_.fixed_point_weight_limit = weight_limit;
      fixed_snapshot_config_.fixed_point_energy_limit_GeV = energy_limit_GeV;
      Kokkos::deep_copy(execution, histograms_, 0LL);
      Kokkos::deep_copy(execution, counters_, DeviceProfileCounters{});
      execution.fence("reset Kokkos resident profile");
      host_shards_.reset(view_);
      downloaded_ = false;
      fixed_snapshot_sealed_ = false;
    }

    bool enabled() const noexcept { return enabled_; }
    void enableHostShards(std::size_t workers, std::size_t budget) {
#ifdef KOKKOS_ENABLE_OPENMP
      if constexpr (std::is_same_v<ExecutionSpace, Kokkos::OpenMP>) {
        if (!enabled_ || downloaded_ || fixed_snapshot_sealed_)
          throw std::logic_error("host profile shards require a fresh profile");
        host_shards_.configure(view_, workers, budget);
        return;
      }
#endif
      throw std::logic_error("profile shards require the OpenMP execution space");
    }
    // Keep the cooperative endpoint's existing interface and opt-in unchanged.
    void enableCooperativeHostShards(std::size_t workers, std::size_t budget) {
      enableHostShards(workers, budget);
    }
    HostProfileShards const& hostShards() const noexcept { return host_shards_; }
    void requireOpenForAccumulation() const {
      if (fixed_snapshot_sealed_)
        throw std::logic_error(
            "profile endpoint already committed; reset before new tracks");
    }
    DeviceProfileAccumulator deviceView() const noexcept { return view_; }
    std::size_t deviceBytes() const noexcept {
      return histograms_.extent(0) * sizeof(long long) +
             counters_.extent(0) * sizeof(DeviceProfileCounters) + host_shards_.bytes();
    }

    void accumulatePhoton(DeviceProfileProjection const projection,
                          PhotonStepView const& steps, std::size_t const step_count,
                          PhotonFinalView const& final_states,
                          std::size_t const final_state_count,
                          ExecutionSpace const& execution = {}) {
      if (!enabled_) return;
      requireOpenForAccumulation();
      auto const accumulator = view_;
      Kokkos::parallel_for(
          "c8_kokkos_accumulate_photon_profile_steps",
          Kokkos::RangePolicy<ExecutionSpace>(execution, 0, step_count),
          KOKKOS_LAMBDA(std::size_t const index) {
            accumulatePhotonProfileStep<KokkosProfileAtomicOperations>(
                projection, accumulator, steps(index));
          });
      Kokkos::parallel_for(
          "c8_kokkos_accumulate_photon_profile_final_states",
          Kokkos::RangePolicy<ExecutionSpace>(execution, 0, final_state_count),
          KOKKOS_LAMBDA(std::size_t const index) {
            accumulatePhotonProfileFinalState<KokkosProfileAtomicOperations>(
                projection, accumulator, steps.data(), step_count, final_states(index));
          });
    }

    void accumulateLepton(DeviceProfileProjection const projection,
                          LeptonStepView const& steps, std::size_t const step_count,
                          LeptonFinalView const& final_states,
                          std::size_t const final_state_count,
                          ExecutionSpace const& execution = {}) {
      if (!enabled_) return;
      requireOpenForAccumulation();
      auto const accumulator = view_;
      Kokkos::parallel_for(
          "c8_kokkos_accumulate_lepton_profile_steps",
          Kokkos::RangePolicy<ExecutionSpace>(execution, 0, step_count),
          KOKKOS_LAMBDA(std::size_t const index) {
            accumulateLeptonProfileStep<KokkosProfileAtomicOperations>(
                projection, accumulator, steps(index));
          });
      Kokkos::parallel_for(
          "c8_kokkos_accumulate_lepton_profile_final_states",
          Kokkos::RangePolicy<ExecutionSpace>(execution, 0, final_state_count),
          KOKKOS_LAMBDA(std::size_t const index) {
            accumulateLeptonProfileFinalState<KokkosProfileAtomicOperations>(
                projection, accumulator, steps.data(), step_count, final_states(index));
          });
    }

    FixedProfileSnapshot downloadFixed(std::string const& identity,
                                       CooperativeEndpoint endpoint,
                                       ExecutionSpace const& execution = {}) {
      if (!enabled_ || downloaded_)
        throw std::logic_error("fixed profile snapshot unavailable/already committed");
      if (identity.empty() || static_cast<unsigned>(endpoint) > 1)
        throw std::invalid_argument("invalid cooperative profile snapshot identity");
      // Final export: a failed copy must not be accepted as a fresh ledger.
      downloaded_ = true;
      fixed_snapshot_sealed_ = true;
      try {
        Kokkos::deep_copy(execution, host_histograms_, histograms_);
        Kokkos::deep_copy(execution, host_counters_, counters_);
        execution.fence("download cooperative fixed profile endpoint");
      } catch (...) {
        try {
          execution.fence("unwind cooperative profile endpoint copy");
        } catch (...) {}
        throw;
      }
      FixedProfileSnapshot out;
      out.shower_identity = identity;
      out.endpoint = endpoint;
      out.config = fixed_snapshot_config_;
      out.weight_units = view_.inverse_weight_scale;
      out.energy_units = view_.inverse_energy_scale;
      out.counters = host_counters_(0);
      out.histograms.resize(histograms_.extent(0));
      for (std::size_t i = 0; i < out.histograms.size(); ++i)
        out.histograms[i] = host_histograms_(i);
      validateFixedProfile(out);
      host_shards_.mergeInto(out);
      return out;
    }

    GpuProfileResult download(ExecutionSpace const& execution = {}) {
      if (!enabled_) throw std::logic_error("Kokkos resident profile is not enabled");
      if (host_shards_.enabled()) {
        // Standalone OpenMP uses the same checked integer merge as cooperative
        // transport. Decode only AFTER the canonical and sharded ledgers have
        // been combined; never add separately rounded floating-point profiles.
        // downloadFixed seals the event even if finalization fails.
        return decodeFixedProfile(downloadFixed("standalone-openmp-profile",
                                                CooperativeEndpoint::OpenMP, execution));
      }
      if (downloaded_)
        throw std::logic_error("Kokkos resident profile was downloaded more than once");
      Kokkos::deep_copy(execution, host_histograms_, histograms_);
      Kokkos::deep_copy(execution, host_counters_, counters_);
      execution.fence("download Kokkos resident profile");
      auto const& counters = host_counters_(0);
      GpuProfileResult result{};
      result.photons.resize(bins_);
      result.electrons.resize(bins_);
      result.positrons.resize(bins_);
      result.muons_minus.resize(bins_);
      result.muons_plus.resize(bins_);
      result.muon_parent_productions.resize(bins_);
      result.energy_loss_GeV.resize(bins_);
      result.muon_energy_loss_GeV.resize(bins_);
      for (std::size_t bin = 0; bin < bins_; ++bin) {
        result.photons[bin] =
            static_cast<double>(host_histograms_(bin)) * view_.inverse_weight_scale;
        result.electrons[bin] = static_cast<double>(host_histograms_(bins_ + bin)) *
                                view_.inverse_weight_scale;
        result.positrons[bin] = static_cast<double>(host_histograms_(2 * bins_ + bin)) *
                                view_.inverse_weight_scale;
        result.muons_minus[bin] = static_cast<double>(host_histograms_(3 * bins_ + bin)) *
                                  view_.inverse_weight_scale;
        result.muons_plus[bin] = static_cast<double>(host_histograms_(4 * bins_ + bin)) *
                                 view_.inverse_weight_scale;
        result.muon_parent_productions[bin] =
            static_cast<double>(host_histograms_(5 * bins_ + bin)) *
            view_.inverse_weight_scale;
        auto const electromagnetic_loss =
            static_cast<double>(host_histograms_(6 * bins_ + bin)) *
            view_.inverse_energy_scale;
        result.muon_energy_loss_GeV[bin] =
            static_cast<double>(host_histograms_(7 * bins_ + bin)) *
            view_.inverse_energy_scale;
        result.energy_loss_GeV[bin] =
            electromagnetic_loss + result.muon_energy_loss_GeV[bin];
        result.weighted_deposited_energy_GeV += result.energy_loss_GeV[bin];
      }
      result.steps = counters.steps;
      result.deposited_steps = counters.deposited_steps;
      result.particle_cuts =
          counters.photon_cuts +
          counters
              .lepton_limits[static_cast<std::size_t>(LeptonTransportLimit::ParticleCut)];
      result.fixed_point_overflows = counters.fixed_point_overflows;
      result.invalid_records = counters.invalid_records;
      result.weighted_medium_rest_mass_input_GeV =
          static_cast<double>(counters.weighted_medium_rest_mass_input) *
          view_.inverse_energy_scale;
      result.weighted_cut_rest_mass_energy_GeV =
          static_cast<double>(counters.weighted_cut_rest_mass_energy) *
          view_.inverse_energy_scale;
      result.weighted_observed_total_energy_GeV =
          static_cast<double>(counters.weighted_observed_total_energy) *
          view_.inverse_energy_scale;
      result.weighted_escaped_total_energy_GeV =
          static_cast<double>(counters.weighted_escaped_total_energy) *
          view_.inverse_energy_scale;
      result.weighted_unwritten_photoelectric_binding_energy_GeV =
          static_cast<double>(counters.weighted_unwritten_photoelectric_binding_energy) *
          view_.inverse_energy_scale;
      result.weighted_observation_cut_overlap_energy_GeV =
          static_cast<double>(counters.weighted_observation_cut_overlap_energy) *
          view_.inverse_energy_scale;
      result.weighted_mass_convention_correction_GeV =
          static_cast<double>(counters.weighted_mass_convention_correction) *
          view_.inverse_energy_scale;
      result.weighted_native_muon_pair_cut_energy_GeV =
          static_cast<double>(counters.weighted_native_muon_pair_cut_energy) *
          view_.inverse_energy_scale;
      downloaded_ = true;
      return result;
    }

  private:
    HostProfileShards host_shards_;
    ProfileProjection fixed_snapshot_config_{};
    Kokkos::View<long long*, memory_space> histograms_{};
    Kokkos::View<DeviceProfileCounters*, memory_space> counters_{};
    Kokkos::View<long long*, host_staging_space> host_histograms_{};
    Kokkos::View<DeviceProfileCounters*, host_staging_space> host_counters_{};
    DeviceProfileAccumulator view_{};
    std::size_t bins_{};
    bool enabled_{};
    bool downloaded_{};
    bool fixed_snapshot_sealed_{};
  };
} // namespace profile_storage_reference

/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <Kokkos_Core.hpp>
#include <corsika/accelerator/em/kokkos/KokkosMemorySpace.hpp>

#include <cstddef>
#include <limits>
#include <stdexcept>

#include <corsika/accelerator/em/detail/ProfileAccumulationStep.hpp>
#include <corsika/accelerator/em/kokkos/KokkosResidentMemoryBudget.hpp>

namespace corsika::accelerator::em::kokkos_detail {

  struct KokkosProfileAtomicOperations {
    KOKKOS_INLINE_FUNCTION static long long add(
        long long* address, long long value) {
#if defined(KOKKOS_ENABLE_CUDA) && defined(__CUDA_ARCH__)
      // CUDA has a native unsigned 64-bit atomic add, while the signed
      // overload selected through Kokkos/desul is implemented as a CAS retry
      // loop.  Two's-complement signed addition has the same bit result as
      // unsigned addition, and the returned pre-add bit pattern preserves the
      // overflow check in checkedProfileAtomicAdd().  This deliberately
      // mirrors the established native-CUDA profile projection path.
      auto const previous = ::atomicAdd(
          reinterpret_cast<unsigned long long*>(address),
          static_cast<unsigned long long>(value));
      return static_cast<long long>(previous);
#else
      return Kokkos::atomic_fetch_add(address, value);
#endif
    }

    KOKKOS_INLINE_FUNCTION static unsigned long long add(
        unsigned long long* address, unsigned long long value) {
#if defined(KOKKOS_ENABLE_CUDA) && defined(__CUDA_ARCH__)
      return ::atomicAdd(address, value);
#else
      return Kokkos::atomic_fetch_add(address, value);
#endif
    }

    KOKKOS_INLINE_FUNCTION static void maximum(
        unsigned long long* address, unsigned long long value) {
#if defined(KOKKOS_ENABLE_CUDA) && defined(__CUDA_ARCH__)
      ::atomicMax(address, value);
#else
      Kokkos::atomic_max(address, value);
#endif
    }
  };

  template <class ExecutionSpace>
  class KokkosProfileAccumulator {
  public:
    using memory_space = typename ExecutionSpace::memory_space;
    using host_staging_space = HostStagingSpace<ExecutionSpace>;
    using PhotonStepView =
        Kokkos::View<gpu::em::PhotonTransportRecord*, memory_space>;
    using PhotonFinalView =
        Kokkos::View<gpu::em::PhotonPairFinalStateRecord*, memory_space>;
    using LeptonStepView =
        Kokkos::View<gpu::em::LeptonTransportRecord*, memory_space>;
    using LeptonFinalView =
        Kokkos::View<gpu::em::BremsFinalStateRecord*, memory_space>;

    static std::size_t projectedDeviceBytes(
        gpu::em::GpuEmConfig::ProfileProjection const& config) {
      if (!config.enabled || !config.accumulate_on_device) return 0;
      if (config.output_bin_count == 0 ||
          !(config.output_bin_width_g_per_cm2 > 0.) ||
          !(config.energy_loss_threshold_g_per_cm2 >= 0.) ||
          !(config.fixed_point_weight_limit > 0.) ||
          !(config.fixed_point_energy_limit_GeV > 0.))
        throw std::invalid_argument(
            "Kokkos resident profile configuration is invalid");
      auto const histograms = checkedMemoryMultiply(
          gpu::em::detail::DeviceProfileHistogramCount,
          config.output_bin_count);
      return checkedMemoryAdd(
          checkedMemoryMultiply(histograms, sizeof(long long)),
          sizeof(gpu::em::detail::DeviceProfileCounters));
    }

    void initialize(gpu::em::GpuEmConfig::ProfileProjection const& config,
                    ExecutionSpace const& execution = {}) {
      if (!config.enabled || !config.accumulate_on_device) return;
      (void)projectedDeviceBytes(config);
      bins_ = config.output_bin_count;
      if (bins_ > std::numeric_limits<std::size_t>::max() /
                      gpu::em::detail::DeviceProfileHistogramCount)
        throw std::length_error(
            "Kokkos resident profile histogram size overflow");
      auto const histogram_values =
          gpu::em::detail::DeviceProfileHistogramCount * bins_;
      histograms_ = Kokkos::View<long long*, memory_space>(
          Kokkos::view_alloc(
              execution, Kokkos::WithoutInitializing,
              "c8_kokkos_profile_histograms"),
          histogram_values);
      counters_ = Kokkos::View<gpu::em::detail::DeviceProfileCounters*,
                               memory_space>(
          Kokkos::view_alloc(
              execution, Kokkos::WithoutInitializing,
              "c8_kokkos_profile_counters"),
          1);
      host_histograms_ = Kokkos::View<long long*, host_staging_space>(
          Kokkos::view_alloc(
              Kokkos::WithoutInitializing,
              "c8_kokkos_profile_host_histograms"),
          histogram_values);
      host_counters_ =
          Kokkos::View<gpu::em::detail::DeviceProfileCounters*,
                       host_staging_space>(
              Kokkos::view_alloc(
                  Kokkos::WithoutInitializing,
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
      view_.energy_loss_threshold_g_per_cm2 =
          config.energy_loss_threshold_g_per_cm2;
      enabled_ = true;
      reset(config.fixed_point_weight_limit,
            config.fixed_point_energy_limit_GeV, execution);
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
      Kokkos::deep_copy(execution, histograms_, 0LL);
      Kokkos::deep_copy(
          execution, counters_, gpu::em::detail::DeviceProfileCounters{});
      execution.fence("reset Kokkos resident profile");
      downloaded_ = false;
    }

    bool enabled() const noexcept { return enabled_; }
    gpu::em::detail::DeviceProfileAccumulator deviceView() const noexcept {
      return view_;
    }
    std::size_t deviceBytes() const noexcept {
      return histograms_.extent(0) * sizeof(long long) +
             counters_.extent(0) *
                 sizeof(gpu::em::detail::DeviceProfileCounters);
    }

    void accumulatePhoton(
        gpu::em::detail::DeviceProfileProjection const projection,
        PhotonStepView const& steps, std::size_t const step_count,
        PhotonFinalView const& final_states,
        std::size_t const final_state_count,
        ExecutionSpace const& execution = {}) {
      if (!enabled_) return;
      auto const accumulator = view_;
      Kokkos::parallel_for(
          "c8_kokkos_accumulate_photon_profile_steps",
          Kokkos::RangePolicy<ExecutionSpace>(execution, 0, step_count),
          KOKKOS_LAMBDA(std::size_t const index) {
            detail::accumulatePhotonProfileStep<KokkosProfileAtomicOperations>(
                projection, accumulator, steps(index));
          });
      Kokkos::parallel_for(
          "c8_kokkos_accumulate_photon_profile_final_states",
          Kokkos::RangePolicy<ExecutionSpace>(execution, 0, final_state_count),
          KOKKOS_LAMBDA(std::size_t const index) {
            detail::accumulatePhotonProfileFinalState<
                KokkosProfileAtomicOperations>(
                projection, accumulator, steps.data(), step_count,
                final_states(index));
          });
    }

    void accumulateLepton(
        gpu::em::detail::DeviceProfileProjection const projection,
        LeptonStepView const& steps, std::size_t const step_count,
        LeptonFinalView const& final_states,
        std::size_t const final_state_count,
        ExecutionSpace const& execution = {}) {
      if (!enabled_) return;
      auto const accumulator = view_;
      Kokkos::parallel_for(
          "c8_kokkos_accumulate_lepton_profile_steps",
          Kokkos::RangePolicy<ExecutionSpace>(execution, 0, step_count),
          KOKKOS_LAMBDA(std::size_t const index) {
            detail::accumulateLeptonProfileStep<KokkosProfileAtomicOperations>(
                projection, accumulator, steps(index));
          });
      Kokkos::parallel_for(
          "c8_kokkos_accumulate_lepton_profile_final_states",
          Kokkos::RangePolicy<ExecutionSpace>(execution, 0, final_state_count),
          KOKKOS_LAMBDA(std::size_t const index) {
            detail::accumulateLeptonProfileFinalState<
                KokkosProfileAtomicOperations>(
                projection, accumulator, steps.data(), step_count,
                final_states(index));
          });
    }

    gpu::em::GpuProfileResult download(
        ExecutionSpace const& execution = {}) {
      if (!enabled_)
        throw std::logic_error("Kokkos resident profile is not enabled");
      if (downloaded_)
        throw std::logic_error(
            "Kokkos resident profile was downloaded more than once");
      Kokkos::deep_copy(execution, host_histograms_, histograms_);
      Kokkos::deep_copy(execution, host_counters_, counters_);
      execution.fence("download Kokkos resident profile");
      auto const& counters = host_counters_(0);
      gpu::em::GpuProfileResult result{};
      result.photons.resize(bins_);
      result.electrons.resize(bins_);
      result.positrons.resize(bins_);
      result.muons_minus.resize(bins_);
      result.muons_plus.resize(bins_);
      result.muon_parent_productions.resize(bins_);
      result.energy_loss_GeV.resize(bins_);
      result.muon_energy_loss_GeV.resize(bins_);
      for (std::size_t bin = 0; bin < bins_; ++bin) {
        result.photons[bin] = static_cast<double>(host_histograms_(bin)) *
                              view_.inverse_weight_scale;
        result.electrons[bin] =
            static_cast<double>(host_histograms_(bins_ + bin)) *
            view_.inverse_weight_scale;
        result.positrons[bin] =
            static_cast<double>(host_histograms_(2 * bins_ + bin)) *
            view_.inverse_weight_scale;
        result.muons_minus[bin] =
            static_cast<double>(host_histograms_(3 * bins_ + bin)) *
            view_.inverse_weight_scale;
        result.muons_plus[bin] =
            static_cast<double>(host_histograms_(4 * bins_ + bin)) *
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
          counters.lepton_limits[static_cast<std::size_t>(
              gpu::em::LeptonTransportLimit::ParticleCut)];
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
          static_cast<double>(
              counters.weighted_unwritten_photoelectric_binding_energy) *
          view_.inverse_energy_scale;
      result.weighted_observation_cut_overlap_energy_GeV =
          static_cast<double>(counters.weighted_observation_cut_overlap_energy) *
          view_.inverse_energy_scale;
      result.weighted_mass_convention_correction_GeV =
          static_cast<double>(counters.weighted_mass_convention_correction) *
          view_.inverse_energy_scale;
      downloaded_ = true;
      return result;
    }

  private:
    Kokkos::View<long long*, memory_space> histograms_{};
    Kokkos::View<gpu::em::detail::DeviceProfileCounters*, memory_space>
        counters_{};
    Kokkos::View<long long*, host_staging_space> host_histograms_{};
    Kokkos::View<gpu::em::detail::DeviceProfileCounters*,
                 host_staging_space>
        host_counters_{};
    gpu::em::detail::DeviceProfileAccumulator view_{};
    std::size_t bins_{};
    bool enabled_{};
    bool downloaded_{};
  };

} // namespace corsika::accelerator::em::kokkos_detail

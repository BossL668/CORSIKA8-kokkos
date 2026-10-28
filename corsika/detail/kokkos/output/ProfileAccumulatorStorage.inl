/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

namespace corsika::kokkos {

  template <typename TExecutionSpace>
  std::size_t ProfileAccumulatorStorage<TExecutionSpace>::projectedDeviceBytes(
      ProfileProjectionConfig const& config) {
    if (!config.enabled_ || !config.accumulateOnDevice_) return 0;
    if (config.outputBinCount_ == 0 || !(config.outputBinWidthGPerCm2_ > 0.) ||
        !(config.energyLossThresholdGPerCm2_ >= 0.) ||
        !(config.fixedPointWeightLimit_ > 0.) || !(config.fixedPointEnergyLimitGeV_ > 0.))
      throw std::invalid_argument("Kokkos resident profile configuration is invalid");
    auto const histograms =
        checked_memory_multiply(ProfileHistogramCount, config.outputBinCount_);
    return checked_memory_add(checked_memory_multiply(histograms, sizeof(long long)),
                              sizeof(ProfileCounters));
  }

  template <typename TExecutionSpace>
  void ProfileAccumulatorStorage<TExecutionSpace>::initialize(
      ProfileProjectionConfig const& config, TExecutionSpace const& execution) {
    if (!config.enabled_ || !config.accumulateOnDevice_) return;
    (void)projectedDeviceBytes(config);
    fixedSnapshotConfig_ = config;
    bins_ = config.outputBinCount_;
    if (bins_ > std::numeric_limits<std::size_t>::max() / ProfileHistogramCount)
      throw std::length_error("Kokkos resident profile histogram size overflow");
    auto const histogram_values = ProfileHistogramCount * bins_;
    histograms_ = Kokkos::View<long long*, memory_space_type>(
        Kokkos::view_alloc(execution, Kokkos::WithoutInitializing,
                           "c8_kokkos_profile_histograms"),
        histogram_values);
    counters_ = Kokkos::View<ProfileCounters*, memory_space_type>(
        Kokkos::view_alloc(execution, Kokkos::WithoutInitializing,
                           "c8_kokkos_profile_counters"),
        1);
    hostHistograms_ = Kokkos::View<long long*, host_staging_space_type>(
        Kokkos::view_alloc(Kokkos::WithoutInitializing,
                           "c8_kokkos_profile_host_histograms"),
        histogram_values);
    hostCounters_ = Kokkos::View<ProfileCounters*, host_staging_space_type>(
        Kokkos::view_alloc(Kokkos::WithoutInitializing,
                           "c8_kokkos_profile_host_counters"),
        1);
    view_.photons_ = histograms_.data();
    view_.electrons_ = view_.photons_ + bins_;
    view_.positrons_ = view_.electrons_ + bins_;
    view_.muonsMinus_ = view_.positrons_ + bins_;
    view_.muonsPlus_ = view_.muonsMinus_ + bins_;
    view_.muonParentProductions_ = view_.muonsPlus_ + bins_;
    view_.energyLoss_ = view_.muonParentProductions_ + bins_;
    view_.muonEnergyLoss_ = view_.energyLoss_ + bins_;
    view_.counters_ = counters_.data();
    view_.bins_ = bins_;
    view_.binWidthGPerCm2_ = config.outputBinWidthGPerCm2_;
    view_.crossingMode_ = config.crossingMode_;
    view_.energyLossThresholdGPerCm2_ = config.energyLossThresholdGPerCm2_;
    enabled_ = true;
    reset(config.fixedPointWeightLimit_, config.fixedPointEnergyLimitGeV_, execution);
  }

  template <typename TExecutionSpace>
  void ProfileAccumulatorStorage<TExecutionSpace>::reset(
      double const weight_limit, double const energy_limit_GeV,
      TExecutionSpace const& execution) {
    if (!enabled_) return;
    if (!(weight_limit > 0.) || !(energy_limit_GeV > 0.))
      throw std::invalid_argument(
          "Kokkos resident profile fixed-point limits must be positive");
    constexpr double FixedPointHeadroom = 0x1p62;
    view_.weightScale_ = FixedPointHeadroom / weight_limit;
    view_.inverseWeightScale_ = 1. / view_.weightScale_;
    view_.energyScale_ = FixedPointHeadroom / energy_limit_GeV;
    view_.inverseEnergyScale_ = 1. / view_.energyScale_;
    fixedSnapshotConfig_.fixedPointWeightLimit_ = weight_limit;
    fixedSnapshotConfig_.fixedPointEnergyLimitGeV_ = energy_limit_GeV;
    Kokkos::deep_copy(execution, histograms_, 0LL);
    Kokkos::deep_copy(execution, counters_, ProfileCounters{});
    execution.fence("reset Kokkos resident profile");
    hostShards_.reset(view_);
    downloaded_ = false;
    fixedSnapshotSealed_ = false;
  }

  template <typename TExecutionSpace>
  bool ProfileAccumulatorStorage<TExecutionSpace>::enabled() const noexcept {
    return enabled_;
  }

  template <typename TExecutionSpace>
  void ProfileAccumulatorStorage<TExecutionSpace>::enableHostShards(
      [[maybe_unused]] std::size_t workers, [[maybe_unused]] std::size_t budget) {
#ifdef KOKKOS_ENABLE_OPENMP
    if constexpr (std::is_same_v<TExecutionSpace, Kokkos::OpenMP>) {
      if (!enabled_ || downloaded_ || fixedSnapshotSealed_)
        throw std::logic_error("host profile shards require a fresh profile");
      hostShards_.configure(view_, workers, budget);
      return;
    }
#endif
    throw std::logic_error("profile shards require the OpenMP execution space");
  }

  template <typename TExecutionSpace>
  void ProfileAccumulatorStorage<TExecutionSpace>::enableCooperativeHostShards(
      std::size_t workers, std::size_t budget) {
    enableHostShards(workers, budget);
  }

  template <typename TExecutionSpace>
  HostProfileShards const& ProfileAccumulatorStorage<TExecutionSpace>::hostShards() const
      noexcept {
    return hostShards_;
  }

  template <typename TExecutionSpace>
  void ProfileAccumulatorStorage<TExecutionSpace>::requireOpenForAccumulation() const {
    if (fixedSnapshotSealed_)
      throw std::logic_error(
          "profile endpoint already committed; reset before new tracks");
  }

  template <typename TExecutionSpace>
  ProfileAccumulatorView ProfileAccumulatorStorage<TExecutionSpace>::deviceView() const
      noexcept {
    return view_;
  }

  template <typename TExecutionSpace>
  std::size_t ProfileAccumulatorStorage<TExecutionSpace>::deviceBytes() const noexcept {
    return histograms_.extent(0) * sizeof(long long) +
           counters_.extent(0) * sizeof(ProfileCounters) + hostShards_.bytes();
  }

  template <typename TExecutionSpace>
  void ProfileAccumulatorStorage<TExecutionSpace>::accumulatePhoton(
      ProfileProjectionView const projection, photon_step_view_type const& steps,
      std::size_t const step_count, photon_final_view_type const& final_states,
      std::size_t const final_state_count, TExecutionSpace const& execution) {
    if (!enabled_) return;
    requireOpenForAccumulation();
    auto const accumulator = view_;
    Kokkos::parallel_for(
        "c8_kokkos_accumulate_photon_profile_steps",
        Kokkos::RangePolicy<TExecutionSpace>(execution, 0, step_count),
        KOKKOS_LAMBDA(std::size_t const index) {
          accumulate_photon_profile_step<ProfileAtomicOperations>(projection, accumulator,
                                                                  steps(index));
        });
    Kokkos::parallel_for(
        "c8_kokkos_accumulate_photon_profile_final_states",
        Kokkos::RangePolicy<TExecutionSpace>(execution, 0, final_state_count),
        KOKKOS_LAMBDA(std::size_t const index) {
          accumulate_photon_profile_final_state<ProfileAtomicOperations>(
              projection, accumulator, steps.data(), step_count, final_states(index));
        });
  }

  template <typename TExecutionSpace>
  void ProfileAccumulatorStorage<TExecutionSpace>::accumulateLepton(
      ProfileProjectionView const projection, lepton_step_view_type const& steps,
      std::size_t const step_count, lepton_final_view_type const& final_states,
      std::size_t const final_state_count, TExecutionSpace const& execution) {
    if (!enabled_) return;
    requireOpenForAccumulation();
    auto const accumulator = view_;
    Kokkos::parallel_for(
        "c8_kokkos_accumulate_lepton_profile_steps",
        Kokkos::RangePolicy<TExecutionSpace>(execution, 0, step_count),
        KOKKOS_LAMBDA(std::size_t const index) {
          accumulate_lepton_profile_step<ProfileAtomicOperations>(projection, accumulator,
                                                                  steps(index));
        });
    Kokkos::parallel_for(
        "c8_kokkos_accumulate_lepton_profile_final_states",
        Kokkos::RangePolicy<TExecutionSpace>(execution, 0, final_state_count),
        KOKKOS_LAMBDA(std::size_t const index) {
          accumulate_lepton_profile_final_state<ProfileAtomicOperations>(
              projection, accumulator, steps.data(), step_count, final_states(index));
        });
  }

  template <typename TExecutionSpace>
  FixedProfileSnapshot ProfileAccumulatorStorage<TExecutionSpace>::downloadFixed(
      std::string const& identity, CooperativeEndpoint endpoint,
      TExecutionSpace const& execution) {
    if (!enabled_ || downloaded_)
      throw std::logic_error("fixed profile snapshot unavailable/already committed");
    if (identity.empty() || static_cast<unsigned>(endpoint) > 1)
      throw std::invalid_argument("invalid cooperative profile snapshot identity");
    // Final export: a failed copy must not be accepted as a fresh ledger.
    downloaded_ = true;
    fixedSnapshotSealed_ = true;
    try {
      Kokkos::deep_copy(execution, hostHistograms_, histograms_);
      Kokkos::deep_copy(execution, hostCounters_, counters_);
      execution.fence("download cooperative fixed profile endpoint");
    } catch (...) {
      try {
        execution.fence("unwind cooperative profile endpoint copy");
      } catch (...) {}
      throw;
    }
    FixedProfileSnapshot out;
    out.showerIdentity_ = identity;
    out.endpoint_ = endpoint;
    out.config_ = fixedSnapshotConfig_;
    out.weightUnits_ = view_.inverseWeightScale_;
    out.energyUnits_ = view_.inverseEnergyScale_;
    out.counters_ = hostCounters_(0);
    out.histograms_.resize(histograms_.extent(0));
    for (std::size_t i = 0; i < out.histograms_.size(); ++i)
      out.histograms_[i] = hostHistograms_(i);
    validate_fixed_profile(out);
    hostShards_.mergeInto(out);
    return out;
  }

  template <typename TExecutionSpace>
  ProfileResult ProfileAccumulatorStorage<TExecutionSpace>::download(
      TExecutionSpace const& execution) {
    if (!enabled_) throw std::logic_error("Kokkos resident profile is not enabled");
    if (hostShards_.enabled()) {
      // Standalone OpenMP uses the same checked integer merge as cooperative
      // transport. Decode only AFTER the canonical and sharded ledgers have
      // been combined; never add separately rounded floating-point profiles.
      // downloadFixed seals the event even if finalization fails.
      return decode_fixed_profile(downloadFixed("standalone-openmp-profile",
                                                CooperativeEndpoint::OpenMP, execution));
    }
    if (downloaded_)
      throw std::logic_error("Kokkos resident profile was downloaded more than once");
    Kokkos::deep_copy(execution, hostHistograms_, histograms_);
    Kokkos::deep_copy(execution, hostCounters_, counters_);
    execution.fence("download Kokkos resident profile");
    auto const& counters = hostCounters_(0);
    ProfileResult result{};
    result.photons_.resize(bins_);
    result.electrons_.resize(bins_);
    result.positrons_.resize(bins_);
    result.muonsMinus_.resize(bins_);
    result.muonsPlus_.resize(bins_);
    result.muonParentProductions_.resize(bins_);
    result.energyLossGeV_.resize(bins_);
    result.muonEnergyLossGeV_.resize(bins_);
    for (std::size_t bin = 0; bin < bins_; ++bin) {
      result.photons_[bin] =
          static_cast<double>(hostHistograms_(bin)) * view_.inverseWeightScale_;
      result.electrons_[bin] =
          static_cast<double>(hostHistograms_(bins_ + bin)) * view_.inverseWeightScale_;
      result.positrons_[bin] = static_cast<double>(hostHistograms_(2 * bins_ + bin)) *
                               view_.inverseWeightScale_;
      result.muonsMinus_[bin] = static_cast<double>(hostHistograms_(3 * bins_ + bin)) *
                                view_.inverseWeightScale_;
      result.muonsPlus_[bin] = static_cast<double>(hostHistograms_(4 * bins_ + bin)) *
                               view_.inverseWeightScale_;
      result.muonParentProductions_[bin] =
          static_cast<double>(hostHistograms_(5 * bins_ + bin)) *
          view_.inverseWeightScale_;
      auto const electromagnetic_loss =
          static_cast<double>(hostHistograms_(6 * bins_ + bin)) *
          view_.inverseEnergyScale_;
      result.muonEnergyLossGeV_[bin] =
          static_cast<double>(hostHistograms_(7 * bins_ + bin)) *
          view_.inverseEnergyScale_;
      result.energyLossGeV_[bin] = electromagnetic_loss + result.muonEnergyLossGeV_[bin];
      result.weightedDepositedEnergyGeV_ += result.energyLossGeV_[bin];
    }
    result.steps_ = counters.steps_;
    result.leptonMaterialBoundaries_ = counters.leptonMaterialBoundaries_;
    result.depositedSteps_ = counters.depositedSteps_;
    result.particleCuts_ =
        counters.photonCuts_ +
        counters
            .leptonLimits_[static_cast<std::size_t>(LeptonTransportLimit::ParticleCut)];
    result.fixedPointOverflows_ = counters.fixedPointOverflows_;
    result.invalidRecords_ = counters.invalidRecords_;
    result.weightedMediumRestMassInputGeV_ =
        static_cast<double>(counters.weightedMediumRestMassInput_) *
        view_.inverseEnergyScale_;
    result.weightedCutRestMassEnergyGeV_ =
        static_cast<double>(counters.weightedCutRestMassEnergy_) *
        view_.inverseEnergyScale_;
    result.weightedObservedTotalEnergyGeV_ =
        static_cast<double>(counters.weightedObservedTotalEnergy_) *
        view_.inverseEnergyScale_;
    result.weightedEscapedTotalEnergyGeV_ =
        static_cast<double>(counters.weightedEscapedTotalEnergy_) *
        view_.inverseEnergyScale_;
    result.weightedUnwrittenPhotoelectricBindingEnergyGeV_ =
        static_cast<double>(counters.weightedUnwrittenPhotoelectricBindingEnergy_) *
        view_.inverseEnergyScale_;
    result.weightedObservationCutOverlapEnergyGeV_ =
        static_cast<double>(counters.weightedObservationCutOverlapEnergy_) *
        view_.inverseEnergyScale_;
    result.weightedMassConventionCorrectionGeV_ =
        static_cast<double>(counters.weightedMassConventionCorrection_) *
        view_.inverseEnergyScale_;
    result.weightedNativeMuonPairCutEnergyGeV_ =
        static_cast<double>(counters.weightedNativeMuonPairCutEnergy_) *
        view_.inverseEnergyScale_;
    downloaded_ = true;
    return result;
  }

} // namespace corsika::kokkos

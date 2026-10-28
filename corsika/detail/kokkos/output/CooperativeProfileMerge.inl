/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

namespace corsika::kokkos {

  inline bool same_profile_config(ProfileProjectionConfig const& a,
                                  ProfileProjectionConfig const& b) {
    if (a.enabled_ != b.enabled_) return false;
    if (a.accumulateOnDevice_ != b.accumulateOnDevice_) return false;
    if (a.axisStepLengthM_ != b.axisStepLengthM_) return false;
    if (a.axisGrammageGPerCm2_ != b.axisGrammageGPerCm2_) return false;
    if (a.outputBinCount_ != b.outputBinCount_) return false;
    if (a.outputBinWidthGPerCm2_ != b.outputBinWidthGPerCm2_) return false;
    if (a.energyLossThresholdGPerCm2_ != b.energyLossThresholdGPerCm2_) return false;
    if (a.fixedPointWeightLimit_ != b.fixedPointWeightLimit_) return false;
    if (a.fixedPointEnergyLimitGeV_ != b.fixedPointEnergyLimitGeV_) return false;
    if (a.crossingMode_ != b.crossingMode_) return false;
    for (int d = 0; d < 3; ++d) {
      if (a.axisStartPositionM_[d] != b.axisStartPositionM_[d]) return false;
      if (a.axisDirection_[d] != b.axisDirection_[d]) return false;
    }
    return true;
  }

  inline void validate_fixed_profile(FixedProfileSnapshot const& s) {
    if (s.counters_.fixedPointOverflows_ || s.counters_.invalidRecords_)
      throw std::runtime_error(
          "cooperative profile rejected: overflows=" +
          std::to_string(s.counters_.fixedPointOverflows_) +
          ", invalid_records=" + std::to_string(s.counters_.invalidRecords_));
    if (s.showerIdentity_.empty() || static_cast<unsigned>(s.endpoint_) > 1 ||
        !s.config_.enabled_ || !s.config_.accumulateOnDevice_ ||
        !s.config_.outputBinCount_ ||
        s.config_.outputBinCount_ > std::numeric_limits<std::size_t>::max() / 8 ||
        s.histograms_.size() != 8 * s.config_.outputBinCount_ ||
        !std::isfinite(s.weightUnits_) || s.weightUnits_ <= 0. ||
        !std::isfinite(s.energyUnits_) || s.energyUnits_ <= 0.)
      throw std::invalid_argument("invalid/overflowed cooperative profile snapshot");
    if (s.weightUnits_ != 1. / (0x1p62 / s.config_.fixedPointWeightLimit_) ||
        s.energyUnits_ != 1. / (0x1p62 / s.config_.fixedPointEnergyLimitGeV_))
      throw std::invalid_argument("cooperative profile scale mismatch");
  }

  inline ProfileCounters merge_profile_counters(ProfileCounters a,
                                                ProfileCounters const& b) {
    auto add = [](auto x, auto y) {
      using value_type = decltype(x);
      if constexpr (std::is_signed_v<value_type>) {
        if ((y > 0 && x > std::numeric_limits<value_type>::max() - y) ||
            (y < 0 && x < std::numeric_limits<value_type>::min() - y))
          throw std::overflow_error("cooperative profile energy counter overflow");
      } else if (y > std::numeric_limits<value_type>::max() - x)
        throw std::overflow_error("cooperative profile counter overflow");
      return x + y;
    };
    a.steps_ = add(a.steps_, b.steps_);
    a.leptonMaterialBoundaries_ =
        add(a.leptonMaterialBoundaries_, b.leptonMaterialBoundaries_);
    a.depositedSteps_ = add(a.depositedSteps_, b.depositedSteps_);
    a.photonCuts_ = add(a.photonCuts_, b.photonCuts_);
    for (unsigned i = 0; i < 8; ++i)
      a.leptonLimits_[i] = add(a.leptonLimits_[i], b.leptonLimits_[i]);
    a.moliereTrials_ = add(a.moliereTrials_, b.moliereTrials_);
    a.moliereDeflections_ = add(a.moliereDeflections_, b.moliereDeflections_);
    a.moliereZeroDeflections_ = add(a.moliereZeroDeflections_, b.moliereZeroDeflections_);
    a.moliereNewtonIterations_ =
        add(a.moliereNewtonIterations_, b.moliereNewtonIterations_);
    a.thinningHillasVertices_ = add(a.thinningHillasVertices_, b.thinningHillasVertices_);
    a.thinningStatisticalVertices_ =
        add(a.thinningStatisticalVertices_, b.thinningStatisticalVertices_);
    a.thinningParticlesDiscarded_ =
        add(a.thinningParticlesDiscarded_, b.thinningParticlesDiscarded_);
    a.fixedPointOverflows_ = add(a.fixedPointOverflows_, b.fixedPointOverflows_);
    a.invalidRecords_ = add(a.invalidRecords_, b.invalidRecords_);
    a.weightedMediumRestMassInput_ =
        add(a.weightedMediumRestMassInput_, b.weightedMediumRestMassInput_);
    a.weightedCutRestMassEnergy_ =
        add(a.weightedCutRestMassEnergy_, b.weightedCutRestMassEnergy_);
    a.weightedObservedTotalEnergy_ =
        add(a.weightedObservedTotalEnergy_, b.weightedObservedTotalEnergy_);
    a.weightedEscapedTotalEnergy_ =
        add(a.weightedEscapedTotalEnergy_, b.weightedEscapedTotalEnergy_);
    a.weightedUnwrittenPhotoelectricBindingEnergy_ =
        add(a.weightedUnwrittenPhotoelectricBindingEnergy_,
            b.weightedUnwrittenPhotoelectricBindingEnergy_);
    a.weightedObservationCutOverlapEnergy_ = add(a.weightedObservationCutOverlapEnergy_,
                                                 b.weightedObservationCutOverlapEnergy_);
    a.weightedMassConventionCorrection_ =
        add(a.weightedMassConventionCorrection_, b.weightedMassConventionCorrection_);
    a.weightedNativeMuonPairCutEnergy_ =
        add(a.weightedNativeMuonPairCutEnergy_, b.weightedNativeMuonPairCutEnergy_);
    a.moliereMaxNewtonIterations_ =
        std::max(a.moliereMaxNewtonIterations_, b.moliereMaxNewtonIterations_);
    return a;
  }

  inline ProfileResult decode_fixed_profile(FixedProfileSnapshot const& s) {
    validate_fixed_profile(s);
    auto bins_ = s.config_.outputBinCount_;
    ProfileAccumulatorView view_{};
    view_.inverseWeightScale_ = s.weightUnits_;
    view_.inverseEnergyScale_ = s.energyUnits_;
    auto hostHistograms_ = [&](std::size_t i) { return s.histograms_.at(i); };
    auto const& counters = s.counters_;
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
    if (counters
            .leptonLimits_[static_cast<std::size_t>(LeptonTransportLimit::ParticleCut)] >
        std::numeric_limits<unsigned long long>::max() - counters.photonCuts_)
      throw std::overflow_error("cooperative decoded particle cut counter overflow");
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

    return result;
  }
  inline CooperativeProfileMerge::CooperativeProfileMerge(std::string identity,
                                                          ProfileProjectionConfig config)
      : identity_(std::move(identity))
      , config_(std::move(config)) {}

  inline void CooperativeProfileMerge::commit(FixedProfileSnapshot const& s) {
    if (finished_) throw std::logic_error("cooperative profile already taken");
    validate_fixed_profile(s);
    auto bit = 1u << static_cast<unsigned>(s.endpoint_);
    if ((seen_ & bit) || s.showerIdentity_ != identity_ ||
        !same_profile_config(s.config_, config_))
      throw std::invalid_argument("duplicate/incompatible cooperative profile commit");
    auto next = seen_ ? merged_ : s;
    if (seen_) {
      if (next.weightUnits_ != s.weightUnits_ || next.energyUnits_ != s.energyUnits_)
        throw std::invalid_argument("cooperative profile scales differ");
      FixedAccumulatorLayout layout{identity_ + "/profile-eight-columns-v1",
                                    s.weightUnits_, s.histograms_.size()};
      merge_fixed_accumulators(next.histograms_, layout, s.histograms_, layout);
      next.counters_ = merge_profile_counters(next.counters_, s.counters_);
    }
    merged_ = std::move(next);
    seen_ |= bit;
  }

  inline FixedProfileSnapshot CooperativeProfileMerge::take() {
    if (seen_ != 3 || finished_)
      throw std::logic_error("profile endpoints not complete/already taken");
    finished_ = true;
    return std::move(merged_);
  }

} // namespace corsika::kokkos

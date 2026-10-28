/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

namespace corsika::kokkos::radio {
  inline bool same_observers(std::vector<RadioObserverSnapshot> const& a,
                             std::vector<RadioObserverSnapshot> const& b) {
    if (a.size() != b.size()) return false;
    for (std::size_t i = 0; i < a.size(); ++i) {
      for (int d = 0; d < 3; ++d)
        if (a[i].positionM_[d] != b[i].positionM_[d]) return false;
      if (a[i].startTimeS_ != b[i].startTimeS_ || a[i].durationS_ != b[i].durationS_ ||
          a[i].sampleRateHz_ != b[i].sampleRateHz_ ||
          a[i].numberOfBins_ != b[i].numberOfBins_)
        return false;
    }
    return true;
  }

  inline bool same_radio_physics(RadioConfig const& a, RadioConfig const& b) {
    if (a.enabled_ != b.enabled_) return false;
    if (a.coreasEnabled_ != b.coreasEnabled_) return false;
    if (a.zhsEnabled_ != b.zhsEnabled_) return false;
    if (a.deterministic_ != b.deterministic_) return false;
    if (a.fixedPointFieldLimitVPerM_ != b.fixedPointFieldLimitVPerM_) return false;
    if (a.zhsSubtrackRefinement_ != b.zhsSubtrackRefinement_) return false;
    if (a.trackDiagnostics_ != b.trackDiagnostics_) return false;
    if (a.propagation_.homogeneousRefractiveIndex_ !=
        b.propagation_.homogeneousRefractiveIndex_)
      return false;
    if (a.propagation_.minimumHeightM_ != b.propagation_.minimumHeightM_) return false;
    if (a.propagation_.maximumHeightM_ != b.propagation_.maximumHeightM_) return false;
    if (a.propagation_.stepM_ != b.propagation_.stepM_) return false;
    if (a.propagation_.inverseStepPerM_ != b.propagation_.inverseStepPerM_) return false;
    if (a.propagation_.slopeRefractivityLower_ != b.propagation_.slopeRefractivityLower_)
      return false;
    if (a.propagation_.slopeRefractivityUpper_ != b.propagation_.slopeRefractivityUpper_)
      return false;
    if (a.propagation_.slopeIntegratedRefractivityLower_ !=
        b.propagation_.slopeIntegratedRefractivityLower_)
      return false;
    if (a.propagation_.slopeIntegratedRefractivityUpper_ !=
        b.propagation_.slopeIntegratedRefractivityUpper_)
      return false;
    if (a.propagation_.refractivity_ != b.propagation_.refractivity_) return false;
    if (a.propagation_.integratedRefractivity_ != b.propagation_.integratedRefractivity_)
      return false;
    return same_observers(a.coreasObservers_, b.coreasObservers_) &&
           same_observers(a.zhsObservers_, b.zhsObservers_);
  }

  inline std::size_t radio_bins(std::vector<RadioObserverSnapshot> const& observers) {
    std::size_t n = 0;
    for (auto const& o : observers) {
      if (!o.numberOfBins_ ||
          o.numberOfBins_ > std::numeric_limits<std::size_t>::max() / 3 - n ||
          !std::isfinite(o.sampleRateHz_) || o.sampleRateHz_ <= 0. ||
          !std::isfinite(o.startTimeS_) || !std::isfinite(o.durationS_) ||
          o.durationS_ < 0.)
        throw std::invalid_argument("invalid cooperative radio observer layout");
      for (double x : o.positionM_)
        if (!std::isfinite(x))
          throw std::invalid_argument("nonfinite cooperative radio observer");
      n += o.numberOfBins_;
    }
    return n;
  }

  inline void validate_fixed_radio(FixedRadioSnapshot const& s) {
    if (s.showerIdentity_.empty() || !s.config_.enabled_ || !s.config_.deterministic_ ||
        (!s.config_.coreasEnabled_ && !s.config_.zhsEnabled_) ||
        !std::isfinite(s.config_.fixedPointFieldLimitVPerM_) ||
        s.config_.fixedPointFieldLimitVPerM_ <= 0. ||
        static_cast<unsigned>(s.endpoint_) > 1 || s.counters_.fixedPointOverflows_)
      throw std::invalid_argument("invalid/overflowed cooperative fixed radio snapshot");
    if (s.coreas_.size() != 3 * radio_bins(s.config_.coreasObservers_) ||
        s.zhs_.size() != 3 * radio_bins(s.config_.zhsObservers_))
      throw std::invalid_argument("cooperative radio snapshot extent mismatch");
  }

  inline RadioCounters merge_radio_counters(RadioCounters a, RadioCounters const& b) {
    auto add = [](unsigned long long x, unsigned long long y) {
      if (y > std::numeric_limits<unsigned long long>::max() - x)
        throw std::overflow_error("cooperative radio counter overflow");
      return x + y;
    };
    a.coreasContributions_ = add(a.coreasContributions_, b.coreasContributions_);
    a.zhsContributions_ = add(a.zhsContributions_, b.zhsContributions_);
    a.zhsSubtracks_ = add(a.zhsSubtracks_, b.zhsSubtracks_);
    a.validTracks_ = add(a.validTracks_, b.validTracks_);
    a.fixedPointOverflows_ = add(a.fixedPointOverflows_, b.fixedPointOverflows_);
    auto sum = [](double x, double y) {
      double z = x + y;
      if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z))
        throw std::overflow_error("nonfinite cooperative radio diagnostics");
      return z;
    };
    a.weightedSegmentCount_ = sum(a.weightedSegmentCount_, b.weightedSegmentCount_);
    a.trackLengthM_ = sum(a.trackLengthM_, b.trackLengthM_);
    a.weightedTrackLengthM_ = sum(a.weightedTrackLengthM_, b.weightedTrackLengthM_);
    a.electronWeightedTrackLengthM_ =
        sum(a.electronWeightedTrackLengthM_, b.electronWeightedTrackLengthM_);
    a.positronWeightedTrackLengthM_ =
        sum(a.positronWeightedTrackLengthM_, b.positronWeightedTrackLengthM_);
    a.signedChargeWeightedTrackLengthM_ =
        sum(a.signedChargeWeightedTrackLengthM_, b.signedChargeWeightedTrackLengthM_);
    a.energyWeightedTrackLengthGeVM_ =
        sum(a.energyWeightedTrackLengthGeVM_, b.energyWeightedTrackLengthGeVM_);
    a.weightedDirectionChangeRad_ =
        sum(a.weightedDirectionChangeRad_, b.weightedDirectionChangeRad_);
    a.weightedDirectionChangeSquaredRad2_ =
        sum(a.weightedDirectionChangeSquaredRad2_, b.weightedDirectionChangeSquaredRad2_);
    a.weightedBetaDeficitTrackLengthM_ =
        sum(a.weightedBetaDeficitTrackLengthM_, b.weightedBetaDeficitTrackLengthM_);
    a.weightedTimeResidualS_ = sum(a.weightedTimeResidualS_, b.weightedTimeResidualS_);
    for (int i = 0; i < 3; ++i)
      a.signedChargeWeightedDirectionChange_[i] =
          sum(a.signedChargeWeightedDirectionChange_[i],
              b.signedChargeWeightedDirectionChange_[i]);
    for (int i = 0; i < 15; ++i)
      a.weightedTrackLengthByKineticEnergyM_[i] =
          sum(a.weightedTrackLengthByKineticEnergyM_[i],
              b.weightedTrackLengthByKineticEnergyM_[i]);
    // Maxima are not additive. Two finite maxima can have an overflowing sum
    // without their (valid) merged maximum overflowing.
    for (double x : {a.maximumSegmentLengthM_, b.maximumSegmentLengthM_,
                     a.maximumDirectionChangeRad_, b.maximumDirectionChangeRad_})
      if (!std::isfinite(x))
        throw std::overflow_error("nonfinite cooperative radio maximum");
    a.maximumSegmentLengthM_ =
        std::max(a.maximumSegmentLengthM_, b.maximumSegmentLengthM_);
    a.maximumDirectionChangeRad_ =
        std::max(a.maximumDirectionChangeRad_, b.maximumDirectionChangeRad_);
    return a;
  }

  inline RadioWaveforms decode_fixed_radio(FixedRadioSnapshot const& s) {
    validate_fixed_radio(s);
    RadioWaveforms result;
    auto decode = [&](auto const& observers, auto const& integers, bool zhs) {
      std::vector<RadioWaveform> out;
      auto bins = radio_bins(observers);
      std::size_t offset = 0;
      for (auto const& observer : observers) {
        auto count = static_cast<std::size_t>(observer.numberOfBins_);
        auto factor = zhs ? observer.sampleRateHz_ : 1.;
        auto inv = s.config_.fixedPointFieldLimitVPerM_ / (FixedPointHeadroom * factor);
        RadioWaveform w;
        w.x_.resize(count);
        w.y_.resize(count);
        w.z_.resize(count);
        for (std::size_t i = 0; i < count; ++i) {
          w.x_[i] = static_cast<double>(integers[offset + i]) * inv;
          w.y_[i] = static_cast<double>(integers[bins + offset + i]) * inv;
          w.z_[i] = static_cast<double>(integers[2 * bins + offset + i]) * inv;
        }
        offset += count;
        out.push_back(std::move(w));
      }
      return out;
    };
    result.coreas_ = decode(s.config_.coreasObservers_, s.coreas_, false);
    result.zhs_ = decode(s.config_.zhsObservers_, s.zhs_, true);
    return result;
  }

  inline CooperativeRadioMerge::CooperativeRadioMerge(std::string identity,
                                                      RadioConfig config)
      : identity_(std::move(identity))
      , config_(std::move(config)) {
    if (identity_.empty())
      throw std::invalid_argument("empty cooperative shower identity");
  }

  inline void CooperativeRadioMerge::commit(FixedRadioSnapshot const& input) {
    if (finished_) throw std::logic_error("cooperative radio output already taken");
    validate_fixed_radio(input);
    auto bit = 1u << static_cast<unsigned>(input.endpoint_);
    if ((seen_ & bit) || input.showerIdentity_ != identity_ ||
        !same_radio_physics(input.config_, config_))
      throw std::invalid_argument("duplicate/incompatible cooperative radio commit");
    auto next = seen_ ? merged_ : input;
    if (seen_) {
      // Config equality above covers each observer's sample rate/scale and
      // propagation model. The layout label also disambiguates field/potential.
      auto merge = [&](auto& dst, auto const& src, char const* algorithm) {
        ::corsika::kokkos::FixedAccumulatorLayout layout{
            identity_ + algorithm, config_.fixedPointFieldLimitVPerM_, dst.size()};
        ::corsika::kokkos::merge_fixed_accumulators(dst, layout, src, layout);
      };
      merge(next.coreas_, input.coreas_, "/coreas-field-soa-v1");
      merge(next.zhs_, input.zhs_, "/zhs-potential-soa-v1");
      next.counters_ = merge_radio_counters(next.counters_, input.counters_);
    }
    merged_ = std::move(next);
    seen_ |= bit;
  }

  inline FixedRadioSnapshot CooperativeRadioMerge::take() {
    if (seen_ != 3 || finished_)
      throw std::logic_error("cooperative radio endpoints not complete/already taken");
    finished_ = true;
    return std::move(merged_);
  }
} // namespace corsika::kokkos::radio

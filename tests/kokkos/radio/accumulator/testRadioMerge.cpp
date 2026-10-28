/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include "RadioAccumulatorComparison.hpp"
#include <limits>
#include <iostream>

namespace {
  using namespace radio_accumulator_test;
  std::pair<cr::FixedRadioSnapshot, ar::detail::FixedRadioSnapshot> snapshot(
      unsigned endpoint) {
    cr::FixedRadioSnapshot a;
    a.showerIdentity_ = "cooperative-production";
    a.endpoint_ = static_cast<cp::CooperativeEndpoint>(endpoint);
    a.config_.enabled_ = true;
    a.config_.propagation_.homogeneousRefractiveIndex_ = 1.0003;
    cr::RadioObserverSnapshot o;
    o.positionM_[0] = 30.;
    o.positionM_[1] = 40.;
    o.numberOfBins_ = 3;
    o.durationS_ = 3.e-9;
    o.sampleRateHz_ = 1.e9;
    a.config_.coreasObservers_ = {o};
    a.config_.zhsObservers_ = {o};
    a.coreas_ = {1, -2, 3, -4, 5, -6, 7, -8, 9};
    a.zhs_ = {-9, 8, -7, 6, -5, 4, -3, 2, -1};
    for (auto& x : a.coreas_) x *= endpoint + 1;
    for (auto& x : a.zhs_) x *= endpoint + 1;
    a.counters_.validTracks_ = endpoint + 2;
    a.counters_.coreasContributions_ = 7;
    a.counters_.zhsContributions_ = 8;
    a.counters_.zhsSubtracks_ = 9;
    a.counters_.weightedSegmentCount_ = 2.5;
    a.counters_.trackLengthM_ = 3.25;
    a.counters_.weightedTrackLengthM_ = 7.5;
    a.counters_.maximumSegmentLengthM_ = 1. + endpoint;
    a.counters_.maximumDirectionChangeRad_ = 0.125 + endpoint;
    a.counters_.signedChargeWeightedDirectionChange_[0] = -.25;
    a.counters_.weightedTrackLengthByKineticEnergyM_[14] = 4.;
    ar::detail::FixedRadioSnapshot b;
    b.shower_identity = a.showerIdentity_;
    b.endpoint = static_cast<ar::detail::CooperativeEndpoint>(endpoint);
    b.config = to_reference(a.config_);
    b.coreas = a.coreas_;
    b.zhs = a.zhs_;
    b.counters.valid_tracks = a.counters_.validTracks_;
    b.counters.coreas_contributions = 7;
    b.counters.zhs_contributions = 8;
    b.counters.zhs_subtracks = 9;
    b.counters.weighted_segment_count = 2.5;
    b.counters.track_length_m = 3.25;
    b.counters.weighted_track_length_m = 7.5;
    b.counters.maximum_segment_length_m = 1. + endpoint;
    b.counters.maximum_direction_change_rad = .125 + endpoint;
    b.counters.signed_charge_weighted_direction_change[0] = -.25;
    b.counters.weighted_track_length_by_kinetic_energy_m[14] = 4.;
    return {a, b};
  }
} // namespace

TEST_CASE("Radio cooperative merge preserves endpoint identity and exact integer output",
          "[kokkos][radio][accumulator]") {
  auto first = snapshot(0), second = snapshot(1);
  for (bool reverse : {false, true}) {
    cr::CooperativeRadioMerge a(first.first.showerIdentity_, first.first.config_);
    ar::detail::CooperativeRadioMerge b(first.second.shower_identity,
                                        first.second.config);
    CHECK_THROWS(a.take());
    CHECK_THROWS(b.take());
    auto x = reverse ? second : first;
    auto y = reverse ? first : second;
    a.commit(x.first);
    b.commit(x.second);
    CHECK_THROWS(a.commit(x.first));
    CHECK_THROWS(b.commit(x.second));
    CHECK_THROWS(a.take());
    CHECK_THROWS(b.take());
    auto invalid = y;
    invalid.first.config_.zhsObservers_[0].sampleRateHz_ *= 2;
    invalid.second.config.zhs_observers[0].sample_rate_Hz *= 2;
    CHECK_THROWS(a.commit(invalid.first));
    CHECK_THROWS(b.commit(invalid.second));
    invalid = y;
    invalid.first.showerIdentity_ = "different";
    invalid.second.shower_identity = "different";
    CHECK_THROWS(a.commit(invalid.first));
    CHECK_THROWS(b.commit(invalid.second));
    a.commit(y.first);
    b.commit(y.second);
    auto actual = a.take();
    auto expected = b.take();
    check_snapshot(actual, expected);
    auto decoded = cr::decode_fixed_radio(actual);
    auto old = ar::detail::decodeFixedRadio(expected);
    CHECK(decoded.coreas_[0].x_ == old.coreas[0].x);
    CHECK(decoded.coreas_[0].y_ == old.coreas[0].y);
    CHECK(decoded.coreas_[0].z_ == old.coreas[0].z);
    CHECK(decoded.zhs_[0].x_ == old.zhs[0].x);
    CHECK(decoded.zhs_[0].y_ == old.zhs[0].y);
    CHECK(decoded.zhs_[0].z_ == old.zhs[0].z);
    CHECK_THROWS(a.take());
    CHECK_THROWS(b.take());
    CHECK_THROWS(a.commit(y.first));
    CHECK_THROWS(b.commit(y.second));
  }
}

TEST_CASE("Radio cooperative merge rejects overflow without a partial commit",
          "[kokkos][radio][accumulator]") {
  for (unsigned kind = 0; kind < 5; ++kind) {
    auto first = snapshot(0), second = snapshot(1), bad = second;
    if (kind == 0) {
      first.first.coreas_[0] = first.second.coreas[0] =
          std::numeric_limits<std::int64_t>::max();
      bad.first.coreas_[0] = bad.second.coreas[0] = 1;
      second.first.coreas_[0] = second.second.coreas[0] = -1;
    } else if (kind == 1) {
      first.first.zhs_[0] = first.second.zhs[0] =
          std::numeric_limits<std::int64_t>::min();
      bad.first.zhs_[0] = bad.second.zhs[0] = -1;
      second.first.zhs_[0] = second.second.zhs[0] = 1;
    } else if (kind == 2) {
      first.first.counters_.validTracks_ = first.second.counters.valid_tracks =
          std::numeric_limits<unsigned long long>::max();
      bad.first.counters_.validTracks_ = bad.second.counters.valid_tracks = 1;
      second.first.counters_.validTracks_ = second.second.counters.valid_tracks = 0;
    } else if (kind == 3) {
      bad.first.counters_.trackLengthM_ = bad.second.counters.track_length_m =
          std::numeric_limits<double>::infinity();
    } else {
      bad.first.counters_.maximumSegmentLengthM_ =
          bad.second.counters.maximum_segment_length_m =
              std::numeric_limits<double>::quiet_NaN();
    }
    cr::CooperativeRadioMerge a(first.first.showerIdentity_, first.first.config_);
    ar::detail::CooperativeRadioMerge b(first.second.shower_identity,
                                        first.second.config);
    a.commit(first.first);
    b.commit(first.second);
    CHECK_THROWS(a.commit(bad.first));
    CHECK_THROWS(b.commit(bad.second));
    a.commit(second.first);
    b.commit(second.second);
    check_snapshot(a.take(), b.take());
  }
  auto max_first = snapshot(0), max_second = snapshot(1);
  max_first.first.counters_.maximumSegmentLengthM_ =
      max_first.second.counters.maximum_segment_length_m = 1.e308;
  max_second.first.counters_.maximumSegmentLengthM_ =
      max_second.second.counters.maximum_segment_length_m = 1.e308;
  auto combined =
      cr::merge_radio_counters(max_first.first.counters_, max_second.first.counters_);
  CHECK(combined.maximumSegmentLengthM_ == 1.e308);
  for (unsigned variant = 0; variant < 5; ++variant) {
    auto bad = snapshot(0);
    switch (variant) {
      case 0:
        bad.first.endpoint_ = static_cast<cp::CooperativeEndpoint>(2);
        bad.second.endpoint = static_cast<ar::detail::CooperativeEndpoint>(2);
        break;
      case 1:
        bad.first.counters_.fixedPointOverflows_ =
            bad.second.counters.fixed_point_overflows = 1;
        break;
      case 2:
        bad.first.zhs_.pop_back();
        bad.second.zhs.pop_back();
        break;
      case 3:
        bad.first.config_.deterministic_ = bad.second.config.deterministic = false;
        break;
      case 4:
        bad.first.config_.fixedPointFieldLimitVPerM_ =
            bad.second.config.fixed_point_field_limit_V_per_m = 0.;
        break;
    }
    CHECK_THROWS(cr::validate_fixed_radio(bad.first));
    CHECK_THROWS(ar::detail::validateFixedRadio(bad.second));
  }
}

TEST_CASE("Radio accumulation completes work on an explicit CUDA execution instance",
          "[kokkos][radio][accumulator]") {
  ensure_runtime();
#if defined(CORSIKA_RADIO_EXPECT_CUDA)
  cudaStream_t stream{};
  REQUIRE(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess);
  {
    execution_type ex(stream);
    auto spec = snapshot(0).first.config_;
    spec.coreasObservers_[0].numberOfBins_ = spec.zhsObservers_[0].numberOfBins_ = 512;
    spec.coreasObservers_[0].durationS_ = spec.zhsObservers_[0].durationS_ = 4.e-6;
    spec.coreasObservers_[0].sampleRateHz_ = spec.zhsObservers_[0].sampleRateHz_ = 128.e6;
    record_type track{};
    track.start_.pid_ = 11;
    track.start_.energyGeV_ = 1.;
    track.start_.weight_ = 1.e-4;
    track.start_.direction_[0] = 1.;
    track.end_ = track.start_;
    track.end_.positionM_[0] = .1;
    track.end_.timeS_ = .1 / (.99 * cr::SpeedOfLightMPerS);
    for (bool fixed : {false, true}) {
      accumulator_type a;
      reference_type b;
      a.initialize(spec, {}, ex);
      b.initialize(to_reference(spec), {}, ex);
      auto old_track = to_reference(track);
      a.enqueueHostLeptonTracks(&track, 1, ex);
      b.enqueueHostLeptonTracks(&old_track, 1, ex);
      if (fixed) {
        auto x = a.downloadFixed("stream", cp::CooperativeEndpoint::Cuda, ex);
        auto y = b.downloadFixed("stream", ar::detail::CooperativeEndpoint::Cuda, ex);
        check_snapshot(x, y);
      } else {
        auto x = a.download(ex);
        auto y = b.download(ex);
        CHECK(x.coreas_[0].x_ == y.coreas[0].x);
        CHECK(x.coreas_[0].y_ == y.coreas[0].y);
        CHECK(x.coreas_[0].z_ == y.coreas[0].z);
        CHECK(x.zhs_[0].x_ == y.zhs[0].x);
        CHECK(x.zhs_[0].y_ == y.zhs[0].y);
        CHECK(x.zhs_[0].z_ == y.zhs[0].z);
      }
      check_statistics(a.statistics(), b.statistics());
    }
    ex.fence("explicit radio stream finished");
  }
  REQUIRE(cudaStreamDestroy(stream) == cudaSuccess);
  std::cout << "explicit nonblocking CUDA radio stream checked\n";
#else
  CHECK(std::is_same_v<execution_type, Kokkos::OpenMP>);
#endif
}

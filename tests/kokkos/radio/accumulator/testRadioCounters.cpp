/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include "RadioAccumulatorChecks.hpp"
#include <Kokkos_Core.hpp>

#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <utility>

namespace {
  using Execution = Kokkos::DefaultExecutionSpace;
  using Accumulator = corsika::kokkos::radio::RadioAccumulator<Execution>;
  using Record = corsika::kokkos::LeptonTransportRecord;
  namespace detail = corsika::kokkos::radio;

  void require(bool condition, char const* message) {
    INFO(message);
    REQUIRE(condition);
  }

  auto configuration(std::size_t observers, bool coreas, bool zhs, bool separate) {
    corsika::kokkos::radio::RadioConfig config;
    config.enabled_ = true;
    config.coreasEnabled_ = coreas;
    config.zhsEnabled_ = zhs;
    config.deterministic_ = true;
    config.propagation_.homogeneousRefractiveIndex_ = 1.0003;
    for (std::size_t i = 0; i < observers; ++i) {
      corsika::kokkos::radio::RadioObserverSnapshot observer;
      observer.positionM_[0] = 30. + 0.1 * i;
      observer.positionM_[1] = 40.;
      observer.durationS_ = 1.e-6;
      observer.sampleRateHz_ = 1.e9;
      observer.numberOfBins_ = 1000;
      if (coreas) config.coreasObservers_.push_back(observer);
      if (separate) observer.positionM_[1] += 1.;
      if (zhs) config.zhsObservers_.push_back(observer);
    }
    return config;
  }

  Record record(std::size_t i) {
    Record value{};
    value.start_.pid_ = i % 2 ? -11 : 11;
    value.start_.energyGeV_ = 1.;
    value.start_.weight_ = 0.0001 * (1 + i % 5);
    value.start_.direction_[0] = 1.;
    value.start_.positionM_[0] = 0.001 * (i % 37);
    value.start_.timeS_ = 1.e-9 * (i % 17);
    value.end_ = value.start_;
    // Invalid zero-length records exercise empty lanes without skipping a
    // collective reduction. The final tile also has unused lanes.
    if (i % 19 != 0) {
      double const length = 0.1 + 0.01 * (i % 7);
      value.end_.positionM_[0] += length;
      value.end_.timeS_ += length / (0.99 * detail::SpeedOfLightMPerS);
    }
    return value;
  }

  void compare(detail::FixedRadioSnapshot const& expected,
               detail::FixedRadioSnapshot const& actual) {
    require(expected.coreas_ == actual.coreas_ && expected.zhs_ == actual.zhs_,
            "counter reduction changed fixed-point waveforms");
    auto const& a = expected.counters_;
    auto const& b = actual.counters_;
    require(a.validTracks_ == b.validTracks_ && a.validTracks_ > 0,
            "valid track count differs");
    require(a.coreasContributions_ == b.coreasContributions_ &&
                a.zhsContributions_ == b.zhsContributions_ &&
                a.zhsSubtracks_ == b.zhsSubtracks_ &&
                a.fixedPointOverflows_ == b.fixedPointOverflows_,
            "projection diagnostic counters differ");
    require(b.fixedPointOverflows_ == 0, "test waveform overflowed");
    auto nonzero = [](auto const& waveform) {
      return std::any_of(waveform.begin(), waveform.end(),
                         [](auto value) { return value != 0; });
    };
    if (!actual.coreas_.empty()) require(nonzero(actual.coreas_), "empty CoREAS signal");
    if (!actual.zhs_.empty()) require(nonzero(actual.zhs_), "empty ZHS signal");
  }

  void test(Execution const& execution, std::size_t observers, bool coreas, bool zhs,
            bool separate = false) {
    auto const config = configuration(observers, coreas, zhs, separate);
    auto const count = Accumulator::DirectProjectionTrackThreshold + 1;
    Accumulator::track_input_view_type records("radio_counter_records", count);
    auto host = Kokkos::create_mirror_view(records);
    for (std::size_t i = 0; i < count; ++i) host(i) = record(i);
    Kokkos::deep_copy(execution, records, host);

    Accumulator reference;
    reference.initialize(config, {}, execution);
    // Two small submissions use the unchanged direct GPU kernel. Their union
    // is exactly the same track list as the single tiled submission below.
    for (std::size_t begin = 0; begin < count;) {
      auto const end = std::min(count, begin + (count + 1) / 2);
      Accumulator::track_input_view_type part =
          Kokkos::subview(records, std::make_pair(begin, end));
      reference.accumulateLeptonTracks(part, end - begin, execution);
      begin = end;
    }
    auto const expected = reference.downloadFixed(
        "counter-reference", ::corsika::kokkos::CooperativeEndpoint::Cuda, execution);

    Accumulator tiled;
    tiled.initialize(config, {}, execution);
    tiled.accumulateLeptonTracks(records, 0, execution);
    tiled.accumulateLeptonTracks(records, count, execution);
    auto const actual = tiled.downloadFixed(
        "counter-tiled", ::corsika::kokkos::CooperativeEndpoint::Cuda, execution);
    compare(expected, actual);
    if constexpr (!Accumulator::IsOpenMP) {
      require(reference.statistics().directProjectionRecords_ == count,
              "reference did not exercise the direct kernel");
      require(tiled.statistics().trackPrecomputedRecords_ == count,
              "candidate did not exercise the tiled kernel");
    }
    std::cout << "observers=" << observers << " coreas=" << coreas << " zhs=" << zhs
              << " separate=" << separate << " passed\n";
  }
} // namespace

TEST_CASE("Production direct and tiled projection counters",
          "[kokkos][radio][accumulator]") {
  radio_accumulator_test::ensure_runtime();
  Execution execution;
  for (auto observers : {3u, 31u, 32u, 33u, 64u}) {
    test(execution, observers, true, false);
    test(execution, observers, false, true);
    test(execution, observers, true, true);
  }
  test(execution, 33, true, true, true);
}

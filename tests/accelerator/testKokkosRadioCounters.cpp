/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 * Distributed under the 3-clause BSD license; see LICENSE.
 */

#include <Kokkos_Core.hpp>
#include <corsika/accelerator/radio/kokkos/KokkosRadioAccumulator.hpp>

#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <utility>

namespace {
using Execution = Kokkos::DefaultExecutionSpace;
using Accumulator =
    corsika::accelerator::radio::kokkos_detail::KokkosRadioAccumulator<Execution>;
using Record = corsika::gpu::em::LeptonTransportRecord;
namespace detail = corsika::accelerator::radio::detail;

void require(bool condition, char const* message) {
  if (!condition) throw std::runtime_error(message);
}

auto configuration(std::size_t observers, bool coreas, bool zhs, bool separate) {
  corsika::gpu::radio::GpuRadioConfig config;
  config.enabled = true;
  config.coreas_enabled = coreas;
  config.zhs_enabled = zhs;
  config.deterministic = true;
  config.propagation.homogeneous_refractive_index = 1.0003;
  for (std::size_t i = 0; i < observers; ++i) {
    corsika::gpu::radio::RadioObserverSnapshot observer;
    observer.position_m[0] = 30. + 0.1 * i;
    observer.position_m[1] = 40.;
    observer.duration_s = 1.e-6;
    observer.sample_rate_Hz = 1.e9;
    observer.number_of_bins = 1000;
    if (coreas) config.coreas_observers.push_back(observer);
    if (separate) observer.position_m[1] += 1.;
    if (zhs) config.zhs_observers.push_back(observer);
  }
  return config;
}

Record record(std::size_t i) {
  Record value{};
  value.start.pid = i % 2 ? -11 : 11;
  value.start.energy_GeV = 1.;
  value.start.weight = 0.0001 * (1 + i % 5);
  value.start.direction[0] = 1.;
  value.start.position_m[0] = 0.001 * (i % 37);
  value.start.time_s = 1.e-9 * (i % 17);
  value.end = value.start;
  // Invalid zero-length records exercise empty lanes without skipping a
  // collective reduction. The final tile also has unused lanes.
  if (i % 19 != 0) {
    double const length = 0.1 + 0.01 * (i % 7);
    value.end.position_m[0] += length;
    value.end.time_s += length / (0.99 * detail::SpeedOfLightMPerS);
  }
  return value;
}

void compare(detail::FixedRadioSnapshot const& expected,
             detail::FixedRadioSnapshot const& actual) {
  require(expected.coreas == actual.coreas && expected.zhs == actual.zhs,
          "counter reduction changed fixed-point waveforms");
  auto const& a = expected.counters;
  auto const& b = actual.counters;
  require(a.valid_tracks == b.valid_tracks && a.valid_tracks > 0,
          "valid track count differs");
  require(a.coreas_contributions == b.coreas_contributions &&
              a.zhs_contributions == b.zhs_contributions &&
              a.zhs_subtracks == b.zhs_subtracks &&
              a.fixed_point_overflows == b.fixed_point_overflows,
          "projection diagnostic counters differ");
  require(b.fixed_point_overflows == 0, "test waveform overflowed");
  auto nonzero = [](auto const& waveform) {
    return std::any_of(waveform.begin(), waveform.end(),
                       [](auto value) { return value != 0; });
  };
  if (!actual.coreas.empty()) require(nonzero(actual.coreas), "empty CoREAS signal");
  if (!actual.zhs.empty()) require(nonzero(actual.zhs), "empty ZHS signal");
}

void test(Execution const& execution, std::size_t observers,
          bool coreas, bool zhs, bool separate = false) {
  auto const config = configuration(observers, coreas, zhs, separate);
  auto const count = Accumulator::DirectProjectionTrackThreshold + 1;
  Accumulator::TrackInputView records("radio_counter_records", count);
  auto host = Kokkos::create_mirror_view(records);
  for (std::size_t i = 0; i < count; ++i) host(i) = record(i);
  Kokkos::deep_copy(execution, records, host);

  Accumulator reference;
  reference.initialize(config, {}, execution);
  // Two small submissions use the unchanged direct GPU kernel. Their union
  // is exactly the same track list as the single tiled submission below.
  for (std::size_t begin = 0; begin < count;) {
    auto const end = std::min(count, begin + (count + 1) / 2);
    Accumulator::TrackInputView part =
        Kokkos::subview(records, std::make_pair(begin, end));
    reference.accumulateLeptonTracks(part, end - begin, execution);
    begin = end;
  }
  auto const expected = reference.downloadFixed(
      "counter-reference", detail::CooperativeEndpoint::Cuda, execution);

  Accumulator tiled;
  tiled.initialize(config, {}, execution);
  tiled.accumulateLeptonTracks(records, 0, execution);
  tiled.accumulateLeptonTracks(records, count, execution);
  auto const actual = tiled.downloadFixed(
      "counter-tiled", detail::CooperativeEndpoint::Cuda, execution);
  compare(expected, actual);
  if constexpr (!Accumulator::IsOpenMP) {
    require(reference.statistics().direct_projection_records == count,
            "reference did not exercise the direct kernel");
    require(tiled.statistics().track_precomputed_records == count,
            "candidate did not exercise the tiled kernel");
  }
  std::cout << "observers=" << observers << " coreas=" << coreas
            << " zhs=" << zhs << " separate=" << separate << " passed\n";
}
} // namespace

int main(int argc, char** argv) {
  Kokkos::ScopeGuard kokkos(argc, argv);
  try {
    Execution execution;
    // Below, at and above the default GPU observer tile width (32), plus
    // partial/multiple tiles and the original sparse-observer workload.
    for (auto observers : {3u, 31u, 32u, 33u, 64u}) {
      test(execution, observers, true, false);
      test(execution, observers, false, true);
      test(execution, observers, true, true);
    }
    test(execution, 33, true, true, true);
    return 0;
  } catch (std::exception const& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}

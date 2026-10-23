/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 * Distributed under the 3-clause BSD license; see LICENSE.
 */

#include <Kokkos_Core.hpp>
#include <corsika/accelerator/radio/kokkos/KokkosRadioAccumulator.hpp>

#include <iostream>
#include <stdexcept>
#include <vector>

namespace {
using Execution = Kokkos::DefaultExecutionSpace;
using Accumulator =
    corsika::accelerator::radio::kokkos_detail::KokkosRadioAccumulator<Execution>;
using Record = corsika::gpu::em::LeptonTransportRecord;

void require(bool condition, char const* message) {
  if (!condition) throw std::runtime_error(message);
}

template <class Action> void requireRejected(Action action, char const* message) {
  bool rejected = false;
  try { action(); } catch (std::logic_error const&) { rejected = true; }
  require(rejected, message);
}

corsika::gpu::radio::GpuRadioConfig configuration() {
  corsika::gpu::radio::GpuRadioConfig config;
  config.enabled = true;
  config.deterministic = true;
  config.propagation.homogeneous_refractive_index = 1.0003;
  corsika::gpu::radio::RadioObserverSnapshot observer;
  observer.position_m[0] = 30.;
  observer.position_m[1] = 40.;
  observer.duration_s = 2.e-6;
  observer.sample_rate_Hz = 1.e9;
  observer.number_of_bins = 2000;
  config.coreas_observers.push_back(observer);
  config.zhs_observers.push_back(observer);
  return config;
}

Record record() {
  Record value{};
  value.start.pid = static_cast<std::int32_t>(corsika::gpu::em::EmPid::Electron);
  value.start.energy_GeV = 1.;
  value.start.weight = 0.001;
  value.start.direction[0] = 1.;
  value.end = value.start;
  value.end.position_m[0] = 0.1;
  value.end.time_s = 0.1 / (0.99 *
      corsika::accelerator::radio::detail::SpeedOfLightMPerS);
  return value;
}

KOKKOS_INLINE_FUNCTION Record varyingRecord(std::size_t index) {
  Record value{};
  value.start.pid = index % 2 ? -11 : 11;
  value.start.energy_GeV = 1. + 0.01 * (index % 7);
  value.start.weight = 0.0001 * (1 + index % 5);
  value.start.direction[0] = 1.;
  value.start.position_m[0] = 0.001 * (index % 37);
  value.start.position_m[1] = 0.001 * (index % 23);
  value.start.time_s = 1.e-9 * (index % 17);
  value.end = value.start;
  value.end.position_m[0] += 0.1;
  value.end.time_s += 0.1 / (0.99 *
      corsika::accelerator::radio::detail::SpeedOfLightMPerS);
  return value;
}

void requireSame(corsika::gpu::radio::GpuRadioWaveforms const& a,
                 corsika::gpu::radio::GpuRadioWaveforms const& b) {
  auto compare = [](auto const& left, auto const& right) {
    require(left.size() == right.size(), "observer count changed");
    bool nonzero = false;
    for (std::size_t i = 0; i < left.size(); ++i) {
      require(left[i].x == right[i].x && left[i].y == right[i].y &&
                  left[i].z == right[i].z,
              "host/device mixed append changed fixed-point radio output");
      for (double v : left[i].y) nonzero = nonzero || v != 0.;
    }
    require(nonzero, "test radio waveform is empty");
  };
  compare(a.coreas, b.coreas);
  compare(a.zhs, b.zhs);
}

void test(Execution const& execution) {
  auto const config = configuration();
  auto const count = Accumulator::HostTrackBatchCapacity + 8;
  std::vector<Record> input(count, record());
  Accumulator::TrackInputView device("host_radio_reference", count);
  auto host = Kokkos::create_mirror_view(device);
  for (std::size_t i = 0; i < count; ++i) host(i) = input[i];
  Kokkos::deep_copy(execution, device, host);
  execution.fence("prepare host radio reference");

  Accumulator reference;
  reference.initialize(config, {}, execution);
  require(reference.deviceBytes() == Accumulator::projectedDeviceBytes(config),
          "bounded host staging missing from initial memory projection");
  reference.accumulateLeptonTracks(device, count, execution);
  auto const expected = reference.download(execution);

  Accumulator mixed;
  mixed.initialize(config, {}, execution);
  // Interleave a native-device record with queued CPU records. Neither append
  // is permitted to clear the waveform already accumulated by the other path.
  mixed.enqueueHostLeptonTracks(input.data(), 3, execution);
  requireRejected([&] { mixed.reset(execution); }, "reset lost queued host tracks");
  mixed.accumulateLeptonTracks(device, 1, execution);
  for (std::size_t i = 3; i < count - 1; ++i)
    mixed.enqueueHostLeptonTracks(&input[i], 1, execution);
  require(mixed.statistics().host_track_batches == 1,
          "full host batch was not submitted exactly once");
  auto const actual = mixed.download(execution); // flush the seven-record tail
  requireSame(expected, actual);
  require(mixed.statistics().host_tracks_enqueued == count - 1 &&
              mixed.statistics().host_track_batches == 2 &&
              mixed.statistics().lepton_tracks == count,
          "host radio batching lost or duplicated tracks");
  require(mixed.statistics().host_to_device_bytes == (count - 1) * sizeof(Record),
          "host radio transfer accounting mismatch");
  requireRejected([&] { mixed.enqueueHostLeptonTracks(input.data(), 1, execution); },
                  "host tracks accepted after final waveform download");
  auto const retained = mixed.deviceBytes();
  mixed.reset(execution);
  require(mixed.deviceBytes() == retained && mixed.statistics().host_tracks_enqueued == 0,
          "new shower did not reuse storage/reset counters");
  mixed.enqueueHostLeptonTracks(input.data(), count, execution);
  requireSame(expected, mixed.download(execution));

  // The cooperative fixed-point export must also flush a partial host batch.
  Accumulator fixed;
  fixed.initialize(config, {}, execution);
  fixed.enqueueHostLeptonTracks(input.data(), 3, execution);
  auto const snapshot = fixed.downloadFixed(
      "host-radio-test", corsika::accelerator::radio::detail::CooperativeEndpoint::Cuda,
      execution);
  require(fixed.statistics().host_track_batches == 1 &&
              fixed.statistics().lepton_tracks == 3 && !snapshot.coreas.empty(),
          "fixed radio export lost the partial host batch");
  execution.fence("complete host radio staging test");
}

// A queued producer stands in for the transport kernel. It rewrites the same
// device-record buffer between projections; host batches must neither touch
// that buffer nor race with projection scratch/waveform reuse. There is no
// explicit fence in the submission loop. Real transport is covered separately
// by full-shower baseline comparisons.
void testInterleavedProducer(Execution const& execution) {
  auto const config = configuration();
  constexpr std::size_t rounds = 4;
  constexpr std::size_t hostCount = Accumulator::HostTrackBatchCapacity + 31;
  Accumulator::TrackInputView device("queued_transport_records", 600);
  Kokkos::View<std::size_t*, typename Execution::memory_space>
      transportState("transport_state_not_owned_by_radio", 600);
  std::vector<Record> all;
  for (std::size_t round = 0; round < rounds; ++round) {
    auto const gpuCount = round % 2 ? 73u : 600u;
    auto const size = gpuCount + hostCount + 11;
    for (std::size_t i = 0; i < size; ++i) all.push_back(varyingRecord(all.size()));
  }
  Accumulator::TrackInputView referenceInput("interleaved_radio_reference", all.size());
  auto mirror = Kokkos::create_mirror_view(referenceInput);
  for (std::size_t i = 0; i < all.size(); ++i) mirror(i) = all[i];
  Kokkos::deep_copy(execution, referenceInput, mirror);
  Accumulator reference;
  reference.initialize(config, {}, execution);
  reference.accumulateLeptonTracks(referenceInput, all.size(), execution);
  auto const expected = reference.download(execution);

  Accumulator mixed;
  mixed.initialize(config, {}, execution);
  std::size_t offset = 0;
  std::size_t lastProducerOffset = 0;
  for (std::size_t round = 0; round < rounds; ++round) {
    auto const gpuCount = round % 2 ? 73u : 600u;
    auto submitProducer = [&](std::size_t count) {
      auto const first = offset;
      lastProducerOffset = first;
      Kokkos::parallel_for("transport_like_radio_record_producer",
          Kokkos::RangePolicy<Execution>(execution, 0, device.extent(0)),
          KOKKOS_LAMBDA(std::size_t i) {
            transportState(i) = first + i;
            device(i) = varyingRecord(first + i);
          });
      mixed.accumulateLeptonTracks(device, count, execution);
      offset += count;
    };
    submitProducer(gpuCount);
    std::vector<Record> transient(all.begin() + offset,
                                  all.begin() + offset + hostCount);
    mixed.enqueueHostLeptonTracks(transient.data(), hostCount, execution);
    // A CPU caller may reuse its storage immediately after enqueue returns.
    for (auto& item : transient) item = Record{};
    offset += hostCount;
    submitProducer(11);
  }
  require(offset == all.size(), "interleaved test input bookkeeping failed");
  requireSame(expected, mixed.download(execution));
  auto state = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, transportState);
  auto retained = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, device);
  for (std::size_t i = 0; i < device.extent(0); ++i) {
    auto const wanted = varyingRecord(lastProducerOffset + i);
    require(state(i) == lastProducerOffset + i &&
                retained(i).start.pid == wanted.start.pid &&
                retained(i).start.weight == wanted.start.weight &&
                retained(i).end.time_s == wanted.end.time_s,
            "host radio submission modified transport-owned state");
  }
  auto const hostTotal = rounds * hostCount;
  require(mixed.statistics().host_tracks_enqueued == hostTotal &&
              mixed.statistics().lepton_tracks == all.size() &&
              mixed.statistics().host_track_batches ==
                  (hostTotal + Accumulator::HostTrackBatchCapacity - 1) /
                      Accumulator::HostTrackBatchCapacity,
          "interleaved producer lost or duplicated a radio batch");
}
} // namespace

int main(int argc, char** argv) {
  Kokkos::ScopeGuard guard(argc, argv);
  try {
    test(Execution{});
    testInterleavedProducer(Execution{});
    std::cout << "host radio append, batching, reset, memory, fixed export and queued producer passed\n";
    return 0;
  } catch (std::exception const& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}

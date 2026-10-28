/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include "RadioAccumulatorChecks.hpp"
#include <Kokkos_Core.hpp>

#include <iostream>
#include <stdexcept>
#include <vector>

namespace {
  using Execution = Kokkos::DefaultExecutionSpace;
  using Accumulator = corsika::kokkos::radio::RadioAccumulator<Execution>;
  using Record = corsika::kokkos::LeptonTransportRecord;

  void require(bool condition, char const* message) {
    INFO(message);
    REQUIRE(condition);
  }

  template <class Action>
  void requireRejected(Action action, char const* message) {
    bool rejected = false;
    try {
      action();
    } catch (std::logic_error const&) { rejected = true; }
    require(rejected, message);
  }

  corsika::kokkos::radio::RadioConfig configuration() {
    corsika::kokkos::radio::RadioConfig config;
    config.enabled_ = true;
    config.deterministic_ = true;
    config.propagation_.homogeneousRefractiveIndex_ = 1.0003;
    corsika::kokkos::radio::RadioObserverSnapshot observer;
    observer.positionM_[0] = 30.;
    observer.positionM_[1] = 40.;
    observer.durationS_ = 2.e-6;
    observer.sampleRateHz_ = 1.e9;
    observer.numberOfBins_ = 2000;
    config.coreasObservers_.push_back(observer);
    config.zhsObservers_.push_back(observer);
    return config;
  }

  Record record() {
    Record value{};
    value.start_.pid_ = static_cast<std::int32_t>(corsika::kokkos::EmPid::Electron);
    value.start_.energyGeV_ = 1.;
    value.start_.weight_ = 0.001;
    value.start_.direction_[0] = 1.;
    value.end_ = value.start_;
    value.end_.positionM_[0] = 0.1;
    value.end_.timeS_ = 0.1 / (0.99 * corsika::kokkos::radio::SpeedOfLightMPerS);
    return value;
  }

  KOKKOS_INLINE_FUNCTION Record varyingRecord(std::size_t index) {
    Record value{};
    value.start_.pid_ = index % 2 ? -11 : 11;
    value.start_.energyGeV_ = 1. + 0.01 * (index % 7);
    value.start_.weight_ = 0.0001 * (1 + index % 5);
    value.start_.direction_[0] = 1.;
    value.start_.positionM_[0] = 0.001 * (index % 37);
    value.start_.positionM_[1] = 0.001 * (index % 23);
    value.start_.timeS_ = 1.e-9 * (index % 17);
    value.end_ = value.start_;
    value.end_.positionM_[0] += 0.1;
    value.end_.timeS_ += 0.1 / (0.99 * corsika::kokkos::radio::SpeedOfLightMPerS);
    return value;
  }

  void requireSame(corsika::kokkos::radio::RadioWaveforms const& a,
                   corsika::kokkos::radio::RadioWaveforms const& b) {
    auto compare = [](auto const& left, auto const& right) {
      require(left.size() == right.size(), "observer count changed");
      bool nonzero = false;
      for (std::size_t i = 0; i < left.size(); ++i) {
        require(left[i].x_ == right[i].x_ && left[i].y_ == right[i].y_ &&
                    left[i].z_ == right[i].z_,
                "host/device mixed append changed fixed-point radio output");
        for (double v : left[i].y_) nonzero = nonzero || v != 0.;
      }
      require(nonzero, "test radio waveform is empty");
    };
    compare(a.coreas_, b.coreas_);
    compare(a.zhs_, b.zhs_);
  }

  void test(Execution const& execution) {
    auto const config = configuration();
    auto const count = Accumulator::HostTrackBatchCapacity + 8;
    std::vector<Record> input(count, record());
    Accumulator::track_input_view_type device("host_radio_reference", count);
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
    require(mixed.statistics().hostTrackBatches_ == 1,
            "full host batch was not submitted exactly once");
    auto const actual = mixed.download(execution); // flush the seven-record tail
    requireSame(expected, actual);
    require(mixed.statistics().hostTracksEnqueued_ == count - 1 &&
                mixed.statistics().hostTrackBatches_ == 2 &&
                mixed.statistics().leptonTracks_ == count,
            "host radio batching lost or duplicated tracks");
    require(mixed.statistics().hostToDeviceBytes_ == (count - 1) * sizeof(Record),
            "host radio transfer accounting mismatch");
    requireRejected([&] { mixed.enqueueHostLeptonTracks(input.data(), 1, execution); },
                    "host tracks accepted after final waveform download");
    auto const retained = mixed.deviceBytes();
    mixed.reset(execution);
    require(
        mixed.deviceBytes() == retained && mixed.statistics().hostTracksEnqueued_ == 0,
        "new shower did not reuse storage/reset counters");
    mixed.enqueueHostLeptonTracks(input.data(), count, execution);
    requireSame(expected, mixed.download(execution));

    // The cooperative fixed-point export must also flush a partial host batch.
    Accumulator fixed;
    fixed.initialize(config, {}, execution);
    fixed.enqueueHostLeptonTracks(input.data(), 3, execution);
    auto const snapshot = fixed.downloadFixed(
        "host-radio-test", corsika::kokkos::CooperativeEndpoint::Cuda, execution);
    require(fixed.statistics().hostTrackBatches_ == 1 &&
                fixed.statistics().leptonTracks_ == 3 && !snapshot.coreas_.empty(),
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
    Accumulator::track_input_view_type device("queued_transport_records", 600);
    Kokkos::View<std::size_t*, typename Execution::memory_space> transportState(
        "transport_state_not_owned_by_radio", 600);
    std::vector<Record> all;
    for (std::size_t round = 0; round < rounds; ++round) {
      auto const gpuCount = round % 2 ? 73u : 600u;
      auto const size = gpuCount + hostCount + 11;
      for (std::size_t i = 0; i < size; ++i) all.push_back(varyingRecord(all.size()));
    }
    Accumulator::track_input_view_type referenceInput("interleaved_radio_reference",
                                                      all.size());
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
        Kokkos::parallel_for(
            "transport_like_radio_record_producer",
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
                  retained(i).start_.pid_ == wanted.start_.pid_ &&
                  retained(i).start_.weight_ == wanted.start_.weight_ &&
                  retained(i).end_.timeS_ == wanted.end_.timeS_,
              "host radio submission modified transport-owned state");
    }
    auto const hostTotal = rounds * hostCount;
    require(mixed.statistics().hostTracksEnqueued_ == hostTotal &&
                mixed.statistics().leptonTracks_ == all.size() &&
                mixed.statistics().hostTrackBatches_ ==
                    (hostTotal + Accumulator::HostTrackBatchCapacity - 1) /
                        Accumulator::HostTrackBatchCapacity,
            "interleaved producer lost or duplicated a radio batch");
  }
} // namespace

TEST_CASE("Production host staging and queued producer", "[kokkos][radio][accumulator]") {
  radio_accumulator_test::ensure_runtime();
  test(Execution{});
  testInterleavedProducer(Execution{});
}

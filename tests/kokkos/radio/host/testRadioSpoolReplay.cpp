/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include <corsika/kokkos/radio/HostRadioCollector.hpp>
#include <corsika/kokkos/radio/RadioTrackSpoolReader.hpp>
#include <corsika/kokkos/radio/RadioTrackSpoolWriter.hpp>
#include "../accumulator/RadioAccumulatorChecks.hpp"
#include <cstdlib>
namespace {
  using namespace radio_accumulator_test;
  struct Backend {
    accumulator_type& accumulator_;
    execution_type execution_;
    bool gpuRadioEnabled() const { return accumulator_.enabled(); }
    void enqueueRadioTracks(boost::span<record_type const> records) {
      accumulator_.enqueueHostLeptonTracks(records.data(), records.size(), execution_);
    }
  };
  struct Directory {
    std::filesystem::path path_;
    Directory() {
      char name[] = "/tmp/c8-radio-replay-XXXXXX";
      auto created = mkdtemp(name);
      if (!created) throw std::runtime_error("test directory");
      path_ = created;
    }
    ~Directory() {
      std::error_code e;
      std::filesystem::remove_all(path_, e);
    }
  };
} // namespace
TEST_CASE("Completed host tracks replay in bounded batches without changing radio",
          "[kokkos][radio][host]") {
  ensure_runtime();
  execution_type execution;
  cr::RadioConfig config;
  config.enabled_ = true;
  config.deterministic_ = true;
  config.propagation_.homogeneousRefractiveIndex_ = 1.0003;
  cr::RadioObserverSnapshot observer;
  observer.positionM_[0] = 30.;
  observer.positionM_[1] = 40.;
  observer.durationS_ = 2.e-6;
  observer.sampleRateHz_ = 1.e9;
  observer.numberOfBins_ = 2000;
  config.coreasObservers_.push_back(observer);
  config.zhsObservers_.push_back(observer);
  Directory tmp;
  auto path = tmp.path_ / "input.bin";
  accumulator_type direct;
  direct.initialize(config, {}, execution);
  Backend directBackend{direct, execution};
  cr::HostRadioCollector collector(directBackend);
  constexpr std::size_t count = 2 * accumulator_type::HostTrackBatchCapacity + 7;
  {
    cr::RadioTrackSpoolWriter writer(path);
    for (std::size_t i = 0; i < count; ++i) {
      corsika::radio::RadioTrackSegment t{};
      t.pdg_ = i % 2 ? -11 : 11;
      t.weight_ = 1.e-4 * (1 + i % 5);
      t.startKineticEnergyGeV_ = 1. + 0.01 * (i % 7);
      t.endKineticEnergyGeV_ = t.startKineticEnergyGeV_;
      t.startPositionM_[0] = 0.001 * (i % 37);
      t.endPositionM_[0] = t.startPositionM_[0] + 0.1;
      t.startDirection_[0] = t.endDirection_[0] = 1.;
      t.startTimeS_ = 1.e-9 * (i % 17);
      t.endTimeS_ = t.startTimeS_ + 0.1 / (0.99 * cr::SpeedOfLightMPerS);
      writer.append(t);
      collector.append(t);
    }
    writer.complete();
  }
  auto expected = direct.download(execution);
  REQUIRE(direct.statistics().hostTracksEnqueued_ == count);
  bool nonzero = false;
  for (auto v : expected.coreas_[0].y_) nonzero = nonzero || v != 0.;
  REQUIRE(nonzero);
  for (auto limit :
       {std::size_t{1}, std::size_t{7}, std::size_t{4096}, std::size_t{65536}}) {
    INFO(limit);
    accumulator_type replay;
    replay.initialize(config, {}, execution);
    Backend backend{replay, execution};
    cr::HostRadioCollector sink(backend);
    cr::RadioTrackSpoolReader reader(path);
    std::vector<corsika::radio::RadioTrackSegment> batch;
    while (reader.readBatch(batch, limit)) {
      REQUIRE(batch.size() <= limit);
      for (auto const& track : batch) sink.append(track);
    }
    REQUIRE(reader.finished());
    REQUIRE(reader.consumed() == count);
    auto actual = replay.download(execution);
    for (auto const& pair : {std::make_pair(&actual.coreas_, &expected.coreas_),
                             std::make_pair(&actual.zhs_, &expected.zhs_)}) {
      REQUIRE(pair.first->size() == pair.second->size());
      for (std::size_t i = 0; i < pair.first->size(); ++i) {
        REQUIRE(pair.first->at(i).x_ == pair.second->at(i).x_);
        REQUIRE(pair.first->at(i).y_ == pair.second->at(i).y_);
        REQUIRE(pair.first->at(i).z_ == pair.second->at(i).z_);
      }
    }
    REQUIRE(replay.statistics().hostTracksEnqueued_ == count);
    REQUIRE(replay.statistics().leptonTracks_ == count);
    REQUIRE(replay.statistics().hostTrackBatches_ == 3);
  }
}

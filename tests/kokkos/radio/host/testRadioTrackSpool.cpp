/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include <corsika/kokkos/radio/RadioTrackSpoolReader.hpp>
#include <corsika/kokkos/radio/RadioTrackSpoolWriter.hpp>

#include <catch2/catch_all.hpp>
#include <cstdlib>
#include <iostream>

namespace fs = std::filesystem;
using corsika::kokkos::radio::RadioTrackSpoolReader;
using corsika::kokkos::radio::RadioTrackSpoolWriter;
using corsika::radio::RadioTrackSegment;

namespace {
  void check(bool ok, char const* message) {
    INFO(message);
    REQUIRE(ok);
  }
  template <typename Action>
  void rejects(Action action, char const* message) {
    bool failed = false;
    try {
      action();
    } catch (std::exception const&) { failed = true; }
    check(failed, message);
  }
  struct Directory {
    fs::path path;
    Directory() {
      char name[] = "/tmp/c8-radio-spool-XXXXXX";
      auto const created = mkdtemp(name);
      check(created != nullptr, "Cannot create test directory");
      path = created;
    }
    ~Directory() {
      std::error_code ignored;
      fs::remove_all(path, ignored);
    }
  };
  RadioTrackSegment track(unsigned index) {
    RadioTrackSegment result{};
    result.pdg_ = index % 2 ? -11 : 11;
    result.weight_ = 0.5 + index;
    result.startTimeS_ = index * 1.e-7;
    result.endTimeS_ = result.startTimeS_ + 3.e-9;
    result.startKineticEnergyGeV_ = 1.25;
    result.endKineticEnergyGeV_ = 1.2;
    for (unsigned axis = 0; axis < 3; ++axis) {
      result.startPositionM_[axis] = axis + index * 0.25;
      result.endPositionM_[axis] = result.startPositionM_[axis] + 0.5;
      result.startDirection_[axis] = axis == 2 ? 1. : 0.;
      result.endDirection_[axis] = result.startDirection_[axis];
    }
    return result;
  }
  void equal(RadioTrackSegment const& a, RadioTrackSegment const& b) {
    check(a.pdg_ == b.pdg_ && a.weight_ == b.weight_ && a.startTimeS_ == b.startTimeS_ &&
              a.endTimeS_ == b.endTimeS_ &&
              a.startKineticEnergyGeV_ == b.startKineticEnergyGeV_ &&
              a.endKineticEnergyGeV_ == b.endKineticEnergyGeV_,
          "Radio spool changed a scalar field");
    for (unsigned axis = 0; axis < 3; ++axis)
      check(a.startPositionM_[axis] == b.startPositionM_[axis] &&
                a.endPositionM_[axis] == b.endPositionM_[axis] &&
                a.startDirection_[axis] == b.startDirection_[axis] &&
                a.endDirection_[axis] == b.endDirection_[axis],
            "Radio spool changed a coordinate");
  }
} // namespace

TEST_CASE("testRadioTrackSpool production contract", "[kokkos][radio][host]") {
  try {
    Directory directory;
    auto const valid = directory.path / "tracks.bin";
    {
      RadioTrackSpoolWriter output(valid);
      for (unsigned i = 0; i < 103; ++i) output.append(track(i));
      check(output.count() == 103, "Wrong written count");
      output.complete();
      rejects([&] { output.complete(); }, "Accepted duplicate completion");
      rejects([&] { output.append(track(1)); }, "Accepted post-completion append");
    }
    rejects([&] { RadioTrackSpoolWriter duplicate(valid); }, "Overwrote existing spool");
    {
      RadioTrackSpoolReader input(valid);
      check(input.count() == 103 && !input.finished(), "Wrong initial reader state");
      std::vector<RadioTrackSegment> batch;
      rejects([&] { input.readBatch(batch, 0); }, "Accepted zero batch limit");
      rejects([&] { input.readBatch(batch, 65537); }, "Accepted unbounded batch limit");
      unsigned count = 0;
      while (input.readBatch(batch, 7)) {
        check(batch.size() <= 7, "Read exceeds batch limit");
        for (auto const& record : batch) equal(record, track(count++));
      }
      check(count == 103 && input.consumed() == input.count() && input.finished(),
            "Lost or duplicated tracks");
    }
    auto const empty = directory.path / "empty.bin";
    {
      RadioTrackSpoolWriter output(empty);
      auto invalid = track(1);
      invalid.pdg_ = 13;
      rejects([&] { output.append(invalid); }, "Accepted muon track");
      invalid = track(1);
      invalid.weight_ = -1.;
      rejects([&] { output.append(invalid); }, "Accepted negative weight");
      invalid = track(1);
      invalid.endTimeS_ = -1.;
      rejects([&] { output.append(invalid); }, "Accepted reversed time");
      invalid = track(1);
      invalid.startPositionM_[1] = std::numeric_limits<double>::quiet_NaN();
      rejects([&] { output.append(invalid); }, "Accepted non-finite coordinates");
      output.complete();
    }
    {
      RadioTrackSpoolReader input(empty);
      std::vector<RadioTrackSegment> batch;
      check(input.finished() && input.count() == 0 && input.readBatch(batch) == 0,
            "Closed empty spool was not accepted");
    }
    auto const incomplete = directory.path / "incomplete.bin";
    {
      RadioTrackSpoolWriter output(incomplete);
      output.append(track(1));
    }
    rejects([&] { RadioTrackSpoolReader input(incomplete); },
            "Accepted incomplete spool");
    for (auto const& name :
         {"bad-magic", "bad-version", "bad-size", "short", "extra", "bad-value"}) {
      auto const malformed = directory.path / name;
      fs::copy_file(valid, malformed);
      if (std::string(name) == "short")
        fs::resize_file(malformed, fs::file_size(malformed) - 1);
      else if (std::string(name) == "extra")
        fs::resize_file(malformed, fs::file_size(malformed) + 1);
      else {
        std::fstream file(malformed, std::ios::binary | std::ios::in | std::ios::out);
        auto const offset = std::string(name) == "bad-magic"
                                ? 0
                                : std::string(name) == "bad-version"
                                      ? 16
                                      : std::string(name) == "bad-size" ? 20 : 32 + 4;
        file.seekp(offset);
        if (std::string(name) == "bad-value")
          corsika::kokkos::radio::radio_spool::write_double(
              file, std::numeric_limits<double>::quiet_NaN());
        else
          file.put('!');
      }
      rejects(
          [&] {
            RadioTrackSpoolReader input(malformed);
            std::vector<RadioTrackSegment> batch;
            input.readBatch(batch);
          },
          "Accepted malformed spool");
    }
    std::cout
        << "Radio track spool: round-trip, bounds and malformed-input checks passed\n";
    return;
  } catch (std::exception const& error) {
    std::cerr << error.what() << '\n';
    FAIL(error.what());
  }
}

/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */
#include <corsika/modules/radio/RadioProcess.hpp>
#include <corsika/modules/radio/detectors/ObserverCollection.hpp>
#include <corsika/modules/radio/observers/TimeDomainObserver.hpp>
#include "ProductionRadioProcess.hpp"
#include <catch2/catch_all.hpp>
#include <parquet/stream_reader.h>
#include <type_traits>

namespace {
  using namespace corsika;
  using Detector = ObserverCollection<TimeDomainObserver>;
  struct Propagator {};
  struct Directory {
    boost::filesystem::path path_ =
        boost::filesystem::unique_path("radio-output-%%%%-%%%%");
    Directory() { boost::filesystem::create_directory(path_); }
    ~Directory() {
      boost::system::error_code ignored;
      boost::filesystem::remove_all(path_, ignored);
    }
  };
  template <bool Zhs, bool Production>
  class Radio final
      : public std::conditional_t<
            Production,
            ProductionRadioProcess<Detector, Radio<Zhs, Production>, Propagator>,
            RadioProcess<Detector, Radio<Zhs, Production>, Propagator>> {
    using Base = std::conditional_t<
        Production, ProductionRadioProcess<Detector, Radio<Zhs, Production>, Propagator>,
        RadioProcess<Detector, Radio<Zhs, Production>, Propagator>>;

  public:
    using Base::Base;
    static constexpr auto algorithm = Zhs ? "ZHS" : "CoREAS";
    unsigned calls_{};
    template <typename Particle>
    ProcessReturn simulate(Step<Particle> const&) {
      ++calls_;
      return ProcessReturn::Ok;
    }
  };
  struct Particle {
    Point position_;
    DirectionVector direction_;
    Code pid_{Code::Electron};
    Point const& getPosition() const { return position_; }
    DirectionVector const& getDirection() const { return direction_; }
    TimeType getTime() const { return 3_ns; }
    HEPEnergyType getKineticEnergy() const { return 2_GeV; }
    Code getPID() const { return pid_; }
    double getWeight() const { return 2.5; }
  };
  struct Sink : radio::RadioTrackSink {
    unsigned calls_{};
    void append(radio::RadioTrackSegment const&) override { ++calls_; }
  };
  struct Row {
    unsigned id_;
    double time_, x_, y_, z_;
  };
  std::vector<Row> read_rows(boost::filesystem::path const& path) {
    parquet::StreamReader reader(parquet::ParquetFileReader::OpenFile(path.string()));
    std::vector<Row> rows;
    while (!reader.eof()) {
      Row r{};
      reader >> r.id_ >> r.time_ >> r.x_ >> r.y_ >> r.z_ >> parquet::EndRow;
      rows.push_back(r);
    }
    return rows;
  }
  template <bool Zhs, bool Production>
  YAML::Node exercise(boost::filesystem::path const& directory, bool empty = false) {
    auto root = get_root_CoordinateSystem();
    Detector detector;
    if (!empty)
      for (unsigned i = 0; i < 2; ++i) {
        TimeDomainObserver observer("antenna" + std::to_string(i),
                                    Point(root, i * 1_m, 0_m, 0_m), root, 0_ns, 3_ns,
                                    1_GHz, 0_ns);
        detector.addObserver(observer);
      }
    Propagator prop;
    Radio<Zhs, Production> process(detector, prop);
    boost::filesystem::create_directory(directory);
    process.startOfLibrary(directory);
    Particle particle{Point(root, 0_m, 0_m, 10_m),
                      DirectionVector(root, {0.6, 0., -0.8})};
    Line line(particle.position_, particle.direction_ * (0.9 * constants::c));
    StraightTrajectory trajectory(line, 1_ns);
    Step step(particle, trajectory);
    SignalPath path(0_ns, 1., 1., 1., DirectionVector(root, {0., 0., 1.}),
                    DirectionVector(root, {0., 0., -1.}), 1_m, {});
    for (unsigned event = 0; event < 3; ++event) {
      Sink sink;
      if (event == 0) {
        process.doContinuous(step, false);
        particle.pid_ = Code::Positron;
        process.doContinuous(step, false);
        particle.pid_ = Code::MuMinus;
        process.doContinuous(step, false);
        CHECK(process.calls_ == (empty ? 0 : 2));
      } else if (event == 1) {
        particle.pid_ = Code::Electron;
        process.setTrackSink(&sink, root);
        process.doContinuous(step, false);
        process.doContinuous(step, false);
        CHECK(process.offloadedSegmentCount() == (empty ? 0 : 2));
        CHECK(sink.calls_ == (empty ? 0 : 2));
        process.setTrackSink(nullptr, {});
      }
      // Deterministic observer inputs isolate writing/conversion from emission.
      // The accepted projection and accumulator tests cover the latter separately.
      if (event < 2)
        for (auto& observer : detector.getObservers())
          for (unsigned k = 0; k < 4; ++k) {
            double value = (event + 1) * (k * k + 1.);
            if constexpr (Zhs) {
              VectorPotential field(root,
                                    {value * 1_V * 1_s / 1_m, 2 * value * 1_V * 1_s / 1_m,
                                     -value * 1_V * 1_s / 1_m});
              observer.receive(observer.getDuration() * (static_cast<double>(k) / 3.),
                               path, field);
            } else {
              ElectricFieldVector field(
                  root, {value * 1_V / 1_m, 2 * value * 1_V / 1_m, -value * 1_V / 1_m});
              observer.receive(observer.getDuration() * (static_cast<double>(k) / 3.),
                               path, field);
            }
          }
      process.endOfShower(event);
      CHECK(process.offloadedSegmentCount() == 0);
      for (auto& observer : detector.getObservers()) {
        for (auto x : observer.getWaveformX()) CHECK(x == 0.);
        for (auto x : observer.getWaveformY()) CHECK(x == 0.);
        for (auto x : observer.getWaveformZ()) CHECK(x == 0.);
      }
    }
    process.writeSummary(directory / "summary.yaml");
    auto result = process.getSummary();
    CHECK(YAML::Dump(result) ==
          YAML::Dump(YAML::LoadFile((directory / "summary.yaml").string())));
    process.endOfLibrary();
    return result;
  }
  template <bool Zhs>
  void compare(bool empty) {
    Directory dir;
    YAML::Node got = exercise<Zhs, false>(dir.path_ / "new", empty);
    YAML::Node expected = exercise<Zhs, true>(dir.path_ / "production", empty);
    REQUIRE(YAML::Dump(got) == YAML::Dump(expected));
    auto rows = read_rows(dir.path_ / "new/observers.parquet");
    auto old = read_rows(dir.path_ / "production/observers.parquet");
    REQUIRE(rows.size() == old.size());
    REQUIRE(rows.size() == (empty ? 0 : 18));
    for (std::size_t i = 0; i < rows.size(); ++i) {
      auto const& r = rows[i];
      auto const& o = old[i];
      CHECK(r.id_ == o.id_);
      CHECK(r.time_ == o.time_);
      CHECK(r.x_ == o.x_);
      CHECK(r.y_ == o.y_);
      CHECK(r.z_ == o.z_);
      unsigned event = i / 6, bin = i % 3;
      CHECK(r.id_ == event);
      CHECK(r.time_ == Catch::Approx(bin + (Zhs ? 0.5 : 0.)));
      double field = event == 2 ? 0.
                                : (Zhs ? -(event + 1.) * (2 * bin + 1) * 1e9
                                       : (event + 1.) * (bin * bin + 1));
      CHECK(r.x_ == Catch::Approx(field));
      CHECK(r.y_ == Catch::Approx(2 * field));
      CHECK(r.z_ == Catch::Approx(-field));
    }
    REQUIRE(got.size() == 3);
    CHECK(got["shower_0"]["segment_count"].as<unsigned>() == (empty ? 0 : 2));
    CHECK(got["shower_1"]["offloaded_segment_count"].as<unsigned>() == (empty ? 0 : 2));
    CHECK(got["shower_1"]["segment_count"].as<unsigned>() == 0);
    CHECK(got["shower_2"]["offloaded_segment_count"].as<unsigned>() == 0);
    CHECK(got["shower_2"]["weighted_track_length_m"].as<double>() == 0.);
    if (!empty) {
      double length = 0.9 * 299792458. * 1e-9;
      CHECK(got["shower_0"]["weighted_track_length_m"].as<double>() ==
            Catch::Approx(5 * length));
      CHECK(got["shower_0"]["signed_charge_weighted_track_length_m"].as<double>() ==
            Catch::Approx(0.).margin(1e-14));
      auto bins = got["shower_0"]["weighted_track_length_by_kinetic_energy"]
                     ["weighted_track_length_m"];
      REQUIRE(bins.size() == 15);
      CHECK(bins[10].as<double>() == Catch::Approx(5 * length));
    }
  }
} // namespace
TEST_CASE("Radio output matches production for multiple CoREAS showers",
          "[radio][output]") {
  compare<false>(false);
}
TEST_CASE("Radio output matches production ZHS differencing and empty event",
          "[radio][output]") {
  compare<true>(false);
}
TEST_CASE("Radio output accepts an empty observer collection", "[radio][output]") {
  compare<false>(true);
}

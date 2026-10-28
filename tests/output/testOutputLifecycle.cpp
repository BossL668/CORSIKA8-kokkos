/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */
#include <corsika/output/OutputManager.hpp>
#include <corsika/output/ParquetStreamer.hpp>
#include <corsika/output/ShowerSummary.hpp>
#include <catch2/catch_all.hpp>
#include <parquet/stream_reader.h>

namespace {
  struct Directory {
    boost::filesystem::path path_ = boost::filesystem::unique_path("output-%%%%-%%%%");
    Directory() { boost::filesystem::create_directory(path_); }
    ~Directory() {
      boost::system::error_code ignored;
      boost::filesystem::remove_all(path_, ignored);
    }
  };
  struct LegacyOutput : corsika::BaseOutput {
    YAML::Node data_;
    unsigned ended_{};
    void startOfLibrary(boost::filesystem::path const&) override {}
    void endOfShower(unsigned id) override { data_["shower_" + std::to_string(id)] = id; }
    void endOfLibrary() override { ++ended_; }
    YAML::Node getSummary() const override { return data_; }
    YAML::Node getConfig() const override { return YAML::Node(); }
  };
} // namespace

TEST_CASE("OutputManager preserves legacy summaries and closes interrupted events") {
  Directory dir;
  LegacyOutput output;
  {
    corsika::OutputManager manager((dir.path_ / "run").string());
    manager.add("legacy", output);
    CHECK_FALSE(manager.showerInProgress());
    manager.startOfShower();
    CHECK(manager.showerInProgress());
    manager.endOfShower();
    CHECK_FALSE(manager.showerInProgress());
    manager.startOfShower();
    CHECK(manager.showerInProgress());
    // The production destructor finishes this pending event and the library.
  }
  CHECK(output.ended_ == 1);
  auto result = YAML::LoadFile((dir.path_ / "run/legacy/summary.yaml").string());
  REQUIRE(result.size() == 2);
  CHECK(result["shower_0"].as<unsigned>() == 0);
  CHECK(result["shower_1"].as<unsigned>() == 1);
}

TEST_CASE("Parquet flush keeps later showers and rejects incomplete rows") {
  Directory dir;
  corsika::ParquetStreamer stream;
  CHECK_THROWS(stream.flushStreamer());
  auto path = (dir.path_ / "values.parquet").string();
  stream.initStreamer(path);
  stream.addField("value", parquet::Repetition::REQUIRED, parquet::Type::DOUBLE,
                  parquet::ConvertedType::NONE);
  stream.buildStreamer();
  *stream.getWriter() << 0u;
  CHECK_THROWS_AS(stream.flushStreamer(), std::logic_error);
  *stream.getWriter() << 1.25 << parquet::EndRow;
  stream.flushStreamer();
  REQUIRE(stream.isInit());
  *stream.getWriter() << 1u << 2.5 << parquet::EndRow;
  stream.flushStreamer();
  stream.closeStreamer();
  CHECK_FALSE(stream.isInit());
  CHECK_THROWS(stream.flushStreamer());
  parquet::StreamReader reader(parquet::ParquetFileReader::OpenFile(path));
  for (unsigned id = 0; id < 2; ++id) {
    REQUIRE_FALSE(reader.eof());
    unsigned event;
    double value;
    reader >> event >> value >> parquet::EndRow;
    CHECK(event == id);
    CHECK(value == 1.25 * (id + 1));
  }
  CHECK(reader.eof());
}

TEST_CASE("ShowerSummary assignment and failed publication retain records") {
  Directory dir;
  corsika::ShowerSummary source, copy;
  source.open(dir.path_);
  source.event(0)["value"] = 10;
  copy = source;
  copy.event(0)["value"] = 20;
  copy.flush();
  source.flush();
  CHECK(source.snapshot()["shower_0"]["value"].as<int>() == 10);
  CHECK(copy.snapshot()["shower_0"]["value"].as<int>() == 20);
  CHECK_THROWS(source.writeSummary(dir.path_ / "missing/summary.yaml"));
  CHECK(source.snapshot()["shower_0"]["value"].as<int>() == 10);
  source.writeSummary(dir.path_ / "summary.yaml");
  CHECK(YAML::LoadFile((dir.path_ / "summary.yaml").string())["shower_0"]["value"]
            .as<int>() == 10);
}

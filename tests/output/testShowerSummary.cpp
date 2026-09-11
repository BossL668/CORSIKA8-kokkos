/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */
#include <catch2/catch_all.hpp>
#include <corsika/output/ShowerSummary.hpp>
#include <corsika/accelerator/em/common/GpuEmRunOutput.hpp>
#include <corsika/output/OutputManager.hpp>

namespace {
  struct SummaryDirectory {
    boost::filesystem::path path = boost::filesystem::unique_path("summary-test-%%%%-%%%%");
    SummaryDirectory() { boost::filesystem::create_directory(path); }
    ~SummaryDirectory() {
      boost::system::error_code ignored;
      boost::filesystem::remove_all(path, ignored);
    }
  };
}

TEST_CASE("ShowerSummary streams ordered records with bounded resident state") {
  SummaryDirectory directory;
  corsika::ShowerSummary summary;
  summary.open(directory.path);
  CHECK(summary.snapshot().IsNull());
  for (unsigned id = 0; id < 2000; ++id) {
    summary.event(id)["value"] = id;
    summary.event(id)["precise"] = 0.12345678901234567;
    summary.flush();
    REQUIRE_FALSE(summary.hasResidentRecord());
  }
  summary.writeSummary(directory.path / "summary.yaml");
  auto decoded = YAML::LoadFile((directory.path / "summary.yaml").string());
  REQUIRE(decoded.size() == 2000);
  unsigned ordinal = 0;
  for (auto const& entry : decoded) {
    CHECK(entry.first.as<std::string>() == "shower_" + std::to_string(ordinal));
    CHECK(entry.second["value"].as<unsigned>() == ordinal++);
    CHECK(entry.second["precise"].as<double>() == 0.12345678901234567);
  }
  CHECK_THROWS(summary.event(0));
  CHECK_FALSE(summary.hasResidentRecord());
}

TEST_CASE("ShowerSummary rewrites only the last entry and isolates copies") {
  SummaryDirectory directory;
  corsika::ShowerSummary summary;
  summary.open(directory.path);
  summary.event(0)["unchanged"] = 42;
  summary.flush();
  summary.event(1)["large"] = std::string(20000, 'x');
  summary.flush();
  summary.event(1).remove("large");
  summary.event(1)["complete"] = false;
  summary.flush();
  auto snapshot = summary.snapshot();
  REQUIRE(snapshot.size() == 2);
  CHECK(snapshot["shower_0"]["unchanged"].as<int>() == 42);
  CHECK_FALSE(snapshot["shower_1"]["large"]);
  CHECK_FALSE(snapshot["shower_1"]["complete"].as<bool>());
  corsika::ShowerSummary copy(summary);
  copy.event(1)["complete"] = true;
  copy.flush();
  CHECK_FALSE(summary.snapshot()["shower_1"]["complete"].as<bool>());
  CHECK(copy.snapshot()["shower_1"]["complete"].as<bool>());
  summary.open(directory.path);
  CHECK(summary.snapshot().IsNull());
}

TEST_CASE("ShowerSummary fails closed on journal failure") {
  SummaryDirectory directory;
  corsika::ShowerSummary summary;
  CHECK_THROWS(summary.open(directory.path / "missing"));
  summary.open(directory.path);
  summary.event(0)["complete"] = true;
  // Simulate storage disappearing: no silent loss of the in-memory record.
  for (auto const& entry : boost::filesystem::directory_iterator(directory.path))
    boost::filesystem::remove(entry.path());
  CHECK_THROWS(summary.flush());
  CHECK(summary.hasResidentRecord());
}

TEST_CASE("GpuEmRunOutput streamed status supports failure after completion") {
  SummaryDirectory directory;
  corsika::gpu::em::GpuEmRunOutput output{YAML::Node()};
  output.startOfLibrary(directory.path);
  output.startOfShower(0);
  output.endOfShower(0);
  YAML::Node metadata;
  metadata["count"] = 123;
  output.recordComplete(0, metadata);
  output.recordIncomplete(0, "injected failure");
  output.writeSummary(directory.path / "summary.yaml");
  auto result = YAML::LoadFile((directory.path / "summary.yaml").string());
  REQUIRE(result.size() == 1);
  CHECK_FALSE(result["shower_0"]["complete"].as<bool>());
  CHECK(result["shower_0"]["statistics"]["count"].as<int>() == 123);
  output.startOfShower(1);
  output.endOfShower(1);
  // OutputManager serializes before endOfLibrary: missing completion must
  // already be marked incomplete in the file, not fixed only in memory.
  output.writeSummary(directory.path / "summary.yaml");
  output.endOfLibrary();
  result = YAML::LoadFile((directory.path / "summary.yaml").string());
  CHECK(result["shower_1"]["status"].as<std::string>() == "incomplete");
}

TEST_CASE("OutputManager dispatches streaming summary without materializing it") {
  SummaryDirectory directory;
  struct StreamingOutput : corsika::BaseOutput {
    mutable unsigned snapshots = 0;
    corsika::ShowerSummary data;
    void startOfLibrary(boost::filesystem::path const& path) override { data.open(path); }
    void endOfShower(unsigned id) override { data.event(id)["value"] = id; data.flush(); }
    void endOfLibrary() override {}
    YAML::Node getConfig() const override { return YAML::Node(); }
    YAML::Node getSummary() const override { ++snapshots; return data.snapshot(); }
    void writeSummary(boost::filesystem::path const& path) const override { data.writeSummary(path); }
  } participant;
  corsika::OutputManager manager((directory.path / "run").string());
  manager.add("stream", participant);
  manager.startOfLibrary();
  manager.startOfShower();
  manager.endOfShower();
  manager.endOfLibrary();
  CHECK(participant.snapshots == 0);
  CHECK(YAML::LoadFile((directory.path / "run/stream/summary.yaml").string()).size() == 1);
}

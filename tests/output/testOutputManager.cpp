/*
 * (c) Copyright 2020 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include <catch2/catch_all.hpp>

#include <boost/filesystem.hpp>

#include <corsika/framework/core/Logging.hpp>

#include <corsika/output/OutputManager.hpp>
#include <corsika/output/NoOutput.hpp>

using namespace corsika;
using Catch::Approx;

struct DummyNoOutput : public NoOutput {
  void check() {
    NoOutput::startOfLibrary("./");
    NoOutput::startOfShower(0);
    NoOutput::endOfShower(0);
    NoOutput::endOfLibrary();
  }
  void checkWrite() { NoOutput::write(Code::Unknown, 1_eV, 1_m, 1_m, 1_ns); }
};

struct DummyOutput : public BaseOutput {

  bool startLibrary_ = false;
  bool startShower_ = false;
  bool endLibrary_ = false;
  bool endShower_ = false;

  void startOfLibrary(boost::filesystem::path const&) override { startLibrary_ = true; }

  YAML::Node getConfig() const final override { return YAML::Node(); }

  void startOfShower(unsigned int const shower = 0) override {
    BaseOutput::startOfShower(shower);
    setInit(true);
    startShower_ = true;
  }

  void endOfShower(unsigned int const) override { endShower_ = true; }

  void endOfLibrary() override { endLibrary_ = true; }

  YAML::Node getSummary() const final override {
    YAML::Node summary;
    summary["test"] = "test";
    return summary;
  }
};

TEST_CASE("OutputManager") {

  logging::set_level(logging::level::info);

  SECTION("standard") {

    // preparation
    if (boost::filesystem::exists("./out_test")) {
      boost::filesystem::remove_all("./out_test");
    }

    // output manager performs nothing, no action, just interface
    OutputManager output("./out_test/check", 0, "", false);

    CHECK(boost::filesystem::is_directory("./out_test/check"));

    DummyOutput test;
    output.add("test", test);

    CHECK_THROWS(output.add(
        "test",
        test)); // should emit warning which cannot be catched, but no action or failure

    output.startOfLibrary();
    CHECK(test.startLibrary_);
    test.startLibrary_ = false;

    CHECK_FALSE(test.isInit());
    output.startOfShower();
    CHECK(test.isInit());
    CHECK(test.startShower_);
    test.startShower_ = false;

    output.endOfShower();
    CHECK(test.endShower_);
    test.endShower_ = false;

    output.endOfLibrary();
    CHECK(test.endLibrary_);
    test.endLibrary_ = false;
  }

  SECTION("compression") {
    std::string const outputDir = "./out_compressed";
    std::string const outputArchive = outputDir + ".tar";

    // preparation
    if (boost::filesystem::exists(outputDir)) {
      boost::filesystem::remove_all(outputDir);
    }
    if (boost::filesystem::exists(outputArchive)) {
      boost::filesystem::remove_all(outputArchive);
    }

    // We make a pointer here because the compression happens at deconstruction
    OutputManager* output = new OutputManager(outputDir, 0, "", true);
    CHECK(boost::filesystem::is_directory(outputDir));
    CHECK(!boost::filesystem::exists(outputArchive)); // compressed file does NOT exist

    // Make an output and open/close shower/lib
    DummyOutput test;
    output->add("test", test);
    output->startOfLibrary();
    output->startOfShower();
    output->endOfShower();
    output->endOfLibrary();

    // Ensure compression happens at deconstruction
    CHECK(!boost::filesystem::exists(outputArchive)); // compressed file does NOT exist
    delete output;
    CHECK(boost::filesystem::exists(outputArchive)); // compressed file DOES exist
  }

  SECTION("auto-write") {

    // preparation
    if (boost::filesystem::exists("./out_test")) {
      boost::filesystem::remove_all("./out_test");
    }

    // output manager performs nothing, no action, just interface
    OutputManager* output = new OutputManager("./out_test/check", 0, "", false);

    CHECK(boost::filesystem::is_directory("./out_test/check"));

    DummyOutput test;
    output->add("test", test);
    output->startOfLibrary();

    // cannot add more after library started
    DummyOutput test2;
    CHECK_THROWS(output->add("test2", test2));

    output->startOfShower();

    // check support for closing automatically
    delete output;
    output = 0;

    CHECK(boost::filesystem::exists("./out_test/check/test/summary.yaml"));
  }

  SECTION("failures") {

    logging::set_level(logging::level::info);

    // preparation
    if (boost::filesystem::exists("./out_test")) {
      boost::filesystem::remove_all("./out_test");
    }

    // output manager performs nothing, no action, just interface
    OutputManager output("./out_test/check", 0, "", false);
    CHECK_THROWS(new OutputManager("./out_test/check", 0, "", false));

    CHECK_THROWS(output.endOfLibrary());

    output.startOfLibrary();

    CHECK_THROWS(output.startOfLibrary());

    output.startOfShower();

    CHECK_THROWS(output.startOfLibrary());

    output.endOfShower();

    CHECK_THROWS(output.startOfLibrary());

    output.endOfLibrary();
  }

  SECTION("NoOutput") {
    // this is one of the classes where testing is a bit useless, but we can at least make
    // sure the interface exists.
    DummyNoOutput nothing;

    nothing.check();
    nothing.checkWrite();
    nothing.getConfig();
    nothing.getSummary();
  }
}

TEST_CASE("OutputManager reopens independent worker libraries") {
  auto const root = boost::filesystem::temp_directory_path() /
                    boost::filesystem::unique_path("c8-worker-output-%%%%-%%%%");
  {
    DummyOutput module;
    OutputManager output((root / "first").string(), 17, "worker test", false);
    output.add("test", module);
    output.startOfLibrary();
    CHECK_THROWS_AS(output.reopenLibrary((root / "second").string(), 23),
                    std::logic_error);
    output.startOfShower();
    output.endOfShower();
    output.endOfLibrary();
    output.flushLibrarySummary();
    CHECK(YAML::LoadFile((root / "first/summary.yaml").string())["showers"].as<int>() ==
          1);
    boost::filesystem::last_write_time(root / "first/summary.yaml", 1);
    output.reopenLibrary((root / "second").string(), 23);
    CHECK(boost::filesystem::last_write_time(root / "first/summary.yaml") == 1);
    CHECK(output.getEventId() == 0);
    CHECK(output.getSummary()["seed"].as<long>() == 23);
    CHECK(output.getConfig()["path"].as<std::string>() == (root / "second").string());
    output.startOfShower();
    output.endOfShower();
    output.endOfLibrary();
    output.flushLibrarySummary();
    CHECK(YAML::LoadFile((root / "second/summary.yaml").string())["showers"].as<int>() ==
          1);
    CHECK(YAML::LoadFile((root / "first/summary.yaml").string())["seed"].as<long>() ==
          17);
    CHECK_THROWS(output.reopenLibrary((root / "first").string(), 29));
    boost::filesystem::last_write_time(root / "second/summary.yaml", 2);
  }
  CHECK(boost::filesystem::last_write_time(root / "second/summary.yaml") == 2);
  boost::filesystem::remove_all(root);
}

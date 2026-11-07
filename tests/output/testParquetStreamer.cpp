/*
 * (c) Copyright 2020 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include <catch2/catch_all.hpp>
#include <parquet/file_reader.h>

#include <boost/filesystem.hpp>

#include <corsika/output/ParquetStreamer.hpp>
#include <corsika/framework/core/Logging.hpp>

using namespace corsika;
using Catch::Approx;

TEST_CASE("ParquetStreamer") {

  logging::set_level(logging::level::info);

  SECTION("standard") {

    // preparation
    if (boost::filesystem::exists("./parquet_test.parquet")) {
      boost::filesystem::remove_all("./parquet_test.parquet");
    }

    ParquetStreamer test;
    CHECK_FALSE(test.isInit());
    CHECK_THROWS(test.getWriter());
    test.initStreamer("./parquet_test.parquet");

    test.addField("testint", parquet::Repetition::REQUIRED, parquet::Type::INT32,
                  parquet::ConvertedType::INT_32);
    test.addField("testfloat", parquet::Repetition::REQUIRED, parquet::Type::FLOAT,
                  parquet::ConvertedType::NONE);

    test.enableCompression(1);

    test.buildStreamer();
    CHECK(test.isInit());

    unsigned int testId = 2;
    int testint = 1;
    double testfloat = 2.0;

    std::shared_ptr<parquet::StreamWriter> writer = test.getWriter();
    (*writer) << testId << testint << static_cast<float>(testfloat) << parquet::EndRow;

    test.closeStreamer();
    CHECK_THROWS(test.getWriter());
    CHECK_FALSE(test.isInit());
    CHECK(boost::filesystem::exists("./parquet_test.parquet"));
  }
}

TEST_CASE("ParquetStreamer does not duplicate columns between worker requests") {
  auto const root = boost::filesystem::temp_directory_path() /
                    boost::filesystem::unique_path("c8-worker-parquet-%%%%-%%%%");
  boost::filesystem::create_directory(root);
  ParquetStreamer streamer;
  for (unsigned request = 0; request < 3; ++request) {
    auto const file = root / (std::to_string(request) + ".parquet");
    streamer.initStreamer(file.string());
    streamer.addField("value", parquet::Repetition::REQUIRED, parquet::Type::INT32,
                      parquet::ConvertedType::INT_32);
    streamer.buildStreamer();
    (*streamer.getWriter()) << 0U << static_cast<int>(request) << parquet::EndRow;
    streamer.closeStreamer();
    auto reader = parquet::ParquetFileReader::OpenFile(file.string());
    CHECK(reader->metadata()->num_columns() == 2);
    CHECK(reader->metadata()->num_rows() == 1);
  }
  boost::filesystem::remove_all(root);
}

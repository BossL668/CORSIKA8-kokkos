/*
 * (c) Copyright 2020 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include <catch2/catch_all.hpp>

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

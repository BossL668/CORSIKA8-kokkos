/*
 * (c) Copyright 2020 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include <catch2/catch_all.hpp>

#include <boost/filesystem.hpp>

#include <corsika/output/ParquetStreamer.hpp>
#include <parquet/file_reader.h>
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

  SECTION("event row groups preserve one file and reject partial rows") {
    ParquetStreamer stream;
    CHECK_THROWS(stream.flushStreamer());
    stream.initStreamer("./parquet_row_groups_test.parquet");
    stream.addField("value", parquet::Repetition::REQUIRED, parquet::Type::INT32,
                    parquet::ConvertedType::INT_32);
    stream.buildStreamer();
    *stream.getWriter() << unsigned{0};
    CHECK_THROWS(stream.flushStreamer());
    *stream.getWriter() << int{9} << parquet::EndRow;
    stream.flushStreamer();
    for (unsigned id = 0; id < 3; ++id) {
      *stream.getWriter() << id << static_cast<int>(10 + id) << parquet::EndRow;
      stream.flushStreamer();
      stream.flushStreamer(); // empty flush must not create extra groups
    }
    stream.closeStreamer();
    auto reader = parquet::ParquetFileReader::OpenFile("./parquet_row_groups_test.parquet", false);
    // Arrow opens the next group at EndRowGroup(); closing the library can
    // retain that one empty trailing group. Repeated empty flushes add none.
    CHECK(reader->metadata()->num_row_groups() == 5);
    CHECK(reader->metadata()->num_rows() == 4);
    CHECK(reader->metadata()->RowGroup(4)->num_rows() == 0);
    CHECK_THROWS(stream.flushStreamer());
  }
}

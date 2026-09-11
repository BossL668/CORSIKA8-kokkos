
/*
 * (c) Copyright 2021 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <string>

// clang-format-off, NOTE: the order of these includes is *important*
// you will get unhelpful compiler errors about unknown
// operator definitions if these are reordered
#include <parquet/stream_writer.h>
#include <parquet/arrow/schema.h>
#include <arrow/io/file.h>
// clang-format-on

namespace corsika {

  /**
   * This class automates the construction of simple tabular
   * Parquet files using the parquet::StreamWriter.
   */
  class ParquetStreamer {

  public:
    /**
     * ParquetStreamer's take no constructor arguments.
     */
    ParquetStreamer();

    /**
     * Initialize the streamer to write to a given file.
     */
    void initStreamer(std::string const& filepath);

    /**
     * Add a field to this streamer.
     */
    template <typename... TArgs>
    void addField(TArgs&&... args);

    /**
     * Enable compression for this streamer.
     */
    void enableCompression(int const level = 9);

    /**
     * Finalize the streamer construction.
     */
    void buildStreamer();

    /**
     * Finish writing this stream.
     */
    void closeStreamer();

    /** Commit completed rows without closing the multi-shower file. */
    void flushStreamer();

    /**
     * Return a reference to the underlying writer.
     */
    std::shared_ptr<parquet::StreamWriter> getWriter();

    /**
     * @return status of streamer
     */
    bool isInit() const { return isInit_; }

  private:
    bool isInit_ = false;                        ///< flag to handle state of writer
    parquet::WriterProperties::Builder builder_; ///< The writer properties builder.
    parquet::schema::NodeVector fields_;         ///< The fields in this file.
    std::shared_ptr<parquet::schema::GroupNode> schema_;   ///< The schema for this file.
    std::shared_ptr<arrow::io::FileOutputStream> outfile_; ///< The output file.
    std::shared_ptr<parquet::StreamWriter> writer_; ///< The stream writer to 'outfile'

  }; // class ParquetStreamer
} // namespace corsika

#include <corsika/detail/output/ParquetStreamer.inl>

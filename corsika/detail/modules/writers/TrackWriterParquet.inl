/*
 * (c) Copyright 2021 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

namespace corsika {

  inline TrackWriterParquet::TrackWriterParquet()
      : output_()
      , showerId_(0) {}

  inline void TrackWriterParquet::startOfLibrary(
      boost::filesystem::path const& directory) {

    // setup the streamer
    output_.initStreamer((directory / "tracks.parquet").string());

    // enable compression with the default level
    output_.enableCompression();

    // build the schema
    output_.addField("pdg", parquet::Repetition::REQUIRED, parquet::Type::INT32,
                     parquet::ConvertedType::INT_32);
    output_.addField("start_energy", parquet::Repetition::REQUIRED, parquet::Type::FLOAT,
                     parquet::ConvertedType::NONE);
    output_.addField("weight", parquet::Repetition::REQUIRED, parquet::Type::FLOAT,
                     parquet::ConvertedType::NONE);
    output_.addField("start_x", parquet::Repetition::REQUIRED, parquet::Type::FLOAT,
                     parquet::ConvertedType::NONE);
    output_.addField("start_y", parquet::Repetition::REQUIRED, parquet::Type::FLOAT,
                     parquet::ConvertedType::NONE);
    output_.addField("start_z", parquet::Repetition::REQUIRED, parquet::Type::FLOAT,
                     parquet::ConvertedType::NONE);
    output_.addField("start_t", parquet::Repetition::REQUIRED, parquet::Type::FLOAT,
                     parquet::ConvertedType::NONE);
    output_.addField("end_x", parquet::Repetition::REQUIRED, parquet::Type::FLOAT,
                     parquet::ConvertedType::NONE);
    output_.addField("end_y", parquet::Repetition::REQUIRED, parquet::Type::FLOAT,
                     parquet::ConvertedType::NONE);
    output_.addField("end_z", parquet::Repetition::REQUIRED, parquet::Type::FLOAT,
                     parquet::ConvertedType::NONE);
    output_.addField("end_energy", parquet::Repetition::REQUIRED, parquet::Type::FLOAT,
                     parquet::ConvertedType::NONE);
    output_.addField("end_t", parquet::Repetition::REQUIRED, parquet::Type::FLOAT,
                     parquet::ConvertedType::NONE);

    // and build the streamer
    output_.buildStreamer();
    showerId_ = 0;
  }

  inline void TrackWriterParquet::startOfShower(unsigned int const showerId) {
    showerId_ = showerId;
  }

  inline void TrackWriterParquet::endOfShower(unsigned int const) {}

  inline void TrackWriterParquet::endOfLibrary() { output_.closeStreamer(); }

  inline void TrackWriterParquet::write(Code const pid, HEPEnergyType const KinenergyPre,
                                        double const weight,
                                        QuantityVector<length_d> const& start,
                                        TimeType const t_start,
                                        QuantityVector<length_d> const& end,
                                        HEPEnergyType const KinenergyPost,
                                        TimeType const t_end) {

    // write the next row - we must write `shower_` first.
    // clang-format off
   *(output_.getWriter())
       << showerId_
       << static_cast<int>(get_PDG(pid))
       << static_cast<float>(KinenergyPre / 1_GeV)
       << static_cast<float>(weight)
       << static_cast<float>(start[0] / 1_m)
       << static_cast<float>(start[1] / 1_m)
       << static_cast<float>(start[2] / 1_m)
       << static_cast<float>(t_start / 1_ns)
       << static_cast<float>(end[0] / 1_m)
       << static_cast<float>(end[1] / 1_m)
       << static_cast<float>(end[2] / 1_m)
       << static_cast<float>(KinenergyPost / 1_GeV)
       << static_cast<float>(t_end / 1_ns)
       << parquet::EndRow;
    // clang-format on
  }

} // namespace corsika
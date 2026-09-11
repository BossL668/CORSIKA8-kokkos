/*
 * (c) Copyright 2021 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/framework/core/ParticleProperties.hpp>
#include <corsika/framework/core/PhysicalUnits.hpp>
#include <corsika/framework/utility/FindXmax.hpp>

#include <corsika/media/ShowerAxis.hpp>

#include <exception>

namespace corsika {

  template <size_t NColumns>
  inline EnergyLossWriterParquet<NColumns>::EnergyLossWriterParquet(
      std::array<const char*, NColumns> const& colNames)
      : columns_(colNames) {}

  template <size_t NColumns>
  inline void EnergyLossWriterParquet<NColumns>::startOfLibrary(
      boost::filesystem::path const& directory) {

    // setup the streamer
    output_.initStreamer((directory / "dEdX.parquet").string());

    // enable compression with the default level
    output_.enableCompression();

    // build the schema
    output_.addField("X", parquet::Repetition::REQUIRED, parquet::Type::FLOAT,
                     parquet::ConvertedType::NONE);
    for (auto const& col : columns_) {
      output_.addField(col, parquet::Repetition::REQUIRED, parquet::Type::FLOAT,
                       parquet::ConvertedType::NONE);
    }

    // and build the streamer
    output_.buildStreamer();
  }

  template <size_t NColumns>
  inline void EnergyLossWriterParquet<NColumns>::write(
      unsigned int const showerId, GrammageType const grammage,
      std::array<HEPEnergyType, NColumns> const& data) {

    // and write the data into the column
    *(output_.getWriter()) << showerId
                           << static_cast<float>(grammage / 1_g * square(1_cm));
    for (HEPEnergyType const& dedx : data) {
      *(output_.getWriter()) << static_cast<float>(dedx / 1_GeV);
    }
    *(output_.getWriter()) << parquet::EndRow;
  }

  template <size_t NColumns>
  inline void EnergyLossWriterParquet<NColumns>::startOfShower(unsigned int const) {}

  template <size_t NColumns>
  inline void EnergyLossWriterParquet<NColumns>::endOfShower(unsigned int const) { output_.flushStreamer(); }

  template <size_t NColumns>
  inline void EnergyLossWriterParquet<NColumns>::endOfLibrary() {
    output_.closeStreamer();
  }

} // namespace corsika

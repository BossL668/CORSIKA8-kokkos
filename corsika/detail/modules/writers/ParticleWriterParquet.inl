/*
 * (c) Copyright 2021 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/framework/core/Logging.hpp>

namespace corsika {

  inline ParticleWriterParquet::ParticleWriterParquet(bool const printZ)
      : output_()
      , showerId_(0)
      , printZ_(printZ) {}

  inline void ParticleWriterParquet::startOfLibrary(
      boost::filesystem::path const& directory) {

    summary_.open(directory);

    // setup the streamer
    output_.initStreamer((directory / "particles.parquet").string());

    // enable compression with the default level
    output_.enableCompression();

    // build the schema
    output_.addField("pdg", parquet::Repetition::REQUIRED, parquet::Type::INT32,
                     parquet::ConvertedType::INT_32);
    output_.addField("kinetic_energy", parquet::Repetition::REQUIRED,
                     parquet::Type::FLOAT, parquet::ConvertedType::NONE);
    output_.addField("x", parquet::Repetition::REQUIRED, parquet::Type::FLOAT,
                     parquet::ConvertedType::NONE);
    output_.addField("y", parquet::Repetition::REQUIRED, parquet::Type::FLOAT,
                     parquet::ConvertedType::NONE);
    if (printZ_) {
      output_.addField("z", parquet::Repetition::REQUIRED, parquet::Type::FLOAT,
                       parquet::ConvertedType::NONE);
    }
    output_.addField("nx", parquet::Repetition::REQUIRED, parquet::Type::FLOAT,
                     parquet::ConvertedType::NONE);
    output_.addField("ny", parquet::Repetition::REQUIRED, parquet::Type::FLOAT,
                     parquet::ConvertedType::NONE);
    output_.addField("nz", parquet::Repetition::REQUIRED, parquet::Type::FLOAT,
                     parquet::ConvertedType::NONE);
    output_.addField("time", parquet::Repetition::REQUIRED, parquet::Type::DOUBLE,
                     parquet::ConvertedType::NONE);
    output_.addField("weight", parquet::Repetition::REQUIRED, parquet::Type::FLOAT,
                     parquet::ConvertedType::NONE);

    // and build the streamer
    output_.buildStreamer();

    showerId_ = 0;
  }

  inline void ParticleWriterParquet::startOfShower(unsigned int const showerId) {
    showerId_ = showerId;

    countHadrons_ = 0;
    countOthers_ = 0;
    countEM_ = 0;
    countMuons_ = 0;

    kineticEnergyHadrons_ = 0_eV;
    kineticEnergyMuons_ = 0_eV;
    kineticEnergyEM_ = 0_eV;
    kineticEnergyOthers_ = 0_eV;

    totalEnergyHadrons_ = 0_eV;
    totalEnergyMuons_ = 0_eV;
    totalEnergyEM_ = 0_eV;
    totalEnergyOthers_ = 0_eV;
  }

  inline void ParticleWriterParquet::endOfShower(unsigned int const) {
    output_.flushStreamer();
    summary_.event(showerId_)["hadron"]["count"] = countHadrons_;
    summary_.event(showerId_)["hadron"]["kinetic_energy"] =
        kineticEnergyHadrons_ / 1_GeV;
    summary_.event(showerId_)["hadron"]["total_energy"] =
        totalEnergyHadrons_ / 1_GeV;

    summary_.event(showerId_)["muon"]["count"] = countMuons_;
    summary_.event(showerId_)["muon"]["kinetic_energy"] =
        kineticEnergyMuons_ / 1_GeV;
    summary_.event(showerId_)["muon"]["total_energy"] =
        totalEnergyMuons_ / 1_GeV;

    summary_.event(showerId_)["em"]["count"] = countEM_;
    summary_.event(showerId_)["em"]["kinetic_energy"] =
        kineticEnergyEM_ / 1_GeV;
    summary_.event(showerId_)["em"]["total_energy"] =
        totalEnergyEM_ / 1_GeV;

    summary_.event(showerId_)["other"]["count"] = countOthers_;
    summary_.event(showerId_)["other"]["kinetic_energy"] =
        kineticEnergyOthers_ / 1_GeV;
    summary_.event(showerId_)["other"]["total_energy"] =
        totalEnergyOthers_ / 1_GeV;
    summary_.flush();
  }

  inline void ParticleWriterParquet::endOfLibrary() { output_.closeStreamer(); }

  inline HEPEnergyType ParticleWriterParquet::getEnergyGround() const {
    return totalEnergyHadrons_ + totalEnergyMuons_ + totalEnergyEM_ + totalEnergyOthers_;
  }

  inline void ParticleWriterParquet::write(Code const pid,
                                           HEPEnergyType const kineticEnergy,
                                           LengthType const x, LengthType const y,
                                           LengthType const z, double const nx,
                                           double const ny, double const nz,
                                           TimeType const t, double const weight) {

    // write the next row - we must write `shower_` first.
    *(output_.getWriter()) << showerId_ << static_cast<int>(get_PDG(pid))
                           << static_cast<float>(kineticEnergy / 1_GeV)
                           << static_cast<float>(x / 1_m) << static_cast<float>(y / 1_m);
    if (printZ_) { *(output_.getWriter()) << static_cast<float>(z / 1_m); }

    *(output_.getWriter()) << static_cast<float>(nx) << static_cast<float>(ny)
                           << static_cast<float>(nz) << static_cast<double>(t / 1_s)
                           << static_cast<float>(weight) << parquet::EndRow;

    if (is_hadron(pid)) {
      ++countHadrons_;
      kineticEnergyHadrons_ += kineticEnergy;
      totalEnergyHadrons_ += kineticEnergy + get_mass(pid);
    } else if (is_muon(pid)) {
      ++countMuons_;
      kineticEnergyMuons_ += kineticEnergy;
      totalEnergyMuons_ += kineticEnergy + get_mass(pid);
    } else if (is_em(pid)) {
      ++countEM_;
      kineticEnergyEM_ += kineticEnergy;
      totalEnergyEM_ += kineticEnergy + get_mass(pid);
    } else {
      ++countOthers_;
      kineticEnergyOthers_ += kineticEnergy;
      totalEnergyOthers_ += kineticEnergy + get_mass(pid);
    }
  }

  /**
   * Return collected library-level summary for output.
   */
  inline YAML::Node ParticleWriterParquet::getSummary() const { return summary_.snapshot(); }

} // namespace corsika

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
#include <algorithm>
#include <iostream>

namespace corsika {

  template <typename TOutput>
  inline LongitudinalWriter<TOutput>::LongitudinalWriter(ShowerAxis const& axis,
                                                         GrammageType dX)
      : LongitudinalWriter<TOutput>{
            axis, static_cast<unsigned int>(axis.getMaximumX() / dX) + 1, dX} {}

  template <typename TOutput>
  inline LongitudinalWriter<TOutput>::LongitudinalWriter(ShowerAxis const& axis,
                                                         size_t nbins, GrammageType dX)
      : TOutput(number_profile::ProfileIndexNames)
      , showerAxis_(axis)
      , dX_(dX)
      , nBins_(nbins)
      , profile_{nbins} {}

  template <typename TOutput>
  inline void LongitudinalWriter<TOutput>::startOfLibrary(
      boost::filesystem::path const& directory) {
    TOutput::startOfLibrary(directory);
  }

  template <typename TOutput>
  inline void LongitudinalWriter<TOutput>::startOfShower(unsigned int const showerId) {
    profile_.clear();
    for (size_t i = 0; i < nBins_; ++i) { profile_.emplace_back(); }
    TOutput::startOfShower(showerId);
  }

  template <typename TOutput>
  inline void LongitudinalWriter<TOutput>::endOfLibrary() {
    TOutput::endOfLibrary();
  }

  template <typename TOutput>
  inline void LongitudinalWriter<TOutput>::endOfShower(unsigned int const showerId) {

    int iRow{0};
    for (number_profile::ProfileData const& row : profile_) {
      // here: write to underlying writer (e.g. parquet)
      TOutput::write(showerId, iRow * dX_, row);
      iRow++;
    }
    TOutput::endOfShower(showerId);
  }

  template <typename TOutput>
  inline void LongitudinalWriter<TOutput>::write(Point const& p0, Point const& p1,
                                                 Code const pid, double const weight) {
    GrammageType const grammageStart = showerAxis_.getProjectedX(p0);
    GrammageType const grammageEnd = showerAxis_.getProjectedX(p1);
    writeProjected(grammageStart, grammageEnd, pid, weight);
  }

  template <typename TOutput>
  inline void LongitudinalWriter<TOutput>::writeProjected(
      GrammageType const grammageStart, GrammageType const grammageEnd,
      Code const pid, double const weight) {

    // Avoid over counting in first bin when backscattered particle goes beyond the
    // injection point.
    if (grammageStart == grammageEnd) { return; }

    // Note: particle may go also "upward", thus, grammageEnd<grammageStart
    size_t const binStart = std::ceil(grammageStart / dX_);
    size_t const binEnd = std::floor(grammageEnd / dX_);

    CORSIKA_LOGGER_TRACE(TOutput::getLogger(),
                         "grammageStart={} End={} binStart={}, end={}",
                         grammageStart / 1_g * square(1_cm),
                         grammageEnd / 1_g * square(1_cm), binStart, binEnd);

    for (size_t bin = binStart; bin <= std::min(binEnd, profile_.size() - 1); ++bin) {
      addBin(bin, pid, weight);
    }
  }

  template <typename TOutput>
  inline void LongitudinalWriter<TOutput>::addBin(
      size_t bin, Code pid, double weight) {
    auto& row = profile_.at(bin);
    if (pid == Code::Photon) {
      row[static_cast<int>(number_profile::ProfileIndex::Photon)] += weight;
    } else if (pid == Code::Positron) {
      row[static_cast<int>(number_profile::ProfileIndex::Positron)] += weight;
    } else if (pid == Code::Electron) {
      row[static_cast<int>(number_profile::ProfileIndex::Electron)] += weight;
    } else if (pid == Code::MuPlus) {
      row[static_cast<int>(number_profile::ProfileIndex::MuPlus)] += weight;
    } else if (pid == Code::MuMinus) {
      row[static_cast<int>(number_profile::ProfileIndex::MuMinus)] += weight;
    } else if (is_hadron(pid)) {
      row[static_cast<int>(number_profile::ProfileIndex::Hadron)] += weight;
    }
    if (is_charged(pid)) {
      row[static_cast<int>(number_profile::ProfileIndex::Charged)] += weight;
    }
  }

  template <typename TOutput>
  inline void LongitudinalWriter<TOutput>::write(GrammageType const Xstart,
                                                 GrammageType const Xend, Code const pid,
                                                 double const weight) {
    double const bstart = Xstart / dX_;
    double const bend = Xend / dX_;

    if (abs(bstart - floor(bstart + 0.5)) > 1e-2 ||
        abs(bend - floor(bend + 0.5)) > 1e-2 || abs(bend - bstart - 1) > 1e-2) {
      CORSIKA_LOGGER_ERROR(TOutput::getLogger(),
                           "CONEX and Corsika8 dX grammage binning are not the same! "
                           "Xstart={} Xend={} dX={}",
                           Xstart / 1_g * square(1_cm), Xend / 1_g * square(1_cm),
                           dX_ / 1_g * square(1_cm));
      throw std::runtime_error(
          "CONEX and Corsika8 dX grammage binning are not the same!");
    }

    size_t const bin = size_t((bend + bstart) / 2);
    if (bin >= profile_.size()) {
      CORSIKA_LOGGER_WARN(TOutput::getLogger(),
                          "Grammage bin {} outside of profile {}. skipping.", bin,
                          profile_.size());
      return;
    }

    addBin(bin, pid, weight);
  }

  template <typename TOutput>
  inline YAML::Node LongitudinalWriter<TOutput>::getConfig() const {

    YAML::Node node;

    node["type"] = "LongitudinalProfile";
    node["units"]["grammage"] = "g/cm^2";
    node["bin-size"] = dX_ / (1_g / square(1_cm));
    node["nbins"] = nBins_;

    return node;
  }

  template <typename TOutput>
  inline YAML::Node LongitudinalWriter<TOutput>::getSummary() const {
    YAML::Node summary;
    return summary;
  }

} // namespace corsika

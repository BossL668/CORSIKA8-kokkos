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

  template <typename TOutput>
  inline EnergyLossWriter<TOutput>::EnergyLossWriter(ShowerAxis const& axis,
                                                     GrammageType dX,
                                                     GrammageType dX_threshold)
      : EnergyLossWriter<TOutput>{axis,
                                  static_cast<unsigned int>(axis.getMaximumX() / dX) + 1,
                                  dX, dX_threshold} {}

  template <typename TOutput>
  inline EnergyLossWriter<TOutput>::EnergyLossWriter(ShowerAxis const& axis,
                                                     unsigned int const nBins,
                                                     GrammageType dX,
                                                     GrammageType dX_threshold)
      : TOutput(dEdX_output::ProfileIndexNames)
      , showerAxis_(axis)
      , dX_(dX)
      , nBins_(nBins)
      , dX_threshold_(dX_threshold) {}

  template <typename TOutput>
  inline void EnergyLossWriter<TOutput>::startOfLibrary(
      boost::filesystem::path const& directory) {
    TOutput::startOfLibrary(directory);
  }

  template <typename TOutput>
  inline void EnergyLossWriter<TOutput>::startOfShower(unsigned int const showerId) {
    TOutput::startOfShower(showerId);
    // reset profile
    profile_.clear();
    profile_.resize(nBins_);
    electromagneticEnergyLost_ = HEPEnergyType::zero();
  }

  template <typename TOutput>
  inline void EnergyLossWriter<TOutput>::endOfLibrary() {
    TOutput::endOfLibrary();
  }

  template <typename TOutput>
  inline void EnergyLossWriter<TOutput>::endOfShower(unsigned int const showerId) {

    // determined Xmax and dEdXmax from quadratic interpolation
    double maximum = 0;
    size_t iMaximum = 0;

    size_t constexpr window_size = 20;
    int constexpr TotalIdx = static_cast<int>(dEdX_output::ProfileIndex::Total);

    auto x_at = [&](size_t i) { return (i + 0.5) * (dX_ / 1_g * square(1_cm)); };
    auto y_at = [&](size_t i) { return profile_[i].at(TotalIdx) / 1_GeV; };

    for (size_t i = 0; i <= profile_.size() - window_size; ++i) {
      double value = 0;
      for (size_t j = 0; j < window_size; ++j) { value += y_at(i + j); }
      if (value > maximum) {
        maximum = value;
        iMaximum = i;
      }
    }

    std::vector<double> xs, ys;
    for (size_t j = 0; j < window_size; ++j) {
      xs.push_back(x_at(iMaximum + j));
      ys.push_back(y_at(iMaximum + j));
    }

    auto [Xmax, dEdXmax] = FindXmax::EstimateParabolaParameters(xs, ys);
    summary_["shower_" + std::to_string(showerId)]["sum_dEdX"] = getEnergyLost() / 1_GeV;
    summary_["shower_" + std::to_string(showerId)]["sum_dEdX_em"] =
        electromagneticEnergyLost_ / 1_GeV;
    summary_["shower_" + std::to_string(showerId)]["Xmax"] = Xmax;
    summary_["shower_" + std::to_string(showerId)]["dEdXmax"] = dEdXmax;

    int iRow{0};
    for (dEdX_output::Profile const& row : profile_) {
      // here: write to underlying writer (e.g. parquet)
      TOutput::write(showerId, iRow * dX_, row);
      iRow++;
    }

    TOutput::endOfShower(showerId);
  }

  template <typename TOutput>
  inline void EnergyLossWriter<TOutput>::write(Point const& p0, Point const& p1,
                                               Code const PID, HEPEnergyType const dE) {

    GrammageType grammageStart = showerAxis_.getProjectedX(p0);
    GrammageType grammageEnd = showerAxis_.getProjectedX(p1);
    writeProjected(grammageStart, grammageEnd, PID, dE);
  }

  template <typename TOutput>
  inline void EnergyLossWriter<TOutput>::writeProjected(
      GrammageType grammageStart, GrammageType grammageEnd,
      Code const PID, HEPEnergyType const dE) {

    if (grammageStart > grammageEnd) { // particle going upstream
      std::swap(grammageStart, grammageEnd);
    }

    GrammageType const deltaX = grammageEnd - grammageStart;

    CORSIKA_LOGGER_TRACE(
        TOutput::getLogger(),
        "dE={} GeV, grammageStart={} g/cm2, End={}g /cm2, deltaX={} g/cm2", dE / 1_GeV,
        grammageStart / 1_g * square(1_cm), grammageEnd / 1_g * square(1_cm),
        deltaX / 1_g * square(1_cm));

    if (deltaX < dX_threshold_) {
      CORSIKA_LOGGER_TRACE(TOutput::getLogger(), "Point-like dE");
      int const maxBin = int(profile_.size() - 1);
      int bin = grammageStart / dX_;
      if (bin < 0) bin = 0;
      if (bin > maxBin) bin = maxBin;
      profile_[bin][static_cast<int>(dEdX_output::ProfileIndex::Total)] += dE;
      if (is_em(PID)) { electromagneticEnergyLost_ += dE; }
      return;
    }

    // only register the range that is covered by the profile
    int const maxBin = int(profile_.size() - 1);
    int binStart = grammageStart / dX_;
    if (binStart < 0) binStart = 0;
    if (binStart > maxBin) binStart = maxBin;
    int binEnd = grammageEnd / dX_;
    if (binEnd < 0) binEnd = 0;
    if (binEnd > maxBin) binEnd = maxBin;

    CORSIKA_LOGGER_TRACE(TOutput::getLogger(), "maxBin={}, binStart={}, binEnd={}",
                         maxBin, binStart, binEnd);

    auto energyCount = HEPEnergyType::zero();

    auto const factor = dE / deltaX; // [ energy / grammage ]
    auto fill = [&](int const bin, GrammageType const weight) {
      auto const increment = factor * weight;
      CORSIKA_LOGGER_TRACE(TOutput::getLogger(),
                           "filling bin={} with weight {} : dE={} GeV ", bin, weight,
                           increment / 1_GeV);
      profile_[bin][static_cast<int>(dEdX_output::ProfileIndex::Total)] += increment;
      energyCount += increment;
    };

    // fill longitudinal profile
    if (binStart == binEnd) {
      fill(binStart, deltaX);
    } else {
      fill(binStart, ((1 + binStart) * dX_ - grammageStart));
      fill(binEnd, (grammageEnd - binEnd * dX_));
      for (int bin = binStart + 1; bin < binEnd; ++bin) { fill(bin, dX_); }
    }

    CORSIKA_LOGGER_TRACE(TOutput::getLogger(), "total energy added to histogram: {} GeV ",
                         energyCount / 1_GeV);
    if (is_em(PID)) { electromagneticEnergyLost_ += energyCount; }
  }

  template <typename TOutput>
  inline void EnergyLossWriter<TOutput>::addBin(
      size_t bin, HEPEnergyType dE) {
    profile_.at(bin)[
        static_cast<int>(dEdX_output::ProfileIndex::Total)] += dE;
  }

  template <typename TOutput>
  inline void EnergyLossWriter<TOutput>::addElectromagneticBin(
      size_t bin, HEPEnergyType dE) {
    addBin(bin, dE);
    electromagneticEnergyLost_ += dE;
  }

  template <typename TOutput>
  inline void EnergyLossWriter<TOutput>::write(Point const& point, Code const PID,
                                               HEPEnergyType const dE) {
    GrammageType grammage = showerAxis_.getProjectedX(point);
    int const maxBin = int(profile_.size() - 1);
    int bin = grammage / dX_;
    if (bin < 0) bin = 0;
    if (bin > maxBin) bin = maxBin;

    CORSIKA_LOGGER_TRACE(TOutput::getLogger(), "add local energy loss bin={} dE={} GeV ",
                         bin, dE / 1_GeV);

    profile_[bin][static_cast<int>(dEdX_output::ProfileIndex::Total)] += dE;
    if (is_em(PID)) { electromagneticEnergyLost_ += dE; }
  }

  template <typename TOutput>
  inline void EnergyLossWriter<TOutput>::write(GrammageType const Xstart,
                                               GrammageType const Xend, Code const PID,
                                               HEPEnergyType const dE) {
    double const bstart = Xstart / dX_;
    double const bend = Xend / dX_;

    if (abs(bstart - floor(bstart + 0.5)) > 1e-2 ||
        abs(bend - floor(bend + 0.5)) > 1e-2 || abs(bend - bstart - 1) > 1e-2) {
      CORSIKA_LOGGER_ERROR(
          TOutput::getLogger(),
          "CascadeEquation (CONEX) and Corsika8 dX grammage binning are not the same! "
          "Xstart={} Xend={} dX={} g/cm2",
          Xstart / 1_g * square(1_cm), Xend / 1_g * square(1_cm),
          dX_ / 1_g * square(1_cm));
      throw std::runtime_error(
          "CONEX and Corsika8 dX grammage binning are not the same!");
    }

    size_t const bin = size_t((bend + bstart) / 2);
    CORSIKA_LOGGER_TRACE(TOutput::getLogger(),
                         "add binned energy loss {} {} bin={} dE={} GeV ", bstart, bend,
                         bin, dE / 1_GeV);
    if (bin >= profile_.size()) {
      CORSIKA_LOGGER_WARN(TOutput::getLogger(),
                          "Grammage bin {} outside of profile {}. skipping.", bin,
                          profile_.size());
      return;
    }
    profile_[bin][static_cast<int>(dEdX_output::ProfileIndex::Total)] += dE;
    if (is_em(PID)) { electromagneticEnergyLost_ += dE; }
  }

  template <typename TOutput>
  inline HEPEnergyType EnergyLossWriter<TOutput>::getEnergyLost() const {
    HEPEnergyType tot = HEPEnergyType::zero();
    for (dEdX_output::Profile const& row : profile_)
      tot += row.at(static_cast<int>(dEdX_output::ProfileIndex::Total));
    return tot;
  }

  template <typename TOutput>
  inline YAML::Node EnergyLossWriter<TOutput>::getConfig() const {

    YAML::Node node;

    node["type"] = "EnergyLoss";
    node["units"]["energy"] = "GeV";
    node["units"]["grammage"] = "g/cm^2";
    node["bin-size"] = dX_ / (1_g / square(1_cm));
    node["nbins"] = nBins_;
    node["grammage_threshold"] = dX_threshold_ / (1_g / square(1_cm));

    return node;
  }

  template <typename TOutput>
  inline YAML::Node EnergyLossWriter<TOutput>::getSummary() const {

    return summary_;
  }

} // namespace corsika

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
  inline ProductionWriter<TOutput>::ProductionWriter(ShowerAxis const& axis,
                                                     GrammageType dX)
      : ProductionWriter<TOutput>{
            axis, static_cast<unsigned int>(axis.getMaximumX() / dX) + 1, dX} {}

  template <typename TOutput>
  inline ProductionWriter<TOutput>::ProductionWriter(ShowerAxis const& axis, size_t nbins,
                                                     GrammageType dX)
      : TOutput(production_profile::ProjectileIndexNames)
      , showerAxis_(axis)
      , dX_(dX)
      , nBins_(nbins)
      , profile_{nbins} {
  }

  template <typename TOutput>
  inline void ProductionWriter<TOutput>::startOfLibrary(
      boost::filesystem::path const& directory) {
    TOutput::startOfLibrary(directory);
    summary_.open(directory);
  }

  template <typename TOutput>
  inline void ProductionWriter<TOutput>::startOfShower(unsigned int const showerId) {
    profile_.clear();
    for (size_t i = 0; i < nBins_; ++i) { profile_.emplace_back(); }
    TOutput::startOfShower(showerId);
  }

  template <typename TOutput>
  inline void ProductionWriter<TOutput>::endOfLibrary() {
    TOutput::endOfLibrary();
  }

  template <typename TOutput>
  inline void ProductionWriter<TOutput>::endOfShower(unsigned int const showerId) {

    // determined XMumax and dNdXmax from quadratic interpolation
    double maximum = 0;
    size_t iMaximum = 0;
    for (size_t i = 0; i < profile_.size() - 3; ++i) {
      double value = (profile_[i + 0].at(
                          static_cast<int>(production_profile::ProjectileIndex::All)) +
                      profile_[i + 1].at(
                          static_cast<int>(production_profile::ProjectileIndex::All)) +
                      profile_[i + 2].at(
                          static_cast<int>(production_profile::ProjectileIndex::All)));
      if (value > maximum) {
        maximum = value;
        iMaximum = i;
      }
    }
    double const dX = dX_ / 1_g * square(1_cm);

    auto [Xmumax, dNdXmumax] = FindXmax::interpolateProfile(
        dX * (0.5 + iMaximum), dX * (1.5 + iMaximum), dX * (2.5 + iMaximum),
        profile_[iMaximum + 0].at(
            static_cast<int>(production_profile::ProjectileIndex::All)),
        profile_[iMaximum + 1].at(
            static_cast<int>(production_profile::ProjectileIndex::All)),
        profile_[iMaximum + 2].at(
            static_cast<int>(production_profile::ProjectileIndex::All)));

    summary_.event(showerId)["XmuMax"] = Xmumax;
    summary_.event(showerId)["dNdXmuMax"] = dNdXmumax;

    // write profile to file
    int iRow{0};
    for (production_profile::ProfileData const& row : profile_) {
      // here: write to underlying writer (e.g. parquet)
      TOutput::write(showerId, iRow * dX_, row);
      iRow++;
    }
    TOutput::endOfShower(showerId);
    summary_.flush();
  }

  template <typename TOutput>
  inline void ProductionWriter<TOutput>::write(Point const& p0, Code const projectile_pid,
                                               double const weight) {

    GrammageType const grammage = showerAxis_.getProjectedX(p0);

    writeProjected(grammage, projectile_pid, weight);
  }

  template <typename TOutput>
  inline void ProductionWriter<TOutput>::writeProjected(
      GrammageType const grammage, Code const projectile_pid,
      double const weight) {

    size_t const bin1 = std::floor(grammage / dX_);
    size_t const bin = std::min(bin1, profile_.size() - 1);

    CORSIKA_LOGGER_TRACE(TOutput::getLogger(), "grammage={} bin={}",
                         grammage / 1_g * square(1_cm), bin);
    addBin(bin, projectile_pid, weight);
  }

  template <typename TOutput>
  inline void ProductionWriter<TOutput>::addBin(
      size_t const bin, Code const projectile_pid,
      double const weight) {
    if (bin >= profile_.size()) {
      CORSIKA_LOGGER_WARN(
          TOutput::getLogger(),
          "Production grammage bin {} outside of profile {}. skipping.",
          bin, profile_.size());
      return;
    }
    if (projectile_pid == Code::Photon) {
      profile_.at(bin)[static_cast<int>(production_profile::ProjectileIndex::Photon)] +=
          weight;
    } else if (projectile_pid == Code::Positron || projectile_pid == Code::Electron) {
      profile_.at(
          bin)[static_cast<int>(production_profile::ProjectileIndex::ElectronPositron)] +=
          weight;
    } else if (is_muon(projectile_pid)) {
      profile_.at(bin)[static_cast<int>(production_profile::ProjectileIndex::Muon)] +=
          weight;
    } else if (is_hadron(projectile_pid)) {
      profile_.at(bin)[static_cast<int>(production_profile::ProjectileIndex::Hadron)] +=
          weight;
      if (projectile_pid == Code::PiPlus || projectile_pid == Code::PiMinus) {
        profile_.at(bin)[static_cast<int>(production_profile::ProjectileIndex::Pion)] +=
            weight;
      } else if (is_kaon(projectile_pid)) {
        profile_.at(bin)[static_cast<int>(production_profile::ProjectileIndex::Kaon)] +=
            weight;
      } else {
        profile_.at(bin)[static_cast<int>(production_profile::ProjectileIndex::Heavy)] +=
            weight;
      }
    }
    profile_.at(bin)[static_cast<int>(production_profile::ProjectileIndex::All)] +=
        weight;
  }

  template <typename TOutput>
  inline YAML::Node ProductionWriter<TOutput>::getConfig() const {

    YAML::Node node;

    node["type"] = "ProductionProfile";
    node["units"]["grammage"] = "g/cm^2";
    node["bin-size"] = dX_ / (1_g / square(1_cm));
    node["nbins"] = nBins_;

    return node;
  }

  template <typename TOutput>
  inline YAML::Node ProductionWriter<TOutput>::getSummary() const {
    return summary_.snapshot();
  }

} // namespace corsika

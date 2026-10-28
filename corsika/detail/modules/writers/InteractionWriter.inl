/*
 * (c) Copyright 2024 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/framework/core/EnergyMomentumOperations.hpp>
#include <corsika/framework/core/Logging.hpp>

namespace corsika {

  template <typename TTracking, typename TOutput>
  inline InteractionWriter<TTracking, TOutput>::InteractionWriter(
      ShowerAxis const& axis, ObservationPlane<TTracking, TOutput> const& obsPlane)
      : obsPlane_(obsPlane)
      , showerAxis_(axis)
      , interactionCounter_(0)
      , showerId_(0) {}

  template <typename TTracking, typename TOutput>
  template <typename TStackView>
  inline void InteractionWriter<TTracking, TOutput>::doSecondaries(TStackView& vS) {
    if (interactionCounter_ != 0) {
      ++interactionCounter_;
      return;
    }

    auto const primary = vS.getProjectile();
    FirstInteractionSnapshot snapshot{primary.getPID(),      primary.getKineticEnergy(),
                                      primary.getPosition(), primary.getDirection(),
                                      primary.getTime(),     {}};
    for (auto particle = vS.begin(); particle != vS.end(); ++particle) {
      snapshot.secondaries_.push_back(InteractionSecondarySnapshot{
          particle.getPID(), particle.getKineticEnergy() + get_mass(particle.getPID()),
          particle.getDirection()});
    }
    recordFirstInteraction(snapshot);
  }

  template <typename TTracking, typename TOutput>
  inline bool InteractionWriter<TTracking, TOutput>::recordFirstInteraction(
      FirstInteractionSnapshot const& snapshot) {
    if (interactionCounter_++) { return false; }
    writeFirstInteraction(snapshot);
    return true;
  }

  template <typename TTracking, typename TOutput>
  inline void InteractionWriter<TTracking, TOutput>::writeFirstInteraction(
      FirstInteractionSnapshot const& snapshot) {
    auto const dX = showerAxis_.getProjectedX(snapshot.position_);
    CORSIKA_LOG_INFO("First interaction at dX {}", dX);
    CORSIKA_LOG_INFO("Primary: {}, E_kin {}", snapshot.parentPid_,
                     snapshot.parentKineticEnergy_);

    Vector const displacement = snapshot.position_ - obsPlane_.getPlane().getCenter();
    auto const x = displacement.dot(obsPlane_.getXAxis());
    auto const y = displacement.dot(obsPlane_.getYAxis());
    auto const z = displacement.dot(obsPlane_.getPlane().getNormal());
    auto const nx = snapshot.direction_.dot(obsPlane_.getXAxis());
    auto const ny = snapshot.direction_.dot(obsPlane_.getYAxis());
    auto const nz = snapshot.direction_.dot(obsPlane_.getPlane().getNormal());
    auto const parent_total_energy =
        snapshot.parentKineticEnergy_ + get_mass(snapshot.parentPid_);
    auto const parent_momentum =
        snapshot.direction_ *
        calculate_momentum(parent_total_energy, get_mass(snapshot.parentPid_));
    auto const px = parent_momentum.dot(obsPlane_.getXAxis());
    auto const py = parent_momentum.dot(obsPlane_.getYAxis());
    auto const pz = parent_momentum.dot(obsPlane_.getPlane().getNormal());

    summary_.event(showerId_)["pdg"] = static_cast<int>(get_PDG(snapshot.parentPid_));
    summary_.event(showerId_)["name"] =
        static_cast<std::string>(get_name(snapshot.parentPid_));
    summary_.event(showerId_)["total_energy"] = parent_total_energy / 1_GeV;
    summary_.event(showerId_)["kinetic_energy"] = snapshot.parentKineticEnergy_ / 1_GeV;
    summary_.event(showerId_)["x"] = x / 1_m;
    summary_.event(showerId_)["y"] = y / 1_m;
    summary_.event(showerId_)["z"] = z / 1_m;
    summary_.event(showerId_)["nx"] = static_cast<double>(nx);
    summary_.event(showerId_)["ny"] = static_cast<double>(ny);
    summary_.event(showerId_)["nz"] = static_cast<double>(nz);
    summary_.event(showerId_)["px"] = static_cast<double>(px / 1_GeV);
    summary_.event(showerId_)["py"] = static_cast<double>(py / 1_GeV);
    summary_.event(showerId_)["pz"] = static_cast<double>(pz / 1_GeV);
    summary_.event(showerId_)["time"] = snapshot.time_ / 1_s;
    summary_.event(showerId_)["slant_depth"] = dX / (1_g / 1_cm / 1_cm);

    for (auto const& secondary : snapshot.secondaries_) {
      if (secondary.totalEnergy_ < get_mass(secondary.pid_)) {
        throw std::runtime_error(
            "first-interaction secondary total energy is below rest mass");
      }
      auto const momentum =
          secondary.direction_ *
          calculate_momentum(secondary.totalEnergy_, get_mass(secondary.pid_));
      *(output_.getWriter())
          << showerId_ << static_cast<int>(get_PDG(secondary.pid_))
          << static_cast<float>(momentum.dot(obsPlane_.getXAxis()) / 1_GeV)
          << static_cast<float>(momentum.dot(obsPlane_.getYAxis()) / 1_GeV)
          << static_cast<float>(momentum.dot(obsPlane_.getPlane().getNormal()) / 1_GeV)
          << parquet::EndRow;
      CORSIKA_LOG_INFO(" 2ndary: {}, E_tot {}", secondary.pid_, secondary.totalEnergy_);
    }
    summary_.event(showerId_)["n_secondaries"] = snapshot.secondaries_.size();
  }

  template <typename TTracking, typename TOutput>
  inline void InteractionWriter<TTracking, TOutput>::startOfLibrary(
      boost::filesystem::path const& directory) {
    summary_.open(directory);
    output_.initStreamer((directory / ("interactions.parquet")).string());

    // enable compression with the default level
    output_.enableCompression();

    output_.addField("pdg", parquet::Repetition::REQUIRED, parquet::Type::INT32,
                     parquet::ConvertedType::INT_32);
    output_.addField("px", parquet::Repetition::REQUIRED, parquet::Type::FLOAT,
                     parquet::ConvertedType::NONE);
    output_.addField("py", parquet::Repetition::REQUIRED, parquet::Type::FLOAT,
                     parquet::ConvertedType::NONE);
    output_.addField("pz", parquet::Repetition::REQUIRED, parquet::Type::FLOAT,
                     parquet::ConvertedType::NONE);

    output_.buildStreamer();

    showerId_ = 0;
    interactionCounter_ = 0;
  }

  template <typename TTracking, typename TOutput>
  inline void InteractionWriter<TTracking, TOutput>::startOfShower(
      unsigned int const showerId) {
    showerId_ = showerId;
    interactionCounter_ = 0;
  }

  template <typename TTracking, typename TOutput>
  inline void InteractionWriter<TTracking, TOutput>::endOfShower(unsigned int const) {
    output_.flushStreamer();
    summary_.flush();
  }

  template <typename TTracking, typename TOutput>
  inline void InteractionWriter<TTracking, TOutput>::endOfLibrary() {
    output_.closeStreamer();
  }

  template <typename TTracking, typename TOutput>
  inline YAML::Node InteractionWriter<TTracking, TOutput>::getConfig() const {
    YAML::Node node;
    node["type"] = "Interactions";
    node["units"]["energy"] = "GeV";
    node["units"]["length"] = "m";
    node["units"]["time"] = "ns";
    node["units"]["grammage"] = "g/cm^2";
    return node;
  }

  template <typename TTracking, typename TOutput>
  inline YAML::Node InteractionWriter<TTracking, TOutput>::getSummary() const {
    return summary_.snapshot();
  }

} // namespace corsika

namespace corsika {
  template <typename TTracking, typename TOutput>
  inline void InteractionWriter<TTracking, TOutput>::writeSummary(
      boost::filesystem::path const& path) const {
    summary_.writeSummary(path);
  }

  template <typename TTracking, typename TOutput>
  inline auto InteractionWriter<TTracking, TOutput>::getInteractionCounter() {
    return interactionCounter_;
  }
} // namespace corsika

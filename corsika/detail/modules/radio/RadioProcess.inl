/*
 * (c) Copyright 2022 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */
#pragma once

#include <corsika/modules/radio/RadioProcess.hpp>

namespace corsika {

  template <typename TObserverCollection, typename TRadioImpl, typename TPropagator>
  inline TRadioImpl&
  RadioProcess<TObserverCollection, TRadioImpl, TPropagator>::implementation() {
    return static_cast<TRadioImpl&>(*this);
  }

  template <typename TObserverCollection, typename TRadioImpl, typename TPropagator>
  inline TRadioImpl const&
  RadioProcess<TObserverCollection, TRadioImpl, TPropagator>::implementation() const {
    return static_cast<TRadioImpl const&>(*this);
  }

  template <typename TObserverCollection, typename TRadioImpl, typename TPropagator>
  inline RadioProcess<TObserverCollection, TRadioImpl, TPropagator>::RadioProcess(
      TObserverCollection& observers, TPropagator& propagator)
      : observers_(observers)
      , propagator_(propagator) {}

  template <typename TObserverCollection, typename TRadioImpl, typename TPropagator>
  template <typename Particle>
  inline ProcessReturn RadioProcess<TObserverCollection, TRadioImpl,
                                    TPropagator>::doContinuous(const Step<Particle>& step,
                                                               const bool) {

    // return immediately if radio process does not have any observers
    if (observers_.size() == 0) return ProcessReturn::Ok;

    // we want the following particles:
    // Code::Electron & Code::Positron

    // we wrap Simulate() in doContinuous as the plan is to add particle level
    // filtering or thinning for calculation of the radio emission. This is
    // important for controlling the runtime of radio (by ignoring particles
    // that aren't going to contribute i.e. heavy hadrons)
    // if (valid(step)) {
    auto const particleID_{step.getParticlePre().getPID()};
    if ((particleID_ == Code::Electron) || (particleID_ == Code::Positron)) {
      auto const length_m =
          (step.getPositionPost() - step.getPositionPre()).getNorm() / 1_m;
      auto const duration_s =
          (step.getTimePost() - step.getTimePre()) / 1_s;
      auto const weight = step.getParticlePre().getWeight();
      auto const kinetic_energy_GeV = step.getEkinPre() / 1_GeV;
      if (std::isfinite(length_m) && length_m > 0. &&
          std::isfinite(duration_s) && duration_s != 0. &&
          std::isfinite(weight) && weight >= 0. &&
          std::isfinite(kinetic_energy_GeV) && kinetic_energy_GeV >= 0.) {
        ++diagnosticSegmentCount_;
        diagnosticWeightedSegmentCount_ += weight;
        diagnosticTrackLengthM_ += length_m;
        diagnosticWeightedTrackLengthM_ += weight * length_m;
        diagnosticEnergyWeightedTrackLengthGeVM_ +=
            weight * length_m * kinetic_energy_GeV;
        diagnosticMaximumSegmentLengthM_ =
            std::max(diagnosticMaximumSegmentLengthM_, length_m);
        if (particleID_ == Code::Electron) {
          diagnosticElectronWeightedTrackLengthM_ += weight * length_m;
          diagnosticSignedChargeTrackLengthM_ -= weight * length_m;
        } else {
          diagnosticPositronWeightedTrackLengthM_ += weight * length_m;
          diagnosticSignedChargeTrackLengthM_ += weight * length_m;
        }
        auto const bin = std::lower_bound(
            diagnosticEnergyUpperEdgesGeV_.begin(),
            diagnosticEnergyUpperEdgesGeV_.end(), kinetic_energy_GeV);
        auto const index = static_cast<std::size_t>(
            std::distance(diagnosticEnergyUpperEdgesGeV_.begin(), bin));
        diagnosticEnergyBinnedTrackLengthM_.at(
            std::min(index, diagnosticEnergyBinnedTrackLengthM_.size() - 1)) +=
            weight * length_m;

        auto const direction_pre =
            step.getDirectionPre().getComponents();
        auto const direction_post =
            step.getDirectionPost().getComponents();
        double direction_dot = 0.;
        std::array<double, 3> direction_delta{};
        for (int axis = 0; axis < 3; ++axis) {
          auto const pre = direction_pre[axis].magnitude();
          auto const post = direction_post[axis].magnitude();
          direction_dot += pre * post;
          direction_delta[axis] = post - pre;
        }
        auto const direction_change_rad =
            std::acos(std::clamp(direction_dot, -1., 1.));
        auto constexpr speed_of_light_m_per_s = 299792458.;
        auto const beta_module =
            length_m / (speed_of_light_m_per_s * duration_s);
        auto const time_residual_s =
            duration_s -
            length_m / speed_of_light_m_per_s;
        diagnosticWeightedDirectionChangeRad_ +=
            weight * direction_change_rad;
        diagnosticWeightedDirectionChangeSquaredRad2_ +=
            weight * direction_change_rad * direction_change_rad;
        diagnosticWeightedBetaDeficitTrackLengthM_ +=
            weight * length_m * (1. - beta_module);
        diagnosticWeightedTimeResidualS_ +=
            weight * time_residual_s;
        diagnosticMaximumDirectionChangeRad_ = std::max(
            diagnosticMaximumDirectionChangeRad_,
            direction_change_rad);
        auto const charge_sign =
            particleID_ == Code::Electron ? -1. : 1.;
        for (int axis = 0; axis < 3; ++axis) {
          diagnosticSignedChargeWeightedDirectionChange_[axis] +=
              charge_sign * weight * direction_delta[axis];
        }
      }
      CORSIKA_LOG_DEBUG("Particle for radio calculation: {} ", particleID_);
      return this->implementation().simulate(step);
    } else {
      CORSIKA_LOG_DEBUG("Particle {} is irrelevant for radio", particleID_);
      return ProcessReturn::Ok;
    }
    //}
  }

  template <typename TObserverCollection, typename TRadioImpl, typename TPropagator>
  template <typename Particle, typename Track>
  inline LengthType
  RadioProcess<TObserverCollection, TRadioImpl, TPropagator>::getMaxStepLength(
      [[maybe_unused]] const Particle& vParticle,
      [[maybe_unused]] const Track& vTrack) const {
    return meter * std::numeric_limits<double>::infinity();
  }

  template <typename TObserverCollection, typename TRadioImpl, typename TPropagator>
  inline void RadioProcess<TObserverCollection, TRadioImpl, TPropagator>::startOfLibrary(
      const boost::filesystem::path& directory) {

    // setup the streamer
    output_.initStreamer((directory / ("observers.parquet")).string());

    // enable compression with the default level
    output_.enableCompression();

    // LCOV_EXCL_START
    // build the schema
    output_.addField("Time", parquet::Repetition::REQUIRED, parquet::Type::DOUBLE,
                     parquet::ConvertedType::NONE);

    output_.addField("Ex", parquet::Repetition::REQUIRED, parquet::Type::DOUBLE,
                     parquet::ConvertedType::NONE);

    output_.addField("Ey", parquet::Repetition::REQUIRED, parquet::Type::DOUBLE,
                     parquet::ConvertedType::NONE);

    output_.addField("Ez", parquet::Repetition::REQUIRED, parquet::Type::DOUBLE,
                     parquet::ConvertedType::NONE);
    // LCOV_EXCL_STOP
    // and build the streamer
    output_.buildStreamer();
  }

  template <typename TObserverCollection, typename TRadioImpl, typename TPropagator>
  inline void RadioProcess<TObserverCollection, TRadioImpl, TPropagator>::endOfShower(
      const unsigned int) {

    // loop over every observer and instruct them to
    // flush data to disk, and then reset the observer
    // before the next event
    for (auto& observer : observers_.getObservers()) {

      auto const sampleRate = observer.getSampleRate() * 1_s;
      auto const radioImplementation =
          static_cast<std::string>(this->implementation().algorithm);

      // get the axis labels for this observer and write the first row.
      axistype axis = observer.implementation().getAxis();

      // get the copy of the waveform data for this event
      std::vector<double> const& dataX = observer.implementation().getWaveformX();
      std::vector<double> const& dataY = observer.implementation().getWaveformY();
      std::vector<double> const& dataZ = observer.implementation().getWaveformZ();

      // check for the axis name
      std::string label = "Unknown";
      if (observer.getDomainLabel() == "Time") {
        label = "Time";
      }
      // LCOV_EXCL_START
      else if (observer.getDomainLabel() == "Frequency") {
        label = "Frequency";
      }
      // LCOV_EXCL_STOP
      if (radioImplementation == "ZHS" && label == "Time") {
        for (size_t i = 0; i < axis.size() - 1; i++) {
          auto time = (axis.at(i + 1) + axis.at(i)) / 2.;
          auto Ex = -(dataX.at(i + 1) - dataX.at(i)) * sampleRate;
          auto Ey = -(dataY.at(i + 1) - dataY.at(i)) * sampleRate;
          auto Ez = -(dataZ.at(i + 1) - dataZ.at(i)) * sampleRate;

          *(output_.getWriter())
              << showerId_ << static_cast<double>(time) << static_cast<double>(Ex)
              << static_cast<double>(Ey) << static_cast<double>(Ez) << parquet::EndRow;
        }
      } else if (radioImplementation == "CoREAS" && label == "Time") {
        for (size_t i = 0; i < axis.size() - 1; i++) {
          *(output_.getWriter())
              << showerId_ << static_cast<double>(axis[i])
              << static_cast<double>(dataX[i]) << static_cast<double>(dataY[i])
              << static_cast<double>(dataZ[i]) << parquet::EndRow;
        }
      }

      observer.reset();
    }

    auto shower = diagnosticSummary_["shower_" + std::to_string(showerId_)];
    shower["segment_count"] = diagnosticSegmentCount_;
    shower["weighted_segment_count"] = diagnosticWeightedSegmentCount_;
    shower["track_length_m"] = diagnosticTrackLengthM_;
    shower["weighted_track_length_m"] = diagnosticWeightedTrackLengthM_;
    shower["electron_weighted_track_length_m"] =
        diagnosticElectronWeightedTrackLengthM_;
    shower["positron_weighted_track_length_m"] =
        diagnosticPositronWeightedTrackLengthM_;
    shower["signed_charge_weighted_track_length_m"] =
        diagnosticSignedChargeTrackLengthM_;
    shower["energy_weighted_track_length_GeV_m"] =
        diagnosticEnergyWeightedTrackLengthGeVM_;
    shower["maximum_segment_length_m"] = diagnosticMaximumSegmentLengthM_;
    shower["weighted_direction_change_rad"] =
        diagnosticWeightedDirectionChangeRad_;
    shower["weighted_direction_change_squared_rad2"] =
        diagnosticWeightedDirectionChangeSquaredRad2_;
    shower["weighted_beta_deficit_track_length_m"] =
        diagnosticWeightedBetaDeficitTrackLengthM_;
    shower["weighted_time_residual_s"] =
        diagnosticWeightedTimeResidualS_;
    shower["maximum_direction_change_rad"] =
        diagnosticMaximumDirectionChangeRad_;
    shower["signed_charge_weighted_direction_change"]["x"] =
        diagnosticSignedChargeWeightedDirectionChange_[0];
    shower["signed_charge_weighted_direction_change"]["y"] =
        diagnosticSignedChargeWeightedDirectionChange_[1];
    shower["signed_charge_weighted_direction_change"]["z"] =
        diagnosticSignedChargeWeightedDirectionChange_[2];
    auto energy_bins = shower["weighted_track_length_by_kinetic_energy"];
    energy_bins["units"]["upper_edge"] = "GeV";
    energy_bins["units"]["weighted_track_length"] = "m";
    for (std::size_t index = 0;
         index < diagnosticEnergyUpperEdgesGeV_.size(); ++index) {
      auto const upper = diagnosticEnergyUpperEdgesGeV_[index];
      energy_bins["upper_edge_GeV"].push_back(
          std::isfinite(upper) ? YAML::Node(upper) : YAML::Node("inf"));
      energy_bins["weighted_track_length_m"].push_back(
          diagnosticEnergyBinnedTrackLengthM_[index]);
    }

    diagnosticSegmentCount_ = 0;
    diagnosticWeightedSegmentCount_ = 0.;
    diagnosticTrackLengthM_ = 0.;
    diagnosticWeightedTrackLengthM_ = 0.;
    diagnosticElectronWeightedTrackLengthM_ = 0.;
    diagnosticPositronWeightedTrackLengthM_ = 0.;
    diagnosticSignedChargeTrackLengthM_ = 0.;
    diagnosticEnergyWeightedTrackLengthGeVM_ = 0.;
    diagnosticMaximumSegmentLengthM_ = 0.;
    diagnosticWeightedDirectionChangeRad_ = 0.;
    diagnosticWeightedDirectionChangeSquaredRad2_ = 0.;
    diagnosticWeightedBetaDeficitTrackLengthM_ = 0.;
    diagnosticWeightedTimeResidualS_ = 0.;
    diagnosticMaximumDirectionChangeRad_ = 0.;
    diagnosticSignedChargeWeightedDirectionChange_.fill(0.);
    diagnosticEnergyBinnedTrackLengthM_.fill(0.);

    // increment our event counter
    showerId_++;
  }

  template <typename TObserverCollection, typename TRadioImpl, typename TPropagator>
  inline void
  RadioProcess<TObserverCollection, TRadioImpl, TPropagator>::endOfLibrary() {
    // One parquet stream stores all showers and uses the mandatory shower
    // column to separate them.  Closing here, rather than after each shower,
    // keeps the streamer valid for multi-event libraries and matches the
    // lifecycle used by the other parquet outputs.
    output_.closeStreamer();
  }

  template <typename TObserverCollection, typename TRadioImpl, typename TPropagator>
  inline YAML::Node
  RadioProcess<TObserverCollection, TRadioImpl, TPropagator>::getConfig() const {

    // top-level YAML node
    YAML::Node config;

    // fill in some basics
    config["type"] = "RadioProcess";
    config["algorithm"] = this->implementation().algorithm;
    config["units"]["time"] = "ns";
    config["units"]["frequency"] = "GHz";
    config["units"]["electric field"] = "V/m";
    config["units"]["distance"] = "m";

    for (auto& observer : observers_.getObservers()) {
      // get the name/location of this observer
      auto name = observer.getName();
      auto location = observer.getLocation().getCoordinates();

      // get the observers config
      config["observers"][name] = observer.getConfig();

      // write the location of this observer
      config["observers"][name]["location"].push_back(location.getX() / 1_m);
      config["observers"][name]["location"].push_back(location.getY() / 1_m);
      config["observers"][name]["location"].push_back(location.getZ() / 1_m);
    }

    return config;
  }

  template <typename TObserverCollection, typename TRadioImpl, typename TPropagator>
  inline YAML::Node
  RadioProcess<TObserverCollection, TRadioImpl, TPropagator>::getSummary() const {
    return diagnosticSummary_;
  }

} // namespace corsika

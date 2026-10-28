/*
 * (c) Copyright 2022 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */
#pragma once

#include <corsika/modules/radio/observers/Observer.hpp>
#include <corsika/modules/radio/propagators/SignalPath.hpp>
#include <yaml-cpp/yaml.h>
#include <vector>

namespace radio_production_observer {
  using namespace corsika;

  /**
   * An implementation of a time-domain observer that has a customized
   * start time, sampling rate, and waveform duration.
   *
   */
  class TimeDomainObserver : public Observer<TimeDomainObserver> {

    TimeType const start_time_;         ///< The start time of this waveform.
    TimeType const duration_;           ///< The duration of this waveform.
    InverseTimeType const sample_rate_; ///< The sampling rate of this observer.
    TimeType const ground_hit_time_; ///< The time the primary particle hits the ground.
    uint64_t const num_bins_;        ///< The number of bins used.
    std::vector<double> waveformEX_; ///< EX polarization.
    std::vector<double> waveformEY_; ///< EY polarization.
    std::vector<double> waveformEZ_; ///< EZ polarization.
    std::vector<long double> const
        time_axis_; ///< The time axis corresponding to the electric field.

  public:
    // import the methods from the observer
    using Observer<TimeDomainObserver>::getName;
    using Observer<TimeDomainObserver>::getLocation;

    /**
     * Construct a new TimeDomainObserver.
     *
     * @param name               The name of this observer.
     * @param location           The location of this observer.
     * @param coordinateSystem   The coordinate system of this observer.
     * @param start_time         The starting time of this waveform.
     * @param duration           The duration of this waveform.
     * @param sample_rate        The sample rate of this waveform.
     * @param ground_hit_time    The time the primary particle hits the ground on a
     * straight vertical line.
     *
     */
    TimeDomainObserver(std::string const& name, Point const& location,
                       CoordinateSystemPtr coordinateSystem, TimeType const& start_time,
                       TimeType const& duration, InverseTimeType const& sample_rate,
                       TimeType const ground_hit_time);

    /**
     * Receive an electric field at this observer.
     *
     * This assumes that the observer will receive
     *  an *instantaneous* electric field modeled as a delta function (or timebin).
     *
     * @param time             The (global) time at which this signal is received.
     * @param receive_vector   The incident unit vector. (not used at the moment)
     * @param field            The incident electric field vector.
     *
     */
    void receive(TimeType const time, SignalPath const& path,
                 ElectricFieldVector const& efield);

    void receive(TimeType const time, SignalPath const& path,
                 VectorPotential const& vectorP);

    /**
     * Return the time-units of each waveform for X polarization
     *
     * This returns them in nanoseconds for ease of use.
     */
    auto const& getWaveformX() const;

    /**
     * Return the time-units of each waveform for Y polarization
     *
     * This returns them in nanoseconds for ease of use.
     */
    auto const& getWaveformY() const;

    /**
     * Return the time-units of each waveform for Z polarization
     *
     * This returns them in nanoseconds for ease of use.
     */
    auto const& getWaveformZ() const;

    /**
     * Return a label that indicates that this is a time
     * domain observer
     *
     * This returns the string "Time".
     */
    std::string const getDomainLabel();

    /**
     * Creates time-units of each waveform.
     *
     * It creates them in nanoseconds for ease of use.
     */
    std::vector<long double> createTimeAxis() const;

    /**
     * Return the time-units of each waveform.
     *
     * This returns them in nanoseconds for ease of use.
     */
    auto const getAxis() const;

    /**
     * Returns the sampling rate of the time domain observer.
     */
    InverseTimeType const& getSampleRate() const;

    /**
     * Returns the start time of detection for the time domain observer.
     */
    TimeType const& getStartTime() const;

    /**
     * Returns the duration covered by the waveform.
     */
    TimeType const& getDuration() const;

    /**
     * Add an externally accumulated waveform to this observer.
     *
     * The CUDA radio backend uses this once per shower to merge device
     * CoREAS electric fields or ZHS vector potentials before the normal
     * RadioProcess::endOfShower output path runs.
     */
    void addWaveform(std::vector<double> const& x, std::vector<double> const& y,
                     std::vector<double> const& z);

    /**
     * Reset the observer before starting a new simulation.
     */
    void reset();

    /**
     * Return a YAML configuration for this observer.
     */
    YAML::Node getConfig() const;

  }; // END: class TimeDomainObserver

} // namespace radio_production_observer

namespace radio_production_observer {
  using namespace corsika;

  inline TimeDomainObserver::TimeDomainObserver(std::string const& name,
                                                Point const& location,
                                                CoordinateSystemPtr coordinateSystem,
                                                TimeType const& start_time,
                                                TimeType const& duration,
                                                InverseTimeType const& sample_rate,
                                                TimeType const ground_hit_time)
      : Observer(name, location, coordinateSystem)
      , start_time_(start_time)
      , duration_(std::abs(duration / 1_s) * 1_s)
      , sample_rate_(std::abs(sample_rate / 1_Hz) * 1_Hz)
      , ground_hit_time_(ground_hit_time)
      , num_bins_(static_cast<std::size_t>(duration_ * sample_rate_ + 1.5l))
      , waveformEX_(num_bins_, 0)
      , waveformEY_(num_bins_, 0)
      , waveformEZ_(num_bins_, 0)
      , time_axis_(createTimeAxis()) {
    if (0_s == duration_) {
      CORSIKA_LOG_WARN(
          "Observer: \"{}\" has a duration of zero. Nothing will be injected into it",
          name);
    } else if (duration_ != duration) {
      CORSIKA_LOG_WARN(
          "Observer: \"{}\" was given a negative duration. Set to absolute value.", name);
    }

    if (sample_rate_ != sample_rate) {
      CORSIKA_LOG_WARN(
          "Observer: \"{}\" was given a negative sampling rate. Set to absolute value.",
          name);
    }
  }

  inline void TimeDomainObserver::receive(TimeType const time, SignalPath const& path,
                                          ElectricFieldVector const& efield) {

    if (time < start_time_ || time > (start_time_ + duration_)) {
      return;
    } else {
      // figure out the correct timebin to store the E-field value.
      // NOTE: static cast is implicitly flooring
      auto timebin{static_cast<std::size_t>(
          std::floor((time - start_time_) * sample_rate_ + 0.5l))};
      CORSIKA_LOG_TRACE(
          "Receiving particle - time {} (bin {}), emit {}, receive {}, efield {}", time,
          timebin, path.emit_, path.receive_, efield);

      auto const dotProd = path.emit_.dot(-path.receive_);
      // If propagation not in a straight line
      // catch case where numerical rounding gives errors
      if (dotProd < (1.0 - 1e-8)) {

        if (dotProd == -1) { // handle by hand (cross product will fail)
          // store the x,y,z electric field components.
          CORSIKA_LOG_TRACE("Rotating efield by 180 deg");
          auto const& components{efield.getComponents(coordinateSystem_)};
          waveformEX_.at(timebin) += -1 * (components.getX() * (1_m / 1_V));
          waveformEY_.at(timebin) += -1 * (components.getY() * (1_m / 1_V));
          waveformEZ_.at(timebin) += -1 * (components.getZ() * (1_m / 1_V));
          return;
        }

        auto const rotation_axis = path.emit_.cross(-path.receive_).normalized();
        CORSIKA_LOG_TRACE("Rotating efield by {} deg about axis {}",
                          asin(dotProd) * 180 / M_PI, rotation_axis);
        auto const sinTh = sqrt(1 - dotProd * dotProd);
        auto const rotated =
            efield * dotProd +
            (rotation_axis.cross(efield) * sinTh +
             rotation_axis * (rotation_axis.dot(efield)) * (1 - dotProd));

        // store the x,y,z electric field components.
        auto const& Electric_field_components{rotated.getComponents(coordinateSystem_)};
        waveformEX_.at(timebin) += (Electric_field_components.getX() * (1_m / 1_V));
        waveformEY_.at(timebin) += (Electric_field_components.getY() * (1_m / 1_V));
        waveformEZ_.at(timebin) += (Electric_field_components.getZ() * (1_m / 1_V));

        return;
      }

      // store the x,y,z electric field components.
      auto const& Electric_field_components{efield.getComponents(coordinateSystem_)};
      waveformEX_.at(timebin) += (Electric_field_components.getX() * (1_m / 1_V));
      waveformEY_.at(timebin) += (Electric_field_components.getY() * (1_m / 1_V));
      waveformEZ_.at(timebin) += (Electric_field_components.getZ() * (1_m / 1_V));
      // TODO: Check how they are stored in memory, row-wise or column-wise? Probably use
      // a 3D object
    }
  }

  inline void TimeDomainObserver::receive(const TimeType time, SignalPath const& path,
                                          const VectorPotential& vectorP) {

    // ZHS contributes bin-integrated potentials, not instantaneous fields.
    // A label just outside the nominal time window can still round into its
    // first/last potential bin. Discarding it before binning leaves those bins
    // incomplete and creates an artificial pulse when RadioProcess computes
    // E[i] = -(A[i+1] - A[i]) / dt. Clip the bin, not the contribution label;
    // keep the existing array length, output time axis and CoREAS semantics.
    auto const bin = std::floor((time - start_time_) * sample_rate_ + 0.5l);
    if (!(bin >= 0 && bin < num_bins_)) {
      return;
    } else {
      auto const timebin = static_cast<std::size_t>(bin);
      CORSIKA_LOG_TRACE(
          "Receiving particle - time {} (bin {}), emit {}, receive {}, vector {}", time,
          timebin, path.emit_, path.receive_, vectorP);

      auto const dotProd = path.emit_.dot(-path.receive_);
      // If propagation not in a straight line
      // catch case where numerical rounding gives errors
      if (dotProd < (1.0 - 1e-8)) {

        if (dotProd == -1) { // handle by hand (cross product will fail)
          // store the x,y,z electric field components.
          auto const& components{vectorP.getComponents(coordinateSystem_)};
          waveformEX_.at(timebin) += -1 * (components.getX() * (1_m / (1_V * 1_s)));
          waveformEY_.at(timebin) += -1 * (components.getY() * (1_m / (1_V * 1_s)));
          waveformEZ_.at(timebin) += -1 * (components.getZ() * (1_m / (1_V * 1_s)));
          return;
        }

        auto const rotation_axis = path.emit_.cross(-path.receive_).normalized();
        CORSIKA_LOG_TRACE("Rotating efield by {} deg about axis {}",
                          asin(dotProd) * 180 / M_PI, rotation_axis);
        auto const sinTh = sqrt(1 - dotProd * dotProd);
        auto const rotated =
            vectorP * dotProd +
            (rotation_axis.cross(vectorP) * sinTh +
             rotation_axis * (rotation_axis.dot(vectorP)) * (1 - dotProd));

        // store the x,y,z electric field components.
        auto const& Vector_potential_components{rotated.getComponents(coordinateSystem_)};
        waveformEX_.at(timebin) +=
            (Vector_potential_components.getX() * (1_m / (1_V * 1_s)));
        waveformEY_.at(timebin) +=
            (Vector_potential_components.getY() * (1_m / (1_V * 1_s)));
        waveformEZ_.at(timebin) +=
            (Vector_potential_components.getZ() * (1_m / (1_V * 1_s)));

        return;
      }

      // store the x,y,z electric field components.
      auto const& Vector_potential_components{vectorP.getComponents(coordinateSystem_)};
      waveformEX_.at(timebin) +=
          (Vector_potential_components.getX() * (1_m / (1_V * 1_s)));
      waveformEY_.at(timebin) +=
          (Vector_potential_components.getY() * (1_m / (1_V * 1_s)));
      waveformEZ_.at(timebin) +=
          (Vector_potential_components.getZ() * (1_m / (1_V * 1_s)));
      // TODO: Check how they are stored in memory, row-wise or column-wise? Probably use
      // a 3D object
    }
  }

  inline auto const& TimeDomainObserver::getWaveformX() const { return waveformEX_; }

  inline auto const& TimeDomainObserver::getWaveformY() const { return waveformEY_; }

  inline auto const& TimeDomainObserver::getWaveformZ() const { return waveformEZ_; }

  inline std::string const TimeDomainObserver::getDomainLabel() { return "Time"; }

  inline std::vector<long double> TimeDomainObserver::createTimeAxis() const {

    // create a 1-D xtensor to store time values so we can print them later.
    std::vector<long double> times(num_bins_, 0);

    // calculate the sample_period
    auto sample_period{1 / sample_rate_};

    // fill in every time-value
    for (uint64_t i = 0; i < num_bins_; i++) {
      // create the current time in nanoseconds
      times.at(i) = static_cast<long double>(
          ((start_time_ - ground_hit_time_) + i * sample_period) / 1_ns);
    }

    return times;
  }

  inline auto const TimeDomainObserver::getAxis() const { return time_axis_; }

  inline InverseTimeType const& TimeDomainObserver::getSampleRate() const {
    return sample_rate_;
  }

  inline TimeType const& TimeDomainObserver::getStartTime() const { return start_time_; }

  inline TimeType const& TimeDomainObserver::getDuration() const { return duration_; }

  inline void TimeDomainObserver::addWaveform(std::vector<double> const& x,
                                              std::vector<double> const& y,
                                              std::vector<double> const& z) {
    if (x.size() != num_bins_ || y.size() != num_bins_ || z.size() != num_bins_) {
      throw std::invalid_argument("radio waveform size does not match observer");
    }
    for (std::size_t index = 0; index < num_bins_; ++index) {
      waveformEX_[index] += x[index];
      waveformEY_[index] += y[index];
      waveformEZ_[index] += z[index];
    }
  }

  inline void TimeDomainObserver::reset() {
    std::fill(waveformEX_.begin(), waveformEX_.end(), 0);
    std::fill(waveformEY_.begin(), waveformEY_.end(), 0);
    std::fill(waveformEZ_.begin(), waveformEZ_.end(), 0);
  }

  inline YAML::Node TimeDomainObserver::getConfig() const {

    // top-level config
    YAML::Node config;

    config["type"] = "TimeDomainObserver";
    config["start time"] = start_time_ / 1_ns;
    config["duration"] = duration_ / 1_ns;
    config["number of bins"] = duration_ * sample_rate_;
    config["sampling frequency"] = sample_rate_ / 1_GHz;

    return config;
  }

} // namespace radio_production_observer

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

namespace corsika {

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
    void addWaveform(std::vector<double> const& x,
                     std::vector<double> const& y,
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

} // namespace corsika

#include <corsika/detail/modules/radio/observers/TimeDomainObserver.inl>

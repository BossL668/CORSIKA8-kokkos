/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include "NativeEmissionReference.hpp"
#include <corsika/framework/core/Step.hpp>
#include <corsika/framework/geometry/RootCoordinateSystem.hpp>
#include <corsika/modules/radio/CoREAS.hpp>
#include <corsika/modules/radio/ZHS.hpp>
#include <corsika/modules/radio/detectors/ObserverCollection.hpp>
#include <corsika/modules/radio/observers/TimeDomainObserver.hpp>
#include <corsika/modules/radio/propagators/SignalPath.hpp>
#include <corsika/kokkos/radio/firn/Fft.hpp>

namespace firn_test {
  using namespace corsika;
  namespace firn = corsika::kokkos::radio::firn;
  namespace {
    struct Particle {
      Point position_;
      DirectionVector direction_;
      TimeType time_;
      double chargeWeight_;
      Point const& getPosition() const { return position_; }
      DirectionVector const& getDirection() const { return direction_; }
      TimeType getTime() const { return time_; }
      HEPEnergyType getKineticEnergy() const { return 1_GeV; }
      Code getPID() const { return chargeWeight_ < 0. ? Code::Electron : Code::Positron; }
      double getWeight() const { return std::abs(chargeWeight_); }
    };
    struct UniformPropagator {
      double index_;
      std::vector<SignalPath> propagate(Particle const&, Point const& start,
                                        Point const& observer) const {
        auto const delta = observer - start;
        auto const distance = delta.getNorm();
        auto const direction = delta.normalized();
        return {SignalPath(index_ * distance / constants::c, index_, index_, index_,
                           direction, -direction, distance, {start, observer})};
      }
    };
  } // namespace
  firn::GeometricalOpticsSpectra native_uniform_emission(
      firn::GeometricalOpticsTrack const& track,
      std::vector<firn::vector_type> const& positions,
      firn::OnlineRadioConfig const& config, double index) {
    logging::set_level(logging::level::err);
    auto const coordinates = get_root_CoordinateSystem();
    auto point = [&](firn::vector_type const& value) {
      return Point(coordinates, value[0] * 1_m, value[1] * 1_m, value[2] * 1_m);
    };
    auto const start = point(track.startM_), end = point(track.endM_);
    auto const duration = (track.endTimeS_ - track.startTimeS_) * 1_s;
    Particle particle{start, (end - start).normalized(), track.startTimeS_ * 1_s,
                      track.chargeWeight_};
    StraightTrajectory trajectory(Line(start, (end - start) / duration), duration);
    Step<Particle> step(particle, trajectory);
    ObserverCollection<TimeDomainObserver> detector;
    for (auto const& position : positions)
      detector.addObserver(TimeDomainObserver(
          "reference", point(position), coordinates, config.timeOriginS_ * 1_s,
          config.grid_.sampleCount_ * config.grid_.sampleIntervalS_ * 1_s,
          1. / (config.grid_.sampleIntervalS_ * 1_s), 0_s,
          VectorPotentialWindowMode::RoundedBins));
    UniformPropagator propagator{index};
    if (config.emissionFormalism_ == firn::EmissionFormalism::ZHS) {
      ZHS<decltype(detector), UniformPropagator> zhs(detector, propagator);
      zhs.simulate(step);
    } else {
      CoREAS<decltype(detector), UniformPropagator> coreas(detector, propagator);
      coreas.simulate(step);
    }
    firn::GeometricalOpticsSpectra result;
    for (auto const& observer : detector.getObservers()) {
      result.total_.emplace_back();
      std::array<std::vector<double>, 3> components{
          observer.getWaveformX(), observer.getWaveformY(), observer.getWaveformZ()};
      for (int component = 0; component < 3; ++component) {
        std::vector<std::complex<double>> values(config.grid_.sampleCount_);
        for (std::size_t sample = 0; sample < values.size(); ++sample) {
          values[sample] = components[component][sample];
          if (config.emissionFormalism_ == firn::EmissionFormalism::ZHS)
            values[sample] =
                (components[component][sample] - components[component][sample + 1]) /
                config.grid_.sampleIntervalS_;
        }
        firn::fft_in_place(values, false);
        auto& output = result.total_.back()[component];
        for (std::size_t frequency = 0; frequency < config.grid_.frequencyCount();
             ++frequency) {
          double const phase =
              config.emissionFormalism_ == firn::EmissionFormalism::ZHS
                  ? -firn::pi * double(frequency) / config.grid_.sampleCount_
                  : 0.;
          output.push_back(values[frequency] * config.grid_.sampleIntervalS_ *
                           std::exp(std::complex<double>(0., phase)));
        }
      }
    }
    return result;
  }
} // namespace firn_test

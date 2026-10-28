/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

namespace scene_reference {
  using namespace corsika;
  using namespace corsika::kokkos::radio;
  namespace snapshot_detail {
    template <typename TDetector>
    std::vector<RadioObserverSnapshot> make_observers(
        TDetector const& detector, CoordinateSystemPtr const& coordinate_system) {
      std::vector<RadioObserverSnapshot> result;
      result.reserve(static_cast<std::size_t>(detector.size()));
      for (auto const& observer : detector.getObservers()) {
        RadioObserverSnapshot snapshot{};
        auto const position = observer.getLocation().getCoordinates(coordinate_system);
        for (int axis = 0; axis < 3; ++axis) {
          snapshot.positionM_[axis] = position[axis] / 1_m;
        }
        snapshot.startTimeS_ = observer.getStartTime() / 1_s;
        snapshot.durationS_ = observer.getDuration() / 1_s;
        snapshot.sampleRateHz_ = observer.getSampleRate() / 1_Hz;
        snapshot.numberOfBins_ = observer.getWaveformX().size();
        result.push_back(snapshot);
      }
      return result;
    }
  } // namespace snapshot_detail
  template <typename TEnvironment, typename TCoreasDetector, typename TZhsDetector>
  RadioConfig make_radio_config(TEnvironment const& environment, Point const& upper_limit,
                                Point const& lower_limit, LengthType const step,
                                TCoreasDetector const& coreas_detector,
                                TZhsDetector const& zhs_detector, bool coreas_enabled,
                                bool zhs_enabled) {
    if (!(step > 0_m)) {
      throw std::invalid_argument("GPU radio propagation table requires a positive step");
    }

    RadioConfig config{};
    config.enabled_ = coreas_enabled || zhs_enabled;
    config.coreasEnabled_ = coreas_enabled;
    config.zhsEnabled_ = zhs_enabled;
    auto& table = config.propagation_;
    auto const coordinate_system = environment.getCoordinateSystem();
    auto const minimum_x = lower_limit.getCoordinates().getX();
    auto const minimum_y = lower_limit.getCoordinates().getY();
    auto const maximum_height = upper_limit.getCoordinates().getZ();
    auto const minimum_height = lower_limit.getCoordinates().getZ() - 1_km;
    auto const inverse_step = 1 / step;
    auto const number_of_bins =
        static_cast<std::size_t>((maximum_height - minimum_height) * inverse_step + 1);
    if (number_of_bins < 11) {
      throw std::invalid_argument(
          "GPU radio propagation table requires at least eleven points");
    }

    table.minimumHeightM_ = minimum_height / 1_m;
    table.maximumHeightM_ = maximum_height / 1_m;
    table.stepM_ = step / 1_m;
    table.inverseStepPerM_ = inverse_step * 1_m;
    table.refractivity_.reserve(number_of_bins);
    table.integratedRefractivity_.reserve(number_of_bins);

    auto const* universe = environment.getUniverse().get();
    for (std::size_t index = 0; index < number_of_bins; ++index) {
      Point point{coordinate_system, minimum_x, minimum_y, minimum_height + index * step};
      auto const* node = universe->getContainingNode(point);
      table.refractivity_.push_back(node->getModelProperties().getRefractiveIndex(point) -
                                    1.);
    }

    // Keep the scalar propagator's current discrete integral exactly. With
    // the production 1 m step this is also the usual rectangular sum.
    auto const step_over_meter = inverse_step * 1_m;
    table.integratedRefractivity_.push_back(table.refractivity_.front() *
                                            step_over_meter);
    for (std::size_t index = 1; index < number_of_bins; ++index) {
      table.integratedRefractivity_.push_back(table.integratedRefractivity_.back() +
                                              table.refractivity_[index] *
                                                  step_over_meter);
    }

    table.slopeRefractivityLower_ =
        (table.refractivity_[10] - table.refractivity_.front()) / 10.;
    table.slopeIntegratedRefractivityLower_ =
        (table.integratedRefractivity_[10] - table.integratedRefractivity_.front()) / 10.;
    auto const last = number_of_bins - 1;
    table.slopeRefractivityUpper_ =
        (table.refractivity_[last] - table.refractivity_[last - 10]) / 10.;
    table.slopeIntegratedRefractivityUpper_ =
        (table.integratedRefractivity_[last] - table.integratedRefractivity_[last - 10]) /
        10.;

    if (coreas_enabled) {
      config.coreasObservers_ =
          snapshot_detail::make_observers(coreas_detector, coordinate_system);
    }
    if (zhs_enabled) {
      config.zhsObservers_ =
          snapshot_detail::make_observers(zhs_detector, coordinate_system);
    }
    return config;
  }
} // namespace scene_reference

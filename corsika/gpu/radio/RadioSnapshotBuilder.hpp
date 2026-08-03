/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <cmath>
#include <cstddef>
#include <stdexcept>

#include <corsika/framework/core/PhysicalUnits.hpp>
#include <corsika/framework/geometry/Point.hpp>
#include <corsika/gpu/radio/Types.hpp>

namespace corsika::gpu::radio {

  namespace snapshot_detail {

    template <typename TDetector>
    std::vector<RadioObserverSnapshot> makeObservers(
        TDetector const& detector, CoordinateSystemPtr const& coordinate_system) {
      std::vector<RadioObserverSnapshot> result;
      result.reserve(static_cast<std::size_t>(detector.size()));
      for (auto const& observer : detector.getObservers()) {
        RadioObserverSnapshot snapshot{};
        auto const position =
            observer.getLocation().getCoordinates(coordinate_system);
        for (int axis = 0; axis < 3; ++axis) {
          snapshot.position_m[axis] = position[axis] / 1_m;
        }
        snapshot.start_time_s = observer.getStartTime() / 1_s;
        snapshot.duration_s = observer.getDuration() / 1_s;
        snapshot.sample_rate_Hz = observer.getSampleRate() / 1_Hz;
        snapshot.number_of_bins = observer.getWaveformX().size();
        result.push_back(snapshot);
      }
      return result;
    }

  } // namespace snapshot_detail

  /**
   * Build the exact device snapshot used by the CUDA radio kernels from the
   * same environment and limits as TabulatedFlatAtmospherePropagator.
   */
  template <typename TEnvironment, typename TCoreasDetector,
            typename TZhsDetector>
  GpuRadioConfig makeGpuRadioConfig(
      TEnvironment const& environment, Point const& upper_limit,
      Point const& lower_limit, LengthType const step,
      TCoreasDetector const& coreas_detector,
      TZhsDetector const& zhs_detector,
      bool coreas_enabled = true, bool zhs_enabled = true) {
    if (!(step > 0_m)) {
      throw std::invalid_argument(
          "GPU radio propagation table requires a positive step");
    }

    GpuRadioConfig config{};
    config.enabled = coreas_enabled || zhs_enabled;
    config.coreas_enabled = coreas_enabled;
    config.zhs_enabled = zhs_enabled;
    auto& table = config.propagation;
    auto const coordinate_system = environment.getCoordinateSystem();
    auto const minimum_x =
        lower_limit.getCoordinates().getX();
    auto const minimum_y =
        lower_limit.getCoordinates().getY();
    auto const maximum_height =
        upper_limit.getCoordinates().getZ();
    auto const minimum_height =
        lower_limit.getCoordinates().getZ() - 1_km;
    auto const inverse_step = 1 / step;
    auto const number_of_bins = static_cast<std::size_t>(
        (maximum_height - minimum_height) * inverse_step + 1);
    if (number_of_bins < 11) {
      throw std::invalid_argument(
          "GPU radio propagation table requires at least eleven points");
    }

    table.minimum_height_m = minimum_height / 1_m;
    table.maximum_height_m = maximum_height / 1_m;
    table.step_m = step / 1_m;
    table.inverse_step_per_m = inverse_step * 1_m;
    table.refractivity.reserve(number_of_bins);
    table.integrated_refractivity.reserve(number_of_bins);

    auto const* universe =
        environment.getUniverse().get();
    for (std::size_t index = 0; index < number_of_bins; ++index) {
      Point point{coordinate_system, minimum_x, minimum_y,
                  minimum_height + index * step};
      auto const* node = universe->getContainingNode(point);
      table.refractivity.push_back(
          node->getModelProperties().getRefractiveIndex(point) - 1.);
    }

    // Keep the scalar propagator's current discrete integral exactly. With
    // the production 1 m step this is also the usual rectangular sum.
    auto const step_over_meter = inverse_step * 1_m;
    table.integrated_refractivity.push_back(
        table.refractivity.front() * step_over_meter);
    for (std::size_t index = 1; index < number_of_bins; ++index) {
      table.integrated_refractivity.push_back(
          table.integrated_refractivity.back() +
          table.refractivity[index] * step_over_meter);
    }

    table.slope_refractivity_lower =
        (table.refractivity[10] - table.refractivity.front()) / 10.;
    table.slope_integrated_refractivity_lower =
        (table.integrated_refractivity[10] -
         table.integrated_refractivity.front()) /
        10.;
    auto const last = number_of_bins - 1;
    table.slope_refractivity_upper =
        (table.refractivity[last] - table.refractivity[last - 10]) /
        10.;
    table.slope_integrated_refractivity_upper =
        (table.integrated_refractivity[last] -
         table.integrated_refractivity[last - 10]) /
        10.;

    if (coreas_enabled) {
      config.coreas_observers =
          snapshot_detail::makeObservers(coreas_detector, coordinate_system);
    }
    if (zhs_enabled) {
      config.zhs_observers =
          snapshot_detail::makeObservers(zhs_detector, coordinate_system);
    }
    return config;
  }

} // namespace corsika::gpu::radio

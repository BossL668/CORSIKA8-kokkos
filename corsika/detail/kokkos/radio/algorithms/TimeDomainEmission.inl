/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/kokkos/radio/RadioProjectionData.hpp>
#include <cmath>

namespace corsika::kokkos::radio::time_domain_detail {
  template <typename TAmplitude, typename TReceiver>
  CORSIKA_HOST_DEVICE inline void deposit_zhs(TimeDomainRay<TAmplitude> const& ray,
                                              double chargeWeight, double intervalS,
                                              std::size_t sampleCount,
                                              TReceiver const& receive) {
    double const width = std::abs(ray.retardedDurationS_);
    double const lower = (ray.arrivalTimeS_ - .5 * width) / intervalS;
    double const upper = (ray.arrivalTimeS_ + .5 * width) / intervalS;
    // The integrated potential stays finite as 1 - n beta.k tends to zero.
    double const areaScale = scalar_constants::ElementaryChargeC * chargeWeight /
                             (4. * Pi * scalar_constants::VacuumPermittivityFPerM *
                              SpeedOfLightMPerS * SpeedOfLightMPerS);
    double first = std::floor(lower + .5), last = std::floor(upper + .5);
    if (last < 0. || first > double(sampleCount)) return;
    first = first < 0. ? 0. : first;
    last = last > double(sampleCount) ? double(sampleCount) : last;
    if (std::floor(lower + .5) == std::floor(upper + .5)) {
      TAmplitude potential[3];
      for (int component = 0; component < 3; ++component)
        potential[component] =
            ray.projectedDisplacement_[component] * areaScale / intervalS;
      receive(static_cast<std::size_t>(first), potential, ray.topology_);
      return;
    }
    for (auto bin = static_cast<std::size_t>(first);
         bin <= static_cast<std::size_t>(last); ++bin) {
      double const begin = lower > double(bin) - .5 ? lower : double(bin) - .5;
      double const finish = upper < double(bin) + .5 ? upper : double(bin) + .5;
      double const fraction = finish > begin ? finish - begin : 0.;
      TAmplitude potential[3];
      for (int component = 0; component < 3; ++component)
        potential[component] =
            ray.projectedDisplacement_[component] * areaScale * fraction / width;
      receive(bin, potential, ray.topology_);
    }
  }

  template <typename TAmplitude, typename TReceiver>
  CORSIKA_HOST_DEVICE inline void deposit_coreas(
      TimeDomainRay<TAmplitude> const& start, TimeDomainRay<TAmplitude> const& end,
      TimeDomainRay<TAmplitude> const& middle, double durationS, double chargeWeight,
      double intervalS, std::size_t sampleCount, TReceiver const& receive,
      double approximationThreshold) {
    double startTime = start.arrivalTimeS_, endTime = end.arrivalTimeS_;
    auto startRay = start, endRay = end;
    double const preDoppler = start.retardedDurationS_ / durationS;
    double const postDoppler = end.retardedDurationS_ / durationS;
    bool const nearCone = std::abs(preDoppler) < approximationThreshold ||
                          std::abs(postDoppler) < approximationThreshold;
    if (nearCone) {
      startRay = middle;
      endRay = middle;
      startTime = middle.arrivalTimeS_ - .5 * middle.retardedDurationS_;
      endTime = middle.arrivalTimeS_ + .5 * middle.retardedDurationS_;
    }
    double const areaScale = scalar_constants::ElementaryChargeC * chargeWeight /
                             (4. * Pi * scalar_constants::VacuumPermittivityFPerM *
                              SpeedOfLightMPerS * SpeedOfLightMPerS);
    double const startBin = std::floor(startTime / intervalS);
    double const endBin = std::floor(endTime / intervalS);
    auto deposit = [&](double bin, TAmplitude const* values) {
      if (bin >= 0. && bin < double(sampleCount))
        receive(static_cast<std::size_t>(bin), values, middle.topology_);
    };
    TAmplitude first[3], second[3];
    if (startBin == endBin) {
      // Linear deposition preserves both the integral and first time moment of
      // an unresolved endpoint pair, on either side of the Cherenkov cone.
      // Combine before deposition to avoid subtracting large fields in one bin.
      double const meanFraction = .5 * (startTime + endTime) / intervalS - startBin;
      for (int component = 0; component < 3; ++component) {
        if (nearCone) {
          // The common midpoint coefficient cancels the Doppler denominator
          // analytically, including its zero limit. No division by zero occurs.
          first[component] = -middle.projectedDisplacement_[component] * areaScale /
                             (intervalS * intervalS);
          second[component] = -first[component];
        } else {
          auto const a = -startRay.projectedDisplacement_[component] * areaScale /
                         (intervalS * startRay.retardedDurationS_);
          auto const b = endRay.projectedDisplacement_[component] * areaScale /
                         (intervalS * endRay.retardedDurationS_);
          auto const moment = (a - b) * (.5 * (endTime - startTime) / intervalS);
          first[component] = (a + b) * (1. - meanFraction) + moment;
          second[component] = (a + b) * meanFraction - moment;
        }
      }
      deposit(startBin, first);
      deposit(startBin + 1., second);
      return;
    }
    for (int endpoint = 0; endpoint < 2; ++endpoint) {
      auto const& ray = endpoint == 0 ? startRay : endRay;
      double const time = endpoint == 0 ? startTime : endTime;
      double const bin = endpoint == 0 ? startBin : endBin;
      double const fraction = time / intervalS - bin;
      double const sign = endpoint == 0 ? -1. : 1.;
      for (int component = 0; component < 3; ++component) {
        auto const field = ray.projectedDisplacement_[component] * (sign * areaScale) /
                           (intervalS * ray.retardedDurationS_);
        first[component] = field * (1. - fraction);
        second[component] = field * fraction;
      }
      deposit(bin, first);
      deposit(bin + 1., second);
    }
  }
} // namespace corsika::kokkos::radio::time_domain_detail

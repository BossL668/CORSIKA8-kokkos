/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

namespace corsika::kokkos::radio::firn {
  inline StraightAverageIndexBackend::StraightAverageIndexBackend(
      std::shared_ptr<VerticalOpticalProfile const> profile)
      : profile_(std::move(profile)) {
    if (!profile_) {
      throw std::runtime_error("straight-average backend needs a profile");
    }
  }

  inline GreenResponseBatch StraightAverageIndexBackend::evaluate(
      vector_type const& source, vector_type const& observer,
      FrequencyGrid const& grid) const {
    grid.validate();
    vector_type const displacement = subtract(observer, source);
    double const distance = norm(displacement);
    if (!(distance > 0.0)) throw std::runtime_error("source equals observer");
    vector_type const direction = multiply(displacement, 1.0 / distance);
    auto lineIntegral = [&](std::function<double(double)> const& value) {
      return distance * detail::simpson(
                            [&](double fraction) {
                              double const z = source[2] + fraction * displacement[2];
                              return value(z);
                            },
                            0.0, 1.0, 256);
    };
    GreenResponseBatch result;
    result.path_.valid_ = true;
    result.path_.topology_ = "straight_path_average_index";
    result.path_.sourceRefractiveIndex_ = profile_->refractiveIndexAtZ(source[2]);
    result.path_.observerRefractiveIndex_ = profile_->refractiveIndexAtZ(observer[2]);
    result.path_.geometricLengthM_ = distance;
    result.path_.opticalPathM_ =
        lineIntegral([&](double z) { return profile_->refractiveIndexAtZ(z); });
    result.path_.travelTimeS_ = result.path_.opticalPathM_ / c_m_per_s;
    result.path_.geometricSpreadingPerM_ = 1.0 / distance;
    result.path_.emitDirection_ = direction;
    result.path_.receiveDirection_ = direction;
    result.dyadicPerM_.resize(grid.frequencyCount());
    double const commonAttenuation =
        profile_->attenuationIsFrequencyIndependent()
            ? lineIntegral([&](double z) { return profile_->attenuationAtZ(z, 0.0); })
            : 0.0;
    for (std::size_t index = 0; index < grid.frequencyCount(); ++index) {
      double const frequency = grid.frequencyHz(index);
      double const attenuation = profile_->attenuationIsFrequencyIndependent()
                                     ? commonAttenuation
                                     : lineIntegral([&](double z) {
                                         return profile_->attenuationAtZ(z, frequency);
                                       });
      double const phase = -2.0 * pi * frequency * result.path_.travelTimeS_;
      std::complex<double> const scale =
          std::polar(std::exp(-attenuation) / distance, phase);
      result.dyadicPerM_[index] = transverse_dyadic(direction, scale);
    }
    result.path_.attenuationExponent_ = commonAttenuation;
    return result;
  }

  inline std::string StraightAverageIndexBackend::name() const {
    return "straight_average_index";
  }
} // namespace corsika::kokkos::radio::firn

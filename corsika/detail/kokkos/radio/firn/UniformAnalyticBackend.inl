/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

namespace corsika::kokkos::radio::firn {
  inline UniformAnalyticBackend::UniformAnalyticBackend(UniformProfile profile)
      : profile_(std::move(profile)) {}

  inline GreenResponseBatch UniformAnalyticBackend::evaluate(
      vector_type const& source, vector_type const& observer,
      FrequencyGrid const& grid) const {
    grid.validate();
    vector_type const displacement = subtract(observer, source);
    double const distance = norm(displacement);
    if (!(distance > 0.0)) throw std::runtime_error("source equals observer");
    vector_type const direction = multiply(displacement, 1.0 / distance);
    MediumSample const sample = profile_.sample(source, 0.0);
    GreenResponseBatch result;
    result.dyadicPerM_.resize(grid.frequencyCount());
    result.path_.valid_ = true;
    result.path_.topology_ = "uniform_direct";
    result.path_.sourceRefractiveIndex_ = sample.refractiveIndex_;
    result.path_.observerRefractiveIndex_ = sample.refractiveIndex_;
    result.path_.geometricLengthM_ = distance;
    result.path_.opticalPathM_ = sample.refractiveIndex_ * distance;
    result.path_.travelTimeS_ = result.path_.opticalPathM_ / c_m_per_s;
    result.path_.attenuationExponent_ = sample.attenuationPerM_ * distance;
    result.path_.geometricSpreadingPerM_ = 1.0 / distance;
    result.path_.emitDirection_ = direction;
    result.path_.receiveDirection_ = direction;
    for (std::size_t index = 0; index < grid.frequencyCount(); ++index) {
      double const frequency = grid.frequencyHz(index);
      double const phase = -2.0 * pi * frequency * result.path_.travelTimeS_;
      std::complex<double> const scale =
          std::polar(std::exp(-result.path_.attenuationExponent_) / distance, phase);
      result.dyadicPerM_[index] = transverse_dyadic(direction, scale);
    }
    return result;
  }

  inline std::string UniformAnalyticBackend::name() const { return "uniform_analytic"; }
} // namespace corsika::kokkos::radio::firn

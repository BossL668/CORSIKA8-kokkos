/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

namespace corsika::kokkos::radio::firn {
  inline StratifiedSurfaceReflectionBackend::StratifiedSurfaceReflectionBackend(
      std::shared_ptr<VerticalOpticalProfile const> profile,
      double exteriorRefractiveIndex)
      : profile_(std::move(profile))
      , solver_(profile_ ? *profile_
                         : throw std::runtime_error("GO needs an optical profile"))
      , exteriorIndex_(exteriorRefractiveIndex) {
    if (!profile_ || !(exteriorIndex_ >= 1.0) || !std::isfinite(exteriorIndex_)) {
      throw std::runtime_error("invalid stratified reflection backend");
    }
  }

  inline GreenResponseBatch StratifiedSurfaceReflectionBackend::evaluate(
      vector_type const& source, vector_type const& observer,
      FrequencyGrid const& grid) const {
    grid.validate();
    PathDiagnostics const path = solver_.surfaceReflected(source, observer, 0.0);
    if (!path.valid_) {
      throw std::runtime_error("stratified surface-reflected ray is absent");
    }
    double const nSurface = profile_->refractiveIndexAtZ(profile_->maximumSupportedZ());
    double const sinIncident = path.invariantP_ / nSurface;
    double const cosIncident = std::sqrt(std::max(0.0, 1.0 - sinIncident * sinIncident));
    // exp(+i omega t) synthesis requires an outward-decaying evanescent wave.
    std::complex<double> const cosTransmitted = std::conj(std::sqrt(std::complex<double>(
        1.0 - std::pow(path.invariantP_ / exteriorIndex_, 2.0), 0.0)));
    std::complex<double> const rS =
        (nSurface * cosIncident - exteriorIndex_ * cosTransmitted) /
        (nSurface * cosIncident + exteriorIndex_ * cosTransmitted);
    std::complex<double> const rP =
        (exteriorIndex_ * cosIncident - nSurface * cosTransmitted) /
        (exteriorIndex_ * cosIncident + nSurface * cosTransmitted);
    vector_type const horizontal = detail::horizontal_unit(source, observer);
    vector_type const s{-horizontal[1], horizontal[0], 0.0};
    vector_type const pSource = normalized(cross(s, path.emitDirection_));
    vector_type const pObserver = normalized(cross(s, path.receiveDirection_));
    GreenResponseBatch result;
    result.path_ = path;
    result.dyadicPerM_.resize(grid.frequencyCount(), zero_matrix());
    for (std::size_t index = 0; index < grid.frequencyCount(); ++index) {
      double const phase = -2.0 * pi * grid.frequencyHz(index) * path.travelTimeS_;
      std::complex<double> const common = std::polar(
          path.geometricSpreadingPerM_ * std::exp(-path.attenuationExponent_), phase);
      for (std::size_t row = 0; row < 3; ++row) {
        for (std::size_t col = 0; col < 3; ++col) {
          result.dyadicPerM_[index][row][col] =
              common * (rS * s[row] * s[col] + rP * pObserver[row] * pSource[col]);
        }
      }
    }
    return result;
  }

  inline std::string StratifiedSurfaceReflectionBackend::name() const {
    return "stratified_surface_reflection";
  }
} // namespace corsika::kokkos::radio::firn

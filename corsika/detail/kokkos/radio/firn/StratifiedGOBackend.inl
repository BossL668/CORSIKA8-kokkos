/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

namespace corsika::kokkos::radio::firn {
  inline StratifiedGOBackend::StratifiedGOBackend(
      std::shared_ptr<VerticalOpticalProfile const> profile)
      : profile_(std::move(profile))
      , solver_(profile_ ? *profile_
                         : throw std::runtime_error("GO needs an optical profile")) {
    if (!profile_) throw std::runtime_error("GO backend needs an optical profile");
  }

  inline GreenResponseBatch StratifiedGOBackend::evaluate(
      vector_type const& source, vector_type const& observer,
      FrequencyGrid const& grid) const {
    grid.validate();
    PathDiagnostics const geometry = solver_.direct(source, observer, 0.0);
    if (!geometry.valid_) {
      throw std::runtime_error("stratified GO has no supported direct path: " +
                               geometry.topology_);
    }
    GreenResponseBatch result;
    result.path_ = geometry;
    result.dyadicPerM_.resize(grid.frequencyCount());
    auto attenuationAtFrequency = [&](double frequency) {
      return std::abs(detail::simpson(
          [&](double z) {
            double const n = profile_->refractiveIndexAtZ(z);
            return profile_->attenuationAtZ(z, frequency) * n /
                   std::sqrt(std::max(n * n - geometry.invariantP_ * geometry.invariantP_,
                                      1.0e-28));
          },
          source[2], observer[2]));
    };
    double const commonAttenuation = profile_->attenuationIsFrequencyIndependent()
                                         ? geometry.attenuationExponent_
                                         : 0.0;
    for (std::size_t index = 0; index < grid.frequencyCount(); ++index) {
      double const frequency = grid.frequencyHz(index);
      // Geometry is frequency independent for the currently supported
      // non-dispersive isotropic profiles.  Re-solving 2049 identical rays for
      // every track subdivision is both wasteful and a production hazard.
      double const attenuation = profile_->attenuationIsFrequencyIndependent()
                                     ? commonAttenuation
                                     : attenuationAtFrequency(frequency);
      double const phase = -2.0 * pi * frequency * geometry.travelTimeS_;
      std::complex<double> const scale =
          std::polar(geometry.geometricSpreadingPerM_ * std::exp(-attenuation), phase);
      result.dyadicPerM_[index] = stratified_polarization_transport_dyadic(
          geometry.emitDirection_, geometry.receiveDirection_, scale);
    }
    return result;
  }

  inline std::string StratifiedGOBackend::name() const { return "stratified_go"; }
} // namespace corsika::kokkos::radio::firn

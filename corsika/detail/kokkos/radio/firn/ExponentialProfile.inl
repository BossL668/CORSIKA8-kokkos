/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

namespace corsika::kokkos::radio::firn {
  inline ExponentialOpticalParameters ExponentialProfile::parameters() const {
    return {deepIndex_, deltaIndex_,      scaleDepthM_,
            surfaceZM_, attenuationPerM_, minimumZM_};
  }
  inline ExponentialProfile::ExponentialProfile(double deepIndex, double deltaIndex,
                                                double scaleDepthM, double surfaceZM,
                                                double attenuationPerM, double minimumZM)
      : deepIndex_(deepIndex)
      , deltaIndex_(deltaIndex)
      , scaleDepthM_(scaleDepthM)
      , surfaceZM_(surfaceZM)
      , attenuationPerM_(attenuationPerM)
      , minimumZM_(minimumZM) {
    if (!(deepIndex_ > 1.0) || !(deltaIndex_ >= 0.0) ||
        !(deltaIndex_ < deepIndex_ - 1.0) || !(scaleDepthM_ > 0.0) ||
        !(attenuationPerM_ >= 0.0) || !(minimumZM_ < surfaceZM_) ||
        !std::isfinite(deepIndex_) || !std::isfinite(deltaIndex_) ||
        !std::isfinite(scaleDepthM_) || !std::isfinite(surfaceZM_) ||
        !std::isfinite(attenuationPerM_) || !std::isfinite(minimumZM_)) {
      throw std::runtime_error("invalid exponential optical profile");
    }
  }

  inline double ExponentialProfile::minimumSupportedZ() const { return minimumZM_; }

  inline double ExponentialProfile::maximumSupportedZ() const { return surfaceZM_; }

  inline double ExponentialProfile::refractiveIndexAtZ(double zM) const {
    return deepIndex_ - deltaIndex_ * std::exp((zM - surfaceZM_) / scaleDepthM_);
  }

  inline double ExponentialProfile::attenuationAtZ(double, double) const {
    return attenuationPerM_;
  }

  inline bool ExponentialProfile::attenuationIsFrequencyIndependent() const {
    return true;
  }

  inline ExponentialEndpointData ExponentialProfile::endpointRayData(
      double q, double deepZM, double shallowZM) const {
    double const a = deepIndex_;
    double const h = scaleDepthM_;
    double const deepDeficit = deltaIndex_ * std::exp((deepZM - surfaceZM_) / h);
    double const shallowDeficit = deltaIndex_ * std::exp((shallowZM - surfaceZM_) / h);
    double const nd = a - deepDeficit, ns = a - shallowDeficit;
    double const p = ns * std::sqrt(1.0 - q * q);
    double const g2 = (a - p) * (a + p), g = std::sqrt(g2);
    double const root = std::sqrt((nd - ns) * (nd + ns) + ns * ns * q * q);
    double const deepCosine = root / nd;
    double const deepT =
        std::log((std::sqrt((a - p) * (nd + p)) + std::sqrt((a + p) * (nd - p))) /
                 std::sqrt(2.0 * p * deepDeficit));
    double const c = (a + p) * ns / (g * (ns + p));
    double const x = c * q, shallowT = std::atanh(x);
    double const integral = 2.0 * (deepT - shallowT) / g;
    double const hyperbolicDifference = std::atanh(deepCosine) - std::atanh(q);
    double const dpdq = -ns * ns * q / p;
    double const deepDerivative =
        h / g2 * (2.0 * a * a / g * deepT - (a * nd + p * p) / root);
    double const dxdq = c * (1.0 + q * dpdq * (1.0 / (a + p) + p / g2 - 1.0 / (ns + p)));
    double const shallowDerivative = 2.0 * h * a * a / (g2 * g) * dpdq * shallowT +
                                     2.0 * p * h / g * dxdq / (1.0 - x * x);
    // Signed q continues smoothly through the horizontal endpoint. Unlike
    // dX/dp, dX/dq remains finite; no angular interval is discarded.
    return {p,
            p * h * integral,
            h * (a * integral - hyperbolicDifference),
            h * (a * a * integral - a * hyperbolicDifference - (root - ns * q)),
            deepDerivative * dpdq - shallowDerivative,
            deepCosine};
  }

  inline double ExponentialProfile::turningLegHorizontal(double p,
                                                         double endpointZM) const {
    double const a = deepIndex_;
    double const deficit =
        deltaIndex_ * std::exp((endpointZM - surfaceZM_) / scaleDepthM_);
    double const n = a - deficit;
    double const gap = a - p;
    double const t =
        std::log((std::sqrt(gap * (n + p)) + std::sqrt((a + p) * (gap - deficit))) /
                 std::sqrt(2.0 * p * deficit));
    return 2.0 * scaleDepthM_ * p * t / std::sqrt(gap * (a + p));
  }

  inline double ExponentialProfile::turningLegHorizontalDerivative(
      double p, double endpointZM) const {
    double const deficit =
        deltaIndex_ * std::exp((endpointZM - surfaceZM_) / scaleDepthM_);
    double const n = deepIndex_ - deficit;
    double const gap = deepIndex_ - p;
    if (!(p > 0.0) || !(deficit > 0.0) || !(gap > deficit)) {
      throw std::runtime_error("invalid exponential turning-leg derivative");
    }
    double const g2 = gap * (deepIndex_ + p);
    double const root = std::sqrt((gap - deficit) * (n + p));
    // atanh(u) = log((A+B)/sqrt(A*A-B*B)); do not round u to one.
    double const t = std::log(
        (std::sqrt(gap * (n + p)) + std::sqrt((deepIndex_ + p) * (gap - deficit))) /
        std::sqrt(2.0 * p * deficit));
    return scaleDepthM_ / g2 *
           (2.0 * deepIndex_ * deepIndex_ / std::sqrt(g2) * t -
            (deepIndex_ * n + p * p) / root);
  }

  inline std::string ExponentialProfile::profileFingerprint() const {
    std::ostringstream stream;
    stream << "exponential_z" << std::setprecision(17) << ':' << deepIndex_ << ':'
           << deltaIndex_ << ':' << scaleDepthM_ << ':' << surfaceZM_ << ':'
           << attenuationPerM_ << ':' << minimumZM_;
    return stream.str();
  }
} // namespace corsika::kokkos::radio::firn

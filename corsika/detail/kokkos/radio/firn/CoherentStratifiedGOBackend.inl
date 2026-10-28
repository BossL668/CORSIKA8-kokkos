/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

namespace corsika::kokkos::radio::firn {
  inline CoherentStratifiedGOBackend::CoherentStratifiedGOBackend(
      std::shared_ptr<VerticalOpticalProfile const> profile,
      double exteriorRefractiveIndex, bool includeSurfaceReflection, bool includeTurning,
      double foldGuardTimeS)
      : profile_(std::move(profile))
      , solver_(profile_ ? *profile_
                         : throw std::runtime_error("GO needs an optical profile"))
      , exteriorIndex_(exteriorRefractiveIndex)
      , includeSurfaceReflection_(includeSurfaceReflection)
      , includeTurning_(includeTurning)
      , foldGuardTimeS_(foldGuardTimeS) {
    if (!profile_ || !(exteriorIndex_ >= 1.0) || !(foldGuardTimeS_ >= 0.0) ||
        !std::isfinite(exteriorIndex_) || !std::isfinite(foldGuardTimeS_)) {
      throw std::runtime_error("invalid coherent stratified GO backend");
    }
    if (!profile_->attenuationIsFrequencyIndependent()) {
      throw std::runtime_error(
          "coherent GO currently requires frequency-independent attenuation");
    }
  }

  inline GreenResponseBatch CoherentStratifiedGOBackend::evaluate(
      vector_type const& source, vector_type const& observer,
      FrequencyGrid const& grid) const {
    grid.validate();
    std::vector<GreenResponseBranch> branches;

    PathDiagnostics const direct = solver_.direct(source, observer, 0.0);
    if (direct.valid_) branches.push_back(goBranch(direct, grid));

    if (includeSurfaceReflection_) {
      PathDiagnostics const reflected = solver_.surfaceReflected(source, observer, 0.0);
      if (reflected.valid_) branches.push_back(reflectedBranch(reflected, grid));
    }

    if (includeTurning_) {
      auto turning = solver_.turningBranches(source, observer, 0.0);
      std::sort(turning.begin(), turning.end(),
                [](PathDiagnostics const& a, PathDiagnostics const& b) {
                  return a.travelTimeS_ < b.travelTimeS_;
                });
      if (turning.size() >= 2 &&
          turning[1].travelTimeS_ - turning[0].travelTimeS_ < foldGuardTimeS_) {
        throw std::runtime_error(
            "turning branches lie inside the configured fold guard; "
            "use a full-wave table");
      }
      for (std::size_t index = 0; index < turning.size(); ++index) {
        // The solver supplies the point-source caustic phase. The +/-pi/4
        // factors of a stationary-phase integral alone are not Green phases.
        turning[index].topology_ = index == 0 ? "turning_short" : "turning_long";
        branches.push_back(goBranch(turning[index], grid));
      }
    }

    if (branches.empty()) {
      throw std::runtime_error(
          "no ordinary GO path exists; shadow/fold requires full-wave response");
    }
    GreenResponseBatch result;
    result.branches_ = std::move(branches);
    result.dyadicPerM_.assign(grid.frequencyCount(), zero_matrix());
    for (auto const& branch : result.branches_) {
      for (std::size_t frequency = 0; frequency < grid.frequencyCount(); ++frequency) {
        for (std::size_t row = 0; row < 3; ++row) {
          for (std::size_t col = 0; col < 3; ++col) {
            result.dyadicPerM_[frequency][row][col] +=
                branch.dyadicPerM_[frequency][row][col];
          }
        }
      }
    }
    auto const representative =
        std::min_element(result.branches_.begin(), result.branches_.end(),
                         [](GreenResponseBranch const& a, GreenResponseBranch const& b) {
                           return a.path_.travelTimeS_ < b.path_.travelTimeS_;
                         });
    result.path_ = representative->path_;
    result.path_.topology_ = "coherent_stratified_go_total";
    return result;
  }

  inline std::string CoherentStratifiedGOBackend::name() const {
    return "coherent_stratified_go";
  }

  inline GreenResponseBranch CoherentStratifiedGOBackend::goBranch(
      PathDiagnostics path, FrequencyGrid const& grid) const {
    GreenResponseBranch result;
    result.path_ = std::move(path);
    result.dyadicPerM_.resize(grid.frequencyCount());
    for (std::size_t index = 0; index < grid.frequencyCount(); ++index) {
      double const phase =
          -2.0 * pi * grid.frequencyHz(index) * result.path_.travelTimeS_ +
          result.path_.phaseOffsetRad_;
      std::complex<double> const scale =
          std::polar(result.path_.geometricSpreadingPerM_ *
                         std::exp(-result.path_.attenuationExponent_),
                     phase);
      result.dyadicPerM_[index] = stratified_polarization_transport_dyadic(
          result.path_.emitDirection_, result.path_.receiveDirection_, scale);
    }
    return result;
  }

  inline GreenResponseBranch CoherentStratifiedGOBackend::reflectedBranch(
      PathDiagnostics path, FrequencyGrid const& grid) const {
    double const nSurface = profile_->refractiveIndexAtZ(profile_->maximumSupportedZ());
    double const sinIncident = path.invariantP_ / nSurface;
    double const cosIncident = std::sqrt(std::max(0.0, 1.0 - sinIncident * sinIncident));
    // With exp(+i omega t) synthesis the outgoing evanescent wave needs
    // Im(k_z)<0, so exp(-i k_z z) decays into the exterior.
    std::complex<double> const cosTransmitted = std::conj(std::sqrt(std::complex<double>(
        1.0 - std::pow(path.invariantP_ / exteriorIndex_, 2.0), 0.0)));
    std::complex<double> const rS =
        (nSurface * cosIncident - exteriorIndex_ * cosTransmitted) /
        (nSurface * cosIncident + exteriorIndex_ * cosTransmitted);
    std::complex<double> const rP =
        (exteriorIndex_ * cosIncident - nSurface * cosTransmitted) /
        (exteriorIndex_ * cosIncident + nSurface * cosTransmitted);
    vector_type s{-path.emitDirection_[1], path.emitDirection_[0], 0.0};
    if (norm(s) < 1.0e-14) s = {0.0, 1.0, 0.0};
    s = normalized(s);
    vector_type const pSource = normalized(cross(s, path.emitDirection_));
    vector_type const pObserver = normalized(cross(s, path.receiveDirection_));
    GreenResponseBranch result;
    result.path_ = std::move(path);
    result.dyadicPerM_.assign(grid.frequencyCount(), zero_matrix());
    for (std::size_t index = 0; index < grid.frequencyCount(); ++index) {
      double const phase =
          -2.0 * pi * grid.frequencyHz(index) * result.path_.travelTimeS_;
      std::complex<double> const common =
          std::polar(result.path_.geometricSpreadingPerM_ *
                         std::exp(-result.path_.attenuationExponent_),
                     phase);
      for (std::size_t row = 0; row < 3; ++row) {
        for (std::size_t col = 0; col < 3; ++col) {
          result.dyadicPerM_[index][row][col] =
              common * (rS * s[row] * s[col] + rP * pObserver[row] * pSource[col]);
        }
      }
    }
    return result;
  }
} // namespace corsika::kokkos::radio::firn

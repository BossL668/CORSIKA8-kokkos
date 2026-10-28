/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

namespace corsika::kokkos::radio::firn {
  inline StratifiedTurningGOBackend::StratifiedTurningGOBackend(
      std::shared_ptr<VerticalOpticalProfile const> profile, std::size_t branchIndex)
      : profile_(std::move(profile))
      , solver_(profile_ ? *profile_
                         : throw std::runtime_error("GO needs an optical profile"))
      , branchIndex_(branchIndex) {
    if (!profile_) throw std::runtime_error("turning GO needs an optical profile");
  }

  inline GreenResponseBatch StratifiedTurningGOBackend::evaluate(
      vector_type const& source, vector_type const& observer,
      FrequencyGrid const& grid) const {
    grid.validate();
    auto branches = solver_.turningBranches(source, observer, 0.0);
    std::sort(branches.begin(), branches.end(),
              [](PathDiagnostics const& a, PathDiagnostics const& b) {
                return a.travelTimeS_ < b.travelTimeS_;
              });
    if (branchIndex_ >= branches.size()) {
      throw std::runtime_error(
          "requested turning branch is absent or lies at a fold; "
          "a full-wave response table is required");
    }
    PathDiagnostics const path = branches[branchIndex_];
    GreenResponseBatch result;
    result.path_ = path;
    result.dyadicPerM_.resize(grid.frequencyCount());
    for (std::size_t index = 0; index < grid.frequencyCount(); ++index) {
      double const phase =
          -2.0 * pi * grid.frequencyHz(index) * path.travelTimeS_ + path.phaseOffsetRad_;
      std::complex<double> const scale = std::polar(
          path.geometricSpreadingPerM_ * std::exp(-path.attenuationExponent_), phase);
      result.dyadicPerM_[index] = stratified_polarization_transport_dyadic(
          path.emitDirection_, path.receiveDirection_, scale);
    }
    return result;
  }

  inline std::string StratifiedTurningGOBackend::name() const {
    return "stratified_turning_go_branch_" + std::to_string(branchIndex_);
  }
} // namespace corsika::kokkos::radio::firn

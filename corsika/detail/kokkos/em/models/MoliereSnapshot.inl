/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <cmath>

namespace corsika::kokkos {

  namespace moliere_snapshot_detail {

    CORSIKA_HOST_DEVICE inline bool finite(double value) {
#if defined(__CUDA_ARCH__)
      return ::isfinite(value);
#else
      return std::isfinite(value);
#endif
    }

  } // namespace moliere_snapshot_detail

  CORSIKA_HOST_DEVICE inline bool valid_moliere_snapshot(
      MoliereSnapshot const& snapshot) {
    using namespace moliere_snapshot_detail;
    if (snapshot.componentCount_ == 0 ||
        snapshot.componentCount_ > MaxMoliereComponents ||
        snapshot.maximumWeightIndex_ >= snapshot.componentCount_ ||
        !finite(snapshot.particleMassMeV_) || !(snapshot.particleMassMeV_ > 0.) ||
        !finite(snapshot.electronMassMeV_) || !(snapshot.electronMassMeV_ > 0.) ||
        !finite(snapshot.fineStructureConstant_) ||
        !(snapshot.fineStructureConstant_ > 0.) || !finite(snapshot.avogadroPerMol_) ||
        !(snapshot.avogadroPerMol_ > 0.) || !finite(snapshot.hbarMeVS_) ||
        !(snapshot.hbarMeVS_ > 0.) || !finite(snapshot.speedOfLightCmPerS_) ||
        !(snapshot.speedOfLightCmPerS_ > 0.) || !finite(snapshot.zSquaredOverAAverage_) ||
        !(snapshot.zSquaredOverAAverage_ > 0.) || !finite(snapshot.inverseWeightZzSum_) ||
        !(snapshot.inverseWeightZzSum_ > 0.) || !finite(snapshot.eulerMascheroni_)) {
      return false;
    }
    for (std::uint32_t index = 0; index < snapshot.componentCount_; ++index) {
      if (!finite(snapshot.components_[index].nuclearCharge_) ||
          !(snapshot.components_[index].nuclearCharge_ > 0.) ||
          !finite(snapshot.components_[index].weightZz_) ||
          !(snapshot.components_[index].weightZz_ > 0.) ||
          !finite(snapshot.components_[index].chi0SquaredMeV2_) ||
          !(snapshot.components_[index].chi0SquaredMeV2_ > 0.) ||
          !finite(snapshot.components_[index].coulombCorrection_) ||
          !(snapshot.components_[index].coulombCorrection_ > 0.)) {
        return false;
      }
    }
    for (std::size_t index = 0; index < MoliereSeriesCoefficientCount; ++index) {
      if (!finite(snapshot.c1_[index]) || !finite(snapshot.c2_[index])) { return false; }
    }
    for (std::size_t index = 0; index < MoliereLargeSeriesCoefficientCount; ++index) {
      if (!finite(snapshot.c2Large_[index]) || !finite(snapshot.s2Large_[index])) {
        return false;
      }
    }
    for (double coefficient : snapshot.c1IntegralLarge_) {
      if (!finite(coefficient)) { return false; }
    }
    return true;
  }

} // namespace corsika::kokkos

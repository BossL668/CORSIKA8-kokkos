/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

namespace corsika::kokkos {
  inline ShowerConfig make_shower_config(BackendConfig const& config) {
    return {config.randomSeed_, config.showerId_, config.thinning_,
            config.profileProjection_.fixedPointWeightLimit_,
            config.profileProjection_.fixedPointEnergyLimitGeV_};
  }
} // namespace corsika::kokkos

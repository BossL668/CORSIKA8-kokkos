/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <cmath>
#include <utility>

#include <corsika/gpu/em/tables/RateTable.hpp>

namespace corsika::gpu::em::tables::testing {

  inline RateTableSet makeFlatRateTableFixture() {
    RateTableSet table;
    table.metadata.proposal_version = "7.6.2";
    table.metadata.generator_version = "flat-test";
    table.metadata.medium_name = "dry_air";
    table.metadata.proposal_medium_hash = 9001;
    table.metadata.energy_cut_MeV = 0.5;
    table.metadata.relative_v_cut = 0.01;
    table.metadata.energy_min_MeV = 1.;
    table.metadata.energy_max_MeV = 100.;
    table.metadata.requested_relative_tolerance = 1.e-3;
    table.metadata.measured_max_relative_error = 5.e-4;
    table.metadata.requested_loss_relative_tolerance = 1.e-3;
    table.metadata.measured_max_loss_relative_error = 4.e-4;
    table.metadata.components = {
        {14, 101, 0.78479, "nitrogen"},
        {16, 102, 0.21052, "oxygen"},
        {40, 103, 0.00469, "argon"}};
    table.metadata.photon_pair_lpm = {
        1.2048e-3,
        2.504e19,
        7.22,
        // Deliberately small in this synthetic fixture so one batch covers
        // both accepted and LPM-suppressed queue branches.
        1.,
        2.8179403262e-13,
        7.2973525693e-3,
        {{101, 7., 182.7},
         {102, 8., 182.7},
         {103, 18., 184.15}}};
    table.metadata.brems_lpm = {
        1.2048e-3,
        2.504e19,
        7.22,
        1.,
        0.5109989461,
        0.5109989461,
        105.6583745,
        2.8179403262e-13,
        7.2973525693e-3,
        {{101, 7., 14.0067, 182.7},
         {102, 8., 15.999, 182.7},
         {103, 18., 39.948, 184.15}}};

    ParticleRateTable photon;
    photon.pdg_id = 22;
    photon.particle_name = "gamma";
    photon.interaction_hash = 9101;
    photon.energies_MeV = {1., 10., 100.};

    RateColumn pair{
        1000000013, 101, "Photopair", "PhotoPairKochMotz",
        "nitrogen", {1., 10., 100.}, {}};
    pair.inverse_cdf.reference_mode = "proposal_interpolated";
    pair.inverse_cdf.energies_MeV = {1., 10., 100.};
    pair.inverse_cdf.quantile_offsets = {0, 3, 7, 10};
    pair.inverse_cdf.quantiles = {
        LossQuantileMinimum, 0.5, LossQuantileMaximum,
        LossQuantileMinimum, 0.25, 0.75, LossQuantileMaximum,
        LossQuantileMinimum, 0.5, LossQuantileMaximum};
    for (std::size_t row = 0; row < 3; ++row) {
      auto const begin =
          static_cast<std::size_t>(
              pair.inverse_cdf.quantile_offsets[row]);
      auto const end =
          static_cast<std::size_t>(
              pair.inverse_cdf.quantile_offsets[row + 1]);
      auto const energy = pair.inverse_cdf.energies_MeV[row];
      for (auto index = begin; index < end; ++index) {
        auto const quantile = pair.inverse_cdf.quantiles[index];
        auto const logit =
            std::log(quantile / (1. - quantile));
        pair.inverse_cdf.v_loss.push_back(
            1.e-2 * std::sqrt(energy) *
            std::exp(0.01 * logit));
      }
    }

    RateColumn pair_final_state{
        2000000013, 101, "PhotopairFinalState",
        "PhotoPairProductionKochMotzSauterNormalizedKinematicSplit",
        "nitrogen",
        {0., 0., 0.}, {}};
    pair_final_state.inverse_cdf = pair.inverse_cdf;
    // Exercise the analytical Koch--Motz path below the table domain while
    // retaining a tabulated high-energy path in the same validation batch.
    pair_final_state.inverse_cdf.energies_MeV = {10., 30., 100.};

    RateColumn compton{
        1000000010, 102, "Compton", "ComptonKleinNishina",
        "oxygen", {0., 2., 4.}, {}};
    compton.inverse_cdf.reference_mode = "proposal_direct";
    compton.inverse_cdf.energies_MeV = {1., 100.};
    compton.inverse_cdf.quantile_offsets = {0, 2, 4};
    compton.inverse_cdf.quantiles = {
        EpairLossQuantileMinimum, EpairLossQuantileMaximum,
        EpairLossQuantileMinimum, EpairLossQuantileMaximum};
    // A kinematically valid constant transfer fraction makes the synthetic
    // end-to-end photon pipeline exercise the GPU Compton final state. The
    // interpolation machinery itself is tested independently with
    // non-constant columns.
    compton.inverse_cdf.v_loss = {0.4, 0.4, 0.4, 0.4};

    RateColumn photoelectric{
        1000000016, 103, "Photoeffect", "PhotoeffectSauter",
        "argon", {1., 1., 1.}, {}};
    photoelectric.inverse_cdf.reference_mode =
        "proposal_direct";
    photoelectric.inverse_cdf.energies_MeV = {1., 100.};
    photoelectric.inverse_cdf.quantile_offsets = {0, 2, 4};
    photoelectric.inverse_cdf.quantiles = {
        LossQuantileMinimum, LossQuantileMaximum,
        LossQuantileMinimum, LossQuantileMaximum};
    // PROPOSAL Photoeffect::CalculateStochasticLoss always returns v=1.
    photoelectric.inverse_cdf.v_loss = {1., 1., 1., 1.};

    RateColumn zero{
        1000000007, 103, "Disabled", "ZeroRate", "argon",
        {0., 0., 0.}, {}};

    photon.columns = {
        std::move(pair), std::move(pair_final_state),
        std::move(compton), std::move(photoelectric),
        std::move(zero)};
    table.particles.push_back(std::move(photon));
    table.continuous_energy_tables = {
        {11,
         "electron",
         "proposal_interpolated",
         0.5109989461,
         1.0109989461,
         4.e-4,
         {1.0109989461, 10., 100.},
         {2., 2.5, 3.},
         {0., 3., 10.}},
        {-11,
         "positron",
         "proposal_interpolated",
         0.5109989461,
         1.0109989461,
         4.e-4,
         {1.0109989461, 10., 100.},
         {2.1, 2.6, 3.1},
         {0., 2.9, 9.8}}};
    return table;
  }

} // namespace corsika::gpu::em::tables::testing

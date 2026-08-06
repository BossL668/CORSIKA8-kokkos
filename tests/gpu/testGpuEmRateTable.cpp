/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include <corsika/gpu/em/tables/RateTable.hpp>

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

  using namespace corsika::gpu::em::tables;

  std::size_t checks = 0;

  void require(bool condition, std::string const& message) {
    ++checks;
    if (!condition) {
      throw std::runtime_error(message);
    }
  }

  template <typename Function>
  void requireThrows(Function function, std::string const& message) {
    ++checks;
    try {
      function();
    } catch (std::exception const&) {
      return;
    }
    throw std::runtime_error(message);
  }

  RateTableSet fixture() {
    RateTableSet table;
    table.metadata.proposal_version = "7.6.2";
    table.metadata.generator_version = "test";
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
        2.2e11,
        2.8179403262e-13,
        7.2973525693e-3,
        {{101, 7., 182.7},
         {102, 8., 182.7},
         {103, 18., 184.15}}};
    table.metadata.brems_lpm = {
        1.2048e-3,
        2.504e19,
        7.22,
        2.2e11,
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
    RateColumn photopair{
        1000000013, 101, "Photopair", "PhotoPairKochMotz", "nitrogen",
        {1., 10., 100.}, {}};
    photopair.inverse_cdf.energies_MeV = {1., 10., 100.};
    photopair.inverse_cdf.reference_mode = "proposal_interpolated";
    photopair.inverse_cdf.quantile_offsets = {0, 3, 6, 9};
    photopair.inverse_cdf.quantiles = {
        LossQuantileMinimum, 0.5, LossQuantileMaximum,
        LossQuantileMinimum, 0.5, LossQuantileMaximum,
        LossQuantileMinimum, 0.5, LossQuantileMaximum};
    for (auto const energy : photopair.inverse_cdf.energies_MeV) {
      for (auto const quantile :
           {LossQuantileMinimum, 0.5, LossQuantileMaximum}) {
        auto const logit = std::log(quantile / (1. - quantile));
        photopair.inverse_cdf.v_loss.push_back(
            1.e-2 * std::sqrt(energy) * std::exp(0.01 * logit));
      }
    }

    RateColumn compton{
        1000000010, 102, "Compton", "ComptonKleinNishina", "oxygen",
        {0., 2., 4.}, {}};
    compton.inverse_cdf.energies_MeV = {1., 100.};
    compton.inverse_cdf.reference_mode = "proposal_direct";
    compton.inverse_cdf.quantile_offsets = {0, 2, 4};
    compton.inverse_cdf.quantiles = {
        EpairLossQuantileMinimum, EpairLossQuantileMaximum,
        EpairLossQuantileMinimum, EpairLossQuantileMaximum};
    compton.inverse_cdf.v_loss = {1., 1., 1., 1.};
    photon.columns = {std::move(photopair), std::move(compton)};
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

  std::filesystem::path temporaryPath() {
    auto const stamp = std::chrono::high_resolution_clock::now()
                           .time_since_epoch()
                           .count();
    return std::filesystem::temp_directory_path() /
           ("c8_gpu_em_rate_table_" + std::to_string(stamp) + ".bin");
  }

  void testSha256() {
    std::string const input = "abc";
    auto const digest =
        sha256(reinterpret_cast<std::uint8_t const*>(input.data()), input.size());
    require(toHex(digest) ==
                "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
            "SHA-256 reference vector differs");
  }

  void testInterpolation() {
    auto const table = fixture();
    auto const& photon = findParticle(table, 22);
    auto const geometric_midpoint = std::sqrt(10.);
    auto const rate =
        interpolateRate(photon, 1000000013, 101, geometric_midpoint);
    require(std::abs(rate - geometric_midpoint) < 1.e-13,
            "positive rate is not log-log interpolated");
    require(interpolateRate(photon, 1000000010, 102, 1.) == 0.,
            "stored threshold endpoint changed");
    require(interpolateTotalRate(photon, 10.) == 12.,
            "stored total rate differs");
    auto const loss_quantile = lossQuantileFromPhiloxWord(1073741824u);
    auto const loss =
        interpolateLossFraction(photon, 1000000013, 101,
                                geometric_midpoint, loss_quantile);
    auto const expected_loss =
        1.e-2 * std::sqrt(geometric_midpoint) *
        std::exp(0.01 *
                 std::log(loss_quantile / (1. - loss_quantile)));
    require(std::abs(loss - expected_loss) < 1.e-14,
            "inverse CDF is not log-bilinearly interpolated");
    require(interpolateLossFraction(
                photon, 1000000010, 102, 10.,
                lossQuantileFromPhiloxWord(3006477107u)) == 1.,
            "constant all-energy-loss inverse CDF changed");
    requireThrows(
        [&] { interpolateTotalRate(photon, 0.5); },
        "energy extrapolation was silently accepted");
    requireThrows(
        [&] { interpolateRate(photon, 123, 101, 10.); },
        "missing process/component column was silently accepted");
    requireThrows(
        [&] {
          interpolateLossFraction(photon, 1000000013, 101, 0.5, 0.5);
        },
        "inverse-CDF energy extrapolation was silently accepted");
    requireThrows(
        [&] {
          interpolateLossFraction(photon, 1000000013, 101, 10., 1.);
        },
        "inverse-CDF quantile one was silently accepted");
    requireThrows(
        [&] {
          interpolateLossFraction(
              photon, 1000000010, 102, 10.,
              lossQuantileFromPhiloxWord(0));
        },
        "column-specific inverse-CDF tail fallback was not enforced");

    auto const& continuous =
        findContinuousEnergyTable(table, 11);
    auto const continuous_midpoint =
        std::sqrt(
            continuous.energies_MeV[0] *
            continuous.energies_MeV[1]);
    require(
        std::abs(
            interpolateContinuousDedx(
                continuous, continuous_midpoint) -
            std::sqrt(2. * 2.5)) <
            1.e-14,
        "continuous dE/dX is not log-log interpolated");
    require(
        std::abs(
            interpolateContinuousRange(
                continuous, continuous_midpoint) -
            1.5) <
            1.e-14,
        "continuous range is not linear in log energy");
    require(
        std::abs(
            interpolateContinuousEnergy(continuous, 6.5) -
            std::sqrt(1000.)) <
            1.e-13,
        "continuous inverse range is not log-energy interpolated");
    require(
        std::abs(
            energyAfterContinuousLoss(continuous, 100., 7.) -
            10.) <
            1.e-13,
        "continuous loss did not subtract range");
    requireThrows(
        [&] {
          energyAfterContinuousLoss(continuous, 10., 3.);
        },
        "continuous transport silently crossed its cut");
  }

  void testRoundTripAndCorruption() {
    auto const path = temporaryPath();
    try {
      auto const original = fixture();
      auto const digest = writeRateTable(path, original);
      auto const loaded = readRateTable(path);
      require(loaded.content_hash == digest, "loaded content hash differs");
      require(calculateContentHash(loaded) == digest,
              "re-encoded content hash differs");
      require(loaded.metadata.proposal_version == "7.6.2",
              "metadata did not survive serialization");
      require(
          loaded.metadata.photon_pair_lpm.e_lpm_MeV ==
                  original.metadata.photon_pair_lpm.e_lpm_MeV &&
              loaded.metadata.photon_pair_lpm.components.size() ==
                  original.metadata.photon_pair_lpm.components.size() &&
              loaded.metadata.photon_pair_lpm.components.front()
                      .radiation_log_constant ==
                  original.metadata.photon_pair_lpm.components.front()
                      .radiation_log_constant,
          "photon-pair LPM metadata did not survive serialization");
      require(
          loaded.metadata.brems_lpm.e_lpm_MeV ==
                  original.metadata.brems_lpm.e_lpm_MeV &&
              loaded.metadata.brems_lpm.lepton_mass_MeV ==
                  original.metadata.brems_lpm.lepton_mass_MeV &&
              loaded.metadata.brems_lpm.components.size() ==
                  original.metadata.brems_lpm.components.size() &&
              loaded.metadata.brems_lpm.components.front()
                      .atomic_mass_number ==
                  original.metadata.brems_lpm.components.front()
                      .atomic_mass_number,
          "bremsstrahlung LPM metadata did not survive serialization");
      require(loaded.particles.front().columns.size() == 2,
              "rate columns did not survive serialization");
      require(
          loaded.continuous_energy_tables.size() == 2 &&
              loaded.continuous_energy_tables.front().pdg_id ==
                  11 &&
              loaded.continuous_energy_tables.front()
                      .range_g_per_cm2.back() ==
                  original.continuous_energy_tables.front()
                      .range_g_per_cm2.back(),
          "continuous energy tables did not survive serialization");
      require(loaded.particles.front()
                      .columns.front()
                      .inverse_cdf.reference_mode ==
                  "proposal_interpolated",
              "inverse-CDF reference mode did not survive serialization");

      {
        std::fstream file(path, std::ios::binary | std::ios::in | std::ios::out);
        require(static_cast<bool>(file), "cannot reopen test table for corruption");
        file.seekg(-1, std::ios::end);
        char byte{};
        file.read(&byte, 1);
        byte ^= 0x01;
        file.seekp(-1, std::ios::end);
        file.write(&byte, 1);
      }
      requireThrows([&] { readRateTable(path); },
                    "corrupted rate table passed SHA-256 validation");
    } catch (...) {
      std::error_code error;
      std::filesystem::remove(path, error);
      throw;
    }
    std::filesystem::remove(path);
  }

  void testValidation() {
    {
      auto monotone = fixture();
      monotone.particles.front()
          .columns.front()
          .inverse_cdf.reference_mode =
          "proposal_interpolated_monotone";
      validateRateTable(monotone);
      ++checks;
    }
    {
      auto fallback = fixture();
      auto& inverse = fallback.particles.front()
                          .columns.front()
                          .inverse_cdf;
      inverse = {};
      inverse.reference_mode =
          SelectedLossCpuFallbackReferenceMode;
      validateRateTable(fallback);
      ++checks;

      inverse.energies_MeV = {1., 10.};
      requireThrows(
          [&] { validateRateTable(fallback); },
          "selected-loss fallback accepted residual inverse-CDF arrays");
    }

    auto table = fixture();
    table.metadata.components.front().number_fraction = 0.7;
    requireThrows([&] { validateRateTable(table); },
                  "invalid medium fractions were accepted");

    table = fixture();
    table.particles.front().columns.front().rates_cm2_per_g.pop_back();
    requireThrows([&] { validateRateTable(table); },
                  "mismatched rate-column length was accepted");

    table = fixture();
    table.particles.front().energies_MeV[1] = 0.5;
    requireThrows([&] { validateRateTable(table); },
                  "non-monotonic energy grid was accepted");

    table = fixture();
    table.particles.front().columns.front().inverse_cdf.v_loss.pop_back();
    requireThrows([&] { validateRateTable(table); },
                  "mismatched inverse-CDF dimensions were accepted");

    table = fixture();
    table.particles.front()
        .columns.front()
        .inverse_cdf.v_loss[1] = 0.;
    requireThrows([&] { validateRateTable(table); },
                  "non-monotonic inverse-CDF row was accepted");

    table = fixture();
    table.particles.front()
        .columns.front()
        .inverse_cdf.reference_mode = "unknown";
    requireThrows([&] { validateRateTable(table); },
                  "unknown inverse-CDF reference mode was accepted");

    table = fixture();
    table.particles.front()
        .columns.front()
        .inverse_cdf.quantiles[3] =
        2. * LossQuantileMinimum;
    requireThrows([&] { validateRateTable(table); },
                  "inconsistent ragged-row quantile domains were accepted");

    table = fixture();
    table.metadata.photon_pair_lpm.e_lpm_MeV = 0.;
    requireThrows([&] { validateRateTable(table); },
                  "invalid LPM characteristic energy was accepted");

    table = fixture();
    table.metadata.photon_pair_lpm.components.front().proposal_hash++;
    requireThrows([&] { validateRateTable(table); },
                  "LPM component not present in the medium was accepted");

    table = fixture();
    table.metadata.photon_pair_lpm.components.pop_back();
    requireThrows([&] { validateRateTable(table); },
                  "incomplete LPM component list was accepted");

    table = fixture();
    table.metadata.brems_lpm.e_lpm_MeV = 0.;
    requireThrows([&] { validateRateTable(table); },
                  "invalid bremsstrahlung LPM energy was accepted");

    table = fixture();
    table.metadata.brems_lpm.components.front().atomic_mass_number = 0.;
    requireThrows(
        [&] { validateRateTable(table); },
        "invalid bremsstrahlung LPM target mass was accepted");

    table = fixture();
    table.metadata.brems_lpm.components.pop_back();
    requireThrows(
        [&] { validateRateTable(table); },
        "incomplete bremsstrahlung LPM component list was accepted");

    table = fixture();
    table.continuous_energy_tables.front()
        .range_g_per_cm2[1] = 0.;
    requireThrows(
        [&] { validateRateTable(table); },
        "non-monotonic continuous range was accepted");

    table = fixture();
    table.continuous_energy_tables.pop_back();
    requireThrows(
        [&] { validateRateTable(table); },
        "missing positron continuous table was accepted");

    table = fixture();
    table.continuous_energy_tables.front()
        .measured_max_relative_error = 2.e-3;
    requireThrows(
        [&] { validateRateTable(table); },
        "inaccurate continuous table was accepted");
  }

  void testCompatibility() {
    auto const table = fixture();
    RateTableRequirements requirements{
        "7.6.2", "dry_air", 9001, 0.5, 0.01, 1., 100., 1.e-3, 1.e-3,
        table.metadata.components};
    validateCompatibility(table, requirements);
    ++checks;

    auto wrong_version = requirements;
    wrong_version.proposal_version = "7.6.3";
    requireThrows([&] { validateCompatibility(table, wrong_version); },
                  "wrong PROPOSAL version was accepted");

    auto wrong_cut = requirements;
    wrong_cut.energy_cut_MeV = 5.;
    requireThrows([&] { validateCompatibility(table, wrong_cut); },
                  "wrong energy cut was accepted");

    auto wrong_medium_hash = requirements;
    wrong_medium_hash.proposal_medium_hash++;
    requireThrows(
        [&] { validateCompatibility(table, wrong_medium_hash); },
        "wrong PROPOSAL medium hash was accepted");

    auto insufficient_domain = requirements;
    insufficient_domain.required_energy_max_MeV = 1000.;
    requireThrows([&] { validateCompatibility(table, insufficient_domain); },
                  "insufficient energy domain was accepted");

    auto insufficient_accuracy = requirements;
    insufficient_accuracy.maximum_relative_error = 1.e-4;
    requireThrows([&] { validateCompatibility(table, insufficient_accuracy); },
                  "insufficient interpolation accuracy was accepted");

    auto insufficient_loss_accuracy = requirements;
    insufficient_loss_accuracy.maximum_loss_relative_error = 1.e-4;
    requireThrows(
        [&] { validateCompatibility(table, insufficient_loss_accuracy); },
        "insufficient inverse-CDF interpolation accuracy was accepted");

    auto wrong_composition = requirements;
    wrong_composition.components.front().proposal_hash++;
    requireThrows([&] { validateCompatibility(table, wrong_composition); },
                  "wrong medium composition was accepted");
  }

} // namespace

int main() {
  try {
    testSha256();
    testInterpolation();
    testRoundTripAndCorruption();
    testValidation();
    testCompatibility();
  } catch (std::exception const& error) {
    std::cerr << "GPU EM rate-table validation failed after " << checks
              << " checks: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
  std::cout << "GPU EM rate-table validation passed: " << checks << " checks\n";
  return EXIT_SUCCESS;
}

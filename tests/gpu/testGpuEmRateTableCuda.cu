/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <GpuEmFlatRateTableFixture.hpp>

#include <corsika/gpu/em/tables/CudaRateTable.hpp>

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

  double quantileAt(double fraction, double minimum, double maximum) {
    auto const lower = std::log(minimum / (1. - minimum));
    auto const upper = std::log(maximum / (1. - maximum));
    auto const coordinate = lower + fraction * (upper - lower);
    if (coordinate >= 0.) {
      return 1. / (1. + std::exp(-coordinate));
    }
    auto const exponential = std::exp(coordinate);
    return exponential / (1. + exponential);
  }

  std::vector<TableQuery> makeFixtureQueries() {
    std::vector<TableQuery> queries;
    for (std::size_t i = 0; i < 256; ++i) {
      auto const fraction =
          (static_cast<double>(i) + 0.5) / 256.;
      auto const energy = std::exp(fraction * std::log(100.));
      auto const quantile = quantileAt(
          fraction, LossQuantileMinimum, LossQuantileMaximum);
      queries.push_back(
          {TableQueryKind::Rate, 22, 1000000013, 101,
           energy, 0.});
      queries.push_back(
          {TableQueryKind::Rate, 22, 1000000010, 102,
           energy, 0.});
      queries.push_back(
          {TableQueryKind::TotalRate, 22, 0, 0, energy, 0.});
      queries.push_back(
          {TableQueryKind::LossFraction, 22, 1000000013,
           101, energy, quantile});
      queries.push_back(
          {TableQueryKind::LossFraction, 22, 2000000013,
           101, energy, quantile});
      queries.push_back(
          {TableQueryKind::LossFraction, 22, 1000000010,
           102, energy,
           quantileAt(fraction, EpairLossQuantileMinimum,
                      EpairLossQuantileMaximum)});
      auto const continuous_energy =
          std::exp(std::log(1.0109989461) +
                   fraction *
                       (std::log(100.) -
                        std::log(1.0109989461)));
      for (auto const pdg_id : {11, -11}) {
        queries.push_back(
            {TableQueryKind::ContinuousDedx, pdg_id, 0, 0,
             continuous_energy, 0.});
        queries.push_back(
            {TableQueryKind::ContinuousRange, pdg_id, 0, 0,
             continuous_energy, 0.});
        auto const maximum_range = pdg_id == 11 ? 10. : 9.8;
        queries.push_back(
            {TableQueryKind::ContinuousEnergy, pdg_id, 0, 0,
             fraction * maximum_range, 0.});
        queries.push_back(
            {TableQueryKind::EnergyAfterContinuousLoss, pdg_id, 0, 0,
             continuous_energy, 0.});
      }
    }
    queries.push_back(
        {TableQueryKind::Rate, 11, 1000000013, 101, 10., 0.});
    queries.push_back(
        {TableQueryKind::Rate, 22, 7, 101, 10., 0.});
    queries.push_back(
        {TableQueryKind::Rate, 22, 1000000013, 101, 0.5, 0.});
    queries.push_back(
        {TableQueryKind::LossFraction, 22, 1000000007,
         103, 10., 0.5});
    queries.push_back(
        {TableQueryKind::LossFraction, 22, 1000000013,
         101, 0.5, 0.5});
    queries.push_back(
        {TableQueryKind::LossFraction, 22, 1000000010,
         102, 10., LossQuantileMinimum});
    queries.push_back(
        {TableQueryKind::Rate, 22, 1000000013, 101, 1., 0.});
    queries.push_back(
        {TableQueryKind::Rate, 22, 1000000013, 101, 100., 0.});
    queries.push_back(
        {TableQueryKind::LossFraction, 22, 1000000013,
         101, 1., LossQuantileMinimum});
    queries.push_back(
        {TableQueryKind::LossFraction, 22, 1000000013,
         101, 100., LossQuantileMaximum});
    queries.push_back(
        {TableQueryKind::ContinuousDedx, 22, 0, 0, 10., 0.});
    queries.push_back(
        {TableQueryKind::ContinuousRange, 11, 0, 0, 1., 0.});
    queries.push_back(
        {TableQueryKind::ContinuousEnergy, 11, 0, 0, -1., 0.});
    queries.push_back(
        {TableQueryKind::EnergyAfterContinuousLoss, 11, 0, 0,
         10., 3.});
    TableQuery invalid{};
    invalid.kind = static_cast<TableQueryKind>(99);
    queries.push_back(invalid);
    return queries;
  }

  std::vector<TableQuery> makeSourceQueries(
      RateTableSet const& source) {
    std::vector<TableQuery> queries;
    constexpr std::size_t SamplesPerColumn = 32;
    for (auto const& particle : source.particles) {
      auto const rate_log_min =
          std::log(particle.energies_MeV.front());
      auto const rate_log_max =
          std::log(particle.energies_MeV.back());
      for (std::size_t i = 0; i < SamplesPerColumn; ++i) {
        auto const fraction =
            (static_cast<double>(i) + 0.5) /
            SamplesPerColumn;
        auto const energy = std::exp(
            rate_log_min +
            fraction * (rate_log_max - rate_log_min));
        queries.push_back(
            {TableQueryKind::TotalRate, particle.pdg_id,
             0, 0, energy, 0.});
        for (auto const& column : particle.columns) {
          queries.push_back(
              {TableQueryKind::Rate, particle.pdg_id,
               column.process_id, column.component_hash,
               energy, 0.});
        }
      }
      for (auto const& column : particle.columns) {
        auto const& inverse = column.inverse_cdf;
        if (inverse.energies_MeV.empty()) {
          continue;
        }
        auto const row_end = static_cast<std::size_t>(
            inverse.quantile_offsets[1]);
        auto const minimum_quantile = inverse.quantiles.front();
        auto const maximum_quantile =
            inverse.quantiles[row_end - 1];
        auto const log_min =
            std::log(inverse.energies_MeV.front());
        auto const log_max =
            std::log(inverse.energies_MeV.back());
        for (std::size_t i = 0; i < SamplesPerColumn; ++i) {
          auto const energy_fraction =
              std::fmod(
                  (static_cast<double>(i) + 0.5) *
                      0.6180339887498948482,
                  1.);
          auto const quantile_fraction =
              std::fmod(
                  (static_cast<double>(i) + 0.5) *
                      0.4142135623730950488,
                  1.);
          queries.push_back(
              {TableQueryKind::LossFraction, particle.pdg_id,
               column.process_id, column.component_hash,
               std::exp(log_min +
                        energy_fraction * (log_max - log_min)),
               quantileAt(quantile_fraction, minimum_quantile,
                          maximum_quantile)});
        }
      }
    }
    for (auto const& continuous : source.continuous_energy_tables) {
      auto const log_min =
          std::log(continuous.energies_MeV.front());
      auto const log_max =
          std::log(continuous.energies_MeV.back());
      auto const range_max =
          continuous.range_g_per_cm2.back();
      for (std::size_t i = 0; i < SamplesPerColumn; ++i) {
        auto const fraction =
            (static_cast<double>(i) + 0.5) /
            SamplesPerColumn;
        auto const energy = std::exp(
            log_min + fraction * (log_max - log_min));
        auto const range =
            interpolateContinuousRange(continuous, energy);
        queries.push_back(
            {TableQueryKind::ContinuousDedx, continuous.pdg_id,
             0, 0, energy, 0.});
        queries.push_back(
            {TableQueryKind::ContinuousRange, continuous.pdg_id,
             0, 0, energy, 0.});
        queries.push_back(
            {TableQueryKind::ContinuousEnergy, continuous.pdg_id,
             0, 0, fraction * range_max, 0.});
        queries.push_back(
            {TableQueryKind::EnergyAfterContinuousLoss,
             continuous.pdg_id, 0, 0, energy, 0.25 * range});
      }
    }
    return queries;
  }

  double hierarchicalReference(RateTableSet const& source,
                               TableQuery const& query) {
    if (query.kind == TableQueryKind::Rate) {
      auto const& particle = findParticle(source, query.pdg_id);
      return interpolateRate(
          particle, query.process_id, query.component_hash,
          query.energy_MeV);
    }
    if (query.kind == TableQueryKind::TotalRate) {
      auto const& particle = findParticle(source, query.pdg_id);
      return interpolateTotalRate(particle, query.energy_MeV);
    }
    if (query.kind == TableQueryKind::LossFraction) {
      auto const& particle = findParticle(source, query.pdg_id);
      return interpolateLossFraction(
          particle, query.process_id, query.component_hash,
          query.energy_MeV, query.quantile);
    }
    auto const& continuous =
        findContinuousEnergyTable(source, query.pdg_id);
    if (query.kind == TableQueryKind::ContinuousDedx) {
      return interpolateContinuousDedx(
          continuous, query.energy_MeV);
    }
    if (query.kind == TableQueryKind::ContinuousRange) {
      return interpolateContinuousRange(
          continuous, query.energy_MeV);
    }
    if (query.kind == TableQueryKind::ContinuousEnergy) {
      return interpolateContinuousEnergy(
          continuous, query.energy_MeV);
    }
    if (query.kind ==
        TableQueryKind::EnergyAfterContinuousLoss) {
      return energyAfterContinuousLoss(
          continuous, query.energy_MeV, query.quantile);
    }
    throw std::invalid_argument(
        "cannot evaluate an invalid hierarchical query");
  }

} // namespace

int main(int argc, char** argv) {
  int device_count = 0;
  auto const cuda_status = cudaGetDeviceCount(&device_count);
  if (cuda_status != cudaSuccess || device_count == 0) {
    std::cerr << "SKIP: no CUDA device is accessible: "
              << cudaGetErrorString(cuda_status) << '\n';
    return 77;
  }

  try {
    if (argc > 2) {
      std::cerr << "usage: testGpuEmRateTableCuda [RATE_TABLE]\n";
      return 2;
    }
    auto const source =
        argc == 2 ? readRateTable(argv[1])
                  : testing::makeFlatRateTableFixture();
    auto const flat = flattenRateTable(source);
    auto const host_view = makeFlatRateTableView(flat);
    auto const queries =
        argc == 2 ? makeSourceQueries(source)
                  : makeFixtureQueries();

    CudaRateTable device_table;
    require(!device_table.initialized(),
            "new CUDA rate table is already initialized");
    requireThrows([&] { device_table.deviceView(); },
                  "uninitialized device view was returned");
    requireThrows(
        [&] {
          CudaRateTable too_small;
          too_small.initialize(
              source, 0, flatRateTableBytes(flat) - 1);
        },
        "CUDA table ignored its memory budget");

    device_table.initialize(source, 0);
    require(device_table.initialized(),
            "CUDA rate table did not initialize");
    require(device_table.deviceBytes() == flatRateTableBytes(flat),
            "CUDA rate-table byte count differs");
    require(device_table.sourceContentHash() == flat.content_hash,
            "CUDA rate-table source hash differs");

    std::vector<TableQueryResult> expected;
    expected.reserve(queries.size());
    for (auto const& query : queries) {
      auto const flat_result = executeTableQuery(host_view, query);
      if (flat_result.status == TableLookupStatus::Success) {
        auto const hierarchical =
            hierarchicalReference(source, query);
        auto const scale =
            std::max({1., std::abs(hierarchical),
                      std::abs(flat_result.value)});
        require(std::abs(flat_result.value - hierarchical) <=
                    2.e-14 * scale,
                "flat/hierarchical host rate-table value differs");
      }
      expected.push_back(flat_result);
    }
    auto const first = device_table.queryForValidation(queries);
    auto const second = device_table.queryForValidation(queries);
    require(first.size() == expected.size() &&
                second.size() == expected.size(),
            "CUDA rate-table result count differs");
    for (std::size_t i = 0; i < expected.size(); ++i) {
      require(first[i].status == expected[i].status,
              "CUDA/CPU rate-table status differs");
      require(second[i].status == first[i].status,
              "repeated CUDA rate-table status differs");
      if (expected[i].status == TableLookupStatus::Success) {
        auto const scale =
            std::max({1., std::abs(expected[i].value),
                      std::abs(first[i].value)});
        require(std::abs(first[i].value - expected[i].value) <=
                    2.e-13 * scale,
                "CUDA/CPU rate-table value differs");
      }
      require(second[i].value == first[i].value,
              "repeated CUDA rate-table value is not deterministic");
    }

    auto const empty =
        device_table.queryForValidation({});
    require(empty.empty(), "empty CUDA query returned values");
    device_table.reset();
    require(!device_table.initialized() &&
                device_table.deviceBytes() == 0,
            "CUDA rate-table reset is incomplete");

    std::cout << "CUDA rate-table validation passed "
              << checks << " checks for " << queries.size()
              << " device queries, " << source.particles.size()
              << " particles (" << flatRateTableBytes(flat)
              << " device bytes)\n";
  } catch (std::exception const& error) {
    std::cerr << "CUDA rate-table validation failed after "
              << checks << " checks: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}

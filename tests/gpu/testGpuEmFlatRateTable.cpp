/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

#include <GpuEmFlatRateTableFixture.hpp>

#include <corsika/gpu/em/ProposalFallback.hpp>
#include <corsika/gpu/em/tables/FlatRateTable.hpp>

namespace {

  using namespace corsika::gpu::em;
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

  void requireClose(double actual, double expected,
                    std::string const& message) {
    auto const scale = std::max({1., std::abs(actual),
                                 std::abs(expected)});
    require(std::abs(actual - expected) <= 2.e-14 * scale, message);
  }

  void testLayout() {
    auto const source = testing::makeFlatRateTableFixture();
    auto const flat = flattenRateTable(source);
    validateFlatRateTable(flat);
    auto const view = makeFlatRateTableView(flat);

    require(view.particle_count == 1, "flat particle count differs");
    require(view.column_count == 5, "flat column count differs");
    require(view.rate_energy_count == 3,
            "flat rate-energy count differs");
    require(view.rate_value_count == 15,
            "flat rate-value count differs");
    require(view.inverse_energy_count == 10,
            "flat inverse-energy count differs");
    require(view.inverse_row_offset_count == 14,
            "flat inverse-row count differs");
    require(view.inverse_value_count == 28,
            "flat inverse-value count differs");
    require(view.continuous_particle_count == 2,
            "flat continuous-particle count differs");
    require(view.continuous_value_count == 6,
            "flat continuous-value count differs");
    require(flat.content_hash == calculateContentHash(source),
            "flat source content hash differs");
    require(flatRateTableBytes(flat) > 0,
            "flat table reports an empty device image");

    auto split_cut_source = source;
    split_cut_source.metadata.energy_cut_MeV = 0.4;
    for (auto& continuous :
         split_cut_source.continuous_energy_tables) {
      continuous.minimum_total_energy_MeV =
          continuous.mass_MeV +
          ContinuousCutSafetyFactor * 0.5;
      continuous.energies_MeV.front() =
          continuous.minimum_total_energy_MeV;
    }
    auto const split_cut_view = makeFlatRateTableView(
        flattenRateTable(split_cut_source));
    requireClose(
        split_cut_view.energy_cut_MeV, 0.4,
        "flat view changed the scalar PROPOSAL stochastic cut");
    requireClose(
        split_cut_view.em_transport_cut_MeV, 0.5,
        "flat view did not preserve a transport cut distinct from the "
        "PROPOSAL stochastic cut");

    auto malformed = flat;
    malformed.column_rate_offsets.pop_back();
    requireThrows([&] { validateFlatRateTable(malformed); },
                  "malformed flat metadata was accepted");
    malformed.inverse_row_offsets[1] =
        malformed.inverse_row_offsets[0];
    requireThrows([&] { validateFlatRateTable(malformed); },
                  "empty flat inverse-CDF row was accepted");
    malformed = flat;
    malformed.continuous_ranges_g_per_cm2[4] =
        malformed.continuous_ranges_g_per_cm2[3];
    requireThrows([&] { validateFlatRateTable(malformed); },
                  "non-increasing flat continuous range was accepted");
  }

  void testHostQueries() {
    auto const source = testing::makeFlatRateTableFixture();
    auto const flat = flattenRateTable(source);
    auto const view = makeFlatRateTableView(flat);
    auto const& photon = findParticle(source, 22);
    auto const particle_index = detail::findParticle(view, 22);
    require(particle_index != view.particle_count,
            "flat photon particle index is missing");

    for (auto const energy :
         {1., std::sqrt(10.), 10., std::sqrt(1000.), 100.}) {
      auto const bracket = detail::rateInterpolationBracket(
          view, particle_index, energy);
      require(bracket.status == TableLookupStatus::Success,
              "shared flat rate interpolation bracket failed");
      auto const column_begin =
          view.particle_column_offsets[particle_index];
      auto const column_end =
          column_begin +
          view.particle_column_counts[particle_index];
      for (auto column = column_begin; column < column_end;
           ++column) {
        auto const direct = detail::interpolateRateColumn(
            view, particle_index, column, energy);
        auto const shared =
            detail::interpolateRateColumnAtBracket(
                view, particle_index, column, bracket);
        require(direct.status == shared.status,
                "shared rate bracket changed a column status");
        requireClose(
            shared.value, direct.value,
            "shared rate bracket changed a column value");
      }

      auto const pair = queryRate(
          view, 22, 1000000013, 101, energy);
      require(pair.status == TableLookupStatus::Success,
              "flat pair-rate query failed");
      requireClose(
          pair.value,
          interpolateRate(photon, 1000000013, 101, energy),
          "flat pair-rate interpolation differs");

      auto const compton = queryRate(
          view, 22, 1000000010, 102, energy);
      require(compton.status == TableLookupStatus::Success,
              "flat Compton-rate query failed");
      requireClose(
          compton.value,
          interpolateRate(photon, 1000000010, 102, energy),
          "flat threshold-rate interpolation differs");

      auto const total = queryTotalRate(view, 22, energy);
      require(total.status == TableLookupStatus::Success,
              "flat total-rate query failed");
      requireClose(total.value, interpolateTotalRate(photon, energy),
                   "flat total-rate interpolation differs");
    }

    require(
        detail::rateInterpolationBracket(
            view, particle_index, 0.5)
                .status ==
            TableLookupStatus::RateEnergyOutOfRange,
        "shared rate bracket accepted an out-of-range energy");
    require(
        detail::rateInterpolationBracket(
            view, particle_index,
            std::numeric_limits<double>::quiet_NaN())
                .status == TableLookupStatus::NonFiniteInput,
        "shared rate bracket accepted a non-finite energy");

    for (auto const energy : {1., std::sqrt(10.), 10., 50., 100.}) {
      for (auto const quantile :
           {LossQuantileMinimum, 0.1, 0.5, 0.9,
            LossQuantileMaximum}) {
        auto const loss = queryLossFraction(
            view, 22, 1000000013, 101, energy, quantile);
        require(loss.status == TableLookupStatus::Success,
                "flat inverse-CDF query failed");
        requireClose(
            loss.value,
            interpolateLossFraction(
                photon, 1000000013, 101, energy, quantile),
            "flat inverse-CDF interpolation differs");
        if (energy >= 10.) {
          auto const split = queryLossFraction(
              view, 22, 2000000013, 101, energy, quantile);
          require(split.status == TableLookupStatus::Success,
                  "flat final-state inverse-CDF query failed");
          requireClose(
              split.value,
              interpolateLossFraction(
                  photon, 2000000013, 101, energy, quantile),
              "flat final-state inverse-CDF interpolation differs");
        }
      }
    }
    require(
        queryLossFraction(
            view, 22, 2000000013, 101, 5., 0.5)
                .status ==
            TableLookupStatus::LossEnergyOutOfRange,
        "below-table pair final-state query was not rejected");

    for (auto const pdg_id : {11, -11}) {
      auto const& continuous =
          findContinuousEnergyTable(source, pdg_id);
      for (auto const energy :
           {continuous.energies_MeV.front(),
            std::sqrt(continuous.energies_MeV.front() * 10.),
            10., std::sqrt(1000.), 100.}) {
        auto const dedx =
            queryContinuousDedx(view, pdg_id, energy);
        require(dedx.status == TableLookupStatus::Success,
                "flat continuous-dEdX query failed");
        requireClose(
            dedx.value,
            interpolateContinuousDedx(continuous, energy),
            "flat continuous-dEdX interpolation differs");
        auto const range =
            queryContinuousRange(view, pdg_id, energy);
        require(range.status == TableLookupStatus::Success,
                "flat continuous-range query failed");
        requireClose(
            range.value,
            interpolateContinuousRange(continuous, energy),
            "flat continuous-range interpolation differs");
        auto const inverse =
            queryContinuousEnergy(view, pdg_id, range.value);
        require(inverse.status == TableLookupStatus::Success,
                "flat inverse continuous-range query failed");
        requireClose(
            inverse.value,
            interpolateContinuousEnergy(continuous, range.value),
            "flat inverse continuous-range interpolation differs");
        auto const grammage = range.value * 0.25;
        auto const after = queryEnergyAfterContinuousLoss(
            view, pdg_id, energy, grammage);
        require(after.status == TableLookupStatus::Success,
                "flat continuous-loss query failed");
        requireClose(
            after.value,
            energyAfterContinuousLoss(
                continuous, energy, grammage),
            "flat continuous-loss interpolation differs");
      }
    }
  }

  void testFallbackStatuses() {
    auto const flat =
        flattenRateTable(testing::makeFlatRateTableFixture());
    auto const view = makeFlatRateTableView(flat);

    require(queryRate(view, 11, 1000000013, 101, 10.).status ==
                TableLookupStatus::ParticleNotFound,
            "missing particle was not reported");
    require(queryRate(view, 22, 123, 101, 10.).status ==
                TableLookupStatus::ColumnNotFound,
            "missing column was not reported");
    require(queryRate(view, 22, 1000000013, 101, 0.5).status ==
                TableLookupStatus::RateEnergyOutOfRange,
            "rate energy fallback was not reported");
    require(queryLossFraction(
                view, 22, 1000000007, 103, 10., 0.5)
                .status ==
                TableLookupStatus::InverseCdfUnavailable,
            "missing inverse CDF was not reported");
    require(queryLossFraction(
                view, 22, 1000000013, 101, 0.5, 0.5)
                .status ==
                TableLookupStatus::LossEnergyOutOfRange,
            "loss energy fallback was not reported");
    require(queryLossFraction(
                view, 22, 1000000010, 102, 10.,
                LossQuantileMinimum)
                .status ==
                TableLookupStatus::LossQuantileOutOfRange,
            "column quantile fallback was not reported");
    require(queryRate(
                view, 22, 1000000013, 101,
                std::numeric_limits<double>::quiet_NaN())
                .status ==
                TableLookupStatus::NonFiniteInput,
            "non-finite query was not rejected");
    require(queryContinuousDedx(view, 22, 10.).status ==
                TableLookupStatus::ContinuousParticleNotFound,
            "missing continuous particle was not reported");
    require(queryContinuousRange(view, 11, 1.).status ==
                TableLookupStatus::ContinuousEnergyOutOfRange,
            "continuous energy fallback was not reported");
    require(queryContinuousEnergy(view, 11, -1.).status ==
                TableLookupStatus::ContinuousRangeOutOfRange,
            "continuous range fallback was not reported");
    require(queryEnergyAfterContinuousLoss(view, 11, 10., 3.).status ==
                TableLookupStatus::TransportCutReached,
            "continuous transport cut was not reported");
    require(queryEnergyAfterContinuousLoss(
                view, 11, 10.,
                std::numeric_limits<double>::quiet_NaN())
                .status == TableLookupStatus::NonFiniteInput,
            "non-finite continuous grammage was not rejected");

    TableQuery invalid{};
    invalid.kind = static_cast<TableQueryKind>(99);
    require(executeTableQuery(view, invalid).status ==
                TableLookupStatus::InvalidQueryKind,
            "invalid query kind was not rejected");

    EmParticleState particle{};
    particle.pid = 22;
    particle.energy_GeV = 1.;
    particle.history_id = 17;
    TableQuery query{TableQueryKind::LossFraction, 22,
                     1000000010, 102, 10.,
                     LossQuantileMinimum};
    auto const result = executeTableQuery(view, query);
    auto const fallback =
        makeTableFallbackEvent(particle, query, result, 9);
    require(fallback.reason ==
                ProposalFallbackReason::LossQuantileOutOfRange,
            "table status was mapped to the wrong fallback reason");
    require(fallback.process_id == query.process_id &&
                fallback.component_hash == query.component_hash,
            "fallback lost the selected process/component");
    require(fallback.particle.history_id == 17 &&
                fallback.random_draw_id == 9,
            "fallback lost its particle or random draw identity");
  }

} // namespace

int main() {
  try {
    testLayout();
    testHostQueries();
    testFallbackStatuses();
  } catch (std::exception const& error) {
    std::cerr << "GPU EM flat rate-table validation failed after "
              << checks << " checks: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
  std::cout << "GPU EM flat rate-table validation passed: "
            << checks << " checks\n";
  return EXIT_SUCCESS;
}

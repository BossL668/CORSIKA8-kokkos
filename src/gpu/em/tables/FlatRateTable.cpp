/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include <corsika/gpu/em/tables/FlatRateTable.hpp>

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>

namespace corsika::gpu::em::tables {

  namespace {

    template <typename Container>
    std::uint32_t checkedSize(Container const& values, char const* name) {
      if (values.size() >
          static_cast<std::size_t>(
              std::numeric_limits<std::uint32_t>::max())) {
        throw std::length_error(
            std::string("flat rate-table ") + name +
            " exceeds the 32-bit device index limit");
      }
      return static_cast<std::uint32_t>(values.size());
    }

    std::uint32_t checkedAdd(std::uint32_t left, std::uint32_t right,
                             char const* name) {
      if (right > std::numeric_limits<std::uint32_t>::max() - left) {
        throw std::length_error(
            std::string("flat rate-table ") + name +
            " offset exceeds the 32-bit device index limit");
      }
      return left + right;
    }

    template <typename T>
    std::size_t bytes(std::vector<T> const& values) {
      return values.size() * sizeof(T);
    }

    bool strictlyIncreasing(double const* values, std::uint32_t count) {
      for (std::uint32_t i = 1; i < count; ++i) {
        if (!(values[i] > values[i - 1])) {
          return false;
        }
      }
      return true;
    }

  } // namespace

  FlatRateTable flattenRateTable(RateTableSet const& source) {
    validateRateTable(source);
    FlatRateTable flat;
    flat.content_hash = calculateContentHash(source);
    flat.energy_cut_MeV = source.metadata.energy_cut_MeV;

    for (auto const& particle : source.particles) {
      flat.particle_pdg_ids.push_back(particle.pdg_id);
      flat.particle_energy_offsets.push_back(
          checkedSize(flat.rate_energies_MeV, "rate-energy"));
      flat.particle_energy_counts.push_back(
          checkedSize(particle.energies_MeV, "particle energy"));
      flat.particle_column_offsets.push_back(
          checkedSize(flat.column_process_ids, "column"));
      flat.particle_column_counts.push_back(
          checkedSize(particle.columns, "particle column"));
      flat.rate_energies_MeV.insert(
          flat.rate_energies_MeV.end(), particle.energies_MeV.begin(),
          particle.energies_MeV.end());

      for (auto const& column : particle.columns) {
        flat.column_process_ids.push_back(column.process_id);
        flat.column_component_hashes.push_back(column.component_hash);
        flat.column_rate_offsets.push_back(
            checkedSize(flat.rates_cm2_per_g, "rate"));
        flat.rates_cm2_per_g.insert(
            flat.rates_cm2_per_g.end(), column.rates_cm2_per_g.begin(),
            column.rates_cm2_per_g.end());

        auto const& inverse = column.inverse_cdf;
        flat.column_inverse_energy_offsets.push_back(
            checkedSize(flat.inverse_energies_MeV, "inverse energy"));
        flat.column_inverse_energy_counts.push_back(
            checkedSize(inverse.energies_MeV, "column inverse energy"));
        flat.column_inverse_row_offsets.push_back(
            checkedSize(flat.inverse_row_offsets, "inverse row-offset"));
        if (inverse.energies_MeV.empty()) {
          continue;
        }

        flat.inverse_energies_MeV.insert(
            flat.inverse_energies_MeV.end(),
            inverse.energies_MeV.begin(), inverse.energies_MeV.end());
        auto const value_base =
            checkedSize(flat.inverse_quantiles, "inverse value");
        for (auto const local_offset : inverse.quantile_offsets) {
          if (local_offset >
              std::numeric_limits<std::uint32_t>::max()) {
            throw std::length_error(
                "inverse-CDF row offset exceeds the 32-bit device index limit");
          }
          flat.inverse_row_offsets.push_back(
              checkedAdd(value_base,
                         static_cast<std::uint32_t>(local_offset),
                         "inverse value"));
        }
        flat.inverse_quantiles.insert(
            flat.inverse_quantiles.end(), inverse.quantiles.begin(),
            inverse.quantiles.end());
        flat.inverse_v_loss.insert(
            flat.inverse_v_loss.end(), inverse.v_loss.begin(),
            inverse.v_loss.end());
      }
    }
    for (auto const& continuous : source.continuous_energy_tables) {
      flat.continuous_pdg_ids.push_back(continuous.pdg_id);
      flat.continuous_energy_offsets.push_back(
          checkedSize(flat.continuous_energies_MeV,
                      "continuous energy"));
      flat.continuous_energy_counts.push_back(
          checkedSize(continuous.energies_MeV,
                      "continuous particle energy"));
      flat.continuous_masses_MeV.push_back(continuous.mass_MeV);
      flat.continuous_minimum_energies_MeV.push_back(
          continuous.minimum_total_energy_MeV);
      flat.continuous_energies_MeV.insert(
          flat.continuous_energies_MeV.end(),
          continuous.energies_MeV.begin(),
          continuous.energies_MeV.end());
      flat.continuous_dEdX_MeV_cm2_per_g.insert(
          flat.continuous_dEdX_MeV_cm2_per_g.end(),
          continuous.dEdX_MeV_cm2_per_g.begin(),
          continuous.dEdX_MeV_cm2_per_g.end());
      flat.continuous_ranges_g_per_cm2.insert(
          flat.continuous_ranges_g_per_cm2.end(),
          continuous.range_g_per_cm2.begin(),
          continuous.range_g_per_cm2.end());
    }
    flat.epair_rho_component_hashes =
        source.epair_rho.component_hashes;
    flat.epair_rho_energies_MeV =
        source.epair_rho.energies_MeV;
    flat.epair_rho_v_coordinates =
        source.epair_rho.v_coordinates;
    flat.epair_rho_quantiles =
        source.epair_rho.rho_quantiles;
    flat.epair_rho_values =
        source.epair_rho.normalized_rho;
    validateFlatRateTable(flat);
    return flat;
  }

  void validateFlatRateTable(FlatRateTable const& table) {
    if (!std::isfinite(table.energy_cut_MeV) ||
        !(table.energy_cut_MeV > 0.)) {
      throw std::invalid_argument(
          "flat rate-table particle cut is invalid");
    }
    auto const particle_count = table.particle_pdg_ids.size();
    if (particle_count == 0 ||
        table.particle_energy_offsets.size() != particle_count ||
        table.particle_energy_counts.size() != particle_count ||
        table.particle_column_offsets.size() != particle_count ||
        table.particle_column_counts.size() != particle_count) {
      throw std::invalid_argument(
          "flat rate-table particle metadata dimensions differ");
    }
    auto const column_count = table.column_process_ids.size();
    if (column_count == 0 ||
        table.column_component_hashes.size() != column_count ||
        table.column_rate_offsets.size() != column_count ||
        table.column_inverse_energy_offsets.size() != column_count ||
        table.column_inverse_energy_counts.size() != column_count ||
        table.column_inverse_row_offsets.size() != column_count) {
      throw std::invalid_argument(
          "flat rate-table column metadata dimensions differ");
    }
    if (table.inverse_quantiles.size() !=
        table.inverse_v_loss.size()) {
      throw std::invalid_argument(
          "flat rate-table inverse-CDF value dimensions differ");
    }
    checkedSize(table.particle_pdg_ids, "particle");
    checkedSize(table.column_process_ids, "column");
    checkedSize(table.rate_energies_MeV, "rate-energy");
    checkedSize(table.rates_cm2_per_g, "rate");
    checkedSize(table.inverse_energies_MeV, "inverse energy");
    checkedSize(table.inverse_row_offsets, "inverse row-offset");
    checkedSize(table.inverse_quantiles, "inverse value");
    auto const continuous_count = table.continuous_pdg_ids.size();
    if (continuous_count == 0 ||
        table.continuous_energy_offsets.size() != continuous_count ||
        table.continuous_energy_counts.size() != continuous_count ||
        table.continuous_masses_MeV.size() != continuous_count ||
        table.continuous_minimum_energies_MeV.size() !=
            continuous_count) {
      throw std::invalid_argument(
          "flat rate-table continuous metadata dimensions differ");
    }
    if (table.continuous_energies_MeV.empty() ||
        table.continuous_dEdX_MeV_cm2_per_g.size() !=
            table.continuous_energies_MeV.size() ||
        table.continuous_ranges_g_per_cm2.size() !=
            table.continuous_energies_MeV.size()) {
      throw std::invalid_argument(
          "flat rate-table continuous value dimensions differ");
    }
    checkedSize(table.continuous_pdg_ids, "continuous particle");
    checkedSize(table.continuous_energies_MeV, "continuous value");
    checkedSize(
        table.epair_rho_component_hashes,
        "Epair rho component");
    checkedSize(
        table.epair_rho_energies_MeV,
        "Epair rho energy");
    checkedSize(
        table.epair_rho_v_coordinates,
        "Epair rho v coordinate");
    checkedSize(
        table.epair_rho_quantiles,
        "Epair rho quantile");
    checkedSize(
        table.epair_rho_values,
        "Epair rho value");

    auto const epair_component_count =
        table.epair_rho_component_hashes.size();
    if (epair_component_count == 0) {
      if (!table.epair_rho_energies_MeV.empty() ||
          !table.epair_rho_v_coordinates.empty() ||
          !table.epair_rho_quantiles.empty() ||
          !table.epair_rho_values.empty()) {
        throw std::invalid_argument(
            "flat Epair rho-table dimensions differ");
      }
    } else {
      if (table.epair_rho_energies_MeV.size() < 2 ||
          table.epair_rho_v_coordinates.size() < 2 ||
          table.epair_rho_quantiles.size() < 2 ||
          !strictlyIncreasing(
              table.epair_rho_energies_MeV.data(),
              checkedSize(
                  table.epair_rho_energies_MeV,
                  "Epair rho energy")) ||
          !strictlyIncreasing(
              table.epair_rho_v_coordinates.data(),
              checkedSize(
                  table.epair_rho_v_coordinates,
                  "Epair rho v coordinate")) ||
          !strictlyIncreasing(
              table.epair_rho_quantiles.data(),
              checkedSize(
                  table.epair_rho_quantiles,
                  "Epair rho quantile"))) {
        throw std::invalid_argument(
            "flat Epair rho-table axes are malformed");
      }
      auto const expected =
          static_cast<std::uint64_t>(epair_component_count) *
          table.epair_rho_energies_MeV.size() *
          table.epair_rho_v_coordinates.size() *
          table.epair_rho_quantiles.size();
      if (expected != table.epair_rho_values.size()) {
        throw std::invalid_argument(
            "flat Epair rho-table value count differs");
      }
      auto const rho_count =
          table.epair_rho_quantiles.size();
      for (std::size_t row = 0;
           row < table.epair_rho_values.size() / rho_count;
           ++row) {
        double previous = -1.;
        for (std::size_t index = 0;
             index < rho_count; ++index) {
          auto const value =
              table.epair_rho_values[row * rho_count + index];
          if (!std::isfinite(value) || value < 0. ||
              value > 1. ||
              value + 1.e-14 < previous) {
            throw std::invalid_argument(
                "flat Epair rho-table row is malformed");
          }
          previous = value;
        }
      }
    }

    for (std::size_t i = 0; i < particle_count; ++i) {
      auto const energy_offset = table.particle_energy_offsets[i];
      auto const energy_count = table.particle_energy_counts[i];
      auto const column_offset = table.particle_column_offsets[i];
      auto const particle_columns = table.particle_column_counts[i];
      if (energy_count < 2 ||
          static_cast<std::size_t>(energy_offset) + energy_count >
              table.rate_energies_MeV.size() ||
          static_cast<std::size_t>(column_offset) + particle_columns >
              column_count ||
          !strictlyIncreasing(table.rate_energies_MeV.data() +
                                  energy_offset,
                              energy_count)) {
        throw std::invalid_argument(
            "flat rate-table particle range is malformed");
      }
      for (auto column = column_offset;
           column < column_offset + particle_columns; ++column) {
        if (static_cast<std::size_t>(
                table.column_rate_offsets[column]) +
                energy_count >
            table.rates_cm2_per_g.size()) {
          throw std::invalid_argument(
              "flat rate-table column rate range is malformed");
        }
      }
    }

    for (std::size_t column = 0; column < column_count; ++column) {
      auto const energy_offset =
          table.column_inverse_energy_offsets[column];
      auto const energy_count =
          table.column_inverse_energy_counts[column];
      auto const row_offset =
          table.column_inverse_row_offsets[column];
      if (energy_count == 0) {
        continue;
      }
      if (energy_count < 2 ||
          static_cast<std::size_t>(energy_offset) + energy_count >
              table.inverse_energies_MeV.size() ||
          static_cast<std::size_t>(row_offset) + energy_count + 1 >
              table.inverse_row_offsets.size() ||
          !strictlyIncreasing(table.inverse_energies_MeV.data() +
                                  energy_offset,
                              energy_count)) {
        throw std::invalid_argument(
            "flat rate-table inverse-energy range is malformed");
      }
      auto const first_begin = table.inverse_row_offsets[row_offset];
      auto const first_end = table.inverse_row_offsets[row_offset + 1];
      if (first_begin >= first_end ||
          first_end > table.inverse_quantiles.size()) {
        throw std::invalid_argument(
            "flat rate-table first inverse-CDF row is malformed");
      }
      auto const minimum_quantile =
          table.inverse_quantiles[first_begin];
      auto const maximum_quantile =
          table.inverse_quantiles[first_end - 1];
      for (std::uint32_t row = 0; row < energy_count; ++row) {
        auto const begin = table.inverse_row_offsets[row_offset + row];
        auto const end = table.inverse_row_offsets[row_offset + row + 1];
        if (begin >= end || end > table.inverse_quantiles.size() ||
            !strictlyIncreasing(
                table.inverse_quantiles.data() + begin, end - begin) ||
            table.inverse_quantiles[begin] != minimum_quantile ||
            table.inverse_quantiles[end - 1] != maximum_quantile) {
          throw std::invalid_argument(
              "flat rate-table inverse-CDF row is malformed");
        }
        for (auto index = begin; index < end; ++index) {
          auto const loss = table.inverse_v_loss[index];
          if (!std::isfinite(loss) || loss < 0. || loss > 1. ||
              (index > begin &&
               loss + 1.e-14 <
                   table.inverse_v_loss[index - 1])) {
            throw std::invalid_argument(
                "flat rate-table inverse-CDF loss row is malformed");
          }
        }
      }
    }

    for (std::size_t particle = 0;
         particle < continuous_count; ++particle) {
      auto const offset = table.continuous_energy_offsets[particle];
      auto const count = table.continuous_energy_counts[particle];
      if (count < 2 ||
          static_cast<std::size_t>(offset) + count >
              table.continuous_energies_MeV.size() ||
          !std::isfinite(table.continuous_masses_MeV[particle]) ||
          table.continuous_masses_MeV[particle] <= 0. ||
          !std::isfinite(
              table.continuous_minimum_energies_MeV[particle]) ||
          table.continuous_minimum_energies_MeV[particle] <=
              table.continuous_masses_MeV[particle] ||
          !strictlyIncreasing(
              table.continuous_energies_MeV.data() + offset, count) ||
          !strictlyIncreasing(
              table.continuous_ranges_g_per_cm2.data() + offset,
              count)) {
        throw std::invalid_argument(
            "flat rate-table continuous particle range is malformed");
      }
      if (table.continuous_energies_MeV[offset] !=
              table.continuous_minimum_energies_MeV[particle] ||
          table.continuous_ranges_g_per_cm2[offset] != 0.) {
        throw std::invalid_argument(
            "flat rate-table continuous transport-cut anchor differs");
      }
      for (std::uint32_t index = 0; index < count; ++index) {
        auto const energy =
            table.continuous_energies_MeV[offset + index];
        auto const dedx =
            table.continuous_dEdX_MeV_cm2_per_g[offset + index];
        auto const range =
            table.continuous_ranges_g_per_cm2[offset + index];
        if (!std::isfinite(energy) || energy <= 0. ||
            !std::isfinite(dedx) || dedx <= 0. ||
            !std::isfinite(range) || range < 0.) {
          throw std::invalid_argument(
              "flat rate-table continuous value is malformed");
        }
      }
    }
  }

  FlatRateTableView makeFlatRateTableView(FlatRateTable const& table) {
    validateFlatRateTable(table);
    return {
        table.particle_pdg_ids.data(),
        table.particle_energy_offsets.data(),
        table.particle_energy_counts.data(),
        table.particle_column_offsets.data(),
        table.particle_column_counts.data(),
        checkedSize(table.particle_pdg_ids, "particle"),
        table.column_process_ids.data(),
        table.column_component_hashes.data(),
        table.column_rate_offsets.data(),
        table.column_inverse_energy_offsets.data(),
        table.column_inverse_energy_counts.data(),
        table.column_inverse_row_offsets.data(),
        checkedSize(table.column_process_ids, "column"),
        table.rate_energies_MeV.data(),
        table.rates_cm2_per_g.data(),
        table.inverse_energies_MeV.data(),
        table.inverse_row_offsets.data(),
        table.inverse_quantiles.data(),
        table.inverse_v_loss.data(),
        checkedSize(table.rate_energies_MeV, "rate-energy"),
        checkedSize(table.rates_cm2_per_g, "rate"),
        checkedSize(table.inverse_energies_MeV, "inverse energy"),
        checkedSize(table.inverse_row_offsets, "inverse row-offset"),
        checkedSize(table.inverse_quantiles, "inverse value"),
        table.continuous_pdg_ids.data(),
        table.continuous_energy_offsets.data(),
        table.continuous_energy_counts.data(),
        table.continuous_masses_MeV.data(),
        table.continuous_minimum_energies_MeV.data(),
        checkedSize(table.continuous_pdg_ids, "continuous particle"),
        table.continuous_energies_MeV.data(),
        table.continuous_dEdX_MeV_cm2_per_g.data(),
        table.continuous_ranges_g_per_cm2.data(),
        checkedSize(table.continuous_energies_MeV,
                    "continuous value"),
        table.energy_cut_MeV,
        table.epair_rho_component_hashes.data(),
        table.epair_rho_energies_MeV.data(),
        table.epair_rho_v_coordinates.data(),
        table.epair_rho_quantiles.data(),
        table.epair_rho_values.data(),
        checkedSize(
            table.epair_rho_component_hashes,
            "Epair rho component"),
        checkedSize(
            table.epair_rho_energies_MeV,
            "Epair rho energy"),
        checkedSize(
            table.epair_rho_v_coordinates,
            "Epair rho v coordinate"),
        checkedSize(
            table.epair_rho_quantiles,
            "Epair rho quantile"),
        checkedSize(
            table.epair_rho_values,
            "Epair rho value")};
  }

  std::size_t flatRateTableBytes(FlatRateTable const& table) {
    validateFlatRateTable(table);
    return bytes(table.particle_pdg_ids) +
           bytes(table.particle_energy_offsets) +
           bytes(table.particle_energy_counts) +
           bytes(table.particle_column_offsets) +
           bytes(table.particle_column_counts) +
           bytes(table.column_process_ids) +
           bytes(table.column_component_hashes) +
           bytes(table.column_rate_offsets) +
           bytes(table.column_inverse_energy_offsets) +
           bytes(table.column_inverse_energy_counts) +
           bytes(table.column_inverse_row_offsets) +
           bytes(table.rate_energies_MeV) +
           bytes(table.rates_cm2_per_g) +
           bytes(table.inverse_energies_MeV) +
           bytes(table.inverse_row_offsets) +
           bytes(table.inverse_quantiles) +
           bytes(table.inverse_v_loss) +
           bytes(table.continuous_pdg_ids) +
           bytes(table.continuous_energy_offsets) +
           bytes(table.continuous_energy_counts) +
           bytes(table.continuous_masses_MeV) +
           bytes(table.continuous_minimum_energies_MeV) +
           bytes(table.continuous_energies_MeV) +
           bytes(table.continuous_dEdX_MeV_cm2_per_g) +
           bytes(table.continuous_ranges_g_per_cm2) +
           bytes(table.epair_rho_component_hashes) +
           bytes(table.epair_rho_energies_MeV) +
           bytes(table.epair_rho_v_coordinates) +
           bytes(table.epair_rho_quantiles) +
           bytes(table.epair_rho_values);
  }

} // namespace corsika::gpu::em::tables

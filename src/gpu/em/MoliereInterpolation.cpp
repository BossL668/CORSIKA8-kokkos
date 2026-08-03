/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include <CubicInterpolation/Axis.h>
#include <CubicInterpolation/CubicSplines.h>
#include <CubicInterpolation/Interpolant.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <stdexcept>
#include <system_error>
#include <utility>

#include <corsika/gpu/em/MoliereScattering.hpp>

namespace corsika::gpu::em {

  namespace {

    using CubicSpline =
        cubic_splines::CubicSplines<double>;
    using CubicInterpolant =
        cubic_splines::Interpolant<CubicSpline>;

    inline constexpr char MoliereCacheMagic[8]{
        'C', '8', 'M', 'O', 'L', 'I', '0', '1'};
    inline constexpr std::uint32_t
        MoliereCacheFormatVersion = 1;

    struct MoliereCacheHeader {
      char magic[8]{};
      std::uint32_t format_version{};
      std::uint32_t snapshot_bytes{};
      std::uint64_t table_bytes{};
      std::uint64_t table_checksum{};
      MoliereSnapshot snapshot{};
    };

    static_assert(
        std::is_trivially_copyable_v<MoliereCacheHeader>);

    std::uint64_t checksumBytes(
        void const* data, std::size_t size) {
      // FNV-1a is sufficient here because the exact snapshot and file size
      // are checked independently. This checksum detects interrupted writes
      // and ordinary cache corruption without adding a crypto dependency.
      auto const* bytes =
          static_cast<unsigned char const*>(data);
      std::uint64_t result = 14695981039346656037ULL;
      for (std::size_t index = 0; index < size; ++index) {
        result ^= bytes[index];
        result *= 1099511628211ULL;
      }
      return result;
    }

    bool finiteInterpolationTable(
        MoliereInterpolationTable const& table) {
      for (auto const& polynomial : table.polynomials) {
        for (double coefficient : polynomial.c) {
          if (!std::isfinite(coefficient)) {
            return false;
          }
        }
      }
      for (double delta : table.initial_guess_delta) {
        if (!std::isfinite(delta)) {
          return false;
        }
      }
      return true;
    }

    bool readInterpolationCache(
        std::filesystem::path const& path,
        MoliereSnapshot const& snapshot,
        MoliereInterpolationTable& table) {
      std::ifstream input(
          path, std::ios::binary | std::ios::ate);
      if (!input) {
        return false;
      }
      auto const expected_size =
          sizeof(MoliereCacheHeader) +
          sizeof(MoliereInterpolationTable);
      auto const file_size = input.tellg();
      if (file_size < 0 ||
          static_cast<std::uintmax_t>(file_size) !=
              expected_size) {
        return false;
      }
      input.seekg(0);
      MoliereCacheHeader header{};
      input.read(
          reinterpret_cast<char*>(&header),
          sizeof(header));
      input.read(
          reinterpret_cast<char*>(&table),
          sizeof(table));
      if (!input ||
          std::memcmp(
              header.magic, MoliereCacheMagic,
              sizeof(MoliereCacheMagic)) != 0 ||
          header.format_version !=
              MoliereCacheFormatVersion ||
          header.snapshot_bytes !=
              sizeof(MoliereSnapshot) ||
          header.table_bytes !=
              sizeof(MoliereInterpolationTable) ||
          std::memcmp(
              &header.snapshot, &snapshot,
              sizeof(snapshot)) != 0 ||
          header.table_checksum !=
              checksumBytes(&table, sizeof(table)) ||
          !finiteInterpolationTable(table)) {
        return false;
      }
      return true;
    }

    void writeInterpolationCache(
        std::filesystem::path const& path,
        MoliereSnapshot const& snapshot,
        MoliereInterpolationTable const& table) {
      auto temporary = path;
      temporary += ".tmp";
      MoliereCacheHeader header{};
      std::memcpy(
          header.magic, MoliereCacheMagic,
          sizeof(MoliereCacheMagic));
      header.format_version = MoliereCacheFormatVersion;
      header.snapshot_bytes = sizeof(MoliereSnapshot);
      header.table_bytes = sizeof(MoliereInterpolationTable);
      header.table_checksum =
          checksumBytes(&table, sizeof(table));
      header.snapshot = snapshot;
      {
        std::ofstream output(
            temporary,
            std::ios::binary | std::ios::trunc);
        if (!output) {
          return;
        }
        output.write(
            reinterpret_cast<char const*>(&header),
            sizeof(header));
        output.write(
            reinterpret_cast<char const*>(&table),
            sizeof(table));
        output.flush();
        if (!output) {
          std::error_code ignored;
          std::filesystem::remove(temporary, ignored);
          return;
        }
      }
      std::error_code error;
      std::filesystem::rename(temporary, path, error);
      if (error) {
        std::filesystem::remove(temporary, error);
      }
    }

    void buildFunctionPolynomials(
        std::function<double(double)> function,
        MoliereCubicPolynomial* output) {
      auto definition = CubicSpline::Definition{};
      definition.f = std::move(function);
      definition.axis =
          std::make_unique<cubic_splines::LinAxis<double>>(
              MoliereInterpolationLower,
              MoliereInterpolationUpper,
              MoliereInterpolationNodeCount);
      CubicInterpolant interpolant(std::move(definition));

      constexpr auto interval_count =
          MoliereInterpolationIntervalCount;
      constexpr auto step =
          (MoliereInterpolationUpper -
           MoliereInterpolationLower) /
          static_cast<double>(interval_count);
      for (std::size_t interval = 0;
           interval < interval_count; ++interval) {
        auto const x0 =
            MoliereInterpolationLower +
            step * static_cast<double>(interval);
        auto const x1 = x0 + step;
        auto const y0 = interpolant.evaluate(x0);
        auto const y1 = interpolant.evaluate(x1);
        auto const derivative0 =
            step * interpolant.prime(x0);
        auto const derivative1 =
            step * interpolant.prime(x1);

        // Cubic Hermite form in q=(x-x0)/step. This is an exact
        // re-expression of the cardinal cubic B-spline on one interval,
        // rather than a second approximation of its values.
        auto& polynomial = output[interval];
        polynomial.c[0] = y0;
        polynomial.c[1] = derivative0;
        polynomial.c[2] =
            3. * (y1 - y0) -
            2. * derivative0 - derivative1;
        polynomial.c[3] =
            2. * (y0 - y1) +
            derivative0 + derivative1;
        for (double coefficient : polynomial.c) {
          if (!std::isfinite(coefficient)) {
            throw std::runtime_error(
                "Moliere interpolation produced a non-finite coefficient");
          }
        }
      }
    }

    bool buildDimensionlessComponentB(
        MoliereSnapshot const& snapshot,
        double reference_B, double beta_squared,
        std::array<double, MaxMoliereComponents>& component_B) {
      if (!(reference_B >= MoliereMinimumB) ||
          !(beta_squared > 0.) ||
          !std::isfinite(reference_B) ||
          !std::isfinite(beta_squared)) {
        return false;
      }
      auto const reference_index =
          snapshot.maximum_weight_index;
      auto const reference_screening =
          snapshot.components[reference_index]
              .chi_0_squared_MeV2 *
          (1.13 +
           snapshot.components[reference_index]
                   .coulomb_correction /
               beta_squared);
      if (!(reference_screening > 0.) ||
          !std::isfinite(reference_screening)) {
        return false;
      }
      auto const reference_rhs =
          reference_B - std::log(reference_B);
      for (std::uint32_t index = 0;
           index < snapshot.component_count; ++index) {
        auto const screening =
            snapshot.components[index]
                .chi_0_squared_MeV2 *
            (1.13 +
             snapshot.components[index]
                     .coulomb_correction /
                 beta_squared);
        if (!(screening > 0.) ||
            !std::isfinite(screening)) {
          return false;
        }
        component_B[index] =
            moliere_detail::solveMoliereB(
                reference_rhs +
                std::log(
                    reference_screening / screening));
        if (!(component_B[index] > 0.) ||
            !std::isfinite(component_B[index])) {
          return false;
        }
      }
      return true;
    }

    double mixtureCumulative(
        MoliereSnapshot const& snapshot,
        MoliereInterpolationView const& interpolation,
        std::array<double, MaxMoliereComponents> const&
            component_B,
        double reference_B, double coordinate) {
      auto result = 0.;
      auto const coordinate_squared =
          coordinate * coordinate;
      auto const inverse_sqrt_pi =
          std::sqrt(1. / moliere_detail::Pi);
      for (std::uint32_t index = 0;
           index < snapshot.component_count; ++index) {
        auto const B = component_B[index];
        auto const inverse_B = 1. / B;
        auto const x =
            coordinate_squared * reference_B * inverse_B;
        auto const inverse_B_squared =
            inverse_B * inverse_B;
        result +=
            snapshot.components[index].weight_zz *
            (0.5 * std::erf(std::sqrt(x)) +
             inverse_sqrt_pi *
                 (moliere_detail::F1M(
                      snapshot, interpolation, x) *
                      inverse_B +
                  moliere_detail::F2M(
                      snapshot, interpolation, x) *
                      inverse_B_squared));
      }
      return result * snapshot.inverse_weight_zz_sum;
    }

    void buildInitialGuessTable(
        MoliereSnapshot const& snapshot,
        MoliereInterpolationView const& interpolation,
        double* output) {
      constexpr auto b_nodes =
          MoliereInitialGuessBNodeCount;
      constexpr auto beta_nodes =
          MoliereInitialGuessBetaSquaredNodeCount;
      constexpr auto coordinate_nodes =
          MoliereInitialGuessCoordinateNodeCount;
      constexpr auto b_step =
          (MoliereInitialGuessBUpper -
           MoliereInitialGuessBLower) /
          static_cast<double>(b_nodes - 1);
      constexpr auto coordinate_step =
          (MoliereInitialGuessCoordinateUpper -
           MoliereInitialGuessCoordinateLower) /
          static_cast<double>(coordinate_nodes - 1);
      constexpr auto beta_step =
          (MoliereInitialGuessBetaSquaredUpper -
           MoliereInitialGuessBetaSquaredLower) /
          static_cast<double>(beta_nodes - 1);
      for (std::size_t b_index = 0;
           b_index < b_nodes; ++b_index) {
        auto const B =
            MoliereInitialGuessBLower +
            static_cast<double>(b_index) * b_step;
        for (std::size_t beta_index = 0;
             beta_index < beta_nodes; ++beta_index) {
          auto const beta_squared =
              MoliereInitialGuessBetaSquaredLower +
              static_cast<double>(beta_index) * beta_step;
          std::array<double, MaxMoliereComponents>
              component_B{};
          auto const valid_components =
              buildDimensionlessComponentB(
                  snapshot, B, beta_squared, component_B);
          for (std::size_t coordinate_index = 0;
               coordinate_index < coordinate_nodes;
               ++coordinate_index) {
            auto const gaussian_coordinate =
                MoliereInitialGuessCoordinateLower +
                static_cast<double>(coordinate_index) *
                    coordinate_step;
            auto const output_index =
                (b_index * beta_nodes + beta_index) *
                    coordinate_nodes +
                coordinate_index;
            if (coordinate_index == 0) {
              output[output_index] = 0.;
              continue;
            }
            if (!valid_components) {
              output[output_index] = 0.;
              continue;
            }
            auto const target =
                0.5 * std::erf(gaussian_coordinate);
            auto lower = 0.;
            auto upper =
                std::max(1., gaussian_coordinate);
            auto upper_cumulative =
                mixtureCumulative(
                    snapshot, interpolation, component_B,
                    B, upper);
            for (int expansion = 0;
                 expansion < 32 &&
                 std::isfinite(upper_cumulative) &&
                 upper_cumulative < target;
                 ++expansion) {
              upper *= 2.;
              upper_cumulative =
                  mixtureCumulative(
                      snapshot, interpolation, component_B,
                      B, upper);
            }
            if (!std::isfinite(upper_cumulative) ||
                upper_cumulative < target) {
              // A zero delta is a safe Gaussian initial guess; the exact
              // multi-component Newton solve remains authoritative.
              output[output_index] = 0.;
              continue;
            }
            for (int iteration = 0; iteration < 24;
                 ++iteration) {
              auto const midpoint = 0.5 * (lower + upper);
              auto const cumulative =
                  mixtureCumulative(
                      snapshot, interpolation, component_B,
                      B, midpoint);
              if (!std::isfinite(cumulative)) {
                lower = upper = gaussian_coordinate;
                break;
              }
              if (cumulative < target) {
                lower = midpoint;
              } else {
                upper = midpoint;
              }
            }
            auto const inverse_coordinate =
                0.5 * (lower + upper);
            auto const delta =
                inverse_coordinate - gaussian_coordinate;
            output[output_index] =
                std::isfinite(delta) ? delta : 0.;
          }
        }
      }
    }

  } // namespace

  MoliereInterpolationTable makeMoliereInterpolationTable(
      MoliereSnapshot const& snapshot) {
    if (!moliere_detail::validSnapshot(snapshot)) {
      throw std::invalid_argument(
          "cannot interpolate an invalid Moliere snapshot");
    }

    MoliereInterpolationTable table;
    constexpr auto intervals =
        MoliereInterpolationIntervalCount;
    buildFunctionPolynomials(
        [&](double x) {
          return moliere_detail::f1M(snapshot, x);
        },
        table.polynomials.data());
    buildFunctionPolynomials(
        [&](double x) {
          return moliere_detail::f2M(snapshot, x);
        },
        table.polynomials.data() + intervals);
    buildFunctionPolynomials(
        [&](double x) {
          return moliere_detail::F1M(snapshot, x);
        },
        table.polynomials.data() + 2 * intervals);
    buildFunctionPolynomials(
        [&](double x) {
          return moliere_detail::F2M(snapshot, x);
        },
        table.polynomials.data() + 3 * intervals);
    auto const interpolation =
        makeMoliereInterpolationView(
            table.polynomials.data());
    buildInitialGuessTable(
        snapshot, interpolation,
        table.initial_guess_delta.data());
    return table;
  }

  MoliereInterpolationTable
  loadOrMakeMoliereInterpolationTable(
      MoliereSnapshot const& snapshot,
      std::filesystem::path const& cache_path) {
    if (!moliere_detail::validSnapshot(snapshot)) {
      throw std::invalid_argument(
          "cannot cache an invalid Moliere snapshot");
    }
    MoliereInterpolationTable table;
    if (!cache_path.empty() &&
        readInterpolationCache(
            cache_path, snapshot, table)) {
      return table;
    }
    table = makeMoliereInterpolationTable(snapshot);
    if (!cache_path.empty()) {
      writeInterpolationCache(
          cache_path, snapshot, table);
    }
    return table;
  }

} // namespace corsika::gpu::em

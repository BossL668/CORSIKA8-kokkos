/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <vector>

#include <corsika/accelerator/AcceleratorMacros.hpp>
#include <corsika/gpu/em/PhotonPairKinematics.hpp>
#include <corsika/gpu/em/ProcessCapabilities.hpp>
#include <corsika/gpu/em/tables/ProposalNativeTable.hpp>
#include <corsika/gpu/em/tables/RateTable.hpp>

#define CORSIKA_GPU_TABLE_HOST_DEVICE C8_ACCELERATOR_INLINE_FUNCTION

namespace corsika::gpu::em::tables {

  /**
   * Non-throwing lookup result used by both host validation and CUDA kernels.
   *
   * Every non-success value is an explicit hybrid-scheduler fallback reason.
   * Device queries never clamp an energy or quantile to the stored domain.
   */
  enum class TableLookupStatus : std::uint32_t {
    Success = 0,
    ParticleNotFound = 1,
    ColumnNotFound = 2,
    RateEnergyOutOfRange = 3,
    InverseCdfUnavailable = 4,
    LossEnergyOutOfRange = 5,
    LossQuantileOutOfRange = 6,
    NonFiniteInput = 7,
    InvalidQueryKind = 8,
    InvalidTableView = 9,
    ContinuousParticleNotFound = 10,
    ContinuousEnergyOutOfRange = 11,
    ContinuousRangeOutOfRange = 12,
    TransportCutReached = 13,
  };

  enum class TableQueryKind : std::uint32_t {
    Rate = 0,
    TotalRate = 1,
    LossFraction = 2,
    ContinuousDedx = 3,
    ContinuousRange = 4,
    ContinuousEnergy = 5,
    EnergyAfterContinuousLoss = 6,
  };

  struct TableQuery {
    TableQueryKind kind{TableQueryKind::Rate};
    std::int32_t pdg_id{};
    std::int32_t process_id{};
    std::uint64_t component_hash{};
    double energy_MeV{};
    double quantile{};
  };

  struct TableQueryResult {
    TableLookupStatus status{TableLookupStatus::InvalidTableView};
    std::uint32_t reserved{};
    double value{};
  };

  /**
   * Result of walking the process/component columns in their canonical
   * PROPOSAL order.  This small POD is shared by the photon selector and the
   * charged-lepton vertex selector so neither kernel needs to know whether
   * the backing storage is a legacy c8emrt table or native PROPOSAL splines.
   */
  struct RateColumnSelectionResult {
    TableLookupStatus status{TableLookupStatus::InvalidTableView};
    std::int32_t process_id{};
    std::uint64_t component_hash{};
    double total_rate{};
    // Proposal-native only: (-sampled_rate)/selected_rate after the exact
    // Interaction::SampleLoss subtraction. Legacy c8emrt selectors leave it
    // zero and keep their independent loss draw unchanged.
    double residual_quantile{};
    std::uint32_t selected{};
  };

  CORSIKA_GPU_TABLE_HOST_DEVICE inline std::uint32_t
  packNativeInverseIterations(std::uint32_t newton,
                              std::uint32_t bisection) {
    auto const bounded_newton = newton > 0xffffu ? 0xffffu : newton;
    auto const bounded_bisection =
        bisection > 0xffffu ? 0xffffu : bisection;
    return bounded_newton | (bounded_bisection << 16u);
  }

  CORSIKA_GPU_TABLE_HOST_DEVICE inline std::uint32_t
  nativeNewtonIterations(TableQueryResult const& result) {
    return result.reserved & 0xffffu;
  }

  CORSIKA_GPU_TABLE_HOST_DEVICE inline std::uint32_t
  nativeBisectionIterations(TableQueryResult const& result) {
    return result.reserved >> 16u;
  }

  /**
   * Pointer-only view copied by value into a CUDA kernel.
   *
   * Particle and column metadata are stored as structure-of-arrays. All
   * offsets are absolute indices into the corresponding flat value array.
   */
  struct FlatRateTableView {
    std::int32_t const* particle_pdg_ids{};
    std::uint32_t const* particle_energy_offsets{};
    std::uint32_t const* particle_energy_counts{};
    std::uint32_t const* particle_column_offsets{};
    std::uint32_t const* particle_column_counts{};
    std::uint32_t particle_count{};

    std::int32_t const* column_process_ids{};
    std::uint64_t const* column_component_hashes{};
    std::uint32_t const* column_rate_offsets{};
    std::uint32_t const* column_inverse_energy_offsets{};
    std::uint32_t const* column_inverse_energy_counts{};
    std::uint32_t const* column_inverse_row_offsets{};
    std::uint32_t column_count{};

    double const* rate_energies_MeV{};
    double const* rates_cm2_per_g{};
    double const* inverse_energies_MeV{};
    std::uint32_t const* inverse_row_offsets{};
    double const* inverse_quantiles{};
    double const* inverse_v_loss{};

    std::uint32_t rate_energy_count{};
    std::uint32_t rate_value_count{};
    std::uint32_t inverse_energy_count{};
    std::uint32_t inverse_row_offset_count{};
    std::uint32_t inverse_value_count{};

    std::int32_t const* continuous_pdg_ids{};
    std::uint32_t const* continuous_energy_offsets{};
    std::uint32_t const* continuous_energy_counts{};
    double const* continuous_masses_MeV{};
    double const* continuous_minimum_energies_MeV{};
    std::uint32_t continuous_particle_count{};

    double const* continuous_energies_MeV{};
    double const* continuous_dEdX_MeV_cm2_per_g{};
    double const* continuous_ranges_g_per_cm2{};
    std::uint32_t continuous_value_count{};
    // Absolute stochastic cut used to generate the PROPOSAL rate columns.
    double energy_cut_MeV{};
    // User-facing EM transport/ParticleCut threshold.  This can differ from
    // energy_cut_MeV because scalar CORSIKA selects the nearest standard
    // PROPOSAL stochastic table below the requested transport cut.
    double em_transport_cut_MeV{};

    std::uint64_t const* epair_rho_component_hashes{};
    double const* epair_rho_energies_MeV{};
    double const* epair_rho_v_coordinates{};
    double const* epair_rho_quantiles{};
    double const* epair_rho_values{};
    std::uint32_t epair_rho_component_count{};
    std::uint32_t epair_rho_energy_count{};
    std::uint32_t epair_rho_v_coordinate_count{};
    std::uint32_t epair_rho_quantile_count{};
    std::uint32_t epair_rho_value_count{};

    // Optional native PROPOSAL core. Legacy c8emrt fields remain unchanged;
    // physics_source==1 selects this read-only native spline view.
    ProposalNativeDeviceView proposal_native{};
    std::uint32_t physics_source{};
    // Three unsigned 64-bit counters: Newton iterations, bisection
    // iterations, and failed native inverse solves. They are diagnostic
    // reductions, not queue append indices, so atomic addition is safe and
    // independent of wavefront scheduling order.
    unsigned long long* native_inverse_counters{};
    // Proposal-native only: user-facing muon kinetic-energy/ParticleCut
    // threshold.  Appended to preserve all legacy c8emrt aggregate
    // initializers and device-view semantics.
    double muon_transport_cut_MeV{};
  };

  /**
   * Owning host representation. Strings and redundant metadata are omitted;
   * the source content hash identifies the exact validated v6 payload.
   */
  struct FlatRateTable {
    Sha256Digest content_hash{};

    std::vector<std::int32_t> particle_pdg_ids;
    std::vector<std::uint32_t> particle_energy_offsets;
    std::vector<std::uint32_t> particle_energy_counts;
    std::vector<std::uint32_t> particle_column_offsets;
    std::vector<std::uint32_t> particle_column_counts;

    std::vector<std::int32_t> column_process_ids;
    std::vector<std::uint64_t> column_component_hashes;
    std::vector<std::uint32_t> column_rate_offsets;
    std::vector<std::uint32_t> column_inverse_energy_offsets;
    std::vector<std::uint32_t> column_inverse_energy_counts;
    std::vector<std::uint32_t> column_inverse_row_offsets;

    std::vector<double> rate_energies_MeV;
    std::vector<double> rates_cm2_per_g;
    std::vector<double> inverse_energies_MeV;
    std::vector<std::uint32_t> inverse_row_offsets;
    std::vector<double> inverse_quantiles;
    std::vector<double> inverse_v_loss;

    std::vector<std::int32_t> continuous_pdg_ids;
    std::vector<std::uint32_t> continuous_energy_offsets;
    std::vector<std::uint32_t> continuous_energy_counts;
    std::vector<double> continuous_masses_MeV;
    std::vector<double> continuous_minimum_energies_MeV;
    std::vector<double> continuous_energies_MeV;
    std::vector<double> continuous_dEdX_MeV_cm2_per_g;
    std::vector<double> continuous_ranges_g_per_cm2;
    double energy_cut_MeV{};

    std::vector<std::uint64_t> epair_rho_component_hashes;
    std::vector<double> epair_rho_energies_MeV;
    std::vector<double> epair_rho_v_coordinates;
    std::vector<double> epair_rho_quantiles;
    std::vector<double> epair_rho_values;
  };

  static_assert(std::is_standard_layout_v<FlatRateTableView>);
  static_assert(std::is_trivially_copyable_v<FlatRateTableView>);
  static_assert(std::is_standard_layout_v<TableQuery>);
  static_assert(std::is_trivially_copyable_v<TableQuery>);
  static_assert(std::is_standard_layout_v<TableQueryResult>);
  static_assert(std::is_trivially_copyable_v<TableQueryResult>);
  static_assert(std::is_standard_layout_v<RateColumnSelectionResult>);
  static_assert(std::is_trivially_copyable_v<RateColumnSelectionResult>);

  FlatRateTable flattenRateTable(RateTableSet const& source);
  void validateFlatRateTable(FlatRateTable const& table);
  FlatRateTableView makeFlatRateTableView(FlatRateTable const& table);
  std::size_t flatRateTableBytes(FlatRateTable const& table);

  struct EpairRhoLookupResult {
    TableLookupStatus status{TableLookupStatus::InvalidTableView};
    std::uint32_t reserved{};
    double normalized_rho{};
  };

  static_assert(std::is_standard_layout_v<EpairRhoLookupResult>);
  static_assert(std::is_trivially_copyable_v<EpairRhoLookupResult>);

  namespace detail {

    CORSIKA_GPU_TABLE_HOST_DEVICE inline bool finiteValue(double value) {
#if defined(__CUDA_ARCH__)
      return ::isfinite(value);
#else
      return std::isfinite(value);
#endif
    }

    CORSIKA_GPU_TABLE_HOST_DEVICE inline double logarithm(double value) {
#if defined(__CUDA_ARCH__)
      return ::log(value);
#else
      return std::log(value);
#endif
    }

    CORSIKA_GPU_TABLE_HOST_DEVICE inline double exponential(double value) {
#if defined(__CUDA_ARCH__)
      return ::exp(value);
#else
      return std::exp(value);
#endif
    }

    CORSIKA_GPU_TABLE_HOST_DEVICE inline double logit(double quantile) {
      return logarithm(quantile / (1. - quantile));
    }

    CORSIKA_GPU_TABLE_HOST_DEVICE inline std::uint32_t lowerBound(
        double const* values, std::uint32_t count, double query) {
      std::uint32_t first = 0;
      while (first < count) {
        auto const step = (count - first) / 2;
        auto const middle = first + step;
        if (values[middle] < query) {
          first = middle + 1;
        } else {
          count = middle;
        }
      }
      return first;
    }

    CORSIKA_GPU_TABLE_HOST_DEVICE inline bool
    validEpairRhoView(FlatRateTableView const& view) {
      if (view.epair_rho_component_count == 0) {
        return false;
      }
      auto const expected =
          static_cast<std::uint64_t>(
              view.epair_rho_component_count) *
          view.epair_rho_energy_count *
          view.epair_rho_v_coordinate_count *
          view.epair_rho_quantile_count;
      return view.epair_rho_energy_count >= 2 &&
             view.epair_rho_v_coordinate_count >= 2 &&
             view.epair_rho_quantile_count >= 2 &&
             expected == view.epair_rho_value_count &&
             view.epair_rho_component_hashes != nullptr &&
             view.epair_rho_energies_MeV != nullptr &&
             view.epair_rho_v_coordinates != nullptr &&
             view.epair_rho_quantiles != nullptr &&
             view.epair_rho_values != nullptr;
    }

    CORSIKA_GPU_TABLE_HOST_DEVICE inline bool interpolationBracket(
        double const* axis, std::uint32_t count, double value,
        bool logarithmic, bool logit_transform,
        std::uint32_t& lower, double& fraction) {
      if (!finiteValue(value) || count < 2 ||
          value < axis[0] || value > axis[count - 1]) {
        return false;
      }
      auto const upper = lowerBound(axis, count, value);
      if (upper == 0) {
        lower = 0;
        fraction = 0.;
        return true;
      }
      if (upper == count) {
        lower = count - 2;
        fraction = 1.;
        return true;
      }
      if (axis[upper] == value) {
        if (upper == count - 1) {
          lower = upper - 1;
          fraction = 1.;
        } else {
          lower = upper;
          fraction = 0.;
        }
        return true;
      }
      lower = upper - 1;
      auto transform =
          [&](double input) {
            if (logit_transform) {
              return logit(input);
            }
            return logarithmic ? logarithm(input) : input;
          };
      auto const first = transform(axis[lower]);
      auto const second = transform(axis[lower + 1]);
      auto const query = transform(value);
      fraction = (query - first) / (second - first);
      return finiteValue(fraction);
    }

    CORSIKA_GPU_TABLE_HOST_DEVICE inline double interpolatePositive(
        double lower, double upper, double fraction) {
      if (lower > 0. && upper > 0.) {
        return exponential(logarithm(lower) +
                           fraction *
                               (logarithm(upper) - logarithm(lower)));
      }
      return lower + fraction * (upper - lower);
    }

    CORSIKA_GPU_TABLE_HOST_DEVICE inline bool validView(
        FlatRateTableView const& view) {
      return view.particle_count > 0 && view.column_count > 0 &&
             view.continuous_particle_count > 0 &&
             view.continuous_value_count > 0 &&
             finiteValue(view.energy_cut_MeV) &&
             view.energy_cut_MeV > 0. &&
             finiteValue(view.em_transport_cut_MeV) &&
             view.em_transport_cut_MeV > 0. &&
             view.particle_pdg_ids != nullptr &&
             view.particle_energy_offsets != nullptr &&
             view.particle_energy_counts != nullptr &&
             view.particle_column_offsets != nullptr &&
             view.particle_column_counts != nullptr &&
             view.column_process_ids != nullptr &&
             view.column_component_hashes != nullptr &&
             view.column_rate_offsets != nullptr &&
             view.column_inverse_energy_offsets != nullptr &&
             view.column_inverse_energy_counts != nullptr &&
             view.column_inverse_row_offsets != nullptr &&
             view.rate_energies_MeV != nullptr &&
             view.rates_cm2_per_g != nullptr &&
             view.continuous_pdg_ids != nullptr &&
             view.continuous_energy_offsets != nullptr &&
             view.continuous_energy_counts != nullptr &&
             view.continuous_masses_MeV != nullptr &&
             view.continuous_minimum_energies_MeV != nullptr &&
             view.continuous_energies_MeV != nullptr &&
             view.continuous_dEdX_MeV_cm2_per_g != nullptr &&
             view.continuous_ranges_g_per_cm2 != nullptr;
    }

    CORSIKA_GPU_TABLE_HOST_DEVICE inline std::uint32_t findParticle(
        FlatRateTableView const& view, std::int32_t pdg_id) {
      for (std::uint32_t index = 0; index < view.particle_count; ++index) {
        if (view.particle_pdg_ids[index] == pdg_id) {
          return index;
        }
      }
      return view.particle_count;
    }

    CORSIKA_GPU_TABLE_HOST_DEVICE inline std::uint32_t findColumn(
        FlatRateTableView const& view, std::uint32_t particle_index,
        std::int32_t process_id, std::uint64_t component_hash) {
      auto const begin = view.particle_column_offsets[particle_index];
      auto const end = begin + view.particle_column_counts[particle_index];
      for (auto index = begin; index < end; ++index) {
        if (view.column_process_ids[index] == process_id &&
            view.column_component_hashes[index] == component_hash) {
          return index;
        }
      }
      return view.column_count;
    }

    CORSIKA_GPU_TABLE_HOST_DEVICE inline std::uint32_t
    findContinuousParticle(FlatRateTableView const& view,
                           std::int32_t pdg_id) {
      for (std::uint32_t index = 0;
           index < view.continuous_particle_count; ++index) {
        if (view.continuous_pdg_ids[index] == pdg_id) {
          return index;
        }
      }
      return view.continuous_particle_count;
    }

    CORSIKA_GPU_TABLE_HOST_DEVICE inline TableQueryResult
    continuousEnergyBracket(FlatRateTableView const& view,
                            std::uint32_t particle_index,
                            double energy_MeV,
                            std::uint32_t& lower,
                            double& fraction) {
      if (!finiteValue(energy_MeV)) {
        return {TableLookupStatus::NonFiniteInput, 0, 0.};
      }
      auto const offset =
          view.continuous_energy_offsets[particle_index];
      auto const count =
          view.continuous_energy_counts[particle_index];
      if (count < 2 ||
          static_cast<std::uint64_t>(offset) + count >
              view.continuous_value_count) {
        return {TableLookupStatus::InvalidTableView, 0, 0.};
      }
      auto const* energies = view.continuous_energies_MeV + offset;
      if (energy_MeV < energies[0] ||
          energy_MeV > energies[count - 1]) {
        return {TableLookupStatus::ContinuousEnergyOutOfRange, 0, 0.};
      }
      auto const upper = lowerBound(energies, count, energy_MeV);
      if (upper == 0) {
        lower = 0;
        fraction = 0.;
      } else if (upper == count) {
        lower = count - 2;
        fraction = 1.;
      } else if (energies[upper] == energy_MeV) {
        if (upper == count - 1) {
          lower = upper - 1;
          fraction = 1.;
        } else {
          lower = upper;
          fraction = 0.;
        }
      } else {
        lower = upper - 1;
        fraction =
            (logarithm(energy_MeV) - logarithm(energies[lower])) /
            (logarithm(energies[lower + 1]) -
             logarithm(energies[lower]));
      }
      return {TableLookupStatus::Success, 0, 0.};
    }

    struct RateInterpolationBracket {
      TableLookupStatus status{
          TableLookupStatus::InvalidTableView};
      std::uint32_t lower{};
      std::uint32_t upper{};
      double fraction{};
    };

    /**
     * Locate the common rate-energy bracket once for all process/component
     * columns of a particle.
     *
     * Every rate column belonging to one particle uses the same energy grid.
     * Keeping the binary search and logarithmic coordinate outside the column
     * loop avoids repeating them for total-rate and process selection queries.
     */
    CORSIKA_GPU_TABLE_HOST_DEVICE inline RateInterpolationBracket
    rateInterpolationBracket(
        FlatRateTableView const& view,
        std::uint32_t particle_index, double energy_MeV) {
      if (!finiteValue(energy_MeV)) {
        return {TableLookupStatus::NonFiniteInput, 0, 0, 0.};
      }
      auto const energy_offset =
          view.particle_energy_offsets[particle_index];
      auto const energy_count =
          view.particle_energy_counts[particle_index];
      if (energy_count < 2 ||
          static_cast<std::uint64_t>(energy_offset) + energy_count >
              view.rate_energy_count) {
        return {TableLookupStatus::InvalidTableView, 0, 0, 0.};
      }
      auto const* energies = view.rate_energies_MeV + energy_offset;
      if (energy_MeV < energies[0] ||
          energy_MeV > energies[energy_count - 1]) {
        return {
            TableLookupStatus::RateEnergyOutOfRange, 0, 0, 0.};
      }
      auto const upper = lowerBound(energies, energy_count, energy_MeV);
      if (upper == 0) {
        return {TableLookupStatus::Success, 0, 0, 0.};
      }
      if (upper == energy_count) {
        return {
            TableLookupStatus::Success,
            energy_count - 1, energy_count - 1, 0.};
      }
      if (energies[upper] == energy_MeV) {
        return {
            TableLookupStatus::Success, upper, upper, 0.};
      }
      auto const lower = upper - 1;
      auto const fraction =
          (logarithm(energy_MeV) - logarithm(energies[lower])) /
          (logarithm(energies[upper]) - logarithm(energies[lower]));
      return {
          TableLookupStatus::Success, lower, upper, fraction};
    }

    CORSIKA_GPU_TABLE_HOST_DEVICE inline TableQueryResult
    interpolateRateColumnAtBracket(
        FlatRateTableView const& view,
        std::uint32_t particle_index,
        std::uint32_t column_index,
        RateInterpolationBracket const& bracket) {
      if (bracket.status != TableLookupStatus::Success) {
        return {bracket.status, 0, 0.};
      }
      auto const energy_count =
          view.particle_energy_counts[particle_index];
      auto const rate_offset =
          view.column_rate_offsets[column_index];
      if (bracket.lower >= energy_count ||
          bracket.upper >= energy_count ||
          static_cast<std::uint64_t>(rate_offset) + energy_count >
              view.rate_value_count) {
        return {TableLookupStatus::InvalidTableView, 0, 0.};
      }
      auto const* rates =
          view.rates_cm2_per_g + rate_offset;
      if (bracket.lower == bracket.upper) {
        return {
            TableLookupStatus::Success, 0,
            rates[bracket.lower]};
      }
      return {TableLookupStatus::Success, 0,
              interpolatePositive(
                  rates[bracket.lower], rates[bracket.upper],
                  bracket.fraction)};
    }

    CORSIKA_GPU_TABLE_HOST_DEVICE inline TableQueryResult
    interpolateRateColumn(
        FlatRateTableView const& view,
        std::uint32_t particle_index,
        std::uint32_t column_index, double energy_MeV) {
      return interpolateRateColumnAtBracket(
          view, particle_index, column_index,
          rateInterpolationBracket(
              view, particle_index, energy_MeV));
    }

    CORSIKA_GPU_TABLE_HOST_DEVICE inline TableQueryResult interpolateLossRow(
        FlatRateTableView const& view, std::uint32_t row_offset_index,
        double quantile) {
      if (view.inverse_row_offset_count < 2 ||
          row_offset_index >= view.inverse_row_offset_count - 1) {
        return {TableLookupStatus::InvalidTableView, 0, 0.};
      }
      auto const begin = view.inverse_row_offsets[row_offset_index];
      auto const end = view.inverse_row_offsets[row_offset_index + 1];
      if (begin >= end || end > view.inverse_value_count) {
        return {TableLookupStatus::InvalidTableView, 0, 0.};
      }
      auto const count = end - begin;
      auto const* quantiles = view.inverse_quantiles + begin;
      auto const* losses = view.inverse_v_loss + begin;
      auto const upper = lowerBound(quantiles, count, quantile);
      if (upper == 0) {
        return {TableLookupStatus::Success, 0, losses[0]};
      }
      if (upper == count) {
        return {TableLookupStatus::Success, 0, losses[count - 1]};
      }
      if (quantiles[upper] == quantile) {
        return {TableLookupStatus::Success, 0, losses[upper]};
      }
      auto const lower = upper - 1;
      auto const fraction =
          (logit(quantile) - logit(quantiles[lower])) /
          (logit(quantiles[upper]) - logit(quantiles[lower]));
      return {TableLookupStatus::Success, 0,
              interpolatePositive(losses[lower], losses[upper], fraction)};
    }

    CORSIKA_GPU_TABLE_HOST_DEVICE inline TableQueryResult fromProposalNative(
        NativeQueryResult const& result, bool continuous = false) {
      switch (result.status) {
      case NativeQueryStatus::Success:
        return {TableLookupStatus::Success,
                packNativeInverseIterations(
                    result.newton_iterations,
                    result.bisection_iterations),
                result.value};
      case NativeQueryStatus::ParticleNotFound:
        return {continuous ? TableLookupStatus::ContinuousParticleNotFound
                           : TableLookupStatus::ParticleNotFound,
                packNativeInverseIterations(
                    result.newton_iterations,
                    result.bisection_iterations), 0.};
      case NativeQueryStatus::ColumnNotFound:
        return {TableLookupStatus::ColumnNotFound,
                packNativeInverseIterations(
                    result.newton_iterations,
                    result.bisection_iterations), 0.};
      case NativeQueryStatus::EnergyOutOfRange:
        return {continuous ? TableLookupStatus::ContinuousEnergyOutOfRange
                           : TableLookupStatus::RateEnergyOutOfRange,
                packNativeInverseIterations(
                    result.newton_iterations,
                    result.bisection_iterations), 0.};
      case NativeQueryStatus::QuantileOutOfRange:
        return {TableLookupStatus::LossQuantileOutOfRange,
                packNativeInverseIterations(
                    result.newton_iterations,
                    result.bisection_iterations), 0.};
      case NativeQueryStatus::UnsupportedKinematics:
      case NativeQueryStatus::RootNotConverged:
        return {TableLookupStatus::InverseCdfUnavailable,
                packNativeInverseIterations(
                    result.newton_iterations,
                    result.bisection_iterations), 0.};
      case NativeQueryStatus::NonFiniteInput:
        return {TableLookupStatus::NonFiniteInput,
                packNativeInverseIterations(
                    result.newton_iterations,
                    result.bisection_iterations), 0.};
      case NativeQueryStatus::InvalidView:
        return {TableLookupStatus::InvalidTableView,
                packNativeInverseIterations(
                    result.newton_iterations,
                    result.bisection_iterations), 0.};
      }
      return {TableLookupStatus::InvalidTableView,
              packNativeInverseIterations(
                  result.newton_iterations,
                  result.bisection_iterations), 0.};
    }

  } // namespace detail

  /**
   * Trilinear interpolation of |rho|/rho_max. Energy uses log(E), while both
   * independent random-number axes use logit coordinates. No clamping is
   * permitted: an unavailable component or out-of-domain query is routed to
   * the explicit PROPOSAL fallback by the final-state kernel.
   */
  CORSIKA_GPU_TABLE_HOST_DEVICE inline EpairRhoLookupResult
  interpolateEpairRho(
      FlatRateTableView const& view,
      std::uint64_t component_hash, double energy_MeV,
      double v_coordinate, double rho_quantile) {
    if (!detail::validEpairRhoView(view)) {
      return {
          TableLookupStatus::InverseCdfUnavailable, 0, 0.};
    }
    std::uint32_t component = 0;
    while (component < view.epair_rho_component_count &&
           view.epair_rho_component_hashes[component] !=
               component_hash) {
      ++component;
    }
    if (component == view.epair_rho_component_count) {
      return {TableLookupStatus::ColumnNotFound, 0, 0.};
    }
    std::uint32_t energy_lower = 0;
    std::uint32_t loss_lower = 0;
    std::uint32_t rho_lower = 0;
    double energy_fraction = 0.;
    double loss_fraction = 0.;
    double rho_fraction = 0.;
    if (!detail::interpolationBracket(
            view.epair_rho_energies_MeV,
            view.epair_rho_energy_count, energy_MeV, true,
            false, energy_lower, energy_fraction)) {
      return {
          TableLookupStatus::LossEnergyOutOfRange, 0, 0.};
    }
    if (!detail::interpolationBracket(
            view.epair_rho_v_coordinates,
            view.epair_rho_v_coordinate_count,
            v_coordinate, true, false, loss_lower,
            loss_fraction) ||
        !detail::interpolationBracket(
            view.epair_rho_quantiles,
            view.epair_rho_quantile_count, rho_quantile,
            false, true, rho_lower, rho_fraction)) {
      return {
          TableLookupStatus::LossQuantileOutOfRange, 0, 0.};
    }
    auto const valueAt =
        [&](std::uint32_t energy_index,
            std::uint32_t loss_index,
            std::uint32_t rho_index) {
          auto const index =
              (((static_cast<std::uint64_t>(component) *
                     view.epair_rho_energy_count +
                 energy_index) *
                    view.epair_rho_v_coordinate_count +
                loss_index) *
                   view.epair_rho_quantile_count +
               rho_index);
          return view.epair_rho_values[index];
        };
    auto interpolate = [](double first, double second,
                          double fraction) {
      return first + fraction * (second - first);
    };
    double energy_values[2]{};
    for (std::uint32_t energy_side = 0;
         energy_side < 2; ++energy_side) {
      double loss_values[2]{};
      for (std::uint32_t loss_side = 0;
           loss_side < 2; ++loss_side) {
        loss_values[loss_side] = interpolate(
            valueAt(
                energy_lower + energy_side,
                loss_lower + loss_side, rho_lower),
            valueAt(
                energy_lower + energy_side,
                loss_lower + loss_side, rho_lower + 1),
            rho_fraction);
      }
      energy_values[energy_side] = interpolate(
          loss_values[0], loss_values[1], loss_fraction);
    }
    auto const value = interpolate(
        energy_values[0], energy_values[1],
        energy_fraction);
    if (!detail::finiteValue(value) || value < 0. ||
        value > 1.) {
      return {TableLookupStatus::InvalidTableView, 0, 0.};
    }
    return {TableLookupStatus::Success, 0, value};
  }

  CORSIKA_GPU_TABLE_HOST_DEVICE inline TableQueryResult queryRate(
      FlatRateTableView const& view, std::int32_t pdg_id,
      std::int32_t process_id, std::uint64_t component_hash,
      double energy_MeV) {
    if (view.physics_source == 1u) {
      return detail::fromProposalNative(queryProposalNativeRate(
          view.proposal_native, pdg_id, process_id, component_hash,
          energy_MeV));
    }
    if (!detail::validView(view)) {
      return {TableLookupStatus::InvalidTableView, 0, 0.};
    }
    auto const particle = detail::findParticle(view, pdg_id);
    if (particle == view.particle_count) {
      return {TableLookupStatus::ParticleNotFound, 0, 0.};
    }
    auto const column =
        detail::findColumn(view, particle, process_id, component_hash);
    if (column == view.column_count) {
      return {TableLookupStatus::ColumnNotFound, 0, 0.};
    }
    return detail::interpolateRateColumn(
        view, particle, column, energy_MeV);
  }

  CORSIKA_GPU_TABLE_HOST_DEVICE inline TableQueryResult queryTotalRate(
      FlatRateTableView const& view, std::int32_t pdg_id,
      double energy_MeV) {
    if (view.physics_source == 1u) {
      return detail::fromProposalNative(queryProposalNativeTotalRate(
          view.proposal_native, pdg_id, energy_MeV));
    }
    if (!detail::validView(view)) {
      return {TableLookupStatus::InvalidTableView, 0, 0.};
    }
    auto const particle = detail::findParticle(view, pdg_id);
    if (particle == view.particle_count) {
      return {TableLookupStatus::ParticleNotFound, 0, 0.};
    }
    auto const begin = view.particle_column_offsets[particle];
    auto const end = begin + view.particle_column_counts[particle];
    auto const bracket = detail::rateInterpolationBracket(
        view, particle, energy_MeV);
    if (bracket.status != TableLookupStatus::Success) {
      return {bracket.status, 0, 0.};
    }
    double total = 0.;
    for (auto column = begin; column < end; ++column) {
      auto const result =
          detail::interpolateRateColumnAtBracket(
              view, particle, column, bracket);
      if (result.status != TableLookupStatus::Success) {
        return result;
      }
      total += result.value;
    }
    if (!detail::finiteValue(total)) {
      return {TableLookupStatus::InvalidTableView, 0, 0.};
    }
    return {TableLookupStatus::Success, 0, total};
  }

  /**
   * Select a process/component with the same cumulative-rate rule used by
   * scalar CORSIKA.  The threshold is an absolute rate in [0,total), not a
   * unit random number.  All columns are still evaluated after a selection
   * so the returned total can be used by the lepton vertex-reselection
   * algorithm without a second table traversal.
   */
  CORSIKA_GPU_TABLE_HOST_DEVICE inline RateColumnSelectionResult
  selectRateColumnByThreshold(
      FlatRateTableView const& view, std::int32_t pdg_id,
      double energy_MeV, double threshold) {
    if (!detail::finiteValue(energy_MeV) ||
        !detail::finiteValue(threshold) || threshold < 0.) {
      return {TableLookupStatus::NonFiniteInput, 0, 0, 0., 0., 0};
    }

    double cumulative = 0.;
    std::int32_t selected_process = 0;
    std::uint64_t selected_component = 0;
    std::int32_t last_positive_process = 0;
    std::uint64_t last_positive_component = 0;
    double selected_residual_quantile = 0.;
    bool found_particle = false;
    bool selected = false;
    bool have_positive = false;

    if (view.physics_source == 1u) {
      if (!native_detail::validView(view.proposal_native)) {
        return {TableLookupStatus::InvalidTableView, 0, 0, 0., 0., 0};
      }
      auto const range = native_detail::dndxParticleRange(
          view.proposal_native, pdg_id);
      for (std::uint32_t index = range.begin;
           index < range.end; ++index) {
        auto const* selected_column = native_detail::selectionDndx(
            view.proposal_native, index);
        if (!selected_column || selected_column->pdg_id != pdg_id)
          return {TableLookupStatus::InvalidTableView, 0, 0,
                  cumulative, 0., 0};
        auto const& column = *selected_column;
        found_particle = true;
        NativeQueryResult native_rate{
            NativeQueryStatus::Success, 0, 0, 0.};
        if (!proposalNativeRateBelowThreshold(column, energy_MeV))
          native_rate = queryProposalNativeRateForColumn(
              view.proposal_native, column, energy_MeV);
        auto const rate = detail::fromProposalNative(native_rate);
        if (rate.status != TableLookupStatus::Success) {
          // Preserve the exact descriptor identity for device-side failure
          // diagnostics.  The caller can then distinguish a bad column from
          // a malformed aggregate query without repeating the traversal.
          return {rate.status, column.process_id, column.component_hash,
                  cumulative, 0., 0};
        }
        if (rate.value > 0.) {
          have_positive = true;
          last_positive_process = column.process_id;
          last_positive_component = column.component_hash;
        }
        cumulative += rate.value;
        if (!selected && threshold < cumulative) {
          selected = true;
          selected_process = column.process_id;
          selected_component = column.component_hash;
          selected_residual_quantile =
              rate.value > 0. ? (cumulative - threshold) / rate.value : 0.;
        }
      }
    } else {
      if (!detail::validView(view)) {
        return {TableLookupStatus::InvalidTableView, 0, 0, 0., 0., 0};
      }
      auto const particle = detail::findParticle(view, pdg_id);
      if (particle == view.particle_count) {
        return {TableLookupStatus::ParticleNotFound, 0, 0, 0., 0., 0};
      }
      found_particle = true;
      auto const bracket =
          detail::rateInterpolationBracket(view, particle, energy_MeV);
      if (bracket.status != TableLookupStatus::Success) {
        return {bracket.status, 0, 0, 0., 0., 0};
      }
      auto const begin = view.particle_column_offsets[particle];
      auto const end = begin + view.particle_column_counts[particle];
      for (auto column = begin; column < end; ++column) {
        auto const rate = detail::interpolateRateColumnAtBracket(
            view, particle, column, bracket);
        if (rate.status != TableLookupStatus::Success) {
          return {rate.status, 0, 0, 0., 0., 0};
        }
        if (rate.value > 0.) {
          have_positive = true;
          last_positive_process = view.column_process_ids[column];
          last_positive_component =
              view.column_component_hashes[column];
        }
        cumulative += rate.value;
        if (!selected && threshold < cumulative) {
          selected = true;
          selected_process = view.column_process_ids[column];
          selected_component = view.column_component_hashes[column];
        }
      }
    }

    if (!found_particle) {
      return {TableLookupStatus::ParticleNotFound, 0, 0, 0., 0., 0};
    }
    if (!detail::finiteValue(cumulative)) {
      return {TableLookupStatus::InvalidTableView, 0, 0, 0., 0., 0};
    }
    // Guard the last representable boundary exactly as the legacy selectors
    // did.  This is only reachable through floating-point summation drift.
    if (!selected && have_positive && threshold < cumulative) {
      selected = true;
      selected_process = last_positive_process;
      selected_component = last_positive_component;
    }
    return {TableLookupStatus::Success, selected_process,
            selected_component, cumulative, selected_residual_quantile,
            selected ? 1u : 0u};
  }

  /** Select from a unit uniform with PROPOSAL 7.6.2 SampleLoss semantics.
   * Native mode walks the exported Rates insertion order, first accumulates
   * the flat per-target total, then evaluates sampled_rate=u*total and
   * subtracts every rate until sampled_rate<0. The residual in the selected
   * interval is the stochastic-loss quantile. */
  CORSIKA_GPU_TABLE_HOST_DEVICE inline RateColumnSelectionResult
  selectRateColumnByUniform(FlatRateTableView const& view,
                            std::int32_t pdg_id, double energy_MeV,
                            double uniform) {
    if (!detail::finiteValue(uniform) || uniform < 0. || uniform >= 1.)
      return {TableLookupStatus::NonFiniteInput, 0, 0, 0., 0., 0};
    if (view.physics_source != 1u) {
      auto const total = queryTotalRate(view, pdg_id, energy_MeV);
      if (total.status != TableLookupStatus::Success)
        return {total.status, 0, 0, 0., 0., 0};
      return selectRateColumnByThreshold(
          view, pdg_id, energy_MeV, uniform * total.value);
    }
    auto const total = queryProposalNativeSelectionTotalRate(
        view.proposal_native, pdg_id, energy_MeV);
    auto const converted_total = detail::fromProposalNative(total);
    if (converted_total.status != TableLookupStatus::Success)
      return {converted_total.status, 0, 0, 0., 0., 0};
    if (!(converted_total.value > 0.))
      return {TableLookupStatus::Success, 0, 0,
              converted_total.value, 0., 0};

    auto sampled_rate = uniform * converted_total.value;
    auto const range = native_detail::dndxParticleRange(
        view.proposal_native, pdg_id);
    for (auto position = range.begin; position < range.end; ++position) {
      auto const* column = native_detail::selectionDndx(
          view.proposal_native, position);
      if (!column || column->pdg_id != pdg_id)
        return {TableLookupStatus::InvalidTableView, 0, 0,
                converted_total.value, 0., 0};
      NativeQueryResult native_rate{NativeQueryStatus::Success, 0, 0, 0.};
      if (!proposalNativeRateBelowThreshold(*column, energy_MeV))
        native_rate = queryProposalNativeRateForColumn(
            view.proposal_native, *column, energy_MeV);
      auto const rate = detail::fromProposalNative(native_rate);
      if (rate.status != TableLookupStatus::Success)
        return {rate.status, column->process_id, column->component_hash,
                converted_total.value, 0., 0};
      sampled_rate -= rate.value;
      if (sampled_rate < 0.) {
        auto const quantile = (-sampled_rate) / rate.value;
        if (!detail::finiteValue(quantile) || quantile < 0. || quantile > 1.)
          return {TableLookupStatus::InvalidTableView, column->process_id,
                  column->component_hash, converted_total.value, 0., 0};
        return {TableLookupStatus::Success, column->process_id,
                column->component_hash, converted_total.value, quantile, 1u};
      }
    }
    return {TableLookupStatus::Success, 0, 0,
            converted_total.value, 0., 0};
  }

  CORSIKA_GPU_TABLE_HOST_DEVICE inline TableQueryResult queryLossFraction(
      FlatRateTableView const& view, std::int32_t pdg_id,
      std::int32_t process_id, std::uint64_t component_hash,
      double energy_MeV, double quantile) {
    if (view.physics_source == 1u) {
      return detail::fromProposalNative(queryProposalNativeLossFraction(
          view.proposal_native, pdg_id, process_id, component_hash,
          energy_MeV, quantile));
    }
    if (!detail::validView(view)) {
      return {TableLookupStatus::InvalidTableView, 0, 0.};
    }
    if (!detail::finiteValue(energy_MeV) ||
        !detail::finiteValue(quantile)) {
      return {TableLookupStatus::NonFiniteInput, 0, 0.};
    }
    auto const particle = detail::findParticle(view, pdg_id);
    if (particle == view.particle_count) {
      return {TableLookupStatus::ParticleNotFound, 0, 0.};
    }
    auto const column =
        detail::findColumn(view, particle, process_id, component_hash);
    if (column == view.column_count) {
      return {TableLookupStatus::ColumnNotFound, 0, 0.};
    }
    auto const energy_count =
        view.column_inverse_energy_counts[column];
    if (energy_count < 2) {
      return {TableLookupStatus::InverseCdfUnavailable, 0, 0.};
    }
    auto const energy_offset =
        view.column_inverse_energy_offsets[column];
    auto const row_offset =
        view.column_inverse_row_offsets[column];
    if (view.inverse_energies_MeV == nullptr ||
        static_cast<std::uint64_t>(energy_offset) + energy_count >
            view.inverse_energy_count ||
        static_cast<std::uint64_t>(row_offset) + energy_count + 1 >
            view.inverse_row_offset_count ||
        view.inverse_quantiles == nullptr ||
        view.inverse_v_loss == nullptr ||
        view.inverse_row_offsets == nullptr) {
      return {TableLookupStatus::InvalidTableView, 0, 0.};
    }
    auto const* energies = view.inverse_energies_MeV + energy_offset;
    if (energy_MeV < energies[0] ||
        energy_MeV > energies[energy_count - 1]) {
      return {TableLookupStatus::LossEnergyOutOfRange, 0, 0.};
    }
    auto const first_begin = view.inverse_row_offsets[row_offset];
    auto const first_end = view.inverse_row_offsets[row_offset + 1];
    if (first_begin >= first_end ||
        first_end > view.inverse_value_count) {
      return {TableLookupStatus::InvalidTableView, 0, 0.};
    }
    if (quantile < view.inverse_quantiles[first_begin] ||
        quantile > view.inverse_quantiles[first_end - 1]) {
      return {TableLookupStatus::LossQuantileOutOfRange, 0, 0.};
    }

    auto const upper =
        detail::lowerBound(energies, energy_count, energy_MeV);
    std::uint32_t lower_energy = 0;
    double energy_fraction = 0.;
    if (upper == 0) {
      lower_energy = 0;
    } else if (upper == energy_count) {
      lower_energy = energy_count - 2;
      energy_fraction = 1.;
    } else if (energies[upper] == energy_MeV) {
      if (upper == energy_count - 1) {
        lower_energy = upper - 1;
        energy_fraction = 1.;
      } else {
        lower_energy = upper;
      }
    } else {
      lower_energy = upper - 1;
      auto const pair_final_state =
          process_id == PhotonPairFinalStateProcessId;
      auto const query_coordinate =
          detail::logarithm(
              pair_final_state
                  ? energy_MeV - PhotonPairThresholdMeV
                  : energy_MeV);
      auto const lower_coordinate =
          detail::logarithm(
              pair_final_state
                  ? energies[lower_energy] -
                        PhotonPairThresholdMeV
                  : energies[lower_energy]);
      auto const upper_coordinate =
          detail::logarithm(
              pair_final_state
                  ? energies[lower_energy + 1] -
                        PhotonPairThresholdMeV
                  : energies[lower_energy + 1]);
      energy_fraction =
          (query_coordinate - lower_coordinate) /
          (upper_coordinate - lower_coordinate);
    }
    auto const lower = detail::interpolateLossRow(
        view, row_offset + lower_energy, quantile);
    if (lower.status != TableLookupStatus::Success) {
      return lower;
    }
    auto const upper_loss = detail::interpolateLossRow(
        view, row_offset + lower_energy + 1, quantile);
    if (upper_loss.status != TableLookupStatus::Success) {
      return upper_loss;
    }
    auto value = detail::interpolatePositive(
        lower.value, upper_loss.value, energy_fraction);
    if (!detail::finiteValue(value) || value < -1.e-14 ||
        value > 1. + 1.e-14) {
      return {TableLookupStatus::InvalidTableView, 0, 0.};
    }
    if (value < 0.) {
      value = 0.;
    } else if (value > 1.) {
      value = 1.;
    }
    return {TableLookupStatus::Success, 0, value};
  }

  CORSIKA_GPU_TABLE_HOST_DEVICE inline TableQueryResult queryContinuousDedx(
      FlatRateTableView const& view, std::int32_t pdg_id,
      double energy_MeV) {
    if (view.physics_source == 1u) {
      return detail::fromProposalNative(queryProposalNativeDedx(
          view.proposal_native, pdg_id, energy_MeV), true);
    }
    if (!detail::validView(view)) {
      return {TableLookupStatus::InvalidTableView, 0, 0.};
    }
    auto const particle =
        detail::findContinuousParticle(view, pdg_id);
    if (particle == view.continuous_particle_count) {
      return {TableLookupStatus::ContinuousParticleNotFound, 0, 0.};
    }
    std::uint32_t lower = 0;
    double fraction = 0.;
    auto const bracket = detail::continuousEnergyBracket(
        view, particle, energy_MeV, lower, fraction);
    if (bracket.status != TableLookupStatus::Success) {
      return bracket;
    }
    auto const offset = view.continuous_energy_offsets[particle];
    auto const value = detail::interpolatePositive(
        view.continuous_dEdX_MeV_cm2_per_g[offset + lower],
        view.continuous_dEdX_MeV_cm2_per_g[offset + lower + 1],
        fraction);
    if (!detail::finiteValue(value) || value <= 0.) {
      return {TableLookupStatus::InvalidTableView, 0, 0.};
    }
    return {TableLookupStatus::Success, 0, value};
  }

  CORSIKA_GPU_TABLE_HOST_DEVICE inline TableQueryResult queryContinuousMass(
      FlatRateTableView const& view, std::int32_t pdg_id) {
    if (view.physics_source == 1u) {
      auto const* utility = native_detail::findUtility(
          view.proposal_native, pdg_id);
      if (!utility)
        return {TableLookupStatus::ContinuousParticleNotFound, 0, 0.};
      if (!detail::finiteValue(utility->particle_mass_MeV) ||
          !(utility->particle_mass_MeV > 0.))
        return {TableLookupStatus::InvalidTableView, 0, 0.};
      return {TableLookupStatus::Success, 0,
              utility->particle_mass_MeV};
    }
    if (!detail::validView(view)) {
      return {TableLookupStatus::InvalidTableView, 0, 0.};
    }
    auto const particle =
        detail::findContinuousParticle(view, pdg_id);
    if (particle == view.continuous_particle_count) {
      return {TableLookupStatus::ContinuousParticleNotFound, 0, 0.};
    }
    auto const value = view.continuous_masses_MeV[particle];
    if (!detail::finiteValue(value) || value <= 0.) {
      return {TableLookupStatus::InvalidTableView, 0, 0.};
    }
    return {TableLookupStatus::Success, 0, value};
  }

  CORSIKA_GPU_TABLE_HOST_DEVICE inline TableQueryResult
  queryContinuousMinimumEnergy(
      FlatRateTableView const& view, std::int32_t pdg_id) {
    if (view.physics_source == 1u) {
      auto const* utility = native_detail::findUtility(
          view.proposal_native, pdg_id);
      if (!utility)
        return {TableLookupStatus::ContinuousParticleNotFound, 0, 0.};
      // PROPOSAL's native displacement interpolant starts at the particle
      // mass.  The scalar CORSIKA step limiter, however, stops at
      // mass + 0.9999 * the user-facing transport cut.  The legacy c8emrt
      // table stored that derived endpoint as its first energy; derive it
      // here so one native spline can serve different transport cuts.
      auto const transport_cut_MeV =
          (pdg_id == 13 || pdg_id == -13)
              ? view.muon_transport_cut_MeV
              : view.em_transport_cut_MeV;
      auto const minimum =
          utility->particle_mass_MeV +
          ContinuousCutSafetyFactor * transport_cut_MeV;
      if (!detail::finiteValue(utility->lower_energy_limit_MeV) ||
          utility->lower_energy_limit_MeV <
              utility->particle_mass_MeV ||
          !detail::finiteValue(transport_cut_MeV) ||
          !(transport_cut_MeV > 0.) ||
          !detail::finiteValue(minimum) ||
          !(minimum > utility->particle_mass_MeV) ||
          minimum < utility->lower_energy_limit_MeV ||
          minimum > utility->spline.axis.high)
        return {TableLookupStatus::InvalidTableView, 0, 0.};
      return {TableLookupStatus::Success, 0, minimum};
    }
    if (!detail::validView(view)) {
      return {TableLookupStatus::InvalidTableView, 0, 0.};
    }
    auto const particle =
        detail::findContinuousParticle(view, pdg_id);
    if (particle == view.continuous_particle_count) {
      return {TableLookupStatus::ContinuousParticleNotFound, 0, 0.};
    }
    auto const value =
        view.continuous_minimum_energies_MeV[particle];
    if (!detail::finiteValue(value) ||
        value <= view.continuous_masses_MeV[particle]) {
      return {TableLookupStatus::InvalidTableView, 0, 0.};
    }
    return {TableLookupStatus::Success, 0, value};
  }

  CORSIKA_GPU_TABLE_HOST_DEVICE inline TableQueryResult queryContinuousRange(
      FlatRateTableView const& view, std::int32_t pdg_id,
      double energy_MeV) {
    if (view.physics_source == 1u) {
      return detail::fromProposalNative(queryProposalNativeRange(
          view.proposal_native, pdg_id, energy_MeV), true);
    }
    if (!detail::validView(view)) {
      return {TableLookupStatus::InvalidTableView, 0, 0.};
    }
    auto const particle =
        detail::findContinuousParticle(view, pdg_id);
    if (particle == view.continuous_particle_count) {
      return {TableLookupStatus::ContinuousParticleNotFound, 0, 0.};
    }
    std::uint32_t lower = 0;
    double fraction = 0.;
    auto const bracket = detail::continuousEnergyBracket(
        view, particle, energy_MeV, lower, fraction);
    if (bracket.status != TableLookupStatus::Success) {
      return bracket;
    }
    auto const offset = view.continuous_energy_offsets[particle];
    auto const value =
        view.continuous_ranges_g_per_cm2[offset + lower] +
        fraction *
            (view.continuous_ranges_g_per_cm2[offset + lower + 1] -
             view.continuous_ranges_g_per_cm2[offset + lower]);
    if (!detail::finiteValue(value) || value < 0.) {
      return {TableLookupStatus::InvalidTableView, 0, 0.};
    }
    return {TableLookupStatus::Success, 0, value};
  }

  CORSIKA_GPU_TABLE_HOST_DEVICE inline TableQueryResult queryContinuousEnergy(
      FlatRateTableView const& view, std::int32_t pdg_id,
      double range_g_per_cm2) {
    if (view.physics_source == 1u) {
      auto result = detail::fromProposalNative(queryProposalNativeEnergy(
          view.proposal_native, pdg_id, range_g_per_cm2), true);
      if (result.status == TableLookupStatus::ContinuousEnergyOutOfRange)
        result.status = TableLookupStatus::ContinuousRangeOutOfRange;
      return result;
    }
    if (!detail::validView(view)) {
      return {TableLookupStatus::InvalidTableView, 0, 0.};
    }
    if (!detail::finiteValue(range_g_per_cm2)) {
      return {TableLookupStatus::NonFiniteInput, 0, 0.};
    }
    auto const particle =
        detail::findContinuousParticle(view, pdg_id);
    if (particle == view.continuous_particle_count) {
      return {TableLookupStatus::ContinuousParticleNotFound, 0, 0.};
    }
    auto const offset = view.continuous_energy_offsets[particle];
    auto const count = view.continuous_energy_counts[particle];
    if (count < 2 ||
        static_cast<std::uint64_t>(offset) + count >
            view.continuous_value_count) {
      return {TableLookupStatus::InvalidTableView, 0, 0.};
    }
    auto const* ranges = view.continuous_ranges_g_per_cm2 + offset;
    auto const* energies = view.continuous_energies_MeV + offset;
    if (range_g_per_cm2 < ranges[0] ||
        range_g_per_cm2 > ranges[count - 1]) {
      return {TableLookupStatus::ContinuousRangeOutOfRange, 0, 0.};
    }
    auto const upper = detail::lowerBound(
        ranges, count, range_g_per_cm2);
    if (upper == 0) {
      return {TableLookupStatus::Success, 0, energies[0]};
    }
    if (upper == count) {
      return {TableLookupStatus::Success, 0, energies[count - 1]};
    }
    if (ranges[upper] == range_g_per_cm2) {
      return {TableLookupStatus::Success, 0, energies[upper]};
    }
    auto const lower = upper - 1;
    auto const fraction =
        (range_g_per_cm2 - ranges[lower]) /
        (ranges[upper] - ranges[lower]);
    auto const value = detail::exponential(
        detail::logarithm(energies[lower]) +
        fraction *
            (detail::logarithm(energies[upper]) -
             detail::logarithm(energies[lower])));
    if (!detail::finiteValue(value) || value <= 0.) {
      return {TableLookupStatus::InvalidTableView, 0, 0.};
    }
    return {TableLookupStatus::Success, 0, value};
  }

  CORSIKA_GPU_TABLE_HOST_DEVICE inline TableQueryResult
  queryEnergyAfterContinuousLoss(
      FlatRateTableView const& view, std::int32_t pdg_id,
      double initial_energy_MeV, double grammage_g_per_cm2) {
    if (!detail::finiteValue(grammage_g_per_cm2) ||
        grammage_g_per_cm2 < 0.) {
      return {TableLookupStatus::NonFiniteInput, 0, 0.};
    }
    auto const initial_range =
        queryContinuousRange(view, pdg_id, initial_energy_MeV);
    if (initial_range.status != TableLookupStatus::Success) {
      return initial_range;
    }
    if (grammage_g_per_cm2 >= initial_range.value &&
        grammage_g_per_cm2 != 0.) {
      return {TableLookupStatus::TransportCutReached, 0, 0.};
    }
    if (view.physics_source == 1u)
      return detail::fromProposalNative(
          queryProposalNativeEnergyAfterContinuousLoss(
              view.proposal_native, pdg_id, initial_energy_MeV,
              grammage_g_per_cm2),
          true);
    return queryContinuousEnergy(
        view, pdg_id, initial_range.value - grammage_g_per_cm2);
  }

  CORSIKA_GPU_TABLE_HOST_DEVICE inline TableQueryResult executeTableQuery(
      FlatRateTableView const& view, TableQuery const& query) {
    if (query.kind == TableQueryKind::Rate) {
      return queryRate(view, query.pdg_id, query.process_id,
                       query.component_hash, query.energy_MeV);
    }
    if (query.kind == TableQueryKind::TotalRate) {
      return queryTotalRate(view, query.pdg_id, query.energy_MeV);
    }
    if (query.kind == TableQueryKind::LossFraction) {
      return queryLossFraction(
          view, query.pdg_id, query.process_id, query.component_hash,
          query.energy_MeV, query.quantile);
    }
    if (query.kind == TableQueryKind::ContinuousDedx) {
      return queryContinuousDedx(
          view, query.pdg_id, query.energy_MeV);
    }
    if (query.kind == TableQueryKind::ContinuousRange) {
      return queryContinuousRange(
          view, query.pdg_id, query.energy_MeV);
    }
    if (query.kind == TableQueryKind::ContinuousEnergy) {
      return queryContinuousEnergy(
          view, query.pdg_id, query.energy_MeV);
    }
    if (query.kind ==
        TableQueryKind::EnergyAfterContinuousLoss) {
      return queryEnergyAfterContinuousLoss(
          view, query.pdg_id, query.energy_MeV, query.quantile);
    }
    return {TableLookupStatus::InvalidQueryKind, 0, 0.};
  }

} // namespace corsika::gpu::em::tables

#undef CORSIKA_GPU_TABLE_HOST_DEVICE

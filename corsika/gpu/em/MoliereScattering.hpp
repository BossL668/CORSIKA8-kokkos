/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <stdexcept>
#include <type_traits>

#include <corsika/accelerator/AcceleratorMacros.hpp>
#include <corsika/gpu/em/tables/RateTable.hpp>

#define CORSIKA_GPU_MOLIERE_HOST_DEVICE C8_ACCELERATOR_INLINE_FUNCTION

namespace corsika::gpu::em {

  inline constexpr std::size_t MaxMoliereComponents = 16;
  inline constexpr std::size_t MoliereSeriesCoefficientCount = 70;
  inline constexpr std::size_t MoliereLargeSeriesCoefficientCount = 13;
  inline constexpr std::size_t MoliereLargeIntegralCoefficientCount = 15;
  inline constexpr std::size_t MoliereInterpolationNodeCount = 100;
  inline constexpr std::size_t MoliereInterpolationIntervalCount =
      MoliereInterpolationNodeCount - 1;
  inline constexpr std::size_t MoliereInterpolationFunctionCount = 4;
  inline constexpr double MoliereInterpolationLower = 0.;
  inline constexpr double MoliereInterpolationUpper = 20.;
  inline constexpr std::size_t MoliereInitialGuessBNodeCount = 64;
  inline constexpr std::size_t
      MoliereInitialGuessBetaSquaredNodeCount = 8;
  inline constexpr std::size_t
      MoliereInitialGuessCoordinateNodeCount = 128;
  inline constexpr double MoliereInitialGuessBLower = 4.5;
  inline constexpr double MoliereInitialGuessBUpper = 64.;
  inline constexpr double
      MoliereInitialGuessBetaSquaredLower = 0.25;
  inline constexpr double
      MoliereInitialGuessBetaSquaredUpper = 1.;
  inline constexpr double MoliereInitialGuessCoordinateLower = 0.;
  inline constexpr double MoliereInitialGuessCoordinateUpper = 5.;
  inline constexpr double MoliereMinimumB = 4.5;
  inline constexpr double MoliereMinimumBEquationRightHandSide =
      2.9959226032237258; // 4.5 - log(4.5)

  struct MoliereComponentSnapshot {
    double nuclear_charge{};
    double weight_zz{};
    // Static screening factors derived once from the versioned medium.
    double chi_0_squared_MeV2{};
    double coulomb_correction{};
  };

  /**
   * Pointer-free state required by PROPOSAL 7.6.2 Moliere.
   *
   * The coefficient arrays are copied from the linked PROPOSAL version by the
   * offline table generator. Keeping them in versioned data avoids embedding
   * a second, silently diverging copy of PROPOSAL's numerical
   * parametrization in the CUDA source.
   */
  struct MoliereSnapshot {
    double particle_mass_MeV{};
    double electron_mass_MeV{};
    double fine_structure_constant{};
    double avogadro_per_mol{};
    double hbar_MeV_s{};
    double speed_of_light_cm_per_s{};
    double z_squared_over_a_average{};
    double inverse_weight_zz_sum{};
    double euler_mascheroni{};
    std::uint32_t component_count{};
    std::uint32_t maximum_weight_index{};
    MoliereComponentSnapshot
        components[MaxMoliereComponents]{};
    double c1[MoliereSeriesCoefficientCount]{};
    double c2[MoliereSeriesCoefficientCount]{};
    double c2_large[MoliereLargeSeriesCoefficientCount]{};
    double s2_large[MoliereLargeSeriesCoefficientCount]{};
    double C1_large[MoliereLargeIntegralCoefficientCount]{};
  };

  /**
   * One cubic polynomial in the normalized interval coordinate q in [0, 1].
   *
   * evaluate(q) = c[0] + q * (c[1] + q * (c[2] + q * c[3]))
   */
  struct MoliereCubicPolynomial {
    double c[4]{};
  };

  /**
   * Host-owned coefficients derived from PROPOSAL's versioned analytical
   * state using the same CubicInterpolation cardinal B-spline implementation
   * as PROPOSAL::MoliereInterpol.
   */
  struct MoliereInterpolationTable {
    std::array<
        MoliereCubicPolynomial,
        MoliereInterpolationFunctionCount *
            MoliereInterpolationIntervalCount>
        polynomials{};
    /**
     * Delta from the Gaussian coordinate to the snapshot-specific
     * multi-component Moliere inverse, tabulated in reference B, beta^2 and
     * quantile coordinate. It is only a Newton initial guess; the full
     * multi-component PDF/CDF still determines the returned angle.
     */
    std::array<
        double,
        MoliereInitialGuessBNodeCount *
            MoliereInitialGuessBetaSquaredNodeCount *
            MoliereInitialGuessCoordinateNodeCount>
        initial_guess_delta{};
  };

  /**
   * Non-owning device view. The four function arrays are kept separate in the
   * logical interface so a kernel only reads the cache lines it actually
   * needs during density or cumulative evaluations.
   */
  struct MoliereInterpolationView {
    MoliereCubicPolynomial const* f1{};
    MoliereCubicPolynomial const* f2{};
    MoliereCubicPolynomial const* F1{};
    MoliereCubicPolynomial const* F2{};
    double const* initial_guess_delta{};
    std::uint32_t interval_count{};
    std::uint32_t initial_guess_b_node_count{};
    std::uint32_t initial_guess_beta_squared_node_count{};
    std::uint32_t initial_guess_coordinate_node_count{};
    std::uint32_t reserved{};
    double lower{};
    double upper{};
    double inverse_step{};
    double initial_guess_b_lower{};
    double initial_guess_b_upper{};
    double initial_guess_inverse_b_step{};
    double initial_guess_beta_squared_lower{};
    double initial_guess_beta_squared_upper{};
    double initial_guess_inverse_beta_squared_step{};
    double initial_guess_coordinate_lower{};
    double initial_guess_coordinate_upper{};
    double initial_guess_inverse_coordinate_step{};
  };

  enum class MoliereStatus : std::uint32_t {
    Success = 0,
    NoDeflection = 1,
    InvalidSnapshot = 2,
    InvalidInput = 3,
    NonConvergent = 4,
    NonFiniteResult = 5,
  };

  struct MoliereAngleResult {
    MoliereStatus status{MoliereStatus::InvalidSnapshot};
    std::uint32_t iterations{};
    double angle_rad{};
  };

  struct MoliereDirectionResult {
    MoliereStatus status{MoliereStatus::InvalidInput};
    std::uint32_t reserved{};
    double direction[3]{};
  };

  static_assert(std::is_standard_layout_v<MoliereSnapshot>);
  static_assert(std::is_trivially_copyable_v<MoliereSnapshot>);
  static_assert(std::is_standard_layout_v<MoliereCubicPolynomial>);
  static_assert(std::is_trivially_copyable_v<MoliereCubicPolynomial>);
  static_assert(std::is_standard_layout_v<MoliereInterpolationTable>);
  static_assert(std::is_trivially_copyable_v<MoliereInterpolationTable>);
  static_assert(
      offsetof(MoliereInterpolationTable, polynomials) == 0);
  static_assert(std::is_standard_layout_v<MoliereInterpolationView>);
  static_assert(std::is_trivially_copyable_v<MoliereInterpolationView>);
  static_assert(std::is_standard_layout_v<MoliereAngleResult>);
  static_assert(std::is_trivially_copyable_v<MoliereAngleResult>);
  static_assert(std::is_standard_layout_v<MoliereDirectionResult>);
  static_assert(std::is_trivially_copyable_v<MoliereDirectionResult>);

  namespace moliere_detail {

    inline constexpr double Pi =
        3.141592653589793238462643383279502884;
    inline constexpr double SqrtTwo =
        1.414213562373095048801688724209698079;

    CORSIKA_GPU_MOLIERE_HOST_DEVICE inline bool finite(
        double value) {
#if defined(__CUDA_ARCH__)
      return ::isfinite(value);
#else
      return std::isfinite(value);
#endif
    }

    CORSIKA_GPU_MOLIERE_HOST_DEVICE inline double absolute(
        double value) {
#if defined(__CUDA_ARCH__)
      return ::fabs(value);
#else
      return std::abs(value);
#endif
    }

    CORSIKA_GPU_MOLIERE_HOST_DEVICE inline double squareRoot(
        double value) {
#if defined(__CUDA_ARCH__)
      return ::sqrt(value);
#else
      return std::sqrt(value);
#endif
    }

    CORSIKA_GPU_MOLIERE_HOST_DEVICE inline double logarithm(
        double value) {
#if defined(__CUDA_ARCH__)
      return ::log(value);
#else
      return std::log(value);
#endif
    }

    CORSIKA_GPU_MOLIERE_HOST_DEVICE inline double exponential(
        double value) {
#if defined(__CUDA_ARCH__)
      return ::exp(value);
#else
      return std::exp(value);
#endif
    }

    CORSIKA_GPU_MOLIERE_HOST_DEVICE inline double power(
        double base, double exponent) {
#if defined(__CUDA_ARCH__)
      return ::pow(base, exponent);
#else
      return std::pow(base, exponent);
#endif
    }

    CORSIKA_GPU_MOLIERE_HOST_DEVICE inline double errorFunction(
        double value) {
#if defined(__CUDA_ARCH__)
      return ::erf(value);
#else
      return std::erf(value);
#endif
    }

    CORSIKA_GPU_MOLIERE_HOST_DEVICE inline bool validSnapshot(
        MoliereSnapshot const& snapshot) {
      if (snapshot.component_count == 0 ||
          snapshot.component_count > MaxMoliereComponents ||
          snapshot.maximum_weight_index >=
              snapshot.component_count ||
          !finite(snapshot.particle_mass_MeV) ||
          !(snapshot.particle_mass_MeV > 0.) ||
          !finite(snapshot.electron_mass_MeV) ||
          !(snapshot.electron_mass_MeV > 0.) ||
          !finite(snapshot.fine_structure_constant) ||
          !(snapshot.fine_structure_constant > 0.) ||
          !finite(snapshot.avogadro_per_mol) ||
          !(snapshot.avogadro_per_mol > 0.) ||
          !finite(snapshot.hbar_MeV_s) ||
          !(snapshot.hbar_MeV_s > 0.) ||
          !finite(snapshot.speed_of_light_cm_per_s) ||
          !(snapshot.speed_of_light_cm_per_s > 0.) ||
          !finite(snapshot.z_squared_over_a_average) ||
          !(snapshot.z_squared_over_a_average > 0.) ||
          !finite(snapshot.inverse_weight_zz_sum) ||
          !(snapshot.inverse_weight_zz_sum > 0.) ||
          !finite(snapshot.euler_mascheroni)) {
        return false;
      }
      for (std::uint32_t index = 0;
           index < snapshot.component_count; ++index) {
        if (!finite(snapshot.components[index].nuclear_charge) ||
            !(snapshot.components[index].nuclear_charge > 0.) ||
          !finite(snapshot.components[index].weight_zz) ||
            !(snapshot.components[index].weight_zz > 0.) ||
            !finite(
                snapshot.components[index]
                    .chi_0_squared_MeV2) ||
            !(snapshot.components[index]
                  .chi_0_squared_MeV2 > 0.) ||
            !finite(
                snapshot.components[index]
                    .coulomb_correction) ||
            !(snapshot.components[index]
                  .coulomb_correction > 0.)) {
          return false;
        }
      }
      for (std::size_t index = 0;
           index < MoliereSeriesCoefficientCount; ++index) {
        if (!finite(snapshot.c1[index]) ||
            !finite(snapshot.c2[index])) {
          return false;
        }
      }
      for (std::size_t index = 0;
           index < MoliereLargeSeriesCoefficientCount; ++index) {
        if (!finite(snapshot.c2_large[index]) ||
            !finite(snapshot.s2_large[index])) {
          return false;
        }
      }
      for (double coefficient : snapshot.C1_large) {
        if (!finite(coefficient)) {
          return false;
        }
      }
      return true;
    }

    CORSIKA_GPU_MOLIERE_HOST_DEVICE inline bool
    validInterpolationView(
        MoliereInterpolationView const& interpolation) {
      return interpolation.f1 != nullptr &&
             interpolation.f2 != nullptr &&
             interpolation.F1 != nullptr &&
             interpolation.F2 != nullptr &&
             interpolation.interval_count ==
                 MoliereInterpolationIntervalCount &&
             finite(interpolation.lower) &&
             finite(interpolation.upper) &&
             finite(interpolation.inverse_step) &&
             interpolation.lower == MoliereInterpolationLower &&
             interpolation.upper == MoliereInterpolationUpper &&
             interpolation.inverse_step > 0. &&
             (interpolation.initial_guess_delta == nullptr ||
              (interpolation.initial_guess_b_node_count ==
                       MoliereInitialGuessBNodeCount &&
               interpolation
                       .initial_guess_beta_squared_node_count ==
                   MoliereInitialGuessBetaSquaredNodeCount &&
               interpolation
                       .initial_guess_coordinate_node_count ==
                   MoliereInitialGuessCoordinateNodeCount &&
               finite(interpolation.initial_guess_b_lower) &&
               finite(interpolation.initial_guess_b_upper) &&
               finite(
                   interpolation.initial_guess_inverse_b_step) &&
               finite(
                   interpolation
                       .initial_guess_beta_squared_lower) &&
               finite(
                   interpolation
                       .initial_guess_beta_squared_upper) &&
               finite(
                   interpolation
                       .initial_guess_inverse_beta_squared_step) &&
               finite(
                   interpolation
                       .initial_guess_coordinate_lower) &&
               finite(
                   interpolation
                       .initial_guess_coordinate_upper) &&
               finite(
                   interpolation
                       .initial_guess_inverse_coordinate_step) &&
               interpolation.initial_guess_b_lower ==
                   MoliereInitialGuessBLower &&
               interpolation.initial_guess_b_upper ==
                   MoliereInitialGuessBUpper &&
               interpolation.initial_guess_inverse_b_step > 0. &&
               interpolation.initial_guess_beta_squared_lower ==
                   MoliereInitialGuessBetaSquaredLower &&
               interpolation.initial_guess_beta_squared_upper ==
                   MoliereInitialGuessBetaSquaredUpper &&
               interpolation
                       .initial_guess_inverse_beta_squared_step >
                   0. &&
               interpolation.initial_guess_coordinate_lower ==
                   MoliereInitialGuessCoordinateLower &&
               interpolation.initial_guess_coordinate_upper ==
                   MoliereInitialGuessCoordinateUpper &&
               interpolation
                       .initial_guess_inverse_coordinate_step >
                   0.));
    }

    CORSIKA_GPU_MOLIERE_HOST_DEVICE inline bool
    hasInterpolationView(
        MoliereInterpolationView const& interpolation) {
      return interpolation.f1 != nullptr ||
             interpolation.f2 != nullptr ||
             interpolation.F1 != nullptr ||
             interpolation.F2 != nullptr ||
             interpolation.interval_count != 0;
    }

    CORSIKA_GPU_MOLIERE_HOST_DEVICE inline double
    evaluateInterpolation(
        MoliereCubicPolynomial const* polynomials,
        MoliereInterpolationView const& interpolation,
        double x) {
      auto coordinate =
          (x - interpolation.lower) *
          interpolation.inverse_step;
      if (coordinate < 0.) {
        coordinate = 0.;
      }
      auto const maximum_coordinate =
          static_cast<double>(interpolation.interval_count);
      if (coordinate > maximum_coordinate) {
        coordinate = maximum_coordinate;
      }
      auto interval =
          static_cast<std::uint32_t>(coordinate);
      if (interval >= interpolation.interval_count) {
        interval = interpolation.interval_count - 1;
      }
      auto const q =
          coordinate - static_cast<double>(interval);
      auto const& polynomial = polynomials[interval];
      return polynomial.c[0] +
             q *
                 (polynomial.c[1] +
                  q *
                      (polynomial.c[2] +
                       q * polynomial.c[3]));
    }

    CORSIKA_GPU_MOLIERE_HOST_DEVICE inline double
    improveMoliereInitialGuess(
        MoliereInterpolationView const& interpolation,
        double reference_B, double beta_squared,
        double gaussian_coordinate) {
      if (interpolation.initial_guess_delta == nullptr ||
          !finite(reference_B) ||
          !finite(beta_squared) ||
          !finite(gaussian_coordinate)) {
        return gaussian_coordinate;
      }
      auto const magnitude = absolute(gaussian_coordinate);
      if (reference_B < interpolation.initial_guess_b_lower ||
          reference_B > interpolation.initial_guess_b_upper ||
          beta_squared <
              interpolation.initial_guess_beta_squared_lower ||
          beta_squared >
              interpolation.initial_guess_beta_squared_upper ||
          magnitude <
              interpolation.initial_guess_coordinate_lower ||
          magnitude >
              interpolation.initial_guess_coordinate_upper) {
        return gaussian_coordinate;
      }
      auto b_coordinate =
          (reference_B -
           interpolation.initial_guess_b_lower) *
          interpolation.initial_guess_inverse_b_step;
      auto coordinate =
          (magnitude -
           interpolation.initial_guess_coordinate_lower) *
          interpolation
              .initial_guess_inverse_coordinate_step;
      auto beta_coordinate =
          (beta_squared -
           interpolation.initial_guess_beta_squared_lower) *
          interpolation
              .initial_guess_inverse_beta_squared_step;
      auto b_index =
          static_cast<std::uint32_t>(b_coordinate);
      auto beta_index =
          static_cast<std::uint32_t>(beta_coordinate);
      auto coordinate_index =
          static_cast<std::uint32_t>(coordinate);
      if (b_index + 1 >=
          interpolation.initial_guess_b_node_count) {
        b_index =
            interpolation.initial_guess_b_node_count - 2;
      }
      if (beta_index + 1 >=
          interpolation
              .initial_guess_beta_squared_node_count) {
        beta_index =
            interpolation
                .initial_guess_beta_squared_node_count -
            2;
      }
      if (coordinate_index + 1 >=
          interpolation
              .initial_guess_coordinate_node_count) {
        coordinate_index =
            interpolation
                .initial_guess_coordinate_node_count -
            2;
      }
      auto const b_fraction =
          b_coordinate - static_cast<double>(b_index);
      auto const beta_fraction =
          beta_coordinate -
          static_cast<double>(beta_index);
      auto const coordinate_fraction =
          coordinate -
          static_cast<double>(coordinate_index);
      auto const coordinate_stride =
          interpolation.initial_guess_coordinate_node_count;
      auto const beta_stride =
          static_cast<std::size_t>(
              interpolation
                  .initial_guess_beta_squared_node_count) *
          coordinate_stride;
      auto const lower_b_offset =
          static_cast<std::size_t>(b_index) * beta_stride;
      auto const upper_b_offset =
          lower_b_offset + beta_stride;
      auto const lower_beta_offset =
          static_cast<std::size_t>(beta_index) *
              coordinate_stride +
          coordinate_index;
      auto const upper_beta_offset =
          lower_beta_offset + coordinate_stride;
      auto const lower_b_lower_beta_offset =
          lower_b_offset + lower_beta_offset;
      auto const lower_b_upper_beta_offset =
          lower_b_offset + upper_beta_offset;
      auto const upper_b_lower_beta_offset =
          upper_b_offset + lower_beta_offset;
      auto const upper_b_upper_beta_offset =
          upper_b_offset + upper_beta_offset;
      auto const lower_b_lower_beta =
          interpolation.initial_guess_delta[
              lower_b_lower_beta_offset] *
              (1. - coordinate_fraction) +
          interpolation.initial_guess_delta[
              lower_b_lower_beta_offset + 1] *
              coordinate_fraction;
      auto const lower_b_upper_beta =
          interpolation.initial_guess_delta[
              lower_b_upper_beta_offset] *
              (1. - coordinate_fraction) +
          interpolation.initial_guess_delta[
              lower_b_upper_beta_offset + 1] *
              coordinate_fraction;
      auto const upper_b_lower_beta =
          interpolation.initial_guess_delta[
              upper_b_lower_beta_offset] *
              (1. - coordinate_fraction) +
          interpolation.initial_guess_delta[
              upper_b_lower_beta_offset + 1] *
              coordinate_fraction;
      auto const upper_b_upper_beta =
          interpolation.initial_guess_delta[
              upper_b_upper_beta_offset] *
              (1. - coordinate_fraction) +
          interpolation.initial_guess_delta[
              upper_b_upper_beta_offset + 1] *
              coordinate_fraction;
      auto const lower_b =
          lower_b_lower_beta * (1. - beta_fraction) +
          lower_b_upper_beta * beta_fraction;
      auto const upper_b =
          upper_b_lower_beta * (1. - beta_fraction) +
          upper_b_upper_beta * beta_fraction;
      auto const corrected_magnitude =
          magnitude +
          lower_b * (1. - b_fraction) +
          upper_b * b_fraction;
      if (!finite(corrected_magnitude) ||
          corrected_magnitude < 0.) {
        return gaussian_coordinate;
      }
      return gaussian_coordinate < 0.
                 ? -corrected_magnitude
                 : corrected_magnitude;
    }

    CORSIKA_GPU_MOLIERE_HOST_DEVICE inline double normalPpf(
        double probability) {
      constexpr double a[] = {
          -3.969683028665376e+01, 2.209460984245205e+02,
          -2.759285104469687e+02, 1.383577518672690e+02,
          -3.066479806614716e+01, 2.506628277459239e+00};
      constexpr double b[] = {
          -5.447609879822406e+01, 1.615858368580409e+02,
          -1.556989798598866e+02, 6.680131188771972e+01,
          -1.328068155288572e+01};
      constexpr double c[] = {
          -7.784894002430293e-03, -3.223964580411365e-01,
          -2.400758277161838e+00, -2.549732539343734e+00,
          4.374664141464968e+00, 2.938163982698783e+00};
      constexpr double d[] = {
          7.784695709041462e-03, 3.224671290700398e-01,
          2.445134137142996e+00, 3.754408661907416e+00};
      constexpr double p_low = 0.02425;
      constexpr double p_high = 1. - p_low;
      double q = 0.;
      double r = 0.;
      double x = 0.;
      if (probability < p_low) {
        q = squareRoot(-2. * logarithm(probability));
        x =
            (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) *
                       q +
                   c[4]) *
                      q +
                  c[5]) /
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) *
                 q +
             1.);
      } else if (probability <= p_high) {
        q = probability - 0.5;
        r = q * q;
        x =
            (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) *
                       r +
                   a[4]) *
                      r +
                  a[5]) *
            q /
            (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) *
                      r +
                  b[4]) *
                     r +
                 1.);
      } else {
        q = squareRoot(-2. * logarithm(1. - probability));
        x =
            -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) *
                        q +
                    c[4]) *
                       q +
                   c[5]) /
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) *
                 q +
             1.);
      }
      // This quantile seeds the exact Moliere Newton solve; it is not the
      // returned variate. Acklam's rational approximation is already far
      // inside that solver's basin, so PROPOSAL's final Halley refinement
      // (one erfc and one exp) only improves a disposable initial guess.
      return x;
    }

    /**
     * Solve B - log(B) = rhs on the physical B > 1 branch.
     *
     * PROPOSAL starts at B=15 and performs six Newton updates. Here the
     * fourth-order asymptotic expansion of -W_{-1}(-exp(-rhs)) supplies an
     * accurate initial value, followed by one cubic Halley update. Across
     * rhs >= 4.5-log(4.5), this agrees with the converged branch at roughly
     * 1e-12 relative accuracy while using two logarithms instead of six.
     */
    CORSIKA_GPU_MOLIERE_HOST_DEVICE inline double
    solveMoliereB(double rhs) {
      if (!finite(rhs) ||
          rhs < MoliereMinimumBEquationRightHandSide) {
        return 0.;
      }
      auto const logarithm_rhs = logarithm(rhs);
      auto const inverse_rhs = 1. / rhs;
      auto const inverse_rhs_squared =
          inverse_rhs * inverse_rhs;
      auto const logarithm_rhs_squared =
          logarithm_rhs * logarithm_rhs;
      auto const logarithm_rhs_cubed =
          logarithm_rhs_squared * logarithm_rhs;
      auto current =
          rhs + logarithm_rhs +
          logarithm_rhs * inverse_rhs +
          logarithm_rhs *
              (2. - logarithm_rhs) * 0.5 *
              inverse_rhs_squared +
          logarithm_rhs *
              (6. - 9. * logarithm_rhs +
               2. * logarithm_rhs_squared) /
              6. * inverse_rhs_squared * inverse_rhs +
          logarithm_rhs *
              (12. - 36. * logarithm_rhs +
               22. * logarithm_rhs_squared -
               3. * logarithm_rhs_cubed) /
              12. * inverse_rhs_squared *
              inverse_rhs_squared;
      if (!finite(current) || !(current > 1.)) {
        return 0.;
      }
      auto const function =
          current - logarithm(current) - rhs;
      auto const inverse_current = 1. / current;
      auto const derivative = 1. - inverse_current;
      auto const denominator =
          2. * derivative * derivative -
          function * inverse_current * inverse_current;
      if (!finite(denominator) || denominator == 0.) {
        return 0.;
      }
      current -=
          2. * function * derivative / denominator;
      return finite(current) &&
                     current >= MoliereMinimumB
                 ? current
                 : 0.;
    }

    CORSIKA_GPU_MOLIERE_HOST_DEVICE inline double f1M(
        MoliereSnapshot const& snapshot, double x) {
      if (x > 12.) {
        return 0.5 * squareRoot(Pi) /
               (power(x, 1.5) *
                power(1. - 4.5 / x, 2. / 3.));
      }
      auto sum = snapshot.c1[69];
      for (int index = 68; index >= 0; --index) {
        sum = sum * x + snapshot.c1[index];
      }
      return sum;
    }

    CORSIKA_GPU_MOLIERE_HOST_DEVICE inline double f2M(
        MoliereSnapshot const& snapshot, double x) {
      if (x > 4.25 * 4.25) {
        constexpr double a =
            0.00013567765224589459194769192063035;
        constexpr double b =
            -0.0022635502525409950842771866774683;
        constexpr double c =
            0.0098037758070269476889935233998585;
        if (x <= 6.5 * 6.5) {
          return a * x + b * squareRoot(x) + c;
        }
        double sum = 0.;
        for (int index = 2; index < 13; ++index) {
          sum +=
              snapshot.c2_large[index] *
              (0.5 * logarithm(x) +
               snapshot.s2_large[index]) *
              power(x, -(index + 0.5));
        }
        return sum;
      }
      auto sum = snapshot.c2[69];
      for (int index = 68; index >= 0; --index) {
        sum = sum * x + snapshot.c2[index];
      }
      return sum;
    }

    CORSIKA_GPU_MOLIERE_HOST_DEVICE inline double F1M(
        MoliereSnapshot const& snapshot, double x) {
      if (x > 12.) {
        auto sum = snapshot.C1_large[14];
        for (int index = 13; index >= 0; --index) {
          sum = snapshot.C1_large[index] + sum / x;
        }
        return sum;
      }
      auto sum = snapshot.c1[69] / (2. * 69. + 1.);
      for (int index = 68; index >= 0; --index) {
        sum =
            sum * x +
            snapshot.c1[index] / (2. * index + 1.);
      }
      return sum * squareRoot(x > 0. ? x : 0.);
    }

    CORSIKA_GPU_MOLIERE_HOST_DEVICE inline double F2M(
        MoliereSnapshot const& snapshot, double x) {
      if (x > 4.25 * 4.25) {
        constexpr double a =
            -0.00026360133958801203364619158975302;
        constexpr double b =
            0.0039965027465457608410459577896745;
        constexpr double c =
            -0.016305842044996649714549974419242;
        if (x <= 6.5 * 6.5) {
          return a * x + b * squareRoot(x) + c;
        }
        double sum = 0.;
        for (int index = 2; index < 13; ++index) {
          sum +=
              -0.5 * snapshot.c2_large[index] / index *
              (0.5 / index + 0.5 * logarithm(x) +
               snapshot.s2_large[index]) *
              power(x, -index);
        }
        return sum;
      }
      auto sum = snapshot.c2[69] / (2. * 69. + 1.);
      for (int index = 68; index >= 0; --index) {
        sum =
            sum * x +
            snapshot.c2[index] / (2. * index + 1.);
      }
      return sum * squareRoot(x > 0. ? x : 0.);
    }

    CORSIKA_GPU_MOLIERE_HOST_DEVICE inline double f1M(
        MoliereSnapshot const& snapshot,
        MoliereInterpolationView const& interpolation,
        double x) {
      if (interpolation.f1 != nullptr &&
          x <= MoliereInterpolationUpper) {
        return evaluateInterpolation(
            interpolation.f1, interpolation, x);
      }
      return f1M(snapshot, x);
    }

    CORSIKA_GPU_MOLIERE_HOST_DEVICE inline double f2M(
        MoliereSnapshot const& snapshot,
        MoliereInterpolationView const& interpolation,
        double x) {
      if (interpolation.f2 != nullptr &&
          x <= MoliereInterpolationUpper) {
        return evaluateInterpolation(
            interpolation.f2, interpolation, x);
      }
      return f2M(snapshot, x);
    }

    CORSIKA_GPU_MOLIERE_HOST_DEVICE inline double F1M(
        MoliereSnapshot const& snapshot,
        MoliereInterpolationView const& interpolation,
        double x) {
      if (interpolation.F1 != nullptr &&
          x <= MoliereInterpolationUpper) {
        return evaluateInterpolation(
            interpolation.F1, interpolation, x);
      }
      return F1M(snapshot, x);
    }

    CORSIKA_GPU_MOLIERE_HOST_DEVICE inline double F2M(
        MoliereSnapshot const& snapshot,
        MoliereInterpolationView const& interpolation,
        double x) {
      if (interpolation.F2 != nullptr &&
          x <= MoliereInterpolationUpper) {
        return evaluateInterpolation(
            interpolation.F2, interpolation, x);
      }
      return F2M(snapshot, x);
    }

    template <std::size_t ComponentCapacity>
    struct DistributionStateForCapacity {
      static_assert(ComponentCapacity > 0);
      static_assert(ComponentCapacity <= MaxMoliereComponents);
      /**
       * Per-step values reused by every Newton evaluation.
       *
       * PROPOSAL's scalar implementation recomputes theta^2/(chi_c^2 B),
       * 1/B and weight/sqrt(pi chi_c^2 B) independently in f(theta) and
       * F(theta). A GPU lepton evaluates both functions several times for
       * each of two projected angles, so keeping these invariant factors in
       * the distribution state removes repeated square roots and divisions
       * without changing the Moliere density or CDF.
       */
      double inverse_scale[ComponentCapacity]{};
      double inverse_B[ComponentCapacity]{};
      double density_weight[ComponentCapacity]{};
      double prefactor{};
      double beta_squared{};
    };

    using DistributionState =
        DistributionStateForCapacity<MaxMoliereComponents>;

    template <std::size_t ComponentCapacity>
    CORSIKA_GPU_MOLIERE_HOST_DEVICE inline MoliereStatus
    buildDistribution(
        MoliereSnapshot const& snapshot, double grammage_g_per_cm2,
        double initial_energy_MeV,
        DistributionStateForCapacity<ComponentCapacity>& state,
        bool validate_snapshot = true) {
      if (snapshot.component_count == 0 ||
          snapshot.component_count > ComponentCapacity) {
        return MoliereStatus::InvalidSnapshot;
      }
      if (validate_snapshot && !validSnapshot(snapshot)) {
        return MoliereStatus::InvalidSnapshot;
      }
      if (!finite(grammage_g_per_cm2) ||
          grammage_g_per_cm2 < 0. ||
          !finite(initial_energy_MeV) ||
          !(initial_energy_MeV >
            snapshot.particle_mass_MeV)) {
        return MoliereStatus::InvalidInput;
      }
      if (grammage_g_per_cm2 == 0.) {
        return MoliereStatus::NoDeflection;
      }
      auto const momentum_squared =
          (initial_energy_MeV -
           snapshot.particle_mass_MeV) *
          (initial_energy_MeV +
           snapshot.particle_mass_MeV);
      auto const beta_squared =
          1. /
          (1. +
           snapshot.particle_mass_MeV *
               snapshot.particle_mass_MeV /
               momentum_squared);
      state.beta_squared = beta_squared;
      auto beta_p_squared =
          momentum_squared / initial_energy_MeV;
      beta_p_squared *= beta_p_squared;
      auto const chi_c_squared =
          4. * Pi * snapshot.avogadro_per_mol *
          snapshot.fine_structure_constant *
          snapshot.fine_structure_constant *
          snapshot.hbar_MeV_s * snapshot.hbar_MeV_s *
          snapshot.speed_of_light_cm_per_s *
          snapshot.speed_of_light_cm_per_s *
          grammage_g_per_cm2 / beta_p_squared *
          snapshot.z_squared_over_a_average;
      if (!finite(chi_c_squared) ||
          !(chi_c_squared > 0.)) {
        return MoliereStatus::NonFiniteResult;
      }
      double maximum_weight_B = 0.;
      for (std::uint32_t index = 0;
           index < snapshot.component_count; ++index) {
        auto const chi_a_squared =
            snapshot.components[index]
                .chi_0_squared_MeV2 /
            momentum_squared *
            (1.13 +
             snapshot.components[index]
                     .coulomb_correction /
                 beta_squared);
        auto const log_chi_ratio =
            logarithm(chi_c_squared / chi_a_squared);
        auto const current = solveMoliereB(
            log_chi_ratio + 1. -
            2. * snapshot.euler_mascheroni);
        if (!(current > 0.)) {
          return MoliereStatus::NoDeflection;
        }
        auto const scale = chi_c_squared * current;
        if (!finite(scale) || !(scale > 0.)) {
          return MoliereStatus::NonFiniteResult;
        }
        state.inverse_scale[index] = 1. / scale;
        state.inverse_B[index] = 1. / current;
        state.density_weight[index] =
            snapshot.components[index].weight_zz /
            squareRoot(scale * Pi);
        if (!finite(state.inverse_scale[index]) ||
            !(state.inverse_scale[index] > 0.) ||
            !finite(state.inverse_B[index]) ||
            !(state.inverse_B[index] > 0.) ||
            !finite(state.density_weight[index]) ||
            !(state.density_weight[index] > 0.)) {
          return MoliereStatus::NonFiniteResult;
        }
        if (index == snapshot.maximum_weight_index) {
          maximum_weight_B = current;
        }
      }
      state.prefactor = squareRoot(
          chi_c_squared * maximum_weight_B);
      return finite(state.prefactor) &&
                     state.prefactor > 0.
                 ? MoliereStatus::Success
                 : MoliereStatus::NonFiniteResult;
    }

    template <std::size_t ComponentCapacity>
    struct DistributionEvaluation {
      double density{};
      double cumulative{};
    };

    /**
     * Evaluate f(theta) and F(theta) in one component traversal.
     *
     * Newton needs both at the same theta. Keeping the two independent
     * functions caused every iteration to calculate the normalized x and
     * reload the component state twice. The two sums remain independent and
     * retain PROPOSAL's component order.
     */
    template <std::size_t ComponentCapacity>
    CORSIKA_GPU_MOLIERE_HOST_DEVICE inline
    DistributionEvaluation<ComponentCapacity>
    evaluateDistribution(
        MoliereSnapshot const& snapshot,
        MoliereInterpolationView const& interpolation,
        DistributionStateForCapacity<ComponentCapacity> const& state,
        double theta) {
      double density_result = 0.;
      double cumulative_result = 0.;
      auto const theta_squared = theta * theta;
      for (std::uint32_t index = 0;
           index < snapshot.component_count; ++index) {
        auto const x =
            theta_squared * state.inverse_scale[index];
        auto const inverse_B = state.inverse_B[index];
        auto const inverse_B_squared =
            inverse_B * inverse_B;
        density_result +=
            state.density_weight[index] *
            (exponential(-x) +
             f1M(snapshot, interpolation, x) * inverse_B +
             f2M(snapshot, interpolation, x) *
                 inverse_B_squared);
        cumulative_result +=
            snapshot.components[index].weight_zz *
            (0.5 *
                 errorFunction(
                     squareRoot(x > 0. ? x : 0.)) +
             squareRoot(1. / Pi) *
                 (F1M(snapshot, interpolation, x) * inverse_B +
                  F2M(snapshot, interpolation, x) *
                      inverse_B_squared));
      }
      density_result *= snapshot.inverse_weight_zz_sum;
      cumulative_result *= snapshot.inverse_weight_zz_sum;
      return {
          density_result,
          theta < 0. ? -cumulative_result
                     : cumulative_result};
    }

    template <std::size_t ComponentCapacity>
    CORSIKA_GPU_MOLIERE_HOST_DEVICE inline MoliereAngleResult
    sampleOneDimensional(
        MoliereSnapshot const& snapshot,
        MoliereInterpolationView const& interpolation,
        DistributionStateForCapacity<ComponentCapacity> const& state,
        double uniform) {
      if (!finite(uniform) || !(uniform > 0.) ||
          !(uniform < 1.)) {
        return {MoliereStatus::InvalidInput, 0, 0.};
      }
      // The symmetric one-dimensional distribution has its exact median at
      // zero. Handling it explicitly avoids PROPOSAL's relative Newton test
      // dividing by a zero iterate for the representable u=0.5 value.
      if (uniform == 0.5) {
        return {MoliereStatus::Success, 0, 0.};
      }
      auto const gaussian_coordinate =
          normalPpf(uniform) / SqrtTwo;
      auto const reference_B =
          1. /
          state.inverse_B[
              snapshot.maximum_weight_index];
      auto current =
          state.prefactor *
          improveMoliereInitialGuess(
              interpolation, reference_B,
              state.beta_squared,
              gaussian_coordinate);
      auto const target = uniform - 0.5;
      for (std::uint32_t iteration = 1;
           iteration <= 100; ++iteration) {
        auto const previous = current;
        auto const evaluation =
            evaluateDistribution(
                snapshot, interpolation, state, previous);
        if (!finite(evaluation.density) ||
            evaluation.density == 0.) {
          return {MoliereStatus::NonFiniteResult,
                  iteration, previous};
        }
        current =
            previous -
            (evaluation.cumulative - target) /
                evaluation.density;
        if (!finite(current)) {
          return {MoliereStatus::NonFiniteResult,
                  iteration, current};
        }
        auto const relative =
            absolute((previous - current) / current);
        if (relative <= 5.e-5) {
          return {
              MoliereStatus::Success, iteration, current};
        }
      }
      return {MoliereStatus::NonConvergent, 100, current};
    }

  } // namespace moliere_detail

  /**
   * Direct device-capable translation of
   * PROPOSAL::Moliere::CalculateScatteringAngle2D().
   *
   * The production CPU path currently uses MoliereInterpol. The analytical
   * Moliere implementation is used here as the primary numerical oracle;
   * their difference is validated separately before integration.
   */
  template <std::size_t ComponentCapacity>
  CORSIKA_GPU_MOLIERE_HOST_DEVICE inline MoliereAngleResult
  sampleMoliereScatteringAngle2DForCapacity(
      MoliereSnapshot const& snapshot,
      MoliereInterpolationView const& interpolation,
      double grammage_g_per_cm2,
      double initial_energy_MeV, double final_energy_MeV,
      double first_uniform, double second_uniform,
      bool validate_snapshot = true) {
    (void)final_energy_MeV;
    if (validate_snapshot &&
        moliere_detail::hasInterpolationView(interpolation) &&
        !moliere_detail::validInterpolationView(
            interpolation)) {
      return {MoliereStatus::InvalidSnapshot, 0, 0.};
    }
    moliere_detail::DistributionStateForCapacity<
        ComponentCapacity>
        state{};
    auto const status =
        moliere_detail::buildDistribution(
            snapshot, grammage_g_per_cm2,
            initial_energy_MeV, state,
            validate_snapshot);
    if (status != MoliereStatus::Success) {
      return {status, 0, 0.};
    }
    auto const first =
        moliere_detail::sampleOneDimensional(
            snapshot, interpolation, state, first_uniform);
    if (first.status != MoliereStatus::Success) {
      return first;
    }
    auto const second =
        moliere_detail::sampleOneDimensional(
            snapshot, interpolation, state, second_uniform);
    if (second.status != MoliereStatus::Success) {
      return second;
    }
    auto const angle = moliere_detail::squareRoot(
        first.angle_rad * first.angle_rad +
        second.angle_rad * second.angle_rad);
    if (!moliere_detail::finite(angle)) {
      return {
          MoliereStatus::NonFiniteResult,
          first.iterations + second.iterations, angle};
    }
    return {
        MoliereStatus::Success,
        first.iterations + second.iterations, angle};
  }

  CORSIKA_GPU_MOLIERE_HOST_DEVICE inline MoliereAngleResult
  sampleMoliereScatteringAngle2D(
      MoliereSnapshot const& snapshot,
      MoliereInterpolationView const& interpolation,
      double grammage_g_per_cm2,
      double initial_energy_MeV, double final_energy_MeV,
      double first_uniform, double second_uniform,
      bool validate_snapshot = true) {
    return sampleMoliereScatteringAngle2DForCapacity<
        MaxMoliereComponents>(
        snapshot, interpolation, grammage_g_per_cm2,
        initial_energy_MeV, final_energy_MeV,
        first_uniform, second_uniform, validate_snapshot);
  }

  CORSIKA_GPU_MOLIERE_HOST_DEVICE inline MoliereAngleResult
  sampleMoliereScatteringAngle2D(
      MoliereSnapshot const& snapshot,
      double grammage_g_per_cm2,
      double initial_energy_MeV, double final_energy_MeV,
      double first_uniform, double second_uniform,
      bool validate_snapshot = true) {
    return sampleMoliereScatteringAngle2D(
        snapshot, {}, grammage_g_per_cm2,
        initial_energy_MeV, final_energy_MeV,
        first_uniform, second_uniform, validate_snapshot);
  }

  /**
   * Apply CORSIKA proposal::ContinuousProcess's two AngleAxis rotations.
   *
   * The first axis follows the same x>0.1 branch as the CPU code and the
   * signs of both rotations are preserved. The final normalization mirrors
   * Step::getDirectionPost().
   */
  CORSIKA_GPU_MOLIERE_HOST_DEVICE inline MoliereDirectionResult
  applyMoliereDirection(
      double const initial_direction[3], double angle_rad,
      double azimuth_uniform) {
    using namespace moliere_detail;
    if (!finite(angle_rad) || angle_rad < 0. ||
        !finite(azimuth_uniform) ||
        !(azimuth_uniform > 0.) ||
        !(azimuth_uniform < 1.)) {
      return {};
    }
    auto const initial_norm_squared =
        initial_direction[0] * initial_direction[0] +
        initial_direction[1] * initial_direction[1] +
        initial_direction[2] * initial_direction[2];
    constexpr double DirectionNormTolerance = 1.e-12;
    constexpr double MinimumDirectionNormSquared =
        (1. - DirectionNormTolerance) *
        (1. - DirectionNormTolerance);
    constexpr double MaximumDirectionNormSquared =
        (1. + DirectionNormTolerance) *
        (1. + DirectionNormTolerance);
    if (!finite(initial_norm_squared) ||
        initial_norm_squared <
            MinimumDirectionNormSquared ||
        initial_norm_squared >
            MaximumDirectionNormSquared) {
      return {};
    }
    double axis1[3]{};
    if (initial_direction[0] > 0.1) {
      axis1[0] = -initial_direction[1];
      axis1[1] = initial_direction[0];
    } else {
      axis1[1] = -initial_direction[2];
      axis1[2] = initial_direction[1];
    }
    auto const axis1_norm =
        squareRoot(
            axis1[0] * axis1[0] +
            axis1[1] * axis1[1] +
            axis1[2] * axis1[2]);
    if (!finite(axis1_norm) || !(axis1_norm > 0.)) {
      return {};
    }
    for (double& component : axis1) {
      component /= axis1_norm;
    }
    double axis1_cross_direction[3]{
        axis1[1] * initial_direction[2] -
            axis1[2] * initial_direction[1],
        axis1[2] * initial_direction[0] -
            axis1[0] * initial_direction[2],
        axis1[0] * initial_direction[1] -
            axis1[1] * initial_direction[0]};
    auto const cos_theta =
#if defined(__CUDA_ARCH__)
        ::cos(angle_rad);
#else
        std::cos(angle_rad);
#endif
    auto const minus_sin_theta =
#if defined(__CUDA_ARCH__)
        -::sin(angle_rad);
#else
        -std::sin(angle_rad);
#endif
    double after_zenith[3]{};
    for (int axis = 0; axis < 3; ++axis) {
      after_zenith[axis] =
          initial_direction[axis] * cos_theta +
          axis1_cross_direction[axis] * minus_sin_theta;
    }
    auto const azimuth =
        azimuth_uniform * 2. * Pi;
    auto const cos_azimuth =
#if defined(__CUDA_ARCH__)
        ::cos(azimuth);
#else
        std::cos(azimuth);
#endif
    auto const minus_sin_azimuth =
#if defined(__CUDA_ARCH__)
        -::sin(azimuth);
#else
        -std::sin(azimuth);
#endif
    double direction_cross_after[3]{
        initial_direction[1] * after_zenith[2] -
            initial_direction[2] * after_zenith[1],
        initial_direction[2] * after_zenith[0] -
            initial_direction[0] * after_zenith[2],
        initial_direction[0] * after_zenith[1] -
            initial_direction[1] * after_zenith[0]};
    auto const parallel =
        initial_direction[0] * after_zenith[0] +
        initial_direction[1] * after_zenith[1] +
        initial_direction[2] * after_zenith[2];
    MoliereDirectionResult result{};
    for (int axis = 0; axis < 3; ++axis) {
      result.direction[axis] =
          after_zenith[axis] * cos_azimuth +
          direction_cross_after[axis] *
              minus_sin_azimuth +
          initial_direction[axis] * parallel *
              (1. - cos_azimuth);
    }
    auto const output_norm =
        squareRoot(
            result.direction[0] * result.direction[0] +
            result.direction[1] * result.direction[1] +
            result.direction[2] * result.direction[2]);
    if (!finite(output_norm) || !(output_norm > 0.)) {
      result.status = MoliereStatus::NonFiniteResult;
      return result;
    }
    for (double& component : result.direction) {
      component /= output_norm;
    }
    result.status = MoliereStatus::Success;
    return result;
  }

  /**
   * Combine curved tracking and multiple scattering exactly as the scalar
   * CORSIKA Step accumulator does.
   *
   * proposal::ContinuousProcess samples a scattered direction relative to the
   * pre-step direction and adds only that direction delta to Step.  Step
   * already contains the tracking delta from the pre-step direction to the
   * curved-trajectory endpoint.  Step::getDirectionPost() therefore returns
   *
   *   normalize(tracked + scattered_from_start - start)
   *
   * rather than a rotation of the tracked endpoint.  Keeping this small
   * algebraic detail identical is important for CPU/GPU shower equivalence
   * when a magnetic field and Moliere scattering act in the same step.
   */
  CORSIKA_GPU_MOLIERE_HOST_DEVICE inline MoliereDirectionResult
  applyMoliereDirectionCpuStepCompatible(
      double const start_direction[3],
      double const tracked_direction[3], double angle_rad,
      double azimuth_uniform) {
    using namespace moliere_detail;
    auto scattered = applyMoliereDirection(
        start_direction, angle_rad, azimuth_uniform);
    if (scattered.status != MoliereStatus::Success) {
      return scattered;
    }
    auto const tracked_norm_squared =
        tracked_direction[0] * tracked_direction[0] +
        tracked_direction[1] * tracked_direction[1] +
        tracked_direction[2] * tracked_direction[2];
    constexpr double DirectionNormTolerance = 1.e-12;
    constexpr double MinimumDirectionNormSquared =
        (1. - DirectionNormTolerance) *
        (1. - DirectionNormTolerance);
    constexpr double MaximumDirectionNormSquared =
        (1. + DirectionNormTolerance) *
        (1. + DirectionNormTolerance);
    if (!finite(tracked_norm_squared) ||
        tracked_norm_squared < MinimumDirectionNormSquared ||
        tracked_norm_squared > MaximumDirectionNormSquared) {
      return {};
    }
    MoliereDirectionResult result{};
    auto norm_squared = 0.;
    for (int axis = 0; axis < 3; ++axis) {
      result.direction[axis] =
          tracked_direction[axis] +
          scattered.direction[axis] -
          start_direction[axis];
      norm_squared +=
          result.direction[axis] * result.direction[axis];
    }
    auto const norm = squareRoot(norm_squared);
    if (!finite(norm) || !(norm > 0.)) {
      result.status = MoliereStatus::NonFiniteResult;
      return result;
    }
    for (double& component : result.direction) {
      component /= norm;
    }
    result.status = MoliereStatus::Success;
    return result;
  }

  inline MoliereSnapshot makeMoliereSnapshot(
      tables::MoliereMetadata const& source) {
    if (!source.enabled ||
        source.components.empty() ||
        source.components.size() > MaxMoliereComponents ||
        source.c1.size() != MoliereSeriesCoefficientCount ||
        source.c2.size() != MoliereSeriesCoefficientCount ||
        source.c2_large.size() !=
            MoliereLargeSeriesCoefficientCount ||
        source.s2_large.size() !=
            MoliereLargeSeriesCoefficientCount ||
        source.C1_large.size() !=
            MoliereLargeIntegralCoefficientCount) {
      throw std::invalid_argument(
          "Moliere metadata is unavailable or malformed");
    }
    MoliereSnapshot snapshot{};
    snapshot.particle_mass_MeV =
        source.particle_mass_MeV;
    snapshot.electron_mass_MeV =
        source.electron_mass_MeV;
    snapshot.fine_structure_constant =
        source.fine_structure_constant;
    snapshot.avogadro_per_mol = source.avogadro_per_mol;
    snapshot.hbar_MeV_s = source.hbar_MeV_s;
    snapshot.speed_of_light_cm_per_s =
        source.speed_of_light_cm_per_s;
    snapshot.euler_mascheroni =
        source.euler_mascheroni;
    snapshot.component_count =
        static_cast<std::uint32_t>(source.components.size());

    double atomic_mass_sum = 0.;
    for (auto const& component : source.components) {
      atomic_mass_sum +=
          component.atoms_in_molecule *
          component.atomic_mass_number;
    }
    double mass_weights[MaxMoliereComponents]{};
    double average_atomic_mass = 0.;
    double average_charge_squared = 0.;
    double weight_zz_sum = 0.;
    for (std::size_t index = 0;
         index < source.components.size(); ++index) {
      auto const& component = source.components[index];
      auto const weight =
          component.atoms_in_molecule *
          component.atomic_mass_number /
          atomic_mass_sum;
      mass_weights[index] = weight;
      average_atomic_mass +=
          weight * component.atomic_mass_number;
      auto const weight_zz =
          weight * component.nuclear_charge *
          component.nuclear_charge;
      weight_zz_sum += weight_zz;
      average_charge_squared +=
          source.particle_mass_MeV ==
                  source.electron_mass_MeV
              ? weight * component.nuclear_charge *
                    (component.nuclear_charge + 1.)
              : weight_zz;
      auto const chi_0 =
          source.electron_mass_MeV *
          source.fine_structure_constant *
          std::pow(
              component.nuclear_charge * 128. /
                  (9. * moliere_detail::Pi *
                   moliere_detail::Pi),
              1. / 3.);
      snapshot.components[index] = {
          component.nuclear_charge,
          weight_zz,
          chi_0 * chi_0,
          3.76 * source.fine_structure_constant *
              source.fine_structure_constant *
              component.nuclear_charge *
              component.nuclear_charge};
    }
    for (std::size_t index = 0;
         index + 1 < source.components.size(); ++index) {
      if (mass_weights[index + 1] > mass_weights[index]) {
        snapshot.maximum_weight_index =
            static_cast<std::uint32_t>(index + 1);
      }
    }
    snapshot.z_squared_over_a_average =
        average_charge_squared / average_atomic_mass;
    snapshot.inverse_weight_zz_sum = 1. / weight_zz_sum;
    std::copy(
        source.c1.begin(), source.c1.end(), snapshot.c1);
    std::copy(
        source.c2.begin(), source.c2.end(), snapshot.c2);
    std::copy(
        source.c2_large.begin(), source.c2_large.end(),
        snapshot.c2_large);
    std::copy(
        source.s2_large.begin(), source.s2_large.end(),
        snapshot.s2_large);
    std::copy(
        source.C1_large.begin(), source.C1_large.end(),
        snapshot.C1_large);
    if (!moliere_detail::validSnapshot(snapshot)) {
      throw std::invalid_argument(
          "Moliere metadata produced an invalid device snapshot");
    }
    return snapshot;
  }

  /**
   * Build the exact production interpolation semantics used by PROPOSAL
   * 7.6.2. The implementation lives in the compiled GPU library so CUDA
   * translation units remain free of CubicInterpolation host internals.
   */
  MoliereInterpolationTable makeMoliereInterpolationTable(
      MoliereSnapshot const& snapshot);

  /**
   * Load the snapshot-specific interpolation table from a validated cache,
   * or rebuild and atomically replace the cache when it is absent, stale or
   * corrupt. The cached table is only a Newton initial guess and never
   * replaces the exact multi-component distribution evaluation.
   */
  MoliereInterpolationTable
  loadOrMakeMoliereInterpolationTable(
      MoliereSnapshot const& snapshot,
      std::filesystem::path const& cache_path);

  inline MoliereInterpolationView makeMoliereInterpolationView(
      MoliereCubicPolynomial const* device_or_host_polynomials,
      double const* device_or_host_initial_guess = nullptr) {
    if (device_or_host_polynomials == nullptr) {
      return {};
    }
    constexpr auto intervals =
        MoliereInterpolationIntervalCount;
    return {
        device_or_host_polynomials,
        device_or_host_polynomials + intervals,
        device_or_host_polynomials + 2 * intervals,
        device_or_host_polynomials + 3 * intervals,
        device_or_host_initial_guess,
        static_cast<std::uint32_t>(intervals),
        device_or_host_initial_guess == nullptr
            ? 0U
            : static_cast<std::uint32_t>(
                  MoliereInitialGuessBNodeCount),
        device_or_host_initial_guess == nullptr
            ? 0U
            : static_cast<std::uint32_t>(
                  MoliereInitialGuessBetaSquaredNodeCount),
        device_or_host_initial_guess == nullptr
            ? 0U
            : static_cast<std::uint32_t>(
                  MoliereInitialGuessCoordinateNodeCount),
        0,
        MoliereInterpolationLower,
        MoliereInterpolationUpper,
        static_cast<double>(intervals) /
            (MoliereInterpolationUpper -
             MoliereInterpolationLower),
        MoliereInitialGuessBLower,
        MoliereInitialGuessBUpper,
        static_cast<double>(
            MoliereInitialGuessBNodeCount - 1) /
            (MoliereInitialGuessBUpper -
             MoliereInitialGuessBLower),
        MoliereInitialGuessBetaSquaredLower,
        MoliereInitialGuessBetaSquaredUpper,
        static_cast<double>(
            MoliereInitialGuessBetaSquaredNodeCount - 1) /
            (MoliereInitialGuessBetaSquaredUpper -
             MoliereInitialGuessBetaSquaredLower),
        MoliereInitialGuessCoordinateLower,
        MoliereInitialGuessCoordinateUpper,
        static_cast<double>(
            MoliereInitialGuessCoordinateNodeCount - 1) /
            (MoliereInitialGuessCoordinateUpper -
             MoliereInitialGuessCoordinateLower)};
  }

} // namespace corsika::gpu::em

#undef CORSIKA_GPU_MOLIERE_HOST_DEVICE

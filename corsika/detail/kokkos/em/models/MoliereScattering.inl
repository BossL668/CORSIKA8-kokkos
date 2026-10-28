/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once
#include <cmath>
namespace corsika::kokkos {
  namespace moliere_detail {
    CORSIKA_HOST_DEVICE inline bool finite(double value) {
#if defined(__CUDA_ARCH__)
      return ::isfinite(value);
#else
      return std::isfinite(value);
#endif
    }

    CORSIKA_HOST_DEVICE inline double absolute(double value) {
#if defined(__CUDA_ARCH__)
      return ::fabs(value);
#else
      return std::abs(value);
#endif
    }

    CORSIKA_HOST_DEVICE inline double square_root(double value) {
#if defined(__CUDA_ARCH__)
      return ::sqrt(value);
#else
      return std::sqrt(value);
#endif
    }

    CORSIKA_HOST_DEVICE inline double logarithm(double value) {
#if defined(__CUDA_ARCH__)
      return ::log(value);
#else
      return std::log(value);
#endif
    }

    CORSIKA_HOST_DEVICE inline double exponential(double value) {
#if defined(__CUDA_ARCH__)
      return ::exp(value);
#else
      return std::exp(value);
#endif
    }

    CORSIKA_HOST_DEVICE inline double power(double base, double exponent) {
#if defined(__CUDA_ARCH__)
      return ::pow(base, exponent);
#else
      return std::pow(base, exponent);
#endif
    }

    CORSIKA_HOST_DEVICE inline double error_function(double value) {
#if defined(__CUDA_ARCH__)
      return ::erf(value);
#else
      return std::erf(value);
#endif
    }

    CORSIKA_HOST_DEVICE inline bool valid_interpolation_view(
        MoliereInterpolationView const& interpolation) {
      return interpolation.getF1().hasStorage() && interpolation.getF2().hasStorage() &&
             interpolation.getIntegralF1().hasStorage() &&
             interpolation.getIntegralF2().hasStorage() &&
             interpolation.getIntervalCount() == MoliereInterpolationIntervalCount &&
             finite(interpolation.getLower()) && finite(interpolation.getUpper()) &&
             finite(interpolation.getInverseStep()) &&
             interpolation.getLower() == MoliereInterpolationLower &&
             interpolation.getUpper() == MoliereInterpolationUpper &&
             interpolation.getInverseStep() > 0. &&
             (!interpolation.getInitialGuessDelta().hasStorage() ||
              (interpolation.getInitialGuessBNodeCount() ==
                   MoliereInitialGuessBNodeCount &&
               interpolation.getInitialGuessBetaSquaredNodeCount() ==
                   MoliereInitialGuessBetaSquaredNodeCount &&
               interpolation.getInitialGuessCoordinateNodeCount() ==
                   MoliereInitialGuessCoordinateNodeCount &&
               finite(interpolation.getInitialGuessBLower()) &&
               finite(interpolation.getInitialGuessBUpper()) &&
               finite(interpolation.getInitialGuessInverseBStep()) &&
               finite(interpolation.getInitialGuessBetaSquaredLower()) &&
               finite(interpolation.getInitialGuessBetaSquaredUpper()) &&
               finite(interpolation.getInitialGuessInverseBetaSquaredStep()) &&
               finite(interpolation.getInitialGuessCoordinateLower()) &&
               finite(interpolation.getInitialGuessCoordinateUpper()) &&
               finite(interpolation.getInitialGuessInverseCoordinateStep()) &&
               interpolation.getInitialGuessBLower() == MoliereInitialGuessBLower &&
               interpolation.getInitialGuessBUpper() == MoliereInitialGuessBUpper &&
               interpolation.getInitialGuessInverseBStep() > 0. &&
               interpolation.getInitialGuessBetaSquaredLower() ==
                   MoliereInitialGuessBetaSquaredLower &&
               interpolation.getInitialGuessBetaSquaredUpper() ==
                   MoliereInitialGuessBetaSquaredUpper &&
               interpolation.getInitialGuessInverseBetaSquaredStep() > 0. &&
               interpolation.getInitialGuessCoordinateLower() ==
                   MoliereInitialGuessCoordinateLower &&
               interpolation.getInitialGuessCoordinateUpper() ==
                   MoliereInitialGuessCoordinateUpper &&
               interpolation.getInitialGuessInverseCoordinateStep() > 0.));
    }

    CORSIKA_HOST_DEVICE inline bool has_interpolation_view(
        MoliereInterpolationView const& interpolation) {
      return interpolation.getF1().hasStorage() || interpolation.getF2().hasStorage() ||
             interpolation.getIntegralF1().hasStorage() ||
             interpolation.getIntegralF2().hasStorage() ||
             interpolation.getIntervalCount() != 0;
    }

    CORSIKA_HOST_DEVICE inline double evaluate_interpolation(
        NativeArrayView<MoliereCubicPolynomial> const& polynomials,
        MoliereInterpolationView const& interpolation, double x) {
      auto coordinate = (x - interpolation.getLower()) * interpolation.getInverseStep();
      if (coordinate < 0.) { coordinate = 0.; }
      auto const maximum_coordinate =
          static_cast<double>(interpolation.getIntervalCount());
      if (coordinate > maximum_coordinate) { coordinate = maximum_coordinate; }
      auto interval = static_cast<std::uint32_t>(coordinate);
      if (interval >= interpolation.getIntervalCount()) {
        interval = interpolation.getIntervalCount() - 1;
      }
      auto const q = coordinate - static_cast<double>(interval);
      auto const& polynomial = polynomials[interval];
      return polynomial.c_[0] +
             q * (polynomial.c_[1] + q * (polynomial.c_[2] + q * polynomial.c_[3]));
    }

    CORSIKA_HOST_DEVICE inline double improve_moliere_initial_guess(
        MoliereInterpolationView const& interpolation, double reference_B,
        double beta_squared, double gaussian_coordinate) {
      if (!interpolation.getInitialGuessDelta().hasStorage() || !finite(reference_B) ||
          !finite(beta_squared) || !finite(gaussian_coordinate)) {
        return gaussian_coordinate;
      }
      auto const magnitude = absolute(gaussian_coordinate);
      if (reference_B < interpolation.getInitialGuessBLower() ||
          reference_B > interpolation.getInitialGuessBUpper() ||
          beta_squared < interpolation.getInitialGuessBetaSquaredLower() ||
          beta_squared > interpolation.getInitialGuessBetaSquaredUpper() ||
          magnitude < interpolation.getInitialGuessCoordinateLower() ||
          magnitude > interpolation.getInitialGuessCoordinateUpper()) {
        return gaussian_coordinate;
      }
      auto b_coordinate = (reference_B - interpolation.getInitialGuessBLower()) *
                          interpolation.getInitialGuessInverseBStep();
      auto coordinate = (magnitude - interpolation.getInitialGuessCoordinateLower()) *
                        interpolation.getInitialGuessInverseCoordinateStep();
      auto beta_coordinate =
          (beta_squared - interpolation.getInitialGuessBetaSquaredLower()) *
          interpolation.getInitialGuessInverseBetaSquaredStep();
      auto b_index = static_cast<std::uint32_t>(b_coordinate);
      auto beta_index = static_cast<std::uint32_t>(beta_coordinate);
      auto coordinate_index = static_cast<std::uint32_t>(coordinate);
      if (b_index + 1 >= interpolation.getInitialGuessBNodeCount()) {
        b_index = interpolation.getInitialGuessBNodeCount() - 2;
      }
      if (beta_index + 1 >= interpolation.getInitialGuessBetaSquaredNodeCount()) {
        beta_index = interpolation.getInitialGuessBetaSquaredNodeCount() - 2;
      }
      if (coordinate_index + 1 >= interpolation.getInitialGuessCoordinateNodeCount()) {
        coordinate_index = interpolation.getInitialGuessCoordinateNodeCount() - 2;
      }
      auto const b_fraction = b_coordinate - static_cast<double>(b_index);
      auto const beta_fraction = beta_coordinate - static_cast<double>(beta_index);
      auto const coordinate_fraction = coordinate - static_cast<double>(coordinate_index);
      auto const coordinate_stride = interpolation.getInitialGuessCoordinateNodeCount();
      auto const beta_stride =
          static_cast<std::size_t>(interpolation.getInitialGuessBetaSquaredNodeCount()) *
          coordinate_stride;
      auto const lower_b_offset = static_cast<std::size_t>(b_index) * beta_stride;
      auto const upper_b_offset = lower_b_offset + beta_stride;
      auto const lower_beta_offset =
          static_cast<std::size_t>(beta_index) * coordinate_stride + coordinate_index;
      auto const upper_beta_offset = lower_beta_offset + coordinate_stride;
      auto const lower_b_lower_beta_offset = lower_b_offset + lower_beta_offset;
      auto const lower_b_upper_beta_offset = lower_b_offset + upper_beta_offset;
      auto const upper_b_lower_beta_offset = upper_b_offset + lower_beta_offset;
      auto const upper_b_upper_beta_offset = upper_b_offset + upper_beta_offset;
      auto const lower_b_lower_beta =
          interpolation.getInitialGuessDelta()[lower_b_lower_beta_offset] *
              (1. - coordinate_fraction) +
          interpolation.getInitialGuessDelta()[lower_b_lower_beta_offset + 1] *
              coordinate_fraction;
      auto const lower_b_upper_beta =
          interpolation.getInitialGuessDelta()[lower_b_upper_beta_offset] *
              (1. - coordinate_fraction) +
          interpolation.getInitialGuessDelta()[lower_b_upper_beta_offset + 1] *
              coordinate_fraction;
      auto const upper_b_lower_beta =
          interpolation.getInitialGuessDelta()[upper_b_lower_beta_offset] *
              (1. - coordinate_fraction) +
          interpolation.getInitialGuessDelta()[upper_b_lower_beta_offset + 1] *
              coordinate_fraction;
      auto const upper_b_upper_beta =
          interpolation.getInitialGuessDelta()[upper_b_upper_beta_offset] *
              (1. - coordinate_fraction) +
          interpolation.getInitialGuessDelta()[upper_b_upper_beta_offset + 1] *
              coordinate_fraction;
      auto const lower_b =
          lower_b_lower_beta * (1. - beta_fraction) + lower_b_upper_beta * beta_fraction;
      auto const upper_b =
          upper_b_lower_beta * (1. - beta_fraction) + upper_b_upper_beta * beta_fraction;
      auto const corrected_magnitude =
          magnitude + lower_b * (1. - b_fraction) + upper_b * b_fraction;
      if (!finite(corrected_magnitude) || corrected_magnitude < 0.) {
        return gaussian_coordinate;
      }
      return gaussian_coordinate < 0. ? -corrected_magnitude : corrected_magnitude;
    }

    CORSIKA_HOST_DEVICE inline double normal_ppf(double probability) {
      constexpr double a[] = {-3.969683028665376e+01, 2.209460984245205e+02,
                              -2.759285104469687e+02, 1.383577518672690e+02,
                              -3.066479806614716e+01, 2.506628277459239e+00};
      constexpr double b[] = {-5.447609879822406e+01, 1.615858368580409e+02,
                              -1.556989798598866e+02, 6.680131188771972e+01,
                              -1.328068155288572e+01};
      constexpr double c[] = {-7.784894002430293e-03, -3.223964580411365e-01,
                              -2.400758277161838e+00, -2.549732539343734e+00,
                              4.374664141464968e+00,  2.938163982698783e+00};
      constexpr double d[] = {7.784695709041462e-03, 3.224671290700398e-01,
                              2.445134137142996e+00, 3.754408661907416e+00};
      constexpr double p_low = 0.02425;
      constexpr double p_high = 1. - p_low;
      double q = 0.;
      double r = 0.;
      double x = 0.;
      if (probability < p_low) {
        q = square_root(-2. * logarithm(probability));
        x = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.);
      } else if (probability <= p_high) {
        q = probability - 0.5;
        r = q * q;
        x = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q /
            (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.);
      } else {
        q = square_root(-2. * logarithm(1. - probability));
        x = -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.);
      }
      // This quantile seeds the exact Moliere Newton solve; it is not the
      // returned variate. Acklam's rational approximation is already far
      // inside that solver's basin, so PROPOSAL's final Halley refinement
      // (one erfc and one exp) only improves a disposable initial guess.
      return x;
    }

    CORSIKA_HOST_DEVICE inline double solve_moliere_b(double rhs) {
      if (!finite(rhs) || rhs < MoliereMinimumBEquationRightHandSide) { return 0.; }
      auto const logarithm_rhs = logarithm(rhs);
      auto const inverse_rhs = 1. / rhs;
      auto const inverse_rhs_squared = inverse_rhs * inverse_rhs;
      auto const logarithm_rhs_squared = logarithm_rhs * logarithm_rhs;
      auto const logarithm_rhs_cubed = logarithm_rhs_squared * logarithm_rhs;
      auto current = rhs + logarithm_rhs + logarithm_rhs * inverse_rhs +
                     logarithm_rhs * (2. - logarithm_rhs) * 0.5 * inverse_rhs_squared +
                     logarithm_rhs *
                         (6. - 9. * logarithm_rhs + 2. * logarithm_rhs_squared) / 6. *
                         inverse_rhs_squared * inverse_rhs +
                     logarithm_rhs *
                         (12. - 36. * logarithm_rhs + 22. * logarithm_rhs_squared -
                          3. * logarithm_rhs_cubed) /
                         12. * inverse_rhs_squared * inverse_rhs_squared;
      if (!finite(current) || !(current > 1.)) { return 0.; }
      auto const function = current - logarithm(current) - rhs;
      auto const inverse_current = 1. / current;
      auto const derivative = 1. - inverse_current;
      auto const denominator =
          2. * derivative * derivative - function * inverse_current * inverse_current;
      if (!finite(denominator) || denominator == 0.) { return 0.; }
      current -= 2. * function * derivative / denominator;
      return finite(current) && current >= MoliereMinimumB ? current : 0.;
    }

    CORSIKA_HOST_DEVICE inline double f1_m(MoliereSnapshot const& snapshot, double x) {
      if (x > 12.) {
        return 0.5 * square_root(Pi) / (power(x, 1.5) * power(1. - 4.5 / x, 2. / 3.));
      }
      auto sum = snapshot.c1_[69];
      for (int index = 68; index >= 0; --index) { sum = sum * x + snapshot.c1_[index]; }
      return sum;
    }

    CORSIKA_HOST_DEVICE inline double f2_m(MoliereSnapshot const& snapshot, double x) {
      if (x > 4.25 * 4.25) {
        constexpr double a = 0.00013567765224589459194769192063035;
        constexpr double b = -0.0022635502525409950842771866774683;
        constexpr double c = 0.0098037758070269476889935233998585;
        if (x <= 6.5 * 6.5) { return a * x + b * square_root(x) + c; }
        double sum = 0.;
        for (int index = 2; index < 13; ++index) {
          sum += snapshot.c2Large_[index] *
                 (0.5 * logarithm(x) + snapshot.s2Large_[index]) *
                 power(x, -(index + 0.5));
        }
        return sum;
      }
      auto sum = snapshot.c2_[69];
      for (int index = 68; index >= 0; --index) { sum = sum * x + snapshot.c2_[index]; }
      return sum;
    }

    CORSIKA_HOST_DEVICE inline double integral_f1_m(MoliereSnapshot const& snapshot,
                                                    double x) {
      if (x > 12.) {
        auto sum = snapshot.c1IntegralLarge_[14];
        for (int index = 13; index >= 0; --index) {
          sum = snapshot.c1IntegralLarge_[index] + sum / x;
        }
        return sum;
      }
      auto sum = snapshot.c1_[69] / (2. * 69. + 1.);
      for (int index = 68; index >= 0; --index) {
        sum = sum * x + snapshot.c1_[index] / (2. * index + 1.);
      }
      return sum * square_root(x > 0. ? x : 0.);
    }

    CORSIKA_HOST_DEVICE inline double integral_f2_m(MoliereSnapshot const& snapshot,
                                                    double x) {
      if (x > 4.25 * 4.25) {
        constexpr double a = -0.00026360133958801203364619158975302;
        constexpr double b = 0.0039965027465457608410459577896745;
        constexpr double c = -0.016305842044996649714549974419242;
        if (x <= 6.5 * 6.5) { return a * x + b * square_root(x) + c; }
        double sum = 0.;
        for (int index = 2; index < 13; ++index) {
          sum += -0.5 * snapshot.c2Large_[index] / index *
                 (0.5 / index + 0.5 * logarithm(x) + snapshot.s2Large_[index]) *
                 power(x, -index);
        }
        return sum;
      }
      auto sum = snapshot.c2_[69] / (2. * 69. + 1.);
      for (int index = 68; index >= 0; --index) {
        sum = sum * x + snapshot.c2_[index] / (2. * index + 1.);
      }
      return sum * square_root(x > 0. ? x : 0.);
    }

    CORSIKA_HOST_DEVICE inline double f1_m(MoliereSnapshot const& snapshot,
                                           MoliereInterpolationView const& interpolation,
                                           double x) {
      if (interpolation.getF1().hasStorage() && x <= MoliereInterpolationUpper) {
        return evaluate_interpolation(interpolation.getF1(), interpolation, x);
      }
      return f1_m(snapshot, x);
    }

    CORSIKA_HOST_DEVICE inline double f2_m(MoliereSnapshot const& snapshot,
                                           MoliereInterpolationView const& interpolation,
                                           double x) {
      if (interpolation.getF2().hasStorage() && x <= MoliereInterpolationUpper) {
        return evaluate_interpolation(interpolation.getF2(), interpolation, x);
      }
      return f2_m(snapshot, x);
    }

    CORSIKA_HOST_DEVICE inline double integral_f1_m(
        MoliereSnapshot const& snapshot, MoliereInterpolationView const& interpolation,
        double x) {
      if (interpolation.getIntegralF1().hasStorage() && x <= MoliereInterpolationUpper) {
        return evaluate_interpolation(interpolation.getIntegralF1(), interpolation, x);
      }
      return integral_f1_m(snapshot, x);
    }

    CORSIKA_HOST_DEVICE inline double integral_f2_m(
        MoliereSnapshot const& snapshot, MoliereInterpolationView const& interpolation,
        double x) {
      if (interpolation.getIntegralF2().hasStorage() && x <= MoliereInterpolationUpper) {
        return evaluate_interpolation(interpolation.getIntegralF2(), interpolation, x);
      }
      return integral_f2_m(snapshot, x);
    }

    template <std::size_t ComponentCapacity>
    CORSIKA_HOST_DEVICE inline MoliereStatus build_distribution(
        MoliereSnapshot const& snapshot, double grammage_g_per_cm2,
        double initial_energy_MeV, DistributionStateForCapacity<ComponentCapacity>& state,
        bool validate_snapshot) {
      if (snapshot.componentCount_ == 0 || snapshot.componentCount_ > ComponentCapacity) {
        return MoliereStatus::InvalidSnapshot;
      }
      if (validate_snapshot && !valid_moliere_snapshot(snapshot)) {
        return MoliereStatus::InvalidSnapshot;
      }
      if (!finite(grammage_g_per_cm2) || grammage_g_per_cm2 < 0. ||
          !finite(initial_energy_MeV) ||
          !(initial_energy_MeV > snapshot.particleMassMeV_)) {
        return MoliereStatus::InvalidInput;
      }
      if (grammage_g_per_cm2 == 0.) { return MoliereStatus::NoDeflection; }
      auto const momentum_squared = (initial_energy_MeV - snapshot.particleMassMeV_) *
                                    (initial_energy_MeV + snapshot.particleMassMeV_);
      auto const beta_squared =
          1. /
          (1. + snapshot.particleMassMeV_ * snapshot.particleMassMeV_ / momentum_squared);
      state.betaSquared_ = beta_squared;
      auto beta_p_squared = momentum_squared / initial_energy_MeV;
      beta_p_squared *= beta_p_squared;
      auto const chi_c_squared =
          4. * Pi * snapshot.avogadroPerMol_ * snapshot.fineStructureConstant_ *
          snapshot.fineStructureConstant_ * snapshot.hbarMeVS_ * snapshot.hbarMeVS_ *
          snapshot.speedOfLightCmPerS_ * snapshot.speedOfLightCmPerS_ *
          grammage_g_per_cm2 / beta_p_squared * snapshot.zSquaredOverAAverage_;
      if (!finite(chi_c_squared) || !(chi_c_squared > 0.)) {
        return MoliereStatus::NonFiniteResult;
      }
      double maximum_weight_B = 0.;
      for (std::uint32_t index = 0; index < snapshot.componentCount_; ++index) {
        auto const chi_a_squared =
            snapshot.components_[index].chi0SquaredMeV2_ / momentum_squared *
            (1.13 + snapshot.components_[index].coulombCorrection_ / beta_squared);
        auto const log_chi_ratio = logarithm(chi_c_squared / chi_a_squared);
        auto const current =
            solve_moliere_b(log_chi_ratio + 1. - 2. * snapshot.eulerMascheroni_);
        if (!(current > 0.)) { return MoliereStatus::NoDeflection; }
        auto const scale = chi_c_squared * current;
        if (!finite(scale) || !(scale > 0.)) { return MoliereStatus::NonFiniteResult; }
        state.inverseScale_[index] = 1. / scale;
        state.inverseB_[index] = 1. / current;
        state.densityWeight_[index] =
            snapshot.components_[index].weightZz_ / square_root(scale * Pi);
        if (!finite(state.inverseScale_[index]) || !(state.inverseScale_[index] > 0.) ||
            !finite(state.inverseB_[index]) || !(state.inverseB_[index] > 0.) ||
            !finite(state.densityWeight_[index]) || !(state.densityWeight_[index] > 0.)) {
          return MoliereStatus::NonFiniteResult;
        }
        if (index == snapshot.maximumWeightIndex_) { maximum_weight_B = current; }
      }
      state.prefactor_ = square_root(chi_c_squared * maximum_weight_B);
      return finite(state.prefactor_) && state.prefactor_ > 0.
                 ? MoliereStatus::Success
                 : MoliereStatus::NonFiniteResult;
    }

    template <std::size_t ComponentCapacity>
    CORSIKA_HOST_DEVICE inline DistributionEvaluation<ComponentCapacity>
    evaluate_distribution(MoliereSnapshot const& snapshot,
                          MoliereInterpolationView const& interpolation,
                          DistributionStateForCapacity<ComponentCapacity> const& state,
                          double theta) {
      double density_result = 0.;
      double cumulative_result = 0.;
      auto const theta_squared = theta * theta;
      for (std::uint32_t index = 0; index < snapshot.componentCount_; ++index) {
        auto const x = theta_squared * state.inverseScale_[index];
        auto const inverse_B = state.inverseB_[index];
        auto const inverse_B_squared = inverse_B * inverse_B;
        density_result +=
            state.densityWeight_[index] *
            (exponential(-x) + f1_m(snapshot, interpolation, x) * inverse_B +
             f2_m(snapshot, interpolation, x) * inverse_B_squared);
        cumulative_result +=
            snapshot.components_[index].weightZz_ *
            (0.5 * error_function(square_root(x > 0. ? x : 0.)) +
             square_root(1. / Pi) *
                 (integral_f1_m(snapshot, interpolation, x) * inverse_B +
                  integral_f2_m(snapshot, interpolation, x) * inverse_B_squared));
      }
      density_result *= snapshot.inverseWeightZzSum_;
      cumulative_result *= snapshot.inverseWeightZzSum_;
      return {density_result, theta < 0. ? -cumulative_result : cumulative_result};
    }

    template <std::size_t ComponentCapacity>
    CORSIKA_HOST_DEVICE inline MoliereAngleResult sample_one_dimensional(
        MoliereSnapshot const& snapshot, MoliereInterpolationView const& interpolation,
        DistributionStateForCapacity<ComponentCapacity> const& state, double uniform) {
      if (!finite(uniform) || !(uniform > 0.) || !(uniform < 1.)) {
        return {MoliereStatus::InvalidInput, 0, 0.};
      }
      // The symmetric one-dimensional distribution has its exact median at
      // zero. Handling it explicitly avoids PROPOSAL's relative Newton test
      // dividing by a zero iterate for the representable u=0.5 value.
      if (uniform == 0.5) { return {MoliereStatus::Success, 0, 0.}; }
      auto const gaussian_coordinate = normal_ppf(uniform) / SqrtTwo;
      auto const reference_B = 1. / state.inverseB_[snapshot.maximumWeightIndex_];
      auto current = state.prefactor_ * improve_moliere_initial_guess(
                                            interpolation, reference_B,
                                            state.betaSquared_, gaussian_coordinate);
      auto const target = uniform - 0.5;
      for (std::uint32_t iteration = 1; iteration <= 100; ++iteration) {
        auto const previous = current;
        auto const evaluation =
            evaluate_distribution(snapshot, interpolation, state, previous);
        if (!finite(evaluation.density_) || evaluation.density_ == 0.) {
          return {MoliereStatus::NonFiniteResult, iteration, previous};
        }
        current = previous - (evaluation.cumulative_ - target) / evaluation.density_;
        if (!finite(current)) {
          return {MoliereStatus::NonFiniteResult, iteration, current};
        }
        auto const relative = absolute((previous - current) / current);
        if (relative <= 5.e-5) { return {MoliereStatus::Success, iteration, current}; }
      }
      return {MoliereStatus::NonConvergent, 100, current};
    }

  } // namespace moliere_detail

  template <std::size_t ComponentCapacity>
  CORSIKA_HOST_DEVICE inline MoliereAngleResult
  sample_moliere_scattering_angle_2d_for_capacity(
      MoliereSnapshot const& snapshot, MoliereInterpolationView const& interpolation,
      double grammage_g_per_cm2, double initial_energy_MeV, double final_energy_MeV,
      double first_uniform, double second_uniform, bool validate_snapshot) {
    (void)final_energy_MeV;
    if (validate_snapshot && moliere_detail::has_interpolation_view(interpolation) &&
        !moliere_detail::valid_interpolation_view(interpolation)) {
      return {MoliereStatus::InvalidSnapshot, 0, 0.};
    }
    moliere_detail::DistributionStateForCapacity<ComponentCapacity> state{};
    auto const status = moliere_detail::build_distribution(
        snapshot, grammage_g_per_cm2, initial_energy_MeV, state, validate_snapshot);
    if (status != MoliereStatus::Success) { return {status, 0, 0.}; }
    auto const first = moliere_detail::sample_one_dimensional(snapshot, interpolation,
                                                              state, first_uniform);
    if (first.status_ != MoliereStatus::Success) { return first; }
    auto const second = moliere_detail::sample_one_dimensional(snapshot, interpolation,
                                                               state, second_uniform);
    if (second.status_ != MoliereStatus::Success) { return second; }
    auto const angle = moliere_detail::square_root(first.angleRad_ * first.angleRad_ +
                                                   second.angleRad_ * second.angleRad_);
    if (!moliere_detail::finite(angle)) {
      return {MoliereStatus::NonFiniteResult, first.iterations_ + second.iterations_,
              angle};
    }
    return {MoliereStatus::Success, first.iterations_ + second.iterations_, angle};
  }

  CORSIKA_HOST_DEVICE inline MoliereAngleResult sample_moliere_scattering_angle_2d(
      MoliereSnapshot const& snapshot, MoliereInterpolationView const& interpolation,
      double grammage_g_per_cm2, double initial_energy_MeV, double final_energy_MeV,
      double first_uniform, double second_uniform, bool validate_snapshot) {
    return sample_moliere_scattering_angle_2d_for_capacity<MaxMoliereComponents>(
        snapshot, interpolation, grammage_g_per_cm2, initial_energy_MeV, final_energy_MeV,
        first_uniform, second_uniform, validate_snapshot);
  }

  CORSIKA_HOST_DEVICE inline MoliereAngleResult sample_moliere_scattering_angle_2d(
      MoliereSnapshot const& snapshot, double grammage_g_per_cm2,
      double initial_energy_MeV, double final_energy_MeV, double first_uniform,
      double second_uniform, bool validate_snapshot) {
    return sample_moliere_scattering_angle_2d(
        snapshot, {}, grammage_g_per_cm2, initial_energy_MeV, final_energy_MeV,
        first_uniform, second_uniform, validate_snapshot);
  }

  CORSIKA_HOST_DEVICE inline MoliereDirectionResult apply_moliere_direction(
      double const (&initial_direction)[3], double angle_rad, double azimuth_uniform) {
    using namespace moliere_detail;
    if (!finite(angle_rad) || angle_rad < 0. || !finite(azimuth_uniform) ||
        !(azimuth_uniform > 0.) || !(azimuth_uniform < 1.)) {
      return {};
    }
    auto const initial_norm_squared = initial_direction[0] * initial_direction[0] +
                                      initial_direction[1] * initial_direction[1] +
                                      initial_direction[2] * initial_direction[2];
    constexpr double DirectionNormTolerance = 1.e-12;
    constexpr double MinimumDirectionNormSquared =
        (1. - DirectionNormTolerance) * (1. - DirectionNormTolerance);
    constexpr double MaximumDirectionNormSquared =
        (1. + DirectionNormTolerance) * (1. + DirectionNormTolerance);
    if (!finite(initial_norm_squared) ||
        initial_norm_squared < MinimumDirectionNormSquared ||
        initial_norm_squared > MaximumDirectionNormSquared) {
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
        square_root(axis1[0] * axis1[0] + axis1[1] * axis1[1] + axis1[2] * axis1[2]);
    if (!finite(axis1_norm) || !(axis1_norm > 0.)) { return {}; }
    for (double& component : axis1) { component /= axis1_norm; }
    double axis1_cross_direction[3]{
        axis1[1] * initial_direction[2] - axis1[2] * initial_direction[1],
        axis1[2] * initial_direction[0] - axis1[0] * initial_direction[2],
        axis1[0] * initial_direction[1] - axis1[1] * initial_direction[0]};
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
      after_zenith[axis] = initial_direction[axis] * cos_theta +
                           axis1_cross_direction[axis] * minus_sin_theta;
    }
    auto const azimuth = azimuth_uniform * 2. * Pi;
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
        initial_direction[1] * after_zenith[2] - initial_direction[2] * after_zenith[1],
        initial_direction[2] * after_zenith[0] - initial_direction[0] * after_zenith[2],
        initial_direction[0] * after_zenith[1] - initial_direction[1] * after_zenith[0]};
    auto const parallel = initial_direction[0] * after_zenith[0] +
                          initial_direction[1] * after_zenith[1] +
                          initial_direction[2] * after_zenith[2];
    MoliereDirectionResult result{};
    for (int axis = 0; axis < 3; ++axis) {
      result.direction_[axis] = after_zenith[axis] * cos_azimuth +
                                direction_cross_after[axis] * minus_sin_azimuth +
                                initial_direction[axis] * parallel * (1. - cos_azimuth);
    }
    auto const output_norm = square_root(result.direction_[0] * result.direction_[0] +
                                         result.direction_[1] * result.direction_[1] +
                                         result.direction_[2] * result.direction_[2]);
    if (!finite(output_norm) || !(output_norm > 0.)) {
      result.status_ = MoliereStatus::NonFiniteResult;
      return result;
    }
    for (double& component : result.direction_) { component /= output_norm; }
    result.status_ = MoliereStatus::Success;
    return result;
  }

  CORSIKA_HOST_DEVICE inline MoliereDirectionResult
  apply_moliere_direction_cpu_step_compatible(double const (&start_direction)[3],
                                              double const (&tracked_direction)[3],
                                              double angle_rad, double azimuth_uniform) {
    using namespace moliere_detail;
    auto scattered = apply_moliere_direction(start_direction, angle_rad, azimuth_uniform);
    if (scattered.status_ != MoliereStatus::Success) { return scattered; }
    auto const tracked_norm_squared = tracked_direction[0] * tracked_direction[0] +
                                      tracked_direction[1] * tracked_direction[1] +
                                      tracked_direction[2] * tracked_direction[2];
    constexpr double DirectionNormTolerance = 1.e-12;
    constexpr double MinimumDirectionNormSquared =
        (1. - DirectionNormTolerance) * (1. - DirectionNormTolerance);
    constexpr double MaximumDirectionNormSquared =
        (1. + DirectionNormTolerance) * (1. + DirectionNormTolerance);
    if (!finite(tracked_norm_squared) ||
        tracked_norm_squared < MinimumDirectionNormSquared ||
        tracked_norm_squared > MaximumDirectionNormSquared) {
      return {};
    }
    MoliereDirectionResult result{};
    auto norm_squared = 0.;
    for (int axis = 0; axis < 3; ++axis) {
      result.direction_[axis] =
          tracked_direction[axis] + scattered.direction_[axis] - start_direction[axis];
      norm_squared += result.direction_[axis] * result.direction_[axis];
    }
    auto const norm = square_root(norm_squared);
    if (!finite(norm) || !(norm > 0.)) {
      result.status_ = MoliereStatus::NonFiniteResult;
      return result;
    }
    for (double& component : result.direction_) { component /= norm; }
    result.status_ = MoliereStatus::Success;
    return result;
  }

} // namespace corsika::kokkos

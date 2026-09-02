/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <cmath>
#include <cstdint>
#include <type_traits>

#include <corsika/accelerator/AcceleratorMacros.hpp>
#include <corsika/gpu/em/BremsLpm.hpp>
#include <corsika/gpu/em/Philox.hpp>

#define CORSIKA_GPU_EPAIR_HOST_DEVICE C8_ACCELERATOR_INLINE_FUNCTION

namespace corsika::gpu::em {

  enum class EpairFinalStateStatus : std::uint32_t {
    Success = 0,
    ComponentNotFound = 1,
    InvalidSnapshot = 2,
    InvalidInput = 3,
    IntegrationFailed = 4,
    NonFiniteResult = 5,
    RejectionEnvelopeUnavailable = 6,
    RejectionEnvelopeExceeded = 7,
    RejectionTrialsExhausted = 8,
  };

  struct EpairFinalStateSample {
    EpairFinalStateStatus status{
        EpairFinalStateStatus::InvalidSnapshot};
    std::uint32_t reserved{};
    double rho{};
    double rho_max{};
  };

  static_assert(std::is_standard_layout_v<EpairFinalStateSample>);
  static_assert(std::is_trivially_copyable_v<EpairFinalStateSample>);

  struct EpairRejectionSample {
    EpairFinalStateSample sample{};
    std::uint32_t trial_count{};
    std::uint32_t reserved{};
  };

  static_assert(std::is_standard_layout_v<EpairRejectionSample>);
  static_assert(std::is_trivially_copyable_v<EpairRejectionSample>);

  inline constexpr int EpairRejectionEnvelopePoints = 8;
  inline constexpr int EpairRejectionEnvelopeRefinementSteps = 12;
  inline constexpr int EpairRejectionMaximumTrials = 256;
  inline constexpr double EpairRejectionEnvelopeSafety = 1.05;
  inline constexpr std::uint64_t
      EpairRejectionCandidateDrawIdBase = 0x100;
  inline constexpr std::uint64_t
      EpairRejectionAcceptanceDrawIdBase = 0x200;

  namespace epair_final_state_detail {

    CORSIKA_GPU_EPAIR_HOST_DEVICE inline bool finite(double value) {
#if defined(__CUDA_ARCH__)
      return ::isfinite(value);
#else
      return std::isfinite(value);
#endif
    }

    CORSIKA_GPU_EPAIR_HOST_DEVICE inline double squareRoot(
        double value) {
#if defined(__CUDA_ARCH__)
      return ::sqrt(value);
#else
      return std::sqrt(value);
#endif
    }

    CORSIKA_GPU_EPAIR_HOST_DEVICE inline double logarithm(
        double value) {
#if defined(__CUDA_ARCH__)
      return ::log(value);
#else
      return std::log(value);
#endif
    }

    CORSIKA_GPU_EPAIR_HOST_DEVICE inline double logarithmOnePlus(
        double value) {
#if defined(__CUDA_ARCH__)
      return ::log1p(value);
#else
      return std::log1p(value);
#endif
    }

    CORSIKA_GPU_EPAIR_HOST_DEVICE inline double exponential(
        double value) {
#if defined(__CUDA_ARCH__)
      return ::exp(value);
#else
      return std::exp(value);
#endif
    }

    CORSIKA_GPU_EPAIR_HOST_DEVICE inline double sine(
        double value) {
#if defined(__CUDA_ARCH__)
      return ::sin(value);
#else
      return std::sin(value);
#endif
    }

    CORSIKA_GPU_EPAIR_HOST_DEVICE inline double power(
        double base, double exponent) {
#if defined(__CUDA_ARCH__)
      return ::pow(base, exponent);
#else
      return std::pow(base, exponent);
#endif
    }

    CORSIKA_GPU_EPAIR_HOST_DEVICE inline double maximum(
        double first, double second) {
#if defined(__CUDA_ARCH__)
      return ::fmax(first, second);
#else
      return std::fmax(first, second);
#endif
    }

    CORSIKA_GPU_EPAIR_HOST_DEVICE inline double absolute(
        double value) {
#if defined(__CUDA_ARCH__)
      return ::fabs(value);
#else
      return std::fabs(value);
#endif
    }

    CORSIKA_GPU_EPAIR_HOST_DEVICE inline double nextToward(
        double value, double direction) {
#if defined(__CUDA_ARCH__)
      return ::nextafter(value, direction);
#else
      return std::nextafter(value, direction);
#endif
    }

    /**
     * Rho-dependent part of PROPOSAL 7.6.2
     * EpairKelnerKokoulinPetrukhin::FunctionToIntegral().
     *
     * CalculateRho() uses this parametrization even though CORSIKA's electron
     * and positron dN/dX columns use EpairForElectronPositron. Factors that are
     * independent of rho are omitted because they cancel in the normalized
     * conditional CDF.
     */
    CORSIKA_GPU_EPAIR_HOST_DEVICE inline double kkpRhoWeight(
        BremsLpmComponentSnapshot const& component,
        double lepton_mass_MeV, double electron_mass_MeV,
        double energy_MeV, double v, double rho) {
      if (!(rho >= 0.) || !(rho < 1.)) {
        return 0.;
      }
      auto r = 1. - rho;
      auto const r2 = r * r;
      auto const beta = v * v / (2. * (1. - v));
      auto const mass_auxiliary =
          lepton_mass_MeV * v / (2. * electron_mass_MeV);
      auto const xi =
          mass_auxiliary * mass_auxiliary * (1. - r2) /
          (1. - v);
      if (!finite(xi) || !(xi > 0.)) {
        return 0.;
      }

      auto const z_to_minus_third =
          power(component.nuclear_charge, -1. / 3.);
      auto const electron_y =
          (5. - r2 + 4. * beta * (1. + r2)) /
          (2. * (1. + 3. * beta) *
                   logarithm(3. + 1. / xi) -
           r2 - 2. * beta * (2. - r2));
      auto const lepton_y =
          (4. + r2 + 3. * beta * (1. + r2)) /
          ((1. + r2) * (1.5 + 2. * beta) *
               logarithm(3. + xi) +
           1. - 1.5 * r2);

      auto const screening_auxiliary =
          1.5 * electron_mass_MeV /
          (lepton_mass_MeV * z_to_minus_third);
      auto const electron_auxiliary =
          (1. + xi) * (1. + electron_y);
      auto const finite_energy_auxiliary =
          2. * electron_mass_MeV *
          1.648721270700128146848650787814163571 *
          component.radiation_log_constant *
          z_to_minus_third /
          (energy_MeV * v * (1. - r2));

      auto electron_diagram =
          logarithm(
              component.radiation_log_constant *
              z_to_minus_third *
              squareRoot(electron_auxiliary) /
              (1. + finite_energy_auxiliary *
                        electron_auxiliary)) -
          0.5 *
              logarithm(
                  1. +
                  screening_auxiliary *
                      screening_auxiliary *
                      electron_auxiliary);
      auto lepton_diagram =
          logarithm(
              component.radiation_log_constant /
              screening_auxiliary * z_to_minus_third /
              (1. + finite_energy_auxiliary *
                        (1. + xi) * (1. + lepton_y)));

      if (electron_diagram > 0.) {
        if (1. / xi < 1.e-5) {
          electron_diagram =
              (1.5 - r2 / 2. + beta * (1. + r2)) /
              xi * electron_diagram;
        } else {
          electron_diagram =
              (((2. + r2) * (1. + beta) +
                xi * (3. + r2)) *
                   logarithmOnePlus(1. / xi) +
               (1. - r2 - beta) / (1. + xi) -
               (3. + r2)) *
              electron_diagram;
        }
      } else {
        electron_diagram = 0.;
      }

      if (lepton_diagram > 0.) {
        lepton_diagram =
            (((1. + r2) * (1. + 1.5 * beta) -
              (1. + 2. * beta) * (1. - r2) / xi) *
                 logarithmOnePlus(xi) +
             xi * (1. - r2 - beta) / (1. + xi) +
             (1. + 2. * beta) * (1. - r2)) *
            lepton_diagram;
      } else {
        lepton_diagram = 0.;
      }

      auto const mass_ratio =
          electron_mass_MeV / lepton_mass_MeV;
      auto const result =
          electron_diagram +
          mass_ratio * mass_ratio * lepton_diagram;
      return finite(result) ? maximum(0., result) : 0.;
    }

    CORSIKA_GPU_EPAIR_HOST_DEVICE inline double
    transformedKkpRhoWeight(
        BremsLpmComponentSnapshot const& component,
        double lepton_mass_MeV, double electron_mass_MeV,
        double energy_MeV, double v, double rho_max,
        double uniform) {
      constexpr double Pi =
          3.141592653589793238462643383279502884;
      auto const half_sine = sine(0.5 * Pi * uniform);
      return kkpRhoWeight(
                 component, lepton_mass_MeV,
                 electron_mass_MeV, energy_MeV, v,
                 rho_max * half_sine * half_sine) *
             sine(Pi * uniform);
    }

    /**
     * Find the single maximum of the Jacobian-transformed KKP density.
     *
     * A coarse grid first brackets the peak. Golden-section refinement then
     * has fixed work and no convergence-dependent device control flow.
     */
    CORSIKA_GPU_EPAIR_HOST_DEVICE inline double
    estimateTransformedKkpMaximum(
        BremsLpmComponentSnapshot const& component,
        double lepton_mass_MeV, double electron_mass_MeV,
        double energy_MeV, double v, double rho_max) {
      double maximum_weight = 0.;
      int maximum_index = 0;
      for (int point = 0;
           point < EpairRejectionEnvelopePoints; ++point) {
        auto const uniform =
            (static_cast<double>(point) + 0.5) /
            static_cast<double>(
                EpairRejectionEnvelopePoints);
        auto const weight = transformedKkpRhoWeight(
            component, lepton_mass_MeV, electron_mass_MeV,
            energy_MeV, v, rho_max, uniform);
        if (weight > maximum_weight) {
          maximum_weight = weight;
          maximum_index = point;
        }
      }
      if (!(maximum_weight > 0.)) {
        return maximum_weight;
      }

      auto const inverse_points =
          1. /
          static_cast<double>(
              EpairRejectionEnvelopePoints);
      auto left = maximum(
          0.,
          (static_cast<double>(maximum_index) - 0.5) *
              inverse_points);
      auto right =
          (static_cast<double>(maximum_index) + 1.5) *
          inverse_points;
      if (right > 1.) {
        right = 1.;
      }
      constexpr double GoldenRatioConjugate =
          0.618033988749894848204586834365638118;
      auto first =
          right - GoldenRatioConjugate * (right - left);
      auto second =
          left + GoldenRatioConjugate * (right - left);
      auto first_weight = transformedKkpRhoWeight(
          component, lepton_mass_MeV, electron_mass_MeV,
          energy_MeV, v, rho_max, first);
      auto second_weight = transformedKkpRhoWeight(
          component, lepton_mass_MeV, electron_mass_MeV,
          energy_MeV, v, rho_max, second);
      maximum_weight = maximum(
          maximum_weight,
          maximum(first_weight, second_weight));
      for (int step = 0;
           step < EpairRejectionEnvelopeRefinementSteps;
           ++step) {
        if (first_weight < second_weight) {
          left = first;
          first = second;
          first_weight = second_weight;
          second =
              left +
              GoldenRatioConjugate * (right - left);
          second_weight = transformedKkpRhoWeight(
              component, lepton_mass_MeV,
              electron_mass_MeV, energy_MeV, v, rho_max,
              second);
          maximum_weight =
              maximum(maximum_weight, second_weight);
        } else {
          right = second;
          second = first;
          second_weight = first_weight;
          first =
              right -
              GoldenRatioConjugate * (right - left);
          first_weight = transformedKkpRhoWeight(
              component, lepton_mass_MeV,
              electron_mass_MeV, energy_MeV, v, rho_max,
              first);
          maximum_weight =
              maximum(maximum_weight, first_weight);
        }
      }
      return maximum_weight;
    }

    /**
     * Integrate in y=-log(1-rho). This resolves the increasingly narrow
     * high-rho tail without an energy-dependent table or endpoint samples.
     */
    CORSIKA_GPU_EPAIR_HOST_DEVICE inline double integrateRhoSegment(
        BremsLpmComponentSnapshot const& component,
        double lepton_mass_MeV, double electron_mass_MeV,
        double energy_MeV, double v, double y_lower,
        double y_upper) {
      constexpr double nodes[4]{
          0.18343464249564980493947614236018398,
          0.52553240991632898581773904918924635,
          0.79666647741362673959155393647583044,
          0.96028985649753623168356086856947299};
      constexpr double weights[4]{
          0.36268378337836198296515044927719561,
          0.31370664587788728733796220198660131,
          0.22238103445337447054435599442624088,
          0.10122853629037625915253135430996219};
      auto const center = 0.5 * (y_lower + y_upper);
      auto const half_width = 0.5 * (y_upper - y_lower);
      double sum = 0.;
      for (int index = 0; index < 4; ++index) {
        auto const displacement = half_width * nodes[index];
        auto const first_y = center - displacement;
        auto const second_y = center + displacement;
        auto const first_jacobian = exponential(-first_y);
        auto const second_jacobian = exponential(-second_y);
        auto const first_rho = 1. - first_jacobian;
        auto const second_rho = 1. - second_jacobian;
        sum += weights[index] *
               (kkpRhoWeight(
                    component, lepton_mass_MeV,
                    electron_mass_MeV, energy_MeV, v,
                    first_rho) *
                    first_jacobian +
                kkpRhoWeight(
                    component, lepton_mass_MeV,
                    electron_mass_MeV, energy_MeV, v,
                    second_rho) *
                    second_jacobian);
      }
      auto const result = half_width * sum;
      return finite(result) && result > 0. ? result : 0.;
    }

    /**
     * Polynomial interpolation used by PROPOSAL::Integral's open Romberg
     * integrator. Keeping the same point selection and correction ordering is
     * important: CalculateRho() exposes this numerical inverse CDF as part of
     * the seeded final-state result.
     */
    CORSIKA_GPU_EPAIR_HOST_DEVICE inline double
    proposalInterpolate(
        double const* x_values, double const* y_values, int start,
        int order, double x, double& error) {
      double c[5]{};
      double d[5]{};
      int nearest = 0;
      auto nearest_distance = absolute(x - x_values[start]);
      for (int index = 0; index < order; ++index) {
        auto const distance =
            absolute(x - x_values[start + index]);
        if (distance < nearest_distance) {
          nearest = index;
          nearest_distance = distance;
        }
        c[index] = y_values[start + index];
        d[index] = y_values[start + index];
      }

      bool take_correction_from_c = false;
      if (nearest == 0) {
        take_correction_from_c = true;
      } else if (nearest == order - 1) {
        take_correction_from_c = false;
      } else {
        take_correction_from_c =
            absolute(x - x_values[start + nearest - 1]) >
            absolute(x - x_values[start + nearest + 1]);
      }

      auto result = y_values[start + nearest];
      error = 0.;
      for (int level = 1; level < order; ++level) {
        for (int index = 0; index < order - level; ++index) {
          auto const first_delta =
              x_values[start + index] - x;
          auto const second_delta =
              x_values[start + index + level] - x;
          auto correction = c[index + 1] - d[index];
          auto const denominator =
              first_delta - second_delta;
          if (denominator != 0.) {
            correction /= denominator;
            c[index] = first_delta * correction;
            d[index] = second_delta * correction;
          } else {
            c[index] = 0.;
            d[index] = 0.;
          }
        }

        if (nearest == 0) {
          take_correction_from_c = true;
        }
        if (nearest == order - level) {
          take_correction_from_c = false;
        }
        if (take_correction_from_c) {
          error = c[nearest];
        } else {
          --nearest;
          error = d[nearest];
        }
        take_correction_from_c = !take_correction_from_c;
        result += error;
      }
      return result;
    }

    CORSIKA_GPU_EPAIR_HOST_DEVICE inline double
    proposalTrapezoid3(
        BremsLpmComponentSnapshot const& component,
        double lepton_mass_MeV, double electron_mass_MeV,
        double energy_MeV, double v, double lower, double upper,
        int point_count, double old_sum) {
      if (point_count == 1) {
        return (upper - lower) *
               kkpRhoWeight(
                   component, lepton_mass_MeV,
                   electron_mass_MeV, energy_MeV, v,
                   0.5 * (upper + lower));
      }

      auto const step_size =
          (upper - lower) / point_count;
      double sum = 0.;
      for (auto x = lower + 0.5 * step_size; x < upper;
           x += step_size) {
        sum += kkpRhoWeight(
            component, lepton_mass_MeV, electron_mass_MeV,
            energy_MeV, v, x);
        x += 2. * step_size;
        sum += kkpRhoWeight(
            component, lepton_mass_MeV, electron_mass_MeV,
            energy_MeV, v, x);
      }
      return old_sum / 3. + sum * step_size;
    }

    CORSIKA_GPU_EPAIR_HOST_DEVICE inline double
    proposalTrapezoid3WithRandomRatio(
        BremsLpmComponentSnapshot const& component,
        double lepton_mass_MeV, double electron_mass_MeV,
        double energy_MeV, double v, double lower, double upper,
        int point_count, double old_sum, int step_number,
        int romberg_order, double random_ratio,
        double& random_x) {
      constexpr double Precision = 1.e-6;
      if (point_count == 1) {
        return (upper - lower) *
               kkpRhoWeight(
                   component, lepton_mass_MeV,
                   electron_mass_MeV, energy_MeV, v,
                   0.5 * (upper + lower));
      }

      auto const step_size =
          (upper - lower) / point_count;
      double target_sum = 0.;
      if (step_number >= romberg_order - 1) {
        target_sum =
            random_ratio * old_sum / (1.5 * step_size);
      }

      double result_sum = 0.;
      double approximate_x = 0.;
      bool found = false;
      for (auto x = lower + 0.5 * step_size; x < upper;
           x += step_size) {
        auto const first_value = kkpRhoWeight(
            component, lepton_mass_MeV, electron_mass_MeV,
            energy_MeV, v, x);
        result_sum += first_value;
        x += 2. * step_size;
        auto const second_value = kkpRhoWeight(
            component, lepton_mass_MeV, electron_mass_MeV,
            energy_MeV, v, x);
        result_sum += second_value;

        if (!found &&
            step_number >= romberg_order - 1 &&
            absolute(result_sum) >= absolute(target_sum)) {
          auto const function_sum =
              first_value + second_value;
          auto integral_difference =
              target_sum -
              (result_sum - function_sum);
          integral_difference *= 1.5 * step_size;

          auto const function_difference =
              second_value - first_value;
          auto const quadratic =
              function_difference / step_size;
          auto const linear =
              (second_value - 5. * first_value) / 2.;
          auto const linear_squared = linear * linear;

          if (absolute(quadratic * integral_difference) <
              Precision * linear_squared) {
            approximate_x =
                integral_difference * 2. / function_sum;
          } else {
            auto determinant =
                linear_squared +
                4. * quadratic * integral_difference;
            if (determinant >= 0.) {
              determinant = squareRoot(determinant);
              if (linear == 0. && determinant == 0.) {
                approximate_x = 0.;
              } else {
                approximate_x =
                    (linear + determinant) / quadratic;
              }
              if (approximate_x < 0. ||
                  approximate_x > 3. * step_size) {
                approximate_x =
                    (linear - determinant) / quadratic;
              } else if (
                  approximate_x < 0. ||
                  approximate_x > 3. * step_size) {
                approximate_x =
                    integral_difference * 2. /
                    function_sum;
              }
            } else {
              approximate_x =
                  integral_difference * 2. /
                  function_sum;
            }
          }
          approximate_x += x - 2.5 * step_size;
          found = true;
        }
      }

      if (step_number >= romberg_order - 1) {
        if (!found) {
          approximate_x = nextToward(upper, lower);
        }
        random_x = approximate_x;
      }
      return old_sum / 3. + result_sum * step_size;
    }

    CORSIKA_GPU_EPAIR_HOST_DEVICE inline bool
    proposalRombergWithRandomRatio(
        BremsLpmComponentSnapshot const& component,
        double lepton_mass_MeV, double electron_mass_MeV,
        double energy_MeV, double v, double lower, double upper,
        double random_ratio, double& result, double& random_x) {
      constexpr int MaxSteps = 12;
      constexpr int RombergOrder = 5;
      constexpr double Precision = 1.e-6;
      double x_values[MaxSteps]{};
      double y_values[MaxSteps]{};
      int point_count = 1;
      double scale = 1.;
      result = 0.;

      for (int step = 0; step < MaxSteps; ++step) {
        result = proposalTrapezoid3WithRandomRatio(
            component, lepton_mass_MeV, electron_mass_MeV,
            energy_MeV, v, lower, upper, point_count, result,
            step, RombergOrder, random_ratio, random_x);
        x_values[step] = scale;
        y_values[step] = result;
        if (step >= RombergOrder - 1) {
          double error = 0.;
          auto const extrapolated = proposalInterpolate(
              x_values, y_values,
              step - (RombergOrder - 1), RombergOrder, 0.,
              error);
          auto relative_error = error;
          if (extrapolated != 0.) {
            relative_error /= extrapolated;
          }
          result = extrapolated;
          if (absolute(relative_error) < Precision) {
            return finite(result);
          }
        }
        point_count *= 3;
        scale /= 9.;
      }
      return false;
    }

    CORSIKA_GPU_EPAIR_HOST_DEVICE inline bool proposalRomberg(
        BremsLpmComponentSnapshot const& component,
        double lepton_mass_MeV, double electron_mass_MeV,
        double energy_MeV, double v, double lower, double upper,
        double reference_value, int romberg_order,
        double& result) {
      constexpr int MaxSteps = 12;
      constexpr double Precision = 1.e-6;
      if (lower == upper) {
        result = 0.;
        return true;
      }
      double x_values[MaxSteps]{};
      double y_values[MaxSteps]{};
      int point_count = 1;
      double scale = 1.;
      result = 0.;
      for (int step = 0; step < MaxSteps; ++step) {
        result = proposalTrapezoid3(
            component, lepton_mass_MeV, electron_mass_MeV,
            energy_MeV, v, lower, upper, point_count, result);
        x_values[step] = scale;
        y_values[step] = result;
        if (step >= romberg_order - 1) {
          double error = 0.;
          auto const extrapolated = proposalInterpolate(
              x_values, y_values,
              step - (romberg_order - 1), romberg_order, 0.,
              error);
          result = extrapolated;
          if (reference_value != 0. &&
              absolute(error / reference_value) < Precision) {
            return finite(result);
          }
        }
        point_count *= 3;
        scale /= 9.;
      }
      return false;
    }

    CORSIKA_GPU_EPAIR_HOST_DEVICE inline bool
    proposalRefineUpperLimit(
        BremsLpmComponentSnapshot const& component,
        double lepton_mass_MeV, double electron_mass_MeV,
        double energy_MeV, double v, double lower, double upper,
        double random_ratio, double total,
        int maximum_refinement_steps, double& random_x) {
      constexpr double Precision = 1.e-6;
      if (maximum_refinement_steps < 1 ||
          maximum_refinement_steps > 20) {
        return false;
      }
      auto lower_bracket = lower;
      auto upper_bracket = upper;
      auto lower_value = -random_ratio * total;
      auto const upper_value = (1. - random_ratio) * total;
      if (lower_value * upper_value > 0.) {
        return false;
      }
      if (lower_value > 0.) {
        auto const temporary = lower_bracket;
        lower_bracket = upper_bracket;
        upper_bracket = temporary;
      }

      auto delta = upper - lower;
      auto old_delta = delta;
      if (random_x < lower || random_x > upper) {
        random_x = 0.5 * (lower + upper);
      }
      auto current_x = random_x;
      double initial_integral = 0.;
      if (!proposalRomberg(
              component, lepton_mass_MeV,
              electron_mass_MeV, energy_MeV, v, lower,
              random_x, total, 5, initial_integral)) {
        return false;
      }
      auto const initial_random_x = random_x;
      auto const initial_function_value =
          initial_integral - random_ratio * total;
      auto function_value = initial_function_value;
      auto derivative = kkpRhoWeight(
          component, lepton_mass_MeV, electron_mass_MeV,
          energy_MeV, v, current_x);

      for (int step = 0;
           step < maximum_refinement_steps; ++step) {
        if (function_value < 0.) {
          lower_bracket = current_x;
        } else {
          upper_bracket = current_x;
        }

        if (((current_x - upper_bracket) * derivative -
             function_value) *
                    ((current_x - lower_bracket) * derivative -
                     function_value) >
                0. ||
            absolute(2. * function_value) >
                absolute(old_delta * derivative)) {
          old_delta = delta;
          delta =
              0.5 * (upper_bracket - lower_bracket);
          current_x = lower_bracket + delta;
          if (lower_bracket == current_x) {
            break;
          }
        } else {
          old_delta = delta;
          delta = function_value / derivative;
          auto const previous_x = current_x;
          current_x -= delta;
          if (previous_x == current_x) {
            break;
          }
        }

        if (absolute(delta) < Precision) {
          break;
        }

        auto segment_lower = initial_random_x;
        auto segment_upper = current_x;
        double sign = 1.;
        if (segment_lower > segment_upper) {
          auto const temporary = segment_lower;
          segment_lower = segment_upper;
          segment_upper = temporary;
          sign = -1.;
        }
        double segment_integral = 0.;
        if (!proposalRomberg(
                component, lepton_mass_MeV,
                electron_mass_MeV, energy_MeV, v,
                segment_lower, segment_upper, total, 2,
                segment_integral)) {
          return false;
        }
        function_value =
            initial_function_value +
            sign * segment_integral;
        derivative = kkpRhoWeight(
            component, lepton_mass_MeV,
            electron_mass_MeV, energy_MeV, v, current_x);
      }
      random_x = current_x;
      return finite(random_x) && random_x >= lower &&
             random_x <= upper;
    }

  } // namespace epair_final_state_detail

  /**
   * Sample the conditional pair asymmetry rho with the three-random-number
   * convention of PROPOSAL's default
   * KelnerKokoulinPetrukhinEpairProduction final-state generator.
   *
   * rho_uniform selects |rho|, sign_uniform selects its sign. The third
   * PROPOSAL random number affects only a direction routine that currently
   * returns the parent direction unchanged and is therefore audited by the
   * caller but is not an argument to this numerical sampler.
   */
  CORSIKA_GPU_EPAIR_HOST_DEVICE inline EpairFinalStateSample
  sampleEpairRhoImpl(
      BremsLpmSnapshot const& snapshot,
      std::uint64_t component_hash, double energy_MeV, double v,
      double rho_uniform, double sign_uniform,
      int refinement_steps) {
    using namespace epair_final_state_detail;
    if (snapshot.component_count == 0 ||
        snapshot.component_count > MaxBremsLpmComponents ||
        !finite(snapshot.lepton_mass_MeV) ||
        !(snapshot.lepton_mass_MeV > 0.) ||
        !finite(snapshot.electron_mass_MeV) ||
        !(snapshot.electron_mass_MeV > 0.)) {
      return {EpairFinalStateStatus::InvalidSnapshot, 0, 0., 0.};
    }
    if (!finite(energy_MeV) ||
        !(energy_MeV >
          snapshot.lepton_mass_MeV +
              2. * snapshot.electron_mass_MeV) ||
        !finite(v) || !(v > 0.) || !(v < 1.) ||
        !finite(rho_uniform) || !(rho_uniform > 0.) ||
        !(rho_uniform < 1.) ||
        !finite(sign_uniform) || !(sign_uniform > 0.) ||
        !(sign_uniform < 1.)) {
      return {EpairFinalStateStatus::InvalidInput, 0, 0., 0.};
    }

    BremsLpmComponentSnapshot component{};
    bool found = false;
    for (std::uint32_t index = 0;
         index < snapshot.component_count; ++index) {
      if (snapshot.components[index].component_hash ==
          component_hash) {
        component = snapshot.components[index];
        found = true;
        break;
      }
    }
    if (!found) {
      return {
          EpairFinalStateStatus::ComponentNotFound, 0, 0., 0.};
    }
    if (!finite(component.nuclear_charge) ||
        !(component.nuclear_charge > 0.) ||
        !finite(component.radiation_log_constant) ||
        !(component.radiation_log_constant > 0.)) {
      return {EpairFinalStateStatus::InvalidSnapshot, 0, 0., 0.};
    }

    auto const threshold_auxiliary =
        1. -
        4. * snapshot.electron_mass_MeV /
            (energy_MeV * v);
    auto const recoil_auxiliary =
        1. -
        6. * snapshot.lepton_mass_MeV *
            snapshot.lepton_mass_MeV /
            (energy_MeV * energy_MeV * (1. - v));
    if (!(threshold_auxiliary > 0.) ||
        !(recoil_auxiliary > 0.)) {
      return {EpairFinalStateStatus::InvalidInput, 0, 0., 0.};
    }
    auto const rho_max =
        squareRoot(threshold_auxiliary) * recoil_auxiliary;
    if (!finite(rho_max) || !(rho_max > 0.) ||
        !(rho_max < 1.)) {
      return {EpairFinalStateStatus::InvalidInput, 0, 0., 0.};
    }

    double total = 0.;
    double rho = 0.;
    if (!proposalRombergWithRandomRatio(
            component, snapshot.lepton_mass_MeV,
            snapshot.electron_mass_MeV, energy_MeV, v, 0.,
            rho_max, rho_uniform, total, rho) ||
        !finite(total)) {
      return {
          EpairFinalStateStatus::IntegrationFailed, 0, 0.,
          rho_max};
    }
    // PROPOSAL initializes rho to zero and changes it only when the numerical
    // integral is positive. Near threshold the KKP final-state weight can be
    // identically zero even though the rate model selected Epair.
    if (!(total > 0.)) {
      return {
          EpairFinalStateStatus::Success, 0, 0., rho_max};
    }
    if (refinement_steps > 0 &&
        !proposalRefineUpperLimit(
            component, snapshot.lepton_mass_MeV,
            snapshot.electron_mass_MeV, energy_MeV, v, 0.,
            rho_max, rho_uniform, total, refinement_steps,
            rho)) {
      return {
          EpairFinalStateStatus::IntegrationFailed, 0, 0.,
          rho_max};
    }
    if (sign_uniform < 0.5) {
      rho = -rho;
    }
    if (!finite(rho) || !(rho > -rho_max) ||
        !(rho < rho_max)) {
      return {
          EpairFinalStateStatus::NonFiniteResult, 0, 0.,
          rho_max};
    }
    return {EpairFinalStateStatus::Success, 0, rho, rho_max};
  }

  CORSIKA_GPU_EPAIR_HOST_DEVICE inline EpairFinalStateSample
  sampleEpairRho(
      BremsLpmSnapshot const& snapshot,
      std::uint64_t component_hash, double energy_MeV, double v,
      double rho_uniform, double sign_uniform) {
    return sampleEpairRhoImpl(
        snapshot, component_hash, energy_MeV, v, rho_uniform,
        sign_uniform, 20);
  }

  /**
   * Sample the same KKP rho density without numerically inverting its CDF.
   *
   * The substitution rho=rho_max*sin^2(pi*u/2) makes the integrable
   * rho->0 endpoint peak smooth after multiplication by the Jacobian. A
   * deterministic 64-point envelope and Philox rejection draws then sample
   * that transformed density directly. The 5% envelope margin is validated
   * over the complete production energy/loss domain. Any observed envelope
   * violation or bounded-loop exhaustion is reported to the caller so it can
   * request the exact CPU PROPOSAL final state instead of accepting bias.
   */
  CORSIKA_GPU_EPAIR_HOST_DEVICE inline EpairRejectionSample
  sampleEpairRhoRejection(
      BremsLpmSnapshot const& snapshot,
      std::uint64_t component_hash, double energy_MeV, double v,
      double sign_uniform, RandomNumberKey random_key) {
    using namespace epair_final_state_detail;
    if (snapshot.component_count == 0 ||
        snapshot.component_count > MaxBremsLpmComponents ||
        !finite(snapshot.lepton_mass_MeV) ||
        !(snapshot.lepton_mass_MeV > 0.) ||
        !finite(snapshot.electron_mass_MeV) ||
        !(snapshot.electron_mass_MeV > 0.)) {
      return {{
          EpairFinalStateStatus::InvalidSnapshot, 0, 0., 0.},
          0, 0};
    }
    if (!finite(energy_MeV) ||
        !(energy_MeV >
          snapshot.lepton_mass_MeV +
              2. * snapshot.electron_mass_MeV) ||
        !finite(v) || !(v > 0.) || !(v < 1.) ||
        !finite(sign_uniform) || !(sign_uniform > 0.) ||
        !(sign_uniform < 1.)) {
      return {{
          EpairFinalStateStatus::InvalidInput, 0, 0., 0.},
          0, 0};
    }

    BremsLpmComponentSnapshot component{};
    bool found = false;
    for (std::uint32_t index = 0;
         index < snapshot.component_count; ++index) {
      if (snapshot.components[index].component_hash ==
          component_hash) {
        component = snapshot.components[index];
        found = true;
        break;
      }
    }
    if (!found) {
      return {{
          EpairFinalStateStatus::ComponentNotFound, 0, 0., 0.},
          0, 0};
    }
    if (!finite(component.nuclear_charge) ||
        !(component.nuclear_charge > 0.) ||
        !finite(component.radiation_log_constant) ||
        !(component.radiation_log_constant > 0.)) {
      return {{
          EpairFinalStateStatus::InvalidSnapshot, 0, 0., 0.},
          0, 0};
    }

    auto const threshold_auxiliary =
        1. -
        4. * snapshot.electron_mass_MeV /
            (energy_MeV * v);
    auto const recoil_auxiliary =
        1. -
        6. * snapshot.lepton_mass_MeV *
            snapshot.lepton_mass_MeV /
            (energy_MeV * energy_MeV * (1. - v));
    if (!(threshold_auxiliary > 0.) ||
        !(recoil_auxiliary > 0.)) {
      return {{
          EpairFinalStateStatus::InvalidInput, 0, 0., 0.},
          0, 0};
    }
    auto const rho_max =
        squareRoot(threshold_auxiliary) * recoil_auxiliary;
    if (!finite(rho_max) || !(rho_max > 0.) ||
        !(rho_max < 1.)) {
      return {{
          EpairFinalStateStatus::InvalidInput, 0, 0.,
          rho_max},
          0, 0};
    }

    constexpr double Pi =
        3.141592653589793238462643383279502884;
    auto envelope = estimateTransformedKkpMaximum(
        component, snapshot.lepton_mass_MeV,
        snapshot.electron_mass_MeV, energy_MeV, v,
        rho_max);
    envelope *= EpairRejectionEnvelopeSafety;
    if (!finite(envelope) || envelope < 0.) {
      return {{
          EpairFinalStateStatus::
              RejectionEnvelopeUnavailable,
          0, 0., rho_max},
          0, 0};
    }
    // This matches PROPOSAL's CalculateRho convention in the near-threshold
    // region where the KKP diagram is zero over the admissible interval.
    if (envelope == 0.) {
      return {{
          EpairFinalStateStatus::Success, 0, 0., rho_max},
          0, 0};
    }

    for (int trial = 0;
         trial < EpairRejectionMaximumTrials; ++trial) {
      random_key.draw_id =
          EpairRejectionCandidateDrawIdBase +
          static_cast<std::uint64_t>(trial);
      auto const candidate_uniform =
          uniformOpen01(random_key);
      random_key.draw_id =
          EpairRejectionAcceptanceDrawIdBase +
          static_cast<std::uint64_t>(trial);
      auto const acceptance_uniform =
          uniformOpen01(random_key);
      auto const transformed_weight =
          transformedKkpRhoWeight(
              component, snapshot.lepton_mass_MeV,
              snapshot.electron_mass_MeV, energy_MeV, v,
              rho_max, candidate_uniform);
      if (!finite(transformed_weight) ||
          transformed_weight < 0.) {
        return {{
            EpairFinalStateStatus::NonFiniteResult, 0, 0.,
            rho_max},
            static_cast<std::uint32_t>(trial + 1), 0};
      }
      if (transformed_weight > envelope) {
        return {{
            EpairFinalStateStatus::
                RejectionEnvelopeExceeded,
            0, 0., rho_max},
            static_cast<std::uint32_t>(trial + 1), 0};
      }
      if (acceptance_uniform * envelope <=
          transformed_weight) {
        auto const half_sine =
            sine(0.5 * Pi * candidate_uniform);
        auto const rho =
            rho_max * half_sine * half_sine;
        auto signed_rho = rho;
        if (sign_uniform < 0.5) {
          signed_rho = -signed_rho;
        }
        return {{
            EpairFinalStateStatus::Success, 0, signed_rho,
            rho_max},
            static_cast<std::uint32_t>(trial + 1), 0};
      }
    }
    return {{
        EpairFinalStateStatus::RejectionTrialsExhausted,
        0, 0., rho_max},
        EpairRejectionMaximumTrials, 0};
  }

} // namespace corsika::gpu::em

#undef CORSIKA_GPU_EPAIR_HOST_DEVICE

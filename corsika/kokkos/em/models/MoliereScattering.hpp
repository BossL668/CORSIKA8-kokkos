/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once
#include <corsika/framework/core/HostDevice.hpp>
#include <corsika/kokkos/em/models/MoliereSnapshot.hpp>
#include <corsika/kokkos/em/tables/MoliereInterpolation.hpp>
#include <cstddef>
#include <cstdint>
#include <type_traits>
namespace corsika::kokkos {
  inline constexpr double MoliereMinimumB = 4.5;
  inline constexpr double MoliereMinimumBEquationRightHandSide = 2.9959226032237258;
  /** Original status/iteration/output contract; inspect the status before use. */
  enum class MoliereStatus : std::uint32_t {
    Success = 0,
    NoDeflection = 1,
    InvalidSnapshot = 2,
    InvalidInput = 3,
    NonConvergent = 4,
    NonFiniteResult = 5,
  };
  /** Original status/iteration/output contract; inspect the status before use. */
  struct MoliereAngleResult {
    MoliereStatus status_{MoliereStatus::InvalidSnapshot};
    std::uint32_t iterations_{};
    double angleRad_{};
  };
  /** Original status/iteration/output contract; inspect the status before use. */
  struct MoliereDirectionResult {
    MoliereStatus status_{MoliereStatus::InvalidInput};
    std::uint32_t reserved_{};
    double direction_[3]{};
  };
  static_assert(std::is_standard_layout_v<MoliereAngleResult>);
  static_assert(std::is_trivially_copyable_v<MoliereAngleResult>);
  static_assert(std::is_standard_layout_v<MoliereDirectionResult>);
  static_assert(std::is_trivially_copyable_v<MoliereDirectionResult>);
  namespace moliere_detail {
    inline constexpr double Pi = moliere_snapshot_detail::Pi;
    inline constexpr double SqrtTwo = 1.414213562373095048801688724209698079;
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
      double inverseScale_[ComponentCapacity]{};
      double inverseB_[ComponentCapacity]{};
      double densityWeight_[ComponentCapacity]{};
      double prefactor_{};
      double betaSquared_{};
    };
    using distribution_state_type = DistributionStateForCapacity<MaxMoliereComponents>;
    template <std::size_t ComponentCapacity>
    struct DistributionEvaluation {
      double density_{};
      double cumulative_{};
    };
    CORSIKA_HOST_DEVICE inline bool finite(double value);
    CORSIKA_HOST_DEVICE inline double absolute(double value);
    CORSIKA_HOST_DEVICE inline double square_root(double value);
    CORSIKA_HOST_DEVICE inline double logarithm(double value);
    CORSIKA_HOST_DEVICE inline double exponential(double value);
    CORSIKA_HOST_DEVICE inline double power(double base, double exponent);
    CORSIKA_HOST_DEVICE inline double error_function(double value);
    CORSIKA_HOST_DEVICE inline bool valid_interpolation_view(
        MoliereInterpolationView const& interpolation);
    CORSIKA_HOST_DEVICE inline bool has_interpolation_view(
        MoliereInterpolationView const& interpolation);
    CORSIKA_HOST_DEVICE inline double evaluate_interpolation(
        NativeArrayView<MoliereCubicPolynomial> const& polynomials,
        MoliereInterpolationView const& interpolation, double x);
    CORSIKA_HOST_DEVICE inline double improve_moliere_initial_guess(
        MoliereInterpolationView const& interpolation, double reference_B,
        double beta_squared, double gaussian_coordinate);
    CORSIKA_HOST_DEVICE inline double normal_ppf(double probability);
    /** Solve B-log(B)=rhs on the physical B>1 branch.
     * The production fourth-order asymptotic seed receives one Halley update;
     * rhs must be at least 4.5-log(4.5), and returned physical B is at least 4.5.
     * This is the existing production solver, not PROPOSAL's six-Newton-update
     * implementation. Invalid input or an unusable branch returns zero. */
    CORSIKA_HOST_DEVICE inline double solve_moliere_b(double rhs);
    CORSIKA_HOST_DEVICE inline double f1_m(MoliereSnapshot const& snapshot, double x);
    CORSIKA_HOST_DEVICE inline double f2_m(MoliereSnapshot const& snapshot, double x);
    CORSIKA_HOST_DEVICE inline double integral_f1_m(MoliereSnapshot const& snapshot,
                                                    double x);
    CORSIKA_HOST_DEVICE inline double integral_f2_m(MoliereSnapshot const& snapshot,
                                                    double x);
    CORSIKA_HOST_DEVICE inline double f1_m(MoliereSnapshot const& snapshot,
                                           MoliereInterpolationView const& interpolation,
                                           double x);
    CORSIKA_HOST_DEVICE inline double f2_m(MoliereSnapshot const& snapshot,
                                           MoliereInterpolationView const& interpolation,
                                           double x);
    CORSIKA_HOST_DEVICE inline double integral_f1_m(
        MoliereSnapshot const& snapshot, MoliereInterpolationView const& interpolation,
        double x);
    CORSIKA_HOST_DEVICE inline double integral_f2_m(
        MoliereSnapshot const& snapshot, MoliereInterpolationView const& interpolation,
        double x);
    template <std::size_t ComponentCapacity>
    CORSIKA_HOST_DEVICE inline MoliereStatus build_distribution(
        MoliereSnapshot const& snapshot, double grammage_g_per_cm2,
        double initial_energy_MeV, DistributionStateForCapacity<ComponentCapacity>& state,
        bool validate_snapshot = true);
    template <std::size_t ComponentCapacity>
    CORSIKA_HOST_DEVICE inline DistributionEvaluation<ComponentCapacity>
    evaluate_distribution(MoliereSnapshot const& snapshot,
                          MoliereInterpolationView const& interpolation,
                          DistributionStateForCapacity<ComponentCapacity> const& state,
                          double theta);
    template <std::size_t ComponentCapacity>
    CORSIKA_HOST_DEVICE inline MoliereAngleResult sample_one_dimensional(
        MoliereSnapshot const& snapshot, MoliereInterpolationView const& interpolation,
        DistributionStateForCapacity<ComponentCapacity> const& state, double uniform);
  } // namespace moliere_detail

  /** Sample two independent projected angles and return their radial norm.
   * Energies are total MeV, grammage g/cm^2, and returned angles radians.
   * final_energy_MeV is deliberately unused, retaining production semantics.
   * A default empty view selects the analytical functions. Uniforms are in (0,1).
   * Skipping snapshot validation requires previously validated, stable inputs. */
  template <std::size_t ComponentCapacity>
  [[nodiscard]] CORSIKA_HOST_DEVICE inline MoliereAngleResult
  sample_moliere_scattering_angle_2d_for_capacity(
      MoliereSnapshot const& snapshot, MoliereInterpolationView const& interpolation,
      double grammage_g_per_cm2, double initial_energy_MeV, double final_energy_MeV,
      double first_uniform, double second_uniform, bool validate_snapshot = true);
  /** Sample using the full component capacity and an explicit interpolation view. */
  [[nodiscard]] CORSIKA_HOST_DEVICE inline MoliereAngleResult
  sample_moliere_scattering_angle_2d(MoliereSnapshot const& snapshot,
                                     MoliereInterpolationView const& interpolation,
                                     double grammage_g_per_cm2, double initial_energy_MeV,
                                     double final_energy_MeV, double first_uniform,
                                     double second_uniform,
                                     bool validate_snapshot = true);
  /** Sample with analytical correction functions and the original Gaussian seed. */
  [[nodiscard]] CORSIKA_HOST_DEVICE inline MoliereAngleResult
  sample_moliere_scattering_angle_2d(MoliereSnapshot const& snapshot,
                                     double grammage_g_per_cm2, double initial_energy_MeV,
                                     double final_energy_MeV, double first_uniform,
                                     double second_uniform,
                                     bool validate_snapshot = true);
  /** Apply the original two rotations to a unit direction in one common frame.
   * The angle is in radians; azimuth_uniform is in (0,1). The original x>0.1
   * axis-selection branch and final direction normalization are retained. */
  [[nodiscard]] CORSIKA_HOST_DEVICE inline MoliereDirectionResult apply_moliere_direction(
      double const (&initial_direction)[3], double angle_rad, double azimuth_uniform);
  /** Match Step's accumulated tracking and scattering direction increments.
   * Return normalize(tracked + scattered_from_start - start), not a rotation
   * of the tracked endpoint. Both input directions use the same frame. */
  [[nodiscard]] CORSIKA_HOST_DEVICE inline MoliereDirectionResult
  apply_moliere_direction_cpu_step_compatible(double const (&start_direction)[3],
                                              double const (&tracked_direction)[3],
                                              double angle_rad, double azimuth_uniform);
} // namespace corsika::kokkos
#include <corsika/detail/kokkos/em/models/MoliereScattering.inl>

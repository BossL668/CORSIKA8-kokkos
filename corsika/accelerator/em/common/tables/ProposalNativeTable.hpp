/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <type_traits>
#include <vector>

#include <corsika/accelerator/AcceleratorMacros.hpp>
#include <corsika/accelerator/em/common/tables/Sha256.hpp>

#define CORSIKA_PROPOSAL_NATIVE_HOST_DEVICE C8_ACCELERATOR_INLINE_FUNCTION

namespace corsika::gpu::em::tables {

  // v6 additionally preserves Displacement::FunctionToIntegral()'s exact
  // cross-list/component dE/dX accumulation order.  Descriptors remain in
  // canonical identity order; a compact index permutation restores the
  // scalar floating-point summation order on the device.
  inline constexpr std::uint32_t ProposalNativeTableFormatVersion = 6;
  inline constexpr char ProposalNativePhysicsSource[] = "proposal-native";
  inline constexpr std::uint64_t NoNativePolynomialCoefficients =
      std::numeric_limits<std::uint64_t>::max();

  enum class NativeAxisType : std::uint32_t {
    Linear = 0,
    Exponential = 1,
    ExponentialMinusOne = 2,
  };

  enum class NativeLossTransform : std::uint32_t {
    RelativeLog = 0,
    RelativeLogPowerOnePointFive = 1,
    OneMinusLog = 2,
  };

  enum class NativeKinematicModel : std::uint32_t {
    Unsupported = 0,
    OnlyStochastic = 1,
    // PhotoPairKochMotz::CalculateStochasticLoss() in PROPOSAL 7.6.2
    // returns one: the stochastic photon is consumed completely.  The
    // exported cumulative dN/dX spline is still used for the interaction
    // rate, but it is not an inverse CDF for an externally observable loss
    // fraction.
    PhotonPairFullLoss = 2,
    Compton = 3,
    BremsElectronScreening = 4,
    ElectronPair = 5,
    IonizationMoller = 6,
    IonizationBhabha = 7,
    IonizationBetheBlochRossi = 8,
    BremsKelnerKokoulinPetrukhin = 9,
    PhotonuclearAllm97 = 10,
  };

  enum class NativeRateModel : std::uint32_t {
    BicubicSpline = 0,
    AnnihilationHeitler = 1,
    PhotoeffectSauter = 2,
    PhotoproductionHeckC7Shadowing = 3,
  };

  enum class NativeQueryStatus : std::uint32_t {
    Success = 0,
    InvalidView = 1,
    NonFiniteInput = 2,
    ParticleNotFound = 3,
    ColumnNotFound = 4,
    EnergyOutOfRange = 5,
    QuantileOutOfRange = 6,
    UnsupportedKinematics = 7,
    RootNotConverged = 8,
  };

  struct NativeQueryResult {
    NativeQueryStatus status{NativeQueryStatus::InvalidView};
    std::uint32_t newton_iterations{};
    std::uint32_t bisection_iterations{};
    double value{};
  };

  struct NativeAxisDescriptor {
    NativeAxisType type{NativeAxisType::Linear};
    std::uint32_t nodes{};
    double low{};
    double high{};
    double step{};
  };

  struct NativeBicubicDescriptor {
    NativeAxisDescriptor energy_axis{};
    NativeAxisDescriptor loss_axis{};
    std::uint32_t rows{};
    std::uint32_t columns{};
    std::uint64_t coefficient_offset{};
    // Optional per-cell polynomial coefficients, derived once on the host
    // with CubicInterpolation 0.1.5's Eigen expression.  The original spline
    // arrays above remain authoritative and are always retained.
    std::uint64_t polynomial_coefficient_offset{
        NoNativePolynomialCoefficients};
  };

  struct NativeCubicDescriptor {
    NativeAxisDescriptor axis{};
    NativeAxisDescriptor value_transform{};
    std::uint64_t coefficient_offset{};
    std::uint32_t coefficient_count{};
    std::uint32_t has_value_transform{};
  };

  struct NativeDndxColumn {
    std::int32_t pdg_id{};
    std::int32_t process_id{};
    std::uint64_t medium_hash{};
    std::uint64_t component_hash{};
    std::uint64_t cross_section_hash{};
    std::uint64_t table_hash{};
    NativeLossTransform loss_transform{NativeLossTransform::RelativeLog};
    NativeKinematicModel kinematic_model{NativeKinematicModel::Unsupported};
    NativeRateModel rate_model{NativeRateModel::BicubicSpline};
    NativeBicubicDescriptor spline{};
    double component_weight{1.};
    double particle_mass_MeV{};
    double component_nuclear_charge{};
    double component_atomic_mass{};
    double component_average_nucleon_mass_MeV{};
    double direct_rate_factor{};
    double lower_energy_limit_MeV{};
    double absolute_cut_MeV{};
    double relative_cut{};
    // PROPOSAL Medium::GetI() converted from eV to MeV.  The native
    // Bethe--Bloch--Rossi ionization boundary is medium dependent through
    // v_min = I / E, so it cannot be reconstructed from component data.
    double medium_mean_excitation_energy_MeV{};
    // Stable ordinal in Interaction::Rates(): cross_list insertion order,
    // followed by the exact per-target order returned by each cross section.
    // Identity lookup remains canonical and uses process/component instead.
    std::uint32_t selection_ordinal{};
    std::uint32_t reserved{};
  };

  struct NativeTotalRateColumn {
    std::int32_t pdg_id{};
    std::uint32_t reserved{};
    std::uint64_t medium_hash{};
    std::uint64_t table_hash{};
    NativeCubicDescriptor spline{};
    double lower_energy_limit_MeV{};
  };

  struct NativeDedxColumn {
    std::int32_t pdg_id{};
    std::int32_t process_id{};
    std::uint64_t medium_hash{};
    std::uint64_t cross_section_hash{};
    // Stable ordinal in PROPOSAL's underlying per-component dE/dX vector.
    // It is assigned before non-interpolant entries are skipped, so it is an
    // identity rather than merely an index in the exported subset.
    std::uint64_t component_index{};
    std::uint64_t table_hash{};
    NativeCubicDescriptor spline{};
    double component_weight{1.};
    double lower_energy_limit_MeV{};
    std::uint32_t accumulation_ordinal{};
    std::uint32_t reserved{};
  };

  struct NativeUtilityColumn {
    std::int32_t pdg_id{};
    std::uint64_t medium_hash{};
    std::uint64_t table_hash{};
    NativeCubicDescriptor spline{};
    double particle_mass_MeV{};
    double lower_energy_limit_MeV{};
    std::uint32_t reverse{};
  };

  struct NativePhysicalConstants {
    double electron_mass_MeV{};
    double charged_pion_mass_MeV{};
    double fine_structure_constant{};
    double classical_electron_radius_cm{};
    double pi{};
    double sqrt_e{};
  };

  struct NativeInteractionIdentity {
    std::int32_t pdg_id{};
    std::uint32_t reserved{};
    std::uint64_t medium_hash{};
    std::uint64_t interaction_hash{};
  };

  /** Owning, host-only canonical export of the calculators used by CORSIKA. */
  struct ProposalNativeTableSet {
    std::uint32_t format_version{ProposalNativeTableFormatVersion};
    std::string proposal_version;
    std::string cubic_interpolation_version;
    NativePhysicalConstants constants{};
    double stochastic_cut_MeV{};
    Sha256Digest content_hash{};
    // Diagnostic state is deliberately excluded from content_hash: a cold
    // build and a warm PROPOSAL cache load must export identical physics.
    std::uint64_t proposal_cache_table_count{};
    std::uint64_t proposal_cache_hit_count{};
    bool proposal_cache_all_hit{};
    std::vector<NativeInteractionIdentity> interaction_identities;
    std::vector<NativeDndxColumn> dndx_columns;
    std::vector<NativeTotalRateColumn> total_rate_columns;
    // Index indirection that walks dndx_columns in Interaction::Rates order.
    // Entries remain grouped by PID and have one-to-one coverage.
    std::vector<std::uint32_t> selection_column_indices;
    std::vector<NativeDedxColumn> dedx_columns;
    // Index indirection that walks dedx_columns in scalar PROPOSAL's exact
    // cross-list/component summation order. Entries remain grouped by PID.
    std::vector<std::uint32_t> dedx_accumulation_indices;
    std::vector<NativeUtilityColumn> utility_columns;
    // All bicubic arrays use the same column-major offset and element count.
    std::vector<double> bicubic_values;
    std::vector<double> bicubic_derivative_energy;
    std::vector<double> bicubic_derivative_loss;
    std::vector<double> bicubic_mixed_derivative;
    // Sixteen row-major coefficients per cell.  Cells are ordered with the
    // energy interval varying fastest: energy + (rows - 1) * loss.
    std::vector<double> bicubic_polynomial_coefficients;
    // Every 1-D interval is represented exactly by node values and first
    // derivatives copied from the cardinal cubic B-spline.
    std::vector<double> cubic_values;
    std::vector<double> cubic_node_derivatives;
  };

  /** Pointer-only POD copied into kernels. */
  struct ProposalNativeDeviceView {
    NativeDndxColumn const* dndx_columns{};
    NativeTotalRateColumn const* total_rate_columns{};
    std::uint32_t const* selection_column_indices{};
    NativeDedxColumn const* dedx_columns{};
    std::uint32_t const* dedx_accumulation_indices{};
    NativeUtilityColumn const* utility_columns{};
    std::uint32_t dndx_column_count{};
    std::uint32_t total_rate_column_count{};
    std::uint32_t selection_column_index_count{};
    std::uint32_t dedx_column_count{};
    std::uint32_t dedx_accumulation_index_count{};
    std::uint32_t utility_column_count{};
    double const* bicubic_values{};
    double const* bicubic_derivative_energy{};
    double const* bicubic_derivative_loss{};
    double const* bicubic_mixed_derivative{};
    std::uint64_t bicubic_coefficient_count{};
    double const* bicubic_polynomial_coefficients{};
    std::uint64_t bicubic_polynomial_coefficient_count{};
    double const* cubic_values{};
    double const* cubic_node_derivatives{};
    std::uint64_t cubic_coefficient_count{};
    NativePhysicalConstants constants{};
  };

  static_assert(std::is_standard_layout_v<NativeAxisDescriptor>);
  static_assert(std::is_trivially_copyable_v<NativeAxisDescriptor>);
  static_assert(std::is_standard_layout_v<NativeDndxColumn>);
  static_assert(std::is_trivially_copyable_v<NativeDndxColumn>);
  static_assert(std::is_standard_layout_v<ProposalNativeDeviceView>);
  static_assert(std::is_trivially_copyable_v<ProposalNativeDeviceView>);

  ProposalNativeDeviceView makeProposalNativeHostView(
      ProposalNativeTableSet const& table);
  std::size_t proposalNativeTableBytes(ProposalNativeTableSet const& table);
  void validateProposalNativeTable(ProposalNativeTableSet const& table);
  Sha256Digest calculateProposalNativeHash(
      ProposalNativeTableSet const& table);

  /**
   * Resolve the version-locked PROPOSAL process/PID/parametrization contract
   * used by the native exporter.  This host-only gate deliberately throws for
   * every tuple that is not explicitly supported; it is exposed so validation
   * tests can prove that an unknown parametrization cannot be uploaded as a
   * superficially valid rate column.
   */
  NativeKinematicModel classifyProposalNativeKinematicModel(
      std::int32_t process_id, std::string const& parametrization,
      std::int32_t pdg_id);

  namespace native_detail {

    CORSIKA_PROPOSAL_NATIVE_HOST_DEVICE inline bool finite(double x) {
#if defined(__CUDA_ARCH__)
      return ::isfinite(x);
#else
      return std::isfinite(x);
#endif
    }

    CORSIKA_PROPOSAL_NATIVE_HOST_DEVICE inline double logarithm(double x) {
#if defined(__CUDA_ARCH__)
      return ::log(x);
#else
      return std::log(x);
#endif
    }

    CORSIKA_PROPOSAL_NATIVE_HOST_DEVICE inline double exponential(double x) {
#if defined(__CUDA_ARCH__)
      return ::exp(x);
#else
      return std::exp(x);
#endif
    }

    CORSIKA_PROPOSAL_NATIVE_HOST_DEVICE inline double logarithmOnePlus(
        double x) {
#if defined(__CUDA_ARCH__)
      return ::log1p(x);
#else
      return std::log1p(x);
#endif
    }

    CORSIKA_PROPOSAL_NATIVE_HOST_DEVICE inline double exponentialMinusOne(
        double x) {
#if defined(__CUDA_ARCH__)
      return ::expm1(x);
#else
      return std::expm1(x);
#endif
    }

    CORSIKA_PROPOSAL_NATIVE_HOST_DEVICE inline double power(
        double x, double exponent) {
#if defined(__CUDA_ARCH__)
      return ::pow(x, exponent);
#else
      return std::pow(x, exponent);
#endif
    }

    CORSIKA_PROPOSAL_NATIVE_HOST_DEVICE inline double squareRoot(double x) {
#if defined(__CUDA_ARCH__)
      return ::sqrt(x);
#else
      return std::sqrt(x);
#endif
    }

    CORSIKA_PROPOSAL_NATIVE_HOST_DEVICE inline double hyperbolicSine(double x) {
#if defined(__CUDA_ARCH__)
      return ::sinh(x);
#else
      return std::sinh(x);
#endif
    }

    CORSIKA_PROPOSAL_NATIVE_HOST_DEVICE inline double arctangent(double x) {
#if defined(__CUDA_ARCH__)
      return ::atan(x);
#else
      return std::atan(x);
#endif
    }

    CORSIKA_PROPOSAL_NATIVE_HOST_DEVICE inline double axisTransform(
        NativeAxisDescriptor const& axis, double x) {
      switch (axis.type) {
      case NativeAxisType::Linear:
        return (x - axis.low) / axis.step;
      case NativeAxisType::Exponential:
        return logarithm(x / axis.low) / axis.step;
      case NativeAxisType::ExponentialMinusOne:
        return (logarithmOnePlus(x / axis.low) -
                0.693147180559945309417232121458176568) /
               axis.step;
      }
      return 0. / 0.;
    }

    CORSIKA_PROPOSAL_NATIVE_HOST_DEVICE inline double axisBackTransform(
        NativeAxisDescriptor const& axis, double coordinate) {
      switch (axis.type) {
      case NativeAxisType::Linear:
        return coordinate * axis.step + axis.low;
      case NativeAxisType::Exponential:
        return axis.low * exponential(coordinate * axis.step);
      case NativeAxisType::ExponentialMinusOne:
        return axis.low * exponentialMinusOne(
                              coordinate * axis.step +
                              0.693147180559945309417232121458176568);
      }
      return 0. / 0.;
    }

    CORSIKA_PROPOSAL_NATIVE_HOST_DEVICE inline double axisDerivative(
        NativeAxisDescriptor const& axis, double x) {
      switch (axis.type) {
      case NativeAxisType::Linear:
        return 1. / axis.step;
      case NativeAxisType::Exponential:
        return 1. / (x * axis.step);
      case NativeAxisType::ExponentialMinusOne:
        return 1. / (axis.step * (x + axis.low));
      }
      return 0. / 0.;
    }

    CORSIKA_PROPOSAL_NATIVE_HOST_DEVICE inline double axisBackDerivative(
        NativeAxisDescriptor const& axis, double coordinate) {
      switch (axis.type) {
      case NativeAxisType::Linear:
        return axis.step;
      case NativeAxisType::Exponential:
        return axis.low * axis.step * exponential(coordinate * axis.step);
      case NativeAxisType::ExponentialMinusOne:
        return axis.low * axis.step *
               exponential(coordinate * axis.step +
                           0.693147180559945309417232121458176568);
      }
      return 0. / 0.;
    }

    CORSIKA_PROPOSAL_NATIVE_HOST_DEVICE inline std::uint32_t intervalNode(
        double coordinate, std::uint32_t nodes) {
      auto node = static_cast<long long>(coordinate >= 0. ? coordinate : -1.);
      if (coordinate < 0.) node = 0;
      if (node > static_cast<long long>(nodes) - 2)
        node = static_cast<long long>(nodes) - 2;
      return static_cast<std::uint32_t>(node);
    }

    CORSIKA_PROPOSAL_NATIVE_HOST_DEVICE inline void hermiteBasis(
        double x, double (&basis)[4]) {
      auto const x2 = x * x;
      auto const x3 = x2 * x;
      basis[0] = 1. - 3. * x2 + 2. * x3;
      basis[1] = 3. * x2 - 2. * x3;
      basis[2] = x - 2. * x2 + x3;
      basis[3] = -x2 + x3;
    }

    CORSIKA_PROPOSAL_NATIVE_HOST_DEVICE inline void hermiteBasisDerivative(
        double x, double (&basis)[4]) {
      auto const x2 = x * x;
      basis[0] = -6. * x + 6. * x2;
      basis[1] = 6. * x - 6. * x2;
      basis[2] = 1. - 4. * x + 3. * x2;
      basis[3] = -2. * x + 3. * x2;
    }

    CORSIKA_PROPOSAL_NATIVE_HOST_DEVICE inline double bicubicElement(
        ProposalNativeDeviceView const&,
        NativeBicubicDescriptor const& spline,
        double const* array, std::uint32_t row, std::uint32_t column) {
      auto const local =
          static_cast<std::uint64_t>(row) +
          static_cast<std::uint64_t>(spline.rows) * column;
      return array[spline.coefficient_offset + local];
    }

    CORSIKA_PROPOSAL_NATIVE_HOST_DEVICE inline double evaluateBicubic(
        ProposalNativeDeviceView const& view,
        NativeBicubicDescriptor const& spline,
        double energy_MeV, double transformed_loss) {
      auto const x0 = axisTransform(spline.energy_axis, energy_MeV);
      auto const x1 = axisTransform(spline.loss_axis, transformed_loss);
      auto const n0 = intervalNode(x0, spline.rows);
      auto const n1 = intervalNode(x1, spline.columns);
      double h0[4]{};
      double h1[4]{};
      hermiteBasis(x0 - n0, h0);
      hermiteBasis(x1 - n1, h1);
      double result = 0.;
      for (std::uint32_t a = 0; a < 4; ++a) {
        for (std::uint32_t b = 0; b < 4; ++b) {
          auto const row = n0 + (a & 1u);
          auto const column = n1 + (b & 1u);
          double const* coefficients = view.bicubic_values;
          if (a >= 2 && b < 2)
            coefficients = view.bicubic_derivative_energy;
          else if (a < 2 && b >= 2)
            coefficients = view.bicubic_derivative_loss;
          else if (a >= 2 && b >= 2)
            coefficients = view.bicubic_mixed_derivative;
          result += h0[a] *
                    bicubicElement(view, spline, coefficients, row, column) *
                    h1[b];
        }
      }
      return result;
    }

    CORSIKA_PROPOSAL_NATIVE_HOST_DEVICE inline double dotFourSequential(
        double a0, double a1, double a2, double a3,
        double b0, double b1, double b2, double b3) {
      auto const p0 = a0 * b0;
      auto const p1 = a1 * b1;
      auto const p2 = a2 * b2;
      auto const p3 = a3 * b3;
      return ((p0 + p1) + p2) + p3;
    }

    /** Evaluate host-precomputed CubicInterpolation polynomial coefficients. */
    CORSIKA_PROPOSAL_NATIVE_HOST_DEVICE inline double
    evaluateBicubicPrecomputed(
        ProposalNativeDeviceView const& view,
        NativeBicubicDescriptor const& spline,
        double energy_MeV, double transformed_loss) {
      if (spline.polynomial_coefficient_offset ==
              NoNativePolynomialCoefficients ||
          view.bicubic_polynomial_coefficients == nullptr)
        return 0. / 0.;
      auto const x0 = axisTransform(spline.energy_axis, energy_MeV);
      auto const x1 = axisTransform(spline.loss_axis, transformed_loss);
      auto const n0 = intervalNode(x0, spline.rows);
      auto const n1 = intervalNode(x1, spline.columns);
      auto const cell = static_cast<std::uint64_t>(n0) +
                        static_cast<std::uint64_t>(spline.rows - 1u) * n1;
      auto const offset = spline.polynomial_coefficient_offset + 16u * cell;
      if (offset > view.bicubic_polynomial_coefficient_count ||
          16u > view.bicubic_polynomial_coefficient_count - offset)
        return 0. / 0.;
      auto const* coefficients =
          view.bicubic_polynomial_coefficients + offset;
      auto const f0 = x0 - n0;
      auto const f1 = x1 - n1;
      double const v0[4]{1., f0, f0 * f0, f0 * f0 * f0};
      double const v1[4]{1., f1, f1 * f1, f1 * f1 * f1};
      double projected[4]{};
      for (int row = 0; row < 4; ++row)
        projected[row] = dotFourSequential(
            coefficients[4 * row], coefficients[4 * row + 1],
            coefficients[4 * row + 2], coefficients[4 * row + 3],
            v1[0], v1[1], v1[2], v1[3]);
      return dotFourSequential(
          v0[0], v0[1], v0[2], v0[3], projected[0], projected[1],
          projected[2], projected[3]);
    }

    CORSIKA_PROPOSAL_NATIVE_HOST_DEVICE inline double
    evaluateBicubicProposal76(
        ProposalNativeDeviceView const& view,
        NativeBicubicDescriptor const& spline,
        double energy_MeV, double transformed_loss) {
      if (spline.polynomial_coefficient_offset !=
              NoNativePolynomialCoefficients &&
          view.bicubic_polynomial_coefficients != nullptr)
        return evaluateBicubicPrecomputed(
            view, spline, energy_MeV, transformed_loss);
      return evaluateBicubic(view, spline, energy_MeV, transformed_loss);
    }


    CORSIKA_PROPOSAL_NATIVE_HOST_DEVICE inline double
    evaluateBicubicLossDerivative(
        ProposalNativeDeviceView const& view,
        NativeBicubicDescriptor const& spline,
        double energy_MeV, double transformed_loss) {
      auto const x0 = axisTransform(spline.energy_axis, energy_MeV);
      auto const x1 = axisTransform(spline.loss_axis, transformed_loss);
      auto const n0 = intervalNode(x0, spline.rows);
      auto const n1 = intervalNode(x1, spline.columns);
      double h0[4]{};
      double dh1[4]{};
      hermiteBasis(x0 - n0, h0);
      hermiteBasisDerivative(x1 - n1, dh1);
      double result = 0.;
      for (std::uint32_t a = 0; a < 4; ++a) {
        for (std::uint32_t b = 0; b < 4; ++b) {
          auto const row = n0 + (a & 1u);
          auto const column = n1 + (b & 1u);
          double const* coefficients = view.bicubic_values;
          if (a >= 2 && b < 2)
            coefficients = view.bicubic_derivative_energy;
          else if (a < 2 && b >= 2)
            coefficients = view.bicubic_derivative_loss;
          else if (a >= 2 && b >= 2)
            coefficients = view.bicubic_mixed_derivative;
          result += h0[a] *
                    bicubicElement(view, spline, coefficients, row, column) *
                    dh1[b];
        }
      }
      // Loss axes in PROPOSAL 7.6.2 are linear. Keep the generic linear
      // coordinate scaling explicit and reject other axes during validation.
      return result / spline.loss_axis.step;
    }

    /**
     * CubicInterpolation 0.1.5 does not use the analytic Bicubic Hermite
     * derivative in find_parameter().  Its BicubicSplines::prime() calls
     * Boost's default sixth-order finite-difference derivative.  Keep this
     * seemingly redundant sampling here because it is part of the scalar
     * PROPOSAL interpolation semantics and affects the final 20-bit Newton
     * stopping point used by a decision tape.
     */
    CORSIKA_PROPOSAL_NATIVE_HOST_DEVICE inline double
    evaluateBicubicLossDerivativeProposal76(
        ProposalNativeDeviceView const& view,
        NativeBicubicDescriptor const& spline,
        double energy_MeV, double transformed_loss) {
      constexpr double epsilon = 2.220446049250313080847263336181640625e-16;
      auto h = power(epsilon / 168., 1. / 7.);
      auto const coordinate = axisTransform(spline.loss_axis, transformed_loss);
      auto const representable = coordinate + h;
      h = representable - coordinate;
      auto const evaluate_at = [&](double internal_coordinate) {
        return evaluateBicubicProposal76(
            view, spline, energy_MeV,
            axisBackTransform(spline.loss_axis, internal_coordinate));
      };
      auto const y1 =
          evaluate_at(coordinate + h) - evaluate_at(coordinate - h);
      auto const y2 =
          evaluate_at(coordinate - 2. * h) - evaluate_at(coordinate + 2. * h);
      auto const y3 =
          evaluate_at(coordinate + 3. * h) - evaluate_at(coordinate - 3. * h);
      auto const coordinate_derivative =
          (y3 + 9. * y2 + 45. * y1) / (60. * h);
      return coordinate_derivative *
             axisDerivative(spline.loss_axis, transformed_loss);
    }

    CORSIKA_PROPOSAL_NATIVE_HOST_DEVICE inline double evaluateCubic(
        ProposalNativeDeviceView const& view,
        NativeCubicDescriptor const& spline, double x) {
      auto const coordinate = axisTransform(spline.axis, x);
      auto const node = intervalNode(coordinate, spline.coefficient_count);
      auto const fraction = coordinate - node;
      auto const offset = spline.coefficient_offset + node;
      double basis[4]{};
      hermiteBasis(fraction, basis);
      auto value =
          basis[0] * view.cubic_values[offset] +
          basis[1] * view.cubic_values[offset + 1] +
          basis[2] * view.cubic_node_derivatives[offset] +
          basis[3] * view.cubic_node_derivatives[offset + 1];
      if (spline.has_value_transform)
        value = axisBackTransform(spline.value_transform, value);
      return value;
    }

    /** Reproduce Interpolant<CubicSplines>::prime() from exported nodes. */
    CORSIKA_PROPOSAL_NATIVE_HOST_DEVICE inline double
    evaluateCubicDerivativeProposal76(
        ProposalNativeDeviceView const& view,
        NativeCubicDescriptor const& spline, double x) {
      auto const coordinate = axisTransform(spline.axis, x);
      auto const node = intervalNode(coordinate, spline.coefficient_count);
      auto const fraction = coordinate - node;
      auto const offset = spline.coefficient_offset + node;
      double basis[4]{};
      double derivative_basis[4]{};
      hermiteBasis(fraction, basis);
      hermiteBasisDerivative(fraction, derivative_basis);
      auto const raw_value =
          basis[0] * view.cubic_values[offset] +
          basis[1] * view.cubic_values[offset + 1] +
          basis[2] * view.cubic_node_derivatives[offset] +
          basis[3] * view.cubic_node_derivatives[offset + 1];
      auto derivative =
          derivative_basis[0] * view.cubic_values[offset] +
          derivative_basis[1] * view.cubic_values[offset + 1] +
          derivative_basis[2] * view.cubic_node_derivatives[offset] +
          derivative_basis[3] * view.cubic_node_derivatives[offset + 1];
      if (spline.has_value_transform)
        derivative *= axisBackDerivative(spline.value_transform, raw_value);
      return derivative * axisDerivative(spline.axis, x);
    }

    struct Proposal76NewtonResult {
      double value{};
      std::uint32_t iterations{};
      bool runtime_error{};
    };

    /** Boost 1.79 newton_raphson_iterate(..., digits=20) without Boost ABI. */
    template <typename Function, typename Derivative>
    CORSIKA_PROPOSAL_NATIVE_HOST_DEVICE inline Proposal76NewtonResult
    proposal76Newton(Function const& function,
                     Derivative const& derivative_function,
                     double initial_guess, double lower, double upper) {
      auto const sign = [](double value) {
        return (0. < value ? 1 : 0) - (value < 0. ? 1 : 0);
      };
      auto residual = 0.;
      auto last_residual = 0.;
      auto result = initial_guess;
      auto guess = initial_guess;
      auto delta = 1.7976931348623157e308;
      auto delta1 = delta;
      auto delta2 = delta;
      auto maximum_residual = 0.;
      auto minimum_residual = 0.;
      constexpr double factor = 1.9073486328125e-6; // 2^(1 - 20)
      std::uint32_t iterations = 0;
      while (iterations < 256) {
        last_residual = residual;
        delta2 = delta1;
        delta1 = delta;
        residual = function(result);
        auto const derivative = derivative_function(result);
        ++iterations;
        if (!finite(residual) || !finite(derivative))
          return {result, iterations, true};
        if (residual == 0.) break;
        if (derivative == 0.) {
          if (last_residual == 0.) {
            guess = result == lower ? upper : lower;
            last_residual = function(guess);
            delta = guess - result;
          }
          if (sign(last_residual) * sign(residual) < 0) {
            delta = delta < 0. ? (result - lower) / 2.
                               : (result - upper) / 2.;
          } else {
            delta = delta < 0. ? (result - upper) / 2.
                               : (result - lower) / 2.;
          }
        } else {
          delta = residual / derivative;
        }
        auto const abs_delta2 = delta2 < 0. ? -delta2 : delta2;
        if ((delta * 2. < 0. ? -(delta * 2.) : delta * 2.) > abs_delta2) {
          auto shift = delta > 0. ? (result - lower) / 2.
                                  : (result - upper) / 2.;
          auto const abs_shift = shift < 0. ? -shift : shift;
          auto const abs_result = result < 0. ? -result : result;
          if (result != 0. && abs_shift > abs_result)
            delta = (delta > 0. ? 1. : -1.) * abs_result;
          else
            delta = shift;
          delta1 = 3. * delta;
          delta2 = 3. * delta;
        }
        guess = result;
        result -= delta;
        if (result <= lower) {
          delta = .5 * (guess - lower);
          result = guess - delta;
          if (result == lower || result == upper) break;
        } else if (result >= upper) {
          delta = .5 * (guess - upper);
          result = guess - delta;
          if (result == lower || result == upper) break;
        }
        if (delta > 0.) {
          upper = guess;
          maximum_residual = residual;
        } else {
          lower = guess;
          minimum_residual = residual;
        }
        if (maximum_residual * minimum_residual > 0.)
          return {result, iterations, true};
        auto const abs_result = result < 0. ? -result : result;
        auto const abs_step = delta < 0. ? -delta : delta;
        if (!(abs_result * factor < abs_step)) break;
      }
      if (iterations == 256) return {result, iterations, true};
      return {result, iterations, false};
    }

    CORSIKA_PROPOSAL_NATIVE_HOST_DEVICE inline bool validView(
        ProposalNativeDeviceView const& view) {
      return view.dndx_column_count > 0 && view.dndx_columns != nullptr &&
             view.total_rate_column_count > 0 &&
             view.total_rate_columns != nullptr &&
             view.selection_column_index_count == view.dndx_column_count &&
             view.selection_column_indices != nullptr &&
             view.dedx_accumulation_index_count ==
                 view.dedx_column_count &&
             (view.dedx_column_count == 0 ||
              (view.dedx_columns != nullptr &&
               view.dedx_accumulation_indices != nullptr)) &&
             view.bicubic_values != nullptr &&
             view.bicubic_derivative_energy != nullptr &&
             view.bicubic_derivative_loss != nullptr &&
             view.bicubic_mixed_derivative != nullptr &&
             view.bicubic_coefficient_count > 0;
    }

    CORSIKA_PROPOSAL_NATIVE_HOST_DEVICE inline NativeDndxColumn const*
    findDndx(ProposalNativeDeviceView const& view, std::int32_t pdg_id,
             std::int32_t process_id, std::uint64_t component_hash) {
      // ProposalNativeTableExporter stores this array in an order equivalent
      // to canonical (PID, process, component) order. Host-side structural
      // validation requires one medium per PID, so this public identity can be
      // found without a device-side hash table or pointer-rich PROPOSAL object.
      std::uint32_t lower = 0;
      std::uint32_t upper = view.dndx_column_count;
      while (lower < upper) {
        auto const middle = lower + (upper - lower) / 2;
        auto const& candidate = view.dndx_columns[middle];
        auto const before =
            candidate.pdg_id < pdg_id ||
            (candidate.pdg_id == pdg_id &&
             (candidate.process_id < process_id ||
              (candidate.process_id == process_id &&
               candidate.component_hash < component_hash)));
        if (before)
          lower = middle + 1;
        else
          upper = middle;
      }
      if (lower < view.dndx_column_count) {
        auto const& column = view.dndx_columns[lower];
        if (column.pdg_id == pdg_id && column.process_id == process_id &&
            column.component_hash == component_hash)
          return &column;
      }
      return nullptr;
    }

    struct NativeDndxParticleRange {
      std::uint32_t begin{};
      std::uint32_t end{};
    };

    CORSIKA_PROPOSAL_NATIVE_HOST_DEVICE inline NativeDndxParticleRange
    dndxParticleRange(ProposalNativeDeviceView const& view,
                      std::int32_t pdg_id) {
      std::uint32_t lower = 0;
      std::uint32_t upper = view.dndx_column_count;
      while (lower < upper) {
        auto const middle = lower + (upper - lower) / 2;
        if (view.dndx_columns[middle].pdg_id < pdg_id)
          lower = middle + 1;
        else
          upper = middle;
      }
      auto const begin = lower;
      upper = view.dndx_column_count;
      while (lower < upper) {
        auto const middle = lower + (upper - lower) / 2;
        if (view.dndx_columns[middle].pdg_id <= pdg_id)
          lower = middle + 1;
        else
          upper = middle;
      }
      return {begin, lower};
    }

    CORSIKA_PROPOSAL_NATIVE_HOST_DEVICE inline NativeDndxColumn const*
    selectionDndx(ProposalNativeDeviceView const& view,
                  std::uint32_t selection_position) {
      if (selection_position >= view.selection_column_index_count)
        return nullptr;
      auto const index = view.selection_column_indices[selection_position];
      if (index >= view.dndx_column_count) return nullptr;
      return &view.dndx_columns[index];
    }

    CORSIKA_PROPOSAL_NATIVE_HOST_DEVICE inline NativeTotalRateColumn const*
    findTotalRate(ProposalNativeDeviceView const& view,
                  std::int32_t pdg_id) {
      std::uint32_t lower = 0;
      std::uint32_t upper = view.total_rate_column_count;
      while (lower < upper) {
        auto const middle = lower + (upper - lower) / 2;
        if (view.total_rate_columns[middle].pdg_id < pdg_id)
          lower = middle + 1;
        else
          upper = middle;
      }
      if (lower < view.total_rate_column_count &&
          view.total_rate_columns[lower].pdg_id == pdg_id)
        return &view.total_rate_columns[lower];
      return nullptr;
    }

    CORSIKA_PROPOSAL_NATIVE_HOST_DEVICE inline NativeUtilityColumn const*
    findUtility(ProposalNativeDeviceView const& view, std::int32_t pdg_id) {
      for (std::uint32_t i = 0; i < view.utility_column_count; ++i)
        if (view.utility_columns[i].pdg_id == pdg_id)
          return &view.utility_columns[i];
      return nullptr;
    }

    struct NativeKinematicLimits {
      double minimum{};
      double maximum{};
      bool valid{};
    };

    CORSIKA_PROPOSAL_NATIVE_HOST_DEVICE inline NativeKinematicLimits
    kinematicLimits(ProposalNativeDeviceView const& view,
                    NativeDndxColumn const& column, double energy_MeV) {
      if (!(energy_MeV > 0.)) return {};
      auto const mass = column.particle_mass_MeV;
      auto minimum = 0.;
      auto maximum = 0.;
      switch (column.kinematic_model) {
      case NativeKinematicModel::OnlyStochastic:
        return {1., 1., true};
      case NativeKinematicModel::PhotonPairFullLoss:
        return {1., 1., true};
      case NativeKinematicModel::Compton:
        maximum =
            1. - 1. /
                     (1. + 2. * energy_MeV /
                                view.constants.electron_mass_MeV);
        break;
      case NativeKinematicModel::BremsElectronScreening:
        if (!(mass > 0.)) return {};
        // BremsElectronScreening inherits Bremsstrahlung's default limits in
        // PROPOSAL 7.6.2.  The Z-dependent 0.75*sqrt(e) recoil expression is
        // specific to EpairProduction and must not be applied here.
        maximum = 1. - mass / energy_MeV;
        if (maximum < 0.) maximum = 0.;
        break;
      case NativeKinematicModel::BremsKelnerKokoulinPetrukhin:
        if (!(mass > 0.) ||
            !(column.component_nuclear_charge > 0.))
          return {};
        // PROPOSAL 7.6.2 BREMSSTRAHLUNG_IMPL(KelnerKokoulinPetrukhin).
        // Retain the source expression order; the production calculator does
        // not enable the optional Lorenz cut.
        maximum =
            1. - 0.75 * view.constants.sqrt_e * (mass / energy_MeV) *
                     power(column.component_nuclear_charge, 1. / 3.);
        if (maximum < 0.) maximum = 0.;
        break;
      case NativeKinematicModel::ElectronPair: {
        if (!(mass > 0.)) return {};
        auto const ratio = mass / energy_MeV;
        minimum = 4. * view.constants.electron_mass_MeV / energy_MeV;
        maximum =
            1. - 0.75 * view.constants.sqrt_e * ratio *
                     power(column.component_nuclear_charge, 1. / 3.);
        auto const second = 1. - 6. * ratio * ratio;
        if (second < maximum) maximum = second;
        if (maximum < minimum) maximum = minimum;
        break;
      }
      case NativeKinematicModel::PhotonuclearAllm97: {
        auto const nucleon_mass =
            column.component_average_nucleon_mass_MeV;
        auto const pion_mass = view.constants.charged_pion_mass_MeV;
        if (!(nucleon_mass > 0.) || !(pion_mass > 0.) || !(mass > 0.))
          return {};
        // PROPOSAL 7.6.2 Photonuclear::GetKinematicLimits().
        minimum =
            (pion_mass + pion_mass * pion_mass /
                             (2. * nucleon_mass)) /
            energy_MeV;
        maximum = 1.;
        if (mass < pion_mass) {
          auto const ratio = mass / nucleon_mass;
          maximum -= nucleon_mass * (1. + ratio * ratio) /
                     (2. * energy_MeV);
        }
        if (maximum < minimum) maximum = minimum;
        break;
      }
      case NativeKinematicModel::IonizationMoller:
        if (!(mass > 0.)) return {};
        maximum = 0.5 * (1. - mass / energy_MeV);
        if (maximum < 0.) maximum = 0.;
        break;
      case NativeKinematicModel::IonizationBhabha:
        if (!(mass > 0.)) return {};
        maximum = 1. - mass / energy_MeV;
        if (maximum < 0.) maximum = 0.;
        break;
      case NativeKinematicModel::IonizationBetheBlochRossi: {
        if (!(mass > 0.) ||
            !(column.medium_mean_excitation_energy_MeV > 0.))
          return {};
        // PROPOSAL 7.6.2 IonizBetheBlochRossi::GetKinematicLimits,
        // including its ordering and final v_max >= v_min clamp.
        auto const mass_ratio =
            view.constants.electron_mass_MeV / mass;
        auto const gamma = energy_MeV / mass;
        minimum =
            column.medium_mean_excitation_energy_MeV / energy_MeV;
        maximum =
            2. * view.constants.electron_mass_MeV *
            (gamma * gamma - 1.) /
            ((1. + 2. * gamma * mass_ratio +
              mass_ratio * mass_ratio) *
             energy_MeV);
        auto const kinetic_limit = 1. - mass / energy_MeV;
        if (kinetic_limit < maximum) maximum = kinetic_limit;
        if (maximum < minimum) maximum = minimum;
        break;
      }
      case NativeKinematicModel::Unsupported:
        return {};
      }
      if (!finite(minimum) || !finite(maximum) || maximum < minimum)
        return {};
      auto cut = minimum;
      if (column.absolute_cut_MeV > 0. && column.relative_cut > 0.) {
        auto requested = column.absolute_cut_MeV / energy_MeV;
        if (column.relative_cut < requested)
          requested = column.relative_cut;
        if (requested < minimum) requested = minimum;
        if (requested > maximum) requested = maximum;
        cut = requested;
      }
      return {cut, maximum, true};
    }

    CORSIKA_PROPOSAL_NATIVE_HOST_DEVICE inline double physicalLoss(
        ProposalNativeDeviceView const& view,
        NativeDndxColumn const& column, double energy_MeV,
        double transformed_loss) {
      auto const limits = kinematicLimits(view, column, energy_MeV);
      if (!limits.valid) return 0. / 0.;
      auto const lower = limits.minimum;
      auto const upper = limits.maximum;
      if (transformed_loss <= 0.) return lower;
      if (transformed_loss >= 1.) return upper;
      if (lower == upper) return lower;
      switch (column.loss_transform) {
      case NativeLossTransform::RelativeLog: {
        if (!(lower > 0.)) return 0. / 0.;
        auto const ratio_log = logarithm(upper / lower);
        return lower * exponential(transformed_loss * ratio_log);
      }
      case NativeLossTransform::RelativeLogPowerOnePointFive: {
        if (!(lower > 0.)) return 0. / 0.;
        auto const ratio_log = logarithm(upper / lower);
        return lower * exponential(
            power(transformed_loss, 1.5) * ratio_log);
      }
      case NativeLossTransform::OneMinusLog: {
        // Preserve PROPOSAL 7.6.2 transform_loss_log operation order.  The
        // log1p/expm1 form is numerically more stable, but changes the final
        // 20-bit Newton iterate and therefore does not reproduce the scalar
        // stochastic-loss decision tape.
        auto const xi = logarithm((1. - lower) / (1. - upper));
        return 1. - (1. - upper) *
                        exponential((1. - transformed_loss) * xi);
      }
      }
      return 0. / 0.;
    }

    CORSIKA_PROPOSAL_NATIVE_HOST_DEVICE inline double transformedLoss(
        ProposalNativeDeviceView const& view,
        NativeDndxColumn const& column, double energy_MeV,
        double physical_loss) {
      auto const limits = kinematicLimits(view, column, energy_MeV);
      if (!limits.valid || physical_loss < limits.minimum ||
          physical_loss > limits.maximum)
        return 0. / 0.;
      auto const lower = limits.minimum;
      auto const upper = limits.maximum;
      if (physical_loss <= lower) return 0.;
      if (physical_loss >= upper) return 1.;
      if (lower == upper) return 0.;
      switch (column.loss_transform) {
      case NativeLossTransform::RelativeLog: {
        if (!(lower > 0.)) return 0. / 0.;
        return logarithm(physical_loss / lower) /
               logarithm(upper / lower);
      }
      case NativeLossTransform::RelativeLogPowerOnePointFive: {
        if (!(lower > 0.)) return 0. / 0.;
        auto const ratio = logarithm(physical_loss / lower) /
                           logarithm(upper / lower);
        return power(ratio, 2. / 3.);
      }
      case NativeLossTransform::OneMinusLog: {
        // Preserve PROPOSAL 7.6.2 retransform_loss_log operation order.
        auto const xi = logarithm((1. - lower) / (1. - upper));
        return 1. - logarithm((1. - physical_loss) / (1. - upper)) / xi;
      }
      }
      return 0. / 0.;
    }

  } // namespace native_detail

  CORSIKA_PROPOSAL_NATIVE_HOST_DEVICE inline bool
  proposalNativeRateBelowThreshold(
      NativeDndxColumn const& column, double energy_MeV) {
    return column.rate_model == NativeRateModel::BicubicSpline &&
           (energy_MeV < column.lower_energy_limit_MeV ||
            energy_MeV < column.spline.energy_axis.low);
  }

  CORSIKA_PROPOSAL_NATIVE_HOST_DEVICE inline NativeQueryResult
  queryProposalNativeCumulativeRate(
      ProposalNativeDeviceView const& view, NativeDndxColumn const& column,
      double energy_MeV, double transformed_loss) {
    if (!native_detail::validView(view))
      return {NativeQueryStatus::InvalidView, 0, 0, 0.};
    if (!native_detail::finite(energy_MeV) ||
        !native_detail::finite(transformed_loss))
      return {NativeQueryStatus::NonFiniteInput, 0, 0, 0.};
    // CrossSectionDNDXInterpolant::evaluate_interpolant returns zero below
    // its process threshold; this is not an out-of-domain shower error.
    if (energy_MeV < column.lower_energy_limit_MeV ||
        energy_MeV < column.spline.energy_axis.low)
      return {NativeQueryStatus::Success, 0, 0, 0.};
    if (energy_MeV > column.spline.energy_axis.high)
      return {NativeQueryStatus::EnergyOutOfRange, 0, 0, 0.};
    if (transformed_loss < column.spline.loss_axis.low ||
        transformed_loss > column.spline.loss_axis.high)
      return {NativeQueryStatus::QuantileOutOfRange, 0, 0, 0.};
    auto value = native_detail::evaluateBicubicProposal76(
                     view, column.spline, energy_MeV, transformed_loss) /
                 column.component_weight;
    // This is the explicit behaviour of
    // CrossSectionDNDXInterpolant::evaluate_interpolant in PROPOSAL 7.6.2.
    if (value < 0.) value = 0.;
    if (!native_detail::finite(value))
      return {NativeQueryStatus::InvalidView, 0, 0, 0.};
    return {NativeQueryStatus::Success, 0, 0, value};
  }

  CORSIKA_PROPOSAL_NATIVE_HOST_DEVICE inline NativeQueryResult
  queryProposalNativeRateForColumn(
      ProposalNativeDeviceView const& view,
      NativeDndxColumn const& column_value, double energy_MeV) {
    if (!native_detail::validView(view))
      return {NativeQueryStatus::InvalidView, 0, 0, 0.};
    auto const* column = &column_value;
    if (proposalNativeRateBelowThreshold(*column, energy_MeV))
      return {NativeQueryStatus::Success, 0, 0, 0.};
    if (column->rate_model == NativeRateModel::AnnihilationHeitler) {
      if (!native_detail::finite(energy_MeV))
        return {NativeQueryStatus::NonFiniteInput, 0, 0, 0.};
      if (energy_MeV <= column->particle_mass_MeV)
        return {NativeQueryStatus::Success, 0, 0, 0.};
      auto const gamma = energy_MeV / column->particle_mass_MeV;
      auto const root = native_detail::power(gamma * gamma - 1., .5);
      auto const aux =
          (gamma * gamma + 4. * gamma + 1.) /
              (gamma * gamma - 1.) *
              native_detail::logarithm(gamma + root) -
          (gamma + 3.) / root;
      auto const rate =
          aux * column->direct_rate_factor / (gamma + 1.);
      return {NativeQueryStatus::Success, 0, 0, rate};
    }
    if (column->rate_model == NativeRateModel::PhotoeffectSauter) {
      if (!native_detail::finite(energy_MeV))
        return {NativeQueryStatus::NonFiniteInput, 0, 0, 0.};
      if (!(energy_MeV > column->lower_energy_limit_MeV))
        return {NativeQueryStatus::Success, 0, 0, 0.};
      auto const alpha = view.constants.fine_structure_constant;
      auto const electron_mass = view.constants.electron_mass_MeV;
      auto const pi = view.constants.pi;
      auto const classical_electron_radius_cm =
          view.constants.classical_electron_radius_cm;
      auto const z = column->component_nuclear_charge;
      auto const ionization = z * z * alpha * alpha * electron_mass / 2.;
      auto const gamma = 1. + (energy_MeV - ionization) / electron_mass;
      auto const gamma_beta =
          native_detail::squareRoot(gamma * gamma - 1.);
      auto const sigma_t = 8. * pi * classical_electron_radius_cm *
                           classical_electron_radius_cm / 3.;
      auto sigma = 1.5 * sigma_t * native_detail::power(z, 5.) *
                   native_detail::power(alpha, 4.) *
                   native_detail::power(electron_mass / energy_MeV, 5.) *
                   native_detail::power(gamma_beta, 3.) *
                   (4. / 3. + gamma * (gamma - 2.) / (gamma + 1.) *
                                  (1. - native_detail::logarithm(
                                              gamma + gamma_beta) /
                                            (gamma * gamma_beta)));
      auto const z_alpha_beta = z * alpha * gamma / gamma_beta;
      auto const z_pi_alpha_beta = pi * z_alpha_beta;
      double correction = 0.;
      if (z_pi_alpha_beta < 10.) {
        correction =
            (1. + z_alpha_beta * z_alpha_beta) * z_pi_alpha_beta /
            native_detail::hyperbolicSine(z_pi_alpha_beta) *
            native_detail::exponential(
                z_alpha_beta *
                (pi - 4. * native_detail::arctangent(1. / z_alpha_beta)));
      } else {
        correction =
            (1. + z_alpha_beta * z_alpha_beta) * 2. * z_pi_alpha_beta *
            native_detail::exponential(
                -4. * z_alpha_beta *
                native_detail::arctangent(1. / z_alpha_beta));
      }
      auto const log_z = native_detail::logarithm(z);
      auto const shell_ratio =
          1. + 0.01481 * log_z * log_z -
          0.000788 * log_z * log_z * log_z;
      auto const rate = sigma * correction * shell_ratio *
                        column->direct_rate_factor;
      return {NativeQueryStatus::Success, 0, 0, rate};
    }
    if (column->rate_model ==
        NativeRateModel::PhotoproductionHeckC7Shadowing) {
      if (!native_detail::finite(energy_MeV))
        return {NativeQueryStatus::NonFiniteInput, 0, 0, 0.};
      if (energy_MeV < column->lower_energy_limit_MeV)
        return {NativeQueryStatus::Success, 0, 0, 0.};
      auto const energy_GeV = energy_MeV * 1.e-3;
      auto const nucleon_mass_GeV =
          column->component_average_nucleon_mass_MeV * 1.e-3;
      auto const s = nucleon_mass_GeV * nucleon_mass_GeV +
                     2. * nucleon_mass_GeV * energy_GeV;
      constexpr double s0 = 1.0761 * 1.0761;
      double nucleon_cross = 0.;
      if (native_detail::squareRoot(s) < 19.39) {
        nucleon_cross =
            (73.7 * native_detail::power(s, 0.073) +
             191.7 * native_detail::power(s, -0.602)) *
            native_detail::squareRoot(1. - s0 / s);
      } else {
        nucleon_cross =
            59.3 * native_detail::power(s, 0.093) +
            120.2 * native_detail::power(s, -0.358);
      }
      constexpr double resonance_mass[3]{1.231, 1.515, 1.680};
      constexpr double resonance_width[3]{0.11, 0.11, 0.125};
      constexpr double resonance_sigma[3]{31.125, 25.567, 17.508};
      constexpr double resonance_window[3]{0.17, 0.38, 0.38};
      for (int index = 0; index < 3; ++index) {
        auto const mass2 = resonance_mass[index] * resonance_mass[index];
        auto const width2 = resonance_width[index] * resonance_width[index];
        auto const breit_wigner =
            s / (energy_GeV * energy_GeV) * resonance_sigma[index] *
            width2 * s /
            ((s - mass2) * (s - mass2) + width2 * s);
        auto threshold = 0.;
        if (energy_GeV >= 0.152 + resonance_window[index])
          threshold = 1.;
        else if (energy_GeV >= 0.152)
          threshold =
              (energy_GeV - 0.152) / resonance_window[index];
        nucleon_cross += breit_wigner * threshold;
      }
      auto const atom_cross =
          nucleon_cross *
          native_detail::power(column->component_atomic_mass, 0.91);
      return {NativeQueryStatus::Success, 0, 0,
              atom_cross * column->direct_rate_factor};
    }
    if (column->rate_model == NativeRateModel::BicubicSpline &&
        (column->pdg_id == 13 || column->pdg_id == -13)) {
      if (!native_detail::finite(energy_MeV))
        return {NativeQueryStatus::NonFiniteInput, 0, 0, 0.};
      if (energy_MeV > column->spline.energy_axis.high)
        return {NativeQueryStatus::EnergyOutOfRange, 0, 0, 0.};
      auto value = native_detail::evaluateBicubicPrecomputed(
                       view, column->spline, energy_MeV, 1.) /
                   column->component_weight;
      if (value < 0.) value = 0.;
      if (!native_detail::finite(value))
        return {NativeQueryStatus::InvalidView, 0, 0, 0.};
      return {NativeQueryStatus::Success, 0, 0, value};
    }
    return queryProposalNativeCumulativeRate(view, *column, energy_MeV, 1.);
  }

  CORSIKA_PROPOSAL_NATIVE_HOST_DEVICE inline NativeQueryResult
  queryProposalNativeRate(
      ProposalNativeDeviceView const& view, std::int32_t pdg_id,
      std::int32_t process_id, std::uint64_t component_hash,
      double energy_MeV) {
    if (!native_detail::validView(view))
      return {NativeQueryStatus::InvalidView, 0, 0, 0.};
    auto const* column = native_detail::findDndx(
        view, pdg_id, process_id, component_hash);
    if (!column)
      return {NativeQueryStatus::ColumnNotFound, 0, 0, 0.};
    if (proposalNativeRateBelowThreshold(*column, energy_MeV))
      return {NativeQueryStatus::Success, 0, 0, 0.};
    return queryProposalNativeRateForColumn(view, *column, energy_MeV);
  }

  CORSIKA_PROPOSAL_NATIVE_HOST_DEVICE inline NativeQueryResult
  queryProposalNativeTotalRate(
      ProposalNativeDeviceView const& view, std::int32_t pdg_id,
      double energy_MeV) {
    if (!native_detail::validView(view))
      return {NativeQueryStatus::InvalidView, 0, 0, 0.};
    if (!native_detail::finite(energy_MeV))
      return {NativeQueryStatus::NonFiniteInput, 0, 0, 0.};
    auto const* column = native_detail::findTotalRate(view, pdg_id);
    if (!column)
      return {NativeQueryStatus::ParticleNotFound, 0, 0, 0.};
    // InteractionBuilder::MeanFreePath uses this independently interpolated
    // rate. Below its refined lower boundary it returns infinity, i.e. zero
    // interaction rate. A negative interpolation excursion has the same
    // scalar behavior.
    if (energy_MeV < column->lower_energy_limit_MeV)
      return {NativeQueryStatus::Success, 0, 0, 0.};
    if (energy_MeV < column->spline.axis.low ||
        energy_MeV > column->spline.axis.high)
      return {NativeQueryStatus::EnergyOutOfRange, 0, 0, 0.};
    auto const value =
        native_detail::evaluateCubic(view, column->spline, energy_MeV);
    if (!native_detail::finite(value))
      return {NativeQueryStatus::InvalidView, 0, 0, 0.};
    return {NativeQueryStatus::Success, 0, 0,
            value < 0. ? 0. : value};
  }

  /** Exact Interaction::Rates flat accumulation used only by the inner
   * PROPOSAL process/component selector. This is deliberately distinct from
   * queryProposalNativeTotalRate(), whose spline controls the outer CORSIKA
   * interaction-distance/rejection decision. */
  CORSIKA_PROPOSAL_NATIVE_HOST_DEVICE inline NativeQueryResult
  queryProposalNativeSelectionTotalRate(
      ProposalNativeDeviceView const& view, std::int32_t pdg_id,
      double energy_MeV) {
    if (!native_detail::validView(view))
      return {NativeQueryStatus::InvalidView, 0, 0, 0.};
    double total = 0.;
    auto const range = native_detail::dndxParticleRange(view, pdg_id);
    bool found = range.begin != range.end;
    for (std::uint32_t i = range.begin; i < range.end; ++i) {
      auto const* selected = native_detail::selectionDndx(view, i);
      if (!selected || selected->pdg_id != pdg_id)
        return {NativeQueryStatus::InvalidView, 0, 0, 0.};
      auto const& column = *selected;
      if (proposalNativeRateBelowThreshold(column, energy_MeV))
        continue;
      auto const rate =
          queryProposalNativeRateForColumn(view, column, energy_MeV);
      if (rate.status != NativeQueryStatus::Success) return rate;
      total += rate.value;
    }
    if (!found)
      return {NativeQueryStatus::ParticleNotFound, 0, 0, 0.};
    return {NativeQueryStatus::Success, 0, 0, total};
  }

  CORSIKA_PROPOSAL_NATIVE_HOST_DEVICE inline NativeQueryResult
  queryProposalNativeDedx(
      ProposalNativeDeviceView const& view, std::int32_t pdg_id,
      double energy_MeV) {
    if (!native_detail::finite(energy_MeV))
      return {NativeQueryStatus::NonFiniteInput, 0, 0, 0.};
    double total = 0.;
    bool found = false;
    for (std::uint32_t position = 0;
         position < view.dedx_accumulation_index_count; ++position) {
      auto const index = view.dedx_accumulation_indices[position];
      if (index >= view.dedx_column_count)
        return {NativeQueryStatus::InvalidView, 0, 0, 0.};
      auto const& column = view.dedx_columns[index];
      if (column.pdg_id != pdg_id) continue;
      found = true;
      // CrossSectionDEDXInterpolant::Calculate() returns zero below the
      // process-specific lower limit.  Other continuous processes for the
      // same particle may already be active, so skipping this column is part
      // of the scalar PROPOSAL summation semantics.
      if (energy_MeV < column.lower_energy_limit_MeV) continue;
      if (energy_MeV < column.spline.axis.low ||
          energy_MeV > column.spline.axis.high)
        return {NativeQueryStatus::EnergyOutOfRange, 0, 0, 0.};
      total += native_detail::evaluateCubic(view, column.spline, energy_MeV) /
               column.component_weight;
    }
    if (!found)
      return {NativeQueryStatus::ParticleNotFound, 0, 0, 0.};
    return {NativeQueryStatus::Success, 0, 0, total};
  }

  CORSIKA_PROPOSAL_NATIVE_HOST_DEVICE inline NativeQueryResult
  queryProposalNativeLossFraction(
      ProposalNativeDeviceView const& view, std::int32_t pdg_id,
      std::int32_t process_id, std::uint64_t component_hash,
      double energy_MeV, double quantile) {
    if (!native_detail::validView(view))
      return {NativeQueryStatus::InvalidView, 0, 0, 0.};
    if (!native_detail::finite(energy_MeV) ||
        !native_detail::finite(quantile))
      return {NativeQueryStatus::NonFiniteInput, 0, 0, 0.};
    if (quantile < 0. || quantile > 1.)
      return {NativeQueryStatus::QuantileOutOfRange, 0, 0, 0.};
    auto const* column = native_detail::findDndx(
        view, pdg_id, process_id, component_hash);
    if (!column)
      return {NativeQueryStatus::ColumnNotFound, 0, 0, 0.};
    // These public transport variables are degenerate at v=1.  Return before
    // considering the backing rate representation: a process may expose a
    // bicubic cumulative-rate table even though it has no loss variable to
    // invert (PhotoPairKochMotz and BKK photon-muon-pair in PROPOSAL 7.6.2).
    if (column->kinematic_model == NativeKinematicModel::OnlyStochastic ||
        column->kinematic_model == NativeKinematicModel::PhotonPairFullLoss)
      return {NativeQueryStatus::Success, 0, 0, 1.};
    if (column->rate_model != NativeRateModel::BicubicSpline) {
      return {NativeQueryStatus::UnsupportedKinematics, 0, 0, 0.};
    }
    auto const limits =
        native_detail::kinematicLimits(view, *column, energy_MeV);
    if (!limits.valid)
      return {NativeQueryStatus::UnsupportedKinematics, 0, 0, 0.};
    if (limits.minimum == limits.maximum)
      return {NativeQueryStatus::Success, 0, 0, limits.minimum};
    auto const total = queryProposalNativeCumulativeRate(
        view, *column, energy_MeV, 1.);
    if (total.status != NativeQueryStatus::Success) return total;
    // CrossSection::CalculateStochasticLoss receives q*CalculatedNdx(), then
    // multiplies that value by the component weight before calling
    // CrossSectionDNDXInterpolant::GetUpperLimit().  Preserve both rounding
    // steps and solve the *unweighted, unclipped* interpolant, exactly as
    // GetUpperLimit does.  Solving the already divided public cumulative rate
    // is algebraically equivalent but can stop at a different 20-bit Newton
    // iterate.
    auto const target = (quantile * total.value) * column->component_weight;
    if (quantile == 0.)
      return {NativeQueryStatus::Success, 0, 0, limits.minimum};
    if (quantile == 1.)
      return {NativeQueryStatus::Success, 0, 0, limits.maximum};

    auto const raw_cumulative = [&](double transformed_loss) {
      return native_detail::evaluateBicubicProposal76(
          view, column->spline, energy_MeV, transformed_loss);
    };
    auto const residual_at = [&](double transformed_loss) {
      return raw_cumulative(transformed_loss) - target;
    };
    auto const sign = [](double value) {
      return (0. < value ? 1 : 0) - (value < 0. ? 1 : 0);
    };

    double lower = column->spline.loss_axis.low;
    double upper = column->spline.loss_axis.high;
    auto lower_residual = residual_at(lower);
    auto upper_residual = residual_at(upper);
    std::uint32_t bisection_iterations = 0;
    bool require_proposal_bisection = false;
    bool exact_root = false;
    double current = 0.;

    // CubicInterpolation::find_parameter first calls Boost bisect with a
    // tolerance equal to 1% of the complete parameter interval.  Match its
    // endpoint tests, sign bookkeeping, midpoint order and strict '<'
    // tolerance test rather than assuming a monotonic '< target' predicate.
    if (!native_detail::finite(lower_residual) ||
        !native_detail::finite(upper_residual)) {
      return {NativeQueryStatus::RootNotConverged, 0, 0, 0.};
    } else if (lower_residual == 0.) {
      current = lower;
      exact_root = true;
    } else if (upper_residual == 0.) {
      current = upper;
      exact_root = true;
    } else if (lower_residual * upper_residual >= 0.) {
      // Boost raises std::runtime_error here; GetUpperLimit catches it and
      // enters PROPOSAL::Bisection below.
      require_proposal_bisection = true;
    } else {
      auto const coarse_tolerance = (upper - lower) * 1.e-2;
      while (!((upper - lower) < coarse_tolerance)) {
        auto const middle = (lower + upper) / 2.;
        auto const middle_residual = residual_at(middle);
        ++bisection_iterations;
        if (!native_detail::finite(middle_residual)) {
          require_proposal_bisection = true;
          break;
        }
        if (middle == upper || middle == lower) break;
        if (middle_residual == 0.) {
          lower = upper = middle;
          exact_root = true;
          break;
        }
        if (sign(middle_residual) * sign(lower_residual) < 0) {
          upper = middle;
          upper_residual = middle_residual;
        } else {
          lower = middle;
          lower_residual = middle_residual;
        }
      }
      current = (lower + upper) / 2.;
    }

    // Reproduce Boost 1.79 newton_raphson_iterate used by
    // CubicInterpolation 0.1.5. `digits=20` means binary digits, not an
    // iteration count.  The deliberately loose stop is observable in a fixed
    // random decision tape, so all updates below retain Boost's order.
    auto delta = 1.7976931348623157e308;
    auto delta1 = delta;
    auto delta2 = delta;
    auto previous_residual = 0.;
    auto maximum_residual = 0.;
    auto minimum_residual = 0.;
    constexpr double newton_factor = 1.9073486328125e-6; // 2^(1 - 20)
    std::uint32_t newton_iterations = 0;
    auto guess = current;
    while (!exact_root && !require_proposal_bisection &&
           newton_iterations < 256) {
      auto const last_residual = previous_residual;
      delta2 = delta1;
      delta1 = delta;
      auto const residual = residual_at(current);
      previous_residual = residual;
      ++newton_iterations;
      if (!native_detail::finite(residual)) {
        require_proposal_bisection = true;
        break;
      }
      if (residual == 0.) {
        exact_root = true;
        break;
      }
      auto derivative =
          native_detail::evaluateBicubicLossDerivativeProposal76(
              view, column->spline, energy_MeV, current);

      if (!native_detail::finite(derivative)) {
        require_proposal_bisection = true;
        break;
      } else if (derivative == 0.) {
        // boost::math::tools::detail::handle_zero_derivative, copied in the
        // same branch order.  `guess` is mutable state in Boost, not merely a
        // temporary midpoint.
        auto mutable_last_residual = last_residual;
        if (mutable_last_residual == 0.) {
          guess = current == lower ? upper : lower;
          mutable_last_residual = residual_at(guess);
          delta = guess - current;
        }
        if (sign(mutable_last_residual) * sign(residual) < 0) {
          delta = delta < 0. ? (current - lower) / 2.
                             : (current - upper) / 2.;
        } else {
          delta = delta < 0. ? (current - upper) / 2.
                             : (current - lower) / 2.;
        }
      } else {
        delta = residual / derivative;
      }
      auto const abs_delta = delta < 0. ? -delta : delta;
      auto const abs_delta2 = delta2 < 0. ? -delta2 : delta2;
      if (2. * abs_delta > abs_delta2) {
        auto shift = delta > 0. ? (current - lower) * .5
                                : (current - upper) * .5;
        auto const abs_shift = shift < 0. ? -shift : shift;
        auto const abs_current = current < 0. ? -current : current;
        if (current != 0. && abs_shift > abs_current)
          // boost::math::tools::newton_raphson_iterate protects an
          // excessively large step with exactly |result|.  Do not add a
          // safety margin here: even a small one changes the final iterate
          // selected by PROPOSAL's deliberately loose 20-bit stopping rule.
          delta = (delta > 0. ? 1. : -1.) * abs_current;
        else
          delta = shift;
        delta1 = 3. * delta;
        delta2 = 3. * delta;
      }

      guess = current;
      current -= delta;
      if (current <= lower) {
        delta = .5 * (guess - lower);
        current = guess - delta;
        if (current == lower || current == upper) break;
      } else if (current >= upper) {
        delta = .5 * (guess - upper);
        current = guess - delta;
        if (current == lower || current == upper) break;
      }
      if (delta > 0.) {
        upper = guess;
        maximum_residual = residual;
      } else {
        lower = guess;
        minimum_residual = residual;
      }
      if (maximum_residual * minimum_residual > 0.) {
        require_proposal_bisection = true;
        break;
      }

      auto const abs_current = current < 0. ? -current : current;
      auto const abs_step = delta < 0. ? -delta : delta;
      if (!(abs_current * newton_factor < abs_step)) break;
    }
    if (newton_iterations == 256) require_proposal_bisection = true;

    // This is PROPOSAL::Bisection(f, 0, 1, 1e-6, 100), which is the exact
    // fallback selected by CrossSectionDNDXInterpolant::GetUpperLimit after a
    // Boost runtime_error.  Do not replace it with a tighter private solver:
    // the midpoint of this coarse bracket is part of scalar PROPOSAL's output.
    if (require_proposal_bisection) {
      lower = column->spline.loss_axis.low;
      upper = column->spline.loss_axis.high;
      lower_residual = residual_at(lower);
      upper_residual = residual_at(upper);
      if (!native_detail::finite(lower_residual) ||
          !native_detail::finite(upper_residual) ||
          lower_residual * upper_residual > 0.)
        return {NativeQueryStatus::RootNotConverged, newton_iterations,
                bisection_iterations, 0.};
      for (std::uint32_t iteration = 0; iteration <= 100; ++iteration) {
        auto const middle = (lower + upper) / 2.;
        auto const middle_residual = residual_at(middle);
        auto const current_lower_residual = residual_at(lower);
        if (!native_detail::finite(middle_residual) ||
            !native_detail::finite(current_lower_residual))
          return {NativeQueryStatus::RootNotConverged, newton_iterations,
                  bisection_iterations, 0.};
        if (sign(middle_residual) == sign(current_lower_residual))
          lower = middle;
        else
          upper = middle;
        ++bisection_iterations;
        if ((upper - lower) < 1.e-6) break;
      }
      current = (lower + upper) / 2.;
    }
    auto const physical =
        native_detail::physicalLoss(view, *column, energy_MeV, current);
    if (!native_detail::finite(physical))
      return {NativeQueryStatus::RootNotConverged, newton_iterations,
              bisection_iterations, 0.};

    return {NativeQueryStatus::Success, newton_iterations,
            bisection_iterations, physical};
  }

  CORSIKA_PROPOSAL_NATIVE_HOST_DEVICE inline NativeQueryResult
  queryProposalNativeRange(ProposalNativeDeviceView const& view,
                           std::int32_t pdg_id, double energy_MeV) {
    auto const* column = native_detail::findUtility(view, pdg_id);
    if (!column)
      return {NativeQueryStatus::ParticleNotFound, 0, 0, 0.};
    if (!native_detail::finite(energy_MeV))
      return {NativeQueryStatus::NonFiniteInput, 0, 0, 0.};
    if (energy_MeV < column->lower_energy_limit_MeV ||
        energy_MeV < column->spline.axis.low ||
        energy_MeV > column->spline.axis.high)
      return {NativeQueryStatus::EnergyOutOfRange, 0, 0, 0.};
    auto const upper =
        native_detail::evaluateCubic(view, column->spline, energy_MeV);
    auto const lower = native_detail::evaluateCubic(
        view, column->spline, column->lower_energy_limit_MeV);
    auto value = column->reverse ? lower - upper : upper - lower;
    if (!native_detail::finite(value) || value < 0.)
      return {NativeQueryStatus::InvalidView, 0, 0, 0.};
    return {NativeQueryStatus::Success, 0, 0, value};
  }

  /**
   * Exact UtilityInterpolant::GetUpperLimit(initial_energy, grammage)
   * semantics used by scalar ContinuousProcess.  Keeping the initial energy
   * is essential: PROPOSAL brackets Newton on [lower_limit, initial_energy],
   * so an inverse of an absolute range over the whole table is not the same
   * floating-point operation even though it solves the same equation.
   */
  CORSIKA_PROPOSAL_NATIVE_HOST_DEVICE inline NativeQueryResult
  queryProposalNativeEnergyAfterContinuousLoss(
      ProposalNativeDeviceView const& view, std::int32_t pdg_id,
      double initial_energy_MeV, double grammage_g_per_cm2) {
    auto const* column = native_detail::findUtility(view, pdg_id);
    if (!column)
      return {NativeQueryStatus::ParticleNotFound, 0, 0, 0.};
    if (!native_detail::finite(initial_energy_MeV) ||
        !native_detail::finite(grammage_g_per_cm2) ||
        grammage_g_per_cm2 < 0.)
      return {NativeQueryStatus::NonFiniteInput, 0, 0, 0.};
    if (initial_energy_MeV < column->lower_energy_limit_MeV ||
        initial_energy_MeV > column->spline.axis.high)
      return {NativeQueryStatus::EnergyOutOfRange, 0, 0, 0.};
    auto const maximum = queryProposalNativeRange(
        view, pdg_id, initial_energy_MeV);
    if (maximum.status != NativeQueryStatus::Success) return maximum;
    if (grammage_g_per_cm2 > maximum.value)
      return {NativeQueryStatus::EnergyOutOfRange, 0, 0, 0.};
    if (grammage_g_per_cm2 == maximum.value)
      return {NativeQueryStatus::Success, 0, 0,
              column->lower_energy_limit_MeV};
    if (grammage_g_per_cm2 == 0.)
      return {NativeQueryStatus::Success, 0, 0, initial_energy_MeV};

    auto signed_grammage = grammage_g_per_cm2;
    if (column->reverse) signed_grammage = -signed_grammage;
    auto const integrated_to_upper = native_detail::evaluateCubic(
        view, column->spline, initial_energy_MeV);
    auto const target = integrated_to_upper - signed_grammage;
    auto const residual = [&](double energy) {
      return native_detail::evaluateCubic(view, column->spline, energy) -
             target;
    };
    auto const derivative = [&](double energy) {
      return native_detail::evaluateCubicDerivativeProposal76(
          view, column->spline, energy);
    };
    auto const sign = [](double value) {
      return (0. < value ? 1 : 0) - (value < 0. ? 1 : 0);
    };
    auto proposal_bisection = [&](double accuracy, double& lower,
                                  double& upper,
                                  std::uint32_t& iterations) {
      auto const lower_value = residual(lower);
      auto const upper_value = residual(upper);
      if (!native_detail::finite(lower_value) ||
          !native_detail::finite(upper_value) ||
          lower_value * upper_value > 0.)
        return false;
      for (std::uint32_t iteration = 0; iteration <= 100; ++iteration) {
        auto const center = (lower + upper) / 2.;
        auto const center_value = residual(center);
        auto const current_lower_value = residual(lower);
        if (!native_detail::finite(center_value) ||
            !native_detail::finite(current_lower_value))
          return false;
        if (sign(center_value) == sign(current_lower_value))
          lower = center;
        else
          upper = center;
        ++iterations;
        if ((upper - lower) < accuracy) return true;
      }
      return true;
    };

    double lower = column->lower_energy_limit_MeV;
    double upper = initial_energy_MeV;
    std::uint32_t bisection_iterations = 0;
    auto const coarse_accuracy =
        (initial_energy_MeV - column->lower_energy_limit_MeV) * 1.e-2;
    if (!proposal_bisection(coarse_accuracy, lower, upper,
                            bisection_iterations))
      return {NativeQueryStatus::RootNotConverged, 0,
              bisection_iterations, 0.};
    auto const initial_guess =
        upper == initial_energy_MeV ? initial_energy_MeV
                                    : (lower + upper) / 2.;
    auto const newton = native_detail::proposal76Newton(
        residual, derivative, initial_guess, lower, upper);
    if (!newton.runtime_error && native_detail::finite(newton.value))
      return {NativeQueryStatus::Success, newton.iterations,
              bisection_iterations, newton.value};

    lower = column->lower_energy_limit_MeV;
    upper = initial_energy_MeV;
    if (!proposal_bisection(1.e-6, lower, upper,
                            bisection_iterations))
      return {NativeQueryStatus::RootNotConverged, newton.iterations,
              bisection_iterations, 0.};
    // UtilityInterpolant returns the first (lower) endpoint of its fallback
    // Bisection pair, unlike CrossSectionDNDX which returns the midpoint.
    return {NativeQueryStatus::Success, newton.iterations,
            bisection_iterations, lower};
  }

  CORSIKA_PROPOSAL_NATIVE_HOST_DEVICE inline NativeQueryResult
  queryProposalNativeEnergy(ProposalNativeDeviceView const& view,
                            std::int32_t pdg_id,
                            double range_g_per_cm2) {
    auto const* column = native_detail::findUtility(view, pdg_id);
    if (!column)
      return {NativeQueryStatus::ParticleNotFound, 0, 0, 0.};
    if (!native_detail::finite(range_g_per_cm2) || range_g_per_cm2 < 0.)
      return {NativeQueryStatus::NonFiniteInput, 0, 0, 0.};
    auto const maximum = queryProposalNativeRange(
        view, pdg_id, column->spline.axis.high);
    if (maximum.status != NativeQueryStatus::Success) return maximum;
    if (range_g_per_cm2 > maximum.value)
      return {NativeQueryStatus::EnergyOutOfRange, 0, 0, 0.};
    double lower = column->lower_energy_limit_MeV;
    double upper = column->spline.axis.high;
    std::uint32_t iterations = 0;
    for (; iterations < 100; ++iterations) {
      // Bisect in PROPOSAL's native logarithmic axis, not linearly in E.
      auto const lower_coordinate =
          native_detail::axisTransform(column->spline.axis, lower);
      auto const upper_coordinate =
          native_detail::axisTransform(column->spline.axis, upper);
      auto const middle = native_detail::axisBackTransform(
          column->spline.axis,
          0.5 * (lower_coordinate + upper_coordinate));
      auto const value = queryProposalNativeRange(view, pdg_id, middle);
      if (value.status != NativeQueryStatus::Success) return value;
      if (value.value < range_g_per_cm2)
        lower = middle;
      else
        upper = middle;
      if ((upper - lower) <= 1.e-11 * upper) break;
    }
    return {NativeQueryStatus::Success, 0, iterations,
            0.5 * (lower + upper)};
  }

} // namespace corsika::gpu::em::tables

#undef CORSIKA_PROPOSAL_NATIVE_HOST_DEVICE

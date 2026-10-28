/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <cmath>

namespace corsika::kokkos {

  namespace proposal_native_detail {

    CORSIKA_HOST_DEVICE inline bool finite(double x) {
#if defined(__CUDA_ARCH__)
      return ::isfinite(x);
#else
      return std::isfinite(x);
#endif
    }

    CORSIKA_HOST_DEVICE inline double logarithm(double x) {
#if defined(__CUDA_ARCH__)
      return ::log(x);
#else
      return std::log(x);
#endif
    }

    CORSIKA_HOST_DEVICE inline double exponential(double x) {
#if defined(__CUDA_ARCH__)
      return ::exp(x);
#else
      return std::exp(x);
#endif
    }

    CORSIKA_HOST_DEVICE inline double logarithm_one_plus(double x) {
#if defined(__CUDA_ARCH__)
      return ::log1p(x);
#else
      return std::log1p(x);
#endif
    }

    CORSIKA_HOST_DEVICE inline double exponential_minus_one(double x) {
#if defined(__CUDA_ARCH__)
      return ::expm1(x);
#else
      return std::expm1(x);
#endif
    }

    CORSIKA_HOST_DEVICE inline double power(double x, double exponent) {
#if defined(__CUDA_ARCH__)
      return ::pow(x, exponent);
#else
      return std::pow(x, exponent);
#endif
    }

    CORSIKA_HOST_DEVICE inline double square_root(double x) {
#if defined(__CUDA_ARCH__)
      return ::sqrt(x);
#else
      return std::sqrt(x);
#endif
    }

    CORSIKA_HOST_DEVICE inline double hyperbolic_sine(double x) {
#if defined(__CUDA_ARCH__)
      return ::sinh(x);
#else
      return std::sinh(x);
#endif
    }

    CORSIKA_HOST_DEVICE inline double arctangent(double x) {
#if defined(__CUDA_ARCH__)
      return ::atan(x);
#else
      return std::atan(x);
#endif
    }

    CORSIKA_HOST_DEVICE inline double axis_transform(NativeAxisDescriptor const& axis,
                                                     double x) {
      switch (axis.type_) {
        case NativeAxisType::Linear:
          return (x - axis.low_) / axis.step_;
        case NativeAxisType::Exponential:
          return logarithm(x / axis.low_) / axis.step_;
        case NativeAxisType::ExponentialMinusOne:
          return (logarithm_one_plus(x / axis.low_) -
                  0.693147180559945309417232121458176568) /
                 axis.step_;
      }
      return 0. / 0.;
    }

    CORSIKA_HOST_DEVICE inline double axis_back_transform(
        NativeAxisDescriptor const& axis, double coordinate) {
      switch (axis.type_) {
        case NativeAxisType::Linear:
          return coordinate * axis.step_ + axis.low_;
        case NativeAxisType::Exponential:
          return axis.low_ * exponential(coordinate * axis.step_);
        case NativeAxisType::ExponentialMinusOne:
          return axis.low_ *
                 exponential_minus_one(coordinate * axis.step_ +
                                       0.693147180559945309417232121458176568);
      }
      return 0. / 0.;
    }

    CORSIKA_HOST_DEVICE inline double axis_derivative(NativeAxisDescriptor const& axis,
                                                      double x) {
      switch (axis.type_) {
        case NativeAxisType::Linear:
          return 1. / axis.step_;
        case NativeAxisType::Exponential:
          return 1. / (x * axis.step_);
        case NativeAxisType::ExponentialMinusOne:
          return 1. / (axis.step_ * (x + axis.low_));
      }
      return 0. / 0.;
    }

    CORSIKA_HOST_DEVICE inline double axis_back_derivative(
        NativeAxisDescriptor const& axis, double coordinate) {
      switch (axis.type_) {
        case NativeAxisType::Linear:
          return axis.step_;
        case NativeAxisType::Exponential:
          return axis.low_ * axis.step_ * exponential(coordinate * axis.step_);
        case NativeAxisType::ExponentialMinusOne:
          return axis.low_ * axis.step_ *
                 exponential(coordinate * axis.step_ +
                             0.693147180559945309417232121458176568);
      }
      return 0. / 0.;
    }

    CORSIKA_HOST_DEVICE inline std::uint32_t interval_node(double coordinate,
                                                           std::uint32_t nodes) {
      auto node = static_cast<long long>(coordinate >= 0. ? coordinate : -1.);
      if (coordinate < 0.) node = 0;
      if (node > static_cast<long long>(nodes) - 2)
        node = static_cast<long long>(nodes) - 2;
      return static_cast<std::uint32_t>(node);
    }

    CORSIKA_HOST_DEVICE inline void hermite_basis(double x, double (&basis)[4]) {
      auto const x2 = x * x;
      auto const x3 = x2 * x;
      basis[0] = 1. - 3. * x2 + 2. * x3;
      basis[1] = 3. * x2 - 2. * x3;
      basis[2] = x - 2. * x2 + x3;
      basis[3] = -x2 + x3;
    }

    CORSIKA_HOST_DEVICE inline void hermite_basis_derivative(double x,
                                                             double (&basis)[4]) {
      auto const x2 = x * x;
      basis[0] = -6. * x + 6. * x2;
      basis[1] = 6. * x - 6. * x2;
      basis[2] = 1. - 4. * x + 3. * x2;
      basis[3] = -2. * x + 3. * x2;
    }

    CORSIKA_HOST_DEVICE inline double bicubic_element(
        ProposalNativeDeviceView const&, NativeBicubicDescriptor const& spline,
        NativeArrayView<double> const& array, std::uint32_t row, std::uint32_t column) {
      auto const local = static_cast<std::uint64_t>(row) +
                         static_cast<std::uint64_t>(spline.rows_) * column;
      return array[spline.coefficientOffset_ + local];
    }

    CORSIKA_HOST_DEVICE inline double evaluate_bicubic(
        ProposalNativeDeviceView const& view, NativeBicubicDescriptor const& spline,
        double energy_MeV, double transformed_loss) {
      auto const x0 = axis_transform(spline.energyAxis_, energy_MeV);
      auto const x1 = axis_transform(spline.lossAxis_, transformed_loss);
      auto const n0 = interval_node(x0, spline.rows_);
      auto const n1 = interval_node(x1, spline.columns_);
      double h0[4]{};
      double h1[4]{};
      hermite_basis(x0 - n0, h0);
      hermite_basis(x1 - n1, h1);
      double result = 0.;
      for (std::uint32_t a = 0; a < 4; ++a) {
        for (std::uint32_t b = 0; b < 4; ++b) {
          auto const row = n0 + (a & 1u);
          auto const column = n1 + (b & 1u);
          auto coefficients = view.getBicubicValues();
          if (a >= 2 && b < 2)
            coefficients = view.getBicubicDerivativeEnergy();
          else if (a < 2 && b >= 2)
            coefficients = view.getBicubicDerivativeLoss();
          else if (a >= 2 && b >= 2)
            coefficients = view.getBicubicMixedDerivative();
          result +=
              h0[a] * bicubic_element(view, spline, coefficients, row, column) * h1[b];
        }
      }
      return result;
    }

    CORSIKA_HOST_DEVICE inline double dot_four_sequential(double a0, double a1, double a2,
                                                          double a3, double b0, double b1,
                                                          double b2, double b3) {
      auto const p0 = a0 * b0;
      auto const p1 = a1 * b1;
      auto const p2 = a2 * b2;
      auto const p3 = a3 * b3;
      return ((p0 + p1) + p2) + p3;
    }

    /* Evaluate host-precomputed CubicInterpolation polynomial coefficients. */
    CORSIKA_HOST_DEVICE inline double evaluate_bicubic_precomputed(
        ProposalNativeDeviceView const& view, NativeBicubicDescriptor const& spline,
        double energy_MeV, double transformed_loss) {
      if (spline.polynomialCoefficientOffset_ == NoNativePolynomialCoefficients ||
          !view.getBicubicPolynomialCoefficients().hasStorage())
        return 0. / 0.;
      auto const x0 = axis_transform(spline.energyAxis_, energy_MeV);
      auto const x1 = axis_transform(spline.lossAxis_, transformed_loss);
      auto const n0 = interval_node(x0, spline.rows_);
      auto const n1 = interval_node(x1, spline.columns_);
      auto const cell = static_cast<std::uint64_t>(n0) +
                        static_cast<std::uint64_t>(spline.rows_ - 1u) * n1;
      auto const offset = spline.polynomialCoefficientOffset_ + 16u * cell;
      if (offset > view.getBicubicPolynomialCoefficients().getSize() ||
          16u > view.getBicubicPolynomialCoefficients().getSize() - offset)
        return 0. / 0.;
      auto const* coefficients = &view.getBicubicPolynomialCoefficients()[offset];
      auto const f0 = x0 - n0;
      auto const f1 = x1 - n1;
      double const v0[4]{1., f0, f0 * f0, f0 * f0 * f0};
      double const v1[4]{1., f1, f1 * f1, f1 * f1 * f1};
      double projected[4]{};
      for (int row = 0; row < 4; ++row)
        projected[row] = dot_four_sequential(
            coefficients[4 * row], coefficients[4 * row + 1], coefficients[4 * row + 2],
            coefficients[4 * row + 3], v1[0], v1[1], v1[2], v1[3]);
      return dot_four_sequential(v0[0], v0[1], v0[2], v0[3], projected[0], projected[1],
                                 projected[2], projected[3]);
    }

    CORSIKA_HOST_DEVICE inline double evaluate_bicubic_proposal76(
        ProposalNativeDeviceView const& view, NativeBicubicDescriptor const& spline,
        double energy_MeV, double transformed_loss) {
      if (spline.polynomialCoefficientOffset_ != NoNativePolynomialCoefficients &&
          view.getBicubicPolynomialCoefficients().hasStorage())
        return evaluate_bicubic_precomputed(view, spline, energy_MeV, transformed_loss);
      return evaluate_bicubic(view, spline, energy_MeV, transformed_loss);
    }

    CORSIKA_HOST_DEVICE inline double evaluate_bicubic_loss_derivative(
        ProposalNativeDeviceView const& view, NativeBicubicDescriptor const& spline,
        double energy_MeV, double transformed_loss) {
      auto const x0 = axis_transform(spline.energyAxis_, energy_MeV);
      auto const x1 = axis_transform(spline.lossAxis_, transformed_loss);
      auto const n0 = interval_node(x0, spline.rows_);
      auto const n1 = interval_node(x1, spline.columns_);
      double h0[4]{};
      double dh1[4]{};
      hermite_basis(x0 - n0, h0);
      hermite_basis_derivative(x1 - n1, dh1);
      double result = 0.;
      for (std::uint32_t a = 0; a < 4; ++a) {
        for (std::uint32_t b = 0; b < 4; ++b) {
          auto const row = n0 + (a & 1u);
          auto const column = n1 + (b & 1u);
          auto coefficients = view.getBicubicValues();
          if (a >= 2 && b < 2)
            coefficients = view.getBicubicDerivativeEnergy();
          else if (a < 2 && b >= 2)
            coefficients = view.getBicubicDerivativeLoss();
          else if (a >= 2 && b >= 2)
            coefficients = view.getBicubicMixedDerivative();
          result +=
              h0[a] * bicubic_element(view, spline, coefficients, row, column) * dh1[b];
        }
      }
      // Loss axes in PROPOSAL 7.6.2 are linear. Keep the generic linear
      // coordinate scaling explicit and reject other axes during validation.
      return result / spline.lossAxis_.step_;
    }

    /*
     * CubicInterpolation 0.1.5 does not use the analytic Bicubic Hermite
     * derivative in find_parameter().  Its BicubicSplines::prime() calls
     * Boost's default sixth-order finite-difference derivative.  Keep this
     * seemingly redundant sampling here because it is part of the scalar
     * PROPOSAL interpolation semantics and affects the final 20-bit Newton
     * stopping point used by a decision tape.
     */
    CORSIKA_HOST_DEVICE inline double evaluate_bicubic_loss_derivative_proposal76(
        ProposalNativeDeviceView const& view, NativeBicubicDescriptor const& spline,
        double energy_MeV, double transformed_loss) {
      constexpr double epsilon = 2.220446049250313080847263336181640625e-16;
      auto h = power(epsilon / 168., 1. / 7.);
      auto const coordinate = axis_transform(spline.lossAxis_, transformed_loss);
      auto const representable = coordinate + h;
      h = representable - coordinate;
      auto const evaluate_at = [&](double internal_coordinate) {
        return evaluate_bicubic_proposal76(
            view, spline, energy_MeV,
            axis_back_transform(spline.lossAxis_, internal_coordinate));
      };
      auto const y1 = evaluate_at(coordinate + h) - evaluate_at(coordinate - h);
      auto const y2 = evaluate_at(coordinate - 2. * h) - evaluate_at(coordinate + 2. * h);
      auto const y3 = evaluate_at(coordinate + 3. * h) - evaluate_at(coordinate - 3. * h);
      auto const coordinate_derivative = (y3 + 9. * y2 + 45. * y1) / (60. * h);
      return coordinate_derivative * axis_derivative(spline.lossAxis_, transformed_loss);
    }

    CORSIKA_HOST_DEVICE inline double evaluate_cubic(ProposalNativeDeviceView const& view,
                                                     NativeCubicDescriptor const& spline,
                                                     double x) {
      auto const coordinate = axis_transform(spline.axis_, x);
      auto const node = interval_node(coordinate, spline.coefficientCount_);
      auto const fraction = coordinate - node;
      auto const offset = spline.coefficientOffset_ + node;
      double basis[4]{};
      hermite_basis(fraction, basis);
      auto value = basis[0] * view.getCubicValues()[offset] +
                   basis[1] * view.getCubicValues()[offset + 1] +
                   basis[2] * view.getCubicNodeDerivatives()[offset] +
                   basis[3] * view.getCubicNodeDerivatives()[offset + 1];
      if (spline.hasValueTransform_)
        value = axis_back_transform(spline.valueTransform_, value);
      return value;
    }

    /* Reproduce Interpolant<CubicSplines>::prime() from exported nodes. */
    CORSIKA_HOST_DEVICE inline double evaluate_cubic_derivative_proposal76(
        ProposalNativeDeviceView const& view, NativeCubicDescriptor const& spline,
        double x) {
      auto const coordinate = axis_transform(spline.axis_, x);
      auto const node = interval_node(coordinate, spline.coefficientCount_);
      auto const fraction = coordinate - node;
      auto const offset = spline.coefficientOffset_ + node;
      double basis[4]{};
      double derivative_basis[4]{};
      hermite_basis(fraction, basis);
      hermite_basis_derivative(fraction, derivative_basis);
      auto const raw_value = basis[0] * view.getCubicValues()[offset] +
                             basis[1] * view.getCubicValues()[offset + 1] +
                             basis[2] * view.getCubicNodeDerivatives()[offset] +
                             basis[3] * view.getCubicNodeDerivatives()[offset + 1];
      auto derivative = derivative_basis[0] * view.getCubicValues()[offset] +
                        derivative_basis[1] * view.getCubicValues()[offset + 1] +
                        derivative_basis[2] * view.getCubicNodeDerivatives()[offset] +
                        derivative_basis[3] * view.getCubicNodeDerivatives()[offset + 1];
      if (spline.hasValueTransform_)
        derivative *= axis_back_derivative(spline.valueTransform_, raw_value);
      return derivative * axis_derivative(spline.axis_, x);
    }

    struct Proposal76NewtonResult {
      double value_{};
      std::uint32_t iterations_{};
      bool runtimeError_{};
    };

    /* Boost 1.79 newton_raphson_iterate(..., digits=20) without Boost ABI. */
    template <typename TFunction, typename TDerivative>
    CORSIKA_HOST_DEVICE inline Proposal76NewtonResult proposal76_newton(
        TFunction const& function, TDerivative const& derivative_function,
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
        if (!finite(residual) || !finite(derivative)) return {result, iterations, true};
        if (residual == 0.) break;
        if (derivative == 0.) {
          if (last_residual == 0.) {
            guess = result == lower ? upper : lower;
            last_residual = function(guess);
            delta = guess - result;
          }
          if (sign(last_residual) * sign(residual) < 0) {
            delta = delta < 0. ? (result - lower) / 2. : (result - upper) / 2.;
          } else {
            delta = delta < 0. ? (result - upper) / 2. : (result - lower) / 2.;
          }
        } else {
          delta = residual / derivative;
        }
        auto const abs_delta2 = delta2 < 0. ? -delta2 : delta2;
        if ((delta * 2. < 0. ? -(delta * 2.) : delta * 2.) > abs_delta2) {
          auto shift = delta > 0. ? (result - lower) / 2. : (result - upper) / 2.;
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
        if (maximum_residual * minimum_residual > 0.) return {result, iterations, true};
        auto const abs_result = result < 0. ? -result : result;
        auto const abs_step = delta < 0. ? -delta : delta;
        if (!(abs_result * factor < abs_step)) break;
      }
      if (iterations == 256) return {result, iterations, true};
      return {result, iterations, false};
    }

    CORSIKA_HOST_DEVICE inline bool valid_view(ProposalNativeDeviceView const& view) {
      return view.getDndxColumns().getSize() > 0 && view.getDndxColumns().hasStorage() &&
             view.getTotalRateColumns().getSize() > 0 &&
             view.getTotalRateColumns().hasStorage() &&
             view.getSelectionColumnIndices().getSize() ==
                 view.getDndxColumns().getSize() &&
             view.getSelectionColumnIndices().hasStorage() &&
             view.getDedxAccumulationIndices().getSize() ==
                 view.getDedxColumns().getSize() &&
             (view.getDedxColumns().getSize() == 0 ||
              (view.getDedxColumns().hasStorage() &&
               view.getDedxAccumulationIndices().hasStorage())) &&
             view.getBicubicValues().hasStorage() &&
             view.getBicubicDerivativeEnergy().hasStorage() &&
             view.getBicubicDerivativeLoss().hasStorage() &&
             view.getBicubicMixedDerivative().hasStorage() &&
             view.getBicubicValues().getSize() > 0;
    }

    CORSIKA_HOST_DEVICE inline NativeDndxColumn const* find_dndx(
        ProposalNativeDeviceView const& view, std::int32_t pdg_id,
        std::int32_t process_id, std::uint64_t component_hash) {
      // ProposalNativeTableExporter stores this array in an order equivalent
      // to canonical (PID, process, component) order. Host-side structural
      // validation requires one medium per PID, so this public identity can be
      // found without a device-side hash table or pointer-rich PROPOSAL object.
      std::uint32_t lower = 0;
      std::uint32_t upper = view.getDndxColumns().getSize();
      while (lower < upper) {
        auto const middle = lower + (upper - lower) / 2;
        auto const columns = view.getDndxColumns();
        auto const& candidate = columns[middle];
        auto const before =
            candidate.pdgId_ < pdg_id ||
            (candidate.pdgId_ == pdg_id && (candidate.processId_ < process_id ||
                                            (candidate.processId_ == process_id &&
                                             candidate.componentHash_ < component_hash)));
        if (before)
          lower = middle + 1;
        else
          upper = middle;
      }
      if (lower < view.getDndxColumns().getSize()) {
        auto const columns = view.getDndxColumns();
        auto const& column = columns[lower];
        if (column.pdgId_ == pdg_id && column.processId_ == process_id &&
            column.componentHash_ == component_hash)
          return &column;
      }
      return nullptr;
    }

    struct NativeDndxParticleRange {
      std::uint32_t begin_{};
      std::uint32_t end_{};
    };

    CORSIKA_HOST_DEVICE inline NativeDndxParticleRange dndx_particle_range(
        ProposalNativeDeviceView const& view, std::int32_t pdg_id) {
      std::uint32_t lower = 0;
      std::uint32_t upper = view.getDndxColumns().getSize();
      while (lower < upper) {
        auto const middle = lower + (upper - lower) / 2;
        if (view.getDndxColumns()[middle].pdgId_ < pdg_id)
          lower = middle + 1;
        else
          upper = middle;
      }
      auto const begin = lower;
      upper = view.getDndxColumns().getSize();
      while (lower < upper) {
        auto const middle = lower + (upper - lower) / 2;
        if (view.getDndxColumns()[middle].pdgId_ <= pdg_id)
          lower = middle + 1;
        else
          upper = middle;
      }
      return {begin, lower};
    }

    CORSIKA_HOST_DEVICE inline NativeDndxColumn const* selection_dndx(
        ProposalNativeDeviceView const& view, std::uint32_t selection_position) {
      if (selection_position >= view.getSelectionColumnIndices().getSize())
        return nullptr;
      auto const index = view.getSelectionColumnIndices()[selection_position];
      if (index >= view.getDndxColumns().getSize()) return nullptr;
      return &view.getDndxColumns()[index];
    }

    CORSIKA_HOST_DEVICE inline NativeTotalRateColumn const* find_total_rate(
        ProposalNativeDeviceView const& view, std::int32_t pdg_id) {
      std::uint32_t lower = 0;
      std::uint32_t upper = view.getTotalRateColumns().getSize();
      while (lower < upper) {
        auto const middle = lower + (upper - lower) / 2;
        if (view.getTotalRateColumns()[middle].pdgId_ < pdg_id)
          lower = middle + 1;
        else
          upper = middle;
      }
      if (lower < view.getTotalRateColumns().getSize() &&
          view.getTotalRateColumns()[lower].pdgId_ == pdg_id)
        return &view.getTotalRateColumns()[lower];
      return nullptr;
    }

    CORSIKA_HOST_DEVICE inline NativeUtilityColumn const* find_utility(
        ProposalNativeDeviceView const& view, std::int32_t pdg_id) {
      for (std::uint32_t i = 0; i < view.getUtilityColumns().getSize(); ++i)
        if (view.getUtilityColumns()[i].pdgId_ == pdg_id)
          return &view.getUtilityColumns()[i];
      return nullptr;
    }

    struct NativeKinematicLimits {
      double minimum_{};
      double maximum_{};
      bool valid_{};
    };

    CORSIKA_HOST_DEVICE inline NativeKinematicLimits kinematic_limits(
        ProposalNativeDeviceView const& view, NativeDndxColumn const& column,
        double energy_MeV) {
      if (!(energy_MeV > 0.)) return {};
      auto const mass = column.particleMassMeV_;
      auto minimum = 0.;
      auto maximum = 0.;
      switch (column.kinematicModel_) {
        case NativeKinematicModel::OnlyStochastic:
          return {1., 1., true};
        case NativeKinematicModel::PhotonPairFullLoss:
          return {1., 1., true};
        case NativeKinematicModel::Compton:
          maximum =
              1. - 1. / (1. + 2. * energy_MeV / view.getConstants().electronMassMeV_);
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
          if (!(mass > 0.) || !(column.componentNuclearCharge_ > 0.)) return {};
          // PROPOSAL 7.6.2 BREMSSTRAHLUNG_IMPL(KelnerKokoulinPetrukhin).
          // Retain the source expression order; the production calculator does
          // not enable the optional Lorenz cut.
          maximum = 1. - 0.75 * view.getConstants().sqrtE_ * (mass / energy_MeV) *
                             power(column.componentNuclearCharge_, 1. / 3.);
          if (maximum < 0.) maximum = 0.;
          break;
        case NativeKinematicModel::ElectronPair: {
          if (!(mass > 0.)) return {};
          auto const ratio = mass / energy_MeV;
          minimum = 4. * view.getConstants().electronMassMeV_ / energy_MeV;
          maximum = 1. - 0.75 * view.getConstants().sqrtE_ * ratio *
                             power(column.componentNuclearCharge_, 1. / 3.);
          auto const second = 1. - 6. * ratio * ratio;
          if (second < maximum) maximum = second;
          if (maximum < minimum) maximum = minimum;
          break;
        }
        case NativeKinematicModel::PhotonuclearAllm97: {
          auto const nucleon_mass = column.componentAverageNucleonMassMeV_;
          auto const pion_mass = view.getConstants().chargedPionMassMeV_;
          if (!(nucleon_mass > 0.) || !(pion_mass > 0.) || !(mass > 0.)) return {};
          // PROPOSAL 7.6.2 Photonuclear::GetKinematicLimits().
          minimum =
              (pion_mass + pion_mass * pion_mass / (2. * nucleon_mass)) / energy_MeV;
          maximum = 1.;
          if (mass < pion_mass) {
            auto const ratio = mass / nucleon_mass;
            maximum -= nucleon_mass * (1. + ratio * ratio) / (2. * energy_MeV);
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
          if (!(mass > 0.) || !(column.mediumMeanExcitationEnergyMeV_ > 0.)) return {};
          // PROPOSAL 7.6.2 IonizBetheBlochRossi::GetKinematicLimits,
          // including its ordering and final v_max >= v_min clamp.
          auto const mass_ratio = view.getConstants().electronMassMeV_ / mass;
          auto const gamma = energy_MeV / mass;
          minimum = column.mediumMeanExcitationEnergyMeV_ / energy_MeV;
          maximum =
              2. * view.getConstants().electronMassMeV_ * (gamma * gamma - 1.) /
              ((1. + 2. * gamma * mass_ratio + mass_ratio * mass_ratio) * energy_MeV);
          auto const kinetic_limit = 1. - mass / energy_MeV;
          if (kinetic_limit < maximum) maximum = kinetic_limit;
          if (maximum < minimum) maximum = minimum;
          break;
        }
        case NativeKinematicModel::Unsupported:
          return {};
      }
      if (!finite(minimum) || !finite(maximum) || maximum < minimum) return {};
      auto cut = minimum;
      if (column.absoluteCutMeV_ > 0. && column.relativeCut_ > 0.) {
        auto requested = column.absoluteCutMeV_ / energy_MeV;
        if (column.relativeCut_ < requested) requested = column.relativeCut_;
        if (requested < minimum) requested = minimum;
        if (requested > maximum) requested = maximum;
        cut = requested;
      }
      return {cut, maximum, true};
    }

    CORSIKA_HOST_DEVICE inline double physical_loss(ProposalNativeDeviceView const& view,
                                                    NativeDndxColumn const& column,
                                                    double energy_MeV,
                                                    double transformed_loss) {
      auto const limits = kinematic_limits(view, column, energy_MeV);
      if (!limits.valid_) return 0. / 0.;
      auto const lower = limits.minimum_;
      auto const upper = limits.maximum_;
      if (transformed_loss <= 0.) return lower;
      if (transformed_loss >= 1.) return upper;
      if (lower == upper) return lower;
      switch (column.lossTransform_) {
        case NativeLossTransform::RelativeLog: {
          if (!(lower > 0.)) return 0. / 0.;
          auto const ratio_log = logarithm(upper / lower);
          return lower * exponential(transformed_loss * ratio_log);
        }
        case NativeLossTransform::RelativeLogPowerOnePointFive: {
          if (!(lower > 0.)) return 0. / 0.;
          auto const ratio_log = logarithm(upper / lower);
          return lower * exponential(power(transformed_loss, 1.5) * ratio_log);
        }
        case NativeLossTransform::OneMinusLog: {
          // Preserve PROPOSAL 7.6.2 transform_loss_log operation order.  The
          // log1p/expm1 form is numerically more stable, but changes the final
          // 20-bit Newton iterate and therefore does not reproduce the scalar
          // stochastic-loss decision tape.
          auto const xi = logarithm((1. - lower) / (1. - upper));
          return 1. - (1. - upper) * exponential((1. - transformed_loss) * xi);
        }
      }
      return 0. / 0.;
    }

    CORSIKA_HOST_DEVICE inline double transformed_loss(
        ProposalNativeDeviceView const& view, NativeDndxColumn const& column,
        double energy_MeV, double physical_loss) {
      auto const limits = kinematic_limits(view, column, energy_MeV);
      if (!limits.valid_ || physical_loss < limits.minimum_ ||
          physical_loss > limits.maximum_)
        return 0. / 0.;
      auto const lower = limits.minimum_;
      auto const upper = limits.maximum_;
      if (physical_loss <= lower) return 0.;
      if (physical_loss >= upper) return 1.;
      if (lower == upper) return 0.;
      switch (column.lossTransform_) {
        case NativeLossTransform::RelativeLog: {
          if (!(lower > 0.)) return 0. / 0.;
          return logarithm(physical_loss / lower) / logarithm(upper / lower);
        }
        case NativeLossTransform::RelativeLogPowerOnePointFive: {
          if (!(lower > 0.)) return 0. / 0.;
          auto const ratio = logarithm(physical_loss / lower) / logarithm(upper / lower);
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

  } // namespace proposal_native_detail

  CORSIKA_HOST_DEVICE inline bool proposal_native_rate_below_threshold(
      NativeDndxColumn const& column, double energy_MeV) {
    return column.rateModel_ == NativeRateModel::BicubicSpline &&
           (energy_MeV < column.lowerEnergyLimitMeV_ ||
            energy_MeV < column.spline_.energyAxis_.low_);
  }

  CORSIKA_HOST_DEVICE inline NativeQueryResult query_proposal_native_cumulative_rate(
      ProposalNativeDeviceView const& view, NativeDndxColumn const& column,
      double energy_MeV, double transformed_loss) {
    if (!proposal_native_detail::valid_view(view))
      return {NativeQueryStatus::InvalidView, 0, 0, 0.};
    if (!proposal_native_detail::finite(energy_MeV) ||
        !proposal_native_detail::finite(transformed_loss))
      return {NativeQueryStatus::NonFiniteInput, 0, 0, 0.};
    // CrossSectionDNDXInterpolant::evaluate_interpolant returns zero below
    // its process threshold; this is not an out-of-domain shower error.
    if (energy_MeV < column.lowerEnergyLimitMeV_ ||
        energy_MeV < column.spline_.energyAxis_.low_)
      return {NativeQueryStatus::Success, 0, 0, 0.};
    if (energy_MeV > column.spline_.energyAxis_.high_)
      return {NativeQueryStatus::EnergyOutOfRange, 0, 0, 0.};
    if (transformed_loss < column.spline_.lossAxis_.low_ ||
        transformed_loss > column.spline_.lossAxis_.high_)
      return {NativeQueryStatus::QuantileOutOfRange, 0, 0, 0.};
    auto value = proposal_native_detail::evaluate_bicubic_proposal76(
                     view, column.spline_, energy_MeV, transformed_loss) /
                 column.componentWeight_;
    // This is the explicit behaviour of
    // CrossSectionDNDXInterpolant::evaluate_interpolant in PROPOSAL 7.6.2.
    if (value < 0.) value = 0.;
    if (!proposal_native_detail::finite(value))
      return {NativeQueryStatus::InvalidView, 0, 0, 0.};
    return {NativeQueryStatus::Success, 0, 0, value};
  }

  CORSIKA_HOST_DEVICE inline NativeQueryResult query_proposal_native_rate_for_column(
      ProposalNativeDeviceView const& view, NativeDndxColumn const& column_value,
      double energy_MeV) {
    if (!proposal_native_detail::valid_view(view))
      return {NativeQueryStatus::InvalidView, 0, 0, 0.};
    auto const* column = &column_value;
    if (proposal_native_rate_below_threshold(*column, energy_MeV))
      return {NativeQueryStatus::Success, 0, 0, 0.};
    if (column->rateModel_ == NativeRateModel::AnnihilationHeitler) {
      if (!proposal_native_detail::finite(energy_MeV))
        return {NativeQueryStatus::NonFiniteInput, 0, 0, 0.};
      if (energy_MeV <= column->particleMassMeV_)
        return {NativeQueryStatus::Success, 0, 0, 0.};
      auto const gamma = energy_MeV / column->particleMassMeV_;
      auto const root = proposal_native_detail::power(gamma * gamma - 1., .5);
      auto const aux = (gamma * gamma + 4. * gamma + 1.) / (gamma * gamma - 1.) *
                           proposal_native_detail::logarithm(gamma + root) -
                       (gamma + 3.) / root;
      auto const rate = aux * column->directRateFactor_ / (gamma + 1.);
      return {NativeQueryStatus::Success, 0, 0, rate};
    }
    if (column->rateModel_ == NativeRateModel::PhotoeffectSauter) {
      if (!proposal_native_detail::finite(energy_MeV))
        return {NativeQueryStatus::NonFiniteInput, 0, 0, 0.};
      if (!(energy_MeV > column->lowerEnergyLimitMeV_))
        return {NativeQueryStatus::Success, 0, 0, 0.};
      auto const alpha = view.getConstants().fineStructureConstant_;
      auto const electron_mass = view.getConstants().electronMassMeV_;
      auto const pi = view.getConstants().pi_;
      auto const classical_electron_radius_cm =
          view.getConstants().classicalElectronRadiusCm_;
      auto const z = column->componentNuclearCharge_;
      auto const ionization = z * z * alpha * alpha * electron_mass / 2.;
      auto const gamma = 1. + (energy_MeV - ionization) / electron_mass;
      auto const gamma_beta = proposal_native_detail::square_root(gamma * gamma - 1.);
      auto const sigma_t =
          8. * pi * classical_electron_radius_cm * classical_electron_radius_cm / 3.;
      auto sigma =
          1.5 * sigma_t * proposal_native_detail::power(z, 5.) *
          proposal_native_detail::power(alpha, 4.) *
          proposal_native_detail::power(electron_mass / energy_MeV, 5.) *
          proposal_native_detail::power(gamma_beta, 3.) *
          (4. / 3. + gamma * (gamma - 2.) / (gamma + 1.) *
                         (1. - proposal_native_detail::logarithm(gamma + gamma_beta) /
                                   (gamma * gamma_beta)));
      auto const z_alpha_beta = z * alpha * gamma / gamma_beta;
      auto const z_pi_alpha_beta = pi * z_alpha_beta;
      double correction = 0.;
      if (z_pi_alpha_beta < 10.) {
        correction =
            (1. + z_alpha_beta * z_alpha_beta) * z_pi_alpha_beta /
            proposal_native_detail::hyperbolic_sine(z_pi_alpha_beta) *
            proposal_native_detail::exponential(
                z_alpha_beta *
                (pi - 4. * proposal_native_detail::arctangent(1. / z_alpha_beta)));
      } else {
        correction = (1. + z_alpha_beta * z_alpha_beta) * 2. * z_pi_alpha_beta *
                     proposal_native_detail::exponential(
                         -4. * z_alpha_beta *
                         proposal_native_detail::arctangent(1. / z_alpha_beta));
      }
      auto const log_z = proposal_native_detail::logarithm(z);
      auto const shell_ratio =
          1. + 0.01481 * log_z * log_z - 0.000788 * log_z * log_z * log_z;
      auto const rate = sigma * correction * shell_ratio * column->directRateFactor_;
      return {NativeQueryStatus::Success, 0, 0, rate};
    }
    if (column->rateModel_ == NativeRateModel::PhotoproductionHeckC7Shadowing) {
      if (!proposal_native_detail::finite(energy_MeV))
        return {NativeQueryStatus::NonFiniteInput, 0, 0, 0.};
      if (energy_MeV < column->lowerEnergyLimitMeV_)
        return {NativeQueryStatus::Success, 0, 0, 0.};
      auto const energy_GeV = energy_MeV * 1.e-3;
      auto const nucleon_mass_GeV = column->componentAverageNucleonMassMeV_ * 1.e-3;
      auto const s =
          nucleon_mass_GeV * nucleon_mass_GeV + 2. * nucleon_mass_GeV * energy_GeV;
      constexpr double s0 = 1.0761 * 1.0761;
      double nucleon_cross = 0.;
      if (proposal_native_detail::square_root(s) < 19.39) {
        nucleon_cross = (73.7 * proposal_native_detail::power(s, 0.073) +
                         191.7 * proposal_native_detail::power(s, -0.602)) *
                        proposal_native_detail::square_root(1. - s0 / s);
      } else {
        nucleon_cross = 59.3 * proposal_native_detail::power(s, 0.093) +
                        120.2 * proposal_native_detail::power(s, -0.358);
      }
      constexpr double resonance_mass[3]{1.231, 1.515, 1.680};
      constexpr double resonance_width[3]{0.11, 0.11, 0.125};
      constexpr double resonance_sigma[3]{31.125, 25.567, 17.508};
      constexpr double resonance_window[3]{0.17, 0.38, 0.38};
      for (int index = 0; index < 3; ++index) {
        auto const mass2 = resonance_mass[index] * resonance_mass[index];
        auto const width2 = resonance_width[index] * resonance_width[index];
        auto const breit_wigner = s / (energy_GeV * energy_GeV) * resonance_sigma[index] *
                                  width2 * s / ((s - mass2) * (s - mass2) + width2 * s);
        auto threshold = 0.;
        if (energy_GeV >= 0.152 + resonance_window[index])
          threshold = 1.;
        else if (energy_GeV >= 0.152)
          threshold = (energy_GeV - 0.152) / resonance_window[index];
        nucleon_cross += breit_wigner * threshold;
      }
      auto const atom_cross = nucleon_cross * proposal_native_detail::power(
                                                  column->componentAtomicMass_, 0.91);
      return {NativeQueryStatus::Success, 0, 0, atom_cross * column->directRateFactor_};
    }
    if (column->rateModel_ == NativeRateModel::BicubicSpline &&
        (column->pdgId_ == 13 || column->pdgId_ == -13)) {
      if (!proposal_native_detail::finite(energy_MeV))
        return {NativeQueryStatus::NonFiniteInput, 0, 0, 0.};
      if (energy_MeV > column->spline_.energyAxis_.high_)
        return {NativeQueryStatus::EnergyOutOfRange, 0, 0, 0.};
      auto value = proposal_native_detail::evaluate_bicubic_precomputed(
                       view, column->spline_, energy_MeV, 1.) /
                   column->componentWeight_;
      if (value < 0.) value = 0.;
      if (!proposal_native_detail::finite(value))
        return {NativeQueryStatus::InvalidView, 0, 0, 0.};
      return {NativeQueryStatus::Success, 0, 0, value};
    }
    return query_proposal_native_cumulative_rate(view, *column, energy_MeV, 1.);
  }

  CORSIKA_HOST_DEVICE inline NativeQueryResult query_proposal_native_rate(
      ProposalNativeDeviceView const& view, std::int32_t pdg_id, std::int32_t process_id,
      std::uint64_t component_hash, double energy_MeV) {
    if (!proposal_native_detail::valid_view(view))
      return {NativeQueryStatus::InvalidView, 0, 0, 0.};
    auto const* column =
        proposal_native_detail::find_dndx(view, pdg_id, process_id, component_hash);
    if (!column) return {NativeQueryStatus::ColumnNotFound, 0, 0, 0.};
    if (proposal_native_rate_below_threshold(*column, energy_MeV))
      return {NativeQueryStatus::Success, 0, 0, 0.};
    return query_proposal_native_rate_for_column(view, *column, energy_MeV);
  }

  CORSIKA_HOST_DEVICE inline NativeQueryResult query_proposal_native_total_rate(
      ProposalNativeDeviceView const& view, std::int32_t pdg_id, double energy_MeV) {
    if (!proposal_native_detail::valid_view(view))
      return {NativeQueryStatus::InvalidView, 0, 0, 0.};
    if (!proposal_native_detail::finite(energy_MeV))
      return {NativeQueryStatus::NonFiniteInput, 0, 0, 0.};
    auto const* column = proposal_native_detail::find_total_rate(view, pdg_id);
    if (!column) return {NativeQueryStatus::ParticleNotFound, 0, 0, 0.};
    // InteractionBuilder::MeanFreePath uses this independently interpolated
    // rate. Below its refined lower boundary it returns infinity, i.e. zero
    // interaction rate. A negative interpolation excursion has the same
    // scalar behavior.
    if (energy_MeV < column->lowerEnergyLimitMeV_)
      return {NativeQueryStatus::Success, 0, 0, 0.};
    if (energy_MeV < column->spline_.axis_.low_ ||
        energy_MeV > column->spline_.axis_.high_)
      return {NativeQueryStatus::EnergyOutOfRange, 0, 0, 0.};
    auto const value =
        proposal_native_detail::evaluate_cubic(view, column->spline_, energy_MeV);
    if (!proposal_native_detail::finite(value))
      return {NativeQueryStatus::InvalidView, 0, 0, 0.};
    return {NativeQueryStatus::Success, 0, 0, value < 0. ? 0. : value};
  }

  /* Exact Interaction::Rates flat accumulation used only by the inner
   * PROPOSAL process/component selector. This is deliberately distinct from
   * queryProposalNativeTotalRate(), whose spline controls the outer CORSIKA
   * interaction-distance/rejection decision. */
  CORSIKA_HOST_DEVICE inline NativeQueryResult query_proposal_native_selection_total_rate(
      ProposalNativeDeviceView const& view, std::int32_t pdg_id, double energy_MeV) {
    if (!proposal_native_detail::valid_view(view))
      return {NativeQueryStatus::InvalidView, 0, 0, 0.};
    double total = 0.;
    auto const range = proposal_native_detail::dndx_particle_range(view, pdg_id);
    bool found = range.begin_ != range.end_;
    for (std::uint32_t i = range.begin_; i < range.end_; ++i) {
      auto const* selected = proposal_native_detail::selection_dndx(view, i);
      if (!selected || selected->pdgId_ != pdg_id)
        return {NativeQueryStatus::InvalidView, 0, 0, 0.};
      auto const& column = *selected;
      if (proposal_native_rate_below_threshold(column, energy_MeV)) continue;
      auto const rate = query_proposal_native_rate_for_column(view, column, energy_MeV);
      if (rate.status_ != NativeQueryStatus::Success) return rate;
      total += rate.value_;
    }
    if (!found) return {NativeQueryStatus::ParticleNotFound, 0, 0, 0.};
    return {NativeQueryStatus::Success, 0, 0, total};
  }

  CORSIKA_HOST_DEVICE inline NativeQueryResult query_proposal_native_dedx(
      ProposalNativeDeviceView const& view, std::int32_t pdg_id, double energy_MeV) {
    if (!proposal_native_detail::finite(energy_MeV))
      return {NativeQueryStatus::NonFiniteInput, 0, 0, 0.};
    double total = 0.;
    bool found = false;
    for (std::uint32_t position = 0;
         position < view.getDedxAccumulationIndices().getSize(); ++position) {
      auto const index = view.getDedxAccumulationIndices()[position];
      if (index >= view.getDedxColumns().getSize())
        return {NativeQueryStatus::InvalidView, 0, 0, 0.};
      auto const columns = view.getDedxColumns();
      auto const& column = columns[index];
      if (column.pdgId_ != pdg_id) continue;
      found = true;
      // CrossSectionDEDXInterpolant::Calculate() returns zero below the
      // process-specific lower limit.  Other continuous processes for the
      // same particle may already be active, so skipping this column is part
      // of the scalar PROPOSAL summation semantics.
      if (energy_MeV < column.lowerEnergyLimitMeV_) continue;
      if (energy_MeV < column.spline_.axis_.low_ ||
          energy_MeV > column.spline_.axis_.high_)
        return {NativeQueryStatus::EnergyOutOfRange, 0, 0, 0.};
      total += proposal_native_detail::evaluate_cubic(view, column.spline_, energy_MeV) /
               column.componentWeight_;
    }
    if (!found) return {NativeQueryStatus::ParticleNotFound, 0, 0, 0.};
    return {NativeQueryStatus::Success, 0, 0, total};
  }

  CORSIKA_HOST_DEVICE inline NativeQueryResult query_proposal_native_loss_fraction(
      ProposalNativeDeviceView const& view, std::int32_t pdg_id, std::int32_t process_id,
      std::uint64_t component_hash, double energy_MeV, double quantile,
      bool photon_split) {
    if (!proposal_native_detail::valid_view(view))
      return {NativeQueryStatus::InvalidView, 0, 0, 0.};
    if (!proposal_native_detail::finite(energy_MeV) ||
        !proposal_native_detail::finite(quantile))
      return {NativeQueryStatus::NonFiniteInput, 0, 0, 0.};
    if (quantile < 0. || quantile > 1.)
      return {NativeQueryStatus::QuantileOutOfRange, 0, 0, 0.};
    auto const* column =
        proposal_native_detail::find_dndx(view, pdg_id, process_id, component_hash);
    if (!column) return {NativeQueryStatus::ColumnNotFound, 0, 0, 0.};
    // Both pair channels consume the photon (v=1). Explicit secondary-split
    // queries invert x instead: E_e-/E_gamma for Koch--Motz, E_mu+/E_gamma
    // for BKK. No extra rate column is inserted into interaction selection.
    bool const muon_split =
        photon_split && process_id == 1000000015 &&
        column->kinematicModel_ == NativeKinematicModel::OnlyStochastic;
    if (photon_split && (pdg_id != 22 ||
                         (!muon_split && column->kinematicModel_ !=
                                             NativeKinematicModel::PhotonPairFullLoss) ||
                         column->lossTransform_ != NativeLossTransform::RelativeLog))
      return {NativeQueryStatus::UnsupportedKinematics, 0, 0, 0.};
    // These public transport variables are degenerate at v=1.  Return before
    // considering the backing rate representation: a process may expose a
    // bicubic cumulative-rate table even though it has no loss variable to
    // invert (PhotoPairKochMotz and BKK photon-muon-pair in PROPOSAL 7.6.2).
    if ((column->kinematicModel_ == NativeKinematicModel::OnlyStochastic &&
         !muon_split) ||
        (column->kinematicModel_ == NativeKinematicModel::PhotonPairFullLoss &&
         !photon_split))
      return {NativeQueryStatus::Success, 0, 0, 1.};
    if (column->rateModel_ != NativeRateModel::BicubicSpline) {
      return {NativeQueryStatus::UnsupportedKinematics, 0, 0, 0.};
    }
    auto xmin = view.getConstants().electronMassMeV_ / energy_MeV;
    double xmax = 1. - xmin;
    if (muon_split) {
      auto const threshold =
          2. * view.getConstants().muonMassMeV_ * view.getConstants().sqrtE_;
      if (!(view.getConstants().muonMassMeV_ > 0.) || !(energy_MeV > threshold))
        return {NativeQueryStatus::EnergyOutOfRange, 0, 0, 0.};
      // Match PhotoMuPairProduction::GetKinematicLimits, including operation
      // order: don't substitute electron limits or infer these from v=1.
      auto const width = proposal_native_detail::square_root(
          1. - 2. * view.getConstants().sqrtE_ * view.getConstants().muonMassMeV_ /
                   energy_MeV);
      xmin = .5 * (1. - width);
      xmax = .5 * (1. + width);
    }
    if (photon_split && (xmin >= .5 || energy_MeV < column->spline_.energyAxis_.low_ ||
                         energy_MeV > column->spline_.energyAxis_.high_))
      return {NativeQueryStatus::EnergyOutOfRange, 0, 0, 0.};
    auto const limits =
        photon_split
            ? proposal_native_detail::NativeKinematicLimits{xmin, xmax, true}
            : proposal_native_detail::kinematic_limits(view, *column, energy_MeV);
    if (!limits.valid_) return {NativeQueryStatus::UnsupportedKinematics, 0, 0, 0.};
    if (limits.minimum_ == limits.maximum_)
      return {NativeQueryStatus::Success, 0, 0, limits.minimum_};
    // PROPOSAL's secondary CalculateRho normalizes the raw, unweighted CDF.
    auto const total = [&]() {
      if (!photon_split)
        return query_proposal_native_cumulative_rate(view, *column, energy_MeV, 1.);
      auto const raw = proposal_native_detail::evaluate_bicubic_proposal76(
          view, column->spline_, energy_MeV, 1.);
      if (!proposal_native_detail::finite(raw))
        return NativeQueryResult{NativeQueryStatus::InvalidView, 0, 0, 0.};
      return NativeQueryResult{NativeQueryStatus::Success, 0, 0, raw < 0. ? 0. : raw};
    }();
    if (total.status_ != NativeQueryStatus::Success) return total;
    // CrossSection::CalculateStochasticLoss receives q*CalculatedNdx(), then
    // multiplies that value by the component weight before calling
    // CrossSectionDNDXInterpolant::GetUpperLimit().  Preserve both rounding
    // steps and solve the *unweighted, unclipped* interpolant, exactly as
    // GetUpperLimit does.  Solving the already divided public cumulative rate
    // is algebraically equivalent but can stop at a different 20-bit Newton
    // iterate.
    auto const target = photon_split
                            ? quantile * total.value_
                            : (quantile * total.value_) * column->componentWeight_;
    if (quantile == 0.) return {NativeQueryStatus::Success, 0, 0, limits.minimum_};
    if (quantile == 1.) return {NativeQueryStatus::Success, 0, 0, limits.maximum_};

    auto const raw_cumulative = [&](double transformed_loss) {
      return proposal_native_detail::evaluate_bicubic_proposal76(
          view, column->spline_, energy_MeV, transformed_loss);
    };
    auto const residual_at = [&](double transformed_loss) {
      return raw_cumulative(transformed_loss) - target;
    };
    auto const sign = [](double value) {
      return (0. < value ? 1 : 0) - (value < 0. ? 1 : 0);
    };

    double lower = column->spline_.lossAxis_.low_;
    double upper = column->spline_.lossAxis_.high_;
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
    if (!proposal_native_detail::finite(lower_residual) ||
        !proposal_native_detail::finite(upper_residual)) {
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
        if (!proposal_native_detail::finite(middle_residual)) {
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
    while (!exact_root && !require_proposal_bisection && newton_iterations < 256) {
      auto const last_residual = previous_residual;
      delta2 = delta1;
      delta1 = delta;
      auto const residual = residual_at(current);
      previous_residual = residual;
      ++newton_iterations;
      if (!proposal_native_detail::finite(residual)) {
        require_proposal_bisection = true;
        break;
      }
      if (residual == 0.) {
        exact_root = true;
        break;
      }
      auto derivative =
          proposal_native_detail::evaluate_bicubic_loss_derivative_proposal76(
              view, column->spline_, energy_MeV, current);

      if (!proposal_native_detail::finite(derivative)) {
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
          delta = delta < 0. ? (current - lower) / 2. : (current - upper) / 2.;
        } else {
          delta = delta < 0. ? (current - upper) / 2. : (current - lower) / 2.;
        }
      } else {
        delta = residual / derivative;
      }
      auto const abs_delta = delta < 0. ? -delta : delta;
      auto const abs_delta2 = delta2 < 0. ? -delta2 : delta2;
      if (2. * abs_delta > abs_delta2) {
        auto shift = delta > 0. ? (current - lower) * .5 : (current - upper) * .5;
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
      lower = column->spline_.lossAxis_.low_;
      upper = column->spline_.lossAxis_.high_;
      lower_residual = residual_at(lower);
      upper_residual = residual_at(upper);
      if (!proposal_native_detail::finite(lower_residual) ||
          !proposal_native_detail::finite(upper_residual) ||
          lower_residual * upper_residual > 0.)
        return {NativeQueryStatus::RootNotConverged, newton_iterations,
                bisection_iterations, 0.};
      for (std::uint32_t iteration = 0; iteration <= 100; ++iteration) {
        auto const middle = (lower + upper) / 2.;
        auto const middle_residual = residual_at(middle);
        auto const current_lower_residual = residual_at(lower);
        if (!proposal_native_detail::finite(middle_residual) ||
            !proposal_native_detail::finite(current_lower_residual))
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
        photon_split
            ? limits.minimum_ * proposal_native_detail::exponential(
                                    current * proposal_native_detail::logarithm(
                                                  limits.maximum_ / limits.minimum_))
            : proposal_native_detail::physical_loss(view, *column, energy_MeV, current);
    if (!proposal_native_detail::finite(physical))
      return {NativeQueryStatus::RootNotConverged, newton_iterations,
              bisection_iterations, 0.};

    return {NativeQueryStatus::Success, newton_iterations, bisection_iterations,
            physical};
  }

  CORSIKA_HOST_DEVICE inline NativeQueryResult query_proposal_native_range(
      ProposalNativeDeviceView const& view, std::int32_t pdg_id, double energy_MeV) {
    auto const* column = proposal_native_detail::find_utility(view, pdg_id);
    if (!column) return {NativeQueryStatus::ParticleNotFound, 0, 0, 0.};
    if (!proposal_native_detail::finite(energy_MeV))
      return {NativeQueryStatus::NonFiniteInput, 0, 0, 0.};
    if (energy_MeV < column->lowerEnergyLimitMeV_ ||
        energy_MeV < column->spline_.axis_.low_ ||
        energy_MeV > column->spline_.axis_.high_)
      return {NativeQueryStatus::EnergyOutOfRange, 0, 0, 0.};
    auto const upper =
        proposal_native_detail::evaluate_cubic(view, column->spline_, energy_MeV);
    auto const lower = proposal_native_detail::evaluate_cubic(
        view, column->spline_, column->lowerEnergyLimitMeV_);
    auto value = column->reverse_ ? lower - upper : upper - lower;
    if (!proposal_native_detail::finite(value) || value < 0.)
      return {NativeQueryStatus::InvalidView, 0, 0, 0.};
    return {NativeQueryStatus::Success, 0, 0, value};
  }

  /*
   * Exact UtilityInterpolant::GetUpperLimit(initial_energy, grammage)
   * semantics used by scalar ContinuousProcess.  Keeping the initial energy
   * is essential: PROPOSAL brackets Newton on [lower_limit, initial_energy],
   * so an inverse of an absolute range over the whole table is not the same
   * floating-point operation even though it solves the same equation.
   */
  CORSIKA_HOST_DEVICE inline NativeQueryResult
  query_proposal_native_energy_after_continuous_loss(ProposalNativeDeviceView const& view,
                                                     std::int32_t pdg_id,
                                                     double initial_energy_MeV,
                                                     double grammage_g_per_cm2) {
    auto const* column = proposal_native_detail::find_utility(view, pdg_id);
    if (!column) return {NativeQueryStatus::ParticleNotFound, 0, 0, 0.};
    if (!proposal_native_detail::finite(initial_energy_MeV) ||
        !proposal_native_detail::finite(grammage_g_per_cm2) || grammage_g_per_cm2 < 0.)
      return {NativeQueryStatus::NonFiniteInput, 0, 0, 0.};
    if (initial_energy_MeV < column->lowerEnergyLimitMeV_ ||
        initial_energy_MeV > column->spline_.axis_.high_)
      return {NativeQueryStatus::EnergyOutOfRange, 0, 0, 0.};
    auto const maximum = query_proposal_native_range(view, pdg_id, initial_energy_MeV);
    if (maximum.status_ != NativeQueryStatus::Success) return maximum;
    if (grammage_g_per_cm2 > maximum.value_)
      return {NativeQueryStatus::EnergyOutOfRange, 0, 0, 0.};
    if (grammage_g_per_cm2 == maximum.value_)
      return {NativeQueryStatus::Success, 0, 0, column->lowerEnergyLimitMeV_};
    if (grammage_g_per_cm2 == 0.)
      return {NativeQueryStatus::Success, 0, 0, initial_energy_MeV};

    auto signed_grammage = grammage_g_per_cm2;
    if (column->reverse_) signed_grammage = -signed_grammage;
    auto const integrated_to_upper =
        proposal_native_detail::evaluate_cubic(view, column->spline_, initial_energy_MeV);
    auto const target = integrated_to_upper - signed_grammage;
    auto const residual = [&](double energy) {
      return proposal_native_detail::evaluate_cubic(view, column->spline_, energy) -
             target;
    };
    auto const derivative = [&](double energy) {
      return proposal_native_detail::evaluate_cubic_derivative_proposal76(
          view, column->spline_, energy);
    };
    auto const sign = [](double value) {
      return (0. < value ? 1 : 0) - (value < 0. ? 1 : 0);
    };
    auto proposal_bisection = [&](double accuracy, double& lower, double& upper,
                                  std::uint32_t& iterations) {
      auto const lower_value = residual(lower);
      auto const upper_value = residual(upper);
      if (!proposal_native_detail::finite(lower_value) ||
          !proposal_native_detail::finite(upper_value) || lower_value * upper_value > 0.)
        return false;
      for (std::uint32_t iteration = 0; iteration <= 100; ++iteration) {
        auto const center = (lower + upper) / 2.;
        auto const center_value = residual(center);
        auto const current_lower_value = residual(lower);
        if (!proposal_native_detail::finite(center_value) ||
            !proposal_native_detail::finite(current_lower_value))
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

    double lower = column->lowerEnergyLimitMeV_;
    double upper = initial_energy_MeV;
    std::uint32_t bisection_iterations = 0;
    auto const coarse_accuracy =
        (initial_energy_MeV - column->lowerEnergyLimitMeV_) * 1.e-2;
    if (!proposal_bisection(coarse_accuracy, lower, upper, bisection_iterations))
      return {NativeQueryStatus::RootNotConverged, 0, bisection_iterations, 0.};
    auto const initial_guess =
        upper == initial_energy_MeV ? initial_energy_MeV : (lower + upper) / 2.;
    auto const newton = proposal_native_detail::proposal76_newton(
        residual, derivative, initial_guess, lower, upper);
    if (!newton.runtimeError_ && proposal_native_detail::finite(newton.value_))
      return {NativeQueryStatus::Success, newton.iterations_, bisection_iterations,
              newton.value_};

    lower = column->lowerEnergyLimitMeV_;
    upper = initial_energy_MeV;
    if (!proposal_bisection(1.e-6, lower, upper, bisection_iterations))
      return {NativeQueryStatus::RootNotConverged, newton.iterations_,
              bisection_iterations, 0.};
    // UtilityInterpolant returns the first (lower) endpoint of its fallback
    // Bisection pair, unlike CrossSectionDNDX which returns the midpoint.
    return {NativeQueryStatus::Success, newton.iterations_, bisection_iterations, lower};
  }

  CORSIKA_HOST_DEVICE inline NativeQueryResult query_proposal_native_energy(
      ProposalNativeDeviceView const& view, std::int32_t pdg_id, double range_g_per_cm2) {
    auto const* column = proposal_native_detail::find_utility(view, pdg_id);
    if (!column) return {NativeQueryStatus::ParticleNotFound, 0, 0, 0.};
    if (!proposal_native_detail::finite(range_g_per_cm2) || range_g_per_cm2 < 0.)
      return {NativeQueryStatus::NonFiniteInput, 0, 0, 0.};
    auto const maximum =
        query_proposal_native_range(view, pdg_id, column->spline_.axis_.high_);
    if (maximum.status_ != NativeQueryStatus::Success) return maximum;
    if (range_g_per_cm2 > maximum.value_)
      return {NativeQueryStatus::EnergyOutOfRange, 0, 0, 0.};
    double lower = column->lowerEnergyLimitMeV_;
    double upper = column->spline_.axis_.high_;
    std::uint32_t iterations = 0;
    for (; iterations < 100; ++iterations) {
      // Bisect in PROPOSAL's native logarithmic axis, not linearly in E.
      auto const lower_coordinate =
          proposal_native_detail::axis_transform(column->spline_.axis_, lower);
      auto const upper_coordinate =
          proposal_native_detail::axis_transform(column->spline_.axis_, upper);
      auto const middle = proposal_native_detail::axis_back_transform(
          column->spline_.axis_, 0.5 * (lower_coordinate + upper_coordinate));
      auto const value = query_proposal_native_range(view, pdg_id, middle);
      if (value.status_ != NativeQueryStatus::Success) return value;
      if (value.value_ < range_g_per_cm2)
        lower = middle;
      else
        upper = middle;
      if ((upper - lower) <= 1.e-11 * upper) break;
    }
    return {NativeQueryStatus::Success, 0, iterations, 0.5 * (lower + upper)};
  }

} // namespace corsika::kokkos

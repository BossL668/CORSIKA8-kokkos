/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <cmath>

namespace corsika::kokkos {

  namespace polynomial_detail {

    KOKKOS_INLINE_FUNCTION PolynomialRoots stable_monic_quadratic(real_type b,
                                                                  real_type c) {
      PolynomialRoots result;
      if (c == 0) {
        result.append(0.);
        if (!(b == 0)) { result.append(-b); }
        return result;
      }
      real_type const multiplied = 4 * c;
      real_type const discriminant = fma(b, b, -multiplied);
      if (discriminant < 0) { return result; }
      if (discriminant == 0) {
        result.append(-b / 2);
        return result;
      }
      real_type const large =
          -(b + (!(b < 0) ? sqrt(discriminant) : -sqrt(discriminant))) / 2;
      result.append(large);
      result.append(c / large);
      return result;
    }

    KOKKOS_INLINE_FUNCTION real_type stable_resolvent(real_type p, real_type q,
                                                      real_type r) {
      real_type const quadraticCoefficient = p * p / 4 - r;
      real_type const negativeConstant = q * q / 8;
      // In this coefficient domain the only positive root is small. Keep the
      // production initial value and its four fixed Halley updates.
      if (p < 0 && r < 0 && negativeConstant < (-p) * quadraticCoefficient / 64 &&
          negativeConstant < p * r / 8) {
        real_type root = negativeConstant / quadraticCoefficient;
        for (unsigned int iteration = 0; iteration < 4; ++iteration) {
          real_type const value =
              ((root + p) * root + quadraticCoefficient) * root - negativeConstant;
          real_type const derivative = (3 * root + 2 * p) * root + quadraticCoefficient;
          real_type const secondDerivative = 6 * root + 2 * p;
          root -= 2 * value * derivative /
                  (2 * derivative * derivative - value * secondDerivative);
        }
        return root;
      }
      if (!(p < 0) && !(quadraticCoefficient < 0)) {
        real_type root = cbrt(negativeConstant);
        if (p > 0) {
          real_type const candidate = sqrt(negativeConstant / p);
          if (candidate < root) { root = candidate; }
        }
        if (quadraticCoefficient > 0) {
          real_type const candidate = negativeConstant / quadraticCoefficient;
          if (candidate < root) { root = candidate; }
        }
        for (unsigned int iteration = 0; iteration < 4; ++iteration) {
          real_type const value =
              ((root + p) * root + quadraticCoefficient) * root - negativeConstant;
          real_type const derivative = (3 * root + 2 * p) * root + quadraticCoefficient;
          real_type const secondDerivative = 6 * root + 2 * p;
          root -= 2 * value * derivative /
                  (2 * derivative * derivative - value * secondDerivative);
        }
        return root;
      }
      real_type const depressedQuadratic = (3 * quadraticCoefficient - p * p) / 9;
      real_type const depressedConstant =
          (p * (9 * quadraticCoefficient - 2 * p * p) + 27 * negativeConstant) / 54;
      real_type const discriminant =
          fma(depressedConstant, depressedConstant,
              depressedQuadratic * depressedQuadratic * depressedQuadratic);
      real_type root;
      if (!(discriminant < 0)) {
        real_type const cubeRoot =
            cbrt(depressedConstant +
                 (!(depressedConstant < 0) ? sqrt(discriminant) : -sqrt(discriminant)));
        root = (cubeRoot == 0 ? real_type(0) : cubeRoot - depressedQuadratic / cubeRoot) -
               p / 3;
      } else {
        real_type argument =
            depressedConstant /
            sqrt(-depressedQuadratic * depressedQuadratic * depressedQuadratic);
        if (argument > 1) { argument = 1; }
        if (argument < -1) { argument = -1; }
        root = 2 * sqrt(-depressedQuadratic) * cos(acos(argument) / 3) - p / 3;
      }
      return root;
    }

  } // namespace polynomial_detail

  KOKKOS_INLINE_FUNCTION PolynomialRoots stable_quartic_roots(double p, double q,
                                                              double r) {
    using real_type = polynomial_detail::real_type;
    PolynomialRoots result;
    // Preserve double power-of-two scaling before conversion to real_type.
    // Zero coefficients must not dominate a small nonzero polynomial's scale.
    int pExponent = -4096;
    int qExponent = -4096;
    int rExponent = -4096;
    if (p != 0) {
      int exponent;
      ::frexp(p, &exponent);
      pExponent = exponent >= 0 ? (exponent + 1) / 2 : exponent / 2;
    }
    if (q != 0) {
      int exponent;
      ::frexp(q, &exponent);
      qExponent = exponent >= 0 ? (exponent + 2) / 3 : exponent / 3;
    }
    if (r != 0) {
      int exponent;
      ::frexp(r, &exponent);
      rExponent = exponent >= 0 ? (exponent + 3) / 4 : exponent / 4;
    }
    int exponent = pExponent > qExponent ? pExponent : qExponent;
    exponent = exponent > rExponent ? exponent : rExponent;
    if (exponent == -4096) {
      result.append(0.);
      return result;
    }
    real_type const scaledP = ::ldexp(p, -2 * exponent);
    real_type const scaledQ = ::ldexp(q, -3 * exponent);
    real_type const scaledR = ::ldexp(r, -4 * exponent);
    double const scale = ::ldexp(1., exponent);
    if (q == 0) {
      auto squaredRoots = polynomial_detail::stable_monic_quadratic(scaledP, scaledR);
      for (unsigned int index = 0; index < squaredRoots.getSize(); ++index) {
        if (squaredRoots[index] < 0) { continue; }
        if (squaredRoots[index] == 0) {
          polynomial_detail::append_distinct(result, 0.);
        } else {
          double const root = static_cast<double>(polynomial_detail::sqrt(
                                  real_type(squaredRoots[index]))) *
                              scale;
          polynomial_detail::append_distinct(result, -root);
          polynomial_detail::append_distinct(result, root);
        }
      }
    } else {
      real_type const resolvent =
          polynomial_detail::stable_resolvent(scaledP, scaledQ, scaledR);
      if (!(resolvent > 0)) {
        result.append(polynomial_detail::infinity());
        return result;
      }
      real_type const linearCoefficient = polynomial_detail::sqrt(2 * resolvent);
      real_type const commonConstant = scaledP / 2 + resolvent;
      real_type const constantDifference = scaledQ / (2 * linearCoefficient);
      real_type firstConstant;
      real_type secondConstant;
      // The constants multiply to scaledR. Form the larger one without
      // cancellation, then obtain the smaller one by division.
      if (!(polynomial_detail::abs(commonConstant + constantDifference) <
            polynomial_detail::abs(commonConstant - constantDifference))) {
        secondConstant = commonConstant + constantDifference;
        firstConstant = scaledR / secondConstant;
      } else {
        firstConstant = commonConstant - constantDifference;
        secondConstant = scaledR / firstConstant;
      }
      auto first =
          polynomial_detail::stable_monic_quadratic(linearCoefficient, firstConstant);
      auto second =
          polynomial_detail::stable_monic_quadratic(-linearCoefficient, secondConstant);
      for (unsigned int index = 0; index < first.getSize(); ++index) {
        polynomial_detail::append_distinct(result, first[index] * scale);
      }
      for (unsigned int index = 0; index < second.getSize(); ++index) {
        polynomial_detail::append_distinct(result, second[index] * scale);
      }
    }
    // Keep the production ascending order before later boundary classification.
    for (unsigned int index = 1; index < result.getSize(); ++index) {
      double const value = result[index];
      unsigned int destination = index;
      while (destination && value < result[destination - 1]) {
        result[destination] = result[destination - 1];
        --destination;
      }
      result[destination] = value;
    }
    return result;
  }

} // namespace corsika::kokkos

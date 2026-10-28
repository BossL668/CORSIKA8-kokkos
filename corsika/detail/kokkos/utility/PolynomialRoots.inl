/*
 * (c) Copyright 2020, 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <cmath>

namespace corsika::kokkos {

  KOKKOS_INLINE_FUNCTION unsigned int PolynomialRoots::getSize() const { return size_; }

  KOKKOS_INLINE_FUNCTION double const& PolynomialRoots::operator[](
      unsigned int index) const {
    return values_[index];
  }

  KOKKOS_INLINE_FUNCTION double& PolynomialRoots::operator[](unsigned int index) {
    return values_[index];
  }

  template <typename TValue>
  KOKKOS_INLINE_FUNCTION void PolynomialRoots::append(TValue value) {
    values_[size_++] = static_cast<double>(value);
  }

  namespace polynomial_detail {

    KOKKOS_INLINE_FUNCTION real_type abs(real_type value) {
      return value < 0 ? -value : value;
    }

    KOKKOS_INLINE_FUNCTION real_type sqrt(real_type value) {
#if defined(__CUDA_ARCH__)
      return extended_sqrt(value);
#else
      return std::sqrt(value);
#endif
    }

    KOKKOS_INLINE_FUNCTION real_type cbrt(real_type value) {
#if defined(__CUDA_ARCH__)
      return extended_cbrt(value);
#else
      return std::cbrt(value);
#endif
    }

    KOKKOS_INLINE_FUNCTION real_type fma(real_type first, real_type second,
                                         real_type third) {
#if defined(__CUDA_ARCH__)
      return first * second + third;
#else
      return std::fma(first, second, third);
#endif
    }

    KOKKOS_INLINE_FUNCTION real_type acos(real_type value) {
#if defined(__CUDA_ARCH__)
      auto angle = ::acos(value.getHigh());
      if (value.getLow() == 0.) { return ExtendedPrecision(angle); }
      if (value.getHigh() == 1.) { return extended_sqrt(2 * (real_type(1) - value)); }
      if (value.getHigh() == -1.) {
        return real_type(3.14159265358979323846) -
               extended_sqrt(2 * (real_type(1) + value));
      }
      return ExtendedPrecision(angle) -
             value.getLow() / ::sqrt((1 - value.getHigh()) * (1 + value.getHigh()));
#else
      return std::acos(value);
#endif
    }

    KOKKOS_INLINE_FUNCTION real_type cos(real_type value) {
#if defined(__CUDA_ARCH__)
      return ExtendedPrecision(::cos(value.getHigh())) -
             ::sin(value.getHigh()) * value.getLow();
#else
      return std::cos(value);
#endif
    }

    KOKKOS_INLINE_FUNCTION void append_distinct(PolynomialRoots& roots, double value) {
      for (unsigned int index = 0; index < roots.getSize(); ++index) {
        if (::fabs(value - roots[index]) <=
            8 * 2.220446049250313e-16 * ::fmax(::fabs(value), ::fabs(roots[index]))) {
          return;
        }
      }
      if (roots.getSize() < 4) { roots.append(value); }
    }

  } // namespace polynomial_detail

  KOKKOS_INLINE_FUNCTION PolynomialRoots linear_roots(double a, double b) {
    PolynomialRoots result;
    if (a != 0.) { result.append(-b / a); }
    return result;
  }

  KOKKOS_INLINE_FUNCTION PolynomialRoots quadratic_roots(polynomial_detail::real_type a,
                                                         polynomial_detail::real_type b,
                                                         polynomial_detail::real_type c,
                                                         double epsilon) {
    using real_type = polynomial_detail::real_type;
    if (polynomial_detail::abs(a) < epsilon) {
      return linear_roots(static_cast<double>(b), static_cast<double>(c));
    }
    if (polynomial_detail::abs(c) < epsilon) {
      auto result = linear_roots(static_cast<double>(a), static_cast<double>(b));
      result.append(0.);
      return result;
    }
    real_type const multiplied = a * 4 * c;
    real_type const error = polynomial_detail::fma(a * 4, c, -multiplied);
    real_type const corrected = polynomial_detail::fma(b, b, -multiplied);
    real_type const radicand = corrected + error;
    PolynomialRoots result;
    if (polynomial_detail::abs(radicand) < epsilon) {
      result.append(-b / (2 * a));
      return result;
    }
    if (radicand < 0.) { return result; }
    real_type const root = c * 2 /
                           (b > 0 ? -b - polynomial_detail::sqrt(radicand)
                                  : -b + polynomial_detail::sqrt(radicand));
    result.append(root);
    result.append(c / (a * root));
    return result;
  }

} // namespace corsika::kokkos

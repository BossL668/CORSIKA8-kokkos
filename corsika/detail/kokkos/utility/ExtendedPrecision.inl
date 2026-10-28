/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <cmath>
#include <limits>

#if defined(__CUDACC__)
#include <math_constants.h>
#endif

namespace corsika::kokkos {

  namespace polynomial_detail {

    // Keep ordinary CUDA products/sums separately rounded. The explicit fma
    // below has a different purpose: recovering the multiplication residual.
    KOKKOS_INLINE_FUNCTION double product(double first, double second) {
#if defined(__CUDA_ARCH__)
      return __dmul_rn(first, second);
#else
      return first * second;
#endif
    }

    KOKKOS_INLINE_FUNCTION double sum(double first, double second) {
#if defined(__CUDA_ARCH__)
      return __dadd_rn(first, second);
#else
      return first + second;
#endif
    }

    KOKKOS_INLINE_FUNCTION double difference(double first, double second) {
#if defined(__CUDA_ARCH__)
      return __dsub_rn(first, second);
#else
      return first - second;
#endif
    }

    KOKKOS_INLINE_FUNCTION double infinity() {
#if defined(__CUDA_ARCH__)
      return CUDART_INF;
#else
      return std::numeric_limits<double>::infinity();
#endif
    }

    KOKKOS_INLINE_FUNCTION ExtendedPrecision normalize(double first, double second) {
      auto total = sum(first, second);
      return {total, difference(second, difference(total, first))};
    }

  } // namespace polynomial_detail

  KOKKOS_INLINE_FUNCTION bool extended_finite(double value) {
#if defined(__CUDA_ARCH__)
    return ::isfinite(value);
#else
    return std::isfinite(value);
#endif
  }

  KOKKOS_INLINE_FUNCTION ExtendedPrecision::ExtendedPrecision() {}

  KOKKOS_INLINE_FUNCTION ExtendedPrecision::ExtendedPrecision(double high)
      : high_(high) {}

  KOKKOS_INLINE_FUNCTION ExtendedPrecision::ExtendedPrecision(double high, double low)
      : high_(high)
      , low_(low) {}

  KOKKOS_INLINE_FUNCTION ExtendedPrecision::operator double() const {
    return polynomial_detail::sum(high_, low_);
  }

  KOKKOS_INLINE_FUNCTION double ExtendedPrecision::getHigh() const { return high_; }

  KOKKOS_INLINE_FUNCTION double ExtendedPrecision::getLow() const { return low_; }

  KOKKOS_INLINE_FUNCTION ExtendedPrecision operator+(ExtendedPrecision first,
                                                     ExtendedPrecision second) {
    using namespace polynomial_detail;
    auto total = sum(first.getHigh(), second.getHigh());
    auto recovered = difference(total, first.getHigh());
    auto error = sum(difference(first.getHigh(), difference(total, recovered)),
                     difference(second.getHigh(), recovered));
    return normalize(total, sum(error, sum(first.getLow(), second.getLow())));
  }

  KOKKOS_INLINE_FUNCTION ExtendedPrecision operator-(ExtendedPrecision value) {
    return {-value.getHigh(), -value.getLow()};
  }

  KOKKOS_INLINE_FUNCTION ExtendedPrecision operator-(ExtendedPrecision first,
                                                     ExtendedPrecision second) {
    return first + (-second);
  }

  KOKKOS_INLINE_FUNCTION ExtendedPrecision operator*(ExtendedPrecision first,
                                                     ExtendedPrecision second) {
    using namespace polynomial_detail;
    auto multiplied = product(first.getHigh(), second.getHigh());
    auto error = ::fma(first.getHigh(), second.getHigh(), -multiplied);
    error = sum(error, sum(product(first.getHigh(), second.getLow()),
                           product(first.getLow(), second.getHigh())));
    error = sum(error, product(first.getLow(), second.getLow()));
    return normalize(multiplied, error);
  }

  KOKKOS_INLINE_FUNCTION ExtendedPrecision operator/(ExtendedPrecision first,
                                                     ExtendedPrecision second) {
    auto quotient = first.getHigh() / second.getHigh();
    if (!extended_finite(quotient)) { return {quotient}; }
    auto residual = first - second * quotient;
    auto correction = residual.getHigh() / second.getHigh();
    residual = residual - second * correction;
    return (ExtendedPrecision(quotient) + correction) +
           residual.getHigh() / second.getHigh();
  }

  KOKKOS_INLINE_FUNCTION bool operator<(ExtendedPrecision first,
                                        ExtendedPrecision second) {
    return first.getHigh() < second.getHigh() ||
           (first.getHigh() == second.getHigh() && first.getLow() < second.getLow());
  }

  KOKKOS_INLINE_FUNCTION bool operator>(ExtendedPrecision first,
                                        ExtendedPrecision second) {
    return second < first;
  }

  KOKKOS_INLINE_FUNCTION bool operator==(ExtendedPrecision first,
                                         ExtendedPrecision second) {
    return first.getHigh() == second.getHigh() && first.getLow() == second.getLow();
  }

  KOKKOS_INLINE_FUNCTION ExtendedPrecision& operator+=(ExtendedPrecision& first,
                                                       ExtendedPrecision second) {
    first = first + second;
    return first;
  }

  KOKKOS_INLINE_FUNCTION ExtendedPrecision& operator-=(ExtendedPrecision& first,
                                                       ExtendedPrecision second) {
    first = first - second;
    return first;
  }

  KOKKOS_INLINE_FUNCTION ExtendedPrecision& operator/=(ExtendedPrecision& first,
                                                       ExtendedPrecision second) {
    first = first / second;
    return first;
  }

  KOKKOS_INLINE_FUNCTION ExtendedPrecision extended_sqrt(ExtendedPrecision value) {
    auto approximation = ::sqrt(value.getHigh());
    if (approximation == 0. || !extended_finite(approximation)) {
      return {approximation};
    }
    return ExtendedPrecision(approximation) +
           (value - ExtendedPrecision(approximation) * approximation) /
               (2 * approximation);
  }

  KOKKOS_INLINE_FUNCTION ExtendedPrecision extended_cbrt(ExtendedPrecision value) {
    ExtendedPrecision approximation(::cbrt(value.getHigh()));
    if (approximation.getHigh() == 0. || !extended_finite(approximation.getHigh())) {
      return approximation;
    }
    for (int iteration = 0; iteration < 2; ++iteration) {
      approximation += (value - approximation * approximation * approximation) /
                       (3 * approximation * approximation);
    }
    return approximation;
  }

} // namespace corsika::kokkos

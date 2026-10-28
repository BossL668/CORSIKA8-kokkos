/*
 * (c) Copyright 2020, 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/kokkos/utility/ExtendedPrecision.hpp>

namespace corsika::kokkos {

  /**
   * Fixed-capacity production root result, without dynamic allocation.
   *
   * Only the first getSize() values are meaningful. An empty result denotes no
   * returned real roots; the stable quartic can instead return an infinite
   * value to indicate an unresolved resolvent. Callers must preserve this
   * distinction. append() requires unused capacity and rounds to double.
   */
  class PolynomialRoots {
  public:
    /// Number of meaningful roots in the fixed four-value storage.
    [[nodiscard]] KOKKOS_INLINE_FUNCTION unsigned int getSize() const;

    /// Read a stored value; index must be less than four, with no bounds check.
    [[nodiscard]] KOKKOS_INLINE_FUNCTION double const& operator[](
        unsigned int index) const;
    /// Modify a stored value without changing the root count; index must be less than
    /// four.
    [[nodiscard]] KOKKOS_INLINE_FUNCTION double& operator[](unsigned int index);

    /**
     * Append after the stored roots, rounding the supplied value to double.
     * Requires getSize() < 4; no capacity or value validation is performed.
     */
    template <typename TValue>
    KOKKOS_INLINE_FUNCTION void append(TValue value);

  private:
    double values_[4]{};
    unsigned int size_{};
  };

  namespace polynomial_detail {
#if defined(__CUDA_ARCH__)
    using real_type = ExtendedPrecision;
#else
    using real_type = long double;
#endif
    KOKKOS_INLINE_FUNCTION real_type abs(real_type value);
    KOKKOS_INLINE_FUNCTION real_type sqrt(real_type value);
    KOKKOS_INLINE_FUNCTION real_type cbrt(real_type value);
    KOKKOS_INLINE_FUNCTION real_type fma(real_type first, real_type second,
                                         real_type third);
    KOKKOS_INLINE_FUNCTION real_type acos(real_type value);
    KOKKOS_INLINE_FUNCTION real_type cos(real_type value);
    KOKKOS_INLINE_FUNCTION void append_distinct(PolynomialRoots& roots, double value);
  } // namespace polynomial_detail

  /// Return the production real root of a*x+b=0 when a is nonzero.
  [[nodiscard]] KOKKOS_INLINE_FUNCTION PolynomialRoots linear_roots(double a, double b);

  /**
   * Preserve the production plane-quadratic solver for a*x*x+b*x+c=0.
   *
   * The absolute epsilon, degeneracy ordering and output order are unchanged.
   * This is distinct from the exact-degeneracy quadratic used inside the stable
   * quartic factorization.
   */
  [[nodiscard]] KOKKOS_INLINE_FUNCTION PolynomialRoots
  quadratic_roots(polynomial_detail::real_type a, polynomial_detail::real_type b,
                  polynomial_detail::real_type c, double epsilon = 1.e-12);

} // namespace corsika::kokkos

#include <corsika/detail/kokkos/utility/PolynomialRoots.inl>

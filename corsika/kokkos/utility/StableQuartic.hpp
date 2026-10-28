/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/kokkos/utility/PolynomialRoots.hpp>

namespace corsika::kokkos {

  namespace polynomial_detail {
    /// Production quadratic with exact degeneracy tests for quartic factorization.
    KOKKOS_INLINE_FUNCTION PolynomialRoots stable_monic_quadratic(real_type b,
                                                                  real_type c);

    /**
     * Production positive resolvent for the depressed quartic.
     *
     * Solves m^3+p*m^2+(p^2/4-r)*m-q^2/8=0 using the unchanged coefficient
     * branches: two use four Halley updates, the others use a closed form.
     * The enclosing quartic reports a nonpositive result as an infinite root.
     */
    KOKKOS_INLINE_FUNCTION real_type stable_resolvent(real_type p, real_type q,
                                                      real_type r);
  } // namespace polynomial_detail

  /**
   * Return production real roots of x^4+p*x^2+q*x+r=0 for finite coefficients.
   *
   * Preserves power-of-two scaling, factorization, root order and duplicate
   * handling. An unresolved resolvent returns one positive infinite value,
   * distinct from an empty real-root set. No residual scan or bisection is
   * performed. This utility does not select a physical boundary crossing.
   */
  [[nodiscard]] KOKKOS_INLINE_FUNCTION PolynomialRoots stable_quartic_roots(double p,
                                                                            double q,
                                                                            double r);

} // namespace corsika::kokkos

#include <corsika/detail/kokkos/utility/StableQuartic.inl>

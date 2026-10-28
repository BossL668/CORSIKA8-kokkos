/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <Kokkos_Macros.hpp>

namespace corsika::kokkos {

  /**
   * Local two-double expansion for the production polynomial resolvent.
   *
   * CUDA long double does not supply the additional precision used by the CPU
   * solver. This record preserves the production local arithmetic; it is not
   * a replacement for particle-state or model-table scalar types.
   */
  class ExtendedPrecision {
  public:
    /// Initialize both words to zero.
    KOKKOS_INLINE_FUNCTION ExtendedPrecision();
    /// Store the supplied leading word and a zero residual, without validation.
    KOKKOS_INLINE_FUNCTION ExtendedPrecision(double high);
    /// Store both supplied words unchanged, without input normalization or validation.
    KOKKOS_INLINE_FUNCTION ExtendedPrecision(double high, double low);
    /// Round the expansion using the production separately rounded high-plus-low sum.
    KOKKOS_INLINE_FUNCTION explicit operator double() const;

    /// Read the leading word without rounding the expansion to one scalar.
    [[nodiscard]] KOKKOS_INLINE_FUNCTION double getHigh() const;
    /// Read the residual word without changing or normalizing the expansion.
    [[nodiscard]] KOKKOS_INLINE_FUNCTION double getLow() const;

  private:
    double high_{};
    double low_{};
  };

  /// Use the production host/device finite classification for a scalar.
  KOKKOS_INLINE_FUNCTION bool extended_finite(double value);

  /**
   * @name Production two-double operators
   * Preserve the production arithmetic, correction order and word comparisons.
   * No extra input normalization, validity check or exceptional-value repair is
   * performed. Compensated operations keep their original normalization steps.
   * @{
   */
  KOKKOS_INLINE_FUNCTION ExtendedPrecision operator+(ExtendedPrecision first,
                                                     ExtendedPrecision second);
  KOKKOS_INLINE_FUNCTION ExtendedPrecision operator-(ExtendedPrecision value);
  KOKKOS_INLINE_FUNCTION ExtendedPrecision operator-(ExtendedPrecision first,
                                                     ExtendedPrecision second);
  KOKKOS_INLINE_FUNCTION ExtendedPrecision operator*(ExtendedPrecision first,
                                                     ExtendedPrecision second);
  KOKKOS_INLINE_FUNCTION ExtendedPrecision operator/(ExtendedPrecision first,
                                                     ExtendedPrecision second);
  KOKKOS_INLINE_FUNCTION bool operator<(ExtendedPrecision first,
                                        ExtendedPrecision second);
  KOKKOS_INLINE_FUNCTION bool operator>(ExtendedPrecision first,
                                        ExtendedPrecision second);
  KOKKOS_INLINE_FUNCTION bool operator==(ExtendedPrecision first,
                                         ExtendedPrecision second);
  KOKKOS_INLINE_FUNCTION ExtendedPrecision& operator+=(ExtendedPrecision& first,
                                                       ExtendedPrecision second);
  KOKKOS_INLINE_FUNCTION ExtendedPrecision& operator-=(ExtendedPrecision& first,
                                                       ExtendedPrecision second);
  KOKKOS_INLINE_FUNCTION ExtendedPrecision& operator/=(ExtendedPrecision& first,
                                                       ExtendedPrecision second);
  /** @} */

  /// Production square-root approximation with one correction.
  KOKKOS_INLINE_FUNCTION ExtendedPrecision extended_sqrt(ExtendedPrecision value);
  /// Production cube-root approximation with two corrections.
  KOKKOS_INLINE_FUNCTION ExtendedPrecision extended_cbrt(ExtendedPrecision value);

  namespace polynomial_detail {
    KOKKOS_INLINE_FUNCTION double product(double first, double second);
    KOKKOS_INLINE_FUNCTION double sum(double first, double second);
    KOKKOS_INLINE_FUNCTION double difference(double first, double second);
    KOKKOS_INLINE_FUNCTION double infinity();
    KOKKOS_INLINE_FUNCTION ExtendedPrecision normalize(double first, double second);
  } // namespace polynomial_detail

} // namespace corsika::kokkos

#include <corsika/detail/kokkos/utility/ExtendedPrecision.inl>

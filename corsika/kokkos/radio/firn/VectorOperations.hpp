/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <array>
#include <complex>
#include <cmath>
#include <stdexcept>

namespace corsika::kokkos::radio::firn {
  /** Scalar SI constants and components in the common local radio coordinates. */
  inline double constexpr c_m_per_s = 299792458.0;
  inline double constexpr mu0_h_per_m = 1.25663706212e-6;
  inline double constexpr elementary_charge_c = 1.602176634e-19;
  inline double constexpr pi = 3.141592653589793238462643383279502884;

  using vector_type = std::array<double, 3>;
  using complex_vector_type = std::array<std::complex<double>, 3>;
  using complex_matrix_type = std::array<std::array<std::complex<double>, 3>, 3>;

  vector_type add(vector_type const& a, vector_type const& b);
  vector_type subtract(vector_type const& a, vector_type const& b);
  vector_type multiply(vector_type const& a, double scale);
  double dot(vector_type const& a, vector_type const& b);
  vector_type cross(vector_type const& a, vector_type const& b);
  double norm(vector_type const& value);
  vector_type normalized(vector_type const& value);
  complex_matrix_type zero_matrix();
  complex_vector_type contract(complex_matrix_type const& matrix,
                               vector_type const& vector);
  bool finite(vector_type const& value);
} // namespace corsika::kokkos::radio::firn
#include <corsika/detail/kokkos/radio/firn/VectorOperations.inl>

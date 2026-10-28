/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

namespace corsika::kokkos::radio::firn {
  inline vector_type add(vector_type const& a, vector_type const& b) {
    return {a[0] + b[0], a[1] + b[1], a[2] + b[2]};
  }

  inline vector_type subtract(vector_type const& a, vector_type const& b) {
    return {a[0] - b[0], a[1] - b[1], a[2] - b[2]};
  }

  inline vector_type multiply(vector_type const& a, double scale) {
    return {a[0] * scale, a[1] * scale, a[2] * scale};
  }

  inline double dot(vector_type const& a, vector_type const& b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  }

  inline vector_type cross(vector_type const& a, vector_type const& b) {
    return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0]};
  }

  inline double norm(vector_type const& value) { return std::sqrt(dot(value, value)); }

  inline vector_type normalized(vector_type const& value) {
    double const length = norm(value);
    if (!(length > 0.0) || !std::isfinite(length)) {
      throw std::runtime_error("cannot normalize a zero or non-finite vector");
    }
    return multiply(value, 1.0 / length);
  }

  inline complex_matrix_type zero_matrix() {
    complex_matrix_type result{};
    for (auto& row : result) {
      for (auto& item : row) item = {0.0, 0.0};
    }
    return result;
  }

  inline complex_vector_type contract(complex_matrix_type const& matrix,
                                      vector_type const& vector) {
    complex_vector_type result{};
    for (std::size_t index = 0; index < 3; ++index) {
      result[index] = 0.0;
      for (std::size_t componentIndex = 0; componentIndex < 3; ++componentIndex)
        result[index] += matrix[index][componentIndex] * vector[componentIndex];
    }
    return result;
  }

  inline bool finite(vector_type const& value) {
    return std::isfinite(value[0]) && std::isfinite(value[1]) && std::isfinite(value[2]);
  }

} // namespace corsika::kokkos::radio::firn

/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/kokkos/radio/firn/VectorOperations.hpp>

namespace corsika::kokkos::radio::firn {

  inline complex_matrix_type transverse_dyadic(vector_type const& propagationDirection,
                                               std::complex<double> scale) {
    vector_type const direction = normalized(propagationDirection);
    complex_matrix_type result = zero_matrix();
    for (std::size_t index = 0; index < 3; ++index) {
      for (std::size_t componentIndex = 0; componentIndex < 3; ++componentIndex) {
        double const projector = (index == componentIndex ? 1.0 : 0.0) -
                                 direction[index] * direction[componentIndex];
        result[index][componentIndex] = scale * projector;
      }
    }
    return result;
  }

  inline complex_matrix_type stratified_polarization_transport_dyadic(
      vector_type const& emitDirection, vector_type const& receiveDirection,
      std::complex<double> scale) {
    vector_type const emit = normalized(emitDirection);
    vector_type const receive = normalized(receiveDirection);
    vector_type horizontal{emit[0], emit[1], 0.0};
    if (norm(horizontal) < 1.0e-14) { horizontal = {receive[0], receive[1], 0.0}; }
    vector_type const s =
        norm(horizontal) < 1.0e-14
            ? vector_type{0.0, 1.0, 0.0}
            : normalized(vector_type{-horizontal[1], horizontal[0], 0.0});
    vector_type const pSource = normalized(cross(s, emit));
    vector_type const pObserver = normalized(cross(s, receive));
    complex_matrix_type result = zero_matrix();
    for (std::size_t index = 0; index < 3; ++index) {
      for (std::size_t componentIndex = 0; componentIndex < 3; ++componentIndex) {
        result[index][componentIndex] =
            scale *
            (s[index] * s[componentIndex] + pObserver[index] * pSource[componentIndex]);
      }
    }
    return result;
  }

} // namespace corsika::kokkos::radio::firn

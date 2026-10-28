/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <cmath>

namespace corsika::kokkos {

  CORSIKA_HOST_DEVICE inline std::uint32_t pack_native_inverse_iterations(
      std::uint32_t newton, std::uint32_t bisection) {
    auto const bounded_newton = newton > 0xffffu ? 0xffffu : newton;
    auto const bounded_bisection = bisection > 0xffffu ? 0xffffu : bisection;
    return bounded_newton | (bounded_bisection << 16u);
  }

  CORSIKA_HOST_DEVICE inline std::uint32_t native_newton_iterations(
      TableQueryResult const& result) {
    return result.reserved_ & 0xffffu;
  }

  CORSIKA_HOST_DEVICE inline std::uint32_t native_bisection_iterations(
      TableQueryResult const& result) {
    return result.reserved_ >> 16u;
  }

  namespace table_query_detail {

    CORSIKA_HOST_DEVICE inline bool finite_value(double value) {
#if defined(__CUDA_ARCH__)
      return ::isfinite(value);
#else
      return std::isfinite(value);
#endif
    }

    CORSIKA_HOST_DEVICE inline TableQueryResult from_proposal_native(
        NativeQueryResult const& result, bool continuous) {
      switch (result.status_) {
        case NativeQueryStatus::Success:
          return {TableLookupStatus::Success,
                  pack_native_inverse_iterations(result.newtonIterations_,
                                                 result.bisectionIterations_),
                  result.value_};
        case NativeQueryStatus::ParticleNotFound:
          return {continuous ? TableLookupStatus::ContinuousParticleNotFound
                             : TableLookupStatus::ParticleNotFound,
                  pack_native_inverse_iterations(result.newtonIterations_,
                                                 result.bisectionIterations_),
                  0.};
        case NativeQueryStatus::ColumnNotFound:
          return {TableLookupStatus::ColumnNotFound,
                  pack_native_inverse_iterations(result.newtonIterations_,
                                                 result.bisectionIterations_),
                  0.};
        case NativeQueryStatus::EnergyOutOfRange:
          return {continuous ? TableLookupStatus::ContinuousEnergyOutOfRange
                             : TableLookupStatus::RateEnergyOutOfRange,
                  pack_native_inverse_iterations(result.newtonIterations_,
                                                 result.bisectionIterations_),
                  0.};
        case NativeQueryStatus::QuantileOutOfRange:
          return {TableLookupStatus::LossQuantileOutOfRange,
                  pack_native_inverse_iterations(result.newtonIterations_,
                                                 result.bisectionIterations_),
                  0.};
        case NativeQueryStatus::UnsupportedKinematics:
        case NativeQueryStatus::RootNotConverged:
          return {TableLookupStatus::InverseCdfUnavailable,
                  pack_native_inverse_iterations(result.newtonIterations_,
                                                 result.bisectionIterations_),
                  0.};
        case NativeQueryStatus::NonFiniteInput:
          return {TableLookupStatus::NonFiniteInput,
                  pack_native_inverse_iterations(result.newtonIterations_,
                                                 result.bisectionIterations_),
                  0.};
        case NativeQueryStatus::InvalidView:
          return {TableLookupStatus::InvalidTableView,
                  pack_native_inverse_iterations(result.newtonIterations_,
                                                 result.bisectionIterations_),
                  0.};
      }
      return {TableLookupStatus::InvalidTableView,
              pack_native_inverse_iterations(result.newtonIterations_,
                                             result.bisectionIterations_),
              0.};
    }
  } // namespace table_query_detail

} // namespace corsika::kokkos

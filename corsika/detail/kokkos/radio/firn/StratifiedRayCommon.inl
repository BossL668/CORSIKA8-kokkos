/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/kokkos/radio/firn/OpticalField.hpp>
#include <functional>

namespace corsika::kokkos::radio::firn {

  namespace detail {

    inline double simpson(std::function<double(double)> const& function, double a,
                          double b, std::size_t intervals = 2048) {
      if (a == b) return 0.0;
      if (intervals < 2) intervals = 2;
      if (intervals % 2) ++intervals;
      double const sign = b > a ? 1.0 : -1.0;
      double const lower = std::min(a, b);
      double const upper = std::max(a, b);
      double const h = (upper - lower) / static_cast<double>(intervals);
      double sum = function(lower) + function(upper);
      for (std::size_t index = 1; index < intervals; ++index) {
        sum += (index % 2 ? 4.0 : 2.0) * function(lower + h * index);
      }
      return sign * sum * h / 3.0;
    }

    inline double horizontal_distance(vector_type const& source,
                                      vector_type const& observer) {
      return std::hypot(observer[0] - source[0], observer[1] - source[1]);
    }

    inline vector_type horizontal_unit(vector_type const& source,
                                       vector_type const& observer) {
      double const distance = horizontal_distance(source, observer);
      if (!(distance > 0.0)) return {1.0, 0.0, 0.0};
      return {(observer[0] - source[0]) / distance, (observer[1] - source[1]) / distance,
              0.0};
    }

  } // namespace detail

} // namespace corsika::kokkos::radio::firn

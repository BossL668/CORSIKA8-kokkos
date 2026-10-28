/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include <corsika/kokkos/environment/ModelEvaluation.hpp>

#include <type_traits>

namespace {

  using point_function_type = double (*)(corsika::kokkos::EnvironmentModel const&,
                                         double const (&)[3]);
  static_assert(std::is_same_v<decltype(&corsika::kokkos::density), point_function_type>);
  static_assert(
      std::is_same_v<decltype(&corsika::kokkos::refractive_index), point_function_type>);

} // namespace

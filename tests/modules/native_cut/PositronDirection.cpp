/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include <corsika/framework/utility/PositronAtRest.hpp>
#ifndef CORSIKA_CUT_NATIVE_ONLY
#include <corsika/kokkos/em/algorithms/PositronAtRest.hpp>
#endif
#include "ProductionPositronAtRest.hpp"
#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>
int main() {
  unsigned checks = 0;
  auto check = [&](bool b) {
    ++checks;
    if (!b) throw std::runtime_error("direction differs");
  };
  try {
    static_assert(!corsika::EnableStoppedPositronAnnihilation);
    for (unsigned i = 0; i <= 256; ++i)
      for (unsigned j = 0; j <= 64; ++j) {
        double const u = double(i) / 256., phi = double(j) / 64.;
#ifdef CORSIKA_CUT_NATIVE_ONLY
        std::array<double, 3> a{}, c{};
#else
        std::array<double, 3> a{}, b{}, c{};
#endif
        corsika::positron_at_rest_direction(u, phi, a.data());
#ifndef CORSIKA_CUT_NATIVE_ONLY
        corsika::kokkos::positron_at_rest_direction(u, phi, b.data());
#endif
        corsika::production_cut_reference::positronAtRestDirection(u, phi, c.data());
        for (unsigned k = 0; k < 3; ++k) {
          check(a[k] == c[k]);
#ifndef CORSIKA_CUT_NATIVE_ONLY
          check(b[k] == c[k]);
#endif
        }
        check(std::abs(a[0] * a[0] + a[1] * a[1] + a[2] * a[2] - 1.) < 1e-14);
      }
    std::cout << "CHECKS=" << checks << '\n';
    return 0;
  } catch (std::exception const& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}

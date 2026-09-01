/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include "CubicInterpolation/Axis.h"
#include "CubicInterpolation/CubicSplines.h"
#include "CubicInterpolation/Interpolant.h"

#include <array>
#include <iomanip>
#include <iostream>
#include <memory>

int main() {
  using Spline = cubic_splines::CubicSplines<double>;
  auto definition = Spline::Definition{};
  definition.f = [](double x) { return x * x * x - 2. * x + 3.; };
  definition.axis =
      std::make_unique<cubic_splines::LinAxis<double>>(
          -2., 3., static_cast<std::size_t>(17));
  cubic_splines::Interpolant<Spline> interpolant(std::move(definition), "", "");

  std::cout << std::hexfloat;
  for (double const x : std::array<double, 5>{-2., -0.75, 0.375, 1.5, 3.}) {
    std::cout << x << ' ' << interpolant.evaluate(x) << ' '
              << interpolant.prime(x) << '\n';
  }
  return 0;
}

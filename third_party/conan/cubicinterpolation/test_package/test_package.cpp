/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include "CubicInterpolation/Axis.h"
#include "CubicInterpolation/CubicSplines.h"
#include "CubicInterpolation/Interpolant.h"

#include <cmath>
#include <cstddef>
#include <iostream>
#include <memory>

namespace {

bool close(double left, double right, double scale = 1.) {
  return std::abs(left - right) <= 1.e-12 * scale;
}

} // namespace

int main() {
  using Spline = cubic_splines::CubicSplines<double>;

  auto definition = Spline::Definition{};
  definition.f = [](double x) { return x * x * x - 2. * x + 3.; };
  definition.axis =
      std::make_unique<cubic_splines::LinAxis<double>>(
          -2., 3., static_cast<std::size_t>(17));

  cubic_splines::Interpolant<Spline> interpolant(std::move(definition), "", "");
  auto const before = interpolant.evaluate(0.375);
  auto const descriptor = interpolant.GetDefinition().GetAxis().ExportDescriptor();
  auto const exported = interpolant.ExportData();
  auto const after = interpolant.evaluate(0.375);

  if (descriptor.type != cubic_splines::AxisType::Linear ||
      descriptor.nodes != 17u || !close(descriptor.low, -2.) ||
      !close(descriptor.high, 3.)) {
    std::cerr << "axis export does not describe the original axis\n";
    return 1;
  }
  if (exported.values.size() != descriptor.nodes ||
      exported.node_derivatives.size() != descriptor.nodes) {
    std::cerr << "coefficient export has an unexpected size\n";
    return 2;
  }
  if (before != after || !std::isfinite(after)) {
    std::cerr << "read-only export changed ordinary spline evaluation\n";
    return 3;
  }

  std::cout.precision(17);
  std::cout << "ordinary_value=" << after << '\n'
            << "nodes=" << exported.values.size() << '\n';
  return 0;
}

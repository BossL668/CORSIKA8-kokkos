/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */
#include <corsika/framework/core/ParticleProperties.hpp>
#include <corsika/modules/thinning/EMThinning.hpp>
#include <corsika/accelerator/em/common/EmThinning.hpp>
#include <corsika/framework/geometry/RootCoordinateSystem.hpp>
#include <tests/common/SetupStack.hpp>
#include <iostream>
#include <random>
#include <stdexcept>
#include "KokkosScalarThinningDriver.hpp"

namespace {
using namespace corsika;
}

int main() {
  constexpr int count = 200000;
  auto& manager = corsika::RNGManager<>::getInstance();
  manager.registerRandomStream("thinning");
  manager.setSeed(20260906);
  auto& rng = manager.getRandomStream("thinning");
  std::mt19937_64 inputs(918273);
  std::uniform_real_distribution<double> uniform;
  auto const& cs = get_root_CoordinateSystem();
  DirectionVector const direction(cs, {0., 0., 1.});
  Point const origin(cs, {0_m, 0_m, 0_m});
  std::vector<accelerator::testing::ThinningInput> host(count);
  for (int i = 0; i < count; ++i) {
    auto& x = host[i];
    x.energy = 1. + 100. * uniform(inputs);
    x.weight = (i % 13 == 0) ? 0. : (0.1 + 10. * uniform(inputs));
    // Both test secondaries must have nonnegative kinetic energy in the real
    // CORSIKA stack; still sample highly asymmetric energy shares.
    double const electron_mass_GeV = get_mass(Code::Electron) / 1_GeV;
    x.e1 = electron_mass_GeV + (x.energy - 2. * electron_mass_GeV) *
                                 (0.00001 + 0.99998 * uniform(inputs));
    x.e2 = x.energy - x.e1;
    x.config = {i % 5 == 0 ? x.energy * 0.9 : x.energy,
                i % 7 == 0 ? x.weight : 0.1 + 100. * uniform(inputs), 1U,
                static_cast<std::uint32_t>(i % 2)};
    // Avoid invalid configuration; zero-weight input is still exercised.
    if (x.config.maximum_weight == 0.) x.config.maximum_weight = 1.;
    auto copied_rng = rng;
    x.u1 = uniform(copied_rng);
    x.u2 = uniform(copied_rng);
    // Use the actual weight-bearing CORSIKA stack and SecondaryView, including
    // its iterator/erase semantics. In particular, do not use raw-pointer mock
    // particle handles: that fixture gave optimization-sensitive stale reads
    // of updated weights at -O3 despite apparently correct survivor masks.
    test::Stack stack;
    auto primary = stack.addParticle(std::make_tuple(
        Code::Photon, x.energy * 1_GeV, direction, origin, 0_ns));
    primary.setWeight(x.weight);
    test::StackView view(primary);
    view.addSecondary(std::make_tuple(
        Code::Electron, x.e1 * 1_GeV - get_mass(Code::Electron), direction));
    view.addSecondary(std::make_tuple(
        Code::Positron, x.e2 * 1_GeV - get_mass(Code::Positron), direction));
    EMThinning scalar(x.config.threshold_GeV * 1_GeV,
                      x.config.maximum_weight, x.config.erase_zero_weight != 0);
    scalar.doSecondaries(view);
    x.expected.keep_mask = 0U;
    for (auto const& child : view) {
      if (child.getPID() == Code::Electron) {
        x.expected.first_weight = child.getWeight();
        x.expected.keep_mask |= 1U;
      } else if (child.getPID() == Code::Positron) {
        x.expected.second_weight = child.getWeight();
        x.expected.keep_mask |= 2U;
      } else {
        throw std::runtime_error("unexpected scalar thinning secondary species");
      }
    }
  }
  auto const comparison = accelerator::testing::compareThinningOnDevice(host);
  std::cout << "scalar thinning oracle cases=" << count
            << " decision_errors=" << comparison.decision_errors
            << " weight_errors_16ulp=" << comparison.weight_errors
            << " maximum_weight_ulp=" << comparison.maximum_weight_ulp
            << " non_bitwise_weights=" << comparison.non_bitwise_weights << '\n';
  return comparison.decision_errors == 0 && comparison.weight_errors == 0 ? 0 : 1;
}

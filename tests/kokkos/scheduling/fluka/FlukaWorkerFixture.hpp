/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */
#pragma once
#include <corsika/media/Environment.hpp>
#include <corsika/media/ExponentialFirn.hpp>
#include <corsika/media/HomogeneousMedium.hpp>
#include <corsika/media/IRefractiveIndexModel.hpp>
#include <corsika/media/UniformRefractiveIndex.hpp>
#include <string>
#include <vector>
namespace corsika::fluka_worker_test {
  inline auto make_environment(std::string const& material) {
    using Interface = IRefractiveIndexModel<IMediumModel>;
    Environment<Interface> environment;
    std::vector<Code> targets;
    if (material == "air") {
      targets = {Code::Hydrogen, Code::Nitrogen, Code::Oxygen, Code::Argon};
    } else if (material == "ice" || material == "firn") {
      targets = {Code::Hydrogen, Code::Oxygen};
    } else if (material == "rock") {
      targets = {Code::Oxygen, get_nucleus_code(28, 14)};
    } else {
      throw std::invalid_argument("Unknown test material");
    }
    // The environment, not the generator adapter, supplies the target set.
    std::vector<double> const fractions = material == "air"
                                              ? std::vector<double>{.25, .25, .25, .25}
                                              : std::vector<double>{2. / 3., 1. / 3.};
    NuclearComposition composition{targets, fractions};
    if (material == "firn") {
      environment.getUniverse()->setModelProperties<ExponentialFirn<Interface>>(
          environment.getCoordinateSystem(), composition);
    } else {
      using Medium = UniformRefractiveIndex<HomogeneousMedium<Interface>>;
      double const density = material == "air" ? 1. : (material == "ice" ? 917. : 2650.);
      environment.getUniverse()->setModelProperties<Medium>(
          1., density * 1_kg / (1_m * 1_m * 1_m), composition);
    }
    return environment;
  }
} // namespace corsika::fluka_worker_test

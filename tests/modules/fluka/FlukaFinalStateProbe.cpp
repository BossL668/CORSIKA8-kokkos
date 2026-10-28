/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include <corsika/framework/core/EnergyMomentumOperations.hpp>
#include <corsika/framework/geometry/RootCoordinateSystem.hpp>
#include <corsika/media/Environment.hpp>
#include <corsika/media/ExponentialFirn.hpp>
#include <corsika/media/HomogeneousMedium.hpp>
#include <corsika/media/IRefractiveIndexModel.hpp>
#include <corsika/media/UniformRefractiveIndex.hpp>
#include <corsika/modules/fluka/InteractionModel.hpp>

#ifdef CORSIKA_FLUKA_PRODUCTION_REFERENCE
#include "ProductionInteractionModel.hpp"
#elif defined(CORSIKA_FLUKA_BASELINE_REFERENCE)
#include "BaselineInteractionModel.hpp"
#endif

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace {
  using namespace corsika;
  using Particle = std::tuple<Code, HEPEnergyType, DirectionVector>;
  using FinalState = std::vector<Particle>;
#ifdef CORSIKA_FLUKA_PRODUCTION_REFERENCE
  using Model = production_fluka_reference::InteractionModel;
#elif defined(CORSIKA_FLUKA_BASELINE_REFERENCE)
  using Model = baseline_fluka_reference::InteractionModel;
#else
  using Model = corsika::fluka::InteractionModel;
#endif

  struct Collector {
    FinalState particles_;
    void addSecondary(Particle particle) { particles_.push_back(std::move(particle)); }
  };

  FinalState generate(Model& model, bool const direct, Code const projectile,
                      Code const target, FourMomentum const& projectileP4,
                      FourMomentum const& targetP4) {
#ifndef CORSIKA_FLUKA_BASELINE_REFERENCE
    if (direct) {
      return model.generateFinalState(projectile, target, projectileP4, targetP4);
    }
#else
    if (direct) { throw std::invalid_argument("Baseline has no final-state API"); }
#endif
    Collector view;
    model.doInteraction(view, projectile, target, projectileP4, targetP4);
    return std::move(view.particles_);
  }

  auto make_environment(std::string const& material) {
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
} // namespace

// Each invocation initializes FLUKA exactly once in a private test directory.
int main(int argc, char** argv) {
  try {
    if (argc != 5) { throw std::invalid_argument("material mode seed output required"); }
    bool const direct = std::string{argv[2]} == "direct";
    if (!direct && std::string{argv[2]} != "stack") {
      throw std::invalid_argument("Unknown test mode");
    }
    auto const seed = std::stoull(argv[3]);
    auto& manager = RNGManager<>::getInstance();
    manager.registerRandomStream("fluka");
    manager.setSeed(seed);
    auto environment = make_environment(argv[1]);
    auto const targets = get_all_elements_in_universe(environment);
    Model model{targets};
    auto const& frame = environment.getCoordinateSystem();
    std::ofstream output{argv[4]};
    if (!output) { throw std::runtime_error("Cannot create final-state audit output"); }
    output << std::hexfloat;
    std::uint64_t supplied = 0;
    std::uint64_t randomHash = 1469598103934665603ULL;
    // Same callback in every executable. Resetting the callback also clears the
    // C8 FLUKA bridge's prefetched buffer; Fortran initialization is not repeated.
    ::fluka::set_rng_function([rng = default_prng_type{seed, 0}, &supplied, &randomHash](
                                  double* values, std::size_t size) mutable {
      std::uniform_real_distribution<double> uniform{0., 1.};
      for (std::size_t i = 0; i < size; ++i) {
        values[i] = uniform(rng);
        std::uint64_t bits = 0;
        static_assert(sizeof(bits) == sizeof(values[i]));
        std::memcpy(&bits, &values[i], sizeof(bits));
        randomHash = (randomHash ^ bits) * 1099511628211ULL;
      }
      supplied += size;
    });

    std::size_t events = 0;
    std::size_t totalSecondaries = 0;
    for (auto target : targets) {
      if (model.getMaterialIndex(target) < 0) {
        throw std::runtime_error("Missing target");
      }
      output << "target " << static_cast<std::int32_t>(get_PDG(target)) << '\n';
      for (auto projectile : {Code::Proton, Code::PiPlus, Code::PiMinus, Code::KPlus}) {
        for (double momentum : {1., 20., 80.}) {
          for (bool tilted : {false, true}) {
            for (bool movingTarget : {false, true}) {
              auto const p = momentum * 1_GeV;
              auto const targetP = (movingTarget ? 0.01 : 0.) * 1_GeV;
              FourMomentum const projectileP4{
                  calculate_total_energy(p, get_mass(projectile)),
                  MomentumVector{frame, tilted ? 0.6 * p : 0_GeV, 0_GeV,
                                 tilted ? 0.8 * p : p}};
              FourMomentum const targetP4{
                  calculate_total_energy(targetP, get_mass(target)),
                  MomentumVector{frame, targetP, 0_GeV, 0_GeV}};
              auto const crossSection =
                  model.getCrossSection(projectile, target, projectileP4, targetP4);
              auto const particles =
                  generate(model, direct, projectile, target, projectileP4, targetP4);
              output << "event " << events++ << ' ' << crossSection / 1_mb << ' '
                     << particles.size() << ' ' << supplied << ' ' << randomHash << '\n';
              if (particles.empty()) { throw std::runtime_error("Empty final state"); }
              for (auto const& [pid, energy, direction] : particles) {
                auto const components = direction.getComponents(frame).getEigenVector();
                if (!std::isfinite(energy / 1_GeV) || energy < 0_GeV ||
                    !components.allFinite() || std::abs(components.norm() - 1.) > 1e-12) {
                  throw std::runtime_error("Invalid final-state particle");
                }
                output << static_cast<std::int32_t>(get_PDG(pid)) << ' ' << energy / 1_GeV
                       << ' ' << components[0] << ' ' << components[1] << ' '
                       << components[2] << '\n';
              }
              totalSecondaries += particles.size();
            }
          }
        }
      }
    }
    for (auto const& [projectile, target] : {std::pair{Code::Electron, Code::Oxygen},
                                             std::pair{Code::Proton, Code::Uranium}}) {
      FourMomentum const projectileP4{
          calculate_total_energy(20_GeV, get_mass(projectile)),
          MomentumVector{frame, 0_GeV, 0_GeV, 20_GeV}};
      FourMomentum const targetP4{get_mass(target), MomentumVector{frame}};
      bool rejected = false;
      try {
        (void)generate(model, direct, projectile, target, projectileP4, targetP4);
      } catch (std::runtime_error const& error) {
        output << "rejected " << error.what() << '\n';
        rejected = true;
      }
      if (!rejected) { throw std::runtime_error("Invalid reaction was accepted"); }
    }
    output << "complete " << events << ' ' << totalSecondaries << ' ' << supplied << ' '
           << randomHash << '\n';
    output.flush();
    if (!output) { throw std::runtime_error("Incomplete final-state audit output"); }
    return 0;
  } catch (std::exception const& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}

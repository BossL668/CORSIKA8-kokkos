/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include <SetupTestEnvironment.hpp>
#include <SetupTestStack.hpp>
#include <corsika/framework/random/RNGManager.hpp>
#include <corsika/modules/Sophia.hpp>
#include <corsika/modules/proposal/HadronicPhotonModel.hpp>
#include <sophia.hpp>
#ifdef COUNTED_MODELS
#include <corsika/modules/proposal/AirShowerInteraction.hpp>
#endif

#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>

namespace {
  using namespace corsika;
  unsigned checks{};
  void require(bool condition, char const* message) {
    ++checks;
    if (!condition) throw std::runtime_error(message);
  }

  // Control rejection only. No test-generated final state is substituted for SOPHIA.
  struct Gate {
    bool accepts_{true};
    unsigned calls_{};
    bool isValid(Code, Code, HEPEnergyType) const { return accepts_; }
    template <typename TView>
    void doInteraction(TView&, Code, Code, FourMomentum const&, FourMomentum const&) {
      ++calls_;
      throw std::runtime_error("Unexpected rejected generator invocation");
    }
  };

  template <typename TView>
  void dump(TView const& view, CoordinateSystemPtr const& coordinates,
            std::ostream& output) {
    output << "particles " << view.getSize() << '\n';
    for (auto const& particle : view) {
      auto const momentum = particle.getMomentum().getComponents(coordinates);
      auto const energy = particle.getEnergy() / 1_GeV;
      require(std::isfinite(energy), "non-finite secondary energy");
      require(std::isfinite(momentum.getX() / 1_GeV) &&
                  std::isfinite(momentum.getY() / 1_GeV) &&
                  std::isfinite(momentum.getZ() / 1_GeV),
              "non-finite secondary momentum");
      output << static_cast<int>(get_PDG(particle.getPID())) << ' ' << energy << ' '
             << momentum.getX() / 1_GeV << ' ' << momentum.getY() / 1_GeV << ' '
             << momentum.getZ() / 1_GeV << '\n';
    }
  }
} // namespace

int main(int argc, char** argv) {
  try {
    require(argc == 2, "evidence filename required");
    std::ofstream output(argv[1]);
    require(bool(output), "cannot open evidence file");
    output << std::hexfloat;
    logging::set_level(logging::level::warn);
    auto& manager = RNGManager<>::getInstance();
    manager.registerRandomStream("sophia");
    manager.registerRandomStream("proposal");
    manager.setSeed(74217);
    auto [environment, csPointer, node] = setup::testing::setup_environment(Code::Oxygen);
    auto const& coordinates = *csPointer;
    corsika::sophia::InteractionModel model;
#ifdef COUNTED_MODELS
    proposal::detail::CountedPhotonModel generator(model);
#else
    auto& generator = model;
#endif
    for (int i = 0; i < 49; ++i) {
      require(so_csydec_.idb[i] <= 0, "official SOPHIA leaves all particles stable");
      output << "idb " << i << ' ' << so_csydec_.idb[i] << '\n';
    }
    std::size_t events = 0, empty = 0;
    for (auto target : {Code::Proton, Code::Neutron}) {
      int id = corsika::sophia::convertToSophiaRaw(target);
      require(corsika::sophia::getSophiaMass(target) / 1_GeV ==
                  std::sqrt(get_sophia_mass2(id)),
              "official mass accessor uses sqrt(AM2)");
      auto const mass = get_mass(target);
      FourMomentum const nucleon(mass,
                                 MomentumVector(coordinates, {0_GeV, 0_GeV, 0_GeV}));
      // Include the previously patched threshold region and resonance/multipion energies.
      for (double value : {.14996, .1516, .16, .2, .5, 1., 2., 10., 50., 70., 1000.}) {
        auto const energy = value * 1_GeV;
        FourMomentum const photon(energy,
                                  MomentumVector(coordinates, {energy, 0_GeV, 0_GeV}));
        auto const valid =
            generator.isValid(Code::Photon, target, (photon + nucleon).getNorm());
        output << "valid " << id << ' ' << value << ' ' << valid << '\n';
        if (!valid) continue;
        for (int repeat = 0; repeat < 64; ++repeat) {
          manager.setSeed(916000 + repeat);
          auto [stack, view] =
              setup::testing::setup_stack(Code::Photon, energy, node, coordinates);
          generator.doInteraction(*view, Code::Photon, target, photon, nucleon);
          output << "sophia " << id << ' ' << value << ' ' << repeat << '\n';
          dump(*view, coordinates, output);
          ++events;
          if (view->getSize() == 0) ++empty;
          // Compare the continuation as well as every ordered final state.
          auto rng = manager.getRandomStream("sophia");
          for (int i = 0; i < 8; ++i) output << "rng " << rng() << '\n';
        }
      }
    }
    require(events >= 1000, "insufficient real generator coverage");
    require(empty > 0, "official threshold behavior was not exercised");
    Gate high;
    proposal::HadronicPhotonModel photonModel(generator, high, 80_GeV);
    for (auto target : {Code::Hydrogen, Code::Nitrogen, Code::Oxygen}) {
      for (auto energy : {.15_GeV, .2_GeV, 2_GeV, 70_GeV}) {
        manager.setSeed(74217);
        auto [stack, view] =
            setup::testing::setup_stack(Code::Photon, energy, node, coordinates);
        FourMomentum const photon(energy,
                                  MomentumVector(coordinates, {energy, 0_GeV, 0_GeV}));
        auto const result =
            photonModel.doHadronicPhotonInteraction(*view, coordinates, photon, target);
        require(result == ProcessReturn::Ok, "official photohadronic return value");
        if (energy > .2_GeV)
          require(view->getSize() > 0, "real SOPHIA final state missing");
        output << "photo " << static_cast<int>(get_PDG(target)) << ' ' << energy / 1_GeV
               << '\n';
        dump(*view, coordinates, output);
        auto rng = manager.getRandomStream("proposal");
        for (int i = 0; i < 8; ++i) output << "proposal-rng " << rng() << '\n';
      }
    }
    require(high.calls_ == 0, "low-energy path called high-energy model");
    Gate low;
    proposal::HadronicPhotonModel<Gate, Gate> rejected(low, high, 80_GeV);
    low.accepts_ = high.accepts_ = false;
    for (auto energy : {2_GeV, 100_GeV}) {
      auto [stack, view] =
          setup::testing::setup_stack(Code::Photon, energy, node, coordinates);
      FourMomentum const photon(energy,
                                MomentumVector(coordinates, {energy, 0_GeV, 0_GeV}));
      auto const result =
          rejected.doHadronicPhotonInteraction(*view, coordinates, photon, Code::Oxygen);
      require(result == ProcessReturn::Ok && view->getSize() == 0,
              "official unsupported model must skip secondary production");
      require(low.calls_ == 0 && high.calls_ == 0, "substitute model was called");
      output << "rejected " << energy / 1_GeV << ' ' << view->getSize() << '\n';
    }
    require(bool(output), "cannot write evidence");
    std::cout << "passed official SOPHIA/photohadronic checks=" << checks
              << " events=" << events << " empty=" << empty << '\n';
  } catch (std::exception const& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}

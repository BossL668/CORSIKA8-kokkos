/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include "LedgerDump.hpp"
#include <SetupTestEnvironment.hpp>
#include <SetupTestStack.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <corsika/framework/random/RNGManager.hpp>
#include <corsika/modules/Sophia.hpp>
#include <corsika/modules/proposal/HadronicInteractionModelFallback.hpp>
#include <corsika/modules/proposal/HadronicPhotonModel.hpp>
#include <corsika/modules/proposal/ThresholdPhotoproductionModel.hpp>
#include <corsika/modules/sophia/ThresholdKinematics.hpp>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sophia.hpp>
#include <stdexcept>
#include <string>
#include <utility>

namespace {
  using namespace corsika;
  unsigned checks{};
  void require(bool value, char const* message) {
    ++checks;
    if (!value) throw std::runtime_error(message);
  }
  // This model never provides a synthetic final state. It only lets the
  // HadronicPhotonModel constructor accept its threshold; later rejection tests
  // must throw before a generator call. Real LE events use SOPHIA below.
  struct RejectableModel {
    bool accepts_{true};
    unsigned calls_{};
    bool isValid(Code, Code, HEPEnergyType) const { return accepts_; }
    template <typename TView>
    void doInteraction(TView&, Code, Code, FourMomentum const&, FourMomentum const&) {
      ++calls_;
      throw std::runtime_error("Unexpected controlled generator invocation");
    }
  };
  HEPEnergyType minimum(Code target) {
#ifdef ORIGINAL_PRODUCTION
    return corsika::sophia::minimumCorsikaComEnergy(target);
#else
    return corsika::sophia::minimum_corsika_com_energy(target);
#endif
  }
  HEPEnergyType internal(Code target, HEPEnergyType energy) {
#ifdef ORIGINAL_PRODUCTION
    return corsika::sophia::internalComEnergy(target, energy);
#else
    return corsika::sophia::internal_com_energy(target, energy);
#endif
  }
  double beta(HEPEnergyType energy, HEPMassType mass) {
#ifdef ORIGINAL_PRODUCTION
    return corsika::sophia::eventgenTargetBeta(energy, mass);
#else
    return corsika::sophia::calculate_eventgen_target_beta(energy, mass);
#endif
  }
  template <typename TView>
  void dump_and_check(TView const& view, CoordinateSystemPtr const& coordinates,
                      FourMomentum const& incoming, std::ostream& output) {
    require(view.getSize() > 0, "actual generator returned no secondaries");
    HEPEnergyType totalEnergy = 0_GeV;
    MomentumVector totalMomentum(coordinates, {0_GeV, 0_GeV, 0_GeV});
    output << "particles " << view.getSize() << '\n';
    for (auto const& particle : view) {
      auto const momentum = particle.getMomentum();
      auto const components = momentum.getComponents(coordinates);
      totalEnergy += particle.getEnergy();
      totalMomentum += momentum;
      output << static_cast<int>(get_PDG(particle.getPID())) << ' '
             << particle.getEnergy() / 1_GeV << ' ' << components.getX() / 1_GeV << ' '
             << components.getY() / 1_GeV << ' ' << components.getZ() / 1_GeV << '\n';
    }
    double const scale = incoming.getTimeLikeComponent() / 1_GeV;
    double const energyResidual = (totalEnergy - incoming.getTimeLikeComponent()) / 1_GeV;
    double const momentumResidual =
        (totalMomentum - incoming.getSpaceLikeComponents()).getNorm() / 1_GeV;
    require(std::abs(energyResidual) <= 1.e-10 * std::max(1., scale), "energy closure");
    require(momentumResidual <= 1.e-10 * std::max(1., scale), "momentum closure");
    output << "residual " << energyResidual << ' ' << momentumResidual << '\n';
  }
} // namespace

int main(int argc, char** argv) {
  try {
    require(argc == 2, "dedicated evidence output required");
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
    corsika::sophia::InteractionModel generator;
    std::array<int, 9> const unstable{24, 25, 26, 31, 32, 39, 40, 41, 42};
    for (int i = 0; i < 49; ++i) {
      bool const restored =
          std::find(unstable.begin(), unstable.end(), i) != unstable.end();
      require(restored ? so_csydec_.idb[i] >= 0 : so_csydec_.idb[i] <= 0,
              "production IDB mask");
      output << "idb " << i << ' ' << so_csydec_.idb[i] << '\n';
    }
    for (Code target : {Code::Proton, Code::Neutron, Code::Hydrogen}) {
      auto const boundary = minimum(target);
      require(!generator.isValid(Code::Photon, target, boundary - 1_eV),
              "threshold below");
      require(generator.isValid(Code::Photon, target, boundary + 1_eV),
              "threshold above");
      output << "threshold " << static_cast<int>(get_PDG(target)) << ' '
             << boundary / 1_GeV << ' ' << internal(target, boundary) / 1_GeV << '\n';
    }
    for (Code target : {Code::Proton, Code::Neutron}) {
      int id = corsika::sophia::convertToSophiaRaw(target);
      auto const mass = corsika::sophia::getSophiaMass(target);
      require(mass / 1_GeV == get_sophia_mass(id), "AM accessor, not sqrt(AM2)");
      output << "mass " << id << ' ' << get_sophia_mass(id) << ' ' << get_sophia_mass2(id)
             << '\n';
      for (double gamma : {.99, 1., 2., 999., 1000., 1001., 1.e6})
        output << "beta " << id << ' ' << gamma << ' ' << beta(gamma * mass, mass)
               << '\n';
      auto const targetMass = get_mass(target);
      auto const threshold = minimum(target) + 1_keV;
      auto const thresholdPhoton =
          (threshold * threshold - targetMass * targetMass) / (2. * targetMass);
      for (HEPEnergyType energy : {thresholdPhoton, 1.2_GeV, 10_GeV, 1000_GeV}) {
        FourMomentum const photon(energy,
                                  MomentumVector(coordinates, {energy, 0_GeV, 0_GeV}));
        FourMomentum const nucleon(targetMass,
                                   MomentumVector(coordinates, {0_GeV, 0_GeV, 0_GeV}));
        auto [stack, viewPointer] =
            setup::testing::setup_stack(Code::Photon, 10_GeV, node, coordinates);
        auto& view = *viewPointer;
        generator.doInteraction(view, Code::Photon, target, photon, nucleon);
        output << "sophia " << id << ' ' << energy / 1_GeV << '\n';
        dump_and_check(view, coordinates, photon + nucleon, output);
      }
    }
    proposal::ThresholdPhotoproductionModel thresholdModel;
    proposal::HadronicInteractionModelFallback<corsika::sophia::InteractionModel,
                                               proposal::ThresholdPhotoproductionModel>
        combined(generator, thresholdModel);
    for (Code target : {Code::Proton, Code::Neutron}) {
      auto const mass = get_mass(target);
      auto const physicalThreshold = mass + get_mass(Code::Pi0);
      auto const upper = minimum(target);
      require(upper > physicalThreshold, "real threshold gap exists");
      require(!thresholdModel.isValid(Code::Photon, target, physicalThreshold - 1_eV),
              "gap lower rejection");
      require(thresholdModel.isValid(Code::Photon, target, physicalThreshold),
              "gap closed lower boundary");
      require(!thresholdModel.isValid(Code::Photon, target, upper),
              "gap open upper boundary");
      auto const center = .5 * (physicalThreshold + upper);
      auto const energy = (center * center - mass * mass) / (2. * mass);
      FourMomentum const photon(energy,
                                MomentumVector(coordinates, {energy, 0_GeV, 0_GeV}));
      FourMomentum const nucleon(mass,
                                 MomentumVector(coordinates, {0_GeV, 0_GeV, 0_GeV}));
      auto& rng = manager.getRandomStream("proposal");
      auto expectedRng = rng;
      std::uniform_real_distribution<double> uniform(0., 1.);
      uniform(expectedRng);
      uniform(expectedRng);
      auto [stack, viewPointer] =
          setup::testing::setup_stack(Code::Photon, 10_GeV, node, coordinates);
      auto& view = *viewPointer;
      combined.doInteraction(view, Code::Photon, target, photon, nucleon);
      require(view.getSize() == 2, "two-body threshold final state");
      // Check subsequent words against exactly two production uniform draws.
      // The diagnostic words are consumed in both fresh-process executables.
      for (int i = 0; i != 16; ++i)
        require(rng() == expectedRng(), "threshold RNG draw order");
      output << "threshold-final " << static_cast<int>(get_PDG(target)) << '\n';
      dump_and_check(view, coordinates, photon + nucleon, output);
    }
#ifdef ORIGINAL_PRODUCTION
    output << "fallback-count " << combined.statistics().preferred_interactions << ' '
           << combined.statistics().fallback_interactions << ' '
           << thresholdModel.statistics().interactions << '\n';
#else
    output << "fallback-count " << combined.statistics().preferredInteractions_ << ' '
           << combined.statistics().fallbackInteractions_ << ' '
           << thresholdModel.statistics().interactions_ << '\n';
#endif
    RejectableModel heGate;
    proposal::HadronicPhotonModel<decltype(combined), RejectableModel> photonModel(
        combined, heGate, 80_GeV);
    heGate.accepts_ = false;
    auto [stack, viewPointer] =
        setup::testing::setup_stack(Code::Photon, 10_GeV, node, coordinates);
    auto& view = *viewPointer;
    HEPEnergyType const photonEnergy = 2_GeV;
    FourMomentum const realPhoton(
        photonEnergy, MomentumVector(coordinates, {photonEnergy, 0_GeV, 0_GeV}));
    photonModel.doHadronicPhotonInteraction(view, coordinates, realPhoton, Code::Oxygen);
    require(view.getSize() > 0, "real LE photohadronic final state");
    require(heGate.calls_ == 0, "LE path did not invoke HE generator");
    auto const& ledger = photonModel.energyLedgerStatistics();
    double const actualTarget = print_ledger(ledger, output);
    require(actualTarget == get_mass(Code::Proton) / 1_GeV ||
                actualTarget == get_mass(Code::Neutron) / 1_GeV,
            "LE ledger records sampled nucleon, not whole oxygen nucleus");
    FourMomentum const actualIncoming(photonEnergy + actualTarget * 1_GeV,
                                      realPhoton.getSpaceLikeComponents());
    output << "photohadronic-real-LE\n";
    dump_and_check(view, coordinates, actualIncoming, output);
    // Only the rejection condition is controlled here; no mock final states.
    heGate.accepts_ = true;
    RejectableModel leGate;
    proposal::HadronicPhotonModel<RejectableModel, RejectableModel> rejected(
        leGate, heGate, 80_GeV);
    leGate.accepts_ = false;
    heGate.accepts_ = false;
    for (auto energy : {2_GeV, 100_GeV}) {
      auto [stack, viewPointer] =
          setup::testing::setup_stack(Code::Photon, 10_GeV, node, coordinates);
      auto& view = *viewPointer;
      FourMomentum const photon(energy,
                                MomentumVector(coordinates, {energy, 0_GeV, 0_GeV}));
      bool threw = false;
      try {
        rejected.doHadronicPhotonInteraction(view, coordinates, photon, Code::Oxygen);
      } catch (std::runtime_error const& error) {
        std::string const text = error.what();
        threw = text.find(energy < 80_GeV
                              ? "low-energy photo-hadronic"
                              : "high-energy photo-hadronic") != std::string::npos;
        output << "rejection " << energy / 1_GeV << ' ' << text << '\n';
      }
      require(threw, "selected LE/HE final-state rejection must throw");
      require(view.getSize() == 0, "rejected interaction did not add final state");
      require(leGate.calls_ == 0 && heGate.calls_ == 0,
              "rejection happened before generator invocation");
    }
    require(bool(output), "cannot write evidence file");
    std::cout << "passed actual SOPHIA/threshold generator checks=" << checks << '\n';
    return 0;
  } catch (std::exception const& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}

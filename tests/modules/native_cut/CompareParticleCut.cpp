/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include <corsika/modules/ParticleCut.hpp>
#include "ProductionParticleCut.hpp"
#include "OfficialParticleCut.hpp"
#include <corsika/setup/SetupStack.hpp>
#include <corsika/framework/geometry/RootCoordinateSystem.hpp>
#include <corsika/media/HomogeneousMedium.hpp>
#include <corsika/media/MediumPropertyModel.hpp>
#include <corsika/media/UniformMagneticField.hpp>
#include <array>
#include <cmath>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <vector>

using namespace corsika;
namespace {
  unsigned checks = 0;
  void check(bool condition, char const* description) {
    ++checks;
    if (!condition) throw std::runtime_error(description);
  }
#define REQUIRE(...) check(bool(__VA_ARGS__), #__VA_ARGS__)
  using EnvironmentType =
      Environment<IMediumPropertyModel<IMagneticFieldModel<IMediumModel>>>;
  using NativeStack = setup::Stack<EnvironmentType>;
  using Entry = std::tuple<int, double, double, double, double>;
  struct RecordingWriter : WriterOff {
    std::vector<Entry> records;
    void write(Point const& point, Code pid, HEPEnergyType deposit) {
      auto d = point.getCoordinates(get_root_CoordinateSystem());
      records.emplace_back(static_cast<int>(get_PDG(pid)), deposit / 1_GeV,
                           d.getX() / 1_m, d.getY() / 1_m, d.getZ() / 1_m);
    }
  };
  struct Input {
    Code pid;
    double energy;
    double time;
    double weight;
  };
  using Candidate = ParticleCut<RecordingWriter>;
  using Production = production_cut_reference::ParticleCut<RecordingWriter>;
  using Official = official_cut_reference::ParticleCut<RecordingWriter>;

  void compare_statistics(Candidate const& candidate, Production const& reference) {
    auto const& a = candidate.statistics();
    auto const& b = reference.statistics();
    REQUIRE(a.particles_ == b.particles);
    REQUIRE(a.invisibleParticles_ == b.invisible_particles);
    REQUIRE(a.weightedKineticEnergyGeV_ == b.weighted_kinetic_energy_GeV);
    REQUIRE(a.weightedRestMassEnergyGeV_ == b.weighted_rest_mass_energy_GeV);
    REQUIRE(a.weightedInvisibleKineticEnergyGeV_ ==
            b.weighted_invisible_kinetic_energy_GeV);
    REQUIRE(a.stoppedPositronAnnihilations_ == b.stopped_positron_annihilations);
    REQUIRE(a.weightedMediumRestMassInputGeV_ == b.weighted_medium_rest_mass_input_GeV);
    REQUIRE(a.byPdg_.size() == b.by_pdg.size());
    for (auto const& [pdg, x] : a.byPdg_) {
      auto const& y = b.by_pdg.at(pdg);
      REQUIRE(x.particles_ == y.particles);
      REQUIRE(x.weightedKineticEnergyGeV_ == y.weighted_kinetic_energy_GeV);
      REQUIRE(x.weightedRestMassEnergyGeV_ == y.weighted_rest_mass_energy_GeV);
    }
  }

  template <class Cut>
  std::vector<Entry> secondaries(Cut& cut, std::vector<Input> const& input) {
    auto const cs = get_root_CoordinateSystem();
    NativeStack stack;
    auto primary = stack.addParticle(std::make_tuple(Code::Proton, 100_GeV,
                                                     DirectionVector(cs, {0., 0., 1.}),
                                                     Point(cs, 1_m, 2_m, 3_m), 0_s));
    typename NativeStack::stack_view_type view(primary);
    for (auto const& item : input) {
      auto p = view.addSecondary(std::make_tuple(item.pid, item.energy * 1_GeV,
                                                 DirectionVector(cs, {0., 0., 1.})));
      p.setTime(item.time * 1_s);
      p.setWeight(item.weight);
    }
    cut.doSecondaries(view);
    std::vector<Entry> remaining;
    for (auto p = view.begin(); p != view.end(); ++p) {
      remaining.emplace_back(static_cast<int>(get_PDG(p.getPID())),
                             p.getKineticEnergy() / 1_GeV, p.getWeight(),
                             p.getTime() / 1_s, 0.);
    }
    REQUIRE(remaining.size() == view.getEntries());
    REQUIRE(stack.getEntries() == view.getEntries() + 1);
    return remaining;
  }

  template <class Cut>
  ProcessReturn continuous(Cut& cut, Input const& item, double postEnergy,
                           double postTime, bool limitFlag) {
    auto const cs = get_root_CoordinateSystem();
    NativeStack stack;
    auto p = stack.addParticle(
        std::make_tuple(item.pid, item.energy * 1_GeV, DirectionVector(cs, {0., 0., 1.}),
                        Point(cs, 1_m, 2_m, 3_m), item.time * 1_s));
    p.setWeight(item.weight);
    StraightTrajectory trajectory(
        Line(p.getPosition(), VelocityVector(cs, {0_m / 1_s, 0_m / 1_s, 1_m / 1_s})),
        1_s);
    Step step(p, trajectory);
    step.add_dEkin((postEnergy - item.energy) * 1_GeV);
    step.add_dt((postTime - item.time - 1.) * 1_s);
    REQUIRE(std::isinf(cut.getMaxStepLength(p, trajectory) / 1_m));
    auto const status = cut.doContinuous(step, limitFlag);
    REQUIRE(stack.getEntries() ==
            1); // ProcessReturn leaves actual removal to the caller.
    REQUIRE(p.getKineticEnergy() == item.energy * 1_GeV);
    return status;
  }

  void fixture(bool invisible) {
    Candidate candidate(1_GeV, 2_GeV, 3_GeV, 4_GeV, 5_GeV, invisible);
    Production production(1_GeV, 2_GeV, 3_GeV, 4_GeV, 5_GeV, invisible);
    Official official(1_GeV, 2_GeV, 3_GeV, 4_GeV, 5_GeV, invisible);
    REQUIRE(candidate.getElectronKineticECut() == production.getElectronKineticECut());
    REQUIRE(candidate.getPhotonKineticECut() == production.getPhotonKineticECut());
    REQUIRE(candidate.getHadronKineticECut() == production.getHadronKineticECut());
    REQUIRE(candidate.getMuonKineticECut() == production.getMuonKineticECut());
    REQUIRE(candidate.getTauKineticECut() == production.getTauKineticECut());
    REQUIRE(YAML::Dump(candidate.getConfig()) == YAML::Dump(production.getConfig()));
    REQUIRE(candidate.getConfig()["stopped_positron_annihilation"].as<std::string>() ==
            "disabled_original_c8_cut");
    compare_statistics(candidate, production);
    std::vector<Input> inputs;
    for (auto const pid :
         {Code::Electron, Code::Positron, Code::Photon, Code::Proton, Code::PiPlus,
          Code::MuPlus, Code::MuMinus, Code::TauPlus, Code::TauMinus, Code::NuE,
          Code::NuMu, get_nucleus_code(16, 8)}) {
      auto const threshold = get_kinetic_energy_propagation_threshold(pid) / 1_GeV *
                             (is_nucleus(pid) ? get_nucleus_A(pid) : 1);
      for (auto energy : {0., std::nextafter(threshold, 0.), threshold,
                          std::nextafter(threshold, INFINITY), threshold + 100.})
        for (auto time : {0., .01, .010000001})
          for (auto weight : {.25, 1., 7.}) inputs.push_back({pid, energy, time, weight});
    }
    auto a = secondaries(candidate, inputs);
    auto b = secondaries(production, inputs);
    auto c = secondaries(official, inputs);
    REQUIRE(a == b);
    REQUIRE(a == c);
    REQUIRE(candidate.records == production.records);
    REQUIRE(candidate.records == official.records);
    compare_statistics(candidate, production);
    // Production statistics accumulate across the inherited writer lifecycle.
    auto count = candidate.statistics().particles_;
    candidate.startOfLibrary("unused-writer-off");
    production.startOfLibrary("unused-writer-off");
    for (unsigned shower = 0; shower < 2; ++shower) {
      candidate.startOfShower(shower);
      production.startOfShower(shower);
      for (auto pid :
           {Code::Electron, Code::Positron, Code::Photon, Code::NuE, Code::MuPlus})
        for (auto energy : {-.1, 0., .5, 1., 10.})
          for (auto time : {0., .01, .010000001})
            for (bool flag : {false, true}) {
              Input input{pid, 100., 0., 2.5};
              auto pa = continuous(candidate, input, energy, time, flag);
              auto pb = continuous(production, input, energy, time, flag);
              auto pc = continuous(official, input, energy, time, flag);
              REQUIRE(pa == pb);
              REQUIRE(pa == pc);
              REQUIRE(candidate.records == production.records);
              REQUIRE(candidate.records == official.records);
              compare_statistics(candidate, production);
            }
      candidate.endOfShower(shower);
      production.endOfShower(shower);
      REQUIRE(candidate.statistics().particles_ > count);
      count = candidate.statistics().particles_;
      compare_statistics(candidate, production);
    }
    candidate.endOfLibrary();
    production.endOfLibrary();
    REQUIRE(candidate.statistics().particles_ == count);
    REQUIRE(candidate.statistics().stoppedPositronAnnihilations_ == 0);
    REQUIRE(candidate.statistics().weightedMediumRestMassInputGeV_ == 0.);
    // Deposits must retain the endpoints: secondaries at z=3m, continuous at z=4m.
    for (auto const& record : candidate.records) {
      REQUIRE(std::get<2>(record) == 1.);
      REQUIRE(std::get<3>(record) == 2.);
      REQUIRE(std::get<4>(record) == 3. || std::get<4>(record) == 4.);
    }
  }

  void other_constructors() {
    std::vector<Input> input{{Code::Electron, 3., 0., 2.},
                             {Code::Positron, 30., 0., .5},
                             {Code::Photon, 3., 0., 1.},
                             {Code::Proton, 30., .010000001, 3.},
                             {Code::NuE, 10., 0., 1.}};
    {
      Candidate c(4_GeV, false);
      Production p(4_GeV, false);
      Official o(4_GeV, false);
      auto a = secondaries(c, input), b = secondaries(p, input),
           d = secondaries(o, input);
      REQUIRE(a == b);
      REQUIRE(a == d);
      REQUIRE(c.records == p.records);
      REQUIRE(c.records == o.records);
      compare_statistics(c, p);
    }
    {
      std::unordered_map<Code const, HEPEnergyType const> cuts{{Code::Electron, 5_GeV},
                                                               {Code::Positron, 50_GeV}};
      Candidate c(cuts, false);
      Production p(cuts, false);
      Official o(cuts, false);
      auto a = secondaries(c, input), b = secondaries(p, input),
           d = secondaries(o, input);
      REQUIRE(a == b);
      REQUIRE(a == d);
      REQUIRE(c.records == p.records);
      REQUIRE(c.records == o.records);
      compare_statistics(c, p);
      REQUIRE(get_kinetic_energy_propagation_threshold(Code::Photon) == 4_GeV);
    }
  }
} // namespace
int main() {
  try {
    logging::set_level(logging::level::warn);
    static_assert(!EnableStoppedPositronAnnihilation);
    static_assert(!production_cut_reference::EnableStoppedPositronAnnihilation);
    auto& manager = RNGManager<>::getInstance();
    manager.registerRandomStream("cascade");
    manager.setSeed(291);
    auto before = manager.getRandomStream("cascade");
    fixture(false);
    fixture(true);
    other_constructors();
    REQUIRE(manager.getRandomStream("cascade")() == before());
    std::cout << "CHECKS=" << checks << '\n';
    return 0;
  } catch (std::exception const& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}

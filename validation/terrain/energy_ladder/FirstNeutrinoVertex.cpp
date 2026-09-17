// Screening diagnostic: transport only the primary neutrino to its first natural
// interaction, using the production scene, scalar stepper and CC+NC generator.
// Daughters are recorded but NEVER transported. This is not a shower simulation.
#include <corsika/framework/process/ProcessSequence.hpp>
#include <corsika/framework/core/ScalarCascadeStepper.hpp>
#include <corsika/modules/neutrino/MountainNeutrinoInteraction.hpp>
#include <corsika/modules/neutrino/TransportedLeptons.hpp>
#include <corsika/modules/transport/InterfaceTracking.hpp>
#include <corsika/setup/SetupStack.hpp>
#include <corsika/setup/SetupC7trackedParticles.hpp>
#include <applications/detail/mountain/TerrainScene.hpp>
#include <chrono>
#include <fstream>
#include <iostream>

using namespace corsika;
using namespace corsika::units::si;
using VertexStack = setup::HybridStack<corsika::terrain::Environment>;

int main(int argc, char** argv) {
  if (argc != 3) { std::cerr << "usage: first_vertex input.yaml output.yaml\n"; return 2; }
  try {
    logging::set_level(logging::level::warn);
    auto config = YAML::LoadFile(argv[1]);
    applications::terrain::Scene scene(config["scene"].as<std::string>());
    terrain::Environment env;
    auto cs = env.getCoordinateSystem();
    auto mesh = scene.build(env);
    interfaces::MeshInterfaceTracking tracking(*mesh.mesh);
    auto& rng = RNGManager<>::getInstance();
    for (auto name : {"cascade", "qgsjet", "sophia", "pythia", "fluka", "proposal", "thinning", "tauola"})
      rng.registerRandomStream(name);
    YAML::Node output;
    output["scope"] = "first natural primary-neutrino vertex only; no tau transport or shower";
    output["input"] = config;
    for (auto c : config["cases"]) {
      auto start = std::chrono::steady_clock::now();
      auto seed = c["seed"].as<unsigned long>();
      auto energy = c["energy_GeV"].as<double>();
      auto pos = c["position_m"].as<std::vector<double>>();
      auto direction = c["direction"].as<std::vector<double>>();
      if (pos.size()!=3 || direction.size()!=3) throw std::runtime_error("invalid vectors");
      Point injection(cs, pos[0]*1_m, pos[1]*1_m, pos[2]*1_m);
      DirectionVector dir(cs, {direction[0], direction[1], direction[2]});
      dir = dir.normalized();
      neutrino::MountainNeutrinoInteraction::validatePrimary(Code::NuTau, energy);
      rng.setSeed(seed);
      // RNGManager::setSeed changes the key only, retaining counters/cache.
      // Recreate the streams in-place to match independent production processes.
      unsigned stream = 0;
      for (auto name : {"cascade", "qgsjet", "sophia", "pythia", "fluka", "proposal", "thinning", "tauola"})
        rng.getRandomStream(name) = default_prng_type(seed, stream++);
      // Fresh generator for each seed: no beam/PDF/event state leaks across cases.
      neutrino::MountainNeutrinoInteraction neutrinos(
          neutrino::withTransportedTaus(setup::C7trackedParticles),
          neutrino::SamplingMode::NaturalCC, neutrino::InteractionChannels::CCAndNC);
      NullModel null;
      auto sequence = make_sequence(neutrinos, null);
      VertexStack stack;
      stack.addParticle(std::make_tuple(Code::NuTau, energy*1_GeV, dir, injection, 0_s));
      ScalarCascadeStepper stepper(env, tracking, sequence, stack);
      stepper.setNodes();
      YAML::Node event;
      event["seed"] = seed;
      event["energy_GeV"] = energy;
      unsigned steps = 0;
      double flight = 0.;
      double maxFlight = c["max_primary_flight_m"].as<double>();
      for (; steps < 32 && stack.getSize() && neutrinos.records().empty() && flight < maxFlight;) {
        auto particle = stack.begin();
        if (particle.getPID()!=Code::NuTau || particle.getGeneration()!=0)
          throw std::runtime_error("attempted to transport a secondary during first-vertex scan");
        stepper.advance(particle);
        ++steps;
        if (neutrinos.records().empty() && stack.getSize())
          flight = (stack.begin().getPosition()-injection).getNorm()/1_m;
      }
      if (steps==32 && neutrinos.records().empty() && flight<maxFlight && stack.getSize())
        throw std::runtime_error("primary step limit reached before vertex/exit");
      event["primary_steps"] = steps;
      event["neutrino"] = neutrinos.summary();
      event["vertex_found"] = !neutrinos.records().empty();
      if (!neutrinos.records().empty()) {
        auto const& r = neutrinos.records().front();
        Point vertex(cs,r.xM*1_m,r.yM*1_m,r.zM*1_m);
        double distance = (vertex-injection).getNorm()/1_m;
        event["vertex_in_requested_interval"] = distance <= maxFlight;
        event["vertex_in_rock"] = mesh.mesh->inside(vertex)==terrain::EInside::kInside;
        event["primary_flight_m"] = distance;
        event["current"] = neutrino::currentName(r.vertex.current);
        event["tau_energy_GeV"] = std::abs(r.vertex.selectedOutgoingLeptonPdg)==15 ? r.vertex.outgoingLeptonP4GeV[0] : 0.;
        event["hadronic_energy_GeV"] = r.vertex.hadronicFinalP4GeV[0];
        event["untransported_secondaries"] = stack.getSize();
      }
      event["elapsed_s"] = std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
      output["events"].push_back(event);
      std::ofstream out(argv[2]); out << output << '\n';
      if (!out) throw std::runtime_error("cannot save first-vertex output");
      std::cerr << "[first_vertex] seed=" << seed << " steps=" << steps
                << " elapsed_s=" << event["elapsed_s"].as<double>() << '\n';
    }
  } catch (std::exception const& e) { std::cerr << e.what() << '\n'; return 1; }
}

/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include "KokkosCpuTransportAlignmentDriver.hpp"

#include <corsika/accelerator/em/KokkosRuntime.hpp>
#include <corsika/accelerator/em/common/tables/ProposalNativeTableExporter.hpp>
#include <corsika/framework/process/ProcessSequence.hpp>
#include <corsika/media/CORSIKA7Atmospheres.hpp>
#include <corsika/modules/ObservationPlane.hpp>
#include <corsika/modules/ParticleCut.hpp>
#include <corsika/modules/proposal/ContinuousProcess.hpp>
#include <corsika/setup/SetupTrajectory.hpp>

#include <SetupTestEnvironment.hpp>
#include <SetupTestStack.hpp>
#include <SetupTestTrajectory.hpp>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

namespace {
  using namespace corsika;
  using namespace corsika::gpu::em;
  using namespace corsika::gpu::em::tables;
  using namespace corsika::accelerator::em::testing;

  void require(bool value, char const* message) {
    if (!value) throw std::runtime_error(message);
  }
  bool close(double a, double b, double relative = 1.e-10, double absolute = 1.e-12) {
    return std::isfinite(a) && std::isfinite(b) &&
           std::abs(a - b) <= absolute + relative * std::max(std::abs(a), std::abs(b));
  }

  struct PointWriter : WriterOff {
    double deposited_GeV{};
    void write(Point const&, Code, HEPEnergyType energy) {
      deposited_GeV += energy / 1_GeV;
    }
  };

  struct CpuExpectation {
    Code code{};
    double grammage{};
    bool cut{};
  };
}

int main() {
  try {
    using namespace corsika;
    using namespace corsika::gpu::em;
    using namespace corsika::gpu::em::tables;
    using namespace corsika::accelerator::em::testing;
    logging::set_level(logging::level::warn);
    PROPOSAL::Logging::SetGlobalLoglevel(spdlog::level::warn);
    RNGManager<>::getInstance().registerRandomStream("proposal");
    accelerator::em::KokkosRuntimeConfig runtime_config;
    runtime_config.threads = 2;
    accelerator::em::KokkosRuntime runtime(runtime_config);

    // Real CORSIKA medium and scalar ContinuousProcess; no synthetic dE/dX
    // is used for this oracle. Reuse the normal air-composition PROPOSAL cache.
    DummyEnvironment env;
    auto const& cs = env.getCoordinateSystem();
    auto world = DummyEnvironment::createNode<Sphere>(Point{cs, 0_m, 0_m, 0_m}, 1e9_m);
    using Model = MediumPropertyModel<
        UniformMagneticField<HomogeneousMedium<DummyEnvironmentInterface>>>;
    world->setModelProperties<Model>(
        Medium::AirDry1Atm, MagneticFieldVector(cs, 0_T, 0_T, 0_T),
        1_kg / (1_m * 1_m * 1_m), standardAirComposition);
    auto* node = world.get();
    env.getUniverse()->addChild(std::move(world));
    for (auto code : proposal::tracked) set_energy_production_threshold(code, 0.5_MeV);
    proposal::ContinuousProcess<> continuous(env);
    auto continuous_views = continuous.nativeCalculatorViews();
    std::vector<std::unique_ptr<PROPOSAL::Interaction>> owned_interactions;
    std::vector<proposal::NativeInteractionCalculatorView> interaction_views;
    for (auto const& view : continuous_views) {
      if (view.projectile != Code::Electron && view.projectile != Code::Positron &&
          view.projectile != Code::MuMinus && view.projectile != Code::MuPlus)
        continue;
      auto medium = *view.medium;
      auto cross = proposal::make_cross_sections(
          view.projectile, medium, view.stochastic_energy_cut, true);
      owned_interactions.push_back(PROPOSAL::make_interaction(cross, true, true));
      interaction_views.push_back(
          {view.projectile, view.medium_hash, view.medium, owned_interactions.back().get(),
           nullptr, nullptr, view.stochastic_energy_cut, view.particle_mass_MeV});
    }
    auto source = exportProposalNativeTables(interaction_views, continuous_views);

    std::vector<CpuTransportAlignmentInput> inputs;
    std::vector<CpuExpectation> expected;
    for (auto code : {Code::Electron, Code::Positron, Code::MuMinus, Code::MuPlus}) {
      auto const muon = is_muon(code);
      auto const mass_MeV = get_mass(code) / 1_MeV;
      for (auto scale : {1., 10., 100.}) {
        auto const cut_MeV = (muon ? 300. : 0.5) * scale;
        auto const crossover = (mass_MeV + cut_MeV) * 0.9999 / 0.9;
        for (auto energy : {mass_MeV + cut_MeV * (1. - 1.e-5),
                            mass_MeV + cut_MeV * (1. + 1.e-5),
                            crossover * (1. - 1.e-5), crossover * (1. + 1.e-5),
                            mass_MeV + 2. * cut_MeV, mass_MeV + 100. * cut_MeV}) {
          ParticleCut<PointWriter> cut(
              (muon ? 0.5 : cut_MeV) * 1_MeV, 0.5_MeV, 0.3_GeV,
              (muon ? cut_MeV : 300.) * 1_MeV, 0.3_GeV, true);
          test::Stack stack;
          auto particle = stack.addParticle(std::make_tuple(
              code, energy * 1_MeV - get_mass(code),
              DirectionVector(cs, {0., 0., -1.}), Point(cs, {0_m, 0_m, 6371100_m}),
              0_ns));
          particle.setNode(node);
          StraightTrajectory track(Line(particle.getPosition(),
              VelocityVector(cs, 0_m / second, 0_m / second, -constants::c)), 1_ns);
          Step step(particle, track);
          auto const scalar_cut = cut.doContinuous(step, false) == ProcessReturn::ParticleAbsorbed;
          CpuTransportAlignmentInput input{};
          input.interaction.status = EmInteractionStatus::NoDiscreteInteraction;
          input.interaction.particle.pid = static_cast<std::int32_t>(get_PDG(code));
          input.interaction.particle.energy_GeV = particle.getEnergy() / 1_GeV;
          input.interaction.particle.direction[2] = -1.;
          input.interaction.particle.position_m[2] = 6371100.;
          input.interaction.particle.weight = 1.;
          input.em_cut_MeV = muon ? 0.5 : cut_MeV;
          input.muon_cut_MeV = muon ? cut_MeV : 300.;
          auto const max_distance = scalar_cut ? 0_m : continuous.getMaxStepLength(particle, track);
          // Homogeneous density 1 kg/m^3 = 0.1 (g/cm^2)/m.
          inputs.push_back(input);
          expected.push_back({code, max_distance / 1_m * 0.1, scalar_cut});
        }
      }
    }
    auto output = runCpuTransportAlignmentOnDevice(source, inputs);
    double maximum_relative_grammage = 0.;
    for (std::size_t i = 0; i < output.size(); ++i) {
      auto const& result = output[i];
      require(result.preparation.fallback_flag == 0, "continuous preparation fallback");
      require(result.preparation.particle_cut == expected[i].cut, "CPU/Kokkos entry cut differs");
      require((result.selection_status == EmInteractionStatus::ParticleCut) == expected[i].cut,
              "CPU/Kokkos selector cut differs");
      require(close(result.preparation.mass_MeV, get_mass(expected[i].code) / 1_MeV, 1.e-14),
              "transport mass differs from scalar stack mass");
      if (!expected[i].cut) {
        auto const actual = result.preparation.maximum_grammage_g_per_cm2;
        maximum_relative_grammage = std::max(maximum_relative_grammage,
            std::abs(actual - expected[i].grammage) / std::max(expected[i].grammage, 1.e-300));
        if (!close(actual, expected[i].grammage)) {
          std::cerr << "grammage case " << i << " scalar=" << expected[i].grammage
                    << " device=" << actual << '\n';
          throw std::runtime_error("real scalar getMaxStepLength differs from Kokkos");
        }
      }
    }

    // Deliberately place the observation plane between the kinetic threshold
    // and the 0.9999 total-energy endpoint. This is the formerly missing
    // simultaneous observation + ParticleCut case, for all four lepton PIDs.
    inputs.clear();
    for (auto code : {Code::Electron, Code::Positron, Code::MuMinus, Code::MuPlus}) {
      auto const mass = get_mass(code) / 1_MeV;
      auto const cut = is_muon(code) ? 300. : 0.5;
      auto const start_energy = mass + cut * (1. + 1.e-5);
      auto const final_energy = mass + cut * (1. - 2.e-5);
      // Only choose the input geometry here; the expected loss below is
      // independently calculated by the real scalar continuous module.
      auto const host = makeProposalNativeHostView(source);
      auto const pid = static_cast<std::int32_t>(get_PDG(code));
      auto const initial_range = queryProposalNativeRange(host, pid, start_energy);
      auto const final_range = queryProposalNativeRange(host, pid, final_energy);
      require(initial_range.status == NativeQueryStatus::Success &&
                  final_range.status == NativeQueryStatus::Success,
              "fixture range lookup failed");
      auto const distance = (initial_range.value - final_range.value) / 0.1;
      for (auto multiplier : {0.01, 1., 20.}) {
        CpuTransportAlignmentInput input{};
        input.propagate = true;
        input.interaction.status = EmInteractionStatus::NoDiscreteInteraction;
        auto& state = input.interaction.particle;
        state.pid = static_cast<std::int32_t>(get_PDG(code));
        state.energy_GeV = start_energy / 1000.;
        state.position_m[2] = 6371100. + multiplier * distance;
        state.direction[2] = -1.;
        state.weight = 2.;
        auto& environment = input.environment;
        environment.number_of_layers = 1;
        environment.atmosphere_layers[0].inner_radius_m = 6371000.;
        environment.atmosphere_layers[0].outer_radius_m = 6471000.;
        environment.atmosphere_layers[0].density_model = DensityModel::Homogeneous;
        environment.atmosphere_layers[0].density_parameter_a = 0.001;
        environment.observation_plane_point_m[2] = 6371100.;
        environment.observation_plane_normal[2] = 1.;
        environment.observation_radius_m = 6371100.;
        inputs.push_back(input);
      }
    }
    output = runCpuTransportAlignmentOnDevice(source, inputs);
    std::size_t dual_hits = 0;
    for (std::size_t i = 0; i < output.size(); ++i) {
      auto const& result = output[i];
      require(result.transport_status == 0, "terminal transport falls back");
      auto const& record = result.transport;
      auto const code = convert_from_PDG(static_cast<PDGCode>(record.start.pid));
      test::Stack stack;
      auto particle = stack.addParticle(std::make_tuple(
          code, record.start.energy_GeV * 1_GeV - get_mass(code),
          DirectionVector(cs, {0., 0., -1.}),
          Point(cs, {0_m, 0_m, record.start.position_m[2] * 1_m}), 0_ns));
      particle.setNode(node);
      particle.setWeight(record.start.weight);
      auto const dt = record.end.time_s * 1_s;
      auto const velocity_z = (record.end.position_m[2] - record.start.position_m[2]) * 1_m / dt;
      auto track = setup::testing::make_track<setup::Trajectory>(
          Line(particle.getPosition(), VelocityVector(cs, 0_m / second, 0_m / second, velocity_z)), dt);
      Step step(particle, track);
      // Run the real scalar continuous process on exactly the same chord.
      continuous.doContinuous(step, true);
      require(close(step.getEkinPost() / 1_GeV + get_mass(code) / 1_GeV,
                    record.end.energy_GeV), "real scalar continuous endpoint differs");
      ParticleCut<PointWriter> cut(0.5_MeV, 0.5_MeV, 0.3_GeV, 0.3_GeV, 0.3_GeV, true);
      ObservationPlane<setup::Tracking, WriterOff> observation(
          Plane(Point(cs, {0_m, 0_m, 6371100_m}), DirectionVector(cs, {0., 0., 1.})),
          DirectionVector(cs, {1., 0., 0.}));
      auto sequence = make_sequence(observation, cut);
      auto const observed = record.limit == LeptonTransportLimit::ObservationSurface ||
                            record.observation_surface_reached_before_cut != 0;
      sequence.doContinuous(step, observed ? ContinuousProcessIndex(&observation)
                                         : ContinuousProcessIndex(nullptr));
      require(observation.statistics().particles == (observed ? 1u : 0u),
              "scalar observation count differs");
      require(cut.statistics().particles == (record.limit == LeptonTransportLimit::ParticleCut ? 1u : 0u),
              "scalar ParticleCut count differs");
      require(close(cut.deposited_GeV, record.cut_deposited_energy_GeV * record.start.weight),
              "scalar point cut deposit differs");
      if (observed && cut.statistics().particles == 1) ++dual_hits;
    }
    require(dual_hits == 4, "did not exercise all four simultaneous observation/cut cases");

    // The CPU adapter passes PROPOSAL kinetic energy into the real stack;
    // compare that stack state with device materialization, not raw PROPOSAL
    // total energy (the two projects use slightly different mass constants).
    struct ExpectedChild { int pid; double energy_GeV; double weight; };
    std::vector<CpuSecondaryAlignmentInput> secondary_inputs;
    std::vector<std::vector<ExpectedChild>> secondary_expected;
    auto const native_e = proposal::particle.at(Code::Electron).mass / 1000.;
    auto const native_mu = proposal::particle.at(Code::MuMinus).mass / 1000.;
    auto add_secondary_case = [&](int pid, int process, double split,
                                  std::vector<ExpectedChild> const& children) {
      for (auto mask : {1u, 2u, 3u}) {
        if (children.size() != 2 && mask != 3u) continue;
        CpuSecondaryAlignmentInput item{};
        item.interaction.status = EmInteractionStatus::Selected;
        item.interaction.process_id = process;
        item.interaction.energy_fraction = process == AnnihilationProcessId ? 1. : 0.1;
        item.interaction.particle_mass_GeV = std::abs(pid) == 13 ? native_mu : native_e;
        item.interaction.particle.pid = pid;
        item.interaction.particle.energy_GeV = 10.;
        item.interaction.particle.weight = 2.;
        item.interaction.particle.direction[2] = 1.;
        item.split = split;
        item.keep_mask = mask;
        secondary_inputs.push_back(item);
        std::vector<ExpectedChild> kept;
        for (std::size_t j = 0; j < children.size(); ++j)
          if (children.size() != 2 || (mask & (1u << j))) kept.push_back(children[j]);
        secondary_expected.push_back(kept);
      }
    };
    add_secondary_case(22, PhotonPairProcessId, 0.4,
                       {{11, 4., 3.}, {-11, 6., 5.}});
    add_secondary_case(22, ComptonProcessId, 0.1,
                       {{22, 9., 3.}, {11, 1. + native_e, 5.}});
    add_secondary_case(22, PhotoelectricProcessId, 0.999,
                       {{11, 9.99 + native_e, 2.}});
    for (int pid : {11, -11}) {
      add_secondary_case(pid, BremsProcessId, 0.4,
                         {{pid, 9., 3.}, {22, 1., 5.}});
      add_secondary_case(pid, IonizationProcessId, 0.4,
                         {{pid, 9., 3.}, {11, 1. + native_e, 5.}});
      add_secondary_case(pid, ElectronPairProcessId, 0.3,
                         {{pid, 9., 2.}, {11, 0.65, 2.}, {-11, 0.35, 2.}});
    }
    for (int pid : {13, -13})
      add_secondary_case(pid, IonizationProcessId, 0.4,
                         {{pid, 9., 3.}, {11, 1. + native_e, 5.}});
    add_secondary_case(-11, AnnihilationProcessId, 0.3,
                       {{22, (10. + native_e) * 0.7, 3.},
                        {22, (10. + native_e) * 0.3, 5.}});
    auto const secondary_outputs = runCpuSecondaryAlignmentOnDevice(secondary_inputs);
    for (std::size_t i = 0; i < secondary_outputs.size(); ++i) {
      auto const& result = secondary_outputs[i];
      require(result.error == 0, "secondary materialization failed");
      require(result.count == secondary_expected[i].size(), "secondary count mismatch");
      test::Stack stack;
      auto primary = stack.addParticle(std::make_tuple(
          Code::Photon, 10_GeV, DirectionVector(cs, {0., 0., 1.}),
          Point(cs, {0_m, 0_m, 0_m}), 0_ns));
      test::StackView view(primary);
      double mass_correction = 0.;
      for (auto const& child : secondary_expected[i]) {
        auto const code = convert_from_PDG(static_cast<PDGCode>(child.pid));
        auto const native_mass = PROPOSAL::ParticleDef::GetParticleDefForType(child.pid).mass * 1_MeV;
        // Exactly the scalar InteractionModel::doSpecifiedInteraction adapter.
        view.addSecondary(std::make_tuple(
            code, child.energy_GeV * 1_GeV - native_mass,
            DirectionVector(cs, {0., 0., 1.})));
        mass_correction += child.weight * (get_mass(code) - native_mass) / 1_GeV;
      }
      std::size_t j = 0;
      for (auto const& child : view) {
        require(result.children[j].pid == static_cast<int>(get_PDG(child.getPID())),
                "secondary species differs from CPU stack");
        require(close(result.children[j].energy_GeV, child.getEnergy() / 1_GeV, 1.e-13),
                "secondary total energy differs from CPU stack");
        require(close(result.children[j].energy_GeV - get_mass(child.getPID()) / 1_GeV,
                      child.getKineticEnergy() / 1_GeV, 1.e-13),
                "secondary kinetic energy differs from CPU stack");
        require(result.children[j].weight == secondary_expected[i][j].weight,
                "secondary weight changed during conversion");
        ++j;
      }
      require(j == result.count, "CPU stack enumeration differs");
      require(close(result.weighted_mass_correction_GeV, mass_correction, 1.e-8, 1.e-18),
              "signed mass-convention closure source differs");
    }
    auto const random_outputs = runCpuRandomDomainAlignmentOnDevice(source, 65536);
    std::size_t legacy_equal = 0, new_equal = 0;
    long double x = 0., y = 0., xx = 0., yy = 0., xy = 0.;
    for (auto const& result : random_outputs) {
      require(result.selection_observed, "native selector did not record its random draw");
      legacy_equal += result.first_moliere_uniform == result.legacy_proposal_selection_uniform;
      new_equal += result.first_moliere_uniform == result.proposal_selection_uniform;
      auto a = static_cast<long double>(result.first_moliere_uniform);
      auto b = static_cast<long double>(result.proposal_selection_uniform);
      x += a; y += b; xx += a * a; yy += b * b; xy += a * b;
    }
    auto const n = static_cast<long double>(random_outputs.size());
    auto const correlation = (xy - x * y / n) / std::sqrt((xx - x * x / n) * (yy - y * y / n));
    require(legacy_equal == random_outputs.size(), "legacy RNG collision reproduction failed");
    require(new_equal == 0, "Moliere and native process selection still reuse a random draw");
    require(std::abs(correlation) < 0.03L, "new independent RNG domains unexpectedly correlated");
    std::cout << std::setprecision(17)
              << "backend=" << runtime.info().backend << " scalar_step_cases=" << expected.size()
              << " terminal_cases=" << output.size() << " dual_observation_cut=" << dual_hits
              << " secondary_adapter_cases=" << secondary_outputs.size()
              << " rng_samples=" << random_outputs.size()
              << " legacy_equal=" << legacy_equal << " new_equal=" << new_equal
              << " new_domain_correlation=" << correlation
              << " max_relative_grammage=" << maximum_relative_grammage << " PASS\n";
  } catch (std::exception const& error) {
    std::cerr << "CPU/Kokkos transport alignment FAIL: " << error.what() << '\n';
    return 1;
  }
}

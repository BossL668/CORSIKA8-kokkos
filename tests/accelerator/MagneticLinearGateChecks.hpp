/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */
#pragma once

#include "KokkosCpuTransportAlignmentDriver.hpp"
#include <corsika/modules/tracking/TrackingLeapFrogCurved.hpp>
#include <corsika/media/CORSIKA7Atmospheres.hpp>
#include <SetupTestEnvironment.hpp>
#include <SetupTestStack.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace magnetic_linear_gate_test {
  // Host oracle uses actual CPU getTrack(), including its zero-field linear
  // trajectory policy, NOT a second copy of the portable magnetic polynomial.
  inline void run(corsika::gpu::em::tables::ProposalNativeTableSet const& tables) {
    using namespace corsika;
    namespace em = gpu::em;
    using accelerator::em::testing::CpuTransportAlignmentInput;
    struct Expected {
      std::array<double, 3> position, direction, plane_normal;
      double time_s, distance_m, chord_m;
      bool linear, external;
    };
    std::vector<CpuTransportAlignmentInput> inputs;
    std::vector<Expected> expected;
    auto const cs = get_root_CoordinateSystem();
    auto const unit_momentum = constants::c *
        convert_HEP_to_SI<MassType::dimension_type>(1_GeV);
    double const scalar_factor = constants::e * (1_T) / unit_momentum * 1_m;
    for (auto code : {Code::Electron, Code::Positron, Code::MuMinus, Code::MuPlus}) {
      double const mass = get_mass(code) / 1_GeV;
      double const threshold_p = scalar_factor * 5.e-5 * 1.e9;
      struct Case { double energy, field, length; std::array<double, 3> direction; bool linear; };
      std::vector<Case> cases{
          {std::hypot(.999 * threshold_p, mass), 5.e-5, 1000., {0., 0., -1.}, false},
          {std::hypot(1.001 * threshold_p, mass), 5.e-5, 1000., {0., 0., -1.}, true},
          {1.e5, 5.e-5, 1000., {0., 0., -1.}, true},
          {1., 0., 1., {0., 0., -1.}, true},
          {1., 5.e-5, 1., {0., 1., 0.}, true},
          {1., 5.e-5, 1., {0., 0., -1.}, false}};
      // Use low-energy e+/- so both sides of p_perp=1 eV can be resolved
      // without cancellation in the existing B^2-(d.B)^2 policy calculation.
      if (!is_muon(code)) {
        double const energy = .0011;
        double const p = std::sqrt((energy - mass) * (energy + mass));
        for (double transverse_p : {.5e-9, 2.e-9}) {
          double const dz = transverse_p / p;
          cases.push_back({energy, 5.e-5, 1.e-6,
                           {0., std::sqrt(1. - dz * dz), -dz}, transverse_p < 1.e-9});
        }
      }
      for (auto const& test : cases) {
        DummyEnvironment env;
        auto world = DummyEnvironment::createNode<Sphere>(Point(cs, 0_m, 0_m, 0_m), 1.e9_m);
        using Model = MediumPropertyModel<UniformMagneticField<HomogeneousMedium<DummyEnvironmentInterface>>>;
        world->setModelProperties<Model>(Medium::AirDry1Atm,
            MagneticFieldVector(cs, 0_T, test.field * 1_T, 0_T),
            1_kg / (1_m * 1_m * 1_m), standardAirComposition);
        auto* node = world.get();
        env.getUniverse()->addChild(std::move(world));
        test::Stack stack;
        auto particle = stack.addParticle(std::make_tuple(code,
            test.energy * 1_GeV - get_mass(code),
            DirectionVector(cs, {test.direction[0], test.direction[1], test.direction[2]}),
            Point(cs, 0_m, 0_m, 6372100_m), 0_ns));
        particle.setNode(node);
        tracking_leapfrog_curved::Tracking tracking;
        auto [track, next_node] = tracking.getTrack(particle);
        (void)next_node;
        auto const dt = test.length * 1_m / particle.getVelocity().getNorm();
        if (dt > track.getDuration()) throw std::runtime_error("magnetic gate fixture exceeds CPU step");
        track.setDuration(dt);
        auto const position = track.getPosition(1.);
        auto const direction = track.getDirection(1.);
        auto const chord = (position - particle.getPosition()).getNorm() / 1_m;
        for (bool external : {false, true}) {
          CpuTransportAlignmentInput input{};
          input.propagate = true;
          input.interaction.status = em::EmInteractionStatus::NoDiscreteInteraction;
          auto& start = input.interaction.particle;
          start.pid = static_cast<int>(get_PDG(code));
          start.energy_GeV = particle.getEnergy() / 1_GeV;
          start.position_m[2] = 6372100.;
          start.weight = 1.;
          auto& environment = input.environment;
          environment.number_of_layers = 1;
          auto& layer = environment.atmosphere_layers[0];
          layer.inner_radius_m = 6371000.; layer.outer_radius_m = 6471000.;
          layer.density_model = em::DensityModel::Homogeneous;
          layer.density_parameter_a = .001;
          environment.magnetic_field_T[1] = test.field;
          environment.observation_radius_m = 6371100.;
          std::array<double, 3> point{position.getX(cs) / 1_m,
              position.getY(cs) / 1_m, position.getZ(cs) / 1_m};
          std::array<double, 3> end_dir{direction.getX(cs), direction.getY(cs), direction.getZ(cs)};
          // Oblique plane exposes a spurious transverse endpoint shift.
          auto normal = test.direction[1] == 0. ? std::array<double, 3>{.6, 0., .8}
                                               : std::array<double, 3>{0., -1., 0.};
          for (int a = 0; a < 3; ++a) {
            start.direction[a] = test.direction[a];
            environment.observation_plane_point_m[a] = point[a];
            environment.observation_plane_normal[a] = normal[a];
          }
          input.external = {test.length, external, external, false};
          inputs.push_back(input);
          expected.push_back({point, end_dir, normal, dt / 1_s, test.length, chord,
                              test.linear, external});
        }
      }
    }
    auto const outputs = accelerator::em::testing::runCpuTransportAlignmentOnDevice(tables, inputs);
    double max_position = 0., max_direction = 0., max_plane = 0., max_time = 0., max_grammage = 0.;
    std::size_t failures = 0, linear_count = 0;
    for (std::size_t i = 0; i < outputs.size(); ++i) {
      auto const& out = outputs[i]; auto const& want = expected[i]; auto const& step = out.transport;
      double pos_error = 0., dir_error = 0., plane_error = 0.;
      for (int a = 0; a < 3; ++a) {
        pos_error = std::max(pos_error, std::abs(step.end.position_m[a] - want.position[a]));
        dir_error = std::max(dir_error, std::abs(step.end.direction[a] - want.direction[a]));
        plane_error += (step.end.position_m[a] - want.position[a]) * want.plane_normal[a];
      }
      auto const time_error = std::abs(step.end.time_s / want.time_s - 1.);
      auto const grammage_error = std::abs(step.traversed_grammage_g_per_cm2 / (.1 * want.chord_m) - 1.);
      bool const linear_ok = !want.linear || (step.magnetic_bending_applied == 0 &&
          step.magnetic_bend_parameter == 0. &&
          step.end.direction[0] == step.start.direction[0]);
      bool const curved_ok = want.linear || step.magnetic_bending_applied == 1;
      bool const ok = out.transport_status == 0 && linear_ok && curved_ok &&
          step.limit == (want.external ? em::LeptonTransportLimit::MaterialBoundary
                                     : em::LeptonTransportLimit::ObservationSurface) &&
          pos_error < 2.e-9 && dir_error < 2.e-12 && std::abs(plane_error) < 2.e-9 &&
          time_error < 2.e-8 && grammage_error < 2.e-8;
      if (!ok) {
        std::cerr << "magnetic gate case=" << i << " linear=" << want.linear
            << " external=" << want.external << " status=" << out.transport_status
            << " fallback=" << static_cast<int>(out.fallback.reason)
            << " bend=" << step.magnetic_bending_applied << " position=" << pos_error
            << " direction=" << dir_error << " time=" << time_error
            << " grammage=" << grammage_error << '\n';
      }
      failures += !ok; linear_count += want.linear;
      max_position = std::max(max_position, pos_error); max_direction = std::max(max_direction, dir_error);
      max_plane = std::max(max_plane, std::abs(plane_error)); max_time = std::max(max_time, time_error);
      max_grammage = std::max(max_grammage, grammage_error);
    }
    std::cout << std::setprecision(17) << "magnetic_gate_cases=" << outputs.size()
        << " linear=" << linear_count << " failures=" << failures
        << " max_position_m=" << max_position << " max_direction=" << max_direction
        << " max_plane_residual_m=" << max_plane << " max_time_relative=" << max_time
        << " max_grammage_relative=" << max_grammage << '\n';
    if (failures) throw std::runtime_error("CPU getTrack/production transport magnetic policy mismatch");
  }
}

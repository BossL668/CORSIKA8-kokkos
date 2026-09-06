/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <utility>

#include <corsika/framework/core/ParticleProperties.hpp>
#include <corsika/framework/core/PhysicalUnits.hpp>
#include <corsika/framework/geometry/Point.hpp>
#include <corsika/framework/geometry/Vector.hpp>
#include <corsika/accelerator/em/common/Types.hpp>
#include <corsika/stack/TransportIdentityStackExtension.hpp>

namespace corsika::gpu::em::router_detail {

  template <typename T, typename = void>
  struct HasGetWeight : std::false_type {};

  template <typename T>
  struct HasGetWeight<
      T, std::void_t<decltype(std::declval<T const&>().getWeight())>>
      : std::true_type {};

  template <typename T, typename = void>
  struct HasSetWeight : std::false_type {};

  template <typename T>
  struct HasSetWeight<
      T, std::void_t<decltype(std::declval<T&>().setWeight(
             std::declval<double>()))>> : std::true_type {};

  template <typename TParticle>
  double getWeightOrOne(TParticle const& particle) {
    if constexpr (HasGetWeight<TParticle>::value) {
      return particle.getWeight();
    } else {
      return 1.;
    }
  }

  template <typename TParticle>
  void setWeightIfSupported(TParticle& particle, double weight) {
    if constexpr (HasSetWeight<TParticle>::value) {
      particle.setWeight(weight);
    } else if (weight != 1.) {
      throw std::runtime_error(
          "GPU particle has a non-unit weight but the CPU stack has no weight field");
    }
  }

  template <typename TParticle>
  EmParticleState toDeviceState(
      TParticle const& particle,
      CoordinateSystemPtr const& coordinate_system,
      transport::HistoryId history_id,
      transport::HistoryId parent_history_id,
      transport::Generation generation,
      transport::StepId step_id) {
    EmParticleState state{};
    state.pid = static_cast<std::int32_t>(particle.getPDG());
    state.energy_GeV = particle.getEnergy() / 1_GeV;

    auto const position =
        particle.getPosition().getCoordinates(coordinate_system);
    auto const direction =
        particle.getDirection().getComponents(coordinate_system);
    for (int axis = 0; axis < 3; ++axis) {
      state.position_m[axis] = position[axis] / meter;
      state.direction[axis] = direction[axis].magnitude();
    }

    state.time_s = particle.getTime() / second;
    state.weight = getWeightOrOne(particle);
    state.history_id = history_id;
    state.parent_history_id = parent_history_id;
    state.generation = generation;
    state.step_id = step_id;
    return state;
  }

  template <typename TStack>
  auto importParticle(
      TStack& stack, EmParticleState const& state,
      CoordinateSystemPtr const& coordinate_system) {
    auto const code =
        convert_from_PDG(static_cast<PDGCode>(state.pid));
    auto const total_energy = state.energy_GeV * 1_GeV;
    auto const mass = get_mass(code);
    if (total_energy < mass) {
      throw std::runtime_error(
          "GPU returned a particle with total energy below its rest mass");
    }

    DirectionVector const direction{
        coordinate_system,
        {state.direction[0], state.direction[1],
         state.direction[2]}};
    Point const position{
        coordinate_system, state.position_m[0] * meter,
        state.position_m[1] * meter,
        state.position_m[2] * meter};

    auto particle = stack.addParticle(std::make_tuple(
        code, total_energy - mass, direction, position,
        state.time_s * second));
    particle.setTransportIdentity(transport::TransportIdentity{
        state.history_id, state.parent_history_id,
        state.generation, state.step_id});
    setWeightIfSupported(particle, state.weight);
    return particle;
  }

} // namespace corsika::gpu::em::router_detail

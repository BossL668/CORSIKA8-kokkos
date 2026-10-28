/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/framework/core/ParticleProperties.hpp>
#include <corsika/framework/core/PhysicalUnits.hpp>
#include <corsika/framework/geometry/Point.hpp>
#include <corsika/framework/geometry/PhysicalGeometry.hpp>
#include <corsika/framework/geometry/Vector.hpp>

#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace corsika::kokkos {

  namespace particle_conversion {

    // Detection traits retain the pinned release's C++17 language level.
    template <typename TParticle, typename = void>
    struct HasWeightGetter : std::false_type {};

    template <typename TParticle>
    struct HasWeightGetter<
        TParticle, std::void_t<decltype(std::declval<TParticle const&>().getWeight())>>
        : std::true_type {};

    template <typename TParticle, typename = void>
    struct HasWeightSetter : std::false_type {};

    template <typename TParticle>
    struct HasWeightSetter<
        TParticle, std::void_t<decltype(std::declval<TParticle&>().setWeight(1.))>>
        : std::true_type {};

  } // namespace particle_conversion

  template <typename TParticle>
  void export_particle_values(TParticle const& particle,
                              CoordinateSystemPtr const& coordinateSystem,
                              ParticleState& state) {
    if (!coordinateSystem) {
      throw std::invalid_argument("Particle value export requires a coordinate system");
    }
    auto const position = particle.getPosition().getCoordinates(coordinateSystem);
    auto const direction = particle.getDirection().getComponents(coordinateSystem);
    auto const pid = static_cast<std::int32_t>(particle.getPDG());
    double const energy = particle.getEnergy() / 1_GeV;
    double const time = particle.getTime() / second;
    double weight = 1.;
    if constexpr (particle_conversion::HasWeightGetter<TParticle>::value) {
      weight = particle.getWeight();
    }

    state.pid_ = pid;
    state.energyGeV_ = energy;
    for (int axis = 0; axis < 3; ++axis) {
      state.positionM_[axis] = position[axis] / meter;
      state.direction_[axis] = direction[axis].magnitude();
    }
    state.timeS_ = time;
    state.weight_ = weight;
  }

  template <typename TParticle>
  void import_particle_values(ParticleState const& state,
                              CoordinateSystemPtr const& coordinateSystem,
                              TParticle& particle) {
    if (!coordinateSystem) {
      throw std::invalid_argument("Particle value import requires a coordinate system");
    }
    // C8's PDG conversion uses abs(int); its minimum cannot be represented positive.
    if (state.pid_ == std::numeric_limits<std::int32_t>::min()) {
      throw std::invalid_argument("Particle value import requires a supported PDG code");
    }
    auto const code = convert_from_PDG(static_cast<PDGCode>(state.pid_));
    if (code == Code::Unknown) {
      throw std::invalid_argument("Particle value import cannot use an unknown particle");
    }
    auto const totalEnergy = state.energyGeV_ * 1_GeV;
    auto const mass = get_mass(code);
    if (totalEnergy < mass) {
      throw std::runtime_error("Particle total energy is below its C8 rest mass");
    }
    if constexpr (!particle_conversion::HasWeightSetter<TParticle>::value) {
      if (state.weight_ != 1.) {
        throw std::runtime_error("A non-unit particle weight requires a weight field");
      }
    }
    DirectionVector const direction{
        coordinateSystem,
        {state.direction_[0], state.direction_[1], state.direction_[2]}};
    Point const position{coordinateSystem, state.positionM_[0] * meter,
                         state.positionM_[1] * meter, state.positionM_[2] * meter};

    particle.setPID(code);
    particle.setKineticEnergy(totalEnergy - mass);
    particle.setDirection(direction);
    particle.setPosition(position);
    particle.setTime(state.timeS_ * second);
    if constexpr (particle_conversion::HasWeightSetter<TParticle>::value) {
      particle.setWeight(state.weight_);
    }
  }

} // namespace corsika::kokkos

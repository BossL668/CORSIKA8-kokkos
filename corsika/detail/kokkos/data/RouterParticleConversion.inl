/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once
namespace corsika::kokkos::particle_transfer {
  template <typename TParticle, typename TModelProperties>
  void export_logical_node(TParticle const& particle, ParticleState& state,
                           EnvironmentTopology<TModelProperties> const& tree) {
    if constexpr (HasEnvironmentNode<TParticle>::value) {
      auto const* native = particle.getNode();
      auto const index = native ? tree.getNodeIndex(*native) : -1;
      if (index < 0)
        throw std::logic_error("CPU logical node is not in the exported environment");
      state.environmentNode_ = index;
    } else
      throw std::logic_error("CPU stack cannot export logical nodes");
  }
  template <typename TParticle, typename TModelProperties>
  void import_logical_node(TParticle& particle, ParticleState const& state,
                           EnvironmentTopology<TModelProperties> const& tree) {
    if constexpr (HasEnvironmentNode<TParticle>::value) {
      auto const node = tree.getNativeNode(state.environmentNode_);
      if (!node)
        throw std::logic_error("device logical node is absent from the host environment");
      particle.setNode(std::addressof(*node));
    } else
      throw std::logic_error("CPU stack cannot import logical nodes");
  }
  template <typename TParticle>
  double get_weight_or_one(TParticle const& particle) {
    if constexpr (HasGetWeight<TParticle>::value) {
      return particle.getWeight();
    } else {
      return 1.;
    }
  }
  template <typename TParticle>
  void set_weight_if_supported(TParticle& particle, double weight) {
    if constexpr (HasSetWeight<TParticle>::value) {
      particle.setWeight(weight);
    } else if (weight != 1.) {
      throw std::runtime_error(
          "GPU particle has a non-unit weight but the CPU stack has no weight field");
    }
  }
  template <typename TParticle, typename TModelProperties>
  ParticleState to_device_state(TParticle const& particle,
                                CoordinateSystemPtr const& coordinateSystem,
                                transport::HistoryId historyId,
                                transport::HistoryId parentHistoryId,
                                transport::Generation generation,
                                transport::StepId stepId,
                                EnvironmentTopology<TModelProperties> const& tree) {
    ParticleState state{};
    state.pid_ = static_cast<std::int32_t>(particle.getPDG());
    state.energyGeV_ = particle.getEnergy() / 1_GeV;

    auto const position = particle.getPosition().getCoordinates(coordinateSystem);
    auto const direction = particle.getDirection().getComponents(coordinateSystem);
    for (int axis = 0; axis < 3; ++axis) {
      state.positionM_[axis] = position[axis] / meter;
      state.direction_[axis] = direction[axis].magnitude();
    }

    state.timeS_ = particle.getTime() / second;
    state.weight_ = get_weight_or_one(particle);
    state.historyId_ = historyId;
    state.parentHistoryId_ = parentHistoryId;
    state.generation_ = generation;
    state.stepId_ = stepId;
    export_logical_node(particle, state, tree);
    return state;
  }
  template <typename TStack, typename TModelProperties>
  auto import_particle(TStack& stack, ParticleState const& state,
                       CoordinateSystemPtr const& coordinateSystem,
                       EnvironmentTopology<TModelProperties> const& tree) {
    auto const code = convert_from_PDG(static_cast<PDGCode>(state.pid_));
    auto const totalEnergy = state.energyGeV_ * 1_GeV;
    auto const mass = get_mass(code);
    if (totalEnergy < mass) {
      throw std::runtime_error(
          "GPU returned a particle with total energy below its rest mass");
    }

    DirectionVector const direction{
        coordinateSystem,
        {state.direction_[0], state.direction_[1], state.direction_[2]}};
    Point const position{coordinateSystem, state.positionM_[0] * meter,
                         state.positionM_[1] * meter, state.positionM_[2] * meter};

    auto particle = stack.addParticle(std::make_tuple(code, totalEnergy - mass, direction,
                                                      position, state.timeS_ * second));
    particle.setTransportIdentity(transport::TransportIdentity{
        state.historyId_, state.parentHistoryId_, state.generation_, state.stepId_});
    set_weight_if_supported(particle, state.weight_);
    import_logical_node(particle, state, tree);
    return particle;
  }
} // namespace corsika::kokkos::particle_transfer

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
#include <corsika/kokkos/data/ParticleState.hpp>
#include <corsika/stack/TransportIdentityStackExtension.hpp>
#include <corsika/kokkos/environment/EnvironmentTopology.hpp>

/** Host-only transfer of complete production particle records.
 * The topology and native environment must outlive the transfer. The topology
 * is mandatory: restore the recorded node, never repeat containment search.
 * Import retains production allocation, rest-mass, weight and identity rules.
 * Compatibility-tagged sub-mass particles must be cut by the caller first.
 */
namespace corsika::kokkos::particle_transfer {

  template <typename T, typename = void>
  struct HasEnvironmentNode : std::false_type {};
  template <typename T>
  struct HasEnvironmentNode<T, std::void_t<decltype(std::declval<T const&>().getNode())>>
      : std::true_type {};

  template <typename TParticle, typename TModelProperties>
  void export_logical_node(TParticle const& particle, ParticleState& state,
                           EnvironmentTopology<TModelProperties> const& tree);

  template <typename TParticle, typename TModelProperties>
  void import_logical_node(TParticle& particle, ParticleState const& state,
                           EnvironmentTopology<TModelProperties> const& tree);

  template <typename T, typename = void>
  struct HasGetWeight : std::false_type {};

  template <typename T>
  struct HasGetWeight<T, std::void_t<decltype(std::declval<T const&>().getWeight())>>
      : std::true_type {};

  template <typename T, typename = void>
  struct HasSetWeight : std::false_type {};

  template <typename T>
  struct HasSetWeight<
      T, std::void_t<decltype(std::declval<T&>().setWeight(std::declval<double>()))>>
      : std::true_type {};

  template <typename TParticle>
  double get_weight_or_one(TParticle const& particle);

  template <typename TParticle>
  void set_weight_if_supported(TParticle& particle, double weight);

  template <typename TParticle, typename TModelProperties>
  ParticleState to_device_state(TParticle const& particle,
                                CoordinateSystemPtr const& coordinateSystem,
                                transport::HistoryId historyId,
                                transport::HistoryId parentHistoryId,
                                transport::Generation generation,
                                transport::StepId stepId,
                                EnvironmentTopology<TModelProperties> const& tree);

  template <typename TStack, typename TModelProperties>
  auto import_particle(TStack& stack, ParticleState const& state,
                       CoordinateSystemPtr const& coordinateSystem,
                       EnvironmentTopology<TModelProperties> const& tree);

} // namespace corsika::kokkos::particle_transfer
#include <corsika/detail/kokkos/data/RouterParticleConversion.inl>

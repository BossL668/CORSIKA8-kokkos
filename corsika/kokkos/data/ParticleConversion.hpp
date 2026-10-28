/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/framework/geometry/CoordinateSystem.hpp>
#include <corsika/kokkos/data/ParticleState.hpp>

namespace corsika::kokkos {

  /**
   * Export particle values to the numerical representation used by transport.
   *
   * This host-only conversion uses C8's coordinate transformations and units:
   * total energy in GeV, position in metres, time in seconds and dimensionless
   * direction components. A particle without a weight getter has unit weight.
   * No direction normalization, cut, random draw or identity allocation occurs.
   *
   * \tparam TParticle C8 particle interface providing getPDG, getEnergy,
   * getPosition, getDirection and getTime; getWeight is optional.
   * \param particle Source particle, not modified by coordinate conversion.
   * \param coordinateSystem Non-null common coordinate system for the output.
   * \param state Existing record whose value fields are replaced. Identity,
   * generation, compatibility tags, medium and logical-node fields are preserved.
   * \pre The particle has a supported PDG code and finite physical values.
   * \throws std::invalid_argument If the common coordinate system is null.
   */
  template <typename TParticle>
  void export_particle_values(TParticle const& particle,
                              CoordinateSystemPtr const& coordinateSystem,
                              ParticleState& state);

  /**
   * Import numerical values into an existing C8 particle, without adding a particle.
   *
   * The record's total energy is converted to kinetic energy using C8's particle
   * mass. Direction components are copied without normalization. This operation
   * neither reads nor changes transport identity, logical-node or medium fields;
   * their mapping belongs to the caller, not to this value conversion.
   *
   * \tparam TParticle C8 particle interface providing setPID, setKineticEnergy,
   * setDirection, setPosition and setTime; setWeight is optional.
   * \param state Numerical input with energy in GeV, position in metres and time
   * in seconds. The record is not modified.
   * \param coordinateSystem Non-null common coordinate system of the record.
   * \param particle Existing destination particle; its stack size is unchanged.
   * \pre Values are finite and the PDG code is supported by C8. Direction is
   * already appropriate for the caller's transport algorithm.
   * \throws std::invalid_argument If the coordinate system is null or the PDG
   * code is unknown. Other unsupported PDG codes retain C8's conversion errors.
   * \throws std::runtime_error If total energy is below the C8 rest mass or a
   * non-unit weight would be lost. These checks precede destination mutation.
   * \note A tagged native muon-pair cut must be handled before calling this
   * ordinary on-shell conversion; compatibility tags do not relax its mass check.
   */
  template <typename TParticle>
  void import_particle_values(ParticleState const& state,
                              CoordinateSystemPtr const& coordinateSystem,
                              TParticle& particle);

} // namespace corsika::kokkos

#include <corsika/detail/kokkos/data/ParticleConversion.inl>

/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <cstdint>
#include <vector>

#include <corsika/gpu/em/BremsLpm.hpp>
#include <corsika/gpu/em/Types.hpp>
#include <corsika/gpu/em/detail/DeviceWorkspace.hpp>

namespace corsika::gpu::em {

  // Draw zero is already used by inverse-CDF loss sampling for BremsProcessId.
  inline constexpr std::uint64_t BremsAzimuthDrawId = 1;
  inline constexpr std::uint64_t BremsLpmDrawId = 2;
  // PROPOSAL's Heitler annihilation final state consumes two random numbers
  // after the direct cross section has selected v=1.
  inline constexpr std::uint64_t AnnihilationRhoDrawId = 1;
  inline constexpr std::uint64_t AnnihilationAzimuthDrawId = 2;
  inline constexpr std::uint64_t IonizationAzimuthDrawId = 1;
  // Default KelnerKokoulinPetrukhinEpairProduction consumes three final-state
  // uniforms. Its direction sampler currently ignores the third value, but
  // the draw remains part of the reproducible/auditable key sequence.
  inline constexpr std::uint64_t EpairRhoDrawId = 1;
  inline constexpr std::uint64_t EpairSignDrawId = 2;
  inline constexpr std::uint64_t EpairDirectionDrawId = 3;
  inline constexpr std::uint64_t EpairLpmDrawId = 4;

  /**
   * Evaluate immutable BremsLPM component terms once with the selected CUDA
   * device's double-precision math implementation.
   */
  BremsLpmPreparedSnapshot prepareBremsLpmSnapshotForCuda(
      BremsLpmSnapshot const& snapshot, int device);

  /**
   * Validation bridge for the GPU implementation of PROPOSAL's default
   * BremsEGS4Approximation followed by CORSIKA's BremsLPM rejection, and
   * HeitlerAnnihilation for positrons, NaivIonization, and the default
   * KelnerKokoulinPetrukhinEpairProduction final state followed by CORSIKA's
   * EpairLPM rejection for electrons and positrons.
   */
  BremsFinalStateBatchResult generateBremsFinalStatesForValidation(
      BremsLpmSnapshot const& lpm_snapshot,
      EmThinningConfig const& thinning,
      std::vector<EmInteractionRecord> const& interactions,
      std::uint64_t random_seed, std::uint64_t shower_id, int device,
      std::uint64_t first_secondary_history_id,
      detail::DeviceWorkspace& workspace);


} // namespace corsika::gpu::em

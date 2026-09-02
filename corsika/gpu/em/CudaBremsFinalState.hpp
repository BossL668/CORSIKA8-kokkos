/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <cstdint>
#include <vector>

#include <corsika/accelerator/em/LeptonFinalStateRandomDomains.hpp>
#include <corsika/gpu/em/BremsLpm.hpp>
#include <corsika/gpu/em/Types.hpp>
#include <corsika/gpu/em/detail/DeviceWorkspace.hpp>

namespace corsika::gpu::em {

  // Source compatibility for code which historically obtained these domains
  // from the native-CUDA header.
  using accelerator::em::AnnihilationAzimuthDrawId;
  using accelerator::em::AnnihilationRhoDrawId;
  using accelerator::em::BremsAzimuthDrawId;
  using accelerator::em::BremsLpmDrawId;
  using accelerator::em::EpairDirectionDrawId;
  using accelerator::em::EpairLpmDrawId;
  using accelerator::em::EpairRhoDrawId;
  using accelerator::em::EpairSignDrawId;
  using accelerator::em::IonizationAzimuthDrawId;

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

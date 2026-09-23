/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <cstdint>

namespace corsika::accelerator::em {

  // Draw zero is used by the inverse-CDF loss sampler.  These domains are
  // shared by native CUDA and Kokkos so execution-space changes cannot alter
  // a particle history's Philox stream.
  inline constexpr std::uint64_t BremsAzimuthDrawId = 1;
  inline constexpr std::uint64_t BremsLpmDrawId = 2;
  inline constexpr std::uint64_t AnnihilationRhoDrawId = 1;
  inline constexpr std::uint64_t AnnihilationAzimuthDrawId = 2;
  inline constexpr std::uint64_t AtRestAnnihilationPolarDrawId = 3;
  inline constexpr std::uint64_t AtRestAnnihilationAzimuthDrawId = 4;
  inline constexpr std::uint64_t IonizationAzimuthDrawId = 1;
  inline constexpr std::uint64_t EpairRhoDrawId = 1;
  inline constexpr std::uint64_t EpairSignDrawId = 2;
  inline constexpr std::uint64_t EpairDirectionDrawId = 3;
  inline constexpr std::uint64_t EpairLpmDrawId = 4;

} // namespace corsika::accelerator::em

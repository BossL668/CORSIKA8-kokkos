/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/framework/core/HostDevice.hpp>

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace corsika::kokkos {

  inline constexpr std::size_t MaxPhotonPairLpmComponents = 16;

  /** One PROPOSAL photon-pair LPM component, without host pointers. */
  struct PhotonPairLpmComponentSnapshot {
    std::uint64_t componentHash_{};
    double nuclearCharge_{};
    double radiationLogConstant_{};
  };

  /** Pointer-free PROPOSAL photon-pair LPM state.
   * Energies are in MeV and the density and length units are explicit in the fields.
   * Member order and initialization retain the production auxiliary-cache layout. */
  struct PhotonPairLpmSnapshot {
    double baselineMassDensityGPerCm3_{};
    double molecularDensityPerCm3_{};
    double sumCharge_{};
    double eLpmMeV_{};
    double classicalElectronRadiusCm_{};
    double fineStructureConstant_{};
    std::uint32_t componentCount_{};
    std::uint32_t reserved_{};
    PhotonPairLpmComponentSnapshot components_[MaxPhotonPairLpmComponents]{};
  };

  /** Outcome of a photon-pair LPM query. */
  enum class PhotonPairLpmStatus : std::uint32_t {
    Success = 0,
    ComponentNotFound = 1,
    InvalidSnapshot = 2,
    InvalidInput = 3,
    NonFiniteResult = 4,
  };

  /** Suppression factor and original query status; no upper probability clamp is applied.
   */
  struct PhotonPairLpmResult {
    PhotonPairLpmStatus status_{PhotonPairLpmStatus::InvalidSnapshot};
    std::uint32_t reserved_{};
    double survivalProbability_{};
  };

  static_assert(std::is_standard_layout_v<PhotonPairLpmSnapshot>);
  static_assert(std::is_trivially_copyable_v<PhotonPairLpmSnapshot>);
  static_assert(std::is_standard_layout_v<PhotonPairLpmResult>);
  static_assert(std::is_trivially_copyable_v<PhotonPairLpmResult>);

  /**
   * Evaluate the production photon-pair LPM suppression formula.
   * energy_MeV is the incident photon total energy; x is the first electron's
   * fraction of the pair energy. The local vertex density is in g/cm^3.
   * Snapshot, input and selected-component checks retain their original order.
   * The returned factor is not clamped to one. This function makes no random draw.
   */
  [[nodiscard]] CORSIKA_HOST_DEVICE inline PhotonPairLpmResult
  photon_pair_lpm_suppression_factor(PhotonPairLpmSnapshot const& snapshot,
                                     std::uint64_t component_hash, double energy_MeV,
                                     double x, double local_mass_density_g_per_cm3);

} // namespace corsika::kokkos

#include <corsika/detail/kokkos/em/models/PhotonPairLpm.inl>

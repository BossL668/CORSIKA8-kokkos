/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/kokkos/ScalarPhysicalConstants.hpp>
#include <corsika/kokkos/data/ParticleState.hpp>
#include <corsika/kokkos/environment/CurvedBoundaryAccess.hpp>
#include <corsika/kokkos/environment/EnvironmentShape.hpp>
#include <corsika/kokkos/environment/EnvironmentTreeView.hpp>
#include <corsika/kokkos/environment/MagneticStep.hpp>
#include <corsika/kokkos/environment/NextEnvironmentBoundary.hpp>

#include <cstdint>

namespace corsika::kokkos {

  /**
   * Numerical outcome of an ordered lepton environment-boundary query.
   *
   * distanceM_ is the leap-frog length parameter, not the endpoint chord.
   * nextNode_ is a logical tree index, not a medium identifier; -1 is outside
   * the tree and -3 is the unassigned default. If failed_ is true, an unsupported
   * shape or incomplete roots made the entire query unusable, including when
   * that candidate did not provide the returned minimum. No particle is moved.
   */
  struct LeptonBoundary {
    double distanceM_{};
    std::int32_t nextNode_{-3};
    bool failed_{};
  };

  /**
   * Construct the production numerical trajectory from a particle and step limit.
   *
   * Converts total energy and rest mass to momentum, speed and SI charge in the
   * original operation order. The caller provides finite E > m > 0, a finite
   * charge number, a unit direction, a consistent field and a validated Success
   * or Linear limit. The supplied gyroradius is retained rather than recomputed.
   * No validation, normalization, model lookup or PID-based mass selection is
   * performed here. Particle, field and tree geometry use one common coordinate
   * system. Tree-view lifetime and memory-accessibility contracts apply.
   *
   * \param tree Exported numerical topology and shapes; no model payload is read.
   * \param particle Starting particle; energyGeV_ is total energy in GeV.
   * \param massGeV Rest mass in GeV/c^2 in the same scalar energy convention.
   * \param chargeNumber Signed charge in units of the elementary charge.
   * \param fieldT Previously selected magnetic field in tesla.
   * \param limit Previously calculated magnetic step limit; Linear selects a
   * straight environment trajectory, and positive-infinity sentinels are retained.
   * \return Fresh per-query access with both diagnostic flags cleared.
   */
  template <typename TTreeView>
  [[nodiscard]] KOKKOS_INLINE_FUNCTION CurvedBoundaryAccess<TTreeView> lepton_access(
      TTreeView const& tree, ParticleState const& particle, double massGeV,
      double chargeNumber, double const (&fieldT)[3],
      MagneticStepLimitResult const& limit);

  /**
   * Select the next environment boundary within the supplied magnetic limit.
   *
   * Uses lepton_access() with its input contracts. particle.environmentNode_
   * must be a valid current tree index, independent of mediumId_. The distance
   * limit is divided by the saved speed before the shared ordered time search,
   * and the selected time is multiplied by that same speed afterwards. These
   * conversions preserve production rounding; no chord length is substituted.
   * Both accumulated access diagnostics are returned through failed_.
   *
   * \param tree Valid numerical environment tree, alive for the query.
   * \param particle Starting particle with an assigned environmentNode_.
   * \param massGeV Rest mass in GeV/c^2.
   * \param chargeNumber Signed charge in elementary-charge units.
   * \param fieldT Previously selected field in tesla in the common system.
   * \param limit Validated magnetic limit with distance and gyroradius in metres.
   * \return Selected distance, next node and accumulated failure status.
   */
  template <typename TTreeView>
  [[nodiscard]] KOKKOS_INLINE_FUNCTION LeptonBoundary
  lepton_boundary(TTreeView const& tree, ParticleState const& particle, double massGeV,
                  double chargeNumber, double const (&fieldT)[3],
                  MagneticStepLimitResult const& limit);

  /**
   * Query an independent observation plane without adding an environment node.
   *
   * Uses the same particle conversion as lepton_access(), but always selects the
   * production curved plane entry even when limit.status_ is Linear. Charge zero
   * still uses the curved accessor's straight fallback. Only a strictly positive
   * entry time is multiplied by the saved speed; other entries return positive
   * infinity. No additional intersection-validity check or distance truncation
   * is introduced. Competition with other step limits remains the caller's job.
   *
   * \param tree Numerical tree view copied into the access; no tree node or model
   * is read for this separate plane query, so an empty view is permitted.
   * \param particle Starting particle satisfying lepton_access()'s contracts;
   * environmentNode_ is not read by this function.
   * \param massGeV Rest mass in GeV/c^2.
   * \param chargeNumber Signed charge in elementary-charge units.
   * \param fieldT Previously selected field in tesla in the common system.
   * \param limit Validated magnetic result; its distance does not cap this query.
   * \param pointM Plane point in metres in the common coordinate system.
   * \param normal Finite unit plane normal in the same coordinate system.
   * \return Forward leap-frog distance in metres, or positive infinity.
   */
  template <typename TTreeView>
  [[nodiscard]] KOKKOS_INLINE_FUNCTION double lepton_observation_distance(
      TTreeView const& tree, ParticleState const& particle, double massGeV,
      double chargeNumber, double const (&fieldT)[3],
      MagneticStepLimitResult const& limit, double const (&pointM)[3],
      double const (&normal)[3]);

} // namespace corsika::kokkos

#include <corsika/detail/kokkos/environment/EnvironmentLeptonBoundary.inl>

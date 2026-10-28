/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <Kokkos_Macros.hpp>
#include <corsika/kokkos/em/algorithms/LeptonContinuousStep.hpp>
#include <corsika/kokkos/em/algorithms/NativeMuonPairCutRecord.hpp>
#include <corsika/kokkos/em/algorithms/NativeMuonPairCut.hpp>
#include <corsika/kokkos/em/algorithms/PhotonTransportStep.hpp>
#include <corsika/kokkos/em/ProcessId.hpp>
#include <corsika/kokkos/environment/EnvironmentLeptonBoundary.hpp>
#include <corsika/kokkos/environment/UniformMagneticField.hpp>

namespace corsika::kokkos {

  namespace lepton_transport_detail {
    // Fixed production policy; stochastic in-flight annihilation is unaffected.
    inline constexpr bool EnableStoppedPositronAnnihilation = false;
    KOKKOS_INLINE_FUNCTION void prepare_stopped_annihilation(
        LeptonTransportRecord& record);
  } // namespace lepton_transport_detail

  /** Advance one charged lepton using the production environment-tree path.
   * The caller provides read-only native continuous tables, an exported tree and
   * an independently configured observation plane in one coordinate system.
   * table and environment owners must outlive the call/kernel in accessible memory.
   * Step competition uses start-tangent grammage; continuous loss uses the completed
   * endpoint chord. No scattering draw, discrete final state or CPU handler runs here.
   *
   * \param table Borrowed native continuous tables and kinetic transport cuts in MeV.
   * \param applyMoliere Whether the later split scattering stage will run.
   * \param environment Read-only, validated C8 environment-tree snapshot view.
   * \param config Query limits, separate observation plane and diagnostic origin.
   * \param interaction Previous sampled-distance, selected, no-interaction or cut record.
   * \param record Exclusive output slot; retained unchanged or partly written on failure.
   * \param fallback Exclusive failure slot; only written when returning 1.
   * \param external Optional non-absorbing material-interface distance candidate.
   * \return 0 for success, 1 for explicit CPU fallback, or 2 for a successful
   * zero-length cut that must bypass split scattering (then be normalized to 0).
   * Input and output objects must not alias. Inspect the return value before using
   * record; no new rollback or output clearing is added to the production policy.
   */
  template <typename TEnvironmentView>
  [[nodiscard]] KOKKOS_INLINE_FUNCTION std::uint32_t transport_lepton(
      ContinuousPhysicsView const& table, bool applyMoliere,
      TEnvironmentView const& environment, TransportGeometryConfig const& config,
      InteractionRecord const& interaction, LeptonTransportRecord& record,
      FallbackRequest& fallback, ExternalTransportBoundary const external = {});

} // namespace corsika::kokkos

#include <corsika/detail/kokkos/em/algorithms/LeptonTransportStep.inl>

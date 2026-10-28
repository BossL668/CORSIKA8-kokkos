/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <Kokkos_Macros.hpp>
#include <corsika/kokkos/em/PhotonTransportRecord.hpp>
#include <corsika/kokkos/em/FallbackRequest.hpp>
#include <corsika/kokkos/em/ParticleId.hpp>
#include <corsika/kokkos/environment/TransportGeometryConfig.hpp>
#include <corsika/kokkos/ScalarPhysicalConstants.hpp>

namespace corsika::kokkos {

  /** Result of one photon step, including the original partial record on failure.
   * fallbackFlag_ == 1 requires caller handling; constructing the request does not
   * execute a CPU handler. Deferred selected photon processes remain in record_.
   */
  struct PhotonTransportOutcome {
    PhotonTransportRecord record_{};
    FallbackRequest fallback_{};
    std::uint32_t fallbackFlag_{};
  };
  static_assert(std::is_trivially_copyable_v<PhotonTransportOutcome>);

  namespace photon_transport_detail {
    KOKKOS_INLINE_FUNCTION FallbackRequest
    make_fallback(InteractionRecord const& interaction, FallbackReason reason);
    KOKKOS_INLINE_FUNCTION bool finite(double value);
    KOKKOS_INLINE_FUNCTION void advance(ParticleState const& start, double distanceM,
                                        ParticleState& end);
    KOKKOS_INLINE_FUNCTION double radius(double const (&centerM)[3],
                                         double const (&positionM)[3],
                                         double (&radial)[3]);
  } // namespace photon_transport_detail

  /** Advance one selected photon on the production environment-tree path.
   *
   * \param environment Validated snapshot arrays, alive throughout the call/kernel.
   * \param config Query parameters, independent observation plane and diagnostic origin.
   * \param interaction Selected, no-discrete-interaction or pre-cut record.
   * \param external Optional external straight material interface.
   * \return Complete step or the unchanged production fallback reason/diagnostics.
   *
   * Input state and geometry are read-only. The step retains original boundary
   * competition, same-value ordering, straight grammage inversion and time cuts.
   * No RNG draw, reaction final state, observation output or CPU replay is executed.
   * On failure a partial record may exist: callers must inspect fallbackFlag_.
   */
  template <typename TEnvironmentView>
  [[nodiscard]] KOKKOS_INLINE_FUNCTION PhotonTransportOutcome transport_photon(
      TEnvironmentView const& environment, TransportGeometryConfig const& config,
      InteractionRecord const& interaction,
      ExternalTransportBoundary const external = {});

} // namespace corsika::kokkos

#include <corsika/detail/kokkos/em/algorithms/PhotonTransportStep.inl>

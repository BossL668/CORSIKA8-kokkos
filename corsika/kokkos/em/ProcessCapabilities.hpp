/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/kokkos/em/ParticleId.hpp>
#include <corsika/kokkos/em/ProcessId.hpp>
#include <corsika/kokkos/em/FallbackRequest.hpp>

namespace corsika::kokkos {

  /** Production final-state routing; does not execute a kernel or CPU handler. */
  enum class ProcessCapability : std::int32_t {
    PhotonPair = 0,
    Bremsstrahlung = 1,
    Compton = 2,
    Photoelectric = 3,
    Annihilation = 4,
    Ionization = 5,
    ElectronPair = 6,
    CpuOnlyFinalState = 7,
    GpuFinalStateNotImplemented = 8,
    PhotonMuonPair = 9,
  };
  /** Preserve production diagnostics, including unknown query-only identifiers. */
  [[nodiscard]] inline char const* em_process_name(std::int32_t process_id) noexcept;
  /** Classify the process without admitting unknown reactions to the CPU-only path. */
  [[nodiscard]] CORSIKA_HOST_DEVICE inline ProcessCapability process_capability(
      std::int32_t pid, std::int32_t process_id);
  /** Distinguish declared CPU processes from unimplemented device processes. */
  [[nodiscard]] CORSIKA_HOST_DEVICE inline FallbackReason process_fallback_reason(
      ProcessCapability capability);
  inline constexpr double NativeSelectionReplayEndpointWidth = 0.;
  /** Keep valid endpoint quantiles on device; CPU-only processes replay on the host.
   * This predicate is not general input validation: callers retain their query guards. */
  [[nodiscard]] CORSIKA_HOST_DEVICE inline bool proposal_native_selection_requires_replay(
      std::int32_t pid, std::int32_t process_id, double residual_quantile);

} // namespace corsika::kokkos

#include <corsika/detail/kokkos/em/ProcessCapabilities.inl>

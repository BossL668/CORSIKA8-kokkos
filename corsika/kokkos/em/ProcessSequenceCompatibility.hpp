/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <cstddef>
#include <stdexcept>
#include <string>
#include <type_traits>

#include <corsika/framework/process/ProcessSequence.hpp>
#include <corsika/framework/process/ProcessTraits.hpp>
#include <corsika/framework/process/SwitchProcessSequence.hpp>

namespace corsika::kokkos {

  /**
   * How a scalar process contract is preserved when an EM particle is routed
   * around ProcessSequence into the resident Kokkos cascade.
   */
  enum class EmStepProcessPolicy {
    ReplacedOnDevice,
    ReplayedFromDeviceRecord,
    DeferredToCpu,
    InapplicableToRoutedEm,
    DiagnosticOnly,
  };

  template <typename TProcess, EmStepProcessPolicy Policy>
  struct EmStepProcessRegistration {
    using process_type = std::decay_t<TProcess>;
    static constexpr EmStepProcessPolicy policy_ = Policy;
  };

  /**
   * Explicit allow-list for scalar process contracts around resident Kokkos EM.
   *
   * Registration declares who preserves a process; it does not execute it.
   * Call validateOrThrow before routing any particles. The caller remains
   * responsible for implementing the declared replacement/replay/CPU path.
   * Cascade-equation callbacks remain in the host cascade loop.
   * Every ContinuousProcess, SecondariesProcess, InteractionProcess,
   * DecayProcess, BoundaryCrossingProcess, and StackProcess in the application
   * sequence must be registered. Unknown types are rejected before the first
   * accelerated shower instead of being silently bypassed.
   */
  template <typename... TRegistrations>
  class EmStepProcessRegistry {
  public:
    static_assert(
        (std::is_same_v<TRegistrations,
                        EmStepProcessRegistration<typename TRegistrations::process_type,
                                                  TRegistrations::policy_>> &&
         ...),
        "GpuEmStepProcessRegistry arguments must be registrations");

    template <typename TSequence>
    static constexpr std::size_t unregisteredContinuousProcessCount();

    template <typename TSequence>
    static constexpr std::size_t unregisteredSecondariesProcessCount();

    template <typename TSequence>
    static constexpr std::size_t unregisteredInteractionProcessCount();

    template <typename TSequence>
    static constexpr std::size_t unregisteredDecayProcessCount();

    template <typename TSequence>
    static constexpr std::size_t unregisteredBoundaryProcessCount();

    template <typename TSequence>
    static constexpr std::size_t unregisteredStackProcessCount();

    template <typename TSequence>
    static constexpr bool compatible();

    template <typename TSequence>
    static void validateOrThrow();

    static constexpr std::size_t registrationCount();

    static constexpr std::size_t replacedOnDeviceCount();

    static constexpr std::size_t replayedFromDeviceRecordCount();

    static constexpr std::size_t deferredToCpuCount();

    static constexpr std::size_t inapplicableToRoutedEmCount();

    static constexpr std::size_t diagnosticOnlyCount();
  };

} // namespace corsika::kokkos

#include <corsika/detail/kokkos/em/ProcessSequenceCompatibility.inl>

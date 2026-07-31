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

namespace corsika::gpu::em {

  /**
   * How a scalar ContinuousProcess is preserved when an EM particle is routed
   * around ProcessSequence into the resident CUDA cascade.
   */
  enum class GpuEmStepProcessPolicy {
    ReplacedOnDevice,
    ReplayedFromDeviceRecord,
  };

  template <typename TProcess, GpuEmStepProcessPolicy Policy>
  struct GpuEmStepProcessRegistration {
    using process_type = std::decay_t<TProcess>;
    static constexpr GpuEmStepProcessPolicy policy = Policy;
  };

  namespace process_sequence_detail {

    template <typename TNode, typename... TRegistrations>
    struct UnregisteredContinuousCount;

    template <typename TNode, typename... TRegistrations>
    struct IsRegistered
        : std::bool_constant<
              (std::is_same_v<
                   std::decay_t<TNode>,
                   typename TRegistrations::process_type> ||
               ...)> {};

    template <typename TNode, typename... TRegistrations>
    struct UnregisteredContinuousCountImpl {
      using node_type = std::decay_t<TNode>;
      static constexpr std::size_t value =
          is_continuous_process_v<node_type> &&
                  !IsRegistered<node_type, TRegistrations...>::value
              ? 1
              : 0;
    };

    template <typename TProcess1, typename TProcess2, int ProcessIndexOffset,
              int IndexOfProcess1, int IndexOfProcess2,
              typename... TRegistrations>
    struct UnregisteredContinuousCountImpl<
        ProcessSequence<TProcess1, TProcess2, ProcessIndexOffset,
                        IndexOfProcess1, IndexOfProcess2>,
        TRegistrations...> {
      using node_type =
          ProcessSequence<TProcess1, TProcess2, ProcessIndexOffset,
                          IndexOfProcess1, IndexOfProcess2>;
      static constexpr std::size_t value =
          IsRegistered<node_type, TRegistrations...>::value
              ? 0
              : UnregisteredContinuousCount<
                    TProcess1, TRegistrations...>::value +
                    UnregisteredContinuousCount<
                        TProcess2, TRegistrations...>::value;
    };

    template <typename TCondition, typename TSequence, typename USequence,
              int IndexFirstProcess, int IndexOfProcess1,
              int IndexOfProcess2, typename... TRegistrations>
    struct UnregisteredContinuousCountImpl<
        SwitchProcessSequence<TCondition, TSequence, USequence,
                              IndexFirstProcess, IndexOfProcess1,
                              IndexOfProcess2>,
        TRegistrations...> {
      using node_type =
          SwitchProcessSequence<TCondition, TSequence, USequence,
                                IndexFirstProcess, IndexOfProcess1,
                                IndexOfProcess2>;
      static constexpr std::size_t value =
          IsRegistered<node_type, TRegistrations...>::value
              ? 0
              : UnregisteredContinuousCount<
                    TSequence, TRegistrations...>::value +
                    UnregisteredContinuousCount<
                        USequence, TRegistrations...>::value;
    };

    template <typename TNode, typename... TRegistrations>
    struct UnregisteredContinuousCount
        : UnregisteredContinuousCountImpl<
              std::decay_t<TNode>, TRegistrations...> {};

    template <GpuEmStepProcessPolicy Policy, typename... TRegistrations>
    inline constexpr std::size_t policyCount =
        (std::size_t{0} + ... +
         (TRegistrations::policy == Policy ? std::size_t{1}
                                           : std::size_t{0}));

  } // namespace process_sequence_detail

  /**
   * Explicit allow-list for ContinuousProcesses bypassed by resident CUDA EM.
   *
   * Unknown interaction/decay/stack/secondaries processes are not rejected by
   * this registry because they do not observe the scalar transport Step.
   * Unknown ContinuousProcesses are rejected before the first CUDA shower.
   */
  template <typename... TRegistrations>
  class GpuEmStepProcessRegistry {
  public:
    static_assert(
        (std::is_same_v<
             TRegistrations,
             GpuEmStepProcessRegistration<
                 typename TRegistrations::process_type,
                 TRegistrations::policy>> &&
         ...),
        "GpuEmStepProcessRegistry arguments must be registrations");

    template <typename TSequence>
    static constexpr std::size_t unregisteredContinuousProcessCount() {
      return process_sequence_detail::UnregisteredContinuousCount<
          TSequence, TRegistrations...>::value;
    }

    template <typename TSequence>
    static constexpr bool compatible() {
      return unregisteredContinuousProcessCount<TSequence>() == 0;
    }

    template <typename TSequence>
    static void validateOrThrow() {
      constexpr auto unregistered =
          unregisteredContinuousProcessCount<TSequence>();
      if constexpr (unregistered != 0) {
        throw std::runtime_error(
            "CUDA EM process-sequence compatibility failed: " +
            std::to_string(unregistered) +
            " ContinuousProcess type(s) can observe or modify an EM Step "
            "but are not registered as device-replaced or record-replayed");
      }
    }

    static constexpr std::size_t registrationCount() {
      return sizeof...(TRegistrations);
    }

    static constexpr std::size_t replacedOnDeviceCount() {
      return process_sequence_detail::policyCount<
          GpuEmStepProcessPolicy::ReplacedOnDevice,
          TRegistrations...>;
    }

    static constexpr std::size_t replayedFromDeviceRecordCount() {
      return process_sequence_detail::policyCount<
          GpuEmStepProcessPolicy::ReplayedFromDeviceRecord,
          TRegistrations...>;
    }
  };

} // namespace corsika::gpu::em

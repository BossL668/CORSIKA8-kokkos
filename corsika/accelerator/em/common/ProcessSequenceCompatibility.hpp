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
   * How a scalar process contract is preserved when an EM particle is routed
   * around ProcessSequence into the resident CUDA cascade.
   */
  enum class GpuEmStepProcessPolicy {
    ReplacedOnDevice,
    ReplayedFromDeviceRecord,
    DeferredToCpu,
    InapplicableToRoutedEm,
    DiagnosticOnly,
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
    struct UnregisteredSecondariesCount;

    template <typename TNode, typename TCategory,
              typename... TRegistrations>
    struct UnregisteredCategoryCount;

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

    template <typename TNode, typename... TRegistrations>
    struct UnregisteredSecondariesCountImpl {
      using node_type = std::decay_t<TNode>;
      static constexpr std::size_t value =
          is_secondaries_process_v<node_type> &&
                  !IsRegistered<node_type, TRegistrations...>::value
              ? 1
              : 0;
    };

    template <typename TProcess1, typename TProcess2, int ProcessIndexOffset,
              int IndexOfProcess1, int IndexOfProcess2,
              typename... TRegistrations>
    struct UnregisteredSecondariesCountImpl<
        ProcessSequence<TProcess1, TProcess2, ProcessIndexOffset,
                        IndexOfProcess1, IndexOfProcess2>,
        TRegistrations...> {
      using node_type =
          ProcessSequence<TProcess1, TProcess2, ProcessIndexOffset,
                          IndexOfProcess1, IndexOfProcess2>;
      static constexpr std::size_t value =
          IsRegistered<node_type, TRegistrations...>::value
              ? 0
              : UnregisteredSecondariesCount<
                    TProcess1, TRegistrations...>::value +
                    UnregisteredSecondariesCount<
                        TProcess2, TRegistrations...>::value;
    };

    template <typename TCondition, typename TSequence, typename USequence,
              int IndexFirstProcess, int IndexOfProcess1,
              int IndexOfProcess2, typename... TRegistrations>
    struct UnregisteredSecondariesCountImpl<
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
              : UnregisteredSecondariesCount<
                    TSequence, TRegistrations...>::value +
                    UnregisteredSecondariesCount<
                        USequence, TRegistrations...>::value;
    };

    template <typename TNode, typename... TRegistrations>
    struct UnregisteredSecondariesCount
        : UnregisteredSecondariesCountImpl<
              std::decay_t<TNode>, TRegistrations...> {};

    struct InteractionProcessCategory {
      template <typename TNode>
      static constexpr bool matches =
          is_interaction_process_v<TNode>;
    };

    struct DecayProcessCategory {
      template <typename TNode>
      static constexpr bool matches =
          is_decay_process_v<TNode>;
    };

    struct BoundaryProcessCategory {
      template <typename TNode>
      static constexpr bool matches =
          is_boundary_process_v<TNode>;
    };

    struct StackProcessCategory {
      template <typename TNode>
      static constexpr bool matches =
          is_stack_process_v<TNode>;
    };

    template <typename TNode, typename TCategory,
              typename... TRegistrations>
    struct UnregisteredCategoryCountImpl {
      using node_type = std::decay_t<TNode>;
      static constexpr std::size_t value =
          TCategory::template matches<node_type> &&
                  !IsRegistered<node_type, TRegistrations...>::value
              ? 1
              : 0;
    };

    template <typename TProcess1, typename TProcess2,
              int ProcessIndexOffset, int IndexOfProcess1,
              int IndexOfProcess2, typename TCategory,
              typename... TRegistrations>
    struct UnregisteredCategoryCountImpl<
        ProcessSequence<TProcess1, TProcess2, ProcessIndexOffset,
                        IndexOfProcess1, IndexOfProcess2>,
        TCategory, TRegistrations...> {
      using node_type =
          ProcessSequence<TProcess1, TProcess2, ProcessIndexOffset,
                          IndexOfProcess1, IndexOfProcess2>;
      static constexpr std::size_t value =
          IsRegistered<node_type, TRegistrations...>::value
              ? 0
              : UnregisteredCategoryCount<
                    TProcess1, TCategory,
                    TRegistrations...>::value +
                    UnregisteredCategoryCount<
                        TProcess2, TCategory,
                        TRegistrations...>::value;
    };

    template <typename TCondition, typename TSequence,
              typename USequence, int IndexFirstProcess,
              int IndexOfProcess1, int IndexOfProcess2,
              typename TCategory, typename... TRegistrations>
    struct UnregisteredCategoryCountImpl<
        SwitchProcessSequence<TCondition, TSequence, USequence,
                              IndexFirstProcess, IndexOfProcess1,
                              IndexOfProcess2>,
        TCategory, TRegistrations...> {
      using node_type =
          SwitchProcessSequence<TCondition, TSequence, USequence,
                                IndexFirstProcess, IndexOfProcess1,
                                IndexOfProcess2>;
      static constexpr std::size_t value =
          IsRegistered<node_type, TRegistrations...>::value
              ? 0
              : UnregisteredCategoryCount<
                    TSequence, TCategory,
                    TRegistrations...>::value +
                    UnregisteredCategoryCount<
                        USequence, TCategory,
                        TRegistrations...>::value;
    };

    template <typename TNode, typename TCategory,
              typename... TRegistrations>
    struct UnregisteredCategoryCount
        : UnregisteredCategoryCountImpl<
              std::decay_t<TNode>, TCategory,
              TRegistrations...> {};

    template <GpuEmStepProcessPolicy Policy, typename... TRegistrations>
    inline constexpr std::size_t policyCount =
        (std::size_t{0} + ... +
         (TRegistrations::policy == Policy ? std::size_t{1}
                                           : std::size_t{0}));

  } // namespace process_sequence_detail

  /**
   * Explicit allow-list for scalar process contracts around resident CUDA EM.
   *
   * Every ContinuousProcess, SecondariesProcess, InteractionProcess,
   * DecayProcess, BoundaryCrossingProcess, and StackProcess in the application
   * sequence must be registered. Unknown types are rejected before the first
   * CUDA shower instead of being silently bypassed.
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
    static constexpr std::size_t unregisteredSecondariesProcessCount() {
      return process_sequence_detail::UnregisteredSecondariesCount<
          TSequence, TRegistrations...>::value;
    }

    template <typename TSequence>
    static constexpr std::size_t unregisteredInteractionProcessCount() {
      return process_sequence_detail::UnregisteredCategoryCount<
          TSequence,
          process_sequence_detail::InteractionProcessCategory,
          TRegistrations...>::value;
    }

    template <typename TSequence>
    static constexpr std::size_t unregisteredDecayProcessCount() {
      return process_sequence_detail::UnregisteredCategoryCount<
          TSequence,
          process_sequence_detail::DecayProcessCategory,
          TRegistrations...>::value;
    }

    template <typename TSequence>
    static constexpr std::size_t unregisteredBoundaryProcessCount() {
      return process_sequence_detail::UnregisteredCategoryCount<
          TSequence,
          process_sequence_detail::BoundaryProcessCategory,
          TRegistrations...>::value;
    }

    template <typename TSequence>
    static constexpr std::size_t unregisteredStackProcessCount() {
      return process_sequence_detail::UnregisteredCategoryCount<
          TSequence,
          process_sequence_detail::StackProcessCategory,
          TRegistrations...>::value;
    }

    template <typename TSequence>
    static constexpr bool compatible() {
      return unregisteredContinuousProcessCount<TSequence>() == 0 &&
             unregisteredSecondariesProcessCount<TSequence>() == 0 &&
             unregisteredInteractionProcessCount<TSequence>() == 0 &&
             unregisteredDecayProcessCount<TSequence>() == 0 &&
             unregisteredBoundaryProcessCount<TSequence>() == 0 &&
             unregisteredStackProcessCount<TSequence>() == 0;
    }

    template <typename TSequence>
    static void validateOrThrow() {
      constexpr auto unregistered =
          unregisteredContinuousProcessCount<TSequence>();
      constexpr auto unregistered_secondaries =
          unregisteredSecondariesProcessCount<TSequence>();
      constexpr auto unregistered_interactions =
          unregisteredInteractionProcessCount<TSequence>();
      constexpr auto unregistered_decays =
          unregisteredDecayProcessCount<TSequence>();
      constexpr auto unregistered_boundaries =
          unregisteredBoundaryProcessCount<TSequence>();
      constexpr auto unregistered_stack =
          unregisteredStackProcessCount<TSequence>();
      if constexpr (unregistered != 0 ||
                    unregistered_secondaries != 0 ||
                    unregistered_interactions != 0 ||
                    unregistered_decays != 0 ||
                    unregistered_boundaries != 0 ||
                    unregistered_stack != 0) {
        throw std::runtime_error(
            "CUDA EM process-sequence compatibility failed: " +
            std::to_string(unregistered) +
            " ContinuousProcess type(s) and " +
            std::to_string(unregistered_secondaries) +
            " SecondariesProcess type(s), " +
            std::to_string(unregistered_interactions) +
            " InteractionProcess type(s), " +
            std::to_string(unregistered_decays) +
            " DecayProcess type(s), " +
            std::to_string(unregistered_boundaries) +
            " BoundaryCrossingProcess type(s), and " +
            std::to_string(unregistered_stack) +
            " StackProcess type(s) can observe or modify routed EM state "
            "but are not explicitly registered");
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

    static constexpr std::size_t deferredToCpuCount() {
      return process_sequence_detail::policyCount<
          GpuEmStepProcessPolicy::DeferredToCpu,
          TRegistrations...>;
    }

    static constexpr std::size_t inapplicableToRoutedEmCount() {
      return process_sequence_detail::policyCount<
          GpuEmStepProcessPolicy::InapplicableToRoutedEm,
          TRegistrations...>;
    }

    static constexpr std::size_t diagnosticOnlyCount() {
      return process_sequence_detail::policyCount<
          GpuEmStepProcessPolicy::DiagnosticOnly,
          TRegistrations...>;
    }
  };

} // namespace corsika::gpu::em

/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once
namespace corsika::kokkos {
  namespace process_sequence_detail {

    template <typename TNode, typename... TRegistrations>
    struct UnregisteredContinuousCount;

    template <typename TNode, typename... TRegistrations>
    struct UnregisteredSecondariesCount;

    template <typename TNode, typename TCategory, typename... TRegistrations>
    struct UnregisteredCategoryCount;

    template <typename TNode, typename... TRegistrations>
    struct IsRegistered
        : std::bool_constant<(std::is_same_v<std::decay_t<TNode>,
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
              int IndexOfProcess1, int IndexOfProcess2, typename... TRegistrations>
    struct UnregisteredContinuousCountImpl<
        ProcessSequence<TProcess1, TProcess2, ProcessIndexOffset, IndexOfProcess1,
                        IndexOfProcess2>,
        TRegistrations...> {
      using node_type = ProcessSequence<TProcess1, TProcess2, ProcessIndexOffset,
                                        IndexOfProcess1, IndexOfProcess2>;
      static constexpr std::size_t value =
          IsRegistered<node_type, TRegistrations...>::value
              ? 0
              : UnregisteredContinuousCount<TProcess1, TRegistrations...>::value +
                    UnregisteredContinuousCount<TProcess2, TRegistrations...>::value;
    };

    template <typename TCondition, typename TSequence, typename USequence,
              int IndexFirstProcess, int IndexOfProcess1, int IndexOfProcess2,
              typename... TRegistrations>
    struct UnregisteredContinuousCountImpl<
        SwitchProcessSequence<TCondition, TSequence, USequence, IndexFirstProcess,
                              IndexOfProcess1, IndexOfProcess2>,
        TRegistrations...> {
      using node_type =
          SwitchProcessSequence<TCondition, TSequence, USequence, IndexFirstProcess,
                                IndexOfProcess1, IndexOfProcess2>;
      static constexpr std::size_t value =
          IsRegistered<node_type, TRegistrations...>::value
              ? 0
              : UnregisteredContinuousCount<TSequence, TRegistrations...>::value +
                    UnregisteredContinuousCount<USequence, TRegistrations...>::value;
    };

    template <typename TNode, typename... TRegistrations>
    struct UnregisteredContinuousCount
        : UnregisteredContinuousCountImpl<std::decay_t<TNode>, TRegistrations...> {};

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
              int IndexOfProcess1, int IndexOfProcess2, typename... TRegistrations>
    struct UnregisteredSecondariesCountImpl<
        ProcessSequence<TProcess1, TProcess2, ProcessIndexOffset, IndexOfProcess1,
                        IndexOfProcess2>,
        TRegistrations...> {
      using node_type = ProcessSequence<TProcess1, TProcess2, ProcessIndexOffset,
                                        IndexOfProcess1, IndexOfProcess2>;
      static constexpr std::size_t value =
          IsRegistered<node_type, TRegistrations...>::value
              ? 0
              : UnregisteredSecondariesCount<TProcess1, TRegistrations...>::value +
                    UnregisteredSecondariesCount<TProcess2, TRegistrations...>::value;
    };

    template <typename TCondition, typename TSequence, typename USequence,
              int IndexFirstProcess, int IndexOfProcess1, int IndexOfProcess2,
              typename... TRegistrations>
    struct UnregisteredSecondariesCountImpl<
        SwitchProcessSequence<TCondition, TSequence, USequence, IndexFirstProcess,
                              IndexOfProcess1, IndexOfProcess2>,
        TRegistrations...> {
      using node_type =
          SwitchProcessSequence<TCondition, TSequence, USequence, IndexFirstProcess,
                                IndexOfProcess1, IndexOfProcess2>;
      static constexpr std::size_t value =
          IsRegistered<node_type, TRegistrations...>::value
              ? 0
              : UnregisteredSecondariesCount<TSequence, TRegistrations...>::value +
                    UnregisteredSecondariesCount<USequence, TRegistrations...>::value;
    };

    template <typename TNode, typename... TRegistrations>
    struct UnregisteredSecondariesCount
        : UnregisteredSecondariesCountImpl<std::decay_t<TNode>, TRegistrations...> {};

    struct InteractionProcessCategory {
      template <typename TNode>
      static constexpr bool matches = is_interaction_process_v<TNode>;
    };

    struct DecayProcessCategory {
      template <typename TNode>
      static constexpr bool matches = is_decay_process_v<TNode>;
    };

    struct BoundaryProcessCategory {
      template <typename TNode>
      static constexpr bool matches = is_boundary_process_v<TNode>;
    };

    struct StackProcessCategory {
      template <typename TNode>
      static constexpr bool matches = is_stack_process_v<TNode>;
    };

    template <typename TNode, typename TCategory, typename... TRegistrations>
    struct UnregisteredCategoryCountImpl {
      using node_type = std::decay_t<TNode>;
      static constexpr std::size_t value =
          TCategory::template matches<node_type> &&
                  !IsRegistered<node_type, TRegistrations...>::value
              ? 1
              : 0;
    };

    template <typename TProcess1, typename TProcess2, int ProcessIndexOffset,
              int IndexOfProcess1, int IndexOfProcess2, typename TCategory,
              typename... TRegistrations>
    struct UnregisteredCategoryCountImpl<
        ProcessSequence<TProcess1, TProcess2, ProcessIndexOffset, IndexOfProcess1,
                        IndexOfProcess2>,
        TCategory, TRegistrations...> {
      using node_type = ProcessSequence<TProcess1, TProcess2, ProcessIndexOffset,
                                        IndexOfProcess1, IndexOfProcess2>;
      static constexpr std::size_t value =
          IsRegistered<node_type, TRegistrations...>::value
              ? 0
              : UnregisteredCategoryCount<TProcess1, TCategory,
                                          TRegistrations...>::value +
                    UnregisteredCategoryCount<TProcess2, TCategory,
                                              TRegistrations...>::value;
    };

    template <typename TCondition, typename TSequence, typename USequence,
              int IndexFirstProcess, int IndexOfProcess1, int IndexOfProcess2,
              typename TCategory, typename... TRegistrations>
    struct UnregisteredCategoryCountImpl<
        SwitchProcessSequence<TCondition, TSequence, USequence, IndexFirstProcess,
                              IndexOfProcess1, IndexOfProcess2>,
        TCategory, TRegistrations...> {
      using node_type =
          SwitchProcessSequence<TCondition, TSequence, USequence, IndexFirstProcess,
                                IndexOfProcess1, IndexOfProcess2>;
      static constexpr std::size_t value =
          IsRegistered<node_type, TRegistrations...>::value
              ? 0
              : UnregisteredCategoryCount<TSequence, TCategory,
                                          TRegistrations...>::value +
                    UnregisteredCategoryCount<USequence, TCategory,
                                              TRegistrations...>::value;
    };

    template <typename TNode, typename TCategory, typename... TRegistrations>
    struct UnregisteredCategoryCount
        : UnregisteredCategoryCountImpl<std::decay_t<TNode>, TCategory,
                                        TRegistrations...> {};

    template <EmStepProcessPolicy Policy, typename... TRegistrations>
    inline constexpr std::size_t policy_count =
        (std::size_t{0} + ... +
         (TRegistrations::policy_ == Policy ? std::size_t{1} : std::size_t{0}));

  } // namespace process_sequence_detail
  template <typename... TRegistrations>
  template <typename TSequence>
  constexpr std::size_t
  EmStepProcessRegistry<TRegistrations...>::unregisteredContinuousProcessCount() {
    return process_sequence_detail::UnregisteredContinuousCount<TSequence,
                                                                TRegistrations...>::value;
  }
  template <typename... TRegistrations>
  template <typename TSequence>
  constexpr std::size_t
  EmStepProcessRegistry<TRegistrations...>::unregisteredSecondariesProcessCount() {
    return process_sequence_detail::UnregisteredSecondariesCount<
        TSequence, TRegistrations...>::value;
  }
  template <typename... TRegistrations>
  template <typename TSequence>
  constexpr std::size_t
  EmStepProcessRegistry<TRegistrations...>::unregisteredInteractionProcessCount() {
    return process_sequence_detail::UnregisteredCategoryCount<
        TSequence, process_sequence_detail::InteractionProcessCategory,
        TRegistrations...>::value;
  }
  template <typename... TRegistrations>
  template <typename TSequence>
  constexpr std::size_t
  EmStepProcessRegistry<TRegistrations...>::unregisteredDecayProcessCount() {
    return process_sequence_detail::UnregisteredCategoryCount<
        TSequence, process_sequence_detail::DecayProcessCategory,
        TRegistrations...>::value;
  }
  template <typename... TRegistrations>
  template <typename TSequence>
  constexpr std::size_t
  EmStepProcessRegistry<TRegistrations...>::unregisteredBoundaryProcessCount() {
    return process_sequence_detail::UnregisteredCategoryCount<
        TSequence, process_sequence_detail::BoundaryProcessCategory,
        TRegistrations...>::value;
  }
  template <typename... TRegistrations>
  template <typename TSequence>
  constexpr std::size_t
  EmStepProcessRegistry<TRegistrations...>::unregisteredStackProcessCount() {
    return process_sequence_detail::UnregisteredCategoryCount<
        TSequence, process_sequence_detail::StackProcessCategory,
        TRegistrations...>::value;
  }
  template <typename... TRegistrations>
  template <typename TSequence>
  constexpr bool EmStepProcessRegistry<TRegistrations...>::compatible() {
    return unregisteredContinuousProcessCount<TSequence>() == 0 &&
           unregisteredSecondariesProcessCount<TSequence>() == 0 &&
           unregisteredInteractionProcessCount<TSequence>() == 0 &&
           unregisteredDecayProcessCount<TSequence>() == 0 &&
           unregisteredBoundaryProcessCount<TSequence>() == 0 &&
           unregisteredStackProcessCount<TSequence>() == 0;
  }
  template <typename... TRegistrations>
  template <typename TSequence>
  void EmStepProcessRegistry<TRegistrations...>::validateOrThrow() {
    constexpr auto unregistered = unregisteredContinuousProcessCount<TSequence>();
    constexpr auto unregisteredSecondaries =
        unregisteredSecondariesProcessCount<TSequence>();
    constexpr auto unregisteredInteractions =
        unregisteredInteractionProcessCount<TSequence>();
    constexpr auto unregisteredDecays = unregisteredDecayProcessCount<TSequence>();
    constexpr auto unregisteredBoundaries = unregisteredBoundaryProcessCount<TSequence>();
    constexpr auto unregisteredStack = unregisteredStackProcessCount<TSequence>();
    if constexpr (unregistered != 0 || unregisteredSecondaries != 0 ||
                  unregisteredInteractions != 0 || unregisteredDecays != 0 ||
                  unregisteredBoundaries != 0 || unregisteredStack != 0) {
      throw std::runtime_error(
          "Kokkos EM process-sequence compatibility failed: " +
          std::to_string(unregistered) + " ContinuousProcess type(s) and " +
          std::to_string(unregisteredSecondaries) + " SecondariesProcess type(s), " +
          std::to_string(unregisteredInteractions) + " InteractionProcess type(s), " +
          std::to_string(unregisteredDecays) + " DecayProcess type(s), " +
          std::to_string(unregisteredBoundaries) +
          " BoundaryCrossingProcess type(s), and " + std::to_string(unregisteredStack) +
          " StackProcess type(s) can observe or modify routed EM state "
          "but are not explicitly registered");
    }
  }
  template <typename... TRegistrations>
  constexpr std::size_t EmStepProcessRegistry<TRegistrations...>::registrationCount() {
    return sizeof...(TRegistrations);
  }
  template <typename... TRegistrations>
  constexpr std::size_t
  EmStepProcessRegistry<TRegistrations...>::replacedOnDeviceCount() {
    return process_sequence_detail::policy_count<EmStepProcessPolicy::ReplacedOnDevice,
                                                 TRegistrations...>;
  }
  template <typename... TRegistrations>
  constexpr std::size_t
  EmStepProcessRegistry<TRegistrations...>::replayedFromDeviceRecordCount() {
    return process_sequence_detail::policy_count<
        EmStepProcessPolicy::ReplayedFromDeviceRecord, TRegistrations...>;
  }
  template <typename... TRegistrations>
  constexpr std::size_t EmStepProcessRegistry<TRegistrations...>::deferredToCpuCount() {
    return process_sequence_detail::policy_count<EmStepProcessPolicy::DeferredToCpu,
                                                 TRegistrations...>;
  }
  template <typename... TRegistrations>
  constexpr std::size_t
  EmStepProcessRegistry<TRegistrations...>::inapplicableToRoutedEmCount() {
    return process_sequence_detail::policy_count<
        EmStepProcessPolicy::InapplicableToRoutedEm, TRegistrations...>;
  }
  template <typename... TRegistrations>
  constexpr std::size_t EmStepProcessRegistry<TRegistrations...>::diagnosticOnlyCount() {
    return process_sequence_detail::policy_count<EmStepProcessPolicy::DiagnosticOnly,
                                                 TRegistrations...>;
  }
} // namespace corsika::kokkos

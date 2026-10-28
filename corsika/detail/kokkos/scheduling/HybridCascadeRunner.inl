/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

namespace corsika::kokkos {

  namespace hybrid_runner_detail {
    template <typename T, typename TTopology, typename = void>
    struct HasEnvironmentTreeSetter : std::false_type {};

    template <typename T, typename TTopology>
    struct HasEnvironmentTreeSetter<
        T, TTopology,
        std::void_t<decltype(std::declval<T&>().setEnvironmentTree(
            std::declval<TTopology const&>()))>> : std::true_type {};

    template <typename T, typename TTopology>
    struct HasConstructorBoundTopology : std::false_type {};

    template <typename TStack, typename TProposalModel, typename TProcessList,
              typename TEnvironment, typename TModelProperties>
    struct HasConstructorBoundTopology<
        ProposalCpuFallbackHandler<TStack, TProposalModel, TProcessList, TEnvironment>,
        EnvironmentTopology<TModelProperties>>
        : std::is_same<typename TEnvironment::BaseNodeType::IModelProperties,
                       TModelProperties> {};
  } // namespace hybrid_runner_detail

  template <typename TProcessRegistry, typename TModelProperties, typename TEnvironment,
            typename TTracking, typename TSequence, typename TOutput, typename TStack,
            typename TBackendFactory, typename TFallbackFactory,
            typename TOutputSinkFactory, typename TConfigureCascade, typename TOnComplete>
  void run_hybrid_cascade(
      CoordinateSystemPtr const& coordinateSystem,
      EnvironmentSnapshot<TModelProperties> const& environmentSnapshot,
      TransportGeometryConfig const& geometry, TEnvironment& environment,
      TTracking& tracking, TSequence& sequence, TOutput& output, TStack& stack,
      HybridRunOptions const& options, TBackendFactory&& backendFactory,
      TFallbackFactory&& fallbackFactory, TOutputSinkFactory&& outputSinkFactory,
      TConfigureCascade&& configureCascade, TOnComplete&& onComplete) {
    TProcessRegistry::template validateOrThrow<TSequence>();
    CORSIKA_LOG_INFO(
        "Accelerated EM process registry accepted {} process contracts "
        "({} device-replaced, {} record-replayed, {} deferred-to-CPU, "
        "{} inapplicable-to-routed-EM, {} diagnostic-only)",
        TProcessRegistry::registrationCount(), TProcessRegistry::replacedOnDeviceCount(),
        TProcessRegistry::replayedFromDeviceRecordCount(),
        TProcessRegistry::deferredToCpuCount(),
        TProcessRegistry::inapplicableToRoutedEmCount(),
        TProcessRegistry::diagnosticOnlyCount());

    auto& backend = std::forward<TBackendFactory>(backendFactory)();
    using topology_type = EnvironmentTopology<TModelProperties>;
    auto fallbackHandler = [&]() {
      if constexpr (std::is_invocable_v<TFallbackFactory, topology_type const&>) {
        return std::forward<TFallbackFactory>(fallbackFactory)(
            environmentSnapshot.getTopology());
      } else {
        return std::forward<TFallbackFactory>(fallbackFactory)();
      }
    }();
    auto outputSink = std::forward<TOutputSinkFactory>(outputSinkFactory)();
    using stack_type = std::remove_reference_t<TStack>;
    using backend_type = std::remove_reference_t<decltype(backend)>;
    using fallback_handler_type = decltype(fallbackHandler);
    if constexpr (hybrid_runner_detail::HasEnvironmentTreeSetter<fallback_handler_type,
                                                                 topology_type>::value) {
      fallbackHandler.setEnvironmentTree(environmentSnapshot.getTopology());
    } else if constexpr (!(hybrid_runner_detail::HasConstructorBoundTopology<
                               fallback_handler_type, topology_type>::value &&
                           std::is_invocable_v<TFallbackFactory, topology_type const&>)) {
      throw std::logic_error(
          "full environment tree requires a node-aware CPU fallback handler");
    }
    using output_sink_type = decltype(outputSink);
    using router_type =
#ifdef CORSIKA_KOKKOS_STATIC_MULTIGPU
        frontier::BufferedFrontierRouter<
#endif
            physical_em_router_type<stack_type, TModelProperties, backend_type,
                                    fallback_handler_type, output_sink_type>
#ifdef CORSIKA_KOKKOS_STATIC_MULTIGPU
            ,
            TModelProperties>
#endif
        ;
    router_type router{
        backend,
        coordinateSystem,
        EnvironmentBorrowedView(bind_environment_data(environmentSnapshot)),
        geometry,
        environmentSnapshot.getTopology(),
        fallbackHandler,
        outputSink};
#ifdef CORSIKA_KOKKOS_STATIC_MULTIGPU
    router.setFrontierEnvironment(environmentSnapshot);
#endif
    router.setRetainRecords(options.retainRecords_);
    router.setFailOnUnexpectedFallback(options.failOnUnexpectedFallback_);

    using tracking_type = std::remove_reference_t<TTracking>;
    using sequence_type = std::remove_reference_t<TSequence>;
    using output_type = std::remove_reference_t<TOutput>;
    HybridCascade<tracking_type, sequence_type, output_type, stack_type, router_type>
        cascade{environment, tracking, sequence, output, stack, router};
    std::forward<TConfigureCascade>(configureCascade)(cascade);
    cascade.run();
    std::forward<TOnComplete>(onComplete)(backend, cascade, router, outputSink,
                                          fallbackHandler);
  }

  template <typename TProcessRegistry, typename... TArguments>
  void run_cuda_hybrid_cascade(TArguments&&... arguments) {
    run_hybrid_cascade<TProcessRegistry>(std::forward<TArguments>(arguments)...);
  }

} // namespace corsika::kokkos

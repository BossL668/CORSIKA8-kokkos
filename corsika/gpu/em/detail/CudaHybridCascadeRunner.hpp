/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/framework/core/HybridCascade.hpp>
#include <corsika/framework/core/Logging.hpp>
#include <corsika/gpu/em/PhysicalCudaEmRouter.hpp>
#include <corsika/gpu/em/ProcessSequenceCompatibility.hpp>

#include <type_traits>
#include <utility>

namespace corsika::gpu::em::detail {

  struct CudaHybridRunOptions {
    bool retain_records{};
    bool fail_on_unexpected_fallback{true};
  };

  /**
   * Execute one CUDA-backed HybridCascade without knowing the concrete
   * environment, physics models, writers, detector types, or report schema.
   * Factories keep application-owned construction policy outside the core;
   * on_complete runs while every typed transport object is still alive.
   */
  template <typename TProcessRegistry, typename TEnvironment,
            typename TTracking, typename TSequence, typename TOutput,
            typename TStack, typename TBackendFactory,
            typename TFallbackFactory,
            typename TOutputSinkFactory, typename TConfigureCascade,
            typename TOnComplete>
  void runCudaHybridCascade(
      CoordinateSystemPtr const& root_cs,
      EnvironmentSnapshot const& environment_snapshot,
      TEnvironment& environment, TTracking& tracking, TSequence& sequence,
      TOutput& output, TStack& stack, CudaHybridRunOptions const& options,
      TBackendFactory&& backend_factory,
      TFallbackFactory&& fallback_factory,
      TOutputSinkFactory&& output_sink_factory,
      TConfigureCascade&& configure_cascade,
      TOnComplete&& on_complete) {
    TProcessRegistry::template validateOrThrow<TSequence>();
    CORSIKA_LOG_INFO(
        "CUDA EM process registry accepted {} process contracts "
        "({} device-replaced, {} record-replayed, {} deferred-to-CPU, "
        "{} inapplicable-to-routed-EM, {} diagnostic-only)",
        TProcessRegistry::registrationCount(),
        TProcessRegistry::replacedOnDeviceCount(),
        TProcessRegistry::replayedFromDeviceRecordCount(),
        TProcessRegistry::deferredToCpuCount(),
        TProcessRegistry::inapplicableToRoutedEmCount(),
        TProcessRegistry::diagnosticOnlyCount());

    auto& backend =
        std::forward<TBackendFactory>(backend_factory)();
    auto fallback_handler =
        std::forward<TFallbackFactory>(fallback_factory)();
    auto output_sink =
        std::forward<TOutputSinkFactory>(output_sink_factory)();
    using Stack = std::remove_reference_t<TStack>;
    using FallbackHandler = decltype(fallback_handler);
    using OutputSink = decltype(output_sink);
    using Router =
        PhysicalCudaEmRouter<Stack, FallbackHandler, OutputSink>;
    Router router{backend, root_cs, environment_snapshot, fallback_handler,
                  output_sink};
    router.setRetainRecords(options.retain_records);
    router.setFailOnUnexpectedFallback(
        options.fail_on_unexpected_fallback);

    using Tracking = std::remove_reference_t<TTracking>;
    using Sequence = std::remove_reference_t<TSequence>;
    using Output = std::remove_reference_t<TOutput>;
    HybridCascade<Tracking, Sequence, Output, Stack, Router> cascade{
        environment, tracking, sequence, output, stack, router};
    std::forward<TConfigureCascade>(configure_cascade)(cascade);
    cascade.run();
    std::forward<TOnComplete>(on_complete)(
        backend, cascade, router, output_sink, fallback_handler);
  }

} // namespace corsika::gpu::em::detail

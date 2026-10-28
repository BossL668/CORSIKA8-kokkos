/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/kokkos/application/cascade/CascadeContext.hpp>
#include <corsika/kokkos/application/cascade/CascadeEventConfig.hpp>
#include <corsika/kokkos/application/BackendOptions.hpp>
#include <corsika/kokkos/application/BackendCli.hpp>
#include <corsika/kokkos/application/DeviceCliOptions.hpp>
#include <corsika/kokkos/application/cascade/RunEnvironmentConfig.hpp>
#include <CLI/App.hpp>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>

#ifdef CORSIKA_KOKKOS_COORDINATOR
#ifndef CORSIKA_KOKKOS_STATIC_MULTIGPU
#define CORSIKA_KOKKOS_STATIC_MULTIGPU 1
#endif
#endif

namespace corsika {
  class OutputManager;
  namespace kokkos {
    class HadronicProcessPool;
  }
} // namespace corsika

namespace corsika::kokkos::application {
  class CascadeRunSession;

  /** Internal prefix/worker CLI state. No physics or scheduling policy is selected here.
   */
  struct CascadeWorkerOptions {
    std::string frontierOut_, frontierIn_;
    double frontierMaximumEnergy_ = 0.;
    unsigned workerId_ = 0;
    bool legacyImport_ = false;
    std::string cpuInput_, cpuOutput_, radioOutput_;
    double cpuMaximumEnergy_ = std::numeric_limits<double>::infinity();
    std::uint64_t historyLimit_ = 0;

    CascadeWorkerOptions();
    void addOptions(CLI::App& app, BackendOptions& backend);
    bool prepare(CLI::App& app, BackendOptions& backend, DeviceSelection const& devices,
                 int nevent, bool hasEnergyRange, bool forceInteraction, bool forceDecay,
                 char const* executable);
  };

  /** Dispatch before the main shower stack frame, models, RNG or CUDA are initialized. */
  int dispatch_cascade(int argc, char** argv,
                       int (*run)(int, char**, DeviceSelection const&),
                       CascadeScene scene = CascadeScene::Atmosphere);

  /** Own run-lifetime backend resources, including the native-only empty implementation.
   * This facade does not start workers or alter the existing lazy initialization.
   */
  class CascadeRuntime {
  public:
    CascadeRuntime();
    ~CascadeRuntime();
    CascadeRuntime(CascadeRuntime const&) = delete;
    CascadeRuntime& operator=(CascadeRuntime const&) = delete;
    void configure(BackendOptions const&, RunEnvironmentConfig const&);
    void registerOutput(OutputManager&);
    HadronicProcessPool* hadronicPool() const;
    void recordIncomplete(unsigned int event, std::string const& message);
    CascadeRunSession& session();

  private:
    struct State;
    std::unique_ptr<State> state_;
  };

  /** The accelerated runner already reported and finalized an incomplete event. */
  struct CascadeAborted {};

  template <typename TCut>
  CascadeEventConfig make_cascade_defaults(CLI::App const& app, std::uint64_t seed,
                                           HEPEnergyType maximum_energy, Code primary,
                                           TCut const& cut,
                                           HEPEnergyType production_threshold,
                                           HEPEnergyType hadronic_threshold,
                                           bool keep_zero_weight);

  template <typename TE, typename TC, typename TP, typename TS, typename TStep,
            typename TAxis, typename TDepth>
  auto make_cascade_geometry(TE& environment, TC const& coordinates, TP const& injection,
                             TS const& surface, TStep const& step, TAxis const& axis,
                             TDepth const& depth, RunEnvironmentConfig const& run);

} // namespace corsika::kokkos::application

#include <corsika/detail/kokkos/application/cascade/CascadeSupport.inl>

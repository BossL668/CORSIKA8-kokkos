/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include <corsika/framework/core/Logging.hpp>

#include <corsika/kokkos/application/cascade/CascadeSupport.hpp>
#include <corsika/kokkos/application/BackendCli.hpp>
#include <corsika/output/OutputManager.hpp>
#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <stdexcept>
#include <vector>

#ifdef CORSIKA_KOKKOS_COORDINATOR
#include <corsika/kokkos/scheduling/AirShowerCoordinator.hpp>
#endif
#if defined(CORSIKA_KOKKOS_STATIC_MULTIGPU) || defined(CORSIKA_KOKKOS_COORDINATOR)
#include <corsika/kokkos/scheduling/StaticFrontier.hpp>
#endif
#ifdef CORSIKA8_WITH_KOKKOS_EM
#include <corsika/kokkos/application/cascade/CascadeRunSession.hpp>
#include <corsika/kokkos/scheduling/HadronicProcessPool.hpp>
#endif

namespace corsika::kokkos::application {

  CascadeWorkerOptions::CascadeWorkerOptions() {
#ifdef CORSIKA_KOKKOS_STATIC_MULTIGPU
    historyLimit_ = kokkos::frontier::HistoryStride;
#endif
  }

  void CascadeWorkerOptions::addOptions([[maybe_unused]] CLI::App& app,
                                        [[maybe_unused]] BackendOptions& backend) {
#ifdef CORSIKA_KOKKOS_STATIC_MULTIGPU
    app.add_option("--prefix-cpu-in", cpuInput_,
                   "Import one statically assigned CPU subshower shard");
    app.add_option("--prefix-cpu-out", cpuOutput_,
                   "Capture all secondary species for static CPU splitting");
    app.add_option("--prefix-cpu-max-energy", cpuMaximumEnergy_,
                   "CPU split energy cap, not a physics cut")
        ->check(CLI::PositiveNumber);
    app.add_option("--frontier-history-limit", historyLimit_,
                   "Exclusive input-history namespace ceiling");
    app.add_flag("--frontier-legacy-import", legacyImport_,
                 "Diagnostic control: import the entire frontier into the scalar stack");
    app.add_option("--frontier-out", frontierOut_,
                   "Capture independent EM subshower roots");
    app.add_option("--frontier-in", frontierIn_,
                   "Import roots of the SAME global primary");
    app.add_option("--frontier-max-energy", frontierMaximumEnergy_,
                   "Root scheduling cap in GeV (not a cut)");
    app.add_option("--frontier-worker-id", workerId_,
                   "Exclusive child history namespace [1,255]");
    app.add_option("--radio-track-out", radioOutput_,
                   "Internal CPU-prefix radio track output")
        ->group("");
    app.add_option("--radio-track-in", backend.radioTrackInputs_,
                   "Internal GPU-worker radio track inputs")
        ->group("");

#endif
  }

  bool CascadeWorkerOptions::prepare(CLI::App& app, BackendOptions& backend,
                                     DeviceSelection const& devices,
                                     [[maybe_unused]] int nevent,
                                     [[maybe_unused]] bool hasEnergyRange,
                                     [[maybe_unused]] bool forceInteraction,
                                     [[maybe_unused]] bool forceDecay,
                                     char const* executable) {
    if (!corsika::kokkos::application::prepare_device_cli(app, backend, devices))
      return false;
    if (!corsika::kokkos::application::prepare_egs4_cli(app, backend)) return false;
#ifdef CORSIKA_KOKKOS_STATIC_MULTIGPU
    if ((!radioOutput_.empty() && (nevent != 1 || backend.emBackend_ != "proposal" ||
                                   backend.radioBackend_ != "cpu" ||
                                   (frontierOut_.empty() && cpuOutput_.empty()) ||
                                   !backend.radioTrackInputs_.empty())) ||
        (!backend.radioTrackInputs_.empty() &&
         (nevent != 1 || backend.emBackend_ != "kokkos-proposal" ||
          backend.radioBackend_ != "kokkos" || frontierIn_.empty()))) {
      CORSIKA_LOG_CRITICAL(
          "Radio track handoff requires a single CPU prefix or a Kokkos PROPOSAL "
          "frontier "
          "worker");
      return false;
    }
#ifndef CORSIKA8_WITH_KOKKOS_EM
    if (!radioOutput_.empty() || !backend.radioTrackInputs_.empty()) {
      CORSIKA_LOG_CRITICAL("Radio track handoff requires a Kokkos-enabled worker build");
      return false;
    }
#endif
    std::vector<std::filesystem::path> radio_input_paths;
    for (auto const& path : backend.radioTrackInputs_) {
      auto const canonical = std::filesystem::canonical(path);
      if (std::find(radio_input_paths.begin(), radio_input_paths.end(), canonical) !=
          radio_input_paths.end()) {
        CORSIKA_LOG_CRITICAL("Duplicate radio track handoff input: {}", path);
        return false;
      }
      radio_input_paths.push_back(canonical);
    }
    if ((!cpuInput_.empty() || !cpuOutput_.empty()) &&
        (nevent != 1 || hasEnergyRange || backend.emBackend_ != "proposal" ||
         forceInteraction || forceDecay || !frontierIn_.empty() || legacyImport_ ||
         (!cpuOutput_.empty() && (!cpuInput_.empty() || !frontierOut_.empty())) ||
         (!cpuInput_.empty() &&
          (frontierOut_.empty() || workerId_ == 0 ||
           historyLimit_ <= (workerId_ + 1ULL) * kokkos::frontier::HistoryStride - 1)))) {
      CORSIKA_LOG_CRITICAL(
          "CPU prefix splitting requires N=1, scalar PROPOSAL, no forced vertex, and "
          "disjoint history namespaces");
      return false;
    }
    if ((!frontierIn_.empty() && !legacyImport_ &&
         backend.emBackend_ != "kokkos-proposal" &&
         backend.emBackend_ != "kokkos-egs4") ||
        (legacyImport_ && frontierIn_.empty())) {
      CORSIKA_LOG_CRITICAL(
          "Buffered frontier input requires the Kokkos worker; legacy control requires "
          "--frontier-in");
      return false;
    }
    if ((!frontierOut_.empty() || !frontierIn_.empty()) &&
        (nevent != 1 || (!frontierOut_.empty() && !frontierIn_.empty()) ||
         hasEnergyRange ||
         (!frontierOut_.empty() &&
          ((backend.emBackend_ != "proposal" && backend.emBackend_ != "kokkos-egs4") ||
           frontierMaximumEnergy_ <= 0.)) ||
         (!frontierIn_.empty() && (forceInteraction || forceDecay || workerId_ == 0)))) {
      CORSIKA_LOG_CRITICAL(
          "Static handoff requires N=1, fixed energy, exclusive input/output; capture "
          "uses "
          "PROPOSAL and a positive cap; workers must not force the primary again");
      return false;
    }
#endif
    if ((backend.kokkosExecution_ == "cuda-openmp" ||
         backend.kokkosExecution_ == "openmp-cuda") &&
        app.count("--hadronic-workers") == 0)
      backend.hadronicWorkers_ = 1;

    if (!kokkos::application::validate_backend_cli_options(backend, executable)) {
      return false;
    }

    return true;
  }

  int dispatch_cascade(int argc, char** argv,
                       int (*run)(int, char**, DeviceSelection const&),
                       [[maybe_unused]] CascadeScene scene) {

    corsika::kokkos::application::DeviceSelection selected_devices;
    try {
      selected_devices = corsika::kokkos::application::parse_devices(argc, argv);
    } catch (std::exception const& error) {
      CORSIKA_LOG_ERROR("GPU selection: {}", error.what());
      return EXIT_FAILURE;
    }
#ifdef CORSIKA_KOKKOS_COORDINATOR
    // Dispatch before initializing Kokkos or any shower/Fortran model state.
    if (kokkos::coordinator::requested(argc, argv))
      return kokkos::coordinator::main_entry(argc, argv);
#endif
    try {
      return run(argc, argv, selected_devices);
    } catch (CascadeAborted const&) { return EXIT_FAILURE; }
  }

  struct CascadeRuntime::State {
#ifdef CORSIKA8_WITH_KOKKOS_EM
    std::unique_ptr<HadronicProcessPool> pool_;
    std::unique_ptr<CascadeRunSession> session_;
#endif
  };

  CascadeRuntime::CascadeRuntime()
      : state_(std::make_unique<State>()) {}
  CascadeRuntime::~CascadeRuntime() = default;

  void CascadeRuntime::configure(
      [[maybe_unused]] BackendOptions const& options,
      [[maybe_unused]] RunEnvironmentConfig const& environment) {
#ifdef CORSIKA8_WITH_KOKKOS_EM
    state_->session_ = std::make_unique<CascadeRunSession>(options, environment);
#endif
  }

  void CascadeRuntime::registerOutput([[maybe_unused]] OutputManager& output) {
#ifdef CORSIKA8_WITH_KOKKOS_EM
    if (state_->session_->getRunOutput())
      output.add("gpu_em", *state_->session_->getRunOutput());
#endif
  }

  HadronicProcessPool* CascadeRuntime::hadronicPool() const {
#ifdef CORSIKA8_WITH_KOKKOS_EM
    return state_->pool_.get();
#else
    return nullptr;
#endif
  }

  void CascadeRuntime::recordIncomplete([[maybe_unused]] unsigned int event,
                                        [[maybe_unused]] std::string const& message) {
#ifdef CORSIKA8_WITH_KOKKOS_EM
    if (state_->session_->getRunOutput())
      state_->session_->getRunOutput()->recordIncomplete(event, message);
#endif
  }

  CascadeRunSession& CascadeRuntime::session() {
#ifdef CORSIKA8_WITH_KOKKOS_EM
    return *state_->session_;
#else
    throw std::logic_error("No accelerated run session in this build");
#endif
  }
} // namespace corsika::kokkos::application

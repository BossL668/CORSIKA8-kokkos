#pragma once
#include "GpuCliOptions.hpp"
#include <CLI/CLI.hpp>
#include <corsika/framework/core/Logging.hpp>
#ifdef CORSIKA8_WITH_NATIVE_EGS4
#include <corsika/accelerator/em/KokkosRuntime.hpp>
#endif

namespace corsika::applications::air_shower {
inline void addEgs4CliOptions(CLI::App& app,GpuCliOptions& o) {
  app.add_option("--egs4-stepfc",o.egs4_stepfc,
    "C7-EGS4 charged-particle step factor (default: 1)")
    ->check(CLI::PositiveNumber)->group("EGS4 EM");
}
inline bool prepareEgs4Cli(CLI::App const& app,GpuCliOptions& o) {
  if(o.em_backend!="egs4") {
    if(app.count("--egs4-stepfc")) {
      CORSIKA_LOG_CRITICAL("--egs4-stepfc requires --em-backend egs4");return false;
    }
    return true;
  }
  if(!app.count("--radio-backend"))o.radio_backend="kokkos";
  if(app.count("--pdg")&&(app.count("--force-interaction")||app.count("--force-decay"))) {
    int id=app["--pdg"]->as<int>();
    if(id==11||id==-11||id==22) {
      CORSIKA_LOG_CRITICAL("Forced scalar primary actions are not supported for EGS4 EM primaries");return false;
    }
  }
  return true;
}
inline bool validateEgs4Execution(GpuCliOptions const& o) {
  if(o.em_backend!="egs4")return true;
#ifndef CORSIKA8_WITH_NATIVE_EGS4
  CORSIKA_LOG_CRITICAL("EGS4 is not compiled; build with CORSIKA_ENABLE_EGS4=ON");return false;
#else
  if(o.hadronic_backend!="scalar"||!o.cuda_replay_trace.empty()) {
    CORSIKA_LOG_CRITICAL("EGS4 requires scalar hadronic scheduling and does not support PROPOSAL replay traces");return false;
  }
  try {
    auto execution=accelerator::em::resolveKokkosExecutionBackend(o.kokkos_execution);
    if(execution!="cuda"&&execution!="openmp")
      throw std::invalid_argument("EGS4 supports cuda or openmp, not cooperative execution");
    if(execution=="cuda"&&o.kokkos_num_threads>1)
      throw std::invalid_argument("CUDA uses one host scheduling thread");
  }catch(std::exception const& error){CORSIKA_LOG_CRITICAL("{}",error.what());return false;}
  return true;
#endif
}
}

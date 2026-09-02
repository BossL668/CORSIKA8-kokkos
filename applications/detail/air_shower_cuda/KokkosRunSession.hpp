/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include "AcceleratedRunEnvironmentConfig.hpp"
#include "GpuCliOptions.hpp"

#include <corsika/accelerator/em/detail/KokkosEmRunSession.hpp>
#include <corsika/gpu/em/GpuEmRunOutput.hpp>
#include <corsika/gpu/em/tables/RateTable.hpp>

#include <memory>

namespace corsika::applications::air_shower {

  /** Application-owned YAML/lifecycle wrapper for the portable backend. */
  class KokkosRunSession {
  public:
    KokkosRunSession(GpuCliOptions const&,
                     AcceleratedRunEnvironmentConfig const&);

    bool enabled() const noexcept { return run_output_ != nullptr; }
    gpu::em::GpuEmRunOutput* runOutput() noexcept {
      return run_output_.get();
    }
    gpu::em::tables::RateTableSet const* loadedRateTable() const noexcept {
      return nullptr;
    }
    bool hasBackend() const noexcept {
      return runtime_session_ && runtime_session_->initialized();
    }
    accelerator::em::detail::KokkosEmRunSession& runtimeSession();

  private:
    std::unique_ptr<accelerator::em::detail::KokkosEmRunSession>
        runtime_session_{};
    std::unique_ptr<gpu::em::GpuEmRunOutput> run_output_{};
  };

} // namespace corsika::applications::air_shower

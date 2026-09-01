/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include "GpuCliOptions.hpp"

#include <corsika/gpu/em/CudaEmBackend.hpp>
#include <corsika/gpu/em/GpuEmRunOutput.hpp>
#include <corsika/gpu/em/tables/RateTable.hpp>

#include <array>
#include <memory>
#include <optional>
#include <string>

namespace corsika::applications::air_shower {

  struct CudaRunEnvironmentConfig {
    std::string geomagnetic_model;
    double geomagnetic_year{};
    double latitude_deg{};
    double longitude_deg{};
    double altitude_m{};
    std::array<double, 3> magnetic_field_T{};
    double maximum_magnetic_deflection_rad{};
    std::array<double, 3> observation_plane_point_m{};
    std::string antenna_file;
  };

  /** Owns CUDA objects whose lifetime spans every shower in one library. */
  class CudaRunSession {
  public:
    CudaRunSession(GpuCliOptions const& options,
                   CudaRunEnvironmentConfig const& environment);

    bool enabled() const noexcept { return run_output_ != nullptr; }

    gpu::em::GpuEmRunOutput* runOutput() noexcept {
      return run_output_.get();
    }

    gpu::em::tables::RateTableSet const* loadedRateTable() const noexcept {
      return loaded_gpu_table_ ? &*loaded_gpu_table_ : nullptr;
    }

    bool hasBackend() const noexcept { return backend_ != nullptr; }

    gpu::em::CudaEmBackend& createBackend();
    gpu::em::CudaEmBackend& backend();

  private:
    std::optional<gpu::em::tables::RateTableSet> loaded_gpu_table_;
    std::unique_ptr<gpu::em::GpuEmRunOutput> run_output_;
    std::unique_ptr<gpu::em::CudaEmBackend> backend_;
  };

} // namespace corsika::applications::air_shower

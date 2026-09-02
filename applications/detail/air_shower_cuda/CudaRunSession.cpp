/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include "CudaRunSession.hpp"

#include <cuda_runtime_api.h>
#include <CubicInterpolation/version.h>
#include <PROPOSAL/version.h>

#include <corsika/gpu/em/tables/ProposalNativeAux.hpp>
#include <corsika/gpu/em/tables/Sha256.hpp>

#include <cstdint>
#include <stdexcept>
#include <utility>

namespace corsika::applications::air_shower {

  namespace {

    std::string cudaVersionString(int const version) {
      return std::to_string(version / 1000) + "." +
             std::to_string((version % 1000) / 10);
    }

  } // namespace

  CudaRunSession::CudaRunSession(
      GpuCliOptions const& options,
      CudaRunEnvironmentConfig const& environment) {
    if (options.em_backend != "cuda") { return; }

    using namespace corsika::gpu::em::tables;
    auto const physics_source =
        options.gpu_physics_source == "proposal-native"
            ? gpu::em::GpuPhysicsSource::ProposalNative
            : gpu::em::GpuPhysicsSource::C8EmRt;
    auto const auxiliary_cache =
        options.gpu_aux_cache_dir.empty()
            ? defaultProposalNativeAuxCacheDirectory()
            : options.gpu_aux_cache_dir;
    runtime_session_ =
        std::make_unique<gpu::em::detail::CudaEmRunSession>(
            physics_source, options.gpu_table_cache, auxiliary_cache);

    cudaDeviceProp properties{};
    auto const property_status =
        cudaGetDeviceProperties(&properties, options.gpu_device);
    if (property_status != cudaSuccess) {
      throw std::runtime_error(
          std::string("cannot query requested CUDA device: ") +
          cudaGetErrorString(property_status));
    }
    int driver_version = 0;
    int runtime_version = 0;
    auto const driver_status = cudaDriverGetVersion(&driver_version);
    auto const runtime_status = cudaRuntimeGetVersion(&runtime_version);
    if (driver_status != cudaSuccess || runtime_status != cudaSuccess) {
      throw std::runtime_error("cannot query CUDA driver/runtime version");
    }

    YAML::Node configuration;
    configuration["backend"] = "CORSIKA8GpuEm";
    configuration["backend_version"] =
        "cuda-em-v3-configurable-magnetic-step";
    configuration["gpu_physics_source"] = options.gpu_physics_source;
    configuration["device"]["index"] = options.gpu_device;
    configuration["device"]["name"] = properties.name;
    configuration["device"]["compute_capability"] =
        std::to_string(properties.major) + "." +
        std::to_string(properties.minor);
    configuration["device"]["total_memory_bytes"] =
        static_cast<std::uint64_t>(properties.totalGlobalMem);
    configuration["cuda"]["driver_version"] =
        cudaVersionString(driver_version);
    configuration["cuda"]["runtime_version"] =
        cudaVersionString(runtime_version);
    configuration["deterministic"] = options.gpu_deterministic;
    configuration["detailed_stage_timing"] =
        options.gpu_detailed_stage_timing;
    configuration["radio_backend"] = options.radio_backend;
    configuration["radio_fixed_point_field_limit_V_per_m"] =
        options.gpu_radio_field_limit;
    configuration["radio_track_diagnostics"] =
        options.gpu_radio_track_diagnostics;
    configuration["minimum_batch_size"] =
        static_cast<std::uint64_t>(options.gpu_min_batch);
    configuration["memory_fraction"] = options.gpu_memory_fraction;
    configuration["accepted_table_tolerance"] =
        options.gpu_table_tolerance;
    configuration["environment"]["geomagnetic_model"] =
        environment.geomagnetic_model;
    configuration["environment"]["geomagnetic_year"] =
        environment.geomagnetic_year;
    configuration["environment"]["latitude_deg"] =
        environment.latitude_deg;
    configuration["environment"]["longitude_deg"] =
        environment.longitude_deg;
    configuration["environment"]["altitude_m"] = environment.altitude_m;
    configuration["environment"]["magnetic_field_T"]["x"] =
        environment.magnetic_field_T[0];
    configuration["environment"]["magnetic_field_T"]["y"] =
        environment.magnetic_field_T[1];
    configuration["environment"]["magnetic_field_T"]["z"] =
        environment.magnetic_field_T[2];
    configuration["environment"]["maximum_magnetic_deflection_rad"] =
        environment.maximum_magnetic_deflection_rad;
    configuration["environment"]["observation_geometry"] = "plane";
    configuration["environment"]["observation_plane_point_m"]["x"] =
        environment.observation_plane_point_m[0];
    configuration["environment"]["observation_plane_point_m"]["y"] =
        environment.observation_plane_point_m[1];
    configuration["environment"]["observation_plane_point_m"]["z"] =
        environment.observation_plane_point_m[2];
    configuration["environment"]["observation_plane_normal"]["x"] = 0.;
    configuration["environment"]["observation_plane_normal"]["y"] = 0.;
    configuration["environment"]["observation_plane_normal"]["z"] = 1.;
    configuration["environment"]["antenna_file"] = environment.antenna_file;
    if (auto const* loaded_gpu_table = loadedRateTable()) {
      auto const& table = *loaded_gpu_table;
      configuration["table"]["path"] = options.gpu_table_cache.string();
      configuration["table"]["format_version"] = RateTableFormatVersion;
      configuration["table"]["sha256"] = toHex(table.content_hash);
      configuration["table"]["proposal_version"] =
          table.metadata.proposal_version;
      configuration["table"]["generator_version"] =
          table.metadata.generator_version;
      configuration["table"]["medium"] = table.metadata.medium_name;
      configuration["table"]["stochastic_cut_MeV"] =
          table.metadata.energy_cut_MeV;
      configuration["table"]["energy_min_MeV"] =
          table.metadata.energy_min_MeV;
      configuration["table"]["energy_max_MeV"] =
          table.metadata.energy_max_MeV;
      configuration["table"]["measured_rate_error"] =
          table.metadata.measured_max_relative_error;
      configuration["table"]["measured_inverse_cdf_error"] =
          table.metadata.measured_max_loss_relative_error;
    } else {
      configuration["table"]["format"] = "PROPOSAL native Hermite";
      configuration["table"]["proposal_version"] = getPROPOSALVersion();
      configuration["table"]["cubic_interpolation_version"] =
          getCubicInterpolationVersion();
      configuration["table"]["aux_cache_directory"] =
          (options.gpu_aux_cache_dir.empty()
               ? defaultProposalNativeAuxCacheDirectory()
               : options.gpu_aux_cache_dir)
              .string();
    }
    run_output_ = std::make_unique<gpu::em::GpuEmRunOutput>(
        std::move(configuration));
  }

  gpu::em::detail::CudaEmRunSession& CudaRunSession::runtimeSession() {
    if (!runtime_session_) {
      throw std::logic_error("CUDA EM run session is disabled");
    }
    return *runtime_session_;
  }

} // namespace corsika::applications::air_shower

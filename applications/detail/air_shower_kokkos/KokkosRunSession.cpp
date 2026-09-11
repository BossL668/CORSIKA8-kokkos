/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include "KokkosRunSession.hpp"

#include <CubicInterpolation/version.h>
#include <PROPOSAL/version.h>

#include <corsika/accelerator/em/common/tables/ProposalNativeAux.hpp>
#include <corsika/accelerator/em/RandomDomains.hpp>

#include <stdexcept>
#include <utility>

namespace corsika::applications::air_shower {

  KokkosRunSession::KokkosRunSession(
      GpuCliOptions const& options,
      AcceleratedRunEnvironmentConfig const& environment) {
    if (options.em_backend != "kokkos") return;
    if (options.gpu_physics_source != "proposal-native")
      throw std::invalid_argument(
          "Kokkos EM requires --gpu-physics-source proposal-native");
    auto const auxiliary_cache =
        options.gpu_aux_cache_dir.empty()
            ? gpu::em::tables::defaultProposalNativeAuxCacheDirectory()
            : options.gpu_aux_cache_dir;
    accelerator::em::KokkosRuntimeConfig runtime_config{};
    runtime_config.device = options.kokkos_device;
    runtime_config.threads = options.kokkos_num_threads;
    runtime_config.execution_backend = options.kokkos_execution;
    runtime_config.cooperative_policy = options.kokkos_cooperative_policy;
    runtime_config.tuning_cache = options.kokkos_tuning_cache;
    runtime_config.require_tuning = options.kokkos_require_tuning;
    runtime_session_ =
        std::make_unique<accelerator::em::detail::KokkosEmRunSession>(
            runtime_config, auxiliary_cache);

    YAML::Node configuration;
    configuration["backend"] = "CORSIKA8KokkosEm";
    configuration["backend_version"] = "kokkos-em-v2-cpu-alignment-rng2";
    configuration["rng_domain_version"] = gpu::em::RandomDomainVersion;
    configuration["physics_alignment_revision"] = "2026-09-06";
    configuration["gpu_physics_source"] = "proposal-native";
    if (options.kokkos_cooperative_policy != "legacy")
      configuration["cooperative_policy"] = options.kokkos_cooperative_policy;
#if defined(CORSIKA8_KOKKOS_BACKEND_CUDA_OPENMP)
    configuration["execution_space"] =
        accelerator::em::resolveKokkosExecutionBackend(options.kokkos_execution) ==
                "openmp" ? "OpenMP" :
        (options.kokkos_execution == "cuda-openmp" ? "CUDA+OpenMP (cooperative)" : "CUDA");
    configuration["compiled_execution_spaces"] = "CUDA,OpenMP";
    configuration["experimental_dual_runtime"] = true;
#elif defined(CORSIKA8_KOKKOS_BACKEND_OPENMP)
    configuration["execution_space"] = "OpenMP";
#elif defined(CORSIKA8_KOKKOS_BACKEND_CUDA)
    configuration["execution_space"] = "CUDA";
#elif defined(CORSIKA8_KOKKOS_BACKEND_HIP)
    configuration["execution_space"] = "HIP";
#elif defined(CORSIKA8_KOKKOS_BACKEND_SYCL)
    configuration["execution_space"] = "SYCL";
#endif
#ifndef CORSIKA8_KOKKOS_VERSION
#error "CORSIKA8_KOKKOS_VERSION must be provided by the Kokkos build"
#endif
    configuration["kokkos"]["version"] = CORSIKA8_KOKKOS_VERSION;
    configuration["kokkos"]["device"] = options.kokkos_device;
    configuration["kokkos"]["threads"] = options.kokkos_num_threads;
    configuration["kokkos"]["tuning_cache"] =
        options.kokkos_tuning_cache.string();
    configuration["kokkos"]["require_tuning"] =
        options.kokkos_require_tuning;
    configuration["deterministic"] = options.gpu_deterministic;
    configuration["radio_backend"] = options.radio_backend;
    configuration["minimum_batch_size"] =
        static_cast<std::uint64_t>(options.gpu_min_batch);
    configuration["requested_resident_batch_limit"] =
        static_cast<std::uint64_t>(options.gpu_resident_batch_limit);
    configuration["memory_fraction"] = options.gpu_memory_fraction;
    configuration["table"]["format"] = "PROPOSAL native Hermite";
    configuration["table"]["proposal_version"] = getPROPOSALVersion();
    configuration["table"]["cubic_interpolation_version"] =
        getCubicInterpolationVersion();
    configuration["table"]["aux_cache_directory"] =
        auxiliary_cache.string();
    configuration["environment"]["geomagnetic_model"] =
        environment.geomagnetic_model;
    configuration["environment"]["geomagnetic_year"] =
        environment.geomagnetic_year;
    configuration["environment"]["latitude_deg"] = environment.latitude_deg;
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
    configuration["environment"]["antenna_file"] = environment.antenna_file;
    run_output_ =
        std::make_unique<gpu::em::GpuEmRunOutput>(std::move(configuration));
  }

  accelerator::em::detail::KokkosEmRunSession&
  KokkosRunSession::runtimeSession() {
    if (!runtime_session_)
      throw std::logic_error("Kokkos EM run session is disabled");
    return *runtime_session_;
  }

} // namespace corsika::applications::air_shower

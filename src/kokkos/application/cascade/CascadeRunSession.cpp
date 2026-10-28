/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include <corsika/kokkos/application/cascade/CascadeRunSession.hpp>

#include <CubicInterpolation/version.h>
#include <PROPOSAL/version.h>

#include <corsika/kokkos/em/proposal/ProposalNativeAux.hpp>
#include <corsika/kokkos/em/random/RandomDomains.hpp>
#include <corsika/kokkos/runtime/BackendRuntime.hpp>

#include <stdexcept>
#include <utility>

namespace corsika::kokkos::application {

  CascadeRunSession::CascadeRunSession(BackendOptions const& options,
                                       RunEnvironmentConfig const& environment) {
    if (options.emBackend_ != "kokkos-proposal") return;
    if (options.gpuPhysicsSource_ != "proposal-native")
      throw std::invalid_argument(
          "Kokkos EM requires --gpu-physics-source proposal-native");
    auto const auxiliaryCache =
        options.gpuAuxCacheDir_.empty()
            ? corsika::kokkos::default_proposal_native_aux_cache_directory()
            : options.gpuAuxCacheDir_;
    corsika::kokkos::RuntimeConfig runtimeConfig{};
    runtimeConfig.device_ = options.kokkosDevice_;
    runtimeConfig.threads_ = options.kokkosNumThreads_;
    runtimeConfig.executionBackend_ = options.kokkosExecution_;
    runtimeConfig.cooperativePolicy_ = options.kokkosCooperativePolicy_;
    runtimeConfig.tuningCache_ = options.kokkosTuningCache_;
    runtimeConfig.requireTuning_ = options.kokkosRequireTuning_;
    runtimeSession_ =
        std::make_unique<corsika::kokkos::EmRunSession>(runtimeConfig, auxiliaryCache);

    YAML::Node configuration;
    configuration["backend"] = "CORSIKA8KokkosEm";
    configuration["backend_version"] = "kokkos-em-v2-cpu-alignment-rng2";
    configuration["rng_domain_version"] = corsika::kokkos::RandomDomainVersion;
    configuration["physics_alignment_revision"] = "2026-09-06";
    configuration["gpu_physics_source"] = "proposal-native";
    if (options.kokkosCooperativePolicy_ != "legacy")
      configuration["cooperative_policy"] = options.kokkosCooperativePolicy_;
#if defined(CORSIKA_KOKKOS_BACKEND_CUDA_OPENMP)
    configuration["execution_space"] =
        corsika::kokkos::resolve_execution_backend(options.kokkosExecution_) == "openmp"
            ? "OpenMP"
            : (options.kokkosExecution_ == "cuda-openmp"
                   ? "CUDA+OpenMP (cooperative)"
                   : options.kokkosExecution_ == "openmp-cuda"
                         ? "OpenMP+CUDA (CPU primary)"
                         : "CUDA");
    configuration["compiled_execution_spaces"] = "CUDA,OpenMP";
    configuration["experimental_dual_runtime"] = true;
#elif defined(CORSIKA_KOKKOS_BACKEND_OPENMP)
    configuration["execution_space"] = "OpenMP";
#elif defined(CORSIKA_KOKKOS_BACKEND_CUDA)
    configuration["execution_space"] = "CUDA";
#elif defined(CORSIKA_KOKKOS_BACKEND_HIP)
    configuration["execution_space"] = "HIP";
#elif defined(CORSIKA_KOKKOS_BACKEND_SYCL)
    configuration["execution_space"] = "SYCL";
#endif
#ifndef CORSIKA_KOKKOS_VERSION
#error "CORSIKA_KOKKOS_VERSION must be provided by the Kokkos build"
#endif
    configuration["kokkos"]["version"] = CORSIKA_KOKKOS_VERSION;
    configuration["kokkos"]["device"] = options.kokkosDevice_;
    configuration["kokkos"]["threads"] = options.kokkosNumThreads_;
    configuration["kokkos"]["tuning_cache"] = options.kokkosTuningCache_.string();
    configuration["kokkos"]["require_tuning"] = options.kokkosRequireTuning_;
    configuration["deterministic"] = options.gpuDeterministic_;
    configuration["radio_backend"] = options.radioBackend_;
    configuration["minimum_batch_size"] =
        static_cast<std::uint64_t>(options.gpuMinBatch_);
    configuration["requested_resident_batch_limit"] =
        static_cast<std::uint64_t>(options.gpuResidentBatchLimit_);
    configuration["memory_fraction"] = options.gpuMemoryFraction_;
    configuration["table"]["format"] = "PROPOSAL native Hermite";
    configuration["table"]["proposal_version"] = getPROPOSALVersion();
    configuration["table"]["cubic_interpolation_version"] =
        getCubicInterpolationVersion();
    configuration["table"]["aux_cache_directory"] = auxiliaryCache.string();
    configuration["environment"]["geomagnetic_model"] = environment.geomagneticModel_;
    configuration["environment"]["geomagnetic_year"] = environment.geomagneticYear_;
    configuration["environment"]["latitude_deg"] = environment.latitudeDeg_;
    configuration["environment"]["longitude_deg"] = environment.longitudeDeg_;
    configuration["environment"]["altitude_m"] = environment.altitudeM_;
    if (environment.scene_ == CascadeScene::HomogeneousIce) {
      configuration["environment"]["scene"] = "homogeneous-south-pole-ice";
      configuration["environment"]["coordinates"] = "earth-centred-polar-NWU";
      configuration["environment"]["surface_role"] = "depth-reference";
    } else if (environment.scene_ == CascadeScene::Firn) {
      configuration["environment"]["scene"] = "exponential-south-pole-firn";
      configuration["environment"]["coordinates"] = "earth-centred-polar-NWU";
      configuration["environment"]["radio_coordinates"] = "surface-centred-NWU";
      configuration["environment"]["surface_role"] = "pure-firn-domain-boundary";
    }
    configuration["environment"]["magnetic_field_T"]["x"] =
        environment.magneticFieldT_[0];
    configuration["environment"]["magnetic_field_T"]["y"] =
        environment.magneticFieldT_[1];
    configuration["environment"]["magnetic_field_T"]["z"] =
        environment.magneticFieldT_[2];
    configuration["environment"]["maximum_magnetic_deflection_rad"] =
        environment.maximumMagneticDeflectionRad_;
    configuration["environment"]["observation_geometry"] = "plane";
    configuration["environment"]["geometry"] = "native-node-tree";
    configuration["environment"]["boundary_search"] = "shared-c8-ordered-tree";
    configuration["environment"]["sphere_solver"] = "stable-quartic-v2";
    configuration["environment"]["antenna_file"] = environment.antennaFile_;
    runOutput_ = std::make_unique<corsika::kokkos::EmRunOutput>(std::move(configuration));
  }

  corsika::kokkos::EmRunSession& CascadeRunSession::getRuntimeSession() {
    if (!runtimeSession_) throw std::logic_error("Kokkos EM run session is disabled");
    return *runtimeSession_;
  }

} // namespace corsika::kokkos::application

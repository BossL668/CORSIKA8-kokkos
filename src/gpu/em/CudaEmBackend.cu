/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include <cuda_runtime.h>

#include <cub/device/device_scan.cuh>
#include <cub/device/device_select.cuh>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include <corsika/gpu/em/CudaEmBackend.hpp>
#include <corsika/gpu/em/CudaInteractionSelector.hpp>
#include <corsika/gpu/em/CudaLeptonTransport.hpp>
#include <corsika/gpu/em/CudaLeptonSelectionTransport.hpp>
#include <corsika/gpu/em/CudaLeptonVertexSelector.hpp>
#include <corsika/gpu/em/CudaPhotonPairFinalState.hpp>
#include <corsika/gpu/em/CudaPhotonSelectionTransport.hpp>
#include <corsika/gpu/em/CudaPhotonTransport.hpp>
#include <corsika/gpu/em/BremsLpm.hpp>
#include <corsika/gpu/em/CudaBremsFinalState.hpp>
#include <corsika/gpu/em/MoliereScattering.hpp>
#include <corsika/gpu/em/Philox.hpp>
#include <corsika/gpu/em/ProcessCapabilities.hpp>
#include <corsika/gpu/em/SphericalAtmosphere.hpp>
#include <corsika/gpu/em/detail/DeviceBatchStages.hpp>
#include <corsika/gpu/em/detail/DeviceWavefrontBucketing.hpp>
#include <corsika/gpu/em/detail/ProfileProjection.hpp>
#include <corsika/gpu/em/tables/CudaRateTable.hpp>
#include <corsika/gpu/em/tables/RateTable.hpp>
#include <corsika/gpu/radio/CudaRadioAccumulator.hpp>

namespace corsika::gpu::em {

  namespace {

    constexpr std::uint32_t ToyBranchingProcessId = 0x544f5901U;
    constexpr double ToyEnergyCutGeV = 5.e-4;
    constexpr unsigned int ThreadsPerBlock = 256;
    constexpr std::size_t LeptonPipelineStageCount = 13;
    constexpr std::size_t LeptonSelectionStage = 0;
    constexpr std::size_t LeptonTransportPhysicsStage = 1;
    constexpr std::size_t LeptonMoliereStage = 2;
    constexpr std::size_t LeptonTransportControlStage = 3;
    constexpr std::size_t LeptonTransportStage = 4;
    constexpr std::size_t LeptonInteractionExtractionStage = 5;
    constexpr std::size_t LeptonVertexSelectionStage = 6;
    constexpr std::size_t
        LeptonFinalStateClassificationStage = 7;
    constexpr std::size_t LeptonFinalStateScanStage = 8;
    constexpr std::size_t LeptonFinalStateSummaryStage = 9;
    constexpr std::size_t LeptonFinalStateWriteStage = 10;
    constexpr std::size_t LeptonFinalStateStage = 11;
    constexpr std::size_t LeptonEndpointCompactionStage = 12;

    struct IsChargedEmParticle {
      __host__ __device__ bool operator()(
          EmParticleState const& particle) const {
        return isChargedLeptonPid(particle.pid);
      }
    };

    void checkCuda(cudaError_t status, char const* operation) {
      if (status == cudaSuccess) {
        return;
      }
      std::ostringstream message;
      message << operation << " failed: " << cudaGetErrorString(status);
      throw std::runtime_error(message.str());
    }

    template <typename TCallable>
    auto measureCudaPipeline(
        TCallable&& callable,
        double& elapsed_total_ms, cudaEvent_t start,
        cudaEvent_t stop) {
      if (start == nullptr || stop == nullptr) {
        throw std::logic_error(
            "physical CUDA timing events are not initialized");
      }
      checkCuda(
          cudaEventRecord(start),
          "cudaEventRecord(physical pipeline start)");
      auto result =
          std::forward<TCallable>(callable)();
      checkCuda(
          cudaEventRecord(stop),
          "cudaEventRecord(physical pipeline stop)");
      checkCuda(
          cudaEventSynchronize(stop),
          "cudaEventSynchronize(physical pipeline stop)");
      float elapsed_ms = 0;
      checkCuda(
          cudaEventElapsedTime(&elapsed_ms, start, stop),
          "cudaEventElapsedTime(physical pipeline)");
      elapsed_total_ms += elapsed_ms;
      return result;
    }

    std::size_t checkedDouble(std::size_t value) {
      if (value > std::numeric_limits<std::size_t>::max() / 2) {
        throw std::overflow_error("GPU particle queue capacity overflow");
      }
      return value * 2;
    }

    std::size_t checkedAdd(std::size_t left, std::size_t right,
                           char const* message) {
      if (right > std::numeric_limits<std::size_t>::max() - left) {
        throw std::overflow_error(message);
      }
      return left + right;
    }

    std::size_t nextCapacity(std::size_t current, std::size_t requested) {
      auto capacity = std::max<std::size_t>(current, 1);
      while (capacity < requested) {
        capacity = checkedDouble(capacity);
      }
      return capacity;
    }

    template <typename TAppendWorkspace>
    std::size_t maximumWorkspaceBatchSize(
        std::size_t byte_limit, std::size_t absolute_limit,
        TAppendWorkspace&& append_workspace) {
      auto fits = [&](std::size_t count) {
        detail::WorkspaceSize input;
        input.add<EmParticleState>(count);
        if (input.bytes() > byte_limit) {
          return false;
        }
        detail::WorkspaceSize output;
        append_workspace(output, count);
        return output.bytes() <= byte_limit;
      };
      if (!fits(1)) {
        throw std::runtime_error(
            "CUDA physical workspace cannot hold one particle");
      }
      std::size_t lower = 1;
      std::size_t upper = 1;
      while (upper < absolute_limit) {
        auto const candidate =
            upper > absolute_limit / 2
                ? absolute_limit
                : upper * 2;
        if (!fits(candidate)) {
          upper = candidate;
          break;
        }
        lower = candidate;
        upper = candidate;
        if (candidate == absolute_limit) {
          return candidate;
        }
      }
      while (lower + 1 < upper) {
        auto const midpoint =
            lower + (upper - lower) / 2;
        if (fits(midpoint)) {
          lower = midpoint;
        } else {
          upper = midpoint;
        }
      }
      return lower;
    }

    bool isZeroHash(std::array<std::uint8_t, 32> const& hash) {
      return std::all_of(hash.begin(), hash.end(),
                         [](std::uint8_t value) {
                           return value == 0;
                         });
    }

    std::size_t processCount(
        tables::RateTableSet const& table) {
      std::size_t count = 0;
      for (auto const& particle : table.particles) {
        count = checkedAdd(
            count, particle.columns.size(),
            "PROPOSAL table process count overflow");
      }
      return count;
    }

    struct DeviceParticleSoA {
      std::int32_t* pid{};
      std::int32_t* medium_id{};
      std::uint32_t* generation{};
      std::uint32_t* reserved{};
      double* energy_GeV{};
      double* position_x_m{};
      double* position_y_m{};
      double* position_z_m{};
      double* direction_x{};
      double* direction_y{};
      double* direction_z{};
      double* time_s{};
      double* weight{};
      std::uint64_t* history_id{};
      std::uint64_t* parent_history_id{};
      std::uint64_t* step_id{};
    };

    __device__ EmParticleState loadParticle(DeviceParticleSoA const& queue,
                                            std::size_t index) {
      EmParticleState particle{};
      particle.pid = queue.pid[index];
      particle.medium_id = queue.medium_id[index];
      particle.generation = queue.generation[index];
      particle.reserved = queue.reserved[index];
      particle.energy_GeV = queue.energy_GeV[index];
      particle.position_m[0] = queue.position_x_m[index];
      particle.position_m[1] = queue.position_y_m[index];
      particle.position_m[2] = queue.position_z_m[index];
      particle.direction[0] = queue.direction_x[index];
      particle.direction[1] = queue.direction_y[index];
      particle.direction[2] = queue.direction_z[index];
      particle.time_s = queue.time_s[index];
      particle.weight = queue.weight[index];
      particle.history_id = queue.history_id[index];
      particle.parent_history_id = queue.parent_history_id[index];
      particle.step_id = queue.step_id[index];
      return particle;
    }

    __device__ void storeParticle(DeviceParticleSoA const& queue,
                                  std::size_t index,
                                  EmParticleState const& particle) {
      queue.pid[index] = particle.pid;
      queue.medium_id[index] = particle.medium_id;
      queue.generation[index] = particle.generation;
      queue.reserved[index] = particle.reserved;
      queue.energy_GeV[index] = particle.energy_GeV;
      queue.position_x_m[index] = particle.position_m[0];
      queue.position_y_m[index] = particle.position_m[1];
      queue.position_z_m[index] = particle.position_m[2];
      queue.direction_x[index] = particle.direction[0];
      queue.direction_y[index] = particle.direction[1];
      queue.direction_z[index] = particle.direction[2];
      queue.time_s[index] = particle.time_s;
      queue.weight[index] = particle.weight;
      queue.history_id[index] = particle.history_id;
      queue.parent_history_id[index] = particle.parent_history_id;
      queue.step_id[index] = particle.step_id;
    }

    __global__ void scatterParticles(EmParticleState const* input,
                                     std::size_t number_of_particles,
                                     DeviceParticleSoA output,
                                     std::size_t output_offset) {
      auto const index =
          static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
      if (index < number_of_particles) {
        storeParticle(output, output_offset + index, input[index]);
      }
    }

    __global__ void gatherParticles(DeviceParticleSoA input,
                                    std::size_t number_of_particles,
                                    EmParticleState* output) {
      auto const index =
          static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
      if (index < number_of_particles) {
        output[index] = loadParticle(input, index);
      }
    }

    __device__ std::uint32_t toyChildCount(double energy_GeV,
                                           std::uint64_t history_id,
                                           std::uint64_t step_id,
                                           std::uint64_t seed,
                                           std::uint64_t shower_id) {
      if (!(energy_GeV > ToyEnergyCutGeV)) {
        return 0;
      }
      RandomNumberKey const key{seed, shower_id, history_id, step_id,
                                ToyBranchingProcessId, 0};
      auto const draw = uniformOpen01(key);
      if (draw < 0.35) {
        return 0;
      }
      if (draw < 0.85) {
        return 1;
      }
      return 2;
    }

    __global__ void countToyChildren(DeviceParticleSoA particles,
                                     std::size_t number_of_particles,
                                     std::uint32_t* child_counts,
                                     std::uint64_t seed,
                                     std::uint64_t shower_id) {
      auto const index =
          static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
      if (index < number_of_particles) {
        child_counts[index] = toyChildCount(
            particles.energy_GeV[index], particles.history_id[index],
            particles.step_id[index], seed, shower_id);
      }
    }

    __device__ EmParticleState advanceToyParticle(EmParticleState particle) {
      constexpr double ToyStepLengthM = 1.;
      constexpr double SpeedOfLightMPerS = 299792458.;
      for (int axis = 0; axis < 3; ++axis) {
        particle.position_m[axis] +=
            ToyStepLengthM * particle.direction[axis];
      }
      particle.time_s += ToyStepLengthM / SpeedOfLightMPerS;
      particle.step_id++;
      return particle;
    }

    __global__ void writeToyChildren(DeviceParticleSoA input,
                                     std::size_t number_of_particles,
                                     std::uint32_t const* child_counts,
                                     std::size_t const* child_offsets,
                                     DeviceParticleSoA output,
                                     std::uint64_t first_child_history_id) {
      auto const index =
          static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
      if (index >= number_of_particles) {
        return;
      }

      auto const count = child_counts[index];
      auto const offset = child_offsets[index];
      auto const parent = loadParticle(input, index);
      if (count == 1) {
        storeParticle(output, offset, advanceToyParticle(parent));
        return;
      }
      if (count != 2) {
        return;
      }

      for (std::uint32_t child_index = 0; child_index < 2; ++child_index) {
        auto child = advanceToyParticle(parent);
        child.energy_GeV = parent.energy_GeV * 0.5;
        child.parent_history_id = parent.history_id;
        child.history_id =
            first_child_history_id + offset + child_index;
        child.generation = parent.generation + 1;
        child.step_id = 0;
        storeParticle(output, offset + child_index, child);
      }
    }

  } // namespace

  class CudaEmBackend::Impl {
  public:
    Impl() = default;

    ~Impl() { release(); }

    void initialize(EnvironmentSnapshot const& environment,
                    ProposalTableSet const& table_descriptor,
                    GpuEmConfig const& requested_config) {
      auto const initialization_start =
          std::chrono::steady_clock::now();
      if (initialized_) {
        throw std::logic_error("CUDA EM backend is already initialized");
      }
      if (environment.number_of_layers > MaxAtmosphereLayers) {
        throw std::invalid_argument(
            "environment snapshot contains more than five atmosphere layers");
      }
      if (table_descriptor.format_version == 0) {
        throw std::invalid_argument("PROPOSAL table metadata has no format version");
      }
      if (requested_config.device < 0) {
        throw std::invalid_argument("CUDA device index must be non-negative");
      }
      if (requested_config.min_batch_size == 0) {
        throw std::invalid_argument("GPU minimum batch size must be positive");
      }
      if (!(requested_config.memory_fraction > 0.) ||
          !(requested_config.memory_fraction <= 1.)) {
        throw std::invalid_argument(
            "GPU memory fraction must be in the interval (0, 1]");
      }
      if (!(requested_config.table_tolerance > 0.)) {
        throw std::invalid_argument("GPU table tolerance must be positive");
      }
      if (requested_config.thinning.enabled != 0 &&
          (!std::isfinite(
               requested_config.thinning.threshold_GeV) ||
           !(requested_config.thinning.threshold_GeV > 0.) ||
           !std::isfinite(
               requested_config.thinning.maximum_weight) ||
           !(requested_config.thinning.maximum_weight > 0.))) {
        throw std::invalid_argument(
            "enabled GPU EM thinning requires a positive finite "
            "threshold and maximum weight");
      }
      if (requested_config.profile_projection.enabled) {
        auto const& projection =
            requested_config.profile_projection;
        if (!std::isfinite(projection.axis_step_length_m) ||
            !(projection.axis_step_length_m > 0.) ||
            projection.axis_grammage_g_per_cm2.size() < 2) {
          throw std::invalid_argument(
              "GPU profile projection requires a positive axis step and "
              "at least two grammage support points");
        }
        auto direction_norm_squared = 0.;
        for (int axis = 0; axis < 3; ++axis) {
          if (!std::isfinite(
                  projection.axis_start_position_m[axis]) ||
              !std::isfinite(
                  projection.axis_direction[axis])) {
            throw std::invalid_argument(
                "GPU profile projection axis is not finite");
          }
          direction_norm_squared +=
              projection.axis_direction[axis] *
              projection.axis_direction[axis];
        }
        if (std::abs(direction_norm_squared - 1.) > 1.e-12 ||
            !std::all_of(
                projection.axis_grammage_g_per_cm2.begin(),
                projection.axis_grammage_g_per_cm2.end(),
                [](double value) {
                  return std::isfinite(value) && value >= 0.;
                }) ||
            !std::is_sorted(
                projection.axis_grammage_g_per_cm2.begin(),
                projection.axis_grammage_g_per_cm2.end())) {
          throw std::invalid_argument(
              "GPU profile projection requires a normalized direction "
              "and finite nondecreasing grammage support");
        }
        if (projection.accumulate_on_device &&
            (projection.output_bin_count == 0 ||
             !std::isfinite(
                 projection.output_bin_width_g_per_cm2) ||
             !(projection.output_bin_width_g_per_cm2 > 0.) ||
             !std::isfinite(
                 projection.energy_loss_threshold_g_per_cm2) ||
             projection.energy_loss_threshold_g_per_cm2 < 0. ||
             !std::isfinite(
                 projection.fixed_point_weight_limit) ||
             !(projection.fixed_point_weight_limit > 0.) ||
             !std::isfinite(
                 projection.fixed_point_energy_limit_GeV) ||
             !(projection.fixed_point_energy_limit_GeV > 0.))) {
          throw std::invalid_argument(
              "resident GPU profile accumulation requires positive finite "
              "binning and fixed-point ranges");
        }
      }

      config_ = requested_config;
      if (environment.number_of_layers != 0 &&
          !atmosphere_detail::validEnvironment(environment)) {
        throw std::invalid_argument(
            "CUDA EM backend requires a valid spherical atmosphere snapshot");
      }
      environment_ = environment;
      checkCuda(cudaSetDevice(config_.device), "cudaSetDevice");

      int device_count = 0;
      checkCuda(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
      if (config_.device >= device_count) {
        throw std::invalid_argument("requested CUDA device does not exist");
      }

      std::size_t free_bytes = 0;
      std::size_t total_bytes = 0;
      checkCuda(cudaMemGetInfo(&free_bytes, &total_bytes), "cudaMemGetInfo");
      memory_budget_bytes_ =
          static_cast<std::size_t>(config_.memory_fraction * free_bytes);
      if (memory_budget_bytes_ == 0) {
        throw std::runtime_error("configured CUDA memory budget is empty");
      }

      initialized_ = true;
      try {
        checkCuda(
            cudaEventCreate(&physical_pipeline_start_),
            "cudaEventCreate(physical pipeline start)");
        checkCuda(
            cudaEventCreate(&physical_pipeline_stop_),
            "cudaEventCreate(physical pipeline stop)");
        if (config_.detailed_stage_timing) {
          for (auto& event : lepton_pipeline_stage_events_) {
            checkCuda(
                cudaEventCreate(&event),
                "cudaEventCreate(lepton pipeline stage)");
          }
          statistics_.lepton_pipeline_timing.enabled = true;
        }
        if (!config_.table_cache.empty()) {
          if (table_descriptor.format_version !=
              tables::RateTableFormatVersion) {
            throw std::runtime_error(
                "PROPOSAL table descriptor format version mismatch");
          }
          auto const source =
              tables::readRateTable(config_.table_cache);
          auto const& metadata = source.metadata;
          if (metadata.requested_relative_tolerance >
                  config_.table_tolerance ||
              metadata.measured_max_relative_error >
                  config_.table_tolerance ||
              metadata.requested_loss_relative_tolerance >
                  config_.table_tolerance ||
              metadata.measured_max_loss_relative_error >
                  config_.table_tolerance) {
            throw std::runtime_error(
                "PROPOSAL table does not satisfy the configured tolerance");
          }
          auto const source_process_count =
              processCount(source);
          if (table_descriptor.process_count != 0 &&
              table_descriptor.process_count !=
                  source_process_count) {
            throw std::runtime_error(
                "PROPOSAL table descriptor process count mismatch");
          }
          if (!isZeroHash(table_descriptor.content_hash) &&
              table_descriptor.content_hash !=
                  source.content_hash) {
            throw std::runtime_error(
                "PROPOSAL table descriptor content hash mismatch");
          }
          proposal_medium_hash_ =
              source.metadata.proposal_medium_hash;
          interaction_hashes_.clear();
          interaction_hashes_.reserve(source.particles.size());
          for (auto const& particle : source.particles) {
            interaction_hashes_.emplace_back(
                particle.pdg_id, particle.interaction_hash);
          }
          photon_pair_lpm_ = makePhotonPairLpmSnapshot(
              source.metadata.photon_pair_lpm);
          brems_lpm_ = makeBremsLpmSnapshot(
              source.metadata.brems_lpm);
          brems_lpm_prepared_ =
              prepareBremsLpmSnapshotForCuda(
                  brems_lpm_, config_.device);
          if (source.metadata.moliere.enabled) {
            moliere_ = makeMoliereSnapshot(
                source.metadata.moliere);
            auto const muon_continuous = std::find_if(
                source.continuous_energy_tables.begin(),
                source.continuous_energy_tables.end(),
                [](tables::ContinuousEnergyTable const& table) {
                  return table.pdg_id ==
                         static_cast<std::int32_t>(
                             EmPid::MuonMinus);
                });
            if (muon_continuous !=
                source.continuous_energy_tables.end()) {
              auto muon_metadata =
                  source.metadata.moliere;
              muon_metadata.particle_mass_MeV =
                  muon_continuous->mass_MeV;
              muon_moliere_ =
                  makeMoliereSnapshot(muon_metadata);
              muon_moliere_available_ = true;
            }
            auto moliere_cache_path =
                config_.table_cache;
            moliere_cache_path +=
                ".moliere-initial-v1.c8cache";
            auto const interpolation =
                loadOrMakeMoliereInterpolationTable(
                    moliere_, moliere_cache_path);
            moliere_interpolation_device_bytes_ =
                sizeof(MoliereInterpolationTable);
            checkCuda(
                cudaMalloc(
                    reinterpret_cast<void**>(
                        &device_moliere_interpolation_),
                    moliere_interpolation_device_bytes_),
                "allocate GPU Moliere interpolation coefficients");
            checkCuda(
                cudaMemcpy(
                    device_moliere_interpolation_,
                    &interpolation,
                    moliere_interpolation_device_bytes_,
                    cudaMemcpyHostToDevice),
                "upload GPU Moliere interpolation coefficients");
            moliere_interpolation_ =
                makeMoliereInterpolationView(
                    device_moliere_interpolation_,
                    reinterpret_cast<double const*>(
                        reinterpret_cast<unsigned char const*>(
                            device_moliere_interpolation_) +
                        offsetof(
                            MoliereInterpolationTable,
                            initial_guess_delta)));
            statistics_.physical_host_to_device_bytes +=
                moliere_interpolation_device_bytes_;
            moliere_available_ = true;
          }
          cuda_rate_table_.initialize(
              source, config_.device, memory_budget_bytes_);
          statistics_.table_device_bytes =
              checkedAdd(
                  cuda_rate_table_.deviceBytes(),
                  moliere_interpolation_device_bytes_,
                  "GPU table byte statistics overflow");
        }
        if (config_.profile_projection.enabled) {
          auto const& projection =
              config_.profile_projection;
          profile_axis_device_bytes_ =
              projection.axis_grammage_g_per_cm2.size() *
              sizeof(double);
          checkCuda(
              cudaMalloc(
                  reinterpret_cast<void**>(
                      &device_profile_axis_grammage_),
                  profile_axis_device_bytes_),
              "allocate GPU ShowerAxis support");
          checkCuda(
              cudaMemcpy(
                  device_profile_axis_grammage_,
                  projection.axis_grammage_g_per_cm2.data(),
                  profile_axis_device_bytes_,
                  cudaMemcpyHostToDevice),
              "upload GPU ShowerAxis support");
          statistics_.physical_host_to_device_bytes +=
              profile_axis_device_bytes_;
          for (int axis = 0; axis < 3; ++axis) {
            device_profile_projection_
                .axis_start_position_m[axis] =
                projection.axis_start_position_m[axis];
            device_profile_projection_.axis_direction[axis] =
                projection.axis_direction[axis];
          }
          device_profile_projection_.axis_step_length_m =
              projection.axis_step_length_m;
          device_profile_projection_
              .axis_grammage_g_per_cm2 =
              device_profile_axis_grammage_;
          device_profile_projection_.axis_support_count =
              projection.axis_grammage_g_per_cm2.size();
          if (projection.accumulate_on_device) {
            if (projection.output_bin_count >
                std::numeric_limits<std::size_t>::max() /
                    detail::DeviceProfileHistogramCount /
                    sizeof(long long)) {
              throw std::overflow_error(
                  "GPU profile histogram allocation size overflow");
            }
            auto const histogram_bytes =
                detail::deviceProfileHistogramBytes(
                    projection.output_bin_count);
            profile_accumulator_device_bytes_ =
                checkedAdd(
                    histogram_bytes,
                    sizeof(detail::DeviceProfileCounters),
                    "GPU profile accumulator size overflow");
            checkCuda(
                cudaMalloc(
                    reinterpret_cast<void**>(
                        &device_profile_histograms_),
                    histogram_bytes),
                "allocate resident GPU profile histograms");
            checkCuda(
                cudaMalloc(
                    reinterpret_cast<void**>(
                        &device_profile_counters_),
                    sizeof(detail::DeviceProfileCounters)),
                "allocate resident GPU profile counters");
            checkCuda(
                cudaMemset(
                    device_profile_histograms_, 0,
                    histogram_bytes),
                "clear resident GPU profile histograms");
            checkCuda(
                cudaMemset(
                    device_profile_counters_, 0,
                    sizeof(detail::DeviceProfileCounters)),
                "clear resident GPU profile counters");
            auto* histogram = device_profile_histograms_;
            device_profile_accumulator_.photons = histogram;
            histogram += projection.output_bin_count;
            device_profile_accumulator_.electrons = histogram;
            histogram += projection.output_bin_count;
            device_profile_accumulator_.positrons = histogram;
            histogram += projection.output_bin_count;
            device_profile_accumulator_.muons_minus = histogram;
            histogram += projection.output_bin_count;
            device_profile_accumulator_.muons_plus = histogram;
            histogram += projection.output_bin_count;
            device_profile_accumulator_
                .muon_parent_productions = histogram;
            histogram += projection.output_bin_count;
            device_profile_accumulator_.energy_loss = histogram;
            histogram += projection.output_bin_count;
            device_profile_accumulator_.muon_energy_loss = histogram;
            device_profile_accumulator_.counters =
                device_profile_counters_;
            device_profile_accumulator_.bins =
                projection.output_bin_count;
            device_profile_accumulator_
                .bin_width_g_per_cm2 =
                projection.output_bin_width_g_per_cm2;
            device_profile_accumulator_
                .energy_loss_threshold_g_per_cm2 =
                projection.energy_loss_threshold_g_per_cm2;
            constexpr double FixedPointHeadroom = 0x1p62;
            device_profile_accumulator_.weight_scale =
                FixedPointHeadroom /
                projection.fixed_point_weight_limit;
            device_profile_accumulator_.inverse_weight_scale =
                1. /
                device_profile_accumulator_.weight_scale;
            device_profile_accumulator_.energy_scale =
                FixedPointHeadroom /
                projection.fixed_point_energy_limit_GeV;
            device_profile_accumulator_.inverse_energy_scale =
                1. /
                device_profile_accumulator_.energy_scale;
            statistics_.profile.enabled = true;
            statistics_.profile.deterministic = true;
            statistics_.profile.bins =
                projection.output_bin_count;
            statistics_.profile.device_bytes =
                profile_accumulator_device_bytes_;
            checkCuda(
                cudaStreamCreateWithFlags(
                    &profile_stream_,
                    cudaStreamNonBlocking),
                "create resident GPU profile stream");
            for (std::size_t slot = 0; slot < 2; ++slot) {
              checkCuda(
                  cudaEventCreate(
                      &profile_start_events_[slot]),
                  "create resident GPU profile start event");
              checkCuda(
                  cudaEventCreate(
                      &profile_done_events_[slot]),
                  "create resident GPU profile done event");
            }
          }
        }
        radio_accumulator_.initialize(
            config_.radio, config_.device, memory_budget_bytes_);
        radio_device_bytes_ = radio_accumulator_.deviceBytes();
        statistics_.physical_host_to_device_bytes +=
            radio_accumulator_.statistics().host_to_device_bytes;
        statistics_.radio = radio_accumulator_.statistics();
        ensureCapacity(config_.min_batch_size);
        if (resident_allocation_bytes_ >=
            memory_budget_bytes_) {
          throw std::runtime_error(
              "no CUDA memory remains for physical batch workspace");
        }
        if (config_.min_batch_size >
            (std::numeric_limits<std::size_t>::max() - 4096) /
                4096) {
          throw std::overflow_error(
              "GPU minimum batch workspace size overflow");
        }
        auto const minimum_workspace_bytes =
            4096 + 4096 * config_.min_batch_size;
        auto available_workspace_bytes =
            memory_budget_bytes_ - resident_allocation_bytes_;
        if (config_.resident_cross_species) {
          auto const maximum_cross_queue_bytes =
              available_workspace_bytes >
                      2 * minimum_workspace_bytes
                  ? available_workspace_bytes -
                        2 * minimum_workspace_bytes
                  : std::size_t{};
          auto const requested_cross_queue_bytes =
              std::min<std::size_t>(
                  512ULL * 1024ULL * 1024ULL,
                  available_workspace_bytes / 8);
          auto const cross_queue_bytes =
              std::min(
                  maximum_cross_queue_bytes,
                  requested_cross_queue_bytes);
          cross_species_queue_capacity_ =
              cross_queue_bytes /
              (2 * sizeof(EmParticleState));
          if (cross_species_queue_capacity_ <
              config_.min_batch_size) {
            throw std::runtime_error(
                "CUDA memory budget cannot hold resident cross-species queues");
          }
          cross_species_queue_device_bytes_ =
              checkedAdd(
                  2 * cross_species_queue_capacity_ *
                      sizeof(EmParticleState),
                  sizeof(std::size_t),
                  "GPU cross-species queue size overflow");
          checkCuda(
              cudaMalloc(
                  reinterpret_cast<void**>(
                      &device_pending_photons_),
                  cross_species_queue_capacity_ *
                      sizeof(EmParticleState)),
              "allocate resident photon cross-species queue");
          checkCuda(
              cudaMalloc(
                  reinterpret_cast<void**>(
                      &device_pending_leptons_),
                  cross_species_queue_capacity_ *
                      sizeof(EmParticleState)),
              "allocate resident lepton cross-species queue");
          checkCuda(
              cudaMalloc(
                  reinterpret_cast<void**>(
                      &device_cross_species_selected_count_),
                  sizeof(std::size_t)),
              "allocate resident cross-species selection count");
          checkCuda(
              cub::DeviceSelect::If(
                  nullptr, cross_species_select_temporary_bytes_,
                  device_pending_photons_,
                  device_pending_leptons_,
                  device_cross_species_selected_count_,
                  cross_species_queue_capacity_,
                  IsChargedEmParticle{}),
              "query resident cross-species selection storage");
          checkCuda(
              cudaMalloc(
                  &device_cross_species_select_temporary_,
                  cross_species_select_temporary_bytes_),
              "allocate resident cross-species selection storage");
          cross_species_queue_device_bytes_ =
              checkedAdd(
                  cross_species_queue_device_bytes_,
                  cross_species_select_temporary_bytes_,
                  "GPU cross-species selection size overflow");
          resident_allocation_bytes_ = checkedAdd(
              resident_allocation_bytes_,
              cross_species_queue_device_bytes_,
              "GPU resident cross-species accounting overflow");
          statistics_.cross_species_queue_device_bytes =
              cross_species_queue_device_bytes_;
          statistics_.cross_species_queue_capacity_per_pid =
              cross_species_queue_capacity_;
          available_workspace_bytes =
              memory_budget_bytes_ - resident_allocation_bytes_;
        }
        auto const first_workspace_limit =
            available_workspace_bytes / 2;
        auto const second_workspace_limit =
            available_workspace_bytes -
            first_workspace_limit;
        if (minimum_workspace_bytes > first_workspace_limit ||
            minimum_workspace_bytes > second_workspace_limit) {
          throw std::runtime_error(
              "CUDA memory budget cannot hold two physical wavefront arenas");
        }
        physical_workspace_.configure(
            config_.device, first_workspace_limit);
        physical_workspace_next_.configure(
            config_.device, second_workspace_limit);
        auto const common_workspace_limit =
            std::min(first_workspace_limit,
                     second_workspace_limit);
        auto const absolute_batch_limit =
            static_cast<std::size_t>(
                std::numeric_limits<std::uint32_t>::max() /
                3U);
        maximum_resident_photon_batch_size_ =
            maximumWorkspaceBatchSize(
                common_workspace_limit,
                absolute_batch_limit,
                [](detail::WorkspaceSize& required,
                   std::size_t count) {
                  detail::
                      appendWavefrontBucketingWorkspace(
                          required, count);
                  detail::
                      appendPhotonDevicePipelineWorkspace(
                          required, count);
                });
        maximum_resident_lepton_batch_size_ =
            maximumWorkspaceBatchSize(
                common_workspace_limit,
                absolute_batch_limit,
                [](detail::WorkspaceSize& required,
                   std::size_t count) {
                  detail::
                      appendWavefrontBucketingWorkspace(
                          required, count);
                  detail::
                      appendLeptonDevicePipelineWorkspace(
                          required, count);
                });
        statistics_.maximum_resident_photon_batch =
            maximum_resident_photon_batch_size_;
        statistics_.maximum_resident_lepton_batch =
            maximum_resident_lepton_batch_size_;
        if (config_.min_batch_size >
                maximum_resident_photon_batch_size_ ||
            config_.min_batch_size >
                maximum_resident_lepton_batch_size_) {
          throw std::runtime_error(
              "configured CUDA minimum batch exceeds physical "
              "workspace batch capacity");
        }
        physical_workspace_.prepare(minimum_workspace_bytes);
        physical_workspace_next_.prepare(
            minimum_workspace_bytes);
        updateWorkspaceStatistics();
        static_host_to_device_bytes_ =
            checkedAdd(
                statistics_.physical_host_to_device_bytes,
                cuda_rate_table_.deviceBytes(),
                "CUDA static upload statistics overflow");
        auto const initialization_stop =
            std::chrono::steady_clock::now();
        one_time_initialization_ms_ =
            std::chrono::duration<double, std::milli>(
                initialization_stop - initialization_start)
                .count();
        shower_ordinal_ = 1;
        radio_accumulator_.reset();
        radio_downloaded_ =
            !radio_accumulator_.enabled();
        resetStatisticsForShower(false);
      } catch (...) {
        release();
        throw;
      }
    }

    void beginShower(
        GpuEmShowerConfig const& shower_config) {
      requireInitialized();
      if (!empty() || pending_photon_count_ != 0 ||
          pending_lepton_count_ != 0) {
        throw std::logic_error(
            "CUDA EM backend cannot begin a shower with a non-empty "
            "particle wavefront");
      }
      if (gpuProfileEnabled() && !profile_downloaded_) {
        throw std::logic_error(
            "CUDA EM backend cannot discard an undownloaded resident "
            "profile at the shower boundary");
      }
      if (gpuRadioEnabled() && !radio_downloaded_) {
        throw std::logic_error(
            "CUDA EM backend cannot discard undownloaded radio waveforms "
            "at the shower boundary");
      }
      if (shower_config.thinning.enabled != 0 &&
          (!std::isfinite(
               shower_config.thinning.threshold_GeV) ||
           !(shower_config.thinning.threshold_GeV > 0.) ||
           !std::isfinite(
               shower_config.thinning.maximum_weight) ||
           !(shower_config.thinning.maximum_weight > 0.))) {
        throw std::invalid_argument(
            "enabled GPU EM thinning requires a positive finite "
            "threshold and maximum weight");
      }
      if (gpuProfileEnabled() &&
          (!std::isfinite(
               shower_config
                   .profile_fixed_point_weight_limit) ||
           !(shower_config
                 .profile_fixed_point_weight_limit > 0.) ||
           !std::isfinite(
               shower_config
                   .profile_fixed_point_energy_limit_GeV) ||
           !(shower_config
                 .profile_fixed_point_energy_limit_GeV > 0.))) {
        throw std::invalid_argument(
            "resident GPU profile requires positive finite per-shower "
            "fixed-point ranges");
      }

      checkCuda(
          cudaSetDevice(config_.device),
          "cudaSetDevice(begin CUDA EM shower)");
      drainProfileStream();
      radio_accumulator_.drain();
      if (gpuProfileEnabled()) {
        auto const histogram_bytes =
            detail::deviceProfileHistogramBytes(
                device_profile_accumulator_.bins);
        checkCuda(
            cudaMemset(
                device_profile_histograms_, 0,
                histogram_bytes),
            "reset resident GPU profile histograms");
        checkCuda(
            cudaMemset(
                device_profile_counters_, 0,
                sizeof(detail::DeviceProfileCounters)),
            "reset resident GPU profile counters");
        constexpr double FixedPointHeadroom = 0x1p62;
        device_profile_accumulator_.weight_scale =
            FixedPointHeadroom /
            shower_config
                .profile_fixed_point_weight_limit;
        device_profile_accumulator_.inverse_weight_scale =
            1. /
            device_profile_accumulator_.weight_scale;
        device_profile_accumulator_.energy_scale =
            FixedPointHeadroom /
            shower_config
                .profile_fixed_point_energy_limit_GeV;
        device_profile_accumulator_.inverse_energy_scale =
            1. /
            device_profile_accumulator_.energy_scale;
        profile_counter_snapshot_ = {};
        profile_slot_pending_ = {};
        profile_downloaded_ = false;
      }
      radio_accumulator_.reset();
      radio_downloaded_ =
          !radio_accumulator_.enabled();

      config_.random_seed = shower_config.random_seed;
      config_.shower_id = shower_config.shower_id;
      config_.thinning = shower_config.thinning;
      config_.profile_projection
          .fixed_point_weight_limit =
          shower_config
              .profile_fixed_point_weight_limit;
      config_.profile_projection
          .fixed_point_energy_limit_GeV =
          shower_config
              .profile_fixed_point_energy_limit_GeV;
      current_size_ = 0;
      host_staging_.clear();
      pending_photon_count_ = 0;
      pending_lepton_count_ = 0;
      pending_photon_head_ = 0;
      pending_lepton_head_ = 0;
      next_history_id_ = 1;
      ++shower_ordinal_;
      resetStatisticsForShower(true);
    }

    bool canTransport(EmParticleState const& particle) const {
      if (particle.pid ==
              static_cast<std::int32_t>(EmPid::Photon) ||
          isElectronOrPositronPid(particle.pid)) {
        return true;
      }
      if (!isMuonPid(particle.pid) ||
          !muon_moliere_available_) {
        return false;
      }
      return std::any_of(
          interaction_hashes_.begin(),
          interaction_hashes_.end(),
          [&](auto const& entry) {
            return entry.first == particle.pid;
          });
    }

    void enqueue(EmParticleState const& particle) {
      requireInitialized();
      if (!canTransport(particle)) {
        throw std::invalid_argument(
            "CUDA EM backend cannot transport the requested particle with the loaded table");
      }
      if (!(particle.energy_GeV >= 0.) || !std::isfinite(particle.energy_GeV)) {
        throw std::invalid_argument("GPU particle energy must be finite and non-negative");
      }
      if (particle.weight < 0. || !std::isfinite(particle.weight)) {
        throw std::invalid_argument(
            "GPU particle weight must be finite and non-negative");
      }
      if (particle.history_id == 0 ||
          particle.history_id == std::numeric_limits<std::uint64_t>::max()) {
        throw std::invalid_argument(
            "GPU particle requires a non-zero, incrementable history ID");
      }

      host_staging_.push_back(particle);
      statistics_.particles_enqueued++;
      next_history_id_ =
          std::max(next_history_id_, particle.history_id + 1);
    }

    EmBatchResult advanceWavefront() {
      requireInitialized();
      appendStaging();

      EmBatchResult result{};
      result.wavefront_index = statistics_.wavefronts;
      result.input_particles = current_size_;
      if (current_size_ == 0) {
        return result;
      }

      // A toy particle produces at most two outputs. Reserving before the scan prevents
      // reallocation from invalidating child counts and offsets.
      try {
        ensureCapacity(checkedDouble(current_size_));
      } catch (...) {
        statistics_.queue_overflows++;
        result.queue_overflow = true;
        throw;
      }

      cudaEvent_t start = nullptr;
      cudaEvent_t stop = nullptr;
      checkCuda(cudaEventCreate(&start), "cudaEventCreate(start)");
      try {
        checkCuda(cudaEventCreate(&stop), "cudaEventCreate(stop)");
        checkCuda(cudaEventRecord(start), "cudaEventRecord(start)");

        auto const blocks = static_cast<unsigned int>(
            (current_size_ + ThreadsPerBlock - 1) / ThreadsPerBlock);
        countToyChildren<<<blocks, ThreadsPerBlock>>>(
            device_current_, current_size_, device_child_counts_,
            config_.random_seed, config_.shower_id);
        checkCuda(cudaGetLastError(), "countToyChildren launch");

        checkCuda(cub::DeviceScan::ExclusiveSum(
                      device_scan_temporary_, scan_temporary_bytes_,
                      device_child_counts_, device_child_offsets_, current_size_),
                  "CUB exclusive child scan");

        std::uint32_t last_count = 0;
        std::size_t last_offset = 0;
        auto const last = current_size_ - 1;
        auto transfer_start = std::chrono::steady_clock::now();
        checkCuda(cudaMemcpy(&last_count, device_child_counts_ + last,
                             sizeof(last_count), cudaMemcpyDeviceToHost),
                  "copy final child count");
        checkCuda(cudaMemcpy(&last_offset, device_child_offsets_ + last,
                             sizeof(last_offset), cudaMemcpyDeviceToHost),
                  "copy final child offset");
        auto transfer_stop = std::chrono::steady_clock::now();
        statistics_.transfer_time_ms +=
            std::chrono::duration<double, std::milli>(transfer_stop - transfer_start)
                .count();

        auto const output_size =
            checkedAdd(last_offset, last_count, "GPU child count overflow");
        auto const first_child_history_id = next_history_id_;
        if (output_size >
            std::numeric_limits<std::uint64_t>::max() - next_history_id_) {
          throw std::overflow_error("GPU secondary history ID overflow");
        }
        if (output_size > 0) {
          writeToyChildren<<<blocks, ThreadsPerBlock>>>(
              device_current_, current_size_, device_child_counts_,
              device_child_offsets_, device_next_, first_child_history_id);
          checkCuda(cudaGetLastError(), "writeToyChildren launch");
        }

        checkCuda(cudaEventRecord(stop), "cudaEventRecord(stop)");
        checkCuda(cudaEventSynchronize(stop), "cudaEventSynchronize(stop)");
        float elapsed_ms = 0;
        checkCuda(cudaEventElapsedTime(&elapsed_ms, start, stop),
                  "cudaEventElapsedTime");
        statistics_.kernel_time_ms += elapsed_ms;

        next_history_id_ += output_size;
        statistics_.wavefronts++;
        statistics_.particles_advanced += current_size_;
        statistics_.particles_produced += output_size;
        std::swap(device_current_, device_next_);
        current_size_ = output_size;
        statistics_.current_particles = current_size_;
        statistics_.peak_particles =
            std::max(statistics_.peak_particles, current_size_);

        result.output_particles = output_size;
      } catch (...) {
        if (stop != nullptr) {
          cudaEventDestroy(stop);
        }
        cudaEventDestroy(start);
        throw;
      }
      cudaEventDestroy(stop);
      cudaEventDestroy(start);
      return result;
    }

    void drain() {
      requireInitialized();
      constexpr std::uint64_t MaximumToyWavefronts = 1000000;
      std::uint64_t iterations = 0;
      while (!empty()) {
        if (++iterations > MaximumToyWavefronts) {
          throw std::runtime_error("toy CUDA wavefront drain exceeded safety limit");
        }
        advanceWavefront();
      }
    }

    bool empty() const { return current_size_ == 0 && host_staging_.empty(); }

    GpuEmStatistics const& statistics() const {
      statistics_.radio = radio_accumulator_.statistics();
      return statistics_;
    }

    std::size_t minimumBatchSize() const {
      requireInitialized();
      return config_.min_batch_size;
    }

    std::size_t maximumResidentPhotonBatchSize() const {
      requireInitialized();
      return maximum_resident_photon_batch_size_;
    }

    std::size_t maximumResidentLeptonBatchSize() const {
      requireInitialized();
      return maximum_resident_lepton_batch_size_;
    }

    std::size_t maximumResidentInputBatchSize() const {
      requireInitialized();
      return std::min(
          maximum_resident_photon_batch_size_,
          maximum_resident_lepton_batch_size_);
    }

    std::size_t pendingPhotonCount() const noexcept {
      return pending_photon_count_;
    }

    std::size_t pendingLeptonCount() const noexcept {
      return pending_lepton_count_;
    }

    bool hasProposalTable() const {
      return cuda_rate_table_.initialized();
    }

    std::array<std::uint8_t, 32> proposalTableHash() const {
      requireInitialized();
      if (!cuda_rate_table_.initialized()) {
        throw std::logic_error(
            "CUDA EM backend has no uploaded PROPOSAL table");
      }
      return cuda_rate_table_.sourceContentHash();
    }

    bool gpuRadioEnabled() const noexcept {
      return radio_accumulator_.enabled();
    }

    radio::GpuRadioWaveforms downloadRadioWaveforms() {
      requireInitialized();
      auto result = radio_accumulator_.downloadWaveforms();
      radio_downloaded_ = true;
      statistics_.radio = radio_accumulator_.statistics();
      statistics_.physical_device_to_host_bytes +=
          statistics_.radio.device_to_host_bytes;
      return result;
    }

    bool gpuProfileEnabled() const noexcept {
      return device_profile_histograms_ != nullptr;
    }

    GpuProfileResult downloadProfile() {
      requireInitialized();
      if (!gpuProfileEnabled()) {
        throw std::logic_error(
            "CUDA resident profile accumulation is not enabled");
      }
      if (profile_downloaded_) {
        throw std::logic_error(
            "CUDA resident profile was downloaded more than once");
      }
      drainProfileStream();
      refreshProfileCounters();
      auto const counters = profile_counter_snapshot_;
      auto const bins = device_profile_accumulator_.bins;
      std::vector<long long> fixed(
          detail::DeviceProfileHistogramCount * bins);
      auto const bytes = fixed.size() * sizeof(long long);
      auto const transfer_start =
          std::chrono::steady_clock::now();
      checkCuda(
          cudaMemcpy(
              fixed.data(), device_profile_histograms_, bytes,
              cudaMemcpyDeviceToHost),
          "download resident GPU profile histograms");
      auto const transfer_stop =
          std::chrono::steady_clock::now();
      auto const transfer_ms =
          std::chrono::duration<double, std::milli>(
              transfer_stop - transfer_start)
              .count();
      statistics_.transfer_time_ms += transfer_ms;
      statistics_.profile.transfer_time_ms += transfer_ms;
      statistics_.physical_device_to_host_bytes += bytes;
      statistics_.profile.device_to_host_bytes += bytes;

      GpuProfileResult result{};
      result.photons.resize(bins);
      result.electrons.resize(bins);
      result.positrons.resize(bins);
      result.muons_minus.resize(bins);
      result.muons_plus.resize(bins);
      result.muon_parent_productions.resize(bins);
      result.energy_loss_GeV.resize(bins);
      result.muon_energy_loss_GeV.resize(bins);
      for (std::size_t bin = 0; bin < bins; ++bin) {
        result.photons[bin] =
            static_cast<double>(fixed[bin]) *
            device_profile_accumulator_
                .inverse_weight_scale;
        result.electrons[bin] =
            static_cast<double>(fixed[bins + bin]) *
            device_profile_accumulator_
                .inverse_weight_scale;
        result.positrons[bin] =
            static_cast<double>(fixed[2 * bins + bin]) *
            device_profile_accumulator_
                .inverse_weight_scale;
        result.muons_minus[bin] =
            static_cast<double>(fixed[3 * bins + bin]) *
            device_profile_accumulator_
                .inverse_weight_scale;
        result.muons_plus[bin] =
            static_cast<double>(fixed[4 * bins + bin]) *
            device_profile_accumulator_
                .inverse_weight_scale;
        result.muon_parent_productions[bin] =
            static_cast<double>(fixed[5 * bins + bin]) *
            device_profile_accumulator_
                .inverse_weight_scale;
        auto const electromagnetic_loss =
            static_cast<double>(fixed[6 * bins + bin]) *
            device_profile_accumulator_
                .inverse_energy_scale;
        result.muon_energy_loss_GeV[bin] =
            static_cast<double>(fixed[7 * bins + bin]) *
            device_profile_accumulator_
                .inverse_energy_scale;
        result.energy_loss_GeV[bin] =
            electromagnetic_loss +
            result.muon_energy_loss_GeV[bin];
        result.weighted_deposited_energy_GeV +=
            result.energy_loss_GeV[bin];
      }
      result.steps = counters.steps;
      result.deposited_steps = counters.deposited_steps;
      result.particle_cuts =
          counters.photon_cuts +
          counters.lepton_limits[
              static_cast<std::size_t>(
                  LeptonTransportLimit::ParticleCut)];
      result.fixed_point_overflows =
          counters.fixed_point_overflows;
      result.invalid_records = counters.invalid_records;
      result.weighted_medium_rest_mass_input_GeV =
          static_cast<double>(
              counters.weighted_medium_rest_mass_input) *
          device_profile_accumulator_.inverse_energy_scale;
      result.weighted_cut_rest_mass_energy_GeV =
          static_cast<double>(
              counters.weighted_cut_rest_mass_energy) *
          device_profile_accumulator_.inverse_energy_scale;
      result.weighted_observed_total_energy_GeV =
          static_cast<double>(
              counters.weighted_observed_total_energy) *
          device_profile_accumulator_.inverse_energy_scale;
      result.weighted_escaped_total_energy_GeV =
          static_cast<double>(
              counters.weighted_escaped_total_energy) *
          device_profile_accumulator_.inverse_energy_scale;
      profile_downloaded_ = true;
      return result;
    }

    EmInteractionBatchResult selectInteractionsForValidation(
        std::vector<EmParticleState> const& particles) {
      requireInitialized();
      if (!cuda_rate_table_.initialized()) {
        throw std::logic_error(
            "interaction selection requires an uploaded PROPOSAL table");
      }
      for (auto const& particle : particles) {
        if (!std::isfinite(particle.energy_GeV) ||
            particle.energy_GeV < 0. ||
            !std::isfinite(particle.weight) ||
            particle.weight < 0. || particle.history_id == 0) {
          throw std::invalid_argument(
              "interaction selection received an invalid particle");
        }
      }
      auto result = selectDiscreteInteractionsForValidation(
          cuda_rate_table_.deviceView(), particles,
          config_.random_seed, config_.shower_id, config_.device,
          physical_workspace_);
      enrichFallbackEvents(result.fallback_events);
      updateWorkspaceStatistics();
      statistics_.interaction_selection_batches++;
      statistics_.interactions_selected +=
          result.interactions.size();
      statistics_.proposal_fallbacks +=
          result.fallback_events.size();
      statistics_.physical_host_to_device_bytes +=
          particles.size() * sizeof(EmParticleState);
      statistics_.physical_device_to_host_bytes +=
          result.interactions.size() *
              sizeof(EmInteractionRecord) +
          result.fallback_events.size() *
              sizeof(ProposalFallbackEvent);
      return result;
    }

    EmFinalStateBatchResult generateFinalStatesForValidation(
        std::vector<EmInteractionRecord> const& interactions,
        std::uint64_t first_secondary_history_id) {
      requireInitialized();
      if (!cuda_rate_table_.initialized()) {
        throw std::logic_error(
            "final-state generation requires an uploaded PROPOSAL table");
      }
      for (auto const& interaction : interactions) {
        auto const& particle = interaction.particle;
        if (!std::isfinite(particle.energy_GeV) ||
            particle.energy_GeV < 0. ||
            !std::isfinite(particle.weight) ||
            particle.weight < 0. || particle.history_id == 0) {
          throw std::invalid_argument(
              "final-state generation received an invalid particle");
        }
        if (interaction.status != EmInteractionStatus::Selected &&
            interaction.status !=
                EmInteractionStatus::NoDiscreteInteraction) {
          throw std::invalid_argument(
              "final-state generation received an invalid interaction status");
        }
        if (interaction.status == EmInteractionStatus::Selected &&
            (!std::isfinite(interaction.energy_fraction) ||
             interaction.energy_fraction < 0. ||
             interaction.energy_fraction > 1.)) {
          throw std::invalid_argument(
              "final-state generation received an invalid loss fraction");
        }
        auto const capability =
            gpuProcessCapability(particle.pid, interaction.process_id);
        if (capability == GpuProcessCapability::PhotonPair ||
            capability == GpuProcessCapability::Compton ||
            capability == GpuProcessCapability::Photoelectric) {
          double norm_squared = 0.;
          for (double component : particle.direction) {
            if (!std::isfinite(component)) {
              throw std::invalid_argument(
                  "photon final-state parent direction is not finite");
            }
            norm_squared += component * component;
          }
          if (!std::isfinite(norm_squared) ||
              std::abs(norm_squared - 1.) > 1.e-12) {
            throw std::invalid_argument(
                "photon final-state parent direction is not normalized");
          }
        }
      }
      auto result = generatePhotonPairFinalStatesForValidation(
          cuda_rate_table_.deviceView(), photon_pair_lpm_,
          config_.thinning, interactions,
          config_.random_seed, config_.shower_id, config_.device,
          first_secondary_history_id, physical_workspace_);
      enrichFallbackEvents(result.fallback_events);
      updateWorkspaceStatistics();
      statistics_.final_state_batches++;
      statistics_.gpu_final_states += result.gpu_interactions;
      statistics_.physical_secondaries_generated +=
          result.secondaries.size();
      accumulateThinningStatistics(
          result.final_state_records);
      statistics_.photon_pair_lpm_trials +=
          result.photon_pair_interactions +
          result.lpm_suppressed.size();
      statistics_.photon_pair_lpm_suppressions +=
          result.lpm_suppressed.size();
      statistics_.photon_pair_final_states +=
          result.photon_pair_interactions;
      statistics_.compton_final_states +=
          result.compton_interactions;
      statistics_.photoelectric_final_states +=
          result.photoelectric_interactions;
      statistics_.proposal_fallbacks +=
          result.fallback_events.size();
      statistics_.physical_host_to_device_bytes +=
          interactions.size() * sizeof(EmInteractionRecord);
      statistics_.physical_device_to_host_bytes +=
          result.final_state_records.size() *
              sizeof(PhotonPairFinalStateRecord) +
          result.secondaries.size() * sizeof(EmParticleState) +
          result.fallback_events.size() *
              sizeof(ProposalFallbackEvent) +
          result.continuations.size() *
              sizeof(EmInteractionRecord) +
          result.lpm_suppressed.size() *
              sizeof(PhotonPairLpmSuppressionRecord);
      return result;
    }

    BremsFinalStateBatchResult
    generateBremsFinalStatesForValidation(
        std::vector<EmInteractionRecord> const& interactions,
        std::uint64_t first_secondary_history_id) {
      requireInitialized();
      if (!cuda_rate_table_.initialized()) {
        throw std::logic_error(
            "bremsstrahlung final-state generation requires an uploaded "
            "PROPOSAL table");
      }
      for (auto const& interaction : interactions) {
        auto const& particle = interaction.particle;
        if (!std::isfinite(particle.energy_GeV) ||
            particle.energy_GeV < 0. ||
            !std::isfinite(particle.weight) ||
            particle.weight < 0. ||
            particle.history_id == 0) {
          throw std::invalid_argument(
              "bremsstrahlung final-state generation received an invalid "
              "particle");
        }
        if (interaction.status != EmInteractionStatus::Selected &&
            interaction.status !=
                EmInteractionStatus::NoDiscreteInteraction) {
          throw std::invalid_argument(
              "bremsstrahlung final-state generation received an invalid "
              "interaction status");
        }
        if (interaction.status == EmInteractionStatus::Selected &&
            (!std::isfinite(interaction.energy_fraction) ||
             interaction.energy_fraction < 0. ||
             interaction.energy_fraction > 1.)) {
          throw std::invalid_argument(
              "bremsstrahlung final-state generation received an invalid "
              "loss fraction");
        }
        auto const capability = gpuProcessCapability(
            particle.pid, interaction.process_id);
        if (capability ==
                GpuProcessCapability::Bremsstrahlung ||
            capability ==
                GpuProcessCapability::Annihilation ||
            capability ==
                GpuProcessCapability::Ionization) {
          double norm_squared = 0.;
          for (double component : particle.direction) {
            if (!std::isfinite(component)) {
              throw std::invalid_argument(
                  "bremsstrahlung parent direction is not finite");
            }
            norm_squared += component * component;
          }
          if (!std::isfinite(norm_squared) ||
              std::abs(norm_squared - 1.) > 1.e-12) {
            throw std::invalid_argument(
                "bremsstrahlung parent direction is not normalized");
          }
        }
      }
      auto result =
          gpu::em::generateBremsFinalStatesForValidation(
              brems_lpm_, config_.thinning, interactions,
              config_.random_seed, config_.shower_id, config_.device,
              first_secondary_history_id,
              physical_workspace_);
      enrichFallbackEvents(result.fallback_events);
      updateWorkspaceStatistics();
      statistics_.final_state_batches++;
      statistics_.gpu_final_states +=
          result.gpu_interactions;
      statistics_.physical_secondaries_generated +=
          result.secondaries.size();
      accumulateThinningStatistics(
          result.final_state_records);
      statistics_.brems_lpm_trials +=
          result.brems_lpm_trials;
      statistics_.brems_lpm_suppressions +=
          result.brems_lpm_suppressions;
      statistics_.brems_final_states +=
          result.brems_interactions;
      statistics_.electron_pair_lpm_trials +=
          result.electron_pair_lpm_trials;
      statistics_.electron_pair_lpm_suppressions +=
          result.electron_pair_lpm_suppressions;
      statistics_.electron_pair_rejection_trials +=
          result.electron_pair_rejection_trials;
      statistics_.electron_pair_zero_weight_samples +=
          result.electron_pair_zero_weight_samples;
      statistics_.electron_pair_rejection_fallbacks +=
          result.electron_pair_rejection_fallbacks;
      statistics_.electron_pair_envelope_violations +=
          result.electron_pair_envelope_violations;
      statistics_.annihilation_final_states +=
          result.annihilation_interactions;
      statistics_.ionization_final_states +=
          result.ionization_interactions;
      statistics_.electron_pair_final_states +=
          result.electron_pair_interactions;
      statistics_.proposal_fallbacks +=
          result.fallback_events.size();
      statistics_.physical_host_to_device_bytes +=
          interactions.size() * sizeof(EmInteractionRecord);
      statistics_.physical_device_to_host_bytes +=
          result.final_state_records.size() *
              sizeof(BremsFinalStateRecord) +
          result.secondaries.size() * sizeof(EmParticleState) +
          result.fallback_events.size() *
              sizeof(ProposalFallbackEvent) +
          result.continuations.size() *
              sizeof(EmInteractionRecord) +
          result.lpm_suppressed.size() *
              sizeof(BremsLpmSuppressionRecord);
      return result;
    }

    PhotonTransportBatchResult transportPhotonsForValidation(
        std::vector<EmInteractionRecord> const& interactions) {
      requireInitialized();
      for (auto const& interaction : interactions) {
        auto const& particle = interaction.particle;
        double direction_norm_squared = 0.;
        for (std::size_t axis = 0; axis < 3; ++axis) {
          if (!std::isfinite(particle.position_m[axis]) ||
              !std::isfinite(particle.direction[axis])) {
            throw std::invalid_argument(
                "photon transport received a non-finite state");
          }
          direction_norm_squared +=
              particle.direction[axis] * particle.direction[axis];
        }
        if (particle.history_id == 0) {
          throw std::invalid_argument(
              "photon transport received an invalid particle identity");
        }
        if (!std::isfinite(direction_norm_squared) ||
            std::abs(direction_norm_squared - 1.) > 1.e-12) {
          throw std::invalid_argument(
              "photon transport direction is not normalized");
        }
      }
      auto result = gpu::em::transportPhotonsForValidation(
          environment_, interactions, config_.device,
          physical_workspace_);
      enrichFallbackEvents(result.fallback_events);
      updateWorkspaceStatistics();
      statistics_.photon_transport_batches++;
      statistics_.proposal_fallbacks +=
          result.fallback_events.size();
      statistics_.physical_host_to_device_bytes +=
          interactions.size() * sizeof(EmInteractionRecord);
      statistics_.physical_device_to_host_bytes +=
          result.records.size() *
              sizeof(PhotonTransportRecord) +
          result.fallback_events.size() *
              sizeof(ProposalFallbackEvent);
      for (auto const& record : result.records) {
        switch (record.limit) {
          case PhotonTransportLimit::Interaction:
            statistics_.photon_transport_interactions++;
            break;
          case PhotonTransportLimit::LayerBoundary:
            statistics_.photon_transport_boundaries++;
            break;
          case PhotonTransportLimit::ObservationSurface:
            statistics_.photon_transport_observations++;
            break;
          case PhotonTransportLimit::EscapedEnvironment:
            statistics_.photon_transport_escapes++;
            break;
          case PhotonTransportLimit::ParticleCut:
            statistics_.photon_transport_cuts++;
            break;
        }
      }
      return result;
    }

    LeptonTransportBatchResult
    transportLeptonsStraightForValidation(
        std::vector<EmInteractionRecord> const& interactions) {
      requireInitialized();
      if (!cuda_rate_table_.initialized()) {
        throw std::logic_error(
            "lepton transport requires an uploaded PROPOSAL table");
      }
      for (auto const& interaction : interactions) {
        auto const& particle = interaction.particle;
        double direction_norm_squared = 0.;
        if (!std::isfinite(particle.energy_GeV) ||
            !(particle.energy_GeV > 0.) ||
            !std::isfinite(particle.weight) ||
            particle.weight < 0. ||
            particle.history_id == 0) {
          throw std::invalid_argument(
              "lepton transport received an invalid particle");
        }
        for (std::size_t axis = 0; axis < 3; ++axis) {
          if (!std::isfinite(particle.position_m[axis]) ||
              !std::isfinite(particle.direction[axis])) {
            throw std::invalid_argument(
                "lepton transport received a non-finite state");
          }
          direction_norm_squared +=
              particle.direction[axis] *
              particle.direction[axis];
        }
        if (!std::isfinite(direction_norm_squared) ||
            std::abs(direction_norm_squared - 1.) > 1.e-12) {
          throw std::invalid_argument(
              "lepton transport direction is not normalized");
        }
      }
      auto result =
          gpu::em::transportLeptonsStraightForValidation(
              cuda_rate_table_.deviceView(), environment_,
              interactions, config_.device, physical_workspace_);
      enrichFallbackEvents(result.fallback_events);
      updateWorkspaceStatistics();
      statistics_.lepton_transport_batches++;
      statistics_.proposal_fallbacks +=
          result.fallback_events.size();
      statistics_.physical_host_to_device_bytes +=
          interactions.size() * sizeof(EmInteractionRecord);
      statistics_.physical_device_to_host_bytes +=
          result.records.size() *
              sizeof(LeptonTransportRecord) +
          result.fallback_events.size() *
              sizeof(ProposalFallbackEvent);
      for (auto const& record : result.records) {
        switch (record.limit) {
          case LeptonTransportLimit::InteractionCandidate:
            statistics_
                .lepton_transport_interaction_candidates++;
            break;
          case LeptonTransportLimit::ContinuousStep:
            statistics_.lepton_transport_continuous_steps++;
            break;
          case LeptonTransportLimit::ParticleCut:
            statistics_.lepton_transport_cuts++;
            break;
          case LeptonTransportLimit::LayerBoundary:
            statistics_.lepton_transport_boundaries++;
            break;
          case LeptonTransportLimit::ObservationSurface:
            statistics_.lepton_transport_observations++;
            break;
          case LeptonTransportLimit::EscapedEnvironment:
            statistics_.lepton_transport_escapes++;
            break;
          case LeptonTransportLimit::MagneticStep:
            statistics_.lepton_transport_magnetic_steps++;
            break;
          case LeptonTransportLimit::DecayCandidate:
            statistics_.lepton_transport_decay_candidates++;
            break;
        }
      }
      return result;
    }

    LeptonVertexSelectionBatchResult
    reselectLeptonInteractionsAtVertexForValidation(
        std::vector<EmInteractionRecord> const& candidates) {
      requireInitialized();
      if (!cuda_rate_table_.initialized()) {
        throw std::logic_error(
            "lepton vertex selection requires an uploaded PROPOSAL table");
      }
      for (auto const& candidate : candidates) {
        auto const& particle = candidate.particle;
        if (candidate.status !=
                EmInteractionStatus::RequiresReselection ||
            !std::isfinite(particle.energy_GeV) ||
            !(particle.energy_GeV > 0.) ||
            !std::isfinite(particle.weight) ||
            particle.weight < 0. ||
            particle.history_id == 0) {
          throw std::invalid_argument(
              "lepton vertex selection received an invalid candidate");
        }
      }
      auto result =
          gpu::em::
              reselectLeptonInteractionsAtVertexForValidation(
                  cuda_rate_table_.deviceView(), candidates,
                  config_.random_seed, config_.shower_id,
                  config_.device, physical_workspace_);
      enrichFallbackEvents(result.fallback_events);
      updateWorkspaceStatistics();
      statistics_.lepton_vertex_selection_batches++;
      statistics_.lepton_vertex_interactions_selected +=
          result.interactions.size();
      statistics_
          .lepton_vertex_no_interaction_continuations +=
          result.continuations.size();
      statistics_.proposal_fallbacks +=
          result.fallback_events.size();
      statistics_.physical_host_to_device_bytes +=
          candidates.size() * sizeof(EmInteractionRecord);
      statistics_.physical_device_to_host_bytes +=
          (result.interactions.size() +
           result.continuations.size()) *
              sizeof(EmInteractionRecord) +
          result.fallback_events.size() *
              sizeof(ProposalFallbackEvent);
      return result;
    }

    LeptonDevicePipelineBatchResult
    runPhysicalLeptonPipeline(
        std::vector<EmParticleState> const& particles,
        std::uint64_t first_secondary_history_id) {
      requireInitialized();
      if (!cuda_rate_table_.initialized()) {
        throw std::logic_error(
            "lepton device pipeline requires an uploaded PROPOSAL table");
      }
      if (first_secondary_history_id == 0) {
        throw std::invalid_argument(
            "lepton device pipeline requires nonzero secondary IDs");
      }
      for (auto const& particle : particles) {
        auto const charged =
            isChargedLeptonPid(particle.pid);
        if (!charged || !std::isfinite(particle.energy_GeV) ||
            !(particle.energy_GeV > 0.) ||
            !std::isfinite(particle.weight) ||
            particle.weight < 0. ||
            particle.history_id == 0) {
          throw std::invalid_argument(
              "lepton device pipeline accepts only valid charged leptons");
        }
        double direction_norm_squared = 0.;
        for (std::size_t axis = 0; axis < 3; ++axis) {
          if (!std::isfinite(particle.position_m[axis]) ||
              !std::isfinite(particle.direction[axis])) {
            throw std::invalid_argument(
                "lepton device pipeline received a non-finite state");
          }
          direction_norm_squared +=
              particle.direction[axis] *
              particle.direction[axis];
        }
        if (!std::isfinite(direction_norm_squared) ||
            std::abs(direction_norm_squared - 1.) > 1.e-12) {
          throw std::invalid_argument(
              "lepton device pipeline direction is not normalized");
        }
      }
      auto result =
          gpu::em::runLeptonDevicePipelineForValidation(
              cuda_rate_table_.deviceView(), brems_lpm_,
              config_.thinning, moliere_,
              muon_moliere_, moliere_interpolation_,
              moliere_available_,
              muon_moliere_available_,
              environment_, particles, config_.random_seed,
              config_.shower_id, config_.device,
              first_secondary_history_id, physical_workspace_);
      enrichFallbackEvents(result.selection_fallback_events);
      enrichFallbackEvents(result.transport_fallback_events);
      enrichFallbackEvents(result.vertex_fallback_events);
      enrichFallbackEvents(result.final_states.fallback_events);
      updateWorkspaceStatistics();
      statistics_.physical_lepton_wavefronts++;
      ++statistics_
            .lepton_selection_transport_summary_fusions;
      ++statistics_
            .pipeline_host_synchronizations_eliminated;
      if (!result.transport_records.empty()) {
        ++statistics_
              .lepton_transport_vertex_summary_fusions;
        ++statistics_
              .pipeline_host_synchronizations_eliminated;
        if (result.interaction_candidates != 0) {
          statistics_
              .pipeline_device_to_host_bytes_eliminated += 12;
          ++statistics_
                .lepton_vertex_final_state_summary_fusions;
          ++statistics_
                .pipeline_host_synchronizations_eliminated;
        }
      }
      statistics_.interaction_selection_batches++;
      statistics_.interactions_selected +=
          result.transport_records.size() +
          result.transport_fallback_events.size();
      statistics_.lepton_transport_batches++;
      for (auto const& record : result.transport_records) {
        if (result.multiple_scattering_enabled) {
          statistics_.moliere_trials++;
          statistics_.moliere_newton_iterations +=
              record.multiple_scattering_iterations;
          statistics_.moliere_max_newton_iterations =
              std::max(
                  statistics_.moliere_max_newton_iterations,
                  record.multiple_scattering_iterations);
          if (record.multiple_scattering_applied != 0) {
            statistics_.moliere_deflections++;
          } else if (
              record.multiple_scattering_status ==
              static_cast<std::uint32_t>(
                  MoliereStatus::NoDeflection)) {
            statistics_.moliere_zero_deflections++;
          }
        }
        switch (record.limit) {
          case LeptonTransportLimit::InteractionCandidate:
            statistics_
                .lepton_transport_interaction_candidates++;
            break;
          case LeptonTransportLimit::ContinuousStep:
            statistics_.lepton_transport_continuous_steps++;
            break;
          case LeptonTransportLimit::ParticleCut:
            statistics_.lepton_transport_cuts++;
            break;
          case LeptonTransportLimit::LayerBoundary:
            statistics_.lepton_transport_boundaries++;
            break;
          case LeptonTransportLimit::ObservationSurface:
            statistics_.lepton_transport_observations++;
            break;
          case LeptonTransportLimit::EscapedEnvironment:
            statistics_.lepton_transport_escapes++;
            break;
          case LeptonTransportLimit::MagneticStep:
            statistics_.lepton_transport_magnetic_steps++;
            break;
          case LeptonTransportLimit::DecayCandidate:
            statistics_.lepton_transport_decay_candidates++;
            break;
        }
      }
      if (result.interaction_candidates != 0) {
        statistics_.lepton_vertex_selection_batches++;
      }
      statistics_.lepton_vertex_interactions_selected +=
          result.vertex_interactions;
      statistics_
          .lepton_vertex_no_interaction_continuations +=
          result.vertex_continuations.size();
      if (result.final_states.input_interactions != 0) {
        statistics_.final_state_batches++;
        ++statistics_
              .lepton_final_state_endpoint_summary_fusions;
        ++statistics_
              .pipeline_host_synchronizations_eliminated;
        statistics_
            .pipeline_device_to_host_bytes_eliminated += 8;
      }
      statistics_.gpu_final_states +=
          result.final_states.gpu_interactions;
      statistics_.physical_secondaries_generated +=
          result.final_states.secondaries.size();
      accumulateThinningStatistics(
          result.final_states.final_state_records);
      statistics_.brems_lpm_trials +=
          result.final_states.brems_lpm_trials;
      statistics_.brems_lpm_suppressions +=
          result.final_states.brems_lpm_suppressions;
      statistics_.brems_final_states +=
          result.final_states.brems_interactions;
      statistics_.electron_pair_lpm_trials +=
          result.final_states.electron_pair_lpm_trials;
      statistics_.electron_pair_lpm_suppressions +=
          result.final_states.electron_pair_lpm_suppressions;
      statistics_.electron_pair_rejection_trials +=
          result.final_states.electron_pair_rejection_trials;
      statistics_.electron_pair_zero_weight_samples +=
          result.final_states.electron_pair_zero_weight_samples;
      statistics_.electron_pair_rejection_fallbacks +=
          result.final_states.electron_pair_rejection_fallbacks;
      statistics_.electron_pair_envelope_violations +=
          result.final_states.electron_pair_envelope_violations;
      statistics_.annihilation_final_states +=
          result.final_states.annihilation_interactions;
      statistics_.ionization_final_states +=
          result.final_states.ionization_interactions;
      statistics_.electron_pair_final_states +=
          result.final_states.electron_pair_interactions;
      statistics_.proposal_fallbacks +=
          result.selection_fallback_events.size() +
          result.transport_fallback_events.size() +
          result.vertex_fallback_events.size() +
          result.final_states.fallback_events.size();
      statistics_.physical_host_to_device_bytes +=
          particles.size() * sizeof(EmParticleState);
      statistics_.physical_device_to_host_bytes +=
          result.transport_records.size() *
              sizeof(LeptonTransportRecord) +
          result.selection_fallback_events.size() *
              sizeof(ProposalFallbackEvent) +
          result.transport_fallback_events.size() *
              sizeof(ProposalFallbackEvent) +
          result.vertex_continuations.size() *
              sizeof(EmInteractionRecord) +
          result.vertex_fallback_events.size() *
              sizeof(ProposalFallbackEvent) +
          result.final_states.final_state_records.size() *
              sizeof(BremsFinalStateRecord) +
          result.final_states.secondaries.size() *
              sizeof(EmParticleState) +
          result.final_states.fallback_events.size() *
              sizeof(ProposalFallbackEvent) +
          result.final_states.continuations.size() *
              sizeof(EmInteractionRecord) +
          result.final_states.lpm_suppressed.size() *
              sizeof(BremsLpmSuppressionRecord) +
          result.next_leptons.size() *
              sizeof(EmParticleState) +
          result.generated_photons.size() *
              sizeof(EmParticleState) +
          result.observations.size() *
              sizeof(ObservationRecord);
      return result;
    }

    PhotonSelectionTransportBatchResult
    selectAndTransportPhotonsForValidation(
        std::vector<EmParticleState> const& particles) {
      requireInitialized();
      if (!cuda_rate_table_.initialized()) {
        throw std::logic_error(
            "selection-transport requires an uploaded PROPOSAL table");
      }
      for (auto const& particle : particles) {
        if (!std::isfinite(particle.energy_GeV) ||
            particle.energy_GeV < 0. ||
            !std::isfinite(particle.weight) ||
            particle.weight < 0. || particle.history_id == 0) {
          throw std::invalid_argument(
              "selection-transport received an invalid particle");
        }
        double direction_norm_squared = 0.;
        for (std::size_t axis = 0; axis < 3; ++axis) {
          if (!std::isfinite(particle.position_m[axis]) ||
              !std::isfinite(particle.direction[axis])) {
            throw std::invalid_argument(
                "selection-transport received a non-finite state");
          }
          direction_norm_squared +=
              particle.direction[axis] *
              particle.direction[axis];
        }
        if (!std::isfinite(direction_norm_squared) ||
            std::abs(direction_norm_squared - 1.) > 1.e-12) {
          throw std::invalid_argument(
              "selection-transport direction is not normalized");
        }
      }
      auto result =
          gpu::em::selectAndTransportPhotonsForValidation(
              cuda_rate_table_.deviceView(), environment_, particles,
              config_.random_seed, config_.shower_id, config_.device,
              physical_workspace_);
      enrichFallbackEvents(result.selection_fallback_events);
      enrichFallbackEvents(result.transport_fallback_events);
      updateWorkspaceStatistics();
      statistics_.interaction_selection_batches++;
      auto const selected_count =
          result.records.size() +
          result.transport_fallback_events.size();
      statistics_.interactions_selected += selected_count;
      statistics_.photon_transport_batches++;
      statistics_.proposal_fallbacks +=
          result.selection_fallback_events.size() +
          result.transport_fallback_events.size();
      statistics_.physical_host_to_device_bytes +=
          particles.size() * sizeof(EmParticleState);
      statistics_.physical_device_to_host_bytes +=
          result.records.size() *
              sizeof(PhotonTransportRecord) +
          result.selection_fallback_events.size() *
              sizeof(ProposalFallbackEvent) +
          result.transport_fallback_events.size() *
              sizeof(ProposalFallbackEvent);
      accumulateTransportStatistics(result.records);
      return result;
    }

    PhotonDevicePipelineBatchResult
    runPhysicalPhotonPipeline(
        std::vector<EmParticleState> const& particles,
        std::uint64_t first_secondary_history_id) {
      requireInitialized();
      if (!cuda_rate_table_.initialized()) {
        throw std::logic_error(
            "photon device pipeline requires an uploaded PROPOSAL table");
      }
      if (first_secondary_history_id == 0) {
        throw std::invalid_argument(
            "photon device pipeline requires nonzero secondary IDs");
      }
      for (auto const& particle : particles) {
        if (!std::isfinite(particle.energy_GeV) ||
            particle.energy_GeV < 0. ||
            !std::isfinite(particle.weight) ||
            particle.weight < 0. || particle.history_id == 0) {
          throw std::invalid_argument(
              "photon device pipeline received an invalid particle");
        }
        double direction_norm_squared = 0.;
        for (std::size_t axis = 0; axis < 3; ++axis) {
          if (!std::isfinite(particle.position_m[axis]) ||
              !std::isfinite(particle.direction[axis])) {
            throw std::invalid_argument(
                "photon device pipeline received a non-finite state");
          }
          direction_norm_squared +=
              particle.direction[axis] *
              particle.direction[axis];
        }
        if (!std::isfinite(direction_norm_squared) ||
            std::abs(direction_norm_squared - 1.) > 1.e-12) {
          throw std::invalid_argument(
              "photon device pipeline direction is not normalized");
        }
      }
      auto result =
          gpu::em::runPhotonDevicePipelineForValidation(
              cuda_rate_table_.deviceView(), photon_pair_lpm_,
              config_.thinning, environment_, particles,
              config_.random_seed, config_.shower_id, config_.device,
              first_secondary_history_id, physical_workspace_);
      enrichFallbackEvents(result.selection_fallback_events);
      enrichFallbackEvents(result.transport_fallback_events);
      enrichFallbackEvents(result.final_states.fallback_events);
      updateWorkspaceStatistics();
      ++statistics_
            .photon_selection_transport_summary_fusions;
      ++statistics_
            .pipeline_host_synchronizations_eliminated;
      if (!result.transport_records.empty()) {
        ++statistics_
              .photon_transport_final_state_summary_fusions;
        ++statistics_
              .pipeline_host_synchronizations_eliminated;
        statistics_
            .pipeline_device_to_host_bytes_eliminated += 4;
      }
      statistics_.interaction_selection_batches++;
      statistics_.interactions_selected +=
          result.transport_records.size() +
          result.transport_fallback_events.size();
      statistics_.photon_transport_batches++;
      accumulateTransportStatistics(result.transport_records);
      if (result.final_states.input_interactions != 0) {
        statistics_.final_state_batches++;
        ++statistics_
              .photon_final_state_endpoint_summary_fusions;
        ++statistics_
              .pipeline_host_synchronizations_eliminated;
        // The old photon stage downloaded 16 final scan values (64 bytes)
        // and then a 6-word endpoint summary (24 bytes).  The fused endpoint
        // summary is 15 words (60 bytes): one synchronization and 28 control
        // bytes are removed per non-empty final-state batch.
        statistics_
            .pipeline_device_to_host_bytes_eliminated += 28;
      }
      statistics_.gpu_final_states +=
          result.final_states.gpu_interactions;
      statistics_.physical_secondaries_generated +=
          result.final_states.secondaries.size();
      accumulateThinningStatistics(
          result.final_states.final_state_records);
      statistics_.photon_pair_lpm_trials +=
          result.final_states.photon_pair_interactions +
          result.final_states.lpm_suppressed.size();
      statistics_.photon_pair_lpm_suppressions +=
          result.final_states.lpm_suppressed.size();
      statistics_.photon_pair_final_states +=
          result.final_states.photon_pair_interactions;
      statistics_.compton_final_states +=
          result.final_states.compton_interactions;
      statistics_.photoelectric_final_states +=
          result.final_states.photoelectric_interactions;
      statistics_.proposal_fallbacks +=
          result.selection_fallback_events.size() +
          result.transport_fallback_events.size() +
          result.final_states.fallback_events.size();
      statistics_.physical_host_to_device_bytes +=
          particles.size() * sizeof(EmParticleState);
      statistics_.physical_device_to_host_bytes +=
          result.transport_records.size() *
              sizeof(PhotonTransportRecord) +
          result.selection_fallback_events.size() *
              sizeof(ProposalFallbackEvent) +
          result.transport_fallback_events.size() *
              sizeof(ProposalFallbackEvent) +
          result.final_states.final_state_records.size() *
              sizeof(PhotonPairFinalStateRecord) +
          result.final_states.secondaries.size() *
              sizeof(EmParticleState) +
          result.final_states.fallback_events.size() *
              sizeof(ProposalFallbackEvent) +
          result.final_states.continuations.size() *
              sizeof(EmInteractionRecord) +
          result.final_states.lpm_suppressed.size() *
              sizeof(PhotonPairLpmSuppressionRecord) +
          result.next_photons.size() * sizeof(EmParticleState) +
          result.observations.size() * sizeof(ObservationRecord);
      return result;
    }

    ResidentPhotonCascadeResult
    runResidentPhotonCascadeForValidation(
        std::vector<EmParticleState> const& particles,
        std::uint64_t first_secondary_history_id,
        std::size_t maximum_wavefronts,
        std::size_t minimum_resident_batch_size) {
      requireInitialized();
      if (!cuda_rate_table_.initialized()) {
        throw std::logic_error(
            "resident photon cascade requires an uploaded PROPOSAL table");
      }
      if (first_secondary_history_id == 0 ||
          maximum_wavefronts == 0 ||
          minimum_resident_batch_size == 0) {
        throw std::invalid_argument(
            "resident photon cascade requires nonzero history/wavefront limits");
      }
      ResidentPhotonCascadeResult result{};
      auto const pending_input =
          std::min(
              pending_photon_count_,
              maximum_resident_photon_batch_size_);
      auto const total_input = checkedAdd(
          particles.size(), pending_input,
          "resident photon input count overflow");
      result.input_particles = total_input;
      if (total_input == 0) {
        result.completed = true;
        return result;
      }
      if (total_input >
          maximum_resident_photon_batch_size_) {
        if (pending_input == 0) {
          result.workspace_limit_checkpoint = true;
          result.remaining_photons = particles;
          return result;
        }
        throw std::runtime_error(
            "resident photon host plus cross-species input exceeds workspace capacity");
      }
      for (auto const& particle : particles) {
        if (particle.pid !=
                static_cast<std::int32_t>(EmPid::Photon) ||
            !std::isfinite(particle.energy_GeV) ||
            particle.energy_GeV < 0. ||
            !std::isfinite(particle.weight) ||
            particle.weight < 0. || particle.history_id == 0) {
          throw std::invalid_argument(
              "resident photon cascade accepts only valid photons");
        }
        double norm_squared = 0.;
        for (std::size_t axis = 0; axis < 3; ++axis) {
          if (!std::isfinite(particle.position_m[axis]) ||
              !std::isfinite(particle.direction[axis])) {
            throw std::invalid_argument(
                "resident photon cascade received a non-finite state");
          }
          norm_squared +=
              particle.direction[axis] *
              particle.direction[axis];
        }
        if (!std::isfinite(norm_squared) ||
            std::abs(norm_squared - 1.) > 1.e-12) {
          throw std::invalid_argument(
              "resident photon direction is not normalized");
        }
      }

      auto appendDevice = [&](auto& destination, auto const* source,
                              std::size_t count,
                              char const* operation) {
        using Vector =
            std::decay_t<decltype(destination)>;
        using Value = typename Vector::value_type;
        if (count == 0) {
          return;
        }
        auto const old_size = destination.size();
        destination.resize(
            checkedAdd(old_size, count,
                       "resident cascade host output size overflow"));
        auto const transfer_start =
            std::chrono::steady_clock::now();
        checkCuda(cudaMemcpy(
                      destination.data() + old_size, source,
                      count * sizeof(Value), cudaMemcpyDeviceToHost),
                  operation);
        auto const transfer_stop =
            std::chrono::steady_clock::now();
        statistics_.transfer_time_ms +=
            std::chrono::duration<double, std::milli>(
                transfer_stop - transfer_start)
                .count();
        statistics_.physical_device_to_host_bytes +=
            count * sizeof(Value);
      };

      detail::DeviceWorkspace* current_workspace =
          &physical_workspace_;
      detail::DeviceWorkspace* output_workspace =
          &physical_workspace_next_;
      waitForProfileInputSlot(
          physicalWorkspaceSlot(current_workspace));
      detail::WorkspaceSize input_required;
      input_required.add<EmParticleState>(total_input);
      current_workspace->prepare(input_required.bytes());
      auto* current_particles =
          current_workspace->acquire<EmParticleState>(
              total_input);
      auto const input_transfer_start =
          std::chrono::steady_clock::now();
      if (pending_input != 0) {
        checkCuda(
            cudaMemcpy(
                current_particles,
                device_pending_photons_ +
                    pending_photon_head_,
                pending_input * sizeof(EmParticleState),
                cudaMemcpyDeviceToDevice),
            "consume resident cross-species photons");
        pending_photon_head_ += pending_input;
        pending_photon_count_ -= pending_input;
        if (pending_photon_count_ == 0) {
          pending_photon_head_ = 0;
        }
        statistics_.cross_species_device_to_device_bytes +=
            pending_input * sizeof(EmParticleState);
      }
      if (!particles.empty()) {
        checkCuda(cudaMemcpy(
                      current_particles + pending_input,
                      particles.data(),
                      particles.size() *
                          sizeof(EmParticleState),
                      cudaMemcpyHostToDevice),
                  "upload resident photon cascade input");
      }
      auto const input_transfer_stop =
          std::chrono::steady_clock::now();
      statistics_.transfer_time_ms +=
          std::chrono::duration<double, std::milli>(
              input_transfer_stop -
              input_transfer_start)
              .count();
      statistics_.physical_host_to_device_bytes +=
          particles.size() * sizeof(EmParticleState);

      auto current_count = total_input;
      auto next_secondary_history_id =
          first_secondary_history_id;
      result.peak_resident_photons = current_count;
      while (current_count != 0 &&
             result.wavefronts < maximum_wavefronts) {
        if (result.wavefronts != 0 &&
            current_count < minimum_resident_batch_size) {
          result.below_minimum_batch_checkpoint = true;
          break;
        }
        detail::WorkspaceSize required;
        detail::appendWavefrontBucketingWorkspace(
            required, current_count);
        detail::appendPhotonDevicePipelineWorkspace(
            required, current_count);
        if (required.bytes() >
            output_workspace->byteLimit()) {
          result.workspace_limit_checkpoint = true;
          break;
        }
        auto const profile_input_slot =
            physicalWorkspaceSlot(output_workspace);
        waitForProfileInputSlot(profile_input_slot);
        output_workspace->prepare(required.bytes());
        detail::DeviceChargedSecondarySink
            charged_secondary_sink{};
        detail::DeviceChargedSecondarySink const*
            charged_secondary_sink_ptr = nullptr;
        if (config_.resident_cross_species) {
          auto const pending_tail =
              pending_lepton_head_ +
              pending_lepton_count_;
          if (pending_tail >
              cross_species_queue_capacity_) {
            throw std::logic_error(
                "resident lepton queue tail exceeds capacity");
          }
          charged_secondary_sink.output =
              device_pending_leptons_ + pending_tail;
          charged_secondary_sink.capacity =
              cross_species_queue_capacity_ -
              pending_tail;
          charged_secondary_sink.temporary_storage =
              device_cross_species_select_temporary_;
          charged_secondary_sink
              .temporary_storage_bytes =
              cross_species_select_temporary_bytes_;
          charged_secondary_sink.selected_count =
              device_cross_species_selected_count_;
          charged_secondary_sink_ptr =
              &charged_secondary_sink;
        }
        auto const pipeline = measureCudaPipeline(
            [&] {
              auto const bucketed =
                  detail::
                      launchWavefrontBucketingOnDevice(
                          current_particles,
                          current_count,
                          *output_workspace);
              return detail::
                  launchPhotonDevicePipelineOnDevice(
                      cuda_rate_table_.deviceView(),
                      photon_pair_lpm_, config_.thinning,
                      environment_, bucketed.particles,
                      current_count, config_.random_seed,
                      config_.shower_id,
                      next_secondary_history_id,
                      *output_workspace,
                      charged_secondary_sink_ptr);
            },
            statistics_.kernel_time_ms,
            physical_pipeline_start_,
            physical_pipeline_stop_);
        if (current_count >=
            detail::MinimumWavefrontRadixSortSize) {
          ++statistics_.wavefront_bucketing_batches;
          statistics_.wavefront_bucketing_particles +=
              current_count;
        } else {
          ++statistics_.wavefront_bucketing_small_batches;
          statistics_.wavefront_bucketing_small_particles +=
              current_count;
        }
        updateWorkspaceStatistics();
        ++result.wavefronts;
        ++statistics_.physical_photon_wavefronts;
        ++statistics_
              .photon_selection_transport_summary_fusions;
        ++statistics_
              .pipeline_host_synchronizations_eliminated;
        if (pipeline.transport.record_count != 0) {
          ++statistics_
                .photon_transport_final_state_summary_fusions;
          ++statistics_
                .pipeline_host_synchronizations_eliminated;
          statistics_
              .pipeline_device_to_host_bytes_eliminated += 4;
        }
        ++statistics_.interaction_selection_batches;
        ++statistics_.photon_transport_batches;
        statistics_.interactions_selected +=
            pipeline.selection.interaction_count;
        result.transport_records +=
            pipeline.transport.record_count;
        auto particle_cuts =
            pipeline.endpoints.particle_cut_count;
        if (gpuProfileEnabled()) {
          scheduleProfileAccumulation(
              profile_input_slot,
              [&](cudaStream_t stream) {
                detail::
                    launchPhotonProfileAccumulationOnDevice(
                        device_profile_projection_,
                        device_profile_accumulator_,
                        pipeline.transport.records,
                        pipeline.transport.record_count,
                        pipeline.final_state.records,
                        pipeline.final_state
                            .gpu_interaction_count,
                        stream);
              });
        } else if (config_.profile_projection.enabled) {
          // Selection has finished before this point, so its compacted
          // interaction array is dead until the next wavefront.  A projected
          // record is smaller than an interaction record; reuse that storage
          // to avoid reserving a second O(wavefront-size) device buffer.
          auto* projected_records =
              reinterpret_cast<ProjectedEmStepRecord*>(
                  pipeline.selection.interactions);
          measureCudaPipeline(
              [&] {
                detail::
                    launchPhotonProfileProjectionOnDevice(
                        device_profile_projection_,
                        pipeline.transport.records,
                        pipeline.transport.record_count,
                        projected_records);
                return 0;
              },
              statistics_.kernel_time_ms,
              physical_pipeline_start_,
              physical_pipeline_stop_);
          auto const projected_begin =
              result.projected_step_records.size();
          appendDevice(
              result.projected_step_records,
              projected_records,
              pipeline.transport.record_count,
              "download projected resident photon steps");
          particle_cuts =
              static_cast<std::size_t>(std::count_if(
                  result.projected_step_records.begin() +
                      static_cast<std::ptrdiff_t>(
                          projected_begin),
                  result.projected_step_records.end(),
                  [](ProjectedEmStepRecord const& record) {
                    return record.transport_limit ==
                           static_cast<std::int32_t>(
                               PhotonTransportLimit::
                                   ParticleCut);
                  }));
          if (particle_cuts !=
              pipeline.endpoints.particle_cut_count) {
            throw std::runtime_error(
                "projected photon cut count differs from device endpoint summary");
          }
        } else {
          auto const transport_record_begin =
              result.step_records.size();
          appendDevice(
              result.step_records,
              pipeline.transport.records,
              pipeline.transport.record_count,
              "download resident photon transport records");
          particle_cuts =
              static_cast<std::size_t>(std::count_if(
                  result.step_records.begin() +
                      static_cast<std::ptrdiff_t>(
                          transport_record_begin),
                  result.step_records.end(),
                  [](PhotonTransportRecord const& record) {
                    return record.limit ==
                           PhotonTransportLimit::ParticleCut;
                  }));
          if (particle_cuts !=
              pipeline.endpoints.particle_cut_count) {
            throw std::runtime_error(
                "photon cut count differs from device endpoint summary");
          }
        }
        result.particle_cuts += particle_cuts;
        statistics_.photon_transport_cuts += particle_cuts;
        result.interaction_vertices +=
            pipeline.at_interaction.interaction_count;
        result.lpm_suppressions +=
            pipeline.final_state.suppression_count;
        auto const terminal_transport_count =
            pipeline.at_interaction.interaction_count +
            pipeline.endpoints.observation_count +
            particle_cuts;
        if (pipeline.transport.record_count <
            terminal_transport_count) {
          throw std::runtime_error(
              "resident photon transport endpoint accounting underflow");
        }
        auto const layer_boundaries =
            pipeline.transport.record_count -
            terminal_transport_count;
        result.layer_boundaries += layer_boundaries;
        statistics_.photon_transport_interactions +=
            pipeline.at_interaction.interaction_count;
        statistics_.photon_transport_boundaries +=
            layer_boundaries;
        if (pipeline.at_interaction.interaction_count != 0) {
          ++statistics_.final_state_batches;
          ++statistics_
                .photon_final_state_endpoint_summary_fusions;
          ++statistics_
                .pipeline_host_synchronizations_eliminated;
          statistics_
              .pipeline_device_to_host_bytes_eliminated += 28;
        }
        statistics_.gpu_final_states +=
            pipeline.final_state.gpu_interaction_count;
        statistics_.physical_secondaries_generated +=
            pipeline.final_state.secondary_count;
        statistics_.photon_pair_lpm_trials +=
            pipeline.final_state.photon_pair_interaction_count +
            pipeline.final_state.suppression_count;
        statistics_.photon_pair_lpm_suppressions +=
            pipeline.final_state.suppression_count;
        statistics_.photon_pair_final_states +=
            pipeline.final_state.photon_pair_interaction_count;
        statistics_.compton_final_states +=
            pipeline.final_state.compton_interaction_count;
        statistics_.photoelectric_final_states +=
            pipeline.final_state
                .photoelectric_interaction_count;
        statistics_.proposal_fallbacks +=
            pipeline.selection.fallback_count +
            pipeline.transport.fallback_count +
            pipeline.final_state.fallback_count;
        if (pipeline.final_state.continuation_count != 0) {
          throw std::runtime_error(
              "resident transported interaction became a no-interaction continuation");
        }

        appendDevice(
            result.fallback_events, pipeline.selection.fallbacks,
            pipeline.selection.fallback_count,
            "download resident selection fallbacks");
        appendDevice(
            result.fallback_events, pipeline.transport.fallbacks,
            pipeline.transport.fallback_count,
            "download resident transport fallbacks");
        appendDevice(
            result.fallback_events, pipeline.final_state.fallbacks,
            pipeline.final_state.fallback_count,
            "download resident final-state fallbacks");
        auto leptons_kept_on_device =
            pipeline.final_state.secondary_count == 0;
        if (config_.resident_cross_species &&
            pipeline.endpoints
                .generated_leptons_compacted) {
          auto const selected =
              pipeline.endpoints.generated_lepton_count;
          auto const pending_tail =
              pending_lepton_head_ +
              pending_lepton_count_;
          if (selected >
                  pipeline.final_state.secondary_count ||
              pending_tail >
                  cross_species_queue_capacity_ ||
              selected >
                  cross_species_queue_capacity_ -
                      pending_tail) {
            throw std::runtime_error(
                "resident charged-secondary selection exceeded queue capacity");
          }
          pending_lepton_count_ += selected;
          statistics_
                  .cross_species_particles_kept_on_device +=
              selected;
          statistics_
                  .cross_species_device_to_device_bytes +=
              selected * sizeof(EmParticleState);
          statistics_.peak_pending_leptons =
              std::max(
                  statistics_.peak_pending_leptons,
                  pending_lepton_count_);
          leptons_kept_on_device = true;
        } else if (
            config_.resident_cross_species &&
            pipeline.final_state.secondary_count != 0) {
          std::vector<EmParticleState> generated_secondaries;
          appendDevice(
              generated_secondaries,
              pipeline.final_state.secondaries,
              pipeline.final_state.secondary_count,
              "download resident electromagnetic secondaries");
          std::vector<EmParticleState> generated_leptons;
          std::copy_if(
              generated_secondaries.begin(),
              generated_secondaries.end(),
              std::back_inserter(
                  generated_leptons),
              [](EmParticleState const& particle) {
                return particle.pid !=
                       static_cast<std::int32_t>(
                           EmPid::Photon);
              });
          auto spilled = rebalanceCrossSpeciesQueue(
              device_pending_leptons_,
              pending_lepton_head_,
              pending_lepton_count_,
              std::move(generated_leptons),
              "resident lepton cross-species queue");
          result.cpu_spill_particles.insert(
              result.cpu_spill_particles.end(),
              std::make_move_iterator(spilled.begin()),
              std::make_move_iterator(spilled.end()));
          statistics_.peak_pending_leptons =
              std::max(
                  statistics_.peak_pending_leptons,
                  pending_lepton_count_);
          leptons_kept_on_device = true;
        }
        if (!leptons_kept_on_device) {
          std::vector<EmParticleState> generated_secondaries;
          appendDevice(
              generated_secondaries,
              pipeline.final_state.secondaries,
              pipeline.final_state.secondary_count,
              "download resident electromagnetic secondaries");
          std::copy_if(
              generated_secondaries.begin(),
              generated_secondaries.end(),
              std::back_inserter(
                  result.electromagnetic_secondaries),
              [](EmParticleState const& particle) {
                return particle.pid !=
                       static_cast<std::int32_t>(
                           EmPid::Photon);
              });
        }
        if (!gpuProfileEnabled()) {
          auto const final_state_record_begin =
              result.final_state_records.size();
          appendDevice(
              result.final_state_records,
              pipeline.final_state.records,
              pipeline.final_state.gpu_interaction_count,
              "download resident final-state records");
          accumulateThinningStatistics(
              result.final_state_records,
              final_state_record_begin);
        }
        auto const observation_begin =
            result.observations.size();
        appendDevice(
            result.observations, pipeline.endpoints.observations,
            pipeline.endpoints.observation_count,
            "download resident observations");
        for (auto index = observation_begin;
             index < result.observations.size(); ++index) {
          if (result.observations[index].status ==
              ObservationStatus::ReachedObservationSurface) {
            ++statistics_.photon_transport_observations;
          } else {
            ++statistics_.photon_transport_escapes;
          }
        }

        if (pipeline.final_state.secondary_count >
            std::numeric_limits<std::uint64_t>::max() -
                next_secondary_history_id) {
          throw std::overflow_error(
              "resident photon cascade secondary history overflow");
        }
        next_secondary_history_id +=
            pipeline.final_state.secondary_count;
        current_particles =
            pipeline.endpoints.next_photons;
        current_count =
            pipeline.endpoints.next_photon_count;
        result.peak_resident_photons =
            std::max(result.peak_resident_photons, current_count);
        std::swap(current_workspace, output_workspace);
      }
      result.completed = current_count == 0;
      if (!result.completed) {
        result.remaining_photons.resize(current_count);
        auto const checkpoint_transfer_start =
            std::chrono::steady_clock::now();
        checkCuda(cudaMemcpy(
                      result.remaining_photons.data(),
                      current_particles,
                      current_count * sizeof(EmParticleState),
                      cudaMemcpyDeviceToHost),
                  "download resident photon cascade checkpoint");
        auto const checkpoint_transfer_stop =
            std::chrono::steady_clock::now();
        statistics_.transfer_time_ms +=
            std::chrono::duration<double, std::milli>(
                checkpoint_transfer_stop -
                checkpoint_transfer_start)
                .count();
        statistics_.physical_device_to_host_bytes +=
            current_count * sizeof(EmParticleState);
      }
      enrichFallbackEvents(result.fallback_events);
      return result;
    }

    ResidentLeptonCascadeResult
    runResidentLeptonCascadeForValidation(
        std::vector<EmParticleState> const& particles,
        std::uint64_t first_secondary_history_id,
        std::size_t maximum_wavefronts,
        std::uint64_t secondary_history_id_limit_exclusive,
        std::size_t minimum_resident_batch_size) {
      requireInitialized();
      if (!cuda_rate_table_.initialized()) {
        throw std::logic_error(
            "resident lepton cascade requires an uploaded PROPOSAL table");
      }
      if (first_secondary_history_id == 0 ||
          maximum_wavefronts == 0 ||
          secondary_history_id_limit_exclusive <=
              first_secondary_history_id ||
          minimum_resident_batch_size == 0) {
        throw std::invalid_argument(
            "resident lepton cascade requires nonzero history/wavefront limits");
      }
      ResidentLeptonCascadeResult result{};
      auto const pending_input =
          std::min(
              pending_lepton_count_,
              maximum_resident_lepton_batch_size_);
      auto const total_input = checkedAdd(
          particles.size(), pending_input,
          "resident lepton input count overflow");
      result.input_particles = total_input;
      if (total_input == 0) {
        result.completed = true;
        return result;
      }
      if (total_input >
          maximum_resident_lepton_batch_size_) {
        if (pending_input == 0) {
          result.workspace_limit_checkpoint = true;
          result.remaining_leptons = particles;
          return result;
        }
        throw std::runtime_error(
            "resident lepton host plus cross-species input exceeds workspace capacity");
      }
      for (auto const& particle : particles) {
        auto const lepton =
            isChargedLeptonPid(particle.pid);
        if (!lepton || !std::isfinite(particle.energy_GeV) ||
            !(particle.energy_GeV > 0.) ||
            !std::isfinite(particle.weight) ||
            particle.weight < 0. ||
            particle.history_id == 0) {
          throw std::invalid_argument(
              "resident lepton cascade accepts only valid charged leptons");
        }
        double norm_squared = 0.;
        for (std::size_t axis = 0; axis < 3; ++axis) {
          if (!std::isfinite(particle.position_m[axis]) ||
              !std::isfinite(particle.direction[axis])) {
            throw std::invalid_argument(
                "resident lepton cascade received a non-finite state");
          }
          norm_squared +=
              particle.direction[axis] *
              particle.direction[axis];
        }
        if (!std::isfinite(norm_squared) ||
            std::abs(norm_squared - 1.) > 1.e-12) {
          throw std::invalid_argument(
              "resident lepton direction is not normalized");
        }
      }

      auto appendDevice = [&](auto& destination, auto const* source,
                              std::size_t count,
                              char const* operation) {
        using Vector =
            std::decay_t<decltype(destination)>;
        using Value = typename Vector::value_type;
        if (count == 0) {
          return;
        }
        auto const old_size = destination.size();
        destination.resize(
            checkedAdd(old_size, count,
                       "resident lepton host output size overflow"));
        auto const transfer_start =
            std::chrono::steady_clock::now();
        checkCuda(cudaMemcpy(
                      destination.data() + old_size, source,
                      count * sizeof(Value),
                      cudaMemcpyDeviceToHost),
                  operation);
        auto const transfer_stop =
            std::chrono::steady_clock::now();
        statistics_.transfer_time_ms +=
            std::chrono::duration<double, std::milli>(
                transfer_stop - transfer_start)
                .count();
        statistics_.physical_device_to_host_bytes +=
            count * sizeof(Value);
      };

      detail::DeviceWorkspace* current_workspace =
          &physical_workspace_;
      detail::DeviceWorkspace* output_workspace =
          &physical_workspace_next_;
      waitForProfileInputSlot(
          physicalWorkspaceSlot(current_workspace));
      detail::WorkspaceSize input_required;
      input_required.add<EmParticleState>(total_input);
      current_workspace->prepare(input_required.bytes());
      auto* current_particles =
          current_workspace->acquire<EmParticleState>(
              total_input);
      auto const input_transfer_start =
          std::chrono::steady_clock::now();
      if (pending_input != 0) {
        checkCuda(
            cudaMemcpy(
                current_particles,
                device_pending_leptons_ +
                    pending_lepton_head_,
                pending_input * sizeof(EmParticleState),
                cudaMemcpyDeviceToDevice),
            "consume resident cross-species leptons");
        pending_lepton_head_ += pending_input;
        pending_lepton_count_ -= pending_input;
        if (pending_lepton_count_ == 0) {
          pending_lepton_head_ = 0;
        }
        statistics_.cross_species_device_to_device_bytes +=
            pending_input * sizeof(EmParticleState);
      }
      if (!particles.empty()) {
        checkCuda(cudaMemcpy(
                      current_particles + pending_input,
                      particles.data(),
                      particles.size() *
                          sizeof(EmParticleState),
                      cudaMemcpyHostToDevice),
                  "upload resident lepton cascade input");
      }
      auto const input_transfer_stop =
          std::chrono::steady_clock::now();
      statistics_.transfer_time_ms +=
          std::chrono::duration<double, std::milli>(
              input_transfer_stop -
              input_transfer_start)
              .count();
      statistics_.physical_host_to_device_bytes +=
          particles.size() * sizeof(EmParticleState);

      auto current_count = total_input;
      auto next_secondary_history_id =
          first_secondary_history_id;
      result.peak_resident_leptons = current_count;
      while (current_count != 0 &&
             result.wavefronts < maximum_wavefronts) {
        if (result.wavefronts != 0 &&
            current_count < minimum_resident_batch_size) {
          result.below_minimum_batch_checkpoint = true;
          break;
        }
        if (next_secondary_history_id >=
            secondary_history_id_limit_exclusive) {
          result.history_range_exhausted = true;
          break;
        }
        auto const remaining_history_ids =
            secondary_history_id_limit_exclusive -
            next_secondary_history_id;
        if (current_count >
            remaining_history_ids / 3) {
          result.history_range_exhausted = true;
          break;
        }
        detail::WorkspaceSize required;
        detail::appendWavefrontBucketingWorkspace(
            required, current_count);
        detail::appendLeptonDevicePipelineWorkspace(
            required, current_count);
        if (required.bytes() >
            output_workspace->byteLimit()) {
          result.workspace_limit_checkpoint = true;
          break;
        }
        auto const radio_input_slot =
            physicalWorkspaceSlot(output_workspace);
        waitForProfileInputSlot(radio_input_slot);
        radio_accumulator_.waitForInputSlot(
            radio_input_slot);
        output_workspace->prepare(required.bytes());
        auto const pipeline = measureCudaPipeline(
            [&] {
              auto const bucketed =
                  detail::
                      launchWavefrontBucketingOnDevice(
                          current_particles,
                          current_count,
                          *output_workspace);
              detail::LeptonPipelineStageEvents
                  stage_events{};
              stage_events.selection_done =
                  lepton_pipeline_stage_events_[
                      LeptonSelectionStage];
              stage_events.transport_physics_done =
                  lepton_pipeline_stage_events_[
                      LeptonTransportPhysicsStage];
              stage_events.moliere_done =
                  lepton_pipeline_stage_events_[
                      LeptonMoliereStage];
              stage_events.transport_control_done =
                  lepton_pipeline_stage_events_[
                      LeptonTransportControlStage];
              stage_events.transport_done =
                  lepton_pipeline_stage_events_[
                      LeptonTransportStage];
              stage_events.interaction_extraction_done =
                  lepton_pipeline_stage_events_[
                      LeptonInteractionExtractionStage];
              stage_events.vertex_selection_done =
                  lepton_pipeline_stage_events_[
                      LeptonVertexSelectionStage];
              stage_events.final_state_classification_done =
                  lepton_pipeline_stage_events_[
                      LeptonFinalStateClassificationStage];
              stage_events.final_state_scan_done =
                  lepton_pipeline_stage_events_[
                      LeptonFinalStateScanStage];
              stage_events.final_state_summary_done =
                  lepton_pipeline_stage_events_[
                      LeptonFinalStateSummaryStage];
              stage_events.final_state_write_done =
                  lepton_pipeline_stage_events_[
                      LeptonFinalStateWriteStage];
              stage_events.final_state_done =
                  lepton_pipeline_stage_events_[
                      LeptonFinalStateStage];
              stage_events.endpoint_compaction_done =
                  lepton_pipeline_stage_events_[
                      LeptonEndpointCompactionStage];
              auto launched = detail::
                  launchLeptonDevicePipelineOnDevice(
                      cuda_rate_table_.deviceView(),
                      brems_lpm_, brems_lpm_prepared_,
                      config_.thinning,
                      moliere_, muon_moliere_,
                      moliere_interpolation_,
                      moliere_available_,
                      muon_moliere_available_,
                      environment_, bucketed.particles,
                      current_count, config_.random_seed,
                      config_.shower_id,
                      next_secondary_history_id,
                      *output_workspace,
                      config_.detailed_stage_timing
                          ? &stage_events
                          : nullptr);
              radio_accumulator_.accumulateLeptonTracksOnDevice(
                  launched.transport.records,
                  launched.transport.record_count,
                  radio_input_slot);
              return launched;
            },
            statistics_.kernel_time_ms,
            physical_pipeline_start_,
            physical_pipeline_stop_);
        if (current_count >=
            detail::MinimumWavefrontRadixSortSize) {
          ++statistics_.wavefront_bucketing_batches;
          statistics_.wavefront_bucketing_particles +=
              current_count;
        } else {
          ++statistics_.wavefront_bucketing_small_batches;
          statistics_.wavefront_bucketing_small_particles +=
              current_count;
        }
        if (config_.detailed_stage_timing) {
          auto elapsed = [&](cudaEvent_t first,
                             cudaEvent_t second) {
            float milliseconds = 0.;
            checkCuda(
                cudaEventElapsedTime(
                    &milliseconds, first, second),
                "cudaEventElapsedTime(lepton pipeline stage)");
            return static_cast<double>(milliseconds);
          };
          auto& timing =
              statistics_.lepton_pipeline_timing;
          ++timing.wavefronts;
          timing.selection_ms += elapsed(
              physical_pipeline_start_,
              lepton_pipeline_stage_events_[
                  LeptonSelectionStage]);
          timing.transport_ms += elapsed(
              lepton_pipeline_stage_events_[
                  LeptonSelectionStage],
              lepton_pipeline_stage_events_[
                  LeptonTransportStage]);
          timing.transport_physics_ms += elapsed(
              lepton_pipeline_stage_events_[
                  LeptonSelectionStage],
              lepton_pipeline_stage_events_[
                  LeptonTransportPhysicsStage]);
          timing.moliere_ms += elapsed(
              lepton_pipeline_stage_events_[
                  LeptonTransportPhysicsStage],
              lepton_pipeline_stage_events_[
                  LeptonMoliereStage]);
          timing.transport_control_ms += elapsed(
              lepton_pipeline_stage_events_[
                  LeptonMoliereStage],
              lepton_pipeline_stage_events_[
                  LeptonTransportControlStage]);
          timing.transport_compaction_ms += elapsed(
              lepton_pipeline_stage_events_[
                  LeptonTransportControlStage],
              lepton_pipeline_stage_events_[
                  LeptonTransportStage]);
          timing.interaction_extraction_ms += elapsed(
              lepton_pipeline_stage_events_[
                  LeptonTransportStage],
              lepton_pipeline_stage_events_[
                  LeptonInteractionExtractionStage]);
          timing.vertex_selection_ms += elapsed(
              lepton_pipeline_stage_events_[
                  LeptonInteractionExtractionStage],
              lepton_pipeline_stage_events_[
                  LeptonVertexSelectionStage]);
          timing.final_state_ms += elapsed(
              lepton_pipeline_stage_events_[
                  LeptonVertexSelectionStage],
              lepton_pipeline_stage_events_[
                  LeptonFinalStateStage]);
          timing.final_state_classification_ms += elapsed(
              lepton_pipeline_stage_events_[
                  LeptonVertexSelectionStage],
              lepton_pipeline_stage_events_[
                  LeptonFinalStateClassificationStage]);
          timing.final_state_scan_ms += elapsed(
              lepton_pipeline_stage_events_[
                  LeptonFinalStateClassificationStage],
              lepton_pipeline_stage_events_[
                  LeptonFinalStateScanStage]);
          timing.final_state_summary_ms += elapsed(
              lepton_pipeline_stage_events_[
                  LeptonFinalStateScanStage],
              lepton_pipeline_stage_events_[
                  LeptonFinalStateSummaryStage]);
          timing.final_state_write_ms += elapsed(
              lepton_pipeline_stage_events_[
                  LeptonFinalStateSummaryStage],
              lepton_pipeline_stage_events_[
                  LeptonFinalStateWriteStage]);
          timing.endpoint_compaction_ms += elapsed(
              lepton_pipeline_stage_events_[
                  LeptonFinalStateStage],
              lepton_pipeline_stage_events_[
                  LeptonEndpointCompactionStage]);
          timing.post_endpoint_ms += elapsed(
              lepton_pipeline_stage_events_[
                  LeptonEndpointCompactionStage],
              physical_pipeline_stop_);
        }
        updateWorkspaceStatistics();
        ++result.wavefronts;
        ++statistics_.physical_lepton_wavefronts;
        ++statistics_
              .lepton_selection_transport_summary_fusions;
        ++statistics_
              .pipeline_host_synchronizations_eliminated;
        if (pipeline.transport.record_count != 0) {
          ++statistics_
                .lepton_transport_vertex_summary_fusions;
          ++statistics_
                .pipeline_host_synchronizations_eliminated;
          if (pipeline.at_interaction.interaction_count != 0) {
            statistics_
                .pipeline_device_to_host_bytes_eliminated += 12;
            ++statistics_
                  .lepton_vertex_final_state_summary_fusions;
            ++statistics_
                  .pipeline_host_synchronizations_eliminated;
          }
        }
        ++statistics_.interaction_selection_batches;
        ++statistics_.lepton_transport_batches;
        statistics_.interactions_selected +=
            pipeline.selection.interaction_count;
        result.transport_records +=
            pipeline.transport.record_count;
        result.interaction_vertices +=
            pipeline.at_interaction.interaction_count;
        result.lpm_suppressions +=
            pipeline.final_state.suppression_count;

        auto accumulate_projected_transport_statistics =
            [&](ProjectedEmStepRecord const& record) {
          if (moliere_available_) {
            ++statistics_.moliere_trials;
            auto const packed =
                static_cast<std::uint32_t>(record.reserved);
            auto const iterations = packed >> 8;
            statistics_.moliere_newton_iterations +=
                iterations;
            statistics_.moliere_max_newton_iterations =
                std::max(
                    statistics_.moliere_max_newton_iterations,
                    iterations);
            if ((record.reserved & 1) != 0) {
              ++statistics_.moliere_deflections;
            } else if (
                ((packed >> 1) & 0x7fU) ==
                static_cast<std::uint32_t>(
                    MoliereStatus::NoDeflection)) {
              ++statistics_.moliere_zero_deflections;
            }
          }
          switch (static_cast<LeptonTransportLimit>(
              record.transport_limit)) {
          case LeptonTransportLimit::InteractionCandidate:
            ++statistics_
                  .lepton_transport_interaction_candidates;
            break;
          case LeptonTransportLimit::ContinuousStep:
            ++statistics_.lepton_transport_continuous_steps;
            break;
          case LeptonTransportLimit::ParticleCut:
            ++statistics_.lepton_transport_cuts;
            break;
          case LeptonTransportLimit::LayerBoundary:
            ++statistics_.lepton_transport_boundaries;
            break;
          case LeptonTransportLimit::ObservationSurface:
            ++statistics_.lepton_transport_observations;
            break;
          case LeptonTransportLimit::EscapedEnvironment:
            ++statistics_.lepton_transport_escapes;
            break;
          case LeptonTransportLimit::MagneticStep:
            ++statistics_.lepton_transport_magnetic_steps;
            break;
          case LeptonTransportLimit::DecayCandidate:
            ++statistics_.lepton_transport_decay_candidates;
            break;
          }
        };
        if (gpuProfileEnabled()) {
          scheduleProfileAccumulation(
              radio_input_slot,
              [&](cudaStream_t stream) {
                detail::
                    launchLeptonProfileAccumulationOnDevice(
                        device_profile_projection_,
                        device_profile_accumulator_,
                        pipeline.transport.records,
                        pipeline.transport.record_count,
                        pipeline.final_state.records,
                        pipeline.final_state
                            .gpu_interaction_count,
                        stream);
              });
        } else if (config_.profile_projection.enabled) {
          // See the photon path above: the selection workspace is no longer
          // live and is large enough for one projected record per input.
          auto* projected_records =
              reinterpret_cast<ProjectedEmStepRecord*>(
                  pipeline.selection.interactions);
          measureCudaPipeline(
              [&] {
                detail::
                    launchLeptonProfileProjectionOnDevice(
                        device_profile_projection_,
                        pipeline.transport.records,
                        pipeline.transport.record_count,
                        projected_records);
                return 0;
              },
              statistics_.kernel_time_ms,
              physical_pipeline_start_,
              physical_pipeline_stop_);
          auto const old_step_count =
              result.projected_step_records.size();
          appendDevice(
              result.projected_step_records,
              projected_records,
              pipeline.transport.record_count,
              "download projected resident lepton steps");
          for (auto index = old_step_count;
               index <
               result.projected_step_records.size();
               ++index) {
            accumulate_projected_transport_statistics(
                result.projected_step_records[index]);
          }
        } else {
          auto const old_step_count =
              result.step_records.size();
          appendDevice(
              result.step_records,
              pipeline.transport.records,
              pipeline.transport.record_count,
              "download resident lepton transport records");
          for (auto index = old_step_count;
               index < result.step_records.size(); ++index) {
            auto const& record = result.step_records[index];
          if (moliere_available_) {
            ++statistics_.moliere_trials;
            statistics_.moliere_newton_iterations +=
                record.multiple_scattering_iterations;
            statistics_.moliere_max_newton_iterations =
                std::max(
                    statistics_.moliere_max_newton_iterations,
                    record.multiple_scattering_iterations);
            if (record.multiple_scattering_applied != 0) {
              ++statistics_.moliere_deflections;
            } else if (
                record.multiple_scattering_status ==
                static_cast<std::uint32_t>(
                    MoliereStatus::NoDeflection)) {
              ++statistics_.moliere_zero_deflections;
            }
          }
          switch (record.limit) {
          case LeptonTransportLimit::InteractionCandidate:
            ++statistics_
                  .lepton_transport_interaction_candidates;
            break;
          case LeptonTransportLimit::ContinuousStep:
            ++statistics_.lepton_transport_continuous_steps;
            break;
          case LeptonTransportLimit::ParticleCut:
            ++statistics_.lepton_transport_cuts;
            break;
          case LeptonTransportLimit::LayerBoundary:
            ++statistics_.lepton_transport_boundaries;
            break;
          case LeptonTransportLimit::ObservationSurface:
            ++statistics_.lepton_transport_observations;
            break;
          case LeptonTransportLimit::EscapedEnvironment:
            ++statistics_.lepton_transport_escapes;
            break;
          case LeptonTransportLimit::MagneticStep:
            ++statistics_.lepton_transport_magnetic_steps;
            break;
          case LeptonTransportLimit::DecayCandidate:
            ++statistics_.lepton_transport_decay_candidates;
            break;
          }
          }
        }
        if (pipeline.at_interaction.interaction_count != 0) {
          ++statistics_.lepton_vertex_selection_batches;
        }
        statistics_.lepton_vertex_interactions_selected +=
            pipeline.vertex.interaction_count;
        statistics_
            .lepton_vertex_no_interaction_continuations +=
            pipeline.vertex.continuation_count;
        if (pipeline.final_state.input_count != 0) {
          ++statistics_.final_state_batches;
          ++statistics_
                .lepton_final_state_endpoint_summary_fusions;
          ++statistics_
                .pipeline_host_synchronizations_eliminated;
          // Before the device control summary, this stage downloaded the
          // final entries of 22 scan arrays (88 bytes), followed by the
          // 5-word endpoint summary (20 bytes). The fused endpoint summary
          // now contains 25 words (100 bytes), including the vertex,
          // final-state and Epair rejection diagnostics. It still saves
          // 8 bytes and one blocking device-to-host transfer per non-empty
          // final-state batch.
          statistics_
              .pipeline_device_to_host_bytes_eliminated += 8;
        }
        statistics_.gpu_final_states +=
            pipeline.final_state.gpu_interaction_count;
        statistics_.physical_secondaries_generated +=
            pipeline.final_state.secondary_count;
        statistics_.brems_lpm_trials +=
            pipeline.final_state.brems_lpm_trial_count;
        statistics_.brems_lpm_suppressions +=
            pipeline.final_state.brems_lpm_suppression_count;
        statistics_.brems_final_states +=
            pipeline.final_state.brems_interaction_count;
        statistics_.electron_pair_lpm_trials +=
            pipeline.final_state.electron_pair_lpm_trial_count;
        statistics_.electron_pair_lpm_suppressions +=
            pipeline.final_state
                .electron_pair_lpm_suppression_count;
        statistics_.electron_pair_rejection_trials +=
            pipeline.final_state
                .electron_pair_rejection_trials;
        statistics_.electron_pair_zero_weight_samples +=
            pipeline.final_state
                .electron_pair_zero_weight_samples;
        statistics_.electron_pair_rejection_fallbacks +=
            pipeline.final_state
                .electron_pair_rejection_fallbacks;
        statistics_.electron_pair_envelope_violations +=
            pipeline.final_state
                .electron_pair_envelope_violations;
        statistics_.annihilation_final_states +=
            pipeline.final_state.annihilation_interaction_count;
        statistics_.ionization_final_states +=
            pipeline.final_state.ionization_interaction_count;
        statistics_.electron_pair_final_states +=
            pipeline.final_state
                .electron_pair_interaction_count;
        statistics_.proposal_fallbacks +=
            pipeline.selection.fallback_count +
            pipeline.transport.fallback_count +
            pipeline.vertex.fallback_count +
            pipeline.final_state.fallback_count;

        appendDevice(
            result.fallback_events,
            pipeline.selection.fallbacks,
            pipeline.selection.fallback_count,
            "download resident lepton selection fallbacks");
        appendDevice(
            result.fallback_events,
            pipeline.transport.fallbacks,
            pipeline.transport.fallback_count,
            "download resident lepton transport fallbacks");
        appendDevice(
            result.fallback_events,
            pipeline.vertex.fallbacks,
            pipeline.vertex.fallback_count,
            "download resident lepton vertex fallbacks");
        appendDevice(
            result.fallback_events,
            pipeline.final_state.fallbacks,
            pipeline.final_state.fallback_count,
            "download resident lepton final-state fallbacks");
        if (config_.resident_cross_species) {
          auto spilled = appendPendingPhotons(
              pipeline.endpoints.generated_photons,
              pipeline.endpoints.generated_photon_count);
          result.cpu_spill_particles.insert(
              result.cpu_spill_particles.end(),
              std::make_move_iterator(spilled.begin()),
              std::make_move_iterator(spilled.end()));
        } else {
          appendDevice(
              result.generated_photons,
              pipeline.endpoints.generated_photons,
              pipeline.endpoints.generated_photon_count,
              "download resident lepton generated photons");
        }
        if (!gpuProfileEnabled()) {
          auto const final_state_record_begin =
              result.final_state_records.size();
          appendDevice(
              result.final_state_records,
              pipeline.final_state.records,
              pipeline.final_state.gpu_interaction_count,
              "download resident lepton final-state records");
          accumulateThinningStatistics(
              result.final_state_records,
              final_state_record_begin);
        }
        appendDevice(
            result.observations,
            pipeline.endpoints.observations,
            pipeline.endpoints.observation_count,
            "download resident lepton observations");
        appendDevice(
            result.decay_candidates,
            pipeline.endpoints.decay_candidates,
            pipeline.endpoints.decay_candidate_count,
            "download resident lepton decay candidates");

        if (pipeline.final_state.secondary_count >
            std::numeric_limits<std::uint64_t>::max() -
                next_secondary_history_id) {
          throw std::overflow_error(
              "resident lepton cascade secondary history overflow");
        }
        next_secondary_history_id +=
            pipeline.final_state.secondary_count;
        current_particles =
            pipeline.endpoints.next_leptons;
        current_count =
            pipeline.endpoints.next_lepton_count;
        result.peak_resident_leptons =
            std::max(
                result.peak_resident_leptons,
                current_count);
        std::swap(current_workspace, output_workspace);
      }
      result.secondary_history_ids_used =
          next_secondary_history_id -
          first_secondary_history_id;
      result.completed = current_count == 0;
      if (!result.completed) {
        result.remaining_leptons.resize(current_count);
        auto const checkpoint_transfer_start =
            std::chrono::steady_clock::now();
        checkCuda(cudaMemcpy(
                      result.remaining_leptons.data(),
                      current_particles,
                      current_count * sizeof(EmParticleState),
                      cudaMemcpyDeviceToHost),
                  "download resident lepton cascade checkpoint");
        auto const checkpoint_transfer_stop =
            std::chrono::steady_clock::now();
        statistics_.transfer_time_ms +=
            std::chrono::duration<double, std::milli>(
                checkpoint_transfer_stop -
                checkpoint_transfer_start)
                .count();
        statistics_.physical_device_to_host_bytes +=
            current_count * sizeof(EmParticleState);
      }
      radio_accumulator_.drain();
      enrichFallbackEvents(result.fallback_events);
      return result;
    }

    PhotonWavefrontBatchResult
    advancePhotonWavefrontForValidation(
        std::vector<EmParticleState> const& particles,
        std::uint64_t first_secondary_history_id) {
      requireInitialized();
      PhotonWavefrontBatchResult result{};
      result.input_particles = particles.size();

      auto pipeline = runPhysicalPhotonPipeline(
          particles, first_secondary_history_id);
      result.selection_fallback_events =
          std::move(
              pipeline.selection_fallback_events);
      result.transport_fallback_events =
          std::move(
              pipeline.transport_fallback_events);
      result.transport_records =
          std::move(pipeline.transport_records);
      result.final_states = std::move(pipeline.final_states);
      result.next_photons = std::move(pipeline.next_photons);
      result.observations = std::move(pipeline.observations);
      statistics_.physical_photon_wavefronts++;
      return result;
    }

    std::vector<EmParticleState> downloadActiveParticles() const {
      requireInitialized();
      std::vector<EmParticleState> particles(current_size_);
      if (current_size_ == 0) {
        return particles;
      }
      auto const start = std::chrono::steady_clock::now();
      auto const blocks = static_cast<unsigned int>(
          (current_size_ + ThreadsPerBlock - 1) / ThreadsPerBlock);
      gatherParticles<<<blocks, ThreadsPerBlock>>>(
          device_current_, current_size_, device_staging_);
      checkCuda(cudaGetLastError(), "gather active GPU particles launch");
      checkCuda(cudaMemcpy(particles.data(), device_staging_,
                           current_size_ * sizeof(EmParticleState),
                           cudaMemcpyDeviceToHost),
                "download active GPU particles");
      auto const stop = std::chrono::steady_clock::now();
      statistics_.transfer_time_ms +=
          std::chrono::duration<double, std::milli>(stop - start).count();
      return particles;
    }

    std::vector<EmParticleState> extractActiveParticlesForTesting() {
      if (!host_staging_.empty()) {
        throw std::logic_error(
            "Cannot extract a GPU wavefront while host particles are staged");
      }
      auto particles = downloadActiveParticles();
      current_size_ = 0;
      statistics_.current_particles = 0;
      return particles;
    }

  private:
    void resetStatisticsForShower(bool reused) {
      GpuEmStatistics next{};
      next.shower_ordinal = shower_ordinal_;
      next.reused_for_shower = reused;
      next.static_host_to_device_bytes =
          static_host_to_device_bytes_;
      next.one_time_initialization_ms =
          one_time_initialization_ms_;
      next.reserved_particles = capacity_;
      next.table_device_bytes =
          checkedAdd(
              cuda_rate_table_.deviceBytes(),
              moliere_interpolation_device_bytes_,
              "CUDA table statistics overflow");
      next.physical_workspace_bytes =
          checkedAdd(
              physical_workspace_.capacityBytes(),
              physical_workspace_next_.capacityBytes(),
              "CUDA workspace statistics overflow");
      next.maximum_resident_photon_batch =
          maximum_resident_photon_batch_size_;
      next.maximum_resident_lepton_batch =
          maximum_resident_lepton_batch_size_;
      next.cross_species_queue_device_bytes =
          cross_species_queue_device_bytes_;
      next.cross_species_queue_capacity_per_pid =
          cross_species_queue_capacity_;
      next.profile.enabled = gpuProfileEnabled();
      next.profile.deterministic =
          gpuProfileEnabled();
      next.profile.bins =
          device_profile_accumulator_.bins;
      next.profile.device_bytes =
          profile_accumulator_device_bytes_;
      next.radio = radio_accumulator_.statistics();
      next.lepton_pipeline_timing.enabled =
          config_.detailed_stage_timing;
      next.peak_device_bytes =
          checkedAdd(
              resident_allocation_bytes_,
              next.physical_workspace_bytes,
              "CUDA resident allocation statistics overflow");
      statistics_ = next;
    }

    std::vector<EmParticleState>
    rebalanceCrossSpeciesQueue(
        EmParticleState* device_queue,
        std::size_t& pending_head,
        std::size_t& pending_count,
        std::vector<EmParticleState> incoming,
        char const* operation) {
      if (device_queue == nullptr ||
          pending_head >
              cross_species_queue_capacity_ ||
          pending_count >
              cross_species_queue_capacity_ -
                  pending_head) {
        throw std::logic_error(
            "invalid resident cross-species queue state");
      }
      auto const combined_count = checkedAdd(
          pending_count, incoming.size(),
          "cross-species spill candidate count overflow");
      struct TaggedQueueParticle {
        EmParticleState particle;
        bool is_incoming{};
      };
      std::vector<TaggedQueueParticle> combined;
      combined.reserve(combined_count);
      if (pending_count != 0) {
        std::vector<EmParticleState> pending(pending_count);
        auto const download_start =
            std::chrono::steady_clock::now();
        checkCuda(
            cudaMemcpy(
                pending.data(),
                device_queue + pending_head,
                pending_count *
                    sizeof(EmParticleState),
                cudaMemcpyDeviceToHost),
            operation);
        auto const download_stop =
            std::chrono::steady_clock::now();
        statistics_.transfer_time_ms +=
            std::chrono::duration<double, std::milli>(
                download_stop - download_start)
                .count();
        statistics_.physical_device_to_host_bytes +=
            pending_count * sizeof(EmParticleState);
        for (auto& particle : pending) {
          combined.push_back(
              TaggedQueueParticle{
                  std::move(particle), false});
        }
      }
      for (auto& particle : incoming) {
        combined.push_back(
            TaggedQueueParticle{
                std::move(particle), true});
      }
      std::stable_sort(
          combined.begin(), combined.end(),
          [](TaggedQueueParticle const& left,
             TaggedQueueParticle const& right) {
            return left.particle.energy_GeV >
                   right.particle.energy_GeV;
          });

      auto const retained_count =
          std::min(
              combined.size(),
              cross_species_queue_capacity_);
      auto const spill_count =
          combined.size() - retained_count;
      if (spill_count != 0) {
        if (retained_count == 0 ||
            combined[retained_count - 1]
                    .particle.energy_GeV <
                combined[retained_count]
                    .particle.energy_GeV) {
          throw std::logic_error(
              "cross-species spill did not select the lowest-energy particles");
        }
        ++statistics_.cross_species_host_spills;
        statistics_.cross_species_particles_spilled_to_cpu +=
            spill_count;
        ++statistics_
              .cross_species_low_energy_ordering_checks;
      }

      std::vector<EmParticleState> retained;
      retained.reserve(retained_count);
      std::size_t retained_incoming = 0;
      for (std::size_t index = 0;
           index < retained_count; ++index) {
        retained_incoming +=
            combined[index].is_incoming ? 1 : 0;
        retained.push_back(
            std::move(combined[index].particle));
      }
      if (retained_count != 0) {
        auto const upload_start =
            std::chrono::steady_clock::now();
        checkCuda(
            cudaMemcpy(
                device_queue, retained.data(),
                retained_count *
                    sizeof(EmParticleState),
                cudaMemcpyHostToDevice),
            operation);
        auto const upload_stop =
            std::chrono::steady_clock::now();
        statistics_.transfer_time_ms +=
            std::chrono::duration<double, std::milli>(
                upload_stop - upload_start)
                .count();
        statistics_.physical_host_to_device_bytes +=
            retained_count * sizeof(EmParticleState);
      }
      statistics_.cross_species_particles_kept_on_device +=
          retained_incoming;
      pending_head = 0;
      pending_count = retained_count;
      ++statistics_.cross_species_spill_rebalances;

      std::vector<EmParticleState> spilled;
      spilled.reserve(spill_count);
      for (std::size_t index = retained_count;
           index < combined.size(); ++index) {
        spilled.push_back(
            std::move(combined[index].particle));
      }
      return spilled;
    }

    std::vector<EmParticleState> appendPendingPhotons(
        EmParticleState const* source, std::size_t count) {
      if (count == 0) {
        return {};
      }
      auto const pending_tail =
          pending_photon_head_ +
          pending_photon_count_;
      if (pending_tail >
              cross_species_queue_capacity_ ||
          count >
              cross_species_queue_capacity_ -
                  pending_tail) {
        std::vector<EmParticleState> incoming(count);
        auto const download_start =
            std::chrono::steady_clock::now();
        checkCuda(
            cudaMemcpy(
                incoming.data(), source,
                count * sizeof(EmParticleState),
                cudaMemcpyDeviceToHost),
            "download photon spill candidates");
        auto const download_stop =
            std::chrono::steady_clock::now();
        statistics_.transfer_time_ms +=
            std::chrono::duration<double, std::milli>(
                download_stop - download_start)
                .count();
        statistics_.physical_device_to_host_bytes +=
            count * sizeof(EmParticleState);
        auto spilled = rebalanceCrossSpeciesQueue(
            device_pending_photons_,
            pending_photon_head_,
            pending_photon_count_,
            std::move(incoming),
            "rebalance resident photon cross-species queue");
        statistics_.peak_pending_photons =
            std::max(
                statistics_.peak_pending_photons,
                pending_photon_count_);
        return spilled;
      }
      checkCuda(
          cudaMemcpy(
              device_pending_photons_ +
                  pending_tail,
              source, count * sizeof(EmParticleState),
              cudaMemcpyDeviceToDevice),
          "append resident cross-species photons");
      pending_photon_count_ += count;
      statistics_.cross_species_particles_kept_on_device +=
          count;
      statistics_.cross_species_device_to_device_bytes +=
          count * sizeof(EmParticleState);
      statistics_.peak_pending_photons =
          std::max(
              statistics_.peak_pending_photons,
              pending_photon_count_);
      return {};
    }

    std::size_t physicalWorkspaceSlot(
        detail::DeviceWorkspace const* workspace) const {
      if (workspace == &physical_workspace_) {
        return 0;
      }
      if (workspace == &physical_workspace_next_) {
        return 1;
      }
      throw std::logic_error(
          "unknown CUDA physical workspace");
    }

    void waitForProfileInputSlot(std::size_t slot) {
      if (slot >= profile_slot_pending_.size()) {
        throw std::out_of_range(
            "resident GPU profile slot is out of range");
      }
      if (!profile_slot_pending_[slot]) {
        return;
      }
      checkCuda(
          cudaEventSynchronize(profile_done_events_[slot]),
          "wait for resident GPU profile input slot");
      float elapsed_ms = 0.;
      checkCuda(
          cudaEventElapsedTime(
              &elapsed_ms, profile_start_events_[slot],
              profile_done_events_[slot]),
          "measure resident GPU profile kernel");
      statistics_.profile.kernel_time_ms += elapsed_ms;
      statistics_.kernel_time_ms += elapsed_ms;
      profile_slot_pending_[slot] = false;
    }

    template <typename Function>
    void scheduleProfileAccumulation(
        std::size_t slot, Function&& launch) {
      if (!gpuProfileEnabled()) {
        return;
      }
      if (slot >= profile_slot_pending_.size() ||
          profile_slot_pending_[slot]) {
        throw std::logic_error(
            "resident GPU profile input slot was reused before completion");
      }
      checkCuda(
          cudaEventRecord(
              profile_start_events_[slot],
              profile_stream_),
          "record resident GPU profile start");
      launch(profile_stream_);
      checkCuda(
          cudaEventRecord(
              profile_done_events_[slot],
              profile_stream_),
          "record resident GPU profile completion");
      profile_slot_pending_[slot] = true;
    }

    void drainProfileStream() {
      if (!gpuProfileEnabled()) {
        return;
      }
      for (std::size_t slot = 0;
           slot < profile_slot_pending_.size(); ++slot) {
        waitForProfileInputSlot(slot);
      }
    }

    detail::DeviceProfileCounters refreshProfileCounters() {
      if (!gpuProfileEnabled()) {
        return {};
      }
      detail::DeviceProfileCounters current{};
      auto const transfer_start =
          std::chrono::steady_clock::now();
      checkCuda(
          cudaMemcpy(
              &current, device_profile_counters_,
              sizeof(current), cudaMemcpyDeviceToHost),
          "download resident GPU profile counters");
      auto const transfer_stop =
          std::chrono::steady_clock::now();
      auto const transfer_ms =
          std::chrono::duration<double, std::milli>(
              transfer_stop - transfer_start)
              .count();
      statistics_.transfer_time_ms += transfer_ms;
      statistics_.profile.transfer_time_ms += transfer_ms;
      statistics_.physical_device_to_host_bytes +=
          sizeof(current);
      statistics_.profile.device_to_host_bytes +=
          sizeof(current);

      auto subtract = [](unsigned long long value,
                         unsigned long long previous,
                         char const* field) {
        if (value < previous) {
          throw std::runtime_error(
              std::string{
                  "resident GPU profile counter decreased: "} +
              field);
        }
        return value - previous;
      };
      detail::DeviceProfileCounters delta{};
      delta.steps = subtract(
          current.steps, profile_counter_snapshot_.steps,
          "steps");
      delta.deposited_steps = subtract(
          current.deposited_steps,
          profile_counter_snapshot_.deposited_steps,
          "deposited steps");
      delta.photon_cuts = subtract(
          current.photon_cuts,
          profile_counter_snapshot_.photon_cuts,
          "photon cuts");
      for (std::size_t index = 0; index < 8; ++index) {
        delta.lepton_limits[index] = subtract(
            current.lepton_limits[index],
            profile_counter_snapshot_
                .lepton_limits[index],
            "lepton limits");
      }
      delta.moliere_trials = subtract(
          current.moliere_trials,
          profile_counter_snapshot_.moliere_trials,
          "Moliere trials");
      delta.moliere_deflections = subtract(
          current.moliere_deflections,
          profile_counter_snapshot_.moliere_deflections,
          "Moliere deflections");
      delta.moliere_zero_deflections = subtract(
          current.moliere_zero_deflections,
          profile_counter_snapshot_
              .moliere_zero_deflections,
          "Moliere zero deflections");
      delta.moliere_newton_iterations = subtract(
          current.moliere_newton_iterations,
          profile_counter_snapshot_
              .moliere_newton_iterations,
          "Moliere Newton iterations");
      delta.thinning_hillas_vertices = subtract(
          current.thinning_hillas_vertices,
          profile_counter_snapshot_
              .thinning_hillas_vertices,
          "Hillas vertices");
      delta.thinning_statistical_vertices = subtract(
          current.thinning_statistical_vertices,
          profile_counter_snapshot_
              .thinning_statistical_vertices,
          "statistical thinning vertices");
      delta.thinning_particles_discarded = subtract(
          current.thinning_particles_discarded,
          profile_counter_snapshot_
              .thinning_particles_discarded,
          "thinned particles");
      delta.fixed_point_overflows = subtract(
          current.fixed_point_overflows,
          profile_counter_snapshot_
              .fixed_point_overflows,
          "fixed-point overflows");
      delta.invalid_records = subtract(
          current.invalid_records,
          profile_counter_snapshot_.invalid_records,
          "invalid records");
      profile_counter_snapshot_ = current;

      statistics_.profile.steps = current.steps;
      statistics_.profile.deposited_steps =
          current.deposited_steps;
      statistics_.profile.fixed_point_overflows =
          current.fixed_point_overflows;
      statistics_.profile.invalid_records =
          current.invalid_records;
      statistics_.thinning_hillas_vertices +=
          delta.thinning_hillas_vertices;
      statistics_.thinning_statistical_vertices +=
          delta.thinning_statistical_vertices;
      statistics_.thinning_particles_discarded +=
          delta.thinning_particles_discarded;
      if (moliere_available_) {
        statistics_.moliere_trials +=
            delta.moliere_trials;
        statistics_.moliere_deflections +=
            delta.moliere_deflections;
        statistics_.moliere_zero_deflections +=
            delta.moliere_zero_deflections;
        statistics_.moliere_newton_iterations +=
            delta.moliere_newton_iterations;
        statistics_.moliere_max_newton_iterations =
            std::max(
                statistics_.moliere_max_newton_iterations,
                static_cast<std::uint32_t>(
                    current.moliere_max_newton_iterations));
      }
      statistics_
          .lepton_transport_interaction_candidates +=
          delta.lepton_limits[
              static_cast<std::size_t>(
                  LeptonTransportLimit::
                      InteractionCandidate)];
      statistics_.lepton_transport_continuous_steps +=
          delta.lepton_limits[
              static_cast<std::size_t>(
                  LeptonTransportLimit::ContinuousStep)];
      statistics_.lepton_transport_cuts +=
          delta.lepton_limits[
              static_cast<std::size_t>(
                  LeptonTransportLimit::ParticleCut)];
      statistics_.lepton_transport_boundaries +=
          delta.lepton_limits[
              static_cast<std::size_t>(
                  LeptonTransportLimit::LayerBoundary)];
      statistics_.lepton_transport_observations +=
          delta.lepton_limits[
              static_cast<std::size_t>(
                  LeptonTransportLimit::
                      ObservationSurface)];
      statistics_.lepton_transport_escapes +=
          delta.lepton_limits[
              static_cast<std::size_t>(
                  LeptonTransportLimit::
                      EscapedEnvironment)];
      statistics_.lepton_transport_magnetic_steps +=
          delta.lepton_limits[
              static_cast<std::size_t>(
                  LeptonTransportLimit::MagneticStep)];
      statistics_.lepton_transport_decay_candidates +=
          delta.lepton_limits[
              static_cast<std::size_t>(
                  LeptonTransportLimit::DecayCandidate)];
      if (current.fixed_point_overflows != 0) {
        throw std::runtime_error(
            "resident GPU profile fixed-point accumulator overflowed");
      }
      if (current.invalid_records != 0) {
        throw std::runtime_error(
            "resident GPU profile encountered an invalid transport or final-state record");
      }
      return delta;
    }

    std::uint64_t interactionHashForPid(
        std::int32_t pid) const {
      auto const found = std::find_if(
          interaction_hashes_.begin(), interaction_hashes_.end(),
          [pid](auto const& entry) {
            return entry.first == pid;
          });
      return found == interaction_hashes_.end()
                 ? std::uint64_t{}
                 : found->second;
    }

    void enrichFallbackEvent(
        ProposalFallbackEvent& event) const {
      event.medium_hash = proposal_medium_hash_;
      event.interaction_hash =
          interactionHashForPid(event.particle.pid);
    }

    void enrichFallbackEvents(
        std::vector<ProposalFallbackEvent>& events) const {
      for (auto& event : events) {
        enrichFallbackEvent(event);
      }
    }

    void accumulateTransportStatistics(
        std::vector<PhotonTransportRecord> const& records) {
      for (auto const& record : records) {
        switch (record.limit) {
          case PhotonTransportLimit::Interaction:
            statistics_.photon_transport_interactions++;
            break;
          case PhotonTransportLimit::LayerBoundary:
            statistics_.photon_transport_boundaries++;
            break;
          case PhotonTransportLimit::ObservationSurface:
            statistics_.photon_transport_observations++;
            break;
          case PhotonTransportLimit::EscapedEnvironment:
            statistics_.photon_transport_escapes++;
            break;
          case PhotonTransportLimit::ParticleCut:
            statistics_.photon_transport_cuts++;
            break;
        }
      }
    }

    void requireInitialized() const {
      if (!initialized_) {
        throw std::logic_error("CUDA EM backend is not initialized");
      }
    }

    std::size_t allocatedBytes(std::size_t capacity,
                               std::size_t scan_bytes) const {
      constexpr std::size_t SoABytesPerParticle =
          4 * sizeof(std::uint32_t) + 9 * sizeof(double) +
          3 * sizeof(std::uint64_t);
      static_assert(SoABytesPerParticle == sizeof(EmParticleState));
      constexpr std::size_t BytesPerParticle =
          3 * SoABytesPerParticle + sizeof(std::uint32_t) +
          sizeof(std::size_t);
      if (capacity >
          (std::numeric_limits<std::size_t>::max() - scan_bytes) /
              BytesPerParticle) {
        return std::numeric_limits<std::size_t>::max();
      }
      return capacity * BytesPerParticle + scan_bytes;
    }

    static void allocateQueue(DeviceParticleSoA& queue,
                              std::size_t capacity) {
      std::int32_t* words32 = nullptr;
      double* reals = nullptr;
      std::uint64_t* words64 = nullptr;
      try {
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&words32),
                             4 * capacity * sizeof(std::uint32_t)),
                  "allocate GPU particle 32-bit SoA fields");
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&reals),
                             9 * capacity * sizeof(double)),
                  "allocate GPU particle real SoA fields");
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&words64),
                             3 * capacity * sizeof(std::uint64_t)),
                  "allocate GPU particle 64-bit SoA fields");
      } catch (...) {
        cudaFree(words64);
        cudaFree(reals);
        cudaFree(words32);
        throw;
      }

      queue.pid = words32;
      queue.medium_id = words32 + capacity;
      queue.generation =
          reinterpret_cast<std::uint32_t*>(words32 + 2 * capacity);
      queue.reserved =
          reinterpret_cast<std::uint32_t*>(words32 + 3 * capacity);
      queue.energy_GeV = reals;
      queue.position_x_m = reals + capacity;
      queue.position_y_m = reals + 2 * capacity;
      queue.position_z_m = reals + 3 * capacity;
      queue.direction_x = reals + 4 * capacity;
      queue.direction_y = reals + 5 * capacity;
      queue.direction_z = reals + 6 * capacity;
      queue.time_s = reals + 7 * capacity;
      queue.weight = reals + 8 * capacity;
      queue.history_id = words64;
      queue.parent_history_id = words64 + capacity;
      queue.step_id = words64 + 2 * capacity;
    }

    static void freeQueue(DeviceParticleSoA& queue) noexcept {
      if (queue.history_id != nullptr) {
        cudaFree(queue.history_id);
      }
      if (queue.energy_GeV != nullptr) {
        cudaFree(queue.energy_GeV);
      }
      if (queue.pid != nullptr) {
        cudaFree(queue.pid);
      }
      queue = {};
    }

    template <typename T>
    static void copyDeviceField(T* destination, T const* source,
                                std::size_t count, char const* operation) {
      checkCuda(cudaMemcpy(destination, source, count * sizeof(T),
                           cudaMemcpyDeviceToDevice),
                operation);
    }

    static void copyQueue(DeviceParticleSoA const& destination,
                          DeviceParticleSoA const& source,
                          std::size_t count) {
      copyDeviceField(destination.pid, source.pid, count, "grow GPU particle PID field");
      copyDeviceField(destination.medium_id, source.medium_id, count,
                      "grow GPU particle medium field");
      copyDeviceField(destination.generation, source.generation, count,
                      "grow GPU particle generation field");
      copyDeviceField(destination.reserved, source.reserved, count,
                      "grow GPU particle reserved field");
      copyDeviceField(destination.energy_GeV, source.energy_GeV, count,
                      "grow GPU particle energy field");
      copyDeviceField(destination.position_x_m, source.position_x_m, count,
                      "grow GPU particle position-x field");
      copyDeviceField(destination.position_y_m, source.position_y_m, count,
                      "grow GPU particle position-y field");
      copyDeviceField(destination.position_z_m, source.position_z_m, count,
                      "grow GPU particle position-z field");
      copyDeviceField(destination.direction_x, source.direction_x, count,
                      "grow GPU particle direction-x field");
      copyDeviceField(destination.direction_y, source.direction_y, count,
                      "grow GPU particle direction-y field");
      copyDeviceField(destination.direction_z, source.direction_z, count,
                      "grow GPU particle direction-z field");
      copyDeviceField(destination.time_s, source.time_s, count,
                      "grow GPU particle time field");
      copyDeviceField(destination.weight, source.weight, count,
                      "grow GPU particle weight field");
      copyDeviceField(destination.history_id, source.history_id, count,
                      "grow GPU particle history field");
      copyDeviceField(destination.parent_history_id, source.parent_history_id,
                      count, "grow GPU particle parent-history field");
      copyDeviceField(destination.step_id, source.step_id, count,
                      "grow GPU particle step field");
    }

    void ensureCapacity(std::size_t requested) {
      if (requested <= capacity_) {
        return;
      }
      auto const new_capacity = nextCapacity(capacity_, requested);

      std::size_t required_scan_bytes = 0;
      checkCuda(cub::DeviceScan::ExclusiveSum(
                    nullptr, required_scan_bytes, device_child_counts_,
                    device_child_offsets_, new_capacity),
                "query CUB scan storage");
      auto const required_bytes =
          allocatedBytes(new_capacity, required_scan_bytes);
      auto const queue_and_table_bytes = checkedAdd(
          checkedAdd(
              required_bytes,
              checkedAdd(
                  cuda_rate_table_.deviceBytes(),
                  moliere_interpolation_device_bytes_,
                  "GPU device table size overflow"),
              "GPU device allocation size overflow"),
          profile_axis_device_bytes_,
          "GPU profile-axis allocation size overflow");
      auto const queue_table_and_profile_bytes = checkedAdd(
          queue_and_table_bytes,
          profile_accumulator_device_bytes_,
          "GPU profile allocation size overflow");
      auto const queue_table_profile_and_cross_bytes =
          checkedAdd(
              queue_table_and_profile_bytes,
              cross_species_queue_device_bytes_,
              "GPU cross-species queue allocation size overflow");
      auto const queue_table_and_radio_bytes = checkedAdd(
          queue_table_profile_and_cross_bytes,
          radio_device_bytes_,
          "GPU radio allocation size overflow");
      auto const total_required_bytes = checkedAdd(
          queue_table_and_radio_bytes,
          checkedAdd(
              physical_workspace_.capacityBytes(),
              physical_workspace_next_.capacityBytes(),
              "GPU physical workspace size overflow"),
          "GPU device allocation plus workspace size overflow");
      if (total_required_bytes > memory_budget_bytes_) {
        std::ostringstream message;
        message << "GPU particle queues and tables require "
                << total_required_bytes
                << " bytes, exceeding the configured " << memory_budget_bytes_
                << "-byte budget";
        throw std::runtime_error(message.str());
      }

      DeviceParticleSoA new_current{};
      DeviceParticleSoA new_next{};
      EmParticleState* new_staging = nullptr;
      std::uint32_t* new_counts = nullptr;
      std::size_t* new_offsets = nullptr;
      void* new_scan_temporary = nullptr;
      try {
        allocateQueue(new_current, new_capacity);
        allocateQueue(new_next, new_capacity);
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&new_staging),
                             new_capacity * sizeof(EmParticleState)),
                  "allocate GPU AoS staging buffer");
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&new_counts),
                             new_capacity * sizeof(std::uint32_t)),
                  "allocate GPU child counts");
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&new_offsets),
                             new_capacity * sizeof(std::size_t)),
                  "allocate GPU child offsets");
        checkCuda(cudaMalloc(&new_scan_temporary, required_scan_bytes),
                  "allocate CUB scan storage");
        if (current_size_ > 0) {
          copyQueue(new_current, device_current_, current_size_);
        }
      } catch (...) {
        cudaFree(new_scan_temporary);
        cudaFree(new_offsets);
        cudaFree(new_counts);
        cudaFree(new_staging);
        freeQueue(new_next);
        freeQueue(new_current);
        throw;
      }

      cudaFree(device_scan_temporary_);
      cudaFree(device_child_offsets_);
      cudaFree(device_child_counts_);
      cudaFree(device_staging_);
      freeQueue(device_next_);
      freeQueue(device_current_);
      device_current_ = new_current;
      device_next_ = new_next;
      device_staging_ = new_staging;
      device_child_counts_ = new_counts;
      device_child_offsets_ = new_offsets;
      device_scan_temporary_ = new_scan_temporary;
      scan_temporary_bytes_ = required_scan_bytes;
      capacity_ = new_capacity;
      resident_allocation_bytes_ = queue_table_and_radio_bytes;
      statistics_.reserved_particles = capacity_;
      statistics_.peak_device_bytes =
          std::max(statistics_.peak_device_bytes,
                   total_required_bytes);
    }

    template <typename TRecord>
    void accumulateThinningStatistics(
        std::vector<TRecord> const& records,
        std::size_t first_record = 0) {
      if (first_record > records.size()) {
        throw std::logic_error(
            "GPU thinning statistics record range is invalid");
      }
      for (auto index = first_record; index < records.size();
           ++index) {
        auto const& record = records[index];
        auto const status =
            static_cast<EmThinningStatus>(
                record.thinning_status);
        if (status == EmThinningStatus::Hillas) {
          ++statistics_.thinning_hillas_vertices;
        } else if (
            status == EmThinningStatus::Statistical) {
          ++statistics_.thinning_statistical_vertices;
        } else if (
            status != EmThinningStatus::NotApplied) {
          throw std::runtime_error(
              "GPU final-state record has an invalid thinning status");
        }
        auto const original_multiplicity =
            record.process_id == PhotoelectricProcessId
                ? std::uint32_t{1}
                : record.process_id ==
                          ElectronPairProcessId
                      ? std::uint32_t{3}
                      : std::uint32_t{2};
        if (record.secondary_count >
            original_multiplicity) {
          throw std::runtime_error(
              "GPU thinning increased final-state multiplicity");
        }
        statistics_.thinning_particles_discarded +=
            original_multiplicity -
            record.secondary_count;
      }
    }

    void updateWorkspaceStatistics() {
      statistics_.physical_workspace_bytes =
          checkedAdd(
              physical_workspace_.capacityBytes(),
              physical_workspace_next_.capacityBytes(),
              "GPU physical workspace statistics overflow");
      auto const total = checkedAdd(
          resident_allocation_bytes_,
          statistics_.physical_workspace_bytes,
          "GPU resident allocation accounting overflow");
      if (total > memory_budget_bytes_) {
        throw std::runtime_error(
            "CUDA physical workspace exceeded backend memory budget");
      }
      statistics_.peak_device_bytes =
          std::max(statistics_.peak_device_bytes, total);
    }

    void appendStaging() {
      if (host_staging_.empty()) {
        return;
      }
      ensureCapacity(checkedAdd(current_size_, host_staging_.size(),
                                "GPU staging queue size overflow"));
      auto const start = std::chrono::steady_clock::now();
      checkCuda(cudaMemcpy(device_staging_, host_staging_.data(),
                           host_staging_.size() * sizeof(EmParticleState),
                           cudaMemcpyHostToDevice),
                "upload staged GPU particles");
      auto const blocks = static_cast<unsigned int>(
          (host_staging_.size() + ThreadsPerBlock - 1) / ThreadsPerBlock);
      scatterParticles<<<blocks, ThreadsPerBlock>>>(
          device_staging_, host_staging_.size(), device_current_, current_size_);
      checkCuda(cudaGetLastError(), "scatter staged GPU particles launch");
      auto const stop = std::chrono::steady_clock::now();
      statistics_.transfer_time_ms +=
          std::chrono::duration<double, std::milli>(stop - start).count();
      current_size_ += host_staging_.size();
      statistics_.current_particles = current_size_;
      statistics_.peak_particles =
          std::max(statistics_.peak_particles, current_size_);
      host_staging_.clear();
    }

    void release() noexcept {
      if (initialized_) {
        cudaSetDevice(config_.device);
      }
      if (profile_stream_ != nullptr) {
        cudaStreamSynchronize(profile_stream_);
      }
      if (device_scan_temporary_ != nullptr) {
        cudaFree(device_scan_temporary_);
      }
      if (device_child_offsets_ != nullptr) {
        cudaFree(device_child_offsets_);
      }
      if (device_child_counts_ != nullptr) {
        cudaFree(device_child_counts_);
      }
      if (device_staging_ != nullptr) {
        cudaFree(device_staging_);
      }
      freeQueue(device_next_);
      freeQueue(device_current_);
      physical_workspace_.release();
      physical_workspace_next_.release();
      radio_accumulator_.release();
      if (device_profile_counters_ != nullptr) {
        cudaFree(device_profile_counters_);
      }
      if (device_profile_histograms_ != nullptr) {
        cudaFree(device_profile_histograms_);
      }
      if (device_cross_species_selected_count_ != nullptr) {
        cudaFree(device_cross_species_selected_count_);
      }
      if (device_cross_species_select_temporary_ != nullptr) {
        cudaFree(device_cross_species_select_temporary_);
      }
      if (device_pending_leptons_ != nullptr) {
        cudaFree(device_pending_leptons_);
      }
      if (device_pending_photons_ != nullptr) {
        cudaFree(device_pending_photons_);
      }
      for (auto& event : profile_done_events_) {
        if (event != nullptr) {
          cudaEventDestroy(event);
        }
      }
      for (auto& event : profile_start_events_) {
        if (event != nullptr) {
          cudaEventDestroy(event);
        }
      }
      if (profile_stream_ != nullptr) {
        cudaStreamDestroy(profile_stream_);
      }
      if (device_profile_axis_grammage_ != nullptr) {
        cudaFree(device_profile_axis_grammage_);
      }
      if (device_moliere_interpolation_ != nullptr) {
        cudaFree(device_moliere_interpolation_);
      }
      if (physical_pipeline_stop_ != nullptr) {
        cudaEventDestroy(physical_pipeline_stop_);
      }
      if (physical_pipeline_start_ != nullptr) {
        cudaEventDestroy(physical_pipeline_start_);
      }
      for (auto& event : lepton_pipeline_stage_events_) {
        if (event != nullptr) {
          cudaEventDestroy(event);
        }
      }
      cuda_rate_table_.reset();
      photon_pair_lpm_ = {};
      brems_lpm_ = {};
      brems_lpm_prepared_ = {};
      moliere_ = {};
      muon_moliere_ = {};
      moliere_interpolation_ = {};
      moliere_available_ = false;
      muon_moliere_available_ = false;
      proposal_medium_hash_ = 0;
      interaction_hashes_.clear();
      environment_ = EnvironmentSnapshot{};
      device_scan_temporary_ = nullptr;
      device_child_offsets_ = nullptr;
      device_child_counts_ = nullptr;
      device_staging_ = nullptr;
      device_profile_axis_grammage_ = nullptr;
      device_moliere_interpolation_ = nullptr;
      device_profile_projection_ = {};
      device_profile_accumulator_ = {};
      device_profile_histograms_ = nullptr;
      device_profile_counters_ = nullptr;
      profile_counter_snapshot_ = {};
      profile_downloaded_ = false;
      radio_downloaded_ = false;
      profile_stream_ = nullptr;
      profile_start_events_ = {};
      profile_done_events_ = {};
      profile_slot_pending_ = {};
      device_pending_photons_ = nullptr;
      device_pending_leptons_ = nullptr;
      device_cross_species_selected_count_ = nullptr;
      device_cross_species_select_temporary_ = nullptr;
      pending_photon_count_ = 0;
      pending_lepton_count_ = 0;
      pending_photon_head_ = 0;
      pending_lepton_head_ = 0;
      cross_species_queue_capacity_ = 0;
      cross_species_queue_device_bytes_ = 0;
      cross_species_select_temporary_bytes_ = 0;
      profile_axis_device_bytes_ = 0;
      profile_accumulator_device_bytes_ = 0;
      radio_device_bytes_ = 0;
      moliere_interpolation_device_bytes_ = 0;
      physical_pipeline_start_ = nullptr;
      physical_pipeline_stop_ = nullptr;
      lepton_pipeline_stage_events_ = {};
      capacity_ = 0;
      current_size_ = 0;
      scan_temporary_bytes_ = 0;
      host_staging_.clear();
      statistics_.table_device_bytes = 0;
      statistics_.physical_workspace_bytes = 0;
      statistics_.maximum_resident_photon_batch = 0;
      statistics_.maximum_resident_lepton_batch = 0;
      maximum_resident_photon_batch_size_ = 0;
      maximum_resident_lepton_batch_size_ = 0;
      resident_allocation_bytes_ = 0;
      shower_ordinal_ = 0;
      static_host_to_device_bytes_ = 0;
      one_time_initialization_ms_ = 0.;
      initialized_ = false;
    }

    GpuEmConfig config_{};
    bool initialized_{};
    std::size_t memory_budget_bytes_{};
    std::size_t capacity_{};
    std::size_t current_size_{};
    std::size_t scan_temporary_bytes_{};
    std::size_t resident_allocation_bytes_{};
    std::size_t maximum_resident_photon_batch_size_{};
    std::size_t maximum_resident_lepton_batch_size_{};
    std::uint64_t next_history_id_{1};
    DeviceParticleSoA device_current_{};
    DeviceParticleSoA device_next_{};
    EmParticleState* device_staging_{};
    std::uint32_t* device_child_counts_{};
    std::size_t* device_child_offsets_{};
    void* device_scan_temporary_{};
    std::vector<EmParticleState> host_staging_{};
    mutable GpuEmStatistics statistics_{};
    tables::CudaRateTable cuda_rate_table_{};
    std::uint64_t proposal_medium_hash_{};
    std::vector<std::pair<std::int32_t, std::uint64_t>>
        interaction_hashes_{};
    PhotonPairLpmSnapshot photon_pair_lpm_{};
    BremsLpmSnapshot brems_lpm_{};
    BremsLpmPreparedSnapshot brems_lpm_prepared_{};
    MoliereSnapshot moliere_{};
    MoliereSnapshot muon_moliere_{};
    MoliereInterpolationView moliere_interpolation_{};
    MoliereCubicPolynomial*
        device_moliere_interpolation_{};
    std::size_t moliere_interpolation_device_bytes_{};
    bool moliere_available_{};
    bool muon_moliere_available_{};
    EnvironmentSnapshot environment_{};
    detail::DeviceWorkspace physical_workspace_{};
    detail::DeviceWorkspace physical_workspace_next_{};
    double* device_profile_axis_grammage_{};
    detail::DeviceProfileProjection
        device_profile_projection_{};
    detail::DeviceProfileAccumulator
        device_profile_accumulator_{};
    long long* device_profile_histograms_{};
    detail::DeviceProfileCounters*
        device_profile_counters_{};
    detail::DeviceProfileCounters
        profile_counter_snapshot_{};
    bool profile_downloaded_{};
    bool radio_downloaded_{};
    cudaStream_t profile_stream_{};
    std::array<cudaEvent_t, 2>
        profile_start_events_{};
    std::array<cudaEvent_t, 2>
        profile_done_events_{};
    std::array<bool, 2> profile_slot_pending_{};
    EmParticleState* device_pending_photons_{};
    EmParticleState* device_pending_leptons_{};
    std::size_t*
        device_cross_species_selected_count_{};
    void* device_cross_species_select_temporary_{};
    std::size_t pending_photon_count_{};
    std::size_t pending_lepton_count_{};
    std::size_t pending_photon_head_{};
    std::size_t pending_lepton_head_{};
    std::size_t cross_species_queue_capacity_{};
    std::size_t cross_species_queue_device_bytes_{};
    std::size_t
        cross_species_select_temporary_bytes_{};
    std::size_t profile_axis_device_bytes_{};
    std::size_t profile_accumulator_device_bytes_{};
    radio::CudaRadioAccumulator radio_accumulator_{};
    std::size_t radio_device_bytes_{};
    std::uint64_t shower_ordinal_{};
    std::uint64_t static_host_to_device_bytes_{};
    double one_time_initialization_ms_{};
    cudaEvent_t physical_pipeline_start_{};
    cudaEvent_t physical_pipeline_stop_{};
    std::array<cudaEvent_t, LeptonPipelineStageCount>
        lepton_pipeline_stage_events_{};
  };

  CudaEmBackend::CudaEmBackend() : impl_(std::make_unique<Impl>()) {}

  CudaEmBackend::~CudaEmBackend() = default;

  CudaEmBackend::CudaEmBackend(CudaEmBackend&&) noexcept = default;

  CudaEmBackend& CudaEmBackend::operator=(CudaEmBackend&&) noexcept = default;

  void CudaEmBackend::initialize(EnvironmentSnapshot const& environment,
                                 ProposalTableSet const& tables,
                                 GpuEmConfig const& config) {
    impl_->initialize(environment, tables, config);
  }

  void CudaEmBackend::beginShower(
      GpuEmShowerConfig const& config) {
    impl_->beginShower(config);
  }

  bool CudaEmBackend::canTransport(EmParticleState const& particle) const {
    return impl_->canTransport(particle);
  }

  void CudaEmBackend::enqueue(EmParticleState const& particle) {
    impl_->enqueue(particle);
  }

  EmBatchResult CudaEmBackend::advanceWavefront() {
    return impl_->advanceWavefront();
  }

  void CudaEmBackend::drain() { impl_->drain(); }

  bool CudaEmBackend::empty() const { return impl_->empty(); }

  GpuEmStatistics const& CudaEmBackend::statistics() const {
    return impl_->statistics();
  }

  std::size_t CudaEmBackend::minimumBatchSize() const {
    return impl_->minimumBatchSize();
  }

  std::size_t
  CudaEmBackend::maximumResidentPhotonBatchSize() const {
    return impl_->maximumResidentPhotonBatchSize();
  }

  std::size_t
  CudaEmBackend::maximumResidentLeptonBatchSize() const {
    return impl_->maximumResidentLeptonBatchSize();
  }

  std::size_t
  CudaEmBackend::maximumResidentInputBatchSize() const {
    return impl_->maximumResidentInputBatchSize();
  }

  std::size_t
  CudaEmBackend::pendingPhotonCount() const noexcept {
    return impl_->pendingPhotonCount();
  }

  std::size_t
  CudaEmBackend::pendingLeptonCount() const noexcept {
    return impl_->pendingLeptonCount();
  }

  bool CudaEmBackend::hasProposalTable() const {
    return impl_->hasProposalTable();
  }

  std::array<std::uint8_t, 32>
  CudaEmBackend::proposalTableHash() const {
    return impl_->proposalTableHash();
  }

  bool CudaEmBackend::gpuRadioEnabled() const noexcept {
    return impl_->gpuRadioEnabled();
  }

  radio::GpuRadioWaveforms
  CudaEmBackend::downloadRadioWaveforms() {
    return impl_->downloadRadioWaveforms();
  }

  bool CudaEmBackend::gpuProfileEnabled() const noexcept {
    return impl_->gpuProfileEnabled();
  }

  GpuProfileResult CudaEmBackend::downloadProfile() {
    return impl_->downloadProfile();
  }

  EmInteractionBatchResult
  CudaEmBackend::selectInteractionsForValidation(
      std::vector<EmParticleState> const& particles) {
    return impl_->selectInteractionsForValidation(particles);
  }

  EmFinalStateBatchResult
  CudaEmBackend::generateFinalStatesForValidation(
      std::vector<EmInteractionRecord> const& interactions,
      std::uint64_t first_secondary_history_id) {
    return impl_->generateFinalStatesForValidation(
        interactions, first_secondary_history_id);
  }

  BremsFinalStateBatchResult
  CudaEmBackend::generateBremsFinalStatesForValidation(
      std::vector<EmInteractionRecord> const& interactions,
      std::uint64_t first_secondary_history_id) {
    return impl_->generateBremsFinalStatesForValidation(
        interactions, first_secondary_history_id);
  }

  PhotonTransportBatchResult
  CudaEmBackend::transportPhotonsForValidation(
      std::vector<EmInteractionRecord> const& interactions) {
    return impl_->transportPhotonsForValidation(interactions);
  }

  LeptonTransportBatchResult
  CudaEmBackend::transportLeptonsStraightForValidation(
      std::vector<EmInteractionRecord> const& interactions) {
    return impl_->transportLeptonsStraightForValidation(
        interactions);
  }

  LeptonVertexSelectionBatchResult
  CudaEmBackend::
      reselectLeptonInteractionsAtVertexForValidation(
          std::vector<EmInteractionRecord> const& candidates) {
    return impl_->
        reselectLeptonInteractionsAtVertexForValidation(
            candidates);
  }

  LeptonDevicePipelineBatchResult
  CudaEmBackend::runLeptonDevicePipelineForValidation(
      std::vector<EmParticleState> const& particles,
      std::uint64_t first_secondary_history_id) {
    return impl_->runPhysicalLeptonPipeline(
        particles, first_secondary_history_id);
  }

  PhotonSelectionTransportBatchResult
  CudaEmBackend::selectAndTransportPhotonsForValidation(
      std::vector<EmParticleState> const& particles) {
    return impl_->selectAndTransportPhotonsForValidation(particles);
  }

  PhotonDevicePipelineBatchResult
  CudaEmBackend::runPhotonDevicePipelineForValidation(
      std::vector<EmParticleState> const& particles,
      std::uint64_t first_secondary_history_id) {
    return impl_->runPhysicalPhotonPipeline(
        particles, first_secondary_history_id);
  }

  ResidentPhotonCascadeResult
  CudaEmBackend::runResidentPhotonCascadeForValidation(
      std::vector<EmParticleState> const& particles,
      std::uint64_t first_secondary_history_id,
      std::size_t maximum_wavefronts,
      std::size_t minimum_resident_batch_size) {
    return impl_->runResidentPhotonCascadeForValidation(
        particles, first_secondary_history_id, maximum_wavefronts,
        minimum_resident_batch_size);
  }

  ResidentLeptonCascadeResult
  CudaEmBackend::runResidentLeptonCascadeForValidation(
      std::vector<EmParticleState> const& particles,
      std::uint64_t first_secondary_history_id,
      std::size_t maximum_wavefronts,
      std::uint64_t secondary_history_id_limit_exclusive,
      std::size_t minimum_resident_batch_size) {
    return impl_->runResidentLeptonCascadeForValidation(
        particles, first_secondary_history_id,
        maximum_wavefronts,
        secondary_history_id_limit_exclusive,
        minimum_resident_batch_size);
  }

  PhotonWavefrontBatchResult
  CudaEmBackend::advancePhotonWavefrontForValidation(
      std::vector<EmParticleState> const& particles,
      std::uint64_t first_secondary_history_id) {
    return impl_->advancePhotonWavefrontForValidation(
        particles, first_secondary_history_id);
  }

  std::vector<EmParticleState> CudaEmBackend::downloadActiveParticles() const {
    return impl_->downloadActiveParticles();
  }

  std::vector<EmParticleState>
  CudaEmBackend::extractActiveParticlesForTesting() {
    return impl_->extractActiveParticlesForTesting();
  }

} // namespace corsika::gpu::em

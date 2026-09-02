/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include <cuda_runtime.h>

#include <cstddef>
#include <sstream>
#include <stdexcept>

#include <corsika/accelerator/em/detail/ProfileProjectionStep.hpp>
#include <corsika/gpu/em/ProcessCapabilities.hpp>
#include <corsika/gpu/em/MoliereScattering.hpp>
#include <corsika/gpu/em/PhotonPairKinematics.hpp>
#include <corsika/gpu/em/detail/ProfileProjection.hpp>

namespace corsika::gpu::em::detail {

  namespace {

    constexpr unsigned int ThreadsPerBlock = 256;
    constexpr double SignedIntegerLimit = 0x1p63;
    constexpr long long SignedIntegerMaximum =
        9223372036854775807LL;
    constexpr long long SignedIntegerMinimum =
        (-9223372036854775807LL - 1LL);

    void checkCuda(cudaError_t status, char const* operation) {
      if (status == cudaSuccess) {
        return;
      }
      std::ostringstream message;
      message << operation << " failed: "
              << cudaGetErrorString(status);
      throw std::runtime_error(message.str());
    }

    __device__ bool checkedAtomicAdd(
        long long* address, long long increment) {
      auto* storage =
          reinterpret_cast<unsigned long long*>(address);
      // CUDA's unsigned 64-bit atomic addition has the same two's-complement
      // bit result as signed addition when no overflow occurs.  It also
      // returns the serialized value that preceded this contribution, so the
      // overflow test remains exact.  The former load/CAS retry loop became
      // extremely expensive when hundreds of millions of shower steps
      // contended on the same longitudinal bins.
      auto const previous_bits = atomicAdd(
          storage, static_cast<unsigned long long>(increment));
      auto const previous =
          static_cast<long long>(previous_bits);
      return !(
          (increment > 0 &&
           previous > SignedIntegerMaximum - increment) ||
          (increment < 0 &&
           previous < SignedIntegerMinimum - increment));
    }

    /**
     * Iteration diagnostics are generated for every transported lepton.
     * Reducing them per active warp avoids adding two highly contended global
     * atomics per record to the profile kernel.
     */
    __device__ void accumulateMoliereIterationCounters(
        DeviceProfileCounters* counters,
        std::uint32_t iterations, unsigned int mask) {
      auto const lane = threadIdx.x & 31U;
      auto sum = static_cast<unsigned long long>(iterations);
      auto maximum = iterations;
      for (unsigned int offset = 16; offset != 0;
           offset >>= 1) {
        auto const source_lane = lane + offset;
        auto const other_sum =
            __shfl_down_sync(mask, sum, offset);
        auto const other_maximum =
            __shfl_down_sync(mask, maximum, offset);
        if (source_lane < 32U &&
            (mask & (1U << source_lane)) != 0U) {
          sum += other_sum;
          maximum =
              maximum > other_maximum
                  ? maximum
                  : other_maximum;
        }
      }
      auto const leader =
          static_cast<unsigned int>(__ffs(mask) - 1);
      if (lane == leader) {
        atomicAdd(
            &counters->moliere_newton_iterations, sum);
        atomicMax(
            &counters->moliere_max_newton_iterations,
            static_cast<unsigned long long>(maximum));
      }
    }

    __device__ void addFixedPoint(
        long long* address, double contribution, double scale,
        DeviceProfileCounters* counters) {
      auto const scaled = contribution * scale;
      if (!isfinite(scaled) ||
          fabs(scaled) >= SignedIntegerLimit ||
          !checkedAtomicAdd(address, __double2ll_rn(scaled))) {
        atomicAdd(
            &counters->fixed_point_overflows, 1ULL);
      }
    }

    __device__ void accumulateParticleProfile(
        DeviceProfileAccumulator const& accumulator,
        std::int32_t pid, double start_grammage,
        double end_grammage, double weight) {
      if (start_grammage == end_grammage ||
          accumulator.bins == 0) {
        return;
      }
      auto const first_value =
          ceil(start_grammage /
               accumulator.bin_width_g_per_cm2);
      auto const last_value =
          floor(end_grammage /
                accumulator.bin_width_g_per_cm2);
      if (!(first_value >= 0.) ||
          !(last_value >= first_value) ||
          first_value >=
              static_cast<double>(accumulator.bins)) {
        return;
      }
      auto const first =
          static_cast<std::size_t>(first_value);
      auto const last = min(
          static_cast<std::size_t>(last_value),
          accumulator.bins - 1);
      long long* profile = nullptr;
      if (pid == static_cast<std::int32_t>(EmPid::Photon)) {
        profile = accumulator.photons;
      } else if (
          pid == static_cast<std::int32_t>(EmPid::Electron)) {
        profile = accumulator.electrons;
      } else if (
          pid == static_cast<std::int32_t>(EmPid::Positron)) {
        profile = accumulator.positrons;
      } else if (
          pid == static_cast<std::int32_t>(EmPid::MuonMinus)) {
        profile = accumulator.muons_minus;
      } else if (
          pid == static_cast<std::int32_t>(EmPid::MuonPlus)) {
        profile = accumulator.muons_plus;
      }
      if (profile == nullptr) {
        atomicAdd(
            &accumulator.counters->invalid_records, 1ULL);
        return;
      }
      for (auto bin = first; bin <= last; ++bin) {
        addFixedPoint(
            profile + bin, weight,
            accumulator.weight_scale,
            accumulator.counters);
      }
    }

    /**
     * Reproduce ProductionWriter's point-vertex binning for the subset of
     * muon production vertices handled entirely on the device.
     *
     * ProductionProfile runs before EM thinning on the scalar path.  The
     * contribution is consequently the incoming parent weight even when the
     * outgoing muon is subsequently discarded or reweighted by thinning.
     */
    __device__ void accumulateMuonParentProduction(
        DeviceProfileAccumulator const& accumulator,
        double grammage, double parent_weight) {
      if (accumulator.bins == 0 ||
          accumulator.muon_parent_productions == nullptr) {
        return;
      }
      auto const bin_value =
          floor(grammage /
                accumulator.bin_width_g_per_cm2);
      if (!isfinite(bin_value)) {
        atomicAdd(
            &accumulator.counters->invalid_records, 1ULL);
        return;
      }
      if (bin_value < 0.) {
        return;
      }
      // CORSIKA 8 icrc2025-beta2 assigns a production vertex with
      // floor(X/dX) and clamps a finite value beyond the configured range to
      // the final bin.  Keep the resident GPU path bit-for-bit aligned with
      // ProductionWriter::writeProjected().
      auto const bin = min(
          static_cast<std::size_t>(bin_value),
          accumulator.bins - 1);
      addFixedPoint(
          accumulator.muon_parent_productions + bin,
          parent_weight, accumulator.weight_scale,
          accumulator.counters);
    }

    __device__ void accumulateEnergyProfile(
        DeviceProfileAccumulator const& accumulator,
        long long* energy_loss,
        double start_grammage, double end_grammage,
        double weighted_deposit_GeV) {
      if (!(weighted_deposit_GeV > 0.) ||
          accumulator.bins == 0) {
        return;
      }
      atomicAdd(
          &accumulator.counters->deposited_steps, 1ULL);
      if (start_grammage > end_grammage) {
        auto const swap = start_grammage;
        start_grammage = end_grammage;
        end_grammage = swap;
      }
      auto const delta = end_grammage - start_grammage;
      auto const max_bin =
          static_cast<int>(accumulator.bins - 1);
      auto first = static_cast<int>(
          start_grammage /
          accumulator.bin_width_g_per_cm2);
      first = first < 0 ? 0 : first;
      first = first > max_bin ? max_bin : first;
      if (delta <
          accumulator.energy_loss_threshold_g_per_cm2) {
        addFixedPoint(
            energy_loss + first,
            weighted_deposit_GeV,
            accumulator.energy_scale,
            accumulator.counters);
        return;
      }
      auto last = static_cast<int>(
          end_grammage /
          accumulator.bin_width_g_per_cm2);
      last = last < 0 ? 0 : last;
      last = last > max_bin ? max_bin : last;
      auto const density = weighted_deposit_GeV / delta;
      if (first == last) {
        addFixedPoint(
            energy_loss + first,
            density * delta, accumulator.energy_scale,
            accumulator.counters);
        return;
      }
      addFixedPoint(
          energy_loss + first,
          density *
          (1. + static_cast<double>(first)) *
                  accumulator.bin_width_g_per_cm2 -
              density * start_grammage,
          accumulator.energy_scale,
          accumulator.counters);
      addFixedPoint(
          energy_loss + last,
          density *
          end_grammage -
              density * static_cast<double>(last) *
                  accumulator.bin_width_g_per_cm2,
          accumulator.energy_scale,
          accumulator.counters);
      for (auto bin = first + 1; bin < last; ++bin) {
        addFixedPoint(
            energy_loss + bin,
            density * accumulator.bin_width_g_per_cm2,
            accumulator.energy_scale,
            accumulator.counters);
      }
    }

    __device__ void accumulateTerminalEnergy(
        DeviceProfileAccumulator const& accumulator,
        double weighted_total_energy_GeV, bool observed) {
      if (!(weighted_total_energy_GeV >= 0.) ||
          !isfinite(weighted_total_energy_GeV)) {
        atomicAdd(
            &accumulator.counters->invalid_records, 1ULL);
        return;
      }
      addFixedPoint(
          observed
              ? &accumulator.counters
                     ->weighted_observed_total_energy
              : &accumulator.counters
                     ->weighted_escaped_total_energy,
          weighted_total_energy_GeV,
          accumulator.energy_scale,
          accumulator.counters);
    }

    template <typename FinalStateRecord>
    __device__ void accumulateThinning(
        FinalStateRecord const& record,
        DeviceProfileCounters* counters) {
      auto const status =
          static_cast<EmThinningStatus>(
              record.thinning_status);
      if (status == EmThinningStatus::Hillas) {
        atomicAdd(
            &counters->thinning_hillas_vertices, 1ULL);
      } else if (
          status == EmThinningStatus::Statistical) {
        atomicAdd(
            &counters->thinning_statistical_vertices,
            1ULL);
      } else if (status != EmThinningStatus::NotApplied) {
        atomicAdd(&counters->invalid_records, 1ULL);
      }
      auto const original_multiplicity =
          record.process_id == PhotoelectricProcessId
              ? 1U
              : record.process_id == ElectronPairProcessId
                    ? 3U
                    : 2U;
      if (record.secondary_count > original_multiplicity) {
        atomicAdd(&counters->invalid_records, 1ULL);
      } else {
        atomicAdd(
            &counters->thinning_particles_discarded,
            static_cast<unsigned long long>(
                original_multiplicity -
                record.secondary_count));
      }
    }

    __global__ void projectPhotonStepsKernel(
        DeviceProfileProjection projection,
        PhotonTransportRecord const* records,
        std::size_t count, ProjectedEmStepRecord* output) {
      auto const index =
          static_cast<std::size_t>(blockIdx.x) * blockDim.x +
          threadIdx.x;
      if (index >= count) {
        return;
      }
      output[index] = accelerator::em::detail::projectPhotonStep(
          projection, records[index]);
    }

    __global__ void projectLeptonStepsKernel(
        DeviceProfileProjection projection,
        LeptonTransportRecord const* records,
        std::size_t count, ProjectedEmStepRecord* output) {
      auto const index =
          static_cast<std::size_t>(blockIdx.x) * blockDim.x +
          threadIdx.x;
      if (index >= count) {
        return;
      }
      output[index] = accelerator::em::detail::projectLeptonStep(
          projection, records[index]);
    }

    __global__ void accumulatePhotonStepsKernel(
        DeviceProfileProjection projection,
        DeviceProfileAccumulator accumulator,
        PhotonTransportRecord const* records,
        std::size_t count) {
      auto const index =
          static_cast<std::size_t>(blockIdx.x) * blockDim.x +
          threadIdx.x;
      if (index >= count) {
        return;
      }
      auto const record = records[index];
      auto const start = accelerator::em::detail::projectProfileGrammage(
          projection, record.start.position_m);
      auto const end = accelerator::em::detail::projectProfileGrammage(
          projection, record.end.position_m);
      accumulateParticleProfile(
          accumulator, record.start.pid, start, end,
          record.start.weight);
      accumulateEnergyProfile(
          accumulator, accumulator.energy_loss, start, end,
          record.cut_deposited_energy_GeV *
              record.start.weight);
      atomicAdd(&accumulator.counters->steps, 1ULL);
      if (record.limit == PhotonTransportLimit::ParticleCut) {
        atomicAdd(
            &accumulator.counters->photon_cuts, 1ULL);
        if (record.observation_surface_reached_before_cut != 0U) {
          accumulateTerminalEnergy(
              accumulator,
              record.end.energy_GeV * record.start.weight,
              true);
        }
      } else if (
          record.limit ==
          PhotonTransportLimit::ObservationSurface) {
        accumulateTerminalEnergy(
            accumulator,
            record.end.energy_GeV * record.start.weight,
            true);
      } else if (
          record.limit ==
          PhotonTransportLimit::EscapedEnvironment) {
        accumulateTerminalEnergy(
            accumulator,
            record.end.energy_GeV * record.start.weight,
            false);
      }
    }

    template <typename TransportRecord>
    __device__ TransportRecord const* findTransportRecord(
        TransportRecord const* records,
        std::size_t count, std::uint64_t input_index) {
      std::size_t first = 0;
      std::size_t last = count;
      while (first < last) {
        auto const middle = first + (last - first) / 2;
        if (records[middle].input_index < input_index) {
          first = middle + 1;
        } else {
          last = middle;
        }
      }
      if (first >= count ||
          records[first].input_index != input_index) {
        return nullptr;
      }
      return records + first;
    }

    __global__ void accumulatePhotonFinalStatesKernel(
        DeviceProfileProjection projection,
        DeviceProfileAccumulator accumulator,
        PhotonTransportRecord const* transport_records,
        std::size_t transport_count,
        PhotonPairFinalStateRecord const* records,
        std::size_t count) {
      auto const index =
          static_cast<std::size_t>(blockIdx.x) * blockDim.x +
          threadIdx.x;
      if (index >= count) {
        return;
      }
      auto const record = records[index];
      accumulateThinning(record, accumulator.counters);
      auto const atomic_electron_process =
          record.process_id == ComptonProcessId ||
          record.process_id == PhotoelectricProcessId;
      if (!atomic_electron_process) {
        return;
      }
      auto const* transport = findTransportRecord(
          transport_records, transport_count,
          record.input_index);
      if (transport == nullptr ||
          transport->start.history_id !=
              record.parent_history_id ||
          transport->interaction.process_id !=
              record.process_id) {
        atomicAdd(
            &accumulator.counters->invalid_records, 1ULL);
        return;
      }
      addFixedPoint(
          &accumulator.counters
               ->weighted_medium_rest_mass_input,
          ElectronMassGeV * transport->start.weight,
          accumulator.energy_scale,
          accumulator.counters);
      if (record.process_id != PhotoelectricProcessId) {
        return;
      }
      if (!isfinite(record.energy_split_fraction) ||
          record.energy_split_fraction < 0. ||
          record.energy_split_fraction > 1.) {
        atomicAdd(
            &accumulator.counters->invalid_records, 1ULL);
        return;
      }
      auto const start = accelerator::em::detail::projectProfileGrammage(
          projection, transport->start.position_m);
      auto const end = accelerator::em::detail::projectProfileGrammage(
          projection, transport->end.position_m);
      auto const deposit =
          transport->end.energy_GeV *
          (1. - record.energy_split_fraction) *
          transport->start.weight;
      accumulateEnergyProfile(
          accumulator, accumulator.energy_loss, start, end,
          deposit);
    }

    __global__ void accumulateLeptonStepsKernel(
        DeviceProfileProjection projection,
        DeviceProfileAccumulator accumulator,
        LeptonTransportRecord const* records,
        std::size_t count) {
      auto const index =
          static_cast<std::size_t>(blockIdx.x) * blockDim.x +
          threadIdx.x;
      // All lanes participate in the ballot before the last partial warp
      // returns.  Do not derive this mask after the profile-bin loops below:
      // their data-dependent trip counts can make __activemask() depend on
      // independent thread scheduling and corrupt an otherwise diagnostic
      // warp reduction.
      auto const warp_mask =
          __ballot_sync(0xffffffffU, index < count);
      if (index >= count) {
        return;
      }
      auto const record = records[index];
      accumulateMoliereIterationCounters(
          accumulator.counters,
          record.multiple_scattering_iterations, warp_mask);
      auto const start = accelerator::em::detail::projectProfileGrammage(
          projection, record.start.position_m);
      auto const end = accelerator::em::detail::projectProfileGrammage(
          projection, record.end.position_m);
      accumulateParticleProfile(
          accumulator, record.start.pid, start, end,
          record.start.weight);
      accumulateEnergyProfile(
          accumulator,
          isMuonPid(record.start.pid)
              ? accumulator.muon_energy_loss
              : accumulator.energy_loss,
          start, end,
          (record.continuous_deposited_energy_GeV +
           record.cut_deposited_energy_GeV) *
              record.start.weight);
      atomicAdd(&accumulator.counters->steps, 1ULL);
      auto const limit =
          static_cast<std::int32_t>(record.limit);
      if (limit < 0 || limit >= 8) {
        atomicAdd(
            &accumulator.counters->invalid_records, 1ULL);
      } else {
        atomicAdd(
            &accumulator.counters->lepton_limits[limit],
            1ULL);
      }
      if (record.limit == LeptonTransportLimit::ParticleCut) {
        addFixedPoint(
            &accumulator.counters
                 ->weighted_cut_rest_mass_energy,
            (isMuonPid(record.start.pid)
                 ? MuonMassGeV
                 : ElectronMassGeV) *
                record.start.weight,
            accumulator.energy_scale,
            accumulator.counters);
        if (record.observation_surface_reached_before_cut != 0U) {
          accumulateTerminalEnergy(
              accumulator,
              record.end.energy_GeV * record.start.weight,
              true);
        }
      } else if (
          record.limit ==
          LeptonTransportLimit::ObservationSurface) {
        accumulateTerminalEnergy(
            accumulator,
            record.end.energy_GeV * record.start.weight,
            true);
      } else if (
          record.limit ==
          LeptonTransportLimit::EscapedEnvironment) {
        accumulateTerminalEnergy(
            accumulator,
            record.end.energy_GeV * record.start.weight,
            false);
      }
      atomicAdd(
          &accumulator.counters->moliere_trials, 1ULL);
      if (record.multiple_scattering_applied != 0) {
        atomicAdd(
            &accumulator.counters->moliere_deflections,
            1ULL);
      } else if (
          record.multiple_scattering_status ==
          static_cast<std::uint32_t>(
              MoliereStatus::NoDeflection)) {
        atomicAdd(
            &accumulator.counters
                 ->moliere_zero_deflections,
            1ULL);
      }
    }

    __global__ void accumulateLeptonFinalStatesKernel(
        DeviceProfileProjection projection,
        DeviceProfileAccumulator accumulator,
        LeptonTransportRecord const* transport_records,
        std::size_t transport_count,
        BremsFinalStateRecord const* records,
        std::size_t count) {
      auto const index =
          static_cast<std::size_t>(blockIdx.x) * blockDim.x +
          threadIdx.x;
      if (index < count) {
        auto const record = records[index];
        accumulateThinning(record, accumulator.counters);
        auto const atomic_electron_process =
            record.process_id == IonizationProcessId ||
            record.process_id == AnnihilationProcessId;
        if (!atomic_electron_process) {
          return;
        }
        auto const* transport = findTransportRecord(
            transport_records, transport_count,
            record.input_index);
        if (transport == nullptr ||
            transport->start.history_id !=
                record.parent_history_id) {
          atomicAdd(
              &accumulator.counters->invalid_records, 1ULL);
          return;
        }
        if (record.process_id == IonizationProcessId &&
            isMuonPid(transport->start.pid)) {
          auto const vertex_grammage =
              accelerator::em::detail::projectProfileGrammage(
              projection, transport->end.position_m);
          accumulateMuonParentProduction(
              accumulator, vertex_grammage,
              transport->start.weight);
        }
        addFixedPoint(
            &accumulator.counters
                 ->weighted_medium_rest_mass_input,
            ElectronMassGeV * transport->start.weight,
            accumulator.energy_scale,
            accumulator.counters);
      }
    }

  } // namespace

  void launchPhotonProfileProjectionOnDevice(
      DeviceProfileProjection const& projection,
      PhotonTransportRecord const* records, std::size_t count,
      ProjectedEmStepRecord* output) {
    if (count == 0) {
      return;
    }
    auto const blocks = static_cast<unsigned int>(
        (count + ThreadsPerBlock - 1) / ThreadsPerBlock);
    projectPhotonStepsKernel<<<blocks, ThreadsPerBlock>>>(
        projection, records, count, output);
    checkCuda(
        cudaGetLastError(),
        "launch photon ShowerAxis projection kernel");
  }

  void launchLeptonProfileProjectionOnDevice(
      DeviceProfileProjection const& projection,
      LeptonTransportRecord const* records, std::size_t count,
      ProjectedEmStepRecord* output) {
    if (count == 0) {
      return;
    }
    auto const blocks = static_cast<unsigned int>(
        (count + ThreadsPerBlock - 1) / ThreadsPerBlock);
    projectLeptonStepsKernel<<<blocks, ThreadsPerBlock>>>(
        projection, records, count, output);
    checkCuda(
        cudaGetLastError(),
        "launch lepton ShowerAxis projection kernel");
  }

  void launchPhotonProfileAccumulationOnDevice(
      DeviceProfileProjection const& projection,
      DeviceProfileAccumulator const& accumulator,
      PhotonTransportRecord const* records, std::size_t count,
      PhotonPairFinalStateRecord const* final_states,
      std::size_t final_state_count,
      cudaStream_t stream) {
    if (count != 0) {
      auto const blocks = static_cast<unsigned int>(
          (count + ThreadsPerBlock - 1) /
          ThreadsPerBlock);
      accumulatePhotonStepsKernel
          <<<blocks, ThreadsPerBlock, 0, stream>>>(
          projection, accumulator, records, count);
      checkCuda(
          cudaGetLastError(),
          "launch resident photon profile accumulation kernel");
    }
    if (final_state_count != 0) {
      auto const blocks = static_cast<unsigned int>(
          (final_state_count + ThreadsPerBlock - 1) /
          ThreadsPerBlock);
      accumulatePhotonFinalStatesKernel
          <<<blocks, ThreadsPerBlock, 0, stream>>>(
              projection, accumulator, records, count,
              final_states, final_state_count);
      checkCuda(
          cudaGetLastError(),
          "launch resident photon deposit accumulation kernel");
    }
  }

  void launchLeptonProfileAccumulationOnDevice(
      DeviceProfileProjection const& projection,
      DeviceProfileAccumulator const& accumulator,
      LeptonTransportRecord const* records, std::size_t count,
      BremsFinalStateRecord const* final_states,
      std::size_t final_state_count,
      cudaStream_t stream) {
    if (count != 0) {
      auto const blocks = static_cast<unsigned int>(
          (count + ThreadsPerBlock - 1) /
          ThreadsPerBlock);
      accumulateLeptonStepsKernel
          <<<blocks, ThreadsPerBlock, 0, stream>>>(
          projection, accumulator, records, count);
      checkCuda(
          cudaGetLastError(),
          "launch resident lepton profile accumulation kernel");
    }
    if (final_state_count != 0) {
      auto const blocks = static_cast<unsigned int>(
          (final_state_count + ThreadsPerBlock - 1) /
          ThreadsPerBlock);
      accumulateLeptonFinalStatesKernel
          <<<blocks, ThreadsPerBlock, 0, stream>>>(
              projection, accumulator, records, count, final_states,
              final_state_count);
      checkCuda(
          cudaGetLastError(),
          "launch resident lepton thinning statistics kernel");
    }
  }

} // namespace corsika::gpu::em::detail

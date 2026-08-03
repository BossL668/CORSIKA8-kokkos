/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

#include <corsika/gpu/em/Types.hpp>

namespace corsika::gpu::em::detail {

  struct DeviceProfileProjection {
    double axis_start_position_m[3]{};
    double axis_direction[3]{};
    double axis_step_length_m{};
    double const* axis_grammage_g_per_cm2{};
    std::size_t axis_support_count{};
  };

  struct DeviceProfileCounters {
    unsigned long long steps{};
    unsigned long long deposited_steps{};
    unsigned long long photon_cuts{};
    unsigned long long lepton_limits[8]{};
    unsigned long long moliere_trials{};
    unsigned long long moliere_deflections{};
    unsigned long long moliere_zero_deflections{};
    unsigned long long moliere_newton_iterations{};
    unsigned long long moliere_max_newton_iterations{};
    unsigned long long thinning_hillas_vertices{};
    unsigned long long thinning_statistical_vertices{};
    unsigned long long thinning_particles_discarded{};
    unsigned long long fixed_point_overflows{};
    unsigned long long invalid_records{};
    long long weighted_medium_rest_mass_input{};
    long long weighted_cut_rest_mass_energy{};
    long long weighted_observed_total_energy{};
    long long weighted_escaped_total_energy{};
  };

  struct DeviceProfileAccumulator {
    long long* photons{};
    long long* electrons{};
    long long* positrons{};
    long long* muons_minus{};
    long long* muons_plus{};
    long long* muon_parent_productions{};
    long long* energy_loss{};
    long long* muon_energy_loss{};
    DeviceProfileCounters* counters{};
    std::size_t bins{};
    double bin_width_g_per_cm2{};
    double energy_loss_threshold_g_per_cm2{};
    double weight_scale{};
    double inverse_weight_scale{};
    double energy_scale{};
    double inverse_energy_scale{};
  };

  void launchPhotonProfileProjectionOnDevice(
      DeviceProfileProjection const&,
      PhotonTransportRecord const* records, std::size_t count,
      ProjectedEmStepRecord* output);

  void launchLeptonProfileProjectionOnDevice(
      DeviceProfileProjection const&,
      LeptonTransportRecord const* records, std::size_t count,
      ProjectedEmStepRecord* output);

  void launchPhotonProfileAccumulationOnDevice(
      DeviceProfileProjection const&,
      DeviceProfileAccumulator const&,
      PhotonTransportRecord const* records, std::size_t count,
      PhotonPairFinalStateRecord const* final_states,
      std::size_t final_state_count,
      cudaStream_t stream = nullptr);

  void launchLeptonProfileAccumulationOnDevice(
      DeviceProfileProjection const&,
      DeviceProfileAccumulator const&,
      LeptonTransportRecord const* records, std::size_t count,
      BremsFinalStateRecord const* final_states,
      std::size_t final_state_count,
      cudaStream_t stream = nullptr);

} // namespace corsika::gpu::em::detail

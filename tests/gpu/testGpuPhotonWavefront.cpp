/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include <cuda_runtime_api.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <iterator>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <GpuEmFlatRateTableFixture.hpp>

#include <corsika/gpu/em/CudaEmBackend.hpp>
#include <corsika/gpu/em/EnvironmentSnapshotBuilder.hpp>
#include <corsika/gpu/em/PhotonPairKinematics.hpp>
#include <corsika/gpu/em/ProcessCapabilities.hpp>
#include <corsika/gpu/em/detail/DeviceWavefrontBucketing.hpp>
#include <corsika/gpu/em/detail/ProfileProjection.hpp>
#include <corsika/gpu/em/tables/RateTable.hpp>

namespace {

  using namespace corsika;
  using namespace corsika::gpu::em;
  using namespace corsika::gpu::em::tables;

  std::size_t checks = 0;

  void require(bool condition, std::string const& message) {
    ++checks;
    if (!condition) {
      throw std::runtime_error(message);
    }
  }

  template <typename Function>
  void requireThrows(
      Function&& function, std::string const& message) {
    ++checks;
    try {
      std::forward<Function>(function)();
    } catch (std::exception const&) {
      return;
    }
    throw std::runtime_error(message);
  }

  std::filesystem::path temporaryPath() {
    auto const stamp =
        std::chrono::high_resolution_clock::now()
            .time_since_epoch()
            .count();
    return std::filesystem::temp_directory_path() /
           ("c8_gpu_photon_wavefront_" +
            std::to_string(stamp) + ".c8emrt");
  }

  std::size_t processCount(RateTableSet const& table) {
    std::size_t count = 0;
    for (auto const& particle : table.particles) {
      count += particle.columns.size();
    }
    return count;
  }

  std::vector<EmParticleState> makePhotons(
      std::size_t count, double earth_radius_m) {
    std::vector<EmParticleState> particles;
    particles.reserve(count);
    for (std::size_t index = 0; index < count; ++index) {
      EmParticleState particle{};
      particle.pid =
          static_cast<std::int32_t>(EmPid::Photon);
      particle.medium_id = 17;
      auto const fraction =
          (static_cast<double>(index % 997) + 0.5) / 997.;
      particle.energy_GeV =
          (2. + 97. * fraction) / 1000.;
      // The lower half has far more than one interaction length before the
      // observation surface; the upper half normally crosses the thin fifth
      // layer before interacting.
      auto const altitude_m =
          index < count / 2 ? 2000. : 110000.;
      particle.position_m[2] =
          earth_radius_m + altitude_m;
      particle.direction[2] = -1.;
      particle.time_s = 1.e-6;
      particle.weight = 1.;
      particle.history_id = index + 1;
      particle.step_id = index % 7;
      particles.push_back(particle);
    }
    return particles;
  }

  void compareDeterministic(
      PhotonWavefrontBatchResult const& left,
      PhotonWavefrontBatchResult const& right) {
    require(left.input_particles == right.input_particles,
            "repeat input count differs");
    require(left.transport_records.size() ==
                right.transport_records.size(),
            "repeat transport count differs");
    require(left.next_photons.size() ==
                right.next_photons.size(),
            "repeat next-photon count differs");
    require(left.observations.size() ==
                right.observations.size(),
            "repeat observation count differs");
    require(left.selection_fallback_events.size() ==
                right.selection_fallback_events.size(),
            "repeat selection fallback count differs");
    require(left.transport_fallback_events.size() ==
                right.transport_fallback_events.size(),
            "repeat transport fallback count differs");
    require(left.final_states.gpu_interactions ==
                right.final_states.gpu_interactions,
            "repeat GPU final-state count differs");
    require(
        left.final_states.photon_pair_interactions ==
                right.final_states.photon_pair_interactions &&
            left.final_states.compton_interactions ==
                right.final_states.compton_interactions &&
            left.final_states.photoelectric_interactions ==
                right.final_states.photoelectric_interactions,
        "repeat per-process final-state count differs");
    require(left.final_states.lpm_suppressed.size() ==
                right.final_states.lpm_suppressed.size(),
            "repeat LPM queue count differs");
    require(left.final_states.fallback_events.size() ==
                right.final_states.fallback_events.size(),
            "repeat final-state fallback count differs");
    for (std::size_t index = 0;
         index < left.transport_records.size(); ++index) {
      auto const& a = left.transport_records[index];
      auto const& b = right.transport_records[index];
      require(a.input_index == b.input_index &&
                  a.limit == b.limit &&
                  a.distance_m == b.distance_m &&
                  a.traversed_grammage_g_per_cm2 ==
                      b.traversed_grammage_g_per_cm2 &&
                  a.observation_surface_reached_before_cut ==
                      b.observation_surface_reached_before_cut,
              "physical transport is not repeatable");
    }
    for (std::size_t index = 0;
         index < left.next_photons.size(); ++index) {
      auto const& a = left.next_photons[index];
      auto const& b = right.next_photons[index];
      require(a.history_id == b.history_id &&
                  a.step_id == b.step_id &&
                  a.position_m[2] == b.position_m[2],
              "next-photon queue is not repeatable");
    }
  }

  void compareFallbacks(
      std::vector<ProposalFallbackEvent> const& left,
      std::vector<ProposalFallbackEvent> const& right,
      std::string const& stage) {
    require(left.size() == right.size(),
            stage + " fallback count differs");
    for (std::size_t index = 0; index < left.size(); ++index) {
      require(
          left[index].input_index == right[index].input_index &&
              left[index].process_id ==
                  right[index].process_id &&
              left[index].reason == right[index].reason &&
              left[index].component_hash ==
                  right[index].component_hash &&
              left[index].random_draw_id ==
                  right[index].random_draw_id,
          stage + " fallback record differs");
    }
  }

  void compareTransportRecords(
      std::vector<PhotonTransportRecord> const& left,
      std::vector<PhotonTransportRecord> const& right) {
    require(left.size() == right.size(),
            "chained transport record count differs");
    for (std::size_t index = 0; index < left.size(); ++index) {
      auto const& a = left[index];
      auto const& b = right[index];
      require(
          a.input_index == b.input_index &&
              a.limit == b.limit &&
              a.start_layer_index == b.start_layer_index &&
              a.end_layer_index == b.end_layer_index &&
              a.distance_m == b.distance_m &&
              a.traversed_grammage_g_per_cm2 ==
                  b.traversed_grammage_g_per_cm2 &&
              a.start_density_g_per_cm3 ==
                  b.start_density_g_per_cm3 &&
              a.end_density_g_per_cm3 ==
                  b.end_density_g_per_cm3 &&
              a.cut_deposited_energy_GeV ==
                  b.cut_deposited_energy_GeV &&
              a.interaction.process_id ==
                  b.interaction.process_id &&
              a.interaction.component_hash ==
                  b.interaction.component_hash &&
              a.interaction.interaction_grammage_g_per_cm2 ==
                  b.interaction
                      .interaction_grammage_g_per_cm2 &&
              a.end.history_id == b.end.history_id &&
              a.end.step_id == b.end.step_id &&
              a.end.position_m[0] == b.end.position_m[0] &&
              a.end.position_m[1] == b.end.position_m[1] &&
              a.end.position_m[2] == b.end.position_m[2] &&
              a.end.time_s == b.end.time_s,
          "device-chained transport differs from legacy bridge");
    }
  }

  void compareFinalStates(
      EmFinalStateBatchResult const& left,
      EmFinalStateBatchResult const& right) {
    require(left.input_interactions == right.input_interactions &&
                left.gpu_interactions == right.gpu_interactions &&
                left.photon_pair_interactions ==
                    right.photon_pair_interactions &&
                left.compton_interactions ==
                    right.compton_interactions &&
                left.photoelectric_interactions ==
                    right.photoelectric_interactions,
            "device-pipeline final-state counts differ");
    require(left.final_state_records.size() ==
                right.final_state_records.size() &&
                left.secondaries.size() ==
                    right.secondaries.size() &&
                left.continuations.size() ==
                    right.continuations.size() &&
                left.lpm_suppressed.size() ==
                    right.lpm_suppressed.size(),
            "device-pipeline final-state queue sizes differ");
    compareFallbacks(
        left.fallback_events, right.fallback_events, "final-state");
    for (std::size_t index = 0;
         index < left.final_state_records.size(); ++index) {
      auto const& a = left.final_state_records[index];
      auto const& b = right.final_state_records[index];
      require(
              a.input_index == b.input_index &&
              a.parent_history_id == b.parent_history_id &&
              a.process_id == b.process_id &&
              a.secondary_offset == b.secondary_offset &&
              a.energy_split_fraction ==
                  b.energy_split_fraction &&
              a.lpm_survival_probability ==
                  b.lpm_survival_probability &&
              a.lpm_uniform == b.lpm_uniform,
          "device-pipeline photon final-state record differs");
    }
    for (std::size_t index = 0;
         index < left.secondaries.size(); ++index) {
      auto const& a = left.secondaries[index];
      auto const& b = right.secondaries[index];
      require(
          a.pid == b.pid && a.history_id == b.history_id &&
              a.parent_history_id == b.parent_history_id &&
              a.energy_GeV == b.energy_GeV &&
              a.direction[0] == b.direction[0] &&
              a.direction[1] == b.direction[1] &&
              a.direction[2] == b.direction[2],
          "device-pipeline secondary differs");
    }
    for (std::size_t index = 0;
         index < left.lpm_suppressed.size(); ++index) {
      auto const& a = left.lpm_suppressed[index];
      auto const& b = right.lpm_suppressed[index];
      require(
          a.input_index == b.input_index &&
              a.particle.history_id == b.particle.history_id &&
              a.particle.step_id == b.particle.step_id &&
              a.survival_probability == b.survival_probability &&
              a.uniform == b.uniform,
          "device-pipeline LPM suppression differs");
    }
  }

} // namespace

int main() {
  int device_count = 0;
  auto const cuda_status = cudaGetDeviceCount(&device_count);
  if (cuda_status != cudaSuccess || device_count == 0) {
    std::cerr << "SKIP: no usable CUDA device\n";
    return 77;
  }

  auto const temporary = temporaryPath();
  try {
    auto source = testing::makeFlatRateTableFixture();
    auto const digest = writeRateTable(temporary, source);
    ProposalTableSet descriptor{};
    descriptor.process_count =
        static_cast<std::uint32_t>(processCount(source));
    descriptor.content_hash = digest;

    auto const earth_radius_m =
        constants::EarthRadius::Mean / 1_m;
    auto const environment =
        makeCorsika7AtmosphereSnapshot(
            AtmosphereId::USStdBK, {0., 0., 0.}, 17,
            earth_radius_m + 100.);
    GpuEmConfig config{};
    config.device = 0;
    config.min_batch_size = 4096;
    config.memory_fraction = 0.10;
    config.table_tolerance = 1.e-3;
    config.random_seed = 0x5741564546524f4eULL;
    config.shower_id = 43;
    config.table_cache = temporary;
    auto const particles = makePhotons(4096, earth_radius_m);
    std::uint64_t constexpr FirstSecondary = 1000000;

    {
      // A synthetic accepted ionization vertex isolates the production
      // profile kernel from transport sampling.  Only the muon parent may be
      // added; an otherwise identical electron vertex is a negative control.
      constexpr std::size_t Bins = 8;
      // The last support value deliberately maps to bin == Bins before
      // clamping.  The icrc2025-beta2 scalar ProductionWriter places that
      // vertex in the final bin, so the resident GPU accumulator must match.
      double const axis_grammage[]{0., 10., 20., 30., 80.};
      double* device_axis = nullptr;
      long long* device_histograms = nullptr;
      corsika::gpu::em::detail::DeviceProfileCounters*
          device_counters = nullptr;
      LeptonTransportRecord* device_transports = nullptr;
      BremsFinalStateRecord* device_final_states = nullptr;
      require(
          cudaMalloc(reinterpret_cast<void**>(&device_axis),
                     sizeof(axis_grammage)) == cudaSuccess &&
              cudaMalloc(
                  reinterpret_cast<void**>(&device_histograms),
                  corsika::gpu::em::detail::
                      deviceProfileHistogramBytes(Bins)) == cudaSuccess &&
              cudaMalloc(
                  reinterpret_cast<void**>(&device_counters),
                  sizeof(corsika::gpu::em::detail::
                             DeviceProfileCounters)) == cudaSuccess &&
              cudaMalloc(
                  reinterpret_cast<void**>(&device_transports),
                  3 * sizeof(LeptonTransportRecord)) == cudaSuccess &&
              cudaMalloc(
                  reinterpret_cast<void**>(&device_final_states),
                  3 * sizeof(BremsFinalStateRecord)) == cudaSuccess,
          "could not allocate synthetic muon production profile fixture");
      require(
          cudaMemcpy(device_axis, axis_grammage,
                     sizeof(axis_grammage),
                     cudaMemcpyHostToDevice) == cudaSuccess &&
              cudaMemset(device_histograms, 0,
                         corsika::gpu::em::detail::
                             deviceProfileHistogramBytes(Bins)) ==
                  cudaSuccess &&
              cudaMemset(device_counters, 0,
                         sizeof(corsika::gpu::em::detail::
                                    DeviceProfileCounters)) ==
                  cudaSuccess,
          "could not initialize synthetic muon production profile fixture");

      LeptonTransportRecord transports[3]{};
      BremsFinalStateRecord final_states[3]{};
      for (std::size_t index = 0; index < 3; ++index) {
        auto& transport = transports[index];
        transport.input_index = index;
        transport.interaction.input_index = index;
        transport.interaction.process_id = IonizationProcessId;
        transport.start.pid =
            index == 1
                ? static_cast<std::int32_t>(EmPid::Electron)
                : static_cast<std::int32_t>(EmPid::MuonMinus);
        transport.start.history_id = 700 + index;
        transport.start.energy_GeV = 10.;
        transport.start.weight =
            index == 0 ? 3. : index == 1 ? 7. : 11.;
        transport.start.position_m[2] = 0.5;
        transport.end = transport.start;
        transport.end.position_m[2] =
            index == 0 ? 2.5 : index == 1 ? 3.5 : 4.;
        transport.limit =
            LeptonTransportLimit::InteractionCandidate;
        auto& final_state = final_states[index];
        final_state.input_index = index;
        final_state.parent_history_id =
            transport.start.history_id;
        final_state.secondary_count = 2;
        final_state.process_id = IonizationProcessId;
      }
      require(
          cudaMemcpy(device_transports, transports,
                     sizeof(transports),
                     cudaMemcpyHostToDevice) == cudaSuccess &&
              cudaMemcpy(device_final_states, final_states,
                         sizeof(final_states),
                         cudaMemcpyHostToDevice) == cudaSuccess,
          "could not upload synthetic muon production profile fixture");

      corsika::gpu::em::detail::DeviceProfileProjection
          projection{};
      projection.axis_direction[2] = 1.;
      projection.axis_step_length_m = 1.;
      projection.axis_grammage_g_per_cm2 = device_axis;
      projection.axis_support_count = 5;
      corsika::gpu::em::detail::DeviceProfileAccumulator
          accumulator{};
      accumulator.photons = device_histograms;
      accumulator.electrons = device_histograms + Bins;
      accumulator.positrons = device_histograms + 2 * Bins;
      accumulator.muons_minus = device_histograms + 3 * Bins;
      accumulator.muons_plus = device_histograms + 4 * Bins;
      accumulator.muon_parent_productions =
          device_histograms + 5 * Bins;
      accumulator.energy_loss = device_histograms + 6 * Bins;
      accumulator.muon_energy_loss =
          device_histograms + 7 * Bins;
      accumulator.counters = device_counters;
      accumulator.bins = Bins;
      accumulator.bin_width_g_per_cm2 = 10.;
      accumulator.energy_loss_threshold_g_per_cm2 = 1.e-4;
      accumulator.weight_scale = 1.;
      accumulator.inverse_weight_scale = 1.;
      accumulator.energy_scale = 1.e6;
      accumulator.inverse_energy_scale = 1.e-6;
      corsika::gpu::em::detail::
          launchLeptonProfileAccumulationOnDevice(
          projection, accumulator, device_transports, 3,
          device_final_states, 3);
      require(
          cudaDeviceSynchronize() == cudaSuccess,
          "synthetic muon production profile kernel failed");

      long long histograms[
          corsika::gpu::em::detail::
              DeviceProfileHistogramCount * Bins]{};
      corsika::gpu::em::detail::DeviceProfileCounters
          counters{};
      require(
          cudaMemcpy(histograms, device_histograms,
                     sizeof(histograms),
                     cudaMemcpyDeviceToHost) == cudaSuccess &&
              cudaMemcpy(&counters, device_counters,
                         sizeof(counters),
                         cudaMemcpyDeviceToHost) == cudaSuccess,
          "could not download synthetic muon production profile fixture");
      auto const electron_begin = histograms + Bins;
      auto const muon_minus_begin = histograms + 3 * Bins;
      auto const parent_begin = histograms + 5 * Bins;
      require(
          electron_begin[0] == 0 &&
              electron_begin[1] == 7 &&
              electron_begin[5] == 7 &&
              electron_begin[6] == 0 &&
              std::accumulate(
                  electron_begin, electron_begin + Bins,
                  0LL) == 35 &&
              muon_minus_begin[0] == 0 &&
              muon_minus_begin[1] == 14 &&
              muon_minus_begin[2] == 14 &&
              muon_minus_begin[3] == 11 &&
              muon_minus_begin[Bins - 1] == 11 &&
              std::accumulate(
                  muon_minus_begin,
                  muon_minus_begin + Bins, 0LL) == 83 &&
              parent_begin[2] == 3 &&
              parent_begin[Bins - 1] == 11 &&
              std::accumulate(parent_begin,
                              parent_begin + Bins, 0LL) == 14 &&
              counters.invalid_records == 0,
          "device lepton or muon-parent profile bin differs from icrc2025-beta2 scalar semantics");

      cudaFree(device_final_states);
      cudaFree(device_transports);
      cudaFree(device_counters);
      cudaFree(device_histograms);
      cudaFree(device_axis);
    }

    auto below_cut = makePhotons(2, earth_radius_m);
    below_cut[0].energy_GeV = 0.0004;
    below_cut[1].energy_GeV = 0.000499;
    CudaEmBackend cut_backend;
    cut_backend.initialize(environment, descriptor, config);
    auto const cut_wave =
        cut_backend.runPhotonDevicePipelineForValidation(
            below_cut, 9000000);
    require(
        cut_wave.selection_fallback_events.empty() &&
            cut_wave.transport_fallback_events.empty() &&
            cut_wave.final_states.fallback_events.empty() &&
            cut_wave.transport_records.size() ==
                below_cut.size() &&
            cut_wave.next_photons.empty() &&
            cut_wave.observations.empty() &&
            cut_wave.final_states.gpu_interactions == 0,
        "sub-cut photons did not terminate entirely on the GPU");
    for (std::size_t index = 0;
         index < cut_wave.transport_records.size(); ++index) {
      auto const& record = cut_wave.transport_records[index];
      require(
          record.limit == PhotonTransportLimit::ParticleCut &&
              record.distance_m == 0. &&
              record.traversed_grammage_g_per_cm2 == 0. &&
              record.cut_deposited_energy_GeV ==
                  below_cut[index].energy_GeV &&
              record.end.step_id ==
                  record.start.step_id + 1,
          "sub-cut photon terminal record differs");
    }
    {
      auto late_at_observation = makePhotons(1, earth_radius_m);
      auto& particle = late_at_observation.front();
      particle.energy_GeV = 0.05;
      particle.position_m[2] = earth_radius_m + 100.0001;
      particle.time_s =
          ParticleCutMaximumTimeS - 0.00005 / 299792458.;
      particle.history_id = 88000001;
      particle.step_id = 3;
      auto const late_wave =
          cut_backend.runPhotonDevicePipelineForValidation(
              late_at_observation, 9050000);
      require(
          late_wave.selection_fallback_events.empty() &&
              late_wave.transport_fallback_events.empty() &&
              late_wave.final_states.fallback_events.empty() &&
              late_wave.transport_records.size() == 1 &&
              late_wave.transport_records[0].limit ==
                  PhotonTransportLimit::ParticleCut &&
              late_wave.transport_records[0]
                      .observation_surface_reached_before_cut ==
                  1U &&
              late_wave.transport_records[0].end.time_s >
                  ParticleCutMaximumTimeS &&
              late_wave.transport_records[0]
                      .cut_deposited_energy_GeV ==
                  particle.energy_GeV &&
              late_wave.observations.size() == 1 &&
              late_wave.observations[0].status ==
                  ObservationStatus::ReachedObservationSurface &&
              late_wave.next_photons.empty() &&
              late_wave.final_states.gpu_interactions == 0,
          "post-step photon time cut does not match scalar observation-before-cut order");
    }
    auto const resident_cut =
        cut_backend.runResidentPhotonCascadeForValidation(
            below_cut, 9100000, 4);
    require(
        resident_cut.completed &&
            resident_cut.wavefronts == 1 &&
            resident_cut.particle_cuts == below_cut.size() &&
            resident_cut.fallback_events.empty() &&
            resident_cut.remaining_photons.empty() &&
            resident_cut.step_records.size() ==
                below_cut.size(),
        "resident sub-cut photon cascade did not finish in one wavefront");

    std::uint64_t saved_h2d_bytes = 0;
    std::uint64_t saved_d2h_bytes = 0;
    std::uint64_t resident_saved_h2d_bytes = 0;
    std::uint64_t resident_saved_d2h_bytes = 0;
    {
      CudaEmBackend legacy;
      legacy.initialize(environment, descriptor, config);
      auto const selection =
          legacy.selectInteractionsForValidation(particles);
      auto const transport =
          legacy.transportPhotonsForValidation(
              selection.interactions);
      std::vector<EmInteractionRecord> at_interaction;
      for (auto const& record : transport.records) {
        if (record.limit == PhotonTransportLimit::Interaction) {
          at_interaction.push_back(record.interaction);
        }
      }
      auto const legacy_final =
          legacy.generateFinalStatesForValidation(
              at_interaction, FirstSecondary);
      auto const legacy_statistics = legacy.statistics();

      CudaEmBackend chained;
      chained.initialize(environment, descriptor, config);
      auto const direct =
          chained.runPhotonDevicePipelineForValidation(
              particles, FirstSecondary);
      auto const chained_statistics = chained.statistics();

      compareFallbacks(
          selection.fallback_events,
          direct.selection_fallback_events, "selection");
      compareFallbacks(
          transport.fallback_events,
          direct.transport_fallback_events, "transport");
      compareTransportRecords(
          transport.records, direct.transport_records);
      compareFinalStates(legacy_final, direct.final_states);
      require(
          legacy_statistics.physical_host_to_device_bytes >
              chained_statistics.physical_host_to_device_bytes,
          "device chaining did not reduce H2D traffic");
      require(
          legacy_statistics.physical_device_to_host_bytes >
              chained_statistics.physical_device_to_host_bytes,
          "device chaining did not reduce D2H traffic");
      require(
          chained_statistics
                  .photon_selection_transport_summary_fusions ==
              1 &&
              chained_statistics
                  .photon_transport_final_state_summary_fusions ==
              1 &&
              chained_statistics
                  .photon_final_state_endpoint_summary_fusions ==
              1 &&
              chained_statistics
                      .pipeline_host_synchronizations_eliminated ==
                  3 &&
              chained_statistics
                      .pipeline_device_to_host_bytes_eliminated ==
                  32,
          "photon final-state/endpoint control summary fusion differs");
      bool deferred_history_overflow_rejected = false;
      try {
        (void)chained.runPhotonDevicePipelineForValidation(
            particles,
            std::numeric_limits<std::uint64_t>::max());
      } catch (std::overflow_error const&) {
        deferred_history_overflow_rejected = true;
      }
      require(
          deferred_history_overflow_rejected,
          "deferred photon control summary accepted overflowing history IDs");

      auto mixed_selection_particles =
          std::vector<EmParticleState>(
              particles.begin(), particles.begin() + 64);
      mixed_selection_particles[3].energy_GeV = 0.5;
      mixed_selection_particles[29].energy_GeV = 0.75;
      CudaEmBackend mixed_staged;
      mixed_staged.initialize(
          environment, descriptor, config);
      auto const mixed_selection =
          mixed_staged.selectInteractionsForValidation(
              mixed_selection_particles);
      auto const mixed_transport =
          mixed_staged.transportPhotonsForValidation(
              mixed_selection.interactions);
      std::vector<EmInteractionRecord>
          mixed_at_interaction;
      for (auto const& record : mixed_transport.records) {
        if (record.limit ==
            PhotonTransportLimit::Interaction) {
          mixed_at_interaction.push_back(
              record.interaction);
        }
      }
      auto const mixed_final =
          mixed_staged.generateFinalStatesForValidation(
              mixed_at_interaction, 7000000);
      CudaEmBackend mixed_chained;
      mixed_chained.initialize(
          environment, descriptor, config);
      auto const mixed_pipeline =
          mixed_chained
              .runPhotonDevicePipelineForValidation(
                  mixed_selection_particles, 7000000);
      require(
          mixed_selection.fallback_events.size() == 2 &&
              mixed_selection.interactions.size() == 62,
          "mixed deferred photon selection did not exercise both branches");
      compareFallbacks(
          mixed_selection.fallback_events,
          mixed_pipeline.selection_fallback_events,
          "mixed deferred selection");
      compareTransportRecords(
          mixed_transport.records,
          mixed_pipeline.transport_records);
      compareFallbacks(
          mixed_transport.fallback_events,
          mixed_pipeline.transport_fallback_events,
          "mixed deferred transport");
      compareFinalStates(
          mixed_final, mixed_pipeline.final_states);
      saved_h2d_bytes =
          legacy_statistics.physical_host_to_device_bytes -
          chained_statistics.physical_host_to_device_bytes;
      saved_d2h_bytes =
          legacy_statistics.physical_device_to_host_bytes -
          chained_statistics.physical_device_to_host_bytes;
      auto const interaction_bytes =
          selection.interactions.size() *
          sizeof(EmInteractionRecord);
      auto const vertex_bytes =
          at_interaction.size() * sizeof(EmInteractionRecord);
      require(saved_h2d_bytes ==
                  interaction_bytes + vertex_bytes,
              "pipeline H2D saving differs from its two compact payloads");
      auto const endpoint_download_bytes =
          direct.next_photons.size() * sizeof(EmParticleState) +
          direct.observations.size() * sizeof(ObservationRecord);
      require(interaction_bytes > endpoint_download_bytes &&
                  saved_d2h_bytes ==
                      interaction_bytes -
                          endpoint_download_bytes,
              "pipeline D2H saving differs after endpoint validation download");
    }

    {
      CudaEmBackend host_loop;
      host_loop.initialize(environment, descriptor, config);
      ResidentPhotonCascadeResult expected{};
      expected.input_particles = particles.size();
      std::vector<EmParticleState> current = particles;
      corsika::gpu::em::detail::DeviceWorkspace
          oracle_bucket_workspace;
      oracle_bucket_workspace.configure(
          config.device, 64ULL * 1024ULL * 1024ULL);
      auto next_history_id = FirstSecondary;
      while (!current.empty() && expected.wavefronts < 64) {
        expected.peak_resident_photons =
            std::max(
                expected.peak_resident_photons,
                current.size());
        auto bucketed =
            corsika::gpu::em::detail::
                bucketWavefrontForValidation(
                    current, config.device,
                    oracle_bucket_workspace);
        current = std::move(bucketed.particles);
        auto const step =
            host_loop.advancePhotonWavefrontForValidation(
                current, next_history_id);
        ++expected.wavefronts;
        expected.transport_records +=
            step.transport_records.size();
        expected.step_records.insert(
            expected.step_records.end(),
            step.transport_records.begin(),
            step.transport_records.end());
        for (auto const& record : step.transport_records) {
          if (record.limit ==
              PhotonTransportLimit::Interaction) {
            ++expected.interaction_vertices;
          } else if (
              record.limit ==
              PhotonTransportLimit::LayerBoundary) {
            ++expected.layer_boundaries;
          } else if (
              record.limit ==
              PhotonTransportLimit::ParticleCut) {
            ++expected.particle_cuts;
          }
        }
        expected.lpm_suppressions +=
            step.final_states.lpm_suppressed.size();
        expected.fallback_events.insert(
            expected.fallback_events.end(),
            step.selection_fallback_events.begin(),
            step.selection_fallback_events.end());
        expected.fallback_events.insert(
            expected.fallback_events.end(),
            step.transport_fallback_events.begin(),
            step.transport_fallback_events.end());
        expected.fallback_events.insert(
            expected.fallback_events.end(),
            step.final_states.fallback_events.begin(),
            step.final_states.fallback_events.end());
        std::copy_if(
            step.final_states.secondaries.begin(),
            step.final_states.secondaries.end(),
            std::back_inserter(
                expected.electromagnetic_secondaries),
            [](EmParticleState const& particle) {
              return particle.pid !=
                     static_cast<std::int32_t>(EmPid::Photon);
            });
        expected.final_state_records.insert(
            expected.final_state_records.end(),
            step.final_states.final_state_records.begin(),
            step.final_states.final_state_records.end());
        expected.observations.insert(
            expected.observations.end(),
            step.observations.begin(), step.observations.end());
        next_history_id += step.final_states.secondaries.size();
        current = step.next_photons;
      }
      expected.completed = current.empty();
      expected.remaining_photons = current;
      require(expected.completed,
              "host oracle photon cascade exceeded wavefront limit");
      auto const host_statistics = host_loop.statistics();

      CudaEmBackend resident;
      resident.initialize(environment, descriptor, config);
      auto const actual =
          resident.runResidentPhotonCascadeForValidation(
              particles, FirstSecondary, 64);
      auto const resident_statistics = resident.statistics();
      require(actual.completed && actual.remaining_photons.empty(),
              "resident photon cascade did not complete");
      require(
          actual.wavefronts == expected.wavefronts &&
              actual.peak_resident_photons ==
                  expected.peak_resident_photons &&
              actual.transport_records ==
                  expected.transport_records &&
              actual.interaction_vertices ==
                  expected.interaction_vertices &&
              actual.layer_boundaries ==
                  expected.layer_boundaries &&
              actual.particle_cuts ==
                  expected.particle_cuts &&
              actual.lpm_suppressions ==
                  expected.lpm_suppressions,
          "resident cascade counters differ from host-loop oracle");
      compareFallbacks(
          actual.fallback_events, expected.fallback_events,
          "resident cascade");
      compareTransportRecords(
          actual.step_records, expected.step_records);
      require(
          actual.electromagnetic_secondaries.size() ==
              expected.electromagnetic_secondaries.size(),
          "resident cascade secondary count differs");
      for (std::size_t index = 0;
           index < actual.electromagnetic_secondaries.size();
           ++index) {
        auto const& a =
            actual.electromagnetic_secondaries[index];
        auto const& b =
            expected.electromagnetic_secondaries[index];
        require(
            a.pid == b.pid && a.history_id == b.history_id &&
                a.parent_history_id == b.parent_history_id &&
                a.energy_GeV == b.energy_GeV &&
                a.direction[0] == b.direction[0] &&
                a.direction[1] == b.direction[1] &&
                a.direction[2] == b.direction[2],
            "resident cascade secondary differs");
      }
      require(actual.final_state_records.size() ==
                  expected.final_state_records.size(),
              "resident cascade final-state record count differs");
      for (std::size_t index = 0;
           index < actual.final_state_records.size(); ++index) {
        require(
            actual.final_state_records[index].parent_history_id ==
                    expected.final_state_records[index]
                        .parent_history_id &&
                actual.final_state_records[index]
                        .energy_split_fraction ==
                    expected.final_state_records[index]
                        .energy_split_fraction,
            "resident cascade final-state record differs");
      }
      require(actual.observations.size() ==
                  expected.observations.size(),
              "resident cascade observation count differs");
      for (std::size_t index = 0;
           index < actual.observations.size(); ++index) {
        require(
            actual.observations[index].status ==
                    expected.observations[index].status &&
                actual.observations[index].particle.history_id ==
                    expected.observations[index]
                        .particle.history_id,
            "resident cascade observation differs");
      }
      require(
          host_statistics.physical_host_to_device_bytes >
                  resident_statistics
                      .physical_host_to_device_bytes &&
              host_statistics.physical_device_to_host_bytes >
                  resident_statistics
                      .physical_device_to_host_bytes,
          "resident cascade did not reduce both transfer directions");
      resident_saved_h2d_bytes =
          host_statistics.physical_host_to_device_bytes -
          resident_statistics.physical_host_to_device_bytes;
      resident_saved_d2h_bytes =
          host_statistics.physical_device_to_host_bytes -
          resident_statistics.physical_device_to_host_bytes;
    }

    {
      auto projected_config = config;
      auto& projection =
          projected_config.profile_projection;
      projection.enabled = true;
      projection.axis_start_position_m[2] =
          earth_radius_m + 120000.;
      projection.axis_direction[2] = -1.;
      projection.axis_step_length_m = 1000.;
      projection.axis_grammage_g_per_cm2.resize(130);
      for (std::size_t index = 0;
           index <
           projection.axis_grammage_g_per_cm2.size();
           ++index) {
        projection.axis_grammage_g_per_cm2[index] =
            10. * static_cast<double>(index);
      }
      projection.output_bin_count = 130;
      projection.output_bin_width_g_per_cm2 = 10.;
      projection.energy_loss_threshold_g_per_cm2 =
          1.e-4;
      projection.fixed_point_weight_limit = 1.e6;
      projection.fixed_point_energy_limit_GeV = 1.e6;

      CudaEmBackend projected_backend;
      projected_backend.initialize(
          environment, descriptor, projected_config);
      auto const projected =
          projected_backend
              .runResidentPhotonCascadeForValidation(
                  particles, FirstSecondary, 2,
                  config.min_batch_size);
      require(
          !projected.projected_step_records.empty() &&
              !projected.final_state_records.empty(),
          "host-projected photon oracle did not retain its records");
      auto const projected_statistics =
          projected_backend.statistics();

      auto accumulator_config = projected_config;
      accumulator_config.profile_projection
          .accumulate_on_device = true;
      CudaEmBackend accumulator_backend;
      accumulator_backend.initialize(
          environment, descriptor, accumulator_config);
      requireThrows(
          [&] {
            accumulator_backend.beginShower(
                makeGpuEmShowerConfig(
                    accumulator_config));
          },
          "backend reuse discarded an undownloaded resident profile");
      auto const accumulated =
          accumulator_backend
              .runResidentPhotonCascadeForValidation(
                  particles, FirstSecondary, 2,
                  config.min_batch_size);
      auto const profile =
          accumulator_backend.downloadProfile();
      auto const accumulator_statistics =
          accumulator_backend.statistics();
      require(
          accumulator_backend.gpuProfileEnabled() &&
              accumulated.projected_step_records.empty() &&
              accumulated.final_state_records.empty(),
          "resident profile path copied per-step records to the host");
      require(
          profile.steps == accumulated.transport_records &&
              profile.fixed_point_overflows == 0 &&
              profile.invalid_records == 0 &&
              profile.photons.size() ==
                  projection.output_bin_count &&
              profile.electrons.size() ==
                  projection.output_bin_count &&
              profile.positrons.size() ==
                  projection.output_bin_count &&
              profile.muon_parent_productions.size() ==
                  projection.output_bin_count &&
              profile.energy_loss_GeV.size() ==
                  projection.output_bin_count,
          "resident photon profile counters or bin counts differ");
      require(
          std::accumulate(
              profile.photons.begin(),
              profile.photons.end(), 0.) > 0. &&
              std::accumulate(
                  profile.energy_loss_GeV.begin(),
                  profile.energy_loss_GeV.end(), 0.) > 0.,
          "resident photon profile did not accumulate tracks and deposits");

      // The resident path used to be checked only for determinism, valid
      // counters, and non-zero output.  Reconstruct the exact fixed-point
      // histogram from the host-projected records of an otherwise identical
      // deterministic cascade so a bin-boundary or track-weight semantic
      // change cannot pass unnoticed.
      require(
          projected.projected_step_records.size() ==
              profile.steps,
          "resident photon profile step count differs from the host-projected oracle");
      constexpr double FixedPointHeadroom = 0x1p62;
      auto const weight_scale =
          FixedPointHeadroom /
          projection.fixed_point_weight_limit;
      auto const inverse_weight_scale = 1. / weight_scale;
      std::vector<long long> expected_photons(
          projection.output_bin_count, 0);
      std::vector<long long> expected_electrons(
          projection.output_bin_count, 0);
      std::vector<long long> expected_positrons(
          projection.output_bin_count, 0);
      for (auto const& record :
           projected.projected_step_records) {
        if (record.start_grammage_g_per_cm2 ==
            record.end_grammage_g_per_cm2) {
          continue;
        }
        auto const first_value = std::ceil(
            record.start_grammage_g_per_cm2 /
            projection.output_bin_width_g_per_cm2);
        auto const last_value = std::floor(
            record.end_grammage_g_per_cm2 /
            projection.output_bin_width_g_per_cm2);
        if (!(first_value >= 0.) ||
            !(last_value >= first_value) ||
            first_value >= static_cast<double>(
                               projection.output_bin_count)) {
          continue;
        }
        auto* expected =
            record.pid ==
                    static_cast<std::int32_t>(EmPid::Photon)
                ? &expected_photons
                : record.pid == static_cast<std::int32_t>(
                                      EmPid::Electron)
                      ? &expected_electrons
                      : record.pid == static_cast<std::int32_t>(
                                            EmPid::Positron)
                            ? &expected_positrons
                            : nullptr;
        require(
            expected != nullptr,
            "host-projected profile oracle contains an invalid EM PID");
        auto const increment = static_cast<long long>(
            std::nearbyint(record.weight * weight_scale));
        auto const first = static_cast<std::size_t>(first_value);
        auto const last = std::min(
            static_cast<std::size_t>(last_value),
            projection.output_bin_count - 1);
        for (auto bin = first; bin <= last; ++bin) {
          expected->at(bin) += increment;
        }
      }
      for (std::size_t bin = 0;
           bin < projection.output_bin_count; ++bin) {
        require(
            profile.photons[bin] ==
                    static_cast<double>(expected_photons[bin]) *
                        inverse_weight_scale &&
                profile.electrons[bin] ==
                    static_cast<double>(expected_electrons[bin]) *
                        inverse_weight_scale &&
                profile.positrons[bin] ==
                    static_cast<double>(expected_positrons[bin]) *
                        inverse_weight_scale,
            "resident particle profile bin differs from the host-projected oracle");
      }
      auto expected_medium_input_GeV = 0.;
      for (auto const& final_state :
           projected.final_state_records) {
        if (final_state.process_id != ComptonProcessId &&
            final_state.process_id !=
                PhotoelectricProcessId) {
          continue;
        }
        auto const step = std::find_if(
            projected.projected_step_records.begin(),
            projected.projected_step_records.end(),
            [&](ProjectedEmStepRecord const& candidate) {
              return candidate.history_id ==
                         final_state.parent_history_id &&
                     candidate.process_id ==
                         final_state.process_id;
            });
        require(
            step != projected.projected_step_records.end(),
            "host-projected atomic-electron state has no transport step");
        expected_medium_input_GeV +=
            ElectronMassGeV * step->weight;
      }
      auto expected_observed_energy_GeV = 0.;
      auto expected_escaped_energy_GeV = 0.;
      for (auto const& observation :
           projected.observations) {
        auto const weighted =
            observation.particle.energy_GeV *
            observation.particle.weight;
        if (observation.status ==
            ObservationStatus::ReachedObservationSurface) {
          expected_observed_energy_GeV += weighted;
        } else {
          expected_escaped_energy_GeV += weighted;
        }
      }
      auto const ledger_tolerance_GeV =
          2.e-9;
      require(
          std::abs(
              profile
                  .weighted_medium_rest_mass_input_GeV -
              expected_medium_input_GeV) <
                  ledger_tolerance_GeV &&
              std::abs(
                  profile
                      .weighted_observed_total_energy_GeV -
                  expected_observed_energy_GeV) <
                  ledger_tolerance_GeV &&
              std::abs(
                  profile
                      .weighted_escaped_total_energy_GeV -
                  expected_escaped_energy_GeV) <
                  ledger_tolerance_GeV &&
              profile
                      .weighted_cut_rest_mass_energy_GeV ==
                  0.,
          "resident photon energy-ledger terms differ from the host oracle");
      require(
          accumulator_statistics
                  .physical_device_to_host_bytes <
              projected_statistics
                  .physical_device_to_host_bytes,
          "resident photon profile did not reduce device-to-host traffic");

      accumulator_backend.beginShower(
          makeGpuEmShowerConfig(accumulator_config));
      auto const repeated_accumulated =
          accumulator_backend
              .runResidentPhotonCascadeForValidation(
                  particles, FirstSecondary, 2,
                  config.min_batch_size);
      auto const repeated_profile =
          accumulator_backend.downloadProfile();
      auto const repeated_statistics =
          accumulator_backend.statistics();
      require(
          repeated_accumulated.transport_records ==
                  accumulated.transport_records &&
              repeated_accumulated.interaction_vertices ==
                  accumulated.interaction_vertices &&
              repeated_accumulated.particle_cuts ==
                  accumulated.particle_cuts &&
              repeated_profile.photons ==
                  profile.photons &&
              repeated_profile.electrons ==
                  profile.electrons &&
              repeated_profile.positrons ==
                  profile.positrons &&
              repeated_profile.muon_parent_productions ==
                  profile.muon_parent_productions &&
              repeated_profile.muon_energy_loss_GeV ==
                  profile.muon_energy_loss_GeV &&
              repeated_profile.energy_loss_GeV ==
                  profile.energy_loss_GeV,
          "reused backend profile is not bitwise identical to its first shower");
      require(
          repeated_profile.steps == profile.steps &&
              repeated_profile.deposited_steps ==
                  profile.deposited_steps &&
              repeated_profile.particle_cuts ==
                  profile.particle_cuts &&
              repeated_profile
                      .weighted_deposited_energy_GeV ==
                  profile
                      .weighted_deposited_energy_GeV &&
              repeated_profile
                      .weighted_medium_rest_mass_input_GeV ==
                  profile
                      .weighted_medium_rest_mass_input_GeV &&
              repeated_profile
                      .weighted_observed_total_energy_GeV ==
                  profile
                      .weighted_observed_total_energy_GeV &&
              repeated_profile
                      .weighted_escaped_total_energy_GeV ==
                  profile
                      .weighted_escaped_total_energy_GeV,
          "reused backend profile counters or energy ledger leaked state");
      require(
          repeated_statistics.shower_ordinal == 2 &&
              repeated_statistics.reused_for_shower &&
              repeated_statistics.table_device_bytes ==
                  accumulator_statistics.table_device_bytes &&
              repeated_statistics.physical_workspace_bytes ==
                  accumulator_statistics
                      .physical_workspace_bytes &&
              repeated_statistics.profile.steps ==
                  accumulator_statistics.profile.steps,
          "reused profile backend did not preserve allocations or reset statistics");

      CudaEmBackend cut_ledger_backend;
      cut_ledger_backend.initialize(
          environment, descriptor, accumulator_config);
      auto cut_leptons = makePhotons(2, earth_radius_m);
      cut_leptons[0].pid =
          static_cast<std::int32_t>(EmPid::Electron);
      cut_leptons[1].pid =
          static_cast<std::int32_t>(EmPid::Positron);
      cut_leptons[0].energy_GeV =
          ElectronMassGeV + 0.0001;
      cut_leptons[1].energy_GeV =
          ElectronMassGeV + 0.0002;
      cut_leptons[0].weight = 2.;
      cut_leptons[1].weight = 3.;
      auto const cut_ledger_cascade =
          cut_ledger_backend
              .runResidentLeptonCascadeForValidation(
                  cut_leptons, 9800000, 4);
      auto const cut_ledger =
          cut_ledger_backend.downloadProfile();
      auto const expected_cut_deposit_GeV =
          0.0001 * 2. + 0.0002 * 3.;
      auto const expected_cut_rest_GeV =
          ElectronMassGeV * 5.;
      require(
          cut_ledger_cascade.completed &&
              cut_ledger_cascade.transport_records == 2 &&
              cut_ledger.particle_cuts == 2 &&
              std::abs(
                  cut_ledger
                          .weighted_deposited_energy_GeV -
                      expected_cut_deposit_GeV) <
                  ledger_tolerance_GeV &&
              std::abs(
                  cut_ledger
                          .weighted_cut_rest_mass_energy_GeV -
                      expected_cut_rest_GeV) <
                  ledger_tolerance_GeV &&
              cut_ledger
                      .weighted_medium_rest_mass_input_GeV ==
                  0. &&
              cut_ledger
                      .weighted_observed_total_energy_GeV ==
                  0. &&
              cut_ledger
                      .weighted_escaped_total_energy_GeV ==
                  0.,
          "resident lepton ParticleCut energy ledger is incomplete");

      auto late_terminal = makePhotons(1, earth_radius_m);
      late_terminal[0].energy_GeV = 0.05;
      late_terminal[0].position_m[2] =
          earth_radius_m + 100.0001;
      late_terminal[0].time_s =
          ParticleCutMaximumTimeS - 0.00005 / 299792458.;
      late_terminal[0].history_id = 9900001;
      late_terminal[0].step_id = 3;
      CudaEmBackend late_terminal_backend;
      late_terminal_backend.initialize(
          environment, descriptor, accumulator_config);
      auto const late_terminal_cascade =
          late_terminal_backend
              .runResidentPhotonCascadeForValidation(
                  late_terminal, 9900100, 4);
      auto const late_terminal_profile =
          late_terminal_backend.downloadProfile();
      require(
          late_terminal_cascade.completed &&
              late_terminal_cascade.particle_cuts == 1 &&
              late_terminal_cascade.observations.size() == 1 &&
              late_terminal_cascade.observations[0].status ==
                  ObservationStatus::ReachedObservationSurface &&
              std::abs(
                  late_terminal_profile
                          .weighted_deposited_energy_GeV -
                      late_terminal[0].energy_GeV) <
                  ledger_tolerance_GeV &&
              std::abs(
                  late_terminal_profile
                          .weighted_observed_total_energy_GeV -
                      late_terminal[0].energy_GeV) <
                  ledger_tolerance_GeV &&
              late_terminal_profile
                      .weighted_escaped_total_energy_GeV ==
                  0.,
          "resident observation-before-time-cut energy ledger is incomplete");

    }

    {
      auto thinned_config = config;
      thinned_config.thinning = {
          1.e20, 1.e20, 1, 1};
      CudaEmBackend thinned;
      thinned.initialize(
          environment, descriptor, thinned_config);
      auto const pipeline =
          thinned.runPhotonDevicePipelineForValidation(
              particles, FirstSecondary);
      auto expected_next_photons = std::count_if(
          pipeline.transport_records.begin(),
          pipeline.transport_records.end(),
          [](PhotonTransportRecord const& record) {
            return record.limit ==
                   PhotonTransportLimit::LayerBoundary;
          });
      expected_next_photons +=
          pipeline.final_states.lpm_suppressed.size();
      require(
          pipeline.final_states.secondaries.size() ==
              pipeline.final_states.gpu_interactions,
          "Hillas photon pipeline did not compact each accepted "
          "final state to one child");
      for (auto const& record :
           pipeline.final_states.final_state_records) {
        require(
            record.secondary_count == 1,
            "Hillas photon pipeline retained the wrong child count");
        if (record.process_id == PhotoelectricProcessId) {
          require(
              record.thinning_status ==
                  static_cast<std::uint32_t>(
                      EmThinningStatus::NotApplied),
              "photon pipeline thinned a 1->1 photoelectric state");
          continue;
        }
        require(
            record.thinning_status ==
                    static_cast<std::uint32_t>(
                        EmThinningStatus::Hillas) &&
                (record.thinning_keep_mask == 0x1U ||
                 record.thinning_keep_mask == 0x2U),
            "photon pipeline has invalid Hillas metadata");
        if (record.process_id == ComptonProcessId &&
            record.thinning_keep_mask == 0x1U) {
          ++expected_next_photons;
          auto const& child =
              pipeline.final_states.secondaries[
                  record.secondary_offset];
          require(
              child.pid ==
                  static_cast<std::int32_t>(EmPid::Photon),
              "Compton keep mask does not address its photon");
        }
      }
      require(
          pipeline.next_photons.size() ==
              static_cast<std::size_t>(
                  expected_next_photons),
          "thinned Compton photon endpoint compaction differs");
      auto const resident =
          thinned.runResidentPhotonCascadeForValidation(
              particles, 5000000, 64);
      require(
          resident.completed &&
              resident.remaining_photons.empty() &&
              !resident.final_state_records.empty() &&
              std::all_of(
                  resident.final_state_records.begin(),
                  resident.final_state_records.end(),
                  [](PhotonPairFinalStateRecord const& record) {
                    return record.secondary_count == 1;
                  }),
          "resident Hillas photon cascade did not complete with compact "
          "one-child final states");
    }

    {
      auto cross_config = config;
      cross_config.resident_cross_species = true;
      CudaEmBackend cross_backend;
      cross_backend.initialize(
          environment, descriptor, cross_config);
      require(
          cross_backend.statistics()
                  .cross_species_queue_capacity_per_pid >=
              cross_config.min_batch_size &&
              cross_backend.statistics()
                      .cross_species_queue_device_bytes > 0,
          "resident cross-species queues were not allocated");

      auto const photon_result =
          cross_backend
              .runResidentPhotonCascadeForValidation(
                  particles, 10000000, 64, 1);
      auto const pending_leptons =
          cross_backend.pendingLeptonCount();
      require(
          photon_result.completed &&
              photon_result.electromagnetic_secondaries.empty() &&
              pending_leptons > 0 &&
              cross_backend.pendingPhotonCount() == 0,
          "photon final states did not remain in the device lepton queue");
      auto const& cross_statistics =
          cross_backend.statistics();
      require(
          cross_statistics
                      .cross_species_particles_kept_on_device >=
                  pending_leptons &&
              cross_statistics
                      .cross_species_device_to_device_bytes > 0 &&
              cross_statistics.cross_species_host_spills == 0,
          "resident cross-species queue accounting is inconsistent");
    }

    {
      auto chunk_config = config;
      chunk_config.min_batch_size = 64;
      chunk_config.memory_fraction = 0.01;
      chunk_config.resident_cross_species = true;
      CudaEmBackend chunk_backend;
      chunk_backend.initialize(
          environment, descriptor, chunk_config);
      auto const maximum_photons =
          chunk_backend.maximumResidentPhotonBatchSize();
      auto const maximum_leptons =
          chunk_backend.maximumResidentLeptonBatchSize();
      auto const large_front =
          makePhotons(maximum_photons, earth_radius_m);
      auto const photon_result =
          chunk_backend
              .runResidentPhotonCascadeForValidation(
                  large_front, 40000000, 64, 1);
      auto const pending_before =
          chunk_backend.pendingLeptonCount();
      require(
          photon_result.completed &&
              pending_before > maximum_leptons &&
              chunk_backend.statistics()
                      .cross_species_host_spills == 0,
          "large photon front did not create a chunked device lepton queue");

      auto const first_chunk =
          chunk_backend
              .runResidentLeptonCascadeForValidation(
                  {}, 50000000, 1,
                  std::numeric_limits<std::uint64_t>::max(), 1);
      require(
          first_chunk.input_particles ==
                  maximum_leptons &&
              chunk_backend.pendingLeptonCount() ==
                  pending_before - maximum_leptons,
          "resident lepton queue did not retain its unconsumed tail");
    }

    {
      auto spill_config = config;
      spill_config.min_batch_size = 64;
      spill_config.memory_fraction = 0.01;
      spill_config.resident_cross_species = true;
      CudaEmBackend spill_backend;
      spill_backend.initialize(
          environment, descriptor, spill_config);
      auto const maximum_photons =
          spill_backend.maximumResidentPhotonBatchSize();
      auto const queue_capacity =
          spill_backend.statistics()
              .cross_species_queue_capacity_per_pid;
      auto const large_front =
          makePhotons(maximum_photons, earth_radius_m);
      bool observed_capacity_spill = false;
      std::size_t previous_pending = 0;
      for (std::size_t attempt = 0; attempt < 64; ++attempt) {
        auto const first_history =
            60000000ULL +
            static_cast<std::uint64_t>(attempt) *
                static_cast<std::uint64_t>(
                    2 * maximum_photons + 1);
        auto const result =
            spill_backend
                .runResidentPhotonCascadeForValidation(
                    large_front, first_history, 64, 1);
        auto const pending =
            spill_backend.pendingLeptonCount();
        if (spill_backend.statistics()
                .cross_species_host_spills != 0) {
          require(
              result.electromagnetic_secondaries.empty() &&
                  !result.cpu_spill_particles.empty() &&
                  pending >= previous_pending &&
                  pending == queue_capacity,
              "full resident lepton queue did not retain the high-energy "
              "front and return only its low-energy tail to the CPU");
          require(
              spill_backend.statistics()
                          .cross_species_particles_spilled_to_cpu ==
                      result.cpu_spill_particles.size() &&
                  spill_backend.statistics()
                          .cross_species_low_energy_ordering_checks ==
                      spill_backend.statistics()
                          .cross_species_host_spills,
              "capacity spill is not a checked, deterministic low-energy "
              "CPU tail");
          observed_capacity_spill = true;
          break;
        }
        require(
            result.electromagnetic_secondaries.empty() &&
                pending > previous_pending &&
                pending <= queue_capacity,
            "resident charged-secondary queue did not grow monotonically");
        previous_pending = pending;
      }
      require(
          observed_capacity_spill,
          "charged-secondary capacity fallback was not exercised");
    }

    CudaEmBackend backend;
    backend.initialize(environment, descriptor, config);
    auto const initialized_workspace_bytes =
        backend.statistics().physical_workspace_bytes;
    auto const maximum_photon_batch =
        backend.maximumResidentPhotonBatchSize();
    auto const maximum_lepton_batch =
        backend.maximumResidentLeptonBatchSize();
    require(
        maximum_photon_batch >= config.min_batch_size &&
            maximum_lepton_batch >= config.min_batch_size &&
            backend.maximumResidentInputBatchSize() ==
                std::min(
                    maximum_photon_batch,
                    maximum_lepton_batch),
        "reported resident workspace capacities are inconsistent");
    require(
        backend.statistics().maximum_resident_photon_batch ==
                maximum_photon_batch &&
            backend.statistics().maximum_resident_lepton_batch ==
                maximum_lepton_batch,
        "resident workspace capacities were not recorded in statistics");
    require(initialized_workspace_bytes >=
                4096 * config.min_batch_size,
            "physical workspace was not preallocated at initialization");

    {
      CudaEmBackend checkpoint_backend;
      checkpoint_backend.initialize(
          environment, descriptor, config);
      auto const checkpoint_capacity =
          checkpoint_backend.maximumResidentPhotonBatchSize();
      require(
          checkpoint_capacity <
              std::numeric_limits<std::size_t>::max(),
          "resident photon capacity cannot be incremented for testing");
      auto const checkpoint_input =
          makePhotons(
              checkpoint_capacity + 1,
              earth_radius_m);
      auto const checkpoint =
          checkpoint_backend
              .runResidentPhotonCascadeForValidation(
                  checkpoint_input, 12000000, 1, 1);
      require(
          checkpoint.workspace_limit_checkpoint &&
              !checkpoint.completed &&
              checkpoint.wavefronts == 0 &&
              checkpoint.transport_records == 0 &&
              checkpoint.remaining_photons.size() ==
                  checkpoint_input.size(),
          "oversize photon front did not create a lossless workspace checkpoint");
      require(
          checkpoint.remaining_photons.front().history_id ==
                  checkpoint_input.front().history_id &&
              checkpoint.remaining_photons.back().history_id ==
                  checkpoint_input.back().history_id,
          "workspace checkpoint changed the resident photon order");
    }

    auto const first =
        backend.advancePhotonWavefrontForValidation(
            particles, FirstSecondary);
    auto const repeat =
        backend.advancePhotonWavefrontForValidation(
            particles, FirstSecondary);
    compareDeterministic(first, repeat);

    require(!first.selection_fallback_events.empty() &&
                first.selection_fallback_events.size() < 100,
            "fixture did not exercise its bounded loss-quantile fallback");
    for (auto const& fallback :
         first.selection_fallback_events) {
      require(
          fallback.reason ==
              ProposalFallbackReason::LossQuantileOutOfRange,
          "selection returned an unexpected fallback reason");
    }
    require(first.transport_fallback_events.empty(),
            "supported spherical transport unexpectedly fell back");
    require(
        first.transport_records.size() +
                first.selection_fallback_events.size() ==
            particles.size(),
        "wavefront lost or duplicated selected/fallback photons");
    require(first.observations.empty(),
            "one wavefront unexpectedly reached an endpoint");

    std::size_t interaction_count = 0;
    std::size_t boundary_count = 0;
    for (std::size_t index = 0;
         index < first.transport_records.size(); ++index) {
      auto const& record = first.transport_records[index];
      require(index == 0 ||
                  record.input_index >
                      first.transport_records[index - 1]
                          .input_index,
              "transport records are not stably ordered");
      if (record.limit == PhotonTransportLimit::Interaction) {
        ++interaction_count;
        require(record.interaction.mass_density_g_per_cm3 > 0.,
                "interaction has no atmosphere density for LPM");
      } else if (
          record.limit == PhotonTransportLimit::LayerBoundary) {
        ++boundary_count;
        require(record.end.step_id ==
                    record.start.step_id + 1,
                "boundary photon did not advance its RNG step");
      } else {
        throw std::runtime_error(
            "unexpected endpoint in mixed wavefront fixture");
      }
    }
    require(interaction_count > 1500,
            "lower atmosphere did not exercise interactions");
    require(boundary_count > 1000,
            "upper atmosphere did not exercise boundary queue");

    auto const final_closure =
        first.final_states.gpu_interactions +
        first.final_states.lpm_suppressed.size() +
        first.final_states.continuations.size() +
        first.final_states.fallback_events.size();
    require(final_closure == interaction_count,
            "final-state dispatch does not close interaction queue");
    require(first.final_states.photon_pair_interactions > 100,
            "wavefront did not generate enough photon pairs");
    require(first.final_states.compton_interactions > 100,
            "wavefront did not generate enough Compton final states");
    require(first.final_states.photoelectric_interactions > 30,
            "wavefront did not generate enough photoelectric final states");
    require(first.final_states.secondaries.size() ==
                2 *
                        (first.final_states
                             .photon_pair_interactions +
                         first.final_states
                             .compton_interactions) +
                    first.final_states
                        .photoelectric_interactions,
            "GPU photon final-state multiplicity does not close");
    for (auto const& fallback :
         first.final_states.fallback_events) {
      require(fallback.process_id != ComptonProcessId,
              "valid fixture Compton interaction fell back to CPU");
    }
    require(
        first.next_photons.size() ==
            boundary_count +
                first.final_states.lpm_suppressed.size() +
                first.final_states.compton_interactions,
        "next wavefront queue does not close all photon continuations");
    std::vector<std::pair<std::uint64_t, std::uint64_t>>
        expected_next;
    for (auto const& record : first.transport_records) {
      if (record.limit == PhotonTransportLimit::LayerBoundary) {
        expected_next.emplace_back(
            record.input_index, record.end.history_id);
      }
    }
    for (auto const& suppression :
         first.final_states.lpm_suppressed) {
      expected_next.emplace_back(
          suppression.input_index,
          suppression.particle.history_id);
    }
    for (auto const& record :
         first.final_states.final_state_records) {
      if (record.process_id == ComptonProcessId) {
        expected_next.emplace_back(
            record.input_index,
            first.final_states
                .secondaries[record.secondary_offset]
                .history_id);
      }
    }
    std::sort(expected_next.begin(), expected_next.end());
    require(expected_next.size() ==
                first.next_photons.size(),
            "source-order oracle has the wrong continuation count");
    for (std::size_t index = 0;
         index < first.next_photons.size(); ++index) {
      require(first.next_photons[index].history_id ==
                  expected_next[index].second,
              "next wavefront queue is not source-stable");
    }

    // Actually consume the returned queue once more. This proves that a
    // boundary state is valid input to the next layer and that an
    // LPM-suppressed state is resampled with its incremented Philox step.
    auto const second =
        backend.advancePhotonWavefrontForValidation(
            first.next_photons, FirstSecondary + 100000);
    require(second.input_particles ==
                first.next_photons.size(),
            "second physical wavefront input differs");
    require(second.transport_fallback_events.empty(),
            "second wavefront transport unexpectedly fell back");
    require(
        second.transport_records.size() +
                second.selection_fallback_events.size() ==
            first.next_photons.size(),
        "second wavefront lost or duplicated photons");

    auto const& statistics = backend.statistics();
    require(statistics.physical_photon_wavefronts == 3,
            "physical wavefront statistic differs");
    require(statistics.photon_transport_batches == 3,
            "transport batch statistic differs");
    require(statistics.interaction_selection_batches == 3,
            "selection batch statistic differs");
    require(statistics.physical_workspace_bytes ==
                initialized_workspace_bytes,
            "pre-sized workspace unexpectedly grew during wavefronts");
    require(statistics.peak_device_bytes >=
                statistics.table_device_bytes +
                    statistics.physical_workspace_bytes,
            "peak device accounting omits table or workspace");

    std::error_code ignored;
    std::filesystem::remove(temporary, ignored);
    std::cout << "GPU photon wavefront passed " << checks
              << " checks: " << interaction_count
              << " interactions, " << boundary_count
              << " boundaries, "
              << first.final_states.photon_pair_interactions
              << " pair and "
              << first.final_states.compton_interactions
              << " Compton and "
              << first.final_states.photoelectric_interactions
              << " photoelectric final states, "
              << first.final_states.fallback_events.size()
              << " explicit CPU fallbacks, saved "
              << saved_h2d_bytes << " H2D + "
              << saved_d2h_bytes
              << " D2H bytes per device pipeline; resident cascade saved "
              << resident_saved_h2d_bytes << " H2D + "
              << resident_saved_d2h_bytes
              << " D2H bytes versus host wavefront loop\n";
    return 0;
  } catch (std::exception const& error) {
    std::error_code ignored;
    std::filesystem::remove(temporary, ignored);
    std::cerr << "GPU photon wavefront failed after " << checks
              << " checks: " << error.what() << '\n';
    return 1;
  }
}

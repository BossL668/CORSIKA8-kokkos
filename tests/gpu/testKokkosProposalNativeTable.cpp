/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include <PROPOSAL/PROPOSAL.h>

#include "KokkosProposalNativeTableTestDriver.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <vector>

#include <corsika/accelerator/em/KokkosRuntime.hpp>
#include <corsika/accelerator/em/KokkosEmBackend.hpp>
#include <corsika/accelerator/em/detail/InteractionSelection.hpp>
#include <corsika/accelerator/em/detail/LeptonFinalStateStep.hpp>
#include <corsika/accelerator/em/detail/LeptonTransportStep.hpp>
#include <corsika/accelerator/em/detail/LeptonVertexSelection.hpp>
#include <corsika/accelerator/em/detail/MoliereStep.hpp>
#include <corsika/accelerator/em/detail/PhotonFinalStateStep.hpp>
#include <corsika/accelerator/em/detail/PhotonTransportStep.hpp>
#include <corsika/accelerator/radio/detail/RadioProjectionStep.hpp>
#include <corsika/framework/core/PhysicalUnits.hpp>
#include <corsika/gpu/em/EnvironmentSnapshotBuilder.hpp>
#include <corsika/gpu/em/tables/ProposalNativeTableExporter.hpp>
#include <corsika/gpu/em/tables/ProposalNativeAux.hpp>
#include <corsika/media/CORSIKA7Atmospheres.hpp>
#include <corsika/modules/proposal/ProposalProcessBase.hpp>

namespace {
  using namespace corsika;
  using namespace corsika::gpu::em::tables;

  struct CalculatorOwner {
    PROPOSAL::Air medium;
    Code code{Code::Unknown};
    PROPOSAL::crosssection_list_t cross_sections;
    std::unique_ptr<PROPOSAL::Interaction> interaction;
    std::unique_ptr<PROPOSAL::Displacement> displacement;
    std::unique_ptr<PROPOSAL::crosssection::PhotoPairLPM> photon_pair_lpm;
    std::unique_ptr<PROPOSAL::crosssection::BremsLPM> brems_lpm;
    double mass_MeV{};
    HEPEnergyType cut{};

    CalculatorOwner(Code const requested, HEPEnergyType const requested_cut)
        : code(requested)
        , mass_MeV(proposal::particle.at(requested).mass)
        , cut(requested_cut) {
      cross_sections = proposal::make_cross_sections(code, medium, cut, true);
      interaction = PROPOSAL::make_interaction(cross_sections, true, true);
      displacement = PROPOSAL::make_displacement(cross_sections, true);
      if (code == Code::Photon) {
        photon_pair_lpm =
            std::make_unique<PROPOSAL::crosssection::PhotoPairLPM>(
                proposal::particle.at(code), medium,
                PROPOSAL::crosssection::PhotoPairKochMotz());
      }
      if (code == Code::Electron || code == Code::Positron) {
        brems_lpm = std::make_unique<PROPOSAL::crosssection::BremsLPM>(
            proposal::particle.at(code), medium,
            PROPOSAL::crosssection::BremsElectronScreening());
      }
    }

    proposal::NativeInteractionCalculatorView interactionView() const {
      return {code, medium.GetHash(), &medium, interaction.get(),
              photon_pair_lpm.get(), brems_lpm.get(), cut, mass_MeV};
    }

    proposal::NativeContinuousCalculatorView continuousView() const {
      return {code, medium.GetHash(), &medium, displacement.get(), cut,
              mass_MeV};
    }
  };

  bool closeEnough(double const left, double const right) {
    auto const scale = std::max({std::abs(left), std::abs(right), 1.e-300});
    return std::isfinite(left) && std::isfinite(right) &&
           std::abs(left - right) <= 2.e-12 * scale;
  }

  bool sameOrClose(double const left, double const right) {
    return left == right || closeEnough(left, right);
  }

  struct HostRadioAtomicOperations {
    static long long add(long long* address, long long value) {
      auto const previous = *address;
      *address += value;
      return previous;
    }
    static unsigned long long add(unsigned long long* address,
                                  unsigned long long value) {
      auto const previous = *address;
      *address += value;
      return previous;
    }
    static double add(double* address, double value) {
      auto const previous = *address;
      *address += value;
      return previous;
    }
    static void maximum(double* address, double value) {
      *address = std::max(*address, value);
    }
  };

  gpu::radio::GpuRadioWaveforms projectRadioOnHost(
      gpu::radio::GpuRadioConfig const& config,
      std::vector<gpu::em::LeptonTransportRecord> const& records) {
    namespace radio_detail = accelerator::radio::detail;
    auto make_observer = [&](gpu::radio::RadioObserverSnapshot const& input,
                             bool zhs) {
      radio_detail::DeviceObserver result{};
      for (int axis = 0; axis < 3; ++axis)
        result.position_m[axis] = input.position_m[axis];
      result.start_time_s = input.start_time_s;
      result.duration_s = input.duration_s;
      result.sample_rate_Hz = input.sample_rate_Hz;
      auto const factor = zhs ? input.sample_rate_Hz : 1.;
      result.fixed_point_scale =
          radio_detail::FixedPointHeadroom * factor /
          config.fixed_point_field_limit_V_per_m;
      result.inverse_fixed_point_scale = 1. / result.fixed_point_scale;
      result.number_of_bins = input.number_of_bins;
      return result;
    };
    auto coreas_observer = make_observer(config.coreas_observers.front(), false);
    auto zhs_observer = make_observer(config.zhs_observers.front(), true);
    auto const bins = static_cast<std::size_t>(coreas_observer.number_of_bins);
    std::vector<long long> coreas_fixed(3 * bins);
    std::vector<long long> zhs_fixed(3 * bins);
    radio_detail::DeviceWaveforms coreas_waveforms{};
    coreas_waveforms.fixed_x = coreas_fixed.data();
    coreas_waveforms.fixed_y = coreas_fixed.data() + bins;
    coreas_waveforms.fixed_z = coreas_fixed.data() + 2 * bins;
    radio_detail::DeviceWaveforms zhs_waveforms{};
    zhs_waveforms.fixed_x = zhs_fixed.data();
    zhs_waveforms.fixed_y = zhs_fixed.data() + bins;
    zhs_waveforms.fixed_z = zhs_fixed.data() + 2 * bins;
    radio_detail::DevicePropagation propagation{};
    propagation.minimum_height_m = config.propagation.minimum_height_m;
    propagation.maximum_height_m = config.propagation.maximum_height_m;
    propagation.step_m = config.propagation.step_m;
    propagation.inverse_step_per_m = config.propagation.inverse_step_per_m;
    propagation.slope_refractivity_lower =
        config.propagation.slope_refractivity_lower;
    propagation.slope_integrated_refractivity_lower =
        config.propagation.slope_integrated_refractivity_lower;
    propagation.slope_refractivity_upper =
        config.propagation.slope_refractivity_upper;
    propagation.slope_integrated_refractivity_upper =
        config.propagation.slope_integrated_refractivity_upper;
    propagation.refractivity = config.propagation.refractivity.data();
    propagation.integrated_refractivity =
        config.propagation.integrated_refractivity.data();
    propagation.table_size = config.propagation.refractivity.size();
    propagation.zhs_subtrack_refinement = config.zhs_subtrack_refinement;
    radio_detail::DeviceRadioCounters counters{};
    for (auto const& record : records) {
      radio_detail::RadioTrackKinematics track{};
      if (!radio_detail::recordTrack<HostRadioAtomicOperations>(
              record, &track, &counters, config.track_diagnostics))
        continue;
      radio_detail::accumulateCoREAS<HostRadioAtomicOperations>(
          track, propagation, coreas_observer, coreas_waveforms, &counters);
      radio_detail::accumulateZHS<HostRadioAtomicOperations>(
          track, propagation, zhs_observer, zhs_waveforms, &counters);
    }
    gpu::radio::GpuRadioWaveforms output{};
    output.coreas.resize(1);
    output.zhs.resize(1);
    auto fill = [&](gpu::radio::RadioWaveform& waveform,
                    std::vector<long long> const& fixed, double inverse_scale) {
      waveform.x.resize(bins);
      waveform.y.resize(bins);
      waveform.z.resize(bins);
      for (std::size_t bin = 0; bin < bins; ++bin) {
        waveform.x[bin] = static_cast<double>(fixed[bin]) * inverse_scale;
        waveform.y[bin] =
            static_cast<double>(fixed[bins + bin]) * inverse_scale;
        waveform.z[bin] =
            static_cast<double>(fixed[2 * bins + bin]) * inverse_scale;
      }
    };
    fill(output.coreas.front(), coreas_fixed,
         coreas_observer.inverse_fixed_point_scale);
    fill(output.zhs.front(), zhs_fixed,
         zhs_observer.inverse_fixed_point_scale);
    return output;
  }
}

int main(int argc, char** argv) {
  try {
    auto checkpoint = [](char const* label) {
      if (std::getenv("C8_KOKKOS_TEST_TRACE") != nullptr)
        std::cerr << "[kokkos-test] " << label << std::endl;
    };
    std::size_t samples = 16384;
    if (argc == 2) samples = std::stoull(argv[1]);
    if (argc > 2 || samples == 0) {
      throw std::invalid_argument(
          "usage: testKokkosProposalNativeTable [SAMPLES]");
    }
    PROPOSAL::Logging::SetGlobalLoglevel(spdlog::level::critical);

    accelerator::em::KokkosEmBackend backend;
    CalculatorOwner photon{Code::Photon, 0.4_MeV};
    CalculatorOwner electron{Code::Electron, 0.4_MeV};
    CalculatorOwner positron{Code::Positron, 0.4_MeV};
    std::vector<proposal::NativeInteractionCalculatorView> interactions{
        photon.interactionView(), electron.interactionView(),
        positron.interactionView()};
    std::vector<proposal::NativeContinuousCalculatorView> continuous{
        electron.continuousView(), positron.continuousView()};
    auto const source = exportProposalNativeTables(interactions, continuous);
    auto const host_view = makeProposalNativeHostView(source);
    gpu::em::GpuEmConfig backend_config{};
    backend_config.physics_source = gpu::em::GpuPhysicsSource::ProposalNative;
    backend_config.em_transport_cut_MeV = 0.4;
    backend_config.muon_transport_cut_MeV = 300.;
    backend_config.random_seed = 0xdecafbad12345678ULL;
    backend_config.shower_id = 73;
    auto const earth_radius_m = 6371000.;
    backend_config.profile_projection.enabled = true;
    backend_config.profile_projection.accumulate_on_device = true;
    backend_config.profile_projection.axis_start_position_m[2] =
        earth_radius_m + 50000.;
    backend_config.profile_projection.axis_direction[2] = -1.;
    backend_config.profile_projection.axis_step_length_m = 1000.;
    backend_config.profile_projection.axis_grammage_g_per_cm2.resize(64);
    for (std::size_t i = 0;
         i < backend_config.profile_projection.axis_grammage_g_per_cm2.size();
         ++i)
      backend_config.profile_projection.axis_grammage_g_per_cm2[i] =
          static_cast<double>(i) * 16.;
    backend_config.profile_projection.output_bin_count = 128;
    backend_config.profile_projection.output_bin_width_g_per_cm2 = 8.;
    backend_config.profile_projection.fixed_point_weight_limit = 1.e12;
    backend_config.profile_projection.fixed_point_energy_limit_GeV = 1.e12;
    backend_config.radio.enabled = true;
    backend_config.radio.coreas_enabled = true;
    backend_config.radio.zhs_enabled = true;
    backend_config.radio.deterministic = true;
    backend_config.radio.track_diagnostics = true;
    backend_config.radio.fixed_point_field_limit_V_per_m = 1.;
    backend_config.radio.propagation.minimum_height_m = earth_radius_m;
    backend_config.radio.propagation.maximum_height_m = earth_radius_m + 1.e5;
    backend_config.radio.propagation.step_m = 1000.;
    backend_config.radio.propagation.inverse_step_per_m = 1. / 1000.;
    for (std::size_t index = 0; index <= 100; ++index) {
      auto const height_m = static_cast<double>(index) * 1000.;
      auto const exponential = std::exp(-height_m / 8000.);
      backend_config.radio.propagation.refractivity.push_back(
          3.e-4 * exponential);
      backend_config.radio.propagation.integrated_refractivity.push_back(
          3.e-4 * 8000. * (1. - exponential));
    }
    gpu::radio::RadioObserverSnapshot observer{};
    observer.position_m[0] = 500.;
    observer.position_m[2] = earth_radius_m + 100.;
    observer.start_time_s = 0.;
    observer.duration_s = 2.e-4;
    observer.sample_rate_Hz = 10.e6;
    observer.number_of_bins = 2001;
    backend_config.radio.coreas_observers.push_back(observer);
    backend_config.radio.zhs_observers.push_back(observer);
    auto const environment = gpu::em::makeCorsika7AtmosphereSnapshot(
        AtmosphereId::USStdBK, {0., 0., 0.}, 0, earth_radius_m + 100.);
    auto const auxiliary = loadOrCreateProposalNativeAux(interactions);
    auto moliere_cache_path = auxiliary.cache_file;
    if (!moliere_cache_path.empty())
      moliere_cache_path += ".moliere-initial-v1.c8cache";
    auto const host_moliere_table = gpu::em::loadOrMakeMoliereInterpolationTable(
        auxiliary.electron_moliere, moliere_cache_path);
    auto const host_moliere_view = gpu::em::makeMoliereInterpolationView(
        host_moliere_table.polynomials.data(),
        host_moliere_table.initial_guess_delta.data());
    backend.initialize(environment, source, auxiliary, backend_config);
    checkpoint("initialized");

    std::vector<ProposalNativeQuery> queries;
    queries.reserve(samples);
    for (std::size_t i = 0; i < samples; ++i) {
      auto const& column = source.dndx_columns[i % source.dndx_columns.size()];
      auto const& axis = column.spline.energy_axis;
      auto const fraction =
          (static_cast<double>((i * 104729U) % 1000003U) + 0.5) / 1000003.;
      auto const coordinate =
          fraction * static_cast<double>(axis.nodes - 1u);
      auto const energy = native_detail::axisBackTransform(axis, coordinate);
      queries.push_back({ProposalNativeQueryKind::Rate, column.pdg_id,
                         column.process_id, column.component_hash, energy, 0.});
    }

    auto const device_output =
        accelerator::em::testing::runKokkosProposalNativeQueries(source, queries);
    checkpoint("native queries");
    if (device_output.source_content_hash != source.content_hash ||
        device_output.device_bytes != proposalNativeTableBytes(source)) {
      throw std::runtime_error("Kokkos native-table identity changed on upload");
    }
    auto const& device_results = device_output.results;
    if (device_results.size() != queries.size()) {
      throw std::runtime_error("Kokkos native-table query count changed");
    }
    double maximum_relative_difference = 0.;
    for (std::size_t i = 0; i < queries.size(); ++i) {
      auto const expected = queryProposalNativeRate(
          host_view, queries[i].pdg_id, queries[i].process_id,
          queries[i].component_hash, queries[i].energy_MeV);
      auto const& actual = device_results[i];
      if (actual.status != expected.status ||
          !closeEnough(actual.value, expected.value)) {
        throw std::runtime_error(
            "Kokkos native-table rate differs from the CPU interpolant");
      }
      auto const scale = std::max(std::abs(expected.value), 1.e-300);
      maximum_relative_difference = std::max(
          maximum_relative_difference,
          std::abs(actual.value - expected.value) / scale);
    }

    auto const photon_total = std::find_if(
        source.total_rate_columns.begin(), source.total_rate_columns.end(),
        [](auto const& column) { return column.pdg_id == 22; });
    if (photon_total == source.total_rate_columns.end()) {
      throw std::runtime_error("native table has no photon total-rate column");
    }
    auto const selection_samples = std::min<std::size_t>(samples, 8192);
    std::vector<gpu::em::EmParticleState> particles(selection_samples);
    for (std::size_t i = 0; i < particles.size(); ++i) {
      auto const fraction =
          (static_cast<double>((i * 65537U) % 999983U) + 0.5) / 999983.;
      auto const coordinate = fraction * static_cast<double>(
          photon_total->spline.axis.nodes - 1u);
      particles[i].pid = 22;
      particles[i].medium_id = 0;
      particles[i].energy_GeV = native_detail::axisBackTransform(
                                    photon_total->spline.axis, coordinate) /
                                1000.;
      particles[i].direction[2] = -1.;
      particles[i].position_m[2] = earth_radius_m + 50000.;
      particles[i].weight = 1.;
      particles[i].history_id = i + 1;
      particles[i].step_id = i % 29;
    }
    auto selected = backend.selectInteractionsForValidation(particles);
    checkpoint("photon selection");
    for (auto const& particle : particles) {
      if (!backend.canTransport(particle))
        throw std::runtime_error(
            "Kokkos production gate rejected a validated photon");
    }
    gpu::em::tables::FlatRateTableView host_physics{};
    host_physics.proposal_native = host_view;
    host_physics.physics_source = 1u;
    host_physics.em_transport_cut_MeV = backend_config.em_transport_cut_MeV;
    host_physics.muon_transport_cut_MeV =
        backend_config.muon_transport_cut_MeV;
    std::vector<gpu::em::EmInteractionRecord> expected_interactions;
    std::vector<gpu::em::ProposalFallbackEvent> expected_fallbacks;
    for (std::size_t i = 0; i < particles.size(); ++i) {
      auto const outcome =
          accelerator::em::detail::selectDiscreteInteraction(
              host_physics, particles[i], i, backend_config.random_seed,
              backend_config.shower_id);
      if (outcome.fallback_flag != 0u) {
        expected_fallbacks.push_back(outcome.fallback);
      } else {
        expected_interactions.push_back(outcome.interaction);
      }
    }
    if (selected.interactions.size() != expected_interactions.size() ||
        selected.fallback_events.size() != expected_fallbacks.size()) {
      throw std::runtime_error(
          "Kokkos interaction selection changed stable output counts");
    }
    for (std::size_t i = 0; i < selected.interactions.size(); ++i) {
      auto const& actual = selected.interactions[i];
      auto const& expected = expected_interactions[i];
      if (actual.particle.history_id != expected.particle.history_id ||
          actual.status != expected.status ||
          actual.process_id != expected.process_id ||
          actual.component_hash != expected.component_hash ||
          actual.distance_uniform != expected.distance_uniform ||
          actual.process_uniform != expected.process_uniform ||
          actual.proposal_selection_uniform !=
              expected.proposal_selection_uniform ||
          !sameOrClose(actual.interaction_grammage_g_per_cm2,
                       expected.interaction_grammage_g_per_cm2) ||
          !sameOrClose(actual.energy_fraction, expected.energy_fraction)) {
        throw std::runtime_error(
            "Kokkos interaction selection differs from the CPU oracle");
      }
    }
    for (std::size_t i = 0; i < selected.fallback_events.size(); ++i) {
      auto const& actual = selected.fallback_events[i];
      auto const& expected = expected_fallbacks[i];
      if (actual.particle.history_id != expected.particle.history_id ||
          actual.reason != expected.reason ||
          actual.process_id != expected.process_id ||
          actual.component_hash != expected.component_hash) {
        throw std::runtime_error(
            "Kokkos interaction fallback order differs from the CPU oracle");
      }
    }

    auto transported = backend.transportPhotonsForValidation(
        selected.interactions);
    checkpoint("photon transport");
    std::vector<gpu::em::PhotonTransportRecord> expected_records;
    std::vector<gpu::em::ProposalFallbackEvent> expected_transport_fallbacks;
    for (auto const& interaction : selected.interactions) {
      auto const outcome = accelerator::em::detail::transportPhoton(
          environment, interaction);
      if (outcome.fallback_flag != 0u) {
        expected_transport_fallbacks.push_back(outcome.fallback);
      } else {
        expected_records.push_back(outcome.record);
      }
    }
    if (transported.records.size() != expected_records.size() ||
        transported.fallback_events.size() !=
            expected_transport_fallbacks.size()) {
      throw std::runtime_error(
          "Kokkos photon transport changed stable output counts");
    }
    for (std::size_t i = 0; i < transported.records.size(); ++i) {
      auto const& actual = transported.records[i];
      auto const& expected = expected_records[i];
      if (actual.start.history_id != expected.start.history_id ||
          actual.limit != expected.limit ||
          actual.start_layer_index != expected.start_layer_index ||
          actual.end_layer_index != expected.end_layer_index ||
          !closeEnough(actual.distance_m, expected.distance_m) ||
          !closeEnough(actual.traversed_grammage_g_per_cm2,
                       expected.traversed_grammage_g_per_cm2) ||
          !closeEnough(actual.end.position_m[2], expected.end.position_m[2]) ||
          !closeEnough(actual.end.time_s, expected.end.time_s)) {
        throw std::runtime_error(
            "Kokkos photon transport differs from the CPU oracle");
      }
    }

    std::vector<gpu::em::EmInteractionRecord> vertex_interactions;
    for (auto const& record : transported.records) {
      if (record.limit == gpu::em::PhotonTransportLimit::Interaction) {
        vertex_interactions.push_back(record.interaction);
      }
    }
    auto const first_secondary_history_id = 10000000ULL;
    auto final_states = backend.generatePhotonFinalStatesForValidation(
        vertex_interactions, first_secondary_history_id);
    checkpoint("photon final states");
    gpu::em::EmFinalStateBatchResult expected_final_states{};
    expected_final_states.input_interactions = vertex_interactions.size();
    std::uint64_t child_offset = 0;
    for (auto const& interaction : vertex_interactions) {
      auto const classification =
          accelerator::em::detail::classifyPhotonFinalState(
              host_physics, auxiliary.photon_pair_lpm,
              backend_config.thinning, interaction,
              backend_config.random_seed, backend_config.shower_id);
      auto const output =
          accelerator::em::detail::materializePhotonFinalState(
              interaction, classification, child_offset,
              first_secondary_history_id);
      if (output.error != 0) {
        throw std::runtime_error("CPU photon final-state oracle failed");
      }
      if (output.has_record != 0) {
        expected_final_states.final_state_records.push_back(output.record);
        for (std::uint32_t child = 0; child < output.secondary_count; ++child)
          expected_final_states.secondaries.push_back(output.secondaries[child]);
        child_offset += output.secondary_count;
        expected_final_states.gpu_interactions++;
        expected_final_states.photon_pair_interactions +=
            classification.photon_pair_flag;
        expected_final_states.compton_interactions +=
            classification.compton_flag;
        expected_final_states.photoelectric_interactions +=
            classification.photoelectric_flag;
      } else if (output.has_fallback != 0) {
        expected_final_states.fallback_events.push_back(output.fallback);
      } else if (output.has_continuation != 0) {
        expected_final_states.continuations.push_back(output.continuation);
      } else if (output.has_suppression != 0) {
        expected_final_states.lpm_suppressed.push_back(output.suppression);
      }
    }
    if (final_states.final_state_records.size() !=
            expected_final_states.final_state_records.size() ||
        final_states.secondaries.size() !=
            expected_final_states.secondaries.size() ||
        final_states.fallback_events.size() !=
            expected_final_states.fallback_events.size() ||
        final_states.continuations.size() !=
            expected_final_states.continuations.size() ||
        final_states.lpm_suppressed.size() !=
            expected_final_states.lpm_suppressed.size() ||
        final_states.photon_pair_interactions !=
            expected_final_states.photon_pair_interactions ||
        final_states.compton_interactions !=
            expected_final_states.compton_interactions ||
        final_states.photoelectric_interactions !=
            expected_final_states.photoelectric_interactions) {
      throw std::runtime_error(
          "Kokkos photon final-state stable output counts changed");
    }
    for (std::size_t i = 0; i < final_states.final_state_records.size(); ++i) {
      auto const& actual = final_states.final_state_records[i];
      auto const& expected = expected_final_states.final_state_records[i];
      if (actual.input_index != expected.input_index ||
          actual.parent_history_id != expected.parent_history_id ||
          actual.secondary_offset != expected.secondary_offset ||
          actual.secondary_count != expected.secondary_count ||
          actual.process_id != expected.process_id ||
          actual.split_uniform != expected.split_uniform ||
          actual.azimuth_uniform != expected.azimuth_uniform ||
          actual.lpm_uniform != expected.lpm_uniform ||
          !sameOrClose(actual.energy_split_fraction,
                       expected.energy_split_fraction) ||
          !sameOrClose(actual.lpm_survival_probability,
                       expected.lpm_survival_probability)) {
        throw std::runtime_error(
            "Kokkos photon final-state record differs from CPU oracle");
      }
    }
    for (std::size_t i = 0; i < final_states.secondaries.size(); ++i) {
      auto const& actual = final_states.secondaries[i];
      auto const& expected = expected_final_states.secondaries[i];
      if (actual.pid != expected.pid ||
          actual.history_id != expected.history_id ||
          actual.parent_history_id != expected.parent_history_id ||
          actual.generation != expected.generation ||
          actual.step_id != expected.step_id ||
          actual.weight != expected.weight ||
          !sameOrClose(actual.energy_GeV, expected.energy_GeV) ||
          !sameOrClose(actual.direction[0], expected.direction[0]) ||
          !sameOrClose(actual.direction[1], expected.direction[1]) ||
          !sameOrClose(actual.direction[2], expected.direction[2])) {
        throw std::runtime_error(
            "Kokkos photon secondary differs from CPU oracle");
      }
    }

    auto const electron_total = std::find_if(
        source.total_rate_columns.begin(), source.total_rate_columns.end(),
        [](auto const& column) { return column.pdg_id == 11; });
    if (electron_total == source.total_rate_columns.end()) {
      throw std::runtime_error("native table has no electron total-rate column");
    }
    auto const lepton_samples = std::min<std::size_t>(samples, 4096);
    std::vector<gpu::em::EmParticleState> leptons(lepton_samples);
    for (std::size_t i = 0; i < leptons.size(); ++i) {
      auto const fraction =
          (static_cast<double>((i * 32771U) % 524287U) + 0.5) / 524287.;
      auto const coordinate = fraction * static_cast<double>(
          electron_total->spline.axis.nodes - 1u);
      auto& particle = leptons[i];
      particle.pid = i % 2 == 0 ? 11 : -11;
      particle.medium_id = 0;
      particle.energy_GeV = native_detail::axisBackTransform(
                                electron_total->spline.axis, coordinate) /
                            1000.;
      particle.position_m[0] = 100. * static_cast<double>(i % 7);
      particle.position_m[2] = earth_radius_m + 50000.;
      particle.direction[0] = 0.1;
      particle.direction[2] = -std::sqrt(0.99);
      particle.weight = 1.;
      particle.history_id = 200000 + i;
      particle.step_id = i % 17;
    }
    auto lepton_selection = backend.selectInteractionsForValidation(leptons);
    auto lepton_transport = backend.transportLeptonsForValidation(
        lepton_selection.interactions);
    checkpoint("lepton transport");
    std::vector<gpu::em::LeptonTransportRecord> expected_lepton_records;
    std::vector<gpu::em::ProposalFallbackEvent> expected_lepton_fallbacks;
    for (auto const& interaction : lepton_selection.interactions) {
      gpu::em::LeptonTransportRecord record{};
      gpu::em::ProposalFallbackEvent fallback{};
      auto const state = accelerator::em::detail::transportLepton(
          host_physics, true, environment, interaction, record, fallback);
      auto const final_state =
          accelerator::em::detail::applyMoliereScatteringStage<4>(
              auxiliary.electron_moliere, auxiliary.muon_moliere,
              host_moliere_view, auxiliary.has_muon_moliere != 0,
              backend_config.random_seed, backend_config.shower_id, record,
              fallback, state);
      if (final_state == 1u) {
        expected_lepton_fallbacks.push_back(fallback);
      } else {
        expected_lepton_records.push_back(record);
      }
    }
    if (lepton_transport.records.size() != expected_lepton_records.size() ||
        lepton_transport.fallback_events.size() !=
            expected_lepton_fallbacks.size()) {
      throw std::runtime_error(
          "Kokkos lepton transport changed stable output counts");
    }
    for (std::size_t i = 0; i < lepton_transport.records.size(); ++i) {
      auto const& actual = lepton_transport.records[i];
      auto const& expected = expected_lepton_records[i];
      if (actual.start.history_id != expected.start.history_id ||
          actual.limit != expected.limit ||
          actual.start_layer_index != expected.start_layer_index ||
          actual.end_layer_index != expected.end_layer_index ||
          !sameOrClose(actual.distance_m, expected.distance_m) ||
          !sameOrClose(actual.traversed_grammage_g_per_cm2,
                       expected.traversed_grammage_g_per_cm2) ||
          !sameOrClose(actual.end.energy_GeV, expected.end.energy_GeV) ||
          std::abs(actual.continuous_deposited_energy_GeV -
                   expected.continuous_deposited_energy_GeV) >
              5.e-12 * std::max(1., expected.start.energy_GeV) ||
          !sameOrClose(actual.end.time_s, expected.end.time_s) ||
          actual.multiple_scattering_status !=
              expected.multiple_scattering_status ||
          actual.multiple_scattering_iterations !=
              expected.multiple_scattering_iterations ||
          !sameOrClose(actual.multiple_scattering_angle_rad,
                       expected.multiple_scattering_angle_rad) ||
          std::abs(actual.end.direction[0] - expected.end.direction[0]) >
              2.e-12 ||
          std::abs(actual.end.direction[1] - expected.end.direction[1]) >
              2.e-12 ||
          std::abs(actual.end.direction[2] - expected.end.direction[2]) >
              2.e-12) {
        std::ostringstream message;
        message << std::setprecision(17);
        message << "Kokkos lepton transport differs from the CPU oracle at "
                << i << ": history=" << actual.start.history_id << '/'
                << expected.start.history_id << ", limit="
                << static_cast<int>(actual.limit) << '/'
                << static_cast<int>(expected.limit) << ", distance="
                << actual.distance_m << '/' << expected.distance_m
                << ", grammage=" << actual.traversed_grammage_g_per_cm2
                << '/' << expected.traversed_grammage_g_per_cm2
                << ", energy=" << actual.end.energy_GeV << '/'
                << expected.end.energy_GeV << ", deposit="
                << actual.continuous_deposited_energy_GeV << '/'
                << expected.continuous_deposited_energy_GeV << ", time="
                << actual.end.time_s << '/' << expected.end.time_s
                << ", moliere_status="
                << actual.multiple_scattering_status << '/'
                << expected.multiple_scattering_status << ", iterations="
                << actual.multiple_scattering_iterations << '/'
                << expected.multiple_scattering_iterations << ", angle="
                << actual.multiple_scattering_angle_rad << '/'
                << expected.multiple_scattering_angle_rad << ", direction=["
                << actual.end.direction[0] << ',' << actual.end.direction[1]
                << ',' << actual.end.direction[2] << "]/["
                << expected.end.direction[0] << ',' << expected.end.direction[1]
                << ',' << expected.end.direction[2] << ']';
        throw std::runtime_error(message.str());
      }
    }
    std::vector<gpu::em::EmInteractionRecord> lepton_candidates;
    for (auto const& record : lepton_transport.records) {
      if (record.limit == gpu::em::LeptonTransportLimit::InteractionCandidate)
        lepton_candidates.push_back(record.interaction);
    }
    auto lepton_vertices =
        backend.selectLeptonVerticesForValidation(lepton_candidates);
    checkpoint("lepton vertices");
    gpu::em::LeptonVertexSelectionBatchResult expected_vertices{};
    expected_vertices.input_candidates = lepton_candidates.size();
    for (auto const& candidate : lepton_candidates) {
      auto const outcome = accelerator::em::detail::selectLeptonVertex(
          host_physics, candidate, backend_config.random_seed,
          backend_config.shower_id);
      if (outcome.interaction_flag != 0u)
        expected_vertices.interactions.push_back(outcome.record);
      else if (outcome.continuation_flag != 0u)
        expected_vertices.continuations.push_back(outcome.record);
      else if (outcome.fallback_flag != 0u)
        expected_vertices.fallback_events.push_back(outcome.fallback);
    }
    if (lepton_vertices.interactions.size() !=
            expected_vertices.interactions.size() ||
        lepton_vertices.continuations.size() !=
            expected_vertices.continuations.size() ||
        lepton_vertices.fallback_events.size() !=
            expected_vertices.fallback_events.size()) {
      throw std::runtime_error(
          "Kokkos lepton vertex selection changed stable output counts");
    }
    for (std::size_t i = 0; i < lepton_vertices.interactions.size(); ++i) {
      auto const& actual = lepton_vertices.interactions[i];
      auto const& expected = expected_vertices.interactions[i];
      if (actual.particle.history_id != expected.particle.history_id ||
          actual.status != expected.status ||
          actual.process_id != expected.process_id ||
          actual.component_hash != expected.component_hash ||
          actual.proposal_selection_uniform !=
              expected.proposal_selection_uniform ||
          !sameOrClose(actual.vertex_total_rate_cm2_per_g,
                       expected.vertex_total_rate_cm2_per_g) ||
          !sameOrClose(actual.energy_fraction, expected.energy_fraction)) {
        throw std::runtime_error(
            "Kokkos lepton vertex selection differs from the CPU oracle");
      }
    }

    auto constexpr first_lepton_secondary_history_id = 2000000ULL;
    auto lepton_final_states = backend.generateLeptonFinalStatesForValidation(
        lepton_vertices.interactions, first_lepton_secondary_history_id);
    checkpoint("lepton final states");
    gpu::em::BremsFinalStateBatchResult expected_lepton_final_states{};
    expected_lepton_final_states.input_interactions =
        lepton_vertices.interactions.size();
    std::uint64_t expected_child_offset = 0;
    for (auto const& interaction : lepton_vertices.interactions) {
      auto const classification =
          accelerator::em::detail::classifyLeptonFinalState(
              auxiliary.brems_lpm, backend_config.thinning, interaction,
              backend_config.random_seed, backend_config.shower_id);
      auto const output =
          accelerator::em::detail::materializeLeptonFinalState(
              interaction, classification, expected_child_offset,
              first_lepton_secondary_history_id,
              auxiliary.brems_lpm.lepton_mass_MeV / 1000.);
      if (output.error != 0u) {
        throw std::runtime_error(
            "CPU lepton final-state oracle failed to materialize");
      }
      if (output.has_record != 0u) {
        expected_lepton_final_states.final_state_records.push_back(
            output.record);
        for (std::uint32_t child = 0; child < output.secondary_count; ++child)
          expected_lepton_final_states.secondaries.push_back(
              output.secondaries[child]);
        expected_lepton_final_states.gpu_interactions++;
      } else if (output.has_fallback != 0u) {
        expected_lepton_final_states.fallback_events.push_back(
            output.fallback);
      } else if (output.has_continuation != 0u) {
        expected_lepton_final_states.continuations.push_back(
            output.continuation);
      } else if (output.has_suppression != 0u) {
        expected_lepton_final_states.lpm_suppressed.push_back(
            output.suppression);
      }
      expected_child_offset += classification.child_count;
      expected_lepton_final_states.brems_interactions +=
          classification.brems_flag;
      expected_lepton_final_states.annihilation_interactions +=
          classification.annihilation_flag;
      expected_lepton_final_states.ionization_interactions +=
          classification.ionization_flag;
      expected_lepton_final_states.electron_pair_interactions +=
          classification.electron_pair_flag;
      expected_lepton_final_states.brems_lpm_trials +=
          classification.brems_flag +
          classification.brems_suppression_flag;
      expected_lepton_final_states.brems_lpm_suppressions +=
          classification.brems_suppression_flag;
      expected_lepton_final_states.electron_pair_lpm_trials +=
          classification.electron_pair_flag +
          classification.electron_pair_suppression_flag;
      expected_lepton_final_states.electron_pair_lpm_suppressions +=
          classification.electron_pair_suppression_flag;
      expected_lepton_final_states.electron_pair_rejection_trials +=
          classification.electron_pair_rejection_trials;
      expected_lepton_final_states.electron_pair_zero_weight_samples +=
          classification.electron_pair_zero_weight_flag;
      expected_lepton_final_states.electron_pair_rejection_fallbacks +=
          classification.electron_pair_rejection_fallback_flag;
      expected_lepton_final_states.electron_pair_envelope_violations +=
          classification.electron_pair_envelope_violation_flag;
    }
    if (lepton_final_states.gpu_interactions !=
            expected_lepton_final_states.gpu_interactions ||
        lepton_final_states.final_state_records.size() !=
            expected_lepton_final_states.final_state_records.size() ||
        lepton_final_states.secondaries.size() !=
            expected_lepton_final_states.secondaries.size() ||
        lepton_final_states.fallback_events.size() !=
            expected_lepton_final_states.fallback_events.size() ||
        lepton_final_states.continuations.size() !=
            expected_lepton_final_states.continuations.size() ||
        lepton_final_states.lpm_suppressed.size() !=
            expected_lepton_final_states.lpm_suppressed.size() ||
        lepton_final_states.brems_interactions !=
            expected_lepton_final_states.brems_interactions ||
        lepton_final_states.annihilation_interactions !=
            expected_lepton_final_states.annihilation_interactions ||
        lepton_final_states.ionization_interactions !=
            expected_lepton_final_states.ionization_interactions ||
        lepton_final_states.electron_pair_interactions !=
            expected_lepton_final_states.electron_pair_interactions ||
        lepton_final_states.brems_lpm_trials !=
            expected_lepton_final_states.brems_lpm_trials ||
        lepton_final_states.brems_lpm_suppressions !=
            expected_lepton_final_states.brems_lpm_suppressions ||
        lepton_final_states.electron_pair_lpm_trials !=
            expected_lepton_final_states.electron_pair_lpm_trials ||
        lepton_final_states.electron_pair_lpm_suppressions !=
            expected_lepton_final_states.electron_pair_lpm_suppressions ||
        lepton_final_states.electron_pair_rejection_trials !=
            expected_lepton_final_states.electron_pair_rejection_trials ||
        lepton_final_states.electron_pair_zero_weight_samples !=
            expected_lepton_final_states.electron_pair_zero_weight_samples ||
        lepton_final_states.electron_pair_rejection_fallbacks !=
            expected_lepton_final_states.electron_pair_rejection_fallbacks ||
        lepton_final_states.electron_pair_envelope_violations !=
            expected_lepton_final_states.electron_pair_envelope_violations) {
      throw std::runtime_error(
          "Kokkos lepton final states changed stable output counts");
    }
    for (std::size_t i = 0;
         i < lepton_final_states.final_state_records.size(); ++i) {
      auto const& actual = lepton_final_states.final_state_records[i];
      auto const& expected =
          expected_lepton_final_states.final_state_records[i];
      if (actual.input_index != expected.input_index ||
          actual.parent_history_id != expected.parent_history_id ||
          actual.secondary_offset != expected.secondary_offset ||
          actual.secondary_count != expected.secondary_count ||
          actual.process_id != expected.process_id ||
          !sameOrClose(actual.photon_energy_fraction,
                       expected.photon_energy_fraction) ||
          actual.final_state_uniform != expected.final_state_uniform ||
          actual.azimuth_uniform != expected.azimuth_uniform ||
          actual.auxiliary_uniform != expected.auxiliary_uniform ||
          !sameOrClose(actual.lpm_survival_probability,
                       expected.lpm_survival_probability) ||
          actual.lpm_uniform != expected.lpm_uniform ||
          actual.final_state_draw_id != expected.final_state_draw_id ||
          actual.azimuth_draw_id != expected.azimuth_draw_id ||
          actual.auxiliary_draw_id != expected.auxiliary_draw_id ||
          actual.lpm_draw_id != expected.lpm_draw_id ||
          actual.thinning_status != expected.thinning_status ||
          actual.thinning_keep_mask != expected.thinning_keep_mask ||
          actual.thinning_first_uniform !=
              expected.thinning_first_uniform ||
          actual.thinning_second_uniform !=
              expected.thinning_second_uniform) {
        throw std::runtime_error(
            "Kokkos lepton final-state record differs from CPU oracle");
      }
    }
    for (std::size_t i = 0; i < lepton_final_states.secondaries.size(); ++i) {
      auto const& actual = lepton_final_states.secondaries[i];
      auto const& expected = expected_lepton_final_states.secondaries[i];
      if (actual.pid != expected.pid ||
          actual.history_id != expected.history_id ||
          actual.parent_history_id != expected.parent_history_id ||
          actual.generation != expected.generation ||
          actual.step_id != expected.step_id ||
          !sameOrClose(actual.energy_GeV, expected.energy_GeV) ||
          !sameOrClose(actual.weight, expected.weight) ||
          // At E >> m, the mathematically equivalent momentum subtraction
          // amplifies the last ULP of device cos/sqrt into an O(1e-8 rad)
          // transverse component.  This is below the existing native-CUDA
          // final-state angular acceptance and does not alter any decision.
          std::abs(actual.direction[0] - expected.direction[0]) > 5.e-8 ||
          std::abs(actual.direction[1] - expected.direction[1]) > 5.e-8 ||
          std::abs(actual.direction[2] - expected.direction[2]) > 5.e-8) {
        std::ostringstream message;
        message << std::setprecision(17)
                << "Kokkos lepton secondary differs from CPU oracle at "
                << i << ": pid=" << actual.pid << '/' << expected.pid
                << ", history=" << actual.history_id << '/'
                << expected.history_id << ", parent="
                << actual.parent_history_id << '/'
                << expected.parent_history_id << ", generation="
                << actual.generation << '/' << expected.generation
                << ", step=" << actual.step_id << '/' << expected.step_id
                << ", energy=" << actual.energy_GeV << '/'
                << expected.energy_GeV << ", weight=" << actual.weight << '/'
                << expected.weight << ", direction=[" << actual.direction[0]
                << ',' << actual.direction[1] << ',' << actual.direction[2]
                << "]/[" << expected.direction[0] << ','
                << expected.direction[1] << ',' << expected.direction[2]
                << ']';
        throw std::runtime_error(message.str());
      }
    }

    auto const resident_photon_count =
        std::min<std::size_t>(particles.size(), 256);
    std::vector<gpu::em::EmParticleState> resident_photons(
        particles.begin(), particles.begin() + resident_photon_count);
    auto resident_photon = backend.runPhotonWavefront(
        resident_photons, 30000000ULL, 4, 1);
    auto resident_photon_repeat = backend.runPhotonWavefront(
        resident_photons, 30000000ULL, 4, 1);
    checkpoint("resident photon");
    if (resident_photon.wavefronts == 0 ||
        resident_photon.input_particles != resident_photon_count ||
        !resident_photon.step_records.empty() ||
        !resident_photon.projected_step_records.empty() ||
        resident_photon.wavefronts != resident_photon_repeat.wavefronts ||
        resident_photon.transport_records !=
            resident_photon_repeat.transport_records ||
        resident_photon.interaction_vertices !=
            resident_photon_repeat.interaction_vertices ||
        resident_photon.final_state_records.size() !=
            resident_photon_repeat.final_state_records.size() ||
        resident_photon.electromagnetic_secondaries.size() !=
            resident_photon_repeat.electromagnetic_secondaries.size() ||
        resident_photon.remaining_photons.size() !=
            resident_photon_repeat.remaining_photons.size()) {
      throw std::runtime_error(
          "Kokkos resident photon wavefront is not repeatable");
    }
    for (std::size_t i = 0;
         i < resident_photon.remaining_photons.size(); ++i) {
      auto const& first = resident_photon.remaining_photons[i];
      auto const& second = resident_photon_repeat.remaining_photons[i];
      if (first.history_id != second.history_id || first.pid != second.pid ||
          first.step_id != second.step_id ||
          first.energy_GeV != second.energy_GeV)
        throw std::runtime_error(
            "Kokkos resident photon checkpoint changed on replay");
    }

    auto const resident_lepton_count =
        std::min<std::size_t>(leptons.size(), 128);
    std::vector<gpu::em::EmParticleState> resident_leptons(
        leptons.begin(), leptons.begin() + resident_lepton_count);
    auto resident_lepton = backend.runLeptonWavefront(
        resident_leptons, 40000000ULL, 3, 50000000ULL, 1);
    auto resident_lepton_repeat = backend.runLeptonWavefront(
        resident_leptons, 40000000ULL, 3, 50000000ULL, 1);
    checkpoint("resident lepton");
    if (resident_lepton.wavefronts == 0 ||
        resident_lepton.input_particles != resident_lepton_count ||
        !resident_lepton.step_records.empty() ||
        !resident_lepton.projected_step_records.empty() ||
        resident_lepton.wavefronts != resident_lepton_repeat.wavefronts ||
        resident_lepton.transport_records !=
            resident_lepton_repeat.transport_records ||
        resident_lepton.interaction_vertices !=
            resident_lepton_repeat.interaction_vertices ||
        resident_lepton.secondary_history_ids_used !=
            resident_lepton_repeat.secondary_history_ids_used ||
        resident_lepton.final_state_records.size() !=
            resident_lepton_repeat.final_state_records.size() ||
        resident_lepton.generated_photons.size() !=
            resident_lepton_repeat.generated_photons.size() ||
        resident_lepton.remaining_leptons.size() !=
            resident_lepton_repeat.remaining_leptons.size()) {
      throw std::runtime_error(
          "Kokkos resident lepton wavefront is not repeatable");
    }
    for (std::size_t i = 0;
         i < resident_lepton.remaining_leptons.size(); ++i) {
      auto const& first = resident_lepton.remaining_leptons[i];
      auto const& second = resident_lepton_repeat.remaining_leptons[i];
      if (first.history_id != second.history_id || first.pid != second.pid ||
          first.step_id != second.step_id ||
          first.energy_GeV != second.energy_GeV)
        throw std::runtime_error(
            "Kokkos resident lepton checkpoint changed on replay");
    }
    auto const profile = backend.downloadProfile();
    checkpoint("profile download");
    auto const expected_profile_steps =
        resident_photon.transport_records +
        resident_photon_repeat.transport_records +
        resident_lepton.transport_records +
        resident_lepton_repeat.transport_records;
    if (profile.steps != expected_profile_steps ||
        profile.photons.size() !=
            backend_config.profile_projection.output_bin_count ||
        profile.fixed_point_overflows != 0 || profile.invalid_records != 0 ||
        !(profile.weighted_deposited_energy_GeV >= 0.)) {
      throw std::runtime_error(
          "Kokkos resident deterministic profile accumulation failed");
    }

    auto const radio_oracle =
        projectRadioOnHost(backend_config.radio, lepton_transport.records);
    auto const radio =
        backend.projectRadioForValidation(lepton_transport.records);
    checkpoint("radio projection");
    auto compare_waveform = [](gpu::radio::RadioWaveform const& actual,
                               gpu::radio::RadioWaveform const& expected) {
      return actual.x == expected.x && actual.y == expected.y &&
             actual.z == expected.z;
    };
    if (radio.coreas.size() != 1 || radio.zhs.size() != 1 ||
        !compare_waveform(radio.coreas.front(), radio_oracle.coreas.front()) ||
        !compare_waveform(radio.zhs.front(), radio_oracle.zhs.front()) ||
        backend.statistics().radio.lepton_tracks == 0 ||
        backend.statistics().radio.coreas_contributions == 0 ||
        backend.statistics().radio.zhs_contributions == 0 ||
        backend.statistics().radio.fixed_point_overflows != 0) {
      throw std::runtime_error(
          "Kokkos CoREAS/ZHS projection differs from the host oracle");
    }

    std::cout << "Kokkos proposal-native table passed " << queries.size()
              << " rate queries and " << selection_samples
              << " interaction selections plus " << transported.records.size()
              << " photon transports on " << backend.runtimeInfo().backend
              << "; selected=" << selected.interactions.size()
              << ", selection_fallbacks=" << selected.fallback_events.size()
              << ", transport_fallbacks="
              << transported.fallback_events.size()
              << ", photon_final_states="
              << final_states.final_state_records.size()
              << ", photon_secondaries=" << final_states.secondaries.size()
              << ", lepton_transports=" << lepton_transport.records.size()
              << ", lepton_fallbacks="
              << lepton_transport.fallback_events.size()
              << ", lepton_vertices=" << lepton_vertices.interactions.size()
              << ", lepton_continuations="
              << lepton_vertices.continuations.size()
              << ", lepton_final_states="
              << lepton_final_states.final_state_records.size()
              << ", lepton_secondaries="
              << lepton_final_states.secondaries.size()
              << ", resident_photon_wavefronts="
              << resident_photon.wavefronts
              << ", resident_lepton_wavefronts="
              << resident_lepton.wavefronts
              << "; bytes=" << device_output.device_bytes
              << ", max_relative_difference="
              << maximum_relative_difference << '\n';
  } catch (std::exception const& error) {
    std::cerr << "Kokkos proposal-native table failed: " << error.what()
              << '\n';
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}

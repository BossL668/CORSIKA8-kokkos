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
#include <memory>
#include <stdexcept>
#include <vector>

#include <corsika/accelerator/em/KokkosRuntime.hpp>
#include <corsika/accelerator/em/KokkosEmBackend.hpp>
#include <corsika/accelerator/em/detail/InteractionSelection.hpp>
#include <corsika/accelerator/em/detail/PhotonFinalStateStep.hpp>
#include <corsika/accelerator/em/detail/PhotonTransportStep.hpp>
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
}

int main(int argc, char** argv) {
  try {
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
    auto const environment = gpu::em::makeCorsika7AtmosphereSnapshot(
        AtmosphereId::USStdBK, {0., 0., 0.}, 0, earth_radius_m + 100.);
    auto const auxiliary = loadOrCreateProposalNativeAux(interactions);
    backend.initialize(environment, source, auxiliary, backend_config);

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

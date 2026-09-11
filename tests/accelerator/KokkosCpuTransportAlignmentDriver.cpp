/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */
#include "KokkosCpuTransportAlignmentDriver.hpp"

#include <Kokkos_Core.hpp>
#include <corsika/accelerator/em/detail/InteractionSelection.hpp>
#include <corsika/accelerator/em/detail/LeptonTransportStep.hpp>
#include <corsika/accelerator/em/detail/LeptonFinalStateStep.hpp>
#include <corsika/accelerator/em/detail/PhotonFinalStateStep.hpp>
#include <corsika/accelerator/em/detail/MoliereStep.hpp>
#include <corsika/accelerator/em/detail/LeptonVertexSelection.hpp>
#include <corsika/accelerator/em/kokkos/KokkosProposalNativeTable.hpp>

namespace corsika::accelerator::em::testing {
  std::vector<CpuTransportAlignmentOutput> runCpuTransportAlignmentOnDevice(
      gpu::em::tables::ProposalNativeTableSet const& source,
      std::vector<CpuTransportAlignmentInput> const& input) {
    using ExecutionSpace = Kokkos::DefaultExecutionSpace;
    kokkos_detail::KokkosProposalNativeTable<ExecutionSpace> table;
    table.initialize(source);
    auto const native = table.deviceView();
    Kokkos::View<CpuTransportAlignmentInput*, ExecutionSpace> inputs(
        "cpu_transport_alignment_inputs", input.size());
    Kokkos::View<CpuTransportAlignmentOutput*, ExecutionSpace> outputs(
        "cpu_transport_alignment_outputs", input.size());
    auto staging = Kokkos::create_mirror_view(inputs);
    for (std::size_t i = 0; i < input.size(); ++i) staging(i) = input[i];
    Kokkos::deep_copy(inputs, staging);
    Kokkos::parallel_for(
        "cpu_transport_alignment", Kokkos::RangePolicy<ExecutionSpace>(0, input.size()),
        KOKKOS_LAMBDA(std::size_t const i) {
          auto const& item = inputs(i);
          auto& result = outputs(i);
          gpu::em::tables::NativePhysicsView physics{};
          physics.proposal_native = native;
          physics.em_transport_cut_MeV = item.em_cut_MeV;
          physics.muon_transport_cut_MeV = item.muon_cut_MeV;
          result.preparation = detail::prepareLeptonContinuousStep(
              physics, item.interaction);
          result.selection_status = detail::selectDiscreteInteraction(
              physics, item.interaction.particle, i, 1234, 5).interaction.status;
          if (item.propagate) {
            result.transport_status = detail::transportLepton(
                physics, false, item.environment, item.interaction,
                result.transport, result.fallback, item.external);
          }
        });
    auto downloaded = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), outputs);
    std::vector<CpuTransportAlignmentOutput> result(input.size());
    for (std::size_t i = 0; i < result.size(); ++i) result[i] = downloaded(i);
    return result;
  }

  std::vector<CpuSecondaryAlignmentOutput> runCpuSecondaryAlignmentOnDevice(
      std::vector<CpuSecondaryAlignmentInput> const& input) {
    using ExecutionSpace = Kokkos::DefaultExecutionSpace;
    Kokkos::View<CpuSecondaryAlignmentInput*, ExecutionSpace> inputs(
        "cpu_secondary_alignment_inputs", input.size());
    Kokkos::View<CpuSecondaryAlignmentOutput*, ExecutionSpace> outputs(
        "cpu_secondary_alignment_outputs", input.size());
    auto staging = Kokkos::create_mirror_view(inputs);
    for (std::size_t i = 0; i < input.size(); ++i) staging(i) = input[i];
    Kokkos::deep_copy(inputs, staging);
    Kokkos::parallel_for("cpu_secondary_alignment",
        Kokkos::RangePolicy<ExecutionSpace>(0, input.size()), KOKKOS_LAMBDA(std::size_t i) {
          auto const& item = inputs(i);
          auto& result = outputs(i);
          if (item.interaction.particle.pid == 22) {
            detail::PhotonFinalStateClassification classification{};
            classification.record_flag = 1;
            auto& sample = classification.parameters;
            sample.process_id = item.interaction.process_id;
            sample.split_fraction = item.split;
            sample.azimuth_uniform = 0.31;
            sample.electron_polar_uniform = 0.4;
            sample.positron_polar_uniform = 0.6;
            sample.thinning_keep_mask = item.keep_mask;
            sample.thinning_first_weight = 3.;
            sample.thinning_second_weight = 5.;
            auto output = detail::materializePhotonFinalState(
                item.interaction, classification, 0, 10);
            result.count = output.secondary_count;
            result.error = output.error;
            result.weighted_mass_correction_GeV = output.record.weighted_mass_convention_correction_GeV;
            for (std::size_t j = 0; j < output.secondary_count; ++j)
              result.children[j] = output.secondaries[j];
          } else {
            detail::LeptonFinalStateClassification classification{};
            classification.record_flag = 1;
            auto& sample = classification.parameters;
            sample.process_id = item.interaction.process_id;
            sample.energy_split_fraction = item.split;
            sample.azimuth_uniform = 0.31;
            sample.thinning_keep_mask = item.keep_mask;
            sample.thinning_first_weight = 3.;
            sample.thinning_second_weight = 5.;
            auto output = detail::materializeLeptonFinalState(
                item.interaction, classification, 0, 10, gpu::em::ElectronMassGeV);
            result.count = output.secondary_count;
            result.error = output.error;
            result.weighted_mass_correction_GeV = output.record.weighted_mass_convention_correction_GeV;
            for (std::size_t j = 0; j < output.secondary_count; ++j)
              result.children[j] = output.secondaries[j];
          }
        });
    auto downloaded = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), outputs);
    std::vector<CpuSecondaryAlignmentOutput> result(input.size());
    for (std::size_t i = 0; i < result.size(); ++i) result[i] = downloaded(i);
    return result;
  }

  std::vector<CpuRandomDomainAlignmentOutput> runCpuRandomDomainAlignmentOnDevice(
      gpu::em::tables::ProposalNativeTableSet const& source, std::size_t samples) {
    using ExecutionSpace = Kokkos::DefaultExecutionSpace;
    kokkos_detail::KokkosProposalNativeTable<ExecutionSpace> table;
    table.initialize(source);
    auto const native = table.deviceView();
    Kokkos::View<CpuRandomDomainAlignmentOutput*, ExecutionSpace> outputs(
        "cpu_random_domain_alignment", samples);
    Kokkos::parallel_for("cpu_random_domain_alignment",
        Kokkos::RangePolicy<ExecutionSpace>(0, samples), KOKKOS_LAMBDA(std::size_t i) {
          using namespace gpu::em;
          auto& result = outputs(i);
          tables::NativePhysicsView physics{};
          physics.proposal_native = native;
          physics.em_transport_cut_MeV = 0.5;
          physics.muon_transport_cut_MeV = 300.;
          LeptonTransportRecord step{};
          step.start.pid = 11;
          step.start.energy_GeV = 10.;
          step.start.direction[2] = 1.;
          step.start.history_id = i + 123;
          step.start.step_id = 7;
          step.end = step.start;
          step.interaction.particle = step.end;
          step.interaction.status = EmInteractionStatus::RequiresReselection;
          step.interaction.process_uniform = 0.5;
          step.interaction.total_rate_cm2_per_g =
              tables::queryTotalRate(physics, 11, 10000.).value;
          // Exercise the real function which records the three Moliere draws.
          // An empty snapshot stops before its physics calculation; this is a
          // RNG-provenance test, not a Moliere scattering-accuracy oracle.
          ProposalFallbackEvent discarded{};
          detail::applyMoliereScattering<MaxMoliereComponents>(
              MoliereSnapshot{}, MoliereInterpolationView{}, true, 7319, 3,
              step, discarded);
          auto const selected = detail::selectLeptonVertex(
              physics, step.interaction, 7319, 3);
          result.first_moliere_uniform = step.multiple_scattering_first_uniform;
          result.proposal_selection_uniform = selected.fallback_flag
              ? selected.fallback.selection_uniform : selected.record.proposal_selection_uniform;
          result.selection_observed = result.proposal_selection_uniform > 0. &&
                                      result.proposal_selection_uniform < 1.;
          RandomNumberKey const legacy{7319, 3, step.start.history_id,
                                       step.start.step_id, 0x454d0003U, 0};
          result.legacy_proposal_selection_uniform = uniformOpen01(legacy);
        });
    auto downloaded = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), outputs);
    std::vector<CpuRandomDomainAlignmentOutput> result(samples);
    for (std::size_t i = 0; i < result.size(); ++i) result[i] = downloaded(i);
    return result;
  }
}

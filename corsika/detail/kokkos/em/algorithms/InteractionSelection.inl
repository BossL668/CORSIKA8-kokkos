/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

namespace corsika::kokkos {

  CORSIKA_HOST_DEVICE inline InteractionSelectionOutcome select_discrete_interaction(
      InteractionPhysicsView const& physics, ParticleState const& particle,
      std::uint64_t const input_index, std::uint64_t const random_seed,
      std::uint64_t const shower_id) {

    InteractionSelectionOutcome output{};
    auto& record = output.interaction_;
    record.particle_ = particle;
    record.inputIndex_ = input_index;
    record.distanceDrawId_ = InteractionDistanceDrawId;
    record.processDrawId_ = InteractionColumnDrawId;
    record.lossDrawId_ = InteractionLossDrawId;
    record.processRandomProcessId_ = InteractionColumnRandomProcessId;
    auto const infinity = HUGE_VAL;
    auto const energy_MeV = particle.energyGeV_ * 1000.;

    if (is_native_muon_pair_cut(particle)) {
      record.status_ = InteractionStatus::ParticleCut;
      record.interactionGrammageGPerCm2_ = infinity;
      return output; // no rate, decay, or momentum evaluation below rest mass
    }

    if (exceeds_particle_cut_time(particle.timeS_)) {
      record.status_ = InteractionStatus::ParticleCut;
      record.interactionGrammageGPerCm2_ = infinity;
      return output;
    }

    auto const charged_lepton = is_charged_lepton_pid(particle.pid_);
    auto const photon = particle.pid_ == static_cast<std::int32_t>(EmPid::Photon);
    if (photon && energy_MeV < physics.getContinuousView().getEmTransportCutMeV()) {
      record.status_ = InteractionStatus::ParticleCut;
      record.interactionGrammageGPerCm2_ = infinity;
      return output;
    }
    if (charged_lepton) {
      auto const mass = query_continuous_mass(physics.getContinuousView(), particle.pid_);
      auto const transport_mass =
          query_continuous_transport_mass(physics.getContinuousView(), particle.pid_);
      if (mass.status_ == TableLookupStatus::Success) {
        record.particleMassGeV_ = mass.value_ / 1000.;
        if (is_muon_pid(particle.pid_) &&
            transport_mass.status_ == TableLookupStatus::Success &&
            particle.energyGeV_ > transport_mass.value_ / 1000.) {
          RandomNumberKey const decay_key{random_seed,
                                          shower_id,
                                          particle.historyId_,
                                          particle.stepId_,
                                          MuonDecayRandomProcessId,
                                          MuonDecayDrawId};
          record.decayUniform_ = uniform_open01(decay_key);
          record.decayDrawId_ = MuonDecayDrawId;
          auto const momentum_GeV =
              ::sqrt((particle.energyGeV_ - transport_mass.value_ / 1000.) *
                     (particle.energyGeV_ + transport_mass.value_ / 1000.));
          record.decayDistanceM_ = -::log(record.decayUniform_) *
                                   MuonDecaySpeedOfLightMPerS * MuonMeanLifetimeS *
                                   momentum_GeV / (transport_mass.value_ / 1000.);
        }
      }
      auto const minimum_energy =
          query_continuous_minimum_energy(physics.getContinuousView(), particle.pid_);
      auto const transport_cut =
          query_continuous_transport_cut(physics.getContinuousView(), particle.pid_);
      auto const below_cut = transport_mass.status_ == TableLookupStatus::Success &&
                             transport_cut.status_ == TableLookupStatus::Success &&
                             energy_MeV - transport_mass.value_ < transport_cut.value_;
      if (below_cut || (minimum_energy.status_ == TableLookupStatus::Success &&
                        energy_MeV <= minimum_energy.value_)) {
        record.status_ = InteractionStatus::ParticleCut;
        record.interactionGrammageGPerCm2_ = infinity;
        return output;
      }
    }

    TableQuery const total_query{
        TableQueryKind::TotalRate, particle.pid_, 0, 0, energy_MeV, 0.};
    auto const total = execute_table_query(physics.getContinuousView(), total_query);
    if (total.status_ != TableLookupStatus::Success) {
      output.fallback_ =
          make_table_fallback_event(particle, total_query, total, 0, input_index);
      output.fallbackFlag_ = 1;
      return output;
    }
    record.totalRateCm2PerG_ = total.value_;
    record.vertexTotalRateCm2PerG_ = total.value_;
    if (!(total.value_ > 0.)) {
      record.status_ = InteractionStatus::NoDiscreteInteraction;
      record.interactionGrammageGPerCm2_ = infinity;
      return output;
    }

    RandomNumberKey const distance_key{random_seed,
                                       shower_id,
                                       particle.historyId_,
                                       particle.stepId_,
                                       InteractionDistanceRandomProcessId,
                                       InteractionDistanceDrawId};
    record.distanceUniform_ = uniform_open01(distance_key);
    record.interactionGrammageGPerCm2_ = -::log(record.distanceUniform_) / total.value_;
    RandomNumberKey const column_key{random_seed,
                                     shower_id,
                                     particle.historyId_,
                                     particle.stepId_,
                                     InteractionColumnRandomProcessId,
                                     InteractionColumnDrawId};
    record.processUniform_ = uniform_open01(column_key);

    // The process is selected at post-continuous-loss energy for charged
    // particles, exactly as in the scalar and native-CUDA paths.
    if (charged_lepton) {
      record.status_ = InteractionStatus::DistanceSampled;
      return output;
    }

    constexpr bool proposal_native = true;
    RateColumnSelectionResult selection{};

    RandomNumberKey const proposal_key{random_seed,
                                       shower_id,
                                       particle.historyId_,
                                       particle.stepId_,
                                       ProposalSelectionRandomProcessId,
                                       ProposalSelectionDrawId};
    record.proposalSelectionUniform_ = uniform_open01(proposal_key);
    record.proposalSelectionRandomProcessId_ = ProposalSelectionRandomProcessId;
    record.proposalSelectionDrawId_ = ProposalSelectionDrawId;
    selection = select_rate_column_by_uniform(physics.getContinuousView().getNativeView(),
                                              particle.pid_, energy_MeV,
                                              record.proposalSelectionUniform_);

    if (selection.status_ != TableLookupStatus::Success) {
      auto const result = TableQueryResult{selection.status_, 0, 0.};
      output.fallback_ =
          make_table_fallback_event(particle, total_query, result, 0, input_index);

      output.fallback_.selectionUniform_ = record.proposalSelectionUniform_;
      output.fallback_.outerAcceptanceUniform_ = record.processUniform_;
      output.fallback_.randomProcessId_ = ProposalSelectionRandomProcessId;
      output.fallback_.randomDrawId_ = ProposalSelectionDrawId;
      output.fallback_.outerAcceptanceRandomProcessId_ = InteractionColumnRandomProcessId;
      output.fallback_.outerAcceptanceDrawId_ = InteractionColumnDrawId;

      output.fallbackFlag_ = 1;
      return output;
    }
    if (selection.selected_ == 0) {
      output.fallback_.particle_ = particle;
      output.fallback_.inputIndex_ = input_index;
      output.fallback_.reason_ = FallbackReason::ZeroTotalRate;
      output.fallbackFlag_ = 1;
      return output;
    }

    record.processId_ = selection.processId_;
    record.componentHash_ = selection.componentHash_;

    record.lossQuantile_ = selection.residualQuantile_;
    if (proposal_native_selection_requires_replay(particle.pid_, record.processId_,
                                                  record.lossQuantile_)) {
      // Selection is not a vertex. The photon must first fly the sampled
      // grammage (or stop at an earlier boundary/cut). Final-state
      // classification completes this request only at the transported vertex.
      record.deferredPhotonFallbackReason_ =
          static_cast<std::int32_t>(FallbackReason::NativeSelectionReplay);
      return output;
    }

    TableQuery const loss_query{
        TableQueryKind::LossFraction, particle.pid_, record.processId_,
        record.componentHash_,        energy_MeV,    record.lossQuantile_};
    auto const loss = execute_table_query(physics.getContinuousView(), loss_query);

    output.nativeNewtonIterations_ = native_newton_iterations(loss);
    output.nativeBisectionIterations_ = native_bisection_iterations(loss);
    output.nativeInverseFailures_ = loss.status_ == TableLookupStatus::Success ? 0u : 1u;

    if (loss.status_ != TableLookupStatus::Success) {
      auto const reason = proposal_fallback_reason(loss.status_);
      if (proposal_fallback_requires_selected_loss(reason)) {
        record.deferredPhotonFallbackReason_ = static_cast<std::int32_t>(reason);
        record.deferredPhotonTableStatus_ = static_cast<std::int32_t>(loss.status_);
        record.energyFraction_ = loss.value_; // diagnostic only; CPU completes v
        return output;
      }
      output.fallback_ = make_table_fallback_event(
          particle, loss_query, loss,
          proposal_native ? ProposalSelectionDrawId : InteractionLossDrawId, input_index);
      output.fallback_.lossQuantile_ = record.lossQuantile_;
      output.fallback_.selectionUniform_ =
          proposal_native ? record.proposalSelectionUniform_ : record.processUniform_;
      output.fallback_.outerAcceptanceUniform_ = record.processUniform_;
      output.fallback_.randomProcessId_ =
          proposal_native ? ProposalSelectionRandomProcessId : 0u;
      output.fallback_.outerAcceptanceRandomProcessId_ = record.processRandomProcessId_;
      output.fallback_.outerAcceptanceDrawId_ = record.processDrawId_;
      output.fallbackFlag_ = 1;
      return output;
    }

    record.status_ = InteractionStatus::Selected;
    record.energyFraction_ = loss.value_;
    return output;
  }

} // namespace corsika::kokkos

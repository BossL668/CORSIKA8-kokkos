/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

namespace corsika::kokkos {

  namespace lepton_vertex_detail {
    CORSIKA_HOST_DEVICE inline FallbackRequest vertex_fallback(
        InteractionRecord const& candidate, FallbackReason const reason) {
      auto event = make_process_fallback_event(candidate, reason);
      event.inputIndex_ = candidate.inputIndex_;
      return event;
    }

    CORSIKA_HOST_DEVICE inline FallbackRequest table_fallback(
        InteractionRecord const& candidate, TableQuery const& query,
        TableQueryResult const& result, std::uint64_t const draw_id,
        double const loss_quantile = 0.) {
      auto event = make_table_fallback_event(candidate.particle_, query, result, draw_id,
                                             candidate.inputIndex_);
      event.interactionVertexReached_ = candidate.interactionVertexReached_;
      auto const native_selection = candidate.proposalSelectionRandomProcessId_ != 0u;
      event.selectionUniform_ = native_selection ? candidate.proposalSelectionUniform_
                                                 : candidate.processUniform_;
      event.lossQuantile_ = loss_quantile;
      event.outerAcceptanceUniform_ = candidate.processUniform_;
      event.outerAcceptanceRandomProcessId_ = candidate.processRandomProcessId_;
      event.outerAcceptanceDrawId_ = candidate.processDrawId_;
      event.randomProcessId_ =
          native_selection ? candidate.proposalSelectionRandomProcessId_ : 0u;
      if (native_selection) event.randomDrawId_ = candidate.proposalSelectionDrawId_;
      return event;
    }
  } // namespace lepton_vertex_detail

  CORSIKA_HOST_DEVICE inline LeptonVertexSelectionOutcome select_lepton_vertex(
      InteractionPhysicsView const& table, InteractionRecord const& candidate,
      std::uint64_t const random_seed, std::uint64_t const shower_id) {

    LeptonVertexSelectionOutcome output{};
    auto record = candidate;
    auto const& particle = record.particle_;
    if (record.status_ == InteractionStatus::AtRestAnnihilation &&
        particle.pid_ == static_cast<std::int32_t>(EmPid::Positron)) {
      output.record_ = record;
      output.interactionFlag_ = 1;
      return output;
    }
    if (!is_charged_lepton_pid(particle.pid_)) {
      output.fallback_ = lepton_vertex_detail::vertex_fallback(
          record, FallbackReason::UnsupportedParticle);
      output.fallbackFlag_ = 1;
      return output;
    }
    if (record.status_ != InteractionStatus::RequiresReselection ||
        !lepton_continuous_detail::finite(record.totalRateCm2PerG_) ||
        !(record.totalRateCm2PerG_ > 0.) ||
        !lepton_continuous_detail::finite(record.processUniform_) ||
        !(record.processUniform_ > 0.) || !(record.processUniform_ < 1.)) {
      output.fallback_ = lepton_vertex_detail::vertex_fallback(
          record, FallbackReason::InvalidFinalState);
      output.fallbackFlag_ = 1;
      return output;
    }

    auto const energy_MeV = particle.energyGeV_ * 1000.;
    auto const outer_threshold = record.processUniform_ * record.totalRateCm2PerG_;
    constexpr bool proposal_native = true;
    RateColumnSelectionResult selection{};

    auto const vertex_total = query_discrete_total_rate(
        table.getContinuousView().getNativeView(), particle.pid_, energy_MeV);
    if (vertex_total.status_ != TableLookupStatus::Success) {
      TableQuery const failed_total_query{
          TableQueryKind::TotalRate, particle.pid_, 0, 0, energy_MeV, 0.};
      output.fallback_ = lepton_vertex_detail::table_fallback(
          record, failed_total_query, vertex_total, record.processDrawId_);
      output.fallbackFlag_ = 1;
      return output;
    }
    record.vertexTotalRateCm2PerG_ = vertex_total.value_;
    if (!(vertex_total.value_ > 0.) || outer_threshold >= vertex_total.value_) {
      if (record.particle_.stepId_ == 0xffffffffffffffffULL) {
        output.fallback_ = lepton_vertex_detail::vertex_fallback(
            record, FallbackReason::InvalidFinalState);
        output.fallbackFlag_ = 1;
        return output;
      }
      record.status_ = InteractionStatus::NoDiscreteInteraction;
      record.processId_ = 0;
      record.componentHash_ = 0;
      record.energyFraction_ = 0.;
      record.lossQuantile_ = 0.;
      record.particle_.stepId_++;
      output.record_ = record;
      output.continuationFlag_ = 1;
      return output;
    }
    RandomNumberKey const proposal_key{random_seed,
                                       shower_id,
                                       particle.historyId_,
                                       particle.stepId_,
                                       ProposalSelectionRandomProcessId,
                                       ProposalSelectionDrawId};
    record.proposalSelectionUniform_ = uniform_open01(proposal_key);
    record.proposalSelectionRandomProcessId_ = ProposalSelectionRandomProcessId;
    record.proposalSelectionDrawId_ = ProposalSelectionDrawId;
    selection = select_rate_column_by_uniform(table.getContinuousView().getNativeView(),
                                              particle.pid_, energy_MeV,
                                              record.proposalSelectionUniform_);

    if (selection.status_ != TableLookupStatus::Success) {
      record.processId_ = selection.processId_;
      record.componentHash_ = selection.componentHash_;
      TableQuery const failed_column_query{TableQueryKind::Rate, particle.pid_,
                                           selection.processId_, selection.componentHash_,
                                           energy_MeV,           0.};
      auto const failure = TableQueryResult{selection.status_, 0, 0.};
      auto fallback = lepton_vertex_detail::table_fallback(
          record, failed_column_query, failure, record.processDrawId_);
      // Preserve the complete threshold inputs.  A non-finite selection
      // status cannot be diagnosed from the post-transport energy alone.
      fallback.diagnosticValue0_ = energy_MeV;
      fallback.diagnosticValue1_ = record.totalRateCm2PerG_;
      fallback.diagnosticValue2_ =
          proposal_native ? record.proposalSelectionUniform_ : record.processUniform_;
      output.fallback_ = fallback;
      output.fallbackFlag_ = 1;
      return output;
    }
    if (!proposal_native) record.vertexTotalRateCm2PerG_ = selection.totalRate_;
    if (!proposal_native &&
        (!(selection.totalRate_ > 0.) || outer_threshold >= selection.totalRate_)) {
      if (record.particle_.stepId_ == 0xffffffffffffffffULL) {
        output.fallback_ = lepton_vertex_detail::vertex_fallback(
            record, FallbackReason::InvalidFinalState);
        output.fallbackFlag_ = 1;
        return output;
      }
      record.status_ = InteractionStatus::NoDiscreteInteraction;
      record.processId_ = 0;
      record.componentHash_ = 0;
      record.energyFraction_ = 0.;
      record.lossQuantile_ = 0.;
      record.particle_.stepId_++;
      output.record_ = record;
      output.continuationFlag_ = 1;
      return output;
    }
    if (selection.selected_ == 0) {
      output.fallback_ =
          lepton_vertex_detail::vertex_fallback(record, FallbackReason::ZeroTotalRate);
      output.fallbackFlag_ = 1;
      return output;
    }

    record.processId_ = selection.processId_;
    record.componentHash_ = selection.componentHash_;

    record.lossQuantile_ = selection.residualQuantile_;
    record.lossDrawId_ = ProposalSelectionDrawId;
    if (proposal_native_selection_requires_replay(particle.pid_, record.processId_,
                                                  record.lossQuantile_)) {
      output.fallback_ = lepton_vertex_detail::vertex_fallback(
          record, FallbackReason::NativeSelectionReplay);
      output.fallbackFlag_ = 1;
      return output;
    }

    TableQuery const loss_query{
        TableQueryKind::LossFraction, particle.pid_, record.processId_,
        record.componentHash_,        energy_MeV,    record.lossQuantile_};
    auto const loss = execute_table_query(table.getContinuousView(), loss_query);
    output.nativeNewtonIterations_ = native_newton_iterations(loss);
    output.nativeBisectionIterations_ = native_bisection_iterations(loss);
    output.nativeInverseFailures_ =
        table.getPhysicsSource() == 1u && loss.status_ != TableLookupStatus::Success ? 1u
                                                                                     : 0u;
    if (loss.status_ != TableLookupStatus::Success) {
      output.fallback_ = lepton_vertex_detail::table_fallback(
          record, loss_query, loss,
          proposal_native ? ProposalSelectionDrawId : InteractionLossDrawId,
          record.lossQuantile_);
      output.fallbackFlag_ = 1;
      return output;
    }
    record.status_ = InteractionStatus::Selected;
    record.energyFraction_ = loss.value_;
    output.record_ = record;
    output.interactionFlag_ = 1;
    return output;
  }

} // namespace corsika::kokkos

/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include "LeptonFinalStateReference.hpp"
#include "PhotonFinalStateChecks.hpp"
#include <corsika/kokkos/em/algorithms/LeptonFinalStateStep.hpp>
#include <cmath>
#include <fstream>
#include <map>
#include <vector>
namespace lepton_final_test {
  namespace current = ::corsika::kokkos;
  namespace reference = ::lepton_final_reference;
  using execution_space_type = Kokkos::DefaultExecutionSpace;
  using memory_space_type = typename execution_space_type::memory_space;
  struct Payload {
    double real_[512]{};
    std::uint64_t integer_[512]{};
    unsigned reals_{}, integers_{};
    KOKKOS_INLINE_FUNCTION void append(double x) { real_[reals_++] = x; }
    template <class T>
    KOKKOS_INLINE_FUNCTION void append(T x) {
      integer_[integers_++] = static_cast<std::uint64_t>(x);
    }
  };
  inline void check(Payload const& a, Payload const& b) {
    REQUIRE(a.reals_ == b.reals_);
    REQUIRE(a.integers_ == b.integers_);
    REQUIRE(a.reals_ <= 512);
    REQUIRE(a.integers_ <= 512);
    for (unsigned i = 0; i < a.reals_; ++i) {
      INFO("lepton real field " << i);
      REQUIRE((a.real_[i] == b.real_[i] ||
               (std::isnan(a.real_[i]) && std::isnan(b.real_[i]))));
    }
    for (unsigned i = 0; i < a.integers_; ++i) {
      INFO("lepton integer field " << i);
      REQUIRE(a.integer_[i] == b.integer_[i]);
    }
  }
  KOKKOS_INLINE_FUNCTION void flatten(Payload& p,
                                      current::FirstInteractionRecord const& v);
  KOKKOS_INLINE_FUNCTION void flatten(Payload& p,
                                      reference::GpuFirstInteractionSnapshot const& v);
  KOKKOS_INLINE_FUNCTION void flatten(Payload& p,
                                      current::LeptonFinalStateRecord const& v);
  KOKKOS_INLINE_FUNCTION void flatten(Payload& p,
                                      reference::BremsFinalStateRecord const& v);
  KOKKOS_INLINE_FUNCTION void flatten(Payload& p,
                                      current::BremsLpmSuppressionRecord const& v);
  KOKKOS_INLINE_FUNCTION void flatten(Payload& p,
                                      reference::BremsLpmSuppressionRecord const& v);
  KOKKOS_INLINE_FUNCTION void flatten(Payload& p, current::ParticleState const& v);
  KOKKOS_INLINE_FUNCTION void flatten(Payload& p, reference::EmParticleState const& v);
  KOKKOS_INLINE_FUNCTION void flatten(Payload& p, current::InteractionRecord const& v);
  KOKKOS_INLINE_FUNCTION void flatten(Payload& p,
                                      reference::EmInteractionRecord const& v);
  KOKKOS_INLINE_FUNCTION void flatten(Payload& p, current::FallbackRequest const& v);
  KOKKOS_INLINE_FUNCTION void flatten(Payload& p,
                                      reference::ProposalFallbackEvent const& v);
  KOKKOS_INLINE_FUNCTION void flatten(Payload& p,
                                      current::LeptonFinalStateParameters const& v);
  KOKKOS_INLINE_FUNCTION void flatten(Payload& p,
                                      reference::LeptonFinalStateParameters const& v);
  KOKKOS_INLINE_FUNCTION void flatten(Payload& p,
                                      current::LeptonFinalStateClassification const& v);
  KOKKOS_INLINE_FUNCTION void flatten(Payload& p,
                                      reference::LeptonFinalStateClassification const& v);
  KOKKOS_INLINE_FUNCTION void flatten(Payload& p,
                                      current::LeptonFinalStateMaterialization const& v);
  KOKKOS_INLINE_FUNCTION void flatten(
      Payload& p, reference::LeptonFinalStateMaterialization const& v);
  KOKKOS_INLINE_FUNCTION void flatten(Payload& p,
                                      current::FirstInteractionRecord const& v) {
    flatten(p, v.parentAtVertex_);
    flatten(p, v.secondaries_[0]);
    flatten(p, v.secondaries_[1]);
    flatten(p, v.secondaries_[2]);
    p.append(v.processId_);
    p.append(v.secondaryCount_);
  }
  KOKKOS_INLINE_FUNCTION void flatten(Payload& p,
                                      reference::GpuFirstInteractionSnapshot const& v) {
    flatten(p, v.parent_at_vertex);
    flatten(p, v.secondaries[0]);
    flatten(p, v.secondaries[1]);
    flatten(p, v.secondaries[2]);
    p.append(v.process_id);
    p.append(v.secondary_count);
  }
  KOKKOS_INLINE_FUNCTION void flatten(Payload& p,
                                      current::LeptonFinalStateRecord const& v) {
    p.append(v.inputIndex_);
    p.append(v.parentHistoryId_);
    p.append(v.secondaryOffset_);
    p.append(v.secondaryCount_);
    p.append(v.processId_);
    p.append(v.photonEnergyFraction_);
    p.append(v.finalStateUniform_);
    p.append(v.azimuthUniform_);
    p.append(v.auxiliaryUniform_);
    p.append(v.lpmSurvivalProbability_);
    p.append(v.lpmUniform_);
    p.append(v.finalStateDrawId_);
    p.append(v.azimuthDrawId_);
    p.append(v.auxiliaryDrawId_);
    p.append(v.lpmDrawId_);
    p.append(v.thinningStatus_);
    p.append(v.thinningKeepMask_);
    p.append(v.thinningFirstUniform_);
    p.append(v.thinningSecondUniform_);
    p.append(v.thinningFirstDrawId_);
    p.append(v.thinningSecondDrawId_);
    p.append(v.weightedMassConventionCorrectionGeV_);
  }
  KOKKOS_INLINE_FUNCTION void flatten(Payload& p,
                                      reference::BremsFinalStateRecord const& v) {
    p.append(v.input_index);
    p.append(v.parent_history_id);
    p.append(v.secondary_offset);
    p.append(v.secondary_count);
    p.append(v.process_id);
    p.append(v.photon_energy_fraction);
    p.append(v.final_state_uniform);
    p.append(v.azimuth_uniform);
    p.append(v.auxiliary_uniform);
    p.append(v.lpm_survival_probability);
    p.append(v.lpm_uniform);
    p.append(v.final_state_draw_id);
    p.append(v.azimuth_draw_id);
    p.append(v.auxiliary_draw_id);
    p.append(v.lpm_draw_id);
    p.append(v.thinning_status);
    p.append(v.thinning_keep_mask);
    p.append(v.thinning_first_uniform);
    p.append(v.thinning_second_uniform);
    p.append(v.thinning_first_draw_id);
    p.append(v.thinning_second_draw_id);
    p.append(v.weighted_mass_convention_correction_GeV);
  }
  KOKKOS_INLINE_FUNCTION void flatten(Payload& p,
                                      current::BremsLpmSuppressionRecord const& v) {
    flatten(p, v.particle_);
    p.append(v.inputIndex_);
    p.append(v.componentHash_);
    p.append(v.survivalProbability_);
    p.append(v.uniform_);
    p.append(v.drawId_);
  }
  KOKKOS_INLINE_FUNCTION void flatten(Payload& p,
                                      reference::BremsLpmSuppressionRecord const& v) {
    flatten(p, v.particle);
    p.append(v.input_index);
    p.append(v.component_hash);
    p.append(v.survival_probability);
    p.append(v.uniform);
    p.append(v.draw_id);
  }
  KOKKOS_INLINE_FUNCTION void flatten(Payload& p, current::ParticleState const& v) {
    p.append(v.pid_);
    p.append(v.mediumId_);
    p.append(v.generation_);
    p.append(v.reserved_);
    p.append(v.energyGeV_);
    p.append(v.positionM_[0]);
    p.append(v.positionM_[1]);
    p.append(v.positionM_[2]);
    p.append(v.direction_[0]);
    p.append(v.direction_[1]);
    p.append(v.direction_[2]);
    p.append(v.timeS_);
    p.append(v.weight_);
    p.append(v.historyId_);
    p.append(v.parentHistoryId_);
    p.append(v.stepId_);
    p.append(v.environmentNode_);
  }
  KOKKOS_INLINE_FUNCTION void flatten(Payload& p, reference::EmParticleState const& v) {
    p.append(v.pid);
    p.append(v.medium_id);
    p.append(v.generation);
    p.append(v.reserved);
    p.append(v.energy_GeV);
    p.append(v.position_m[0]);
    p.append(v.position_m[1]);
    p.append(v.position_m[2]);
    p.append(v.direction[0]);
    p.append(v.direction[1]);
    p.append(v.direction[2]);
    p.append(v.time_s);
    p.append(v.weight);
    p.append(v.history_id);
    p.append(v.parent_history_id);
    p.append(v.step_id);
    p.append(v.environment_node);
  }
  KOKKOS_INLINE_FUNCTION void flatten(Payload& p, current::InteractionRecord const& v) {
    flatten(p, v.particle_);
    p.append(v.deferredPhotonFallbackReason_);
    p.append(v.deferredPhotonTableStatus_);
    p.append(v.interactionVertexReached_);
    p.append(v.inputIndex_);
    p.append(v.processId_);
    p.append(v.status_);
    p.append(v.componentHash_);
    p.append(v.totalRateCm2PerG_);
    p.append(v.vertexTotalRateCm2PerG_);
    p.append(v.interactionGrammageGPerCm2_);
    p.append(v.massDensityGPerCm3_);
    p.append(v.particleMassGeV_);
    p.append(v.decayDistanceM_);
    p.append(v.decayUniform_);
    p.append(v.decayDrawId_);
    p.append(v.energyFraction_);
    p.append(v.distanceUniform_);
    p.append(v.processUniform_);
    p.append(v.lossQuantile_);
    p.append(v.distanceDrawId_);
    p.append(v.processDrawId_);
    p.append(v.lossDrawId_);
    p.append(v.proposalSelectionUniform_);
    p.append(v.processRandomProcessId_);
    p.append(v.proposalSelectionRandomProcessId_);
    p.append(v.proposalSelectionDrawId_);
  }
  KOKKOS_INLINE_FUNCTION void flatten(Payload& p,
                                      reference::EmInteractionRecord const& v) {
    flatten(p, v.particle);
    p.append(v.deferred_photon_fallback_reason);
    p.append(v.deferred_photon_table_status);
    p.append(v.interaction_vertex_reached);
    p.append(v.input_index);
    p.append(v.process_id);
    p.append(v.status);
    p.append(v.component_hash);
    p.append(v.total_rate_cm2_per_g);
    p.append(v.vertex_total_rate_cm2_per_g);
    p.append(v.interaction_grammage_g_per_cm2);
    p.append(v.mass_density_g_per_cm3);
    p.append(v.particle_mass_GeV);
    p.append(v.decay_distance_m);
    p.append(v.decay_uniform);
    p.append(v.decay_draw_id);
    p.append(v.energy_fraction);
    p.append(v.distance_uniform);
    p.append(v.process_uniform);
    p.append(v.loss_quantile);
    p.append(v.distance_draw_id);
    p.append(v.process_draw_id);
    p.append(v.loss_draw_id);
    p.append(v.proposal_selection_uniform);
    p.append(v.process_random_process_id);
    p.append(v.proposal_selection_random_process_id);
    p.append(v.proposal_selection_draw_id);
  }
  KOKKOS_INLINE_FUNCTION void flatten(Payload& p, current::FallbackRequest const& v) {
    flatten(p, v.particle_);
    p.append(v.interactionVertexReached_);
    p.append(v.inputIndex_);
    p.append(v.processId_);
    p.append(v.reason_);
    p.append(v.diagnosticStatus_);
    p.append(v.diagnosticReserved_);
    p.append(v.diagnosticValue0_);
    p.append(v.diagnosticValue1_);
    p.append(v.diagnosticValue2_);
    p.append(v.componentHash_);
    p.append(v.mediumHash_);
    p.append(v.interactionHash_);
    p.append(v.energyFraction_);
    p.append(v.selectionUniform_);
    p.append(v.lossQuantile_);
    p.append(v.finalStateUniform_);
    p.append(v.outerAcceptanceUniform_);
    p.append(v.randomProcessId_);
    p.append(v.outerAcceptanceRandomProcessId_);
    p.append(v.randomDrawId_);
    p.append(v.outerAcceptanceDrawId_);
    p.append(v.finalStateDrawId_);
  }
  KOKKOS_INLINE_FUNCTION void flatten(Payload& p,
                                      reference::ProposalFallbackEvent const& v) {
    flatten(p, v.particle);
    p.append(v.interaction_vertex_reached);
    p.append(v.input_index);
    p.append(v.process_id);
    p.append(v.reason);
    p.append(v.diagnostic_status);
    p.append(v.diagnostic_reserved);
    p.append(v.diagnostic_value0);
    p.append(v.diagnostic_value1);
    p.append(v.diagnostic_value2);
    p.append(v.component_hash);
    p.append(v.medium_hash);
    p.append(v.interaction_hash);
    p.append(v.energy_fraction);
    p.append(v.selection_uniform);
    p.append(v.loss_quantile);
    p.append(v.final_state_uniform);
    p.append(v.outer_acceptance_uniform);
    p.append(v.random_process_id);
    p.append(v.outer_acceptance_random_process_id);
    p.append(v.random_draw_id);
    p.append(v.outer_acceptance_draw_id);
    p.append(v.final_state_draw_id);
  }
  KOKKOS_INLINE_FUNCTION void flatten(Payload& p,
                                      current::LeptonFinalStateParameters const& v) {
    p.append(v.processId_);
    p.append(v.thinningStatus_);
    p.append(v.thinningKeepMask_);
    p.append(v.finalStateUniform_);
    p.append(v.energySplitFraction_);
    p.append(v.azimuthUniform_);
    p.append(v.auxiliaryUniform_);
    p.append(v.lpmSurvivalProbability_);
    p.append(v.lpmUniform_);
    p.append(v.thinningFirstUniform_);
    p.append(v.thinningSecondUniform_);
    p.append(v.thinningFirstWeight_);
    p.append(v.thinningSecondWeight_);
  }
  KOKKOS_INLINE_FUNCTION void flatten(Payload& p,
                                      reference::LeptonFinalStateParameters const& v) {
    p.append(v.process_id);
    p.append(v.thinning_status);
    p.append(v.thinning_keep_mask);
    p.append(v.final_state_uniform);
    p.append(v.energy_split_fraction);
    p.append(v.azimuth_uniform);
    p.append(v.auxiliary_uniform);
    p.append(v.lpm_survival_probability);
    p.append(v.lpm_uniform);
    p.append(v.thinning_first_uniform);
    p.append(v.thinning_second_uniform);
    p.append(v.thinning_first_weight);
    p.append(v.thinning_second_weight);
  }
  KOKKOS_INLINE_FUNCTION void flatten(Payload& p,
                                      current::LeptonFinalStateClassification const& v) {
    flatten(p, v.parameters_);
    flatten(p, v.fallback_);
    p.append(v.childCount_);
    p.append(v.recordFlag_);
    p.append(v.fallbackFlag_);
    p.append(v.continuationFlag_);
    p.append(v.suppressionFlag_);
    p.append(v.bremsFlag_);
    p.append(v.annihilationFlag_);
    p.append(v.ionizationFlag_);
    p.append(v.electronPairFlag_);
    p.append(v.bremsSuppressionFlag_);
    p.append(v.electronPairSuppressionFlag_);
    p.append(v.electronPairRejectionTrials_);
    p.append(v.electronPairZeroWeightFlag_);
    p.append(v.electronPairRejectionFallbackFlag_);
    p.append(v.electronPairEnvelopeViolationFlag_);
  }
  KOKKOS_INLINE_FUNCTION void flatten(
      Payload& p, reference::LeptonFinalStateClassification const& v) {
    flatten(p, v.parameters);
    flatten(p, v.fallback);
    p.append(v.child_count);
    p.append(v.record_flag);
    p.append(v.fallback_flag);
    p.append(v.continuation_flag);
    p.append(v.suppression_flag);
    p.append(v.brems_flag);
    p.append(v.annihilation_flag);
    p.append(v.ionization_flag);
    p.append(v.electron_pair_flag);
    p.append(v.brems_suppression_flag);
    p.append(v.electron_pair_suppression_flag);
    p.append(v.electron_pair_rejection_trials);
    p.append(v.electron_pair_zero_weight_flag);
    p.append(v.electron_pair_rejection_fallback_flag);
    p.append(v.electron_pair_envelope_violation_flag);
  }
  KOKKOS_INLINE_FUNCTION void flatten(Payload& p,
                                      current::LeptonFinalStateMaterialization const& v) {
    flatten(p, v.record_);
    flatten(p, v.secondaries_[0]);
    flatten(p, v.secondaries_[1]);
    flatten(p, v.secondaries_[2]);
    flatten(p, v.fallback_);
    flatten(p, v.continuation_);
    flatten(p, v.suppression_);
    flatten(p, v.firstInteraction_);
    p.append(v.secondaryCount_);
    p.append(v.hasRecord_);
    p.append(v.hasFallback_);
    p.append(v.hasContinuation_);
    p.append(v.hasSuppression_);
    p.append(v.hasFirstInteraction_);
    p.append(v.error_);
  }
  KOKKOS_INLINE_FUNCTION void flatten(
      Payload& p, reference::LeptonFinalStateMaterialization const& v) {
    flatten(p, v.record);
    flatten(p, v.secondaries[0]);
    flatten(p, v.secondaries[1]);
    flatten(p, v.secondaries[2]);
    flatten(p, v.fallback);
    flatten(p, v.continuation);
    flatten(p, v.suppression);
    flatten(p, v.first_interaction);
    p.append(v.secondary_count);
    p.append(v.has_record);
    p.append(v.has_fallback);
    p.append(v.has_continuation);
    p.append(v.has_suppression);
    p.append(v.has_first_interaction);
    p.append(v.error);
  }
  struct Input {
    current::InteractionRecord interaction_{};
    current::BremsLpmSnapshot lpm_{};
    current::BremsLpmPreparedSnapshot prepared_{};
    current::EmThinningConfig thinning_{};
    std::uint64_t seed_{20261024}, shower_{7}, offset_{19}, firstId_{1000000};
    unsigned usePrepared_{};
  };
  KOKKOS_INLINE_FUNCTION reference::BremsLpmSnapshot convert(
      current::BremsLpmSnapshot const& s) {
    reference::BremsLpmSnapshot r{};
    r.baseline_mass_density_g_per_cm3 = s.baselineMassDensityGPerCm3_;
    r.molecular_density_per_cm3 = s.molecularDensityPerCm3_;
    r.sum_charge = s.sumCharge_;
    r.e_lpm_MeV = s.eLpmMeV_;
    r.lepton_mass_MeV = s.leptonMassMeV_;
    r.electron_mass_MeV = s.electronMassMeV_;
    r.muon_mass_MeV = s.muonMassMeV_;
    r.classical_electron_radius_cm = s.classicalElectronRadiusCm_;
    r.fine_structure_constant = s.fineStructureConstant_;
    r.component_count = s.componentCount_;
    r.reserved = s.reserved_;
    r.muon_baseline_mass_density_g_per_cm3 = s.muonBaselineMassDensityGPerCm3_;
    r.muon_molecular_density_per_cm3 = s.muonMolecularDensityPerCm3_;
    r.muon_sum_charge = s.muonSumCharge_;
    r.muon_e_lpm_MeV = s.muonELpmMeV_;
    for (unsigned i = 0; i < current::MaxBremsLpmComponents; ++i) {
      r.components[i].component_hash = s.components_[i].componentHash_;
      r.components[i].nuclear_charge = s.components_[i].nuclearCharge_;
      r.components[i].atomic_mass_number = s.components_[i].atomicMassNumber_;
      r.components[i].radiation_log_constant = s.components_[i].radiationLogConstant_;
    }
    return r;
  }
  KOKKOS_INLINE_FUNCTION reference::BremsLpmPreparedSnapshot convert(
      current::BremsLpmPreparedSnapshot const& s) {
    reference::BremsLpmPreparedSnapshot r{};
    r.valid = s.valid_;
    r.component_count = s.componentCount_;
    for (unsigned i = 0; i < current::MaxBremsLpmComponents; ++i) {
      r.components[i].component_hash = s.components_[i].componentHash_;
      r.components[i].s1 = s.components_[i].s1_;
      r.components[i].logarithm_s1 = s.components_[i].logarithmS1_;
    }
    return r;
  }
  KOKKOS_INLINE_FUNCTION current::LeptonFinalStateClassification classify(
      Input const& q) {
    if (q.usePrepared_)
      return current::classify_lepton_final_state(q.lpm_, q.prepared_, q.thinning_,
                                                  q.interaction_, q.seed_, q.shower_);
    return current::classify_lepton_final_state(q.lpm_, q.thinning_, q.interaction_,
                                                q.seed_, q.shower_);
  }
  KOKKOS_INLINE_FUNCTION Payload evaluate(Input const& q) {
    auto const c = classify(q);
    auto const m = current::materialize_lepton_final_state(
        q.interaction_, c, q.offset_, q.firstId_, q.lpm_.electronMassMeV_ / 1000.);
    Payload result;
    flatten(result, c);
    flatten(result, m);
    return result;
  }
  KOKKOS_INLINE_FUNCTION Payload expected(Input const& q) {
    auto const lpm = convert(q.lpm_);
    auto const interaction = continuous_test::convert(q.interaction_);
    auto const& t = q.thinning_;
    reference::EmThinningConfig thinning{t.thresholdGeV_, t.maximumWeight_, t.enabled_,
                                         t.eraseZeroWeight_};
    auto const c = q.usePrepared_ ? reference::classifyLeptonFinalState(
                                        lpm, convert(q.prepared_), thinning, interaction,
                                        q.seed_, q.shower_)
                                  : reference::classifyLeptonFinalState(
                                        lpm, thinning, interaction, q.seed_, q.shower_);
    auto const m = reference::materializeLeptonFinalState(
        interaction, c, q.offset_, q.firstId_, lpm.electron_mass_MeV / 1000.);
    Payload result;
    flatten(result, c);
    flatten(result, m);
    return result;
  }
  inline Input input(int pid, int process, double energy) {
    Input q;
    q.interaction_ = photon_final_test::input(process, energy).interaction_;
    q.interaction_.particle_.pid_ = pid;
    q.interaction_.particleMassGeV_ =
        current::is_muon_pid(pid) ? .1056583745 : .00051099895;
    q.lpm_.electronMassMeV_ = .51099895;
    q.lpm_.leptonMassMeV_ = .51099895;
    q.lpm_.muonMassMeV_ = 105.6583745;
    q.interaction_.energyFraction_ = process == current::AnnihilationProcessId ? 1. : .25;
    return q;
  }
  inline void compare_device(std::vector<Input> const& cases) {
    Kokkos::View<Input*, memory_space_type> in("lepton vertices", cases.size());
    auto host = Kokkos::create_mirror_view(in);
    for (std::size_t i = 0; i < cases.size(); ++i) host(i) = cases[i];
    Kokkos::deep_copy(in, host);
    Kokkos::View<unsigned*, memory_space_type> mismatch("lepton mismatches",
                                                        cases.size());
    for (bool reverse : {false, true}) {
      Kokkos::parallel_for(
          "production lepton final-state comparison",
          Kokkos::RangePolicy<execution_space_type>(0, cases.size()),
          KOKKOS_LAMBDA(std::size_t work) {
            auto i = reverse ? in.extent(0) - work - 1 : work;
            auto a = evaluate(in(i));
            auto b = expected(in(i));
            unsigned n = a.reals_ != b.reals_ || a.integers_ != b.integers_;
            for (unsigned k = 0; k < a.reals_; ++k)
              n += !(a.real_[k] == b.real_[k] ||
                     (a.real_[k] != a.real_[k] && b.real_[k] != b.real_[k]));
            for (unsigned k = 0; k < a.integers_; ++k)
              n += a.integer_[k] != b.integer_[k];
            mismatch(i) = n;
          });
      Kokkos::fence();
      auto answer = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, mismatch);
      for (std::size_t i = 0; i < cases.size(); ++i) {
        INFO("lepton device input " << i << " reverse " << reverse);
        REQUIRE(answer(i) == 0);
      }
    }
  }
} // namespace lepton_final_test

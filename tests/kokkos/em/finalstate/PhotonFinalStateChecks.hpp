/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include "PhotonFinalStateReference.hpp"
#include "../continuous/ContinuousAdapters.hpp"
#include <corsika/kokkos/em/algorithms/PhotonFinalStateStep.hpp>
#include <corsika/kokkos/em/random/RandomDomains.hpp>
#include <corsika/kokkos/em/execution/NativeTableStorage.hpp>
#include "../tables/ProposalNativeQueryTestData.hpp"
#include <corsika/kokkos/runtime/Runtime.hpp>
#include <Kokkos_Core.hpp>
#include <catch2/catch_all.hpp>
#include <cmath>
#include <fstream>
#include <vector>
#include <map>
namespace photon_final_test {
  namespace current = ::corsika::kokkos;
  namespace reference = ::photon_final_reference;
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
  KOKKOS_INLINE_FUNCTION void flatten(Payload& p,
                                      current::FirstInteractionRecord const& v);
  KOKKOS_INLINE_FUNCTION void flatten(Payload& p,
                                      reference::GpuFirstInteractionSnapshot const& v);
  KOKKOS_INLINE_FUNCTION void flatten(Payload& p,
                                      current::PhotonFinalStateRecord const& v);
  KOKKOS_INLINE_FUNCTION void flatten(Payload& p,
                                      reference::PhotonFinalStateRecord const& v);
  KOKKOS_INLINE_FUNCTION void flatten(Payload& p,
                                      current::PhotonPairLpmSuppressionRecord const& v);
  KOKKOS_INLINE_FUNCTION void flatten(Payload& p,
                                      reference::PhotonPairLpmSuppressionRecord const& v);
  KOKKOS_INLINE_FUNCTION void flatten(Payload& p, current::ParticleState const& v);
  KOKKOS_INLINE_FUNCTION void flatten(Payload& p, reference::EmParticleState const& v);
  KOKKOS_INLINE_FUNCTION void flatten(Payload& p, current::InteractionRecord const& v);
  KOKKOS_INLINE_FUNCTION void flatten(Payload& p,
                                      reference::EmInteractionRecord const& v);
  KOKKOS_INLINE_FUNCTION void flatten(Payload& p, current::FallbackRequest const& v);
  KOKKOS_INLINE_FUNCTION void flatten(Payload& p,
                                      reference::ProposalFallbackEvent const& v);
  KOKKOS_INLINE_FUNCTION void flatten(Payload& p,
                                      current::PhotonFinalStateParameters const& v);
  KOKKOS_INLINE_FUNCTION void flatten(Payload& p,
                                      reference::PhotonFinalStateParameters const& v);
  KOKKOS_INLINE_FUNCTION void flatten(Payload& p,
                                      current::PhotonFinalStateClassification const& v);
  KOKKOS_INLINE_FUNCTION void flatten(Payload& p,
                                      reference::PhotonFinalStateClassification const& v);
  KOKKOS_INLINE_FUNCTION void flatten(Payload& p,
                                      current::PhotonFinalStateMaterialization const& v);
  KOKKOS_INLINE_FUNCTION void flatten(
      Payload& p, reference::PhotonFinalStateMaterialization const& v);
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
                                      current::PhotonFinalStateRecord const& v) {
    p.append(v.inputIndex_);
    p.append(v.parentHistoryId_);
    p.append(v.secondaryOffset_);
    p.append(v.secondaryCount_);
    p.append(v.processId_);
    p.append(v.energySplitFraction_);
    p.append(v.splitUniform_);
    p.append(v.azimuthUniform_);
    p.append(v.electronPolarUniform_);
    p.append(v.positronPolarUniform_);
    p.append(v.lpmSurvivalProbability_);
    p.append(v.lpmUniform_);
    p.append(v.splitDrawId_);
    p.append(v.azimuthDrawId_);
    p.append(v.electronPolarDrawId_);
    p.append(v.positronPolarDrawId_);
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
                                      reference::PhotonFinalStateRecord const& v) {
    p.append(v.input_index);
    p.append(v.parent_history_id);
    p.append(v.secondary_offset);
    p.append(v.secondary_count);
    p.append(v.process_id);
    p.append(v.energy_split_fraction);
    p.append(v.split_uniform);
    p.append(v.azimuth_uniform);
    p.append(v.electron_polar_uniform);
    p.append(v.positron_polar_uniform);
    p.append(v.lpm_survival_probability);
    p.append(v.lpm_uniform);
    p.append(v.split_draw_id);
    p.append(v.azimuth_draw_id);
    p.append(v.electron_polar_draw_id);
    p.append(v.positron_polar_draw_id);
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
                                      current::PhotonPairLpmSuppressionRecord const& v) {
    flatten(p, v.particle_);
    p.append(v.inputIndex_);
    p.append(v.componentHash_);
    p.append(v.survivalProbability_);
    p.append(v.uniform_);
    p.append(v.drawId_);
  }
  KOKKOS_INLINE_FUNCTION void flatten(
      Payload& p, reference::PhotonPairLpmSuppressionRecord const& v) {
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
                                      current::PhotonFinalStateParameters const& v) {
    p.append(v.processId_);
    p.append(v.thinningStatus_);
    p.append(v.thinningKeepMask_);
    p.append(v.splitFraction_);
    p.append(v.splitUniform_);
    p.append(v.splitDrawId_);
    p.append(v.azimuthUniform_);
    p.append(v.electronPolarUniform_);
    p.append(v.positronPolarUniform_);
    p.append(v.lpmSurvivalProbability_);
    p.append(v.lpmUniform_);
    p.append(v.thinningFirstUniform_);
    p.append(v.thinningSecondUniform_);
    p.append(v.thinningFirstWeight_);
    p.append(v.thinningSecondWeight_);
    p.append(v.secondaryMassGeV_);
  }
  KOKKOS_INLINE_FUNCTION void flatten(Payload& p,
                                      reference::PhotonFinalStateParameters const& v) {
    p.append(v.process_id);
    p.append(v.thinning_status);
    p.append(v.thinning_keep_mask);
    p.append(v.split_fraction);
    p.append(v.split_uniform);
    p.append(v.split_draw_id);
    p.append(v.azimuth_uniform);
    p.append(v.electron_polar_uniform);
    p.append(v.positron_polar_uniform);
    p.append(v.lpm_survival_probability);
    p.append(v.lpm_uniform);
    p.append(v.thinning_first_uniform);
    p.append(v.thinning_second_uniform);
    p.append(v.thinning_first_weight);
    p.append(v.thinning_second_weight);
    p.append(v.secondary_mass_GeV);
  }
  KOKKOS_INLINE_FUNCTION void flatten(Payload& p,
                                      current::PhotonFinalStateClassification const& v) {
    flatten(p, v.parameters_);
    flatten(p, v.fallback_);
    p.append(v.childCount_);
    p.append(v.recordFlag_);
    p.append(v.fallbackFlag_);
    p.append(v.continuationFlag_);
    p.append(v.suppressionFlag_);
    p.append(v.photonPairFlag_);
    p.append(v.comptonFlag_);
    p.append(v.photoelectricFlag_);
    p.append(v.photonMuonPairFlag_);
  }
  KOKKOS_INLINE_FUNCTION void flatten(
      Payload& p, reference::PhotonFinalStateClassification const& v) {
    flatten(p, v.parameters);
    flatten(p, v.fallback);
    p.append(v.child_count);
    p.append(v.record_flag);
    p.append(v.fallback_flag);
    p.append(v.continuation_flag);
    p.append(v.suppression_flag);
    p.append(v.photon_pair_flag);
    p.append(v.compton_flag);
    p.append(v.photoelectric_flag);
    p.append(v.photon_muon_pair_flag);
  }
  KOKKOS_INLINE_FUNCTION void flatten(Payload& p,
                                      current::PhotonFinalStateMaterialization const& v) {
    flatten(p, v.record_);
    flatten(p, v.secondaries_[0]);
    flatten(p, v.secondaries_[1]);
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
      Payload& p, reference::PhotonFinalStateMaterialization const& v) {
    flatten(p, v.record);
    flatten(p, v.secondaries[0]);
    flatten(p, v.secondaries[1]);
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
    current::PhotonPairLpmSnapshot lpm_{};
    current::EmThinningConfig thinning_{};
    std::uint64_t seed_{20261024}, shower_{7}, offset_{19}, firstId_{1000000};
  };
  KOKKOS_INLINE_FUNCTION reference::PhotonPairLpmSnapshot old_lpm(
      current::PhotonPairLpmSnapshot const& s) {
    reference::PhotonPairLpmSnapshot r{};
    r.baseline_mass_density_g_per_cm3 = s.baselineMassDensityGPerCm3_;
    r.molecular_density_per_cm3 = s.molecularDensityPerCm3_;
    r.sum_charge = s.sumCharge_;
    r.e_lpm_MeV = s.eLpmMeV_;
    r.classical_electron_radius_cm = s.classicalElectronRadiusCm_;
    r.fine_structure_constant = s.fineStructureConstant_;
    r.component_count = s.componentCount_;
    r.reserved = s.reserved_;
    for (unsigned i = 0; i < current::MaxPhotonPairLpmComponents; ++i)
      r.components[i] = {s.components_[i].componentHash_, s.components_[i].nuclearCharge_,
                         s.components_[i].radiationLogConstant_};
    return r;
  }
  KOKKOS_INLINE_FUNCTION Payload evaluate(current::ProposalNativeDeviceView const& table,
                                          Input const& input) {
    current::ContinuousPhysicsView physics(table, .5, 10.);
    auto const c = current::classify_photon_final_state(
        physics, input.lpm_, input.thinning_, input.interaction_, input.seed_,
        input.shower_);
    auto const m = current::materialize_photon_final_state(input.interaction_, c,
                                                           input.offset_, input.firstId_);
    Payload result;
    flatten(result, c);
    flatten(result, m);
    return result;
  }
  KOKKOS_INLINE_FUNCTION Payload expected(
      native_table_reference::ProposalNativeDeviceView const& table, Input const& input) {
    reference::tables::NativePhysicsView physics{};
    physics.proposal_native = table;
    physics.em_transport_cut_MeV = .5;
    physics.muon_transport_cut_MeV = 10.;
    auto const interaction = continuous_test::convert(input.interaction_);
    auto const& t = input.thinning_;
    reference::EmThinningConfig thinning{t.thresholdGeV_, t.maximumWeight_, t.enabled_,
                                         t.eraseZeroWeight_};
    auto const c = reference::classifyPhotonFinalState(
        physics, old_lpm(input.lpm_), thinning, interaction, input.seed_, input.shower_);
    auto const m = reference::materializePhotonFinalState(interaction, c, input.offset_,
                                                          input.firstId_);
    Payload result;
    flatten(result, c);
    flatten(result, m);
    return result;
  }
  inline void check(Payload const& a, Payload const& b) {
    REQUIRE(a.reals_ == b.reals_);
    REQUIRE(a.integers_ == b.integers_);
    REQUIRE(a.reals_ <= 512);
    REQUIRE(a.integers_ <= 512);
    for (unsigned i = 0; i < a.reals_; ++i) {
      INFO("real field " << i);
      REQUIRE((a.real_[i] == b.real_[i] ||
               (std::isnan(a.real_[i]) && std::isnan(b.real_[i]))));
    }
    for (unsigned i = 0; i < a.integers_; ++i) {
      INFO("integer field " << i);
      REQUIRE(a.integer_[i] == b.integer_[i]);
    }
  }
  inline Input input(int process, double energy, std::uint64_t component = 17) {
    Input q;
    auto& r = q.interaction_;
    r.status_ = current::InteractionStatus::Selected;
    r.interactionVertexReached_ = 1;
    r.inputIndex_ = 123;
    r.processId_ = process;
    r.componentHash_ = component;
    r.energyFraction_ = .25;
    r.lossQuantile_ = .375;
    r.lossDrawId_ = 0;
    r.processUniform_ = .75;
    r.proposalSelectionUniform_ = .125;
    r.proposalSelectionRandomProcessId_ = current::ProposalSelectionRandomProcessId;
    r.processRandomProcessId_ = current::InteractionColumnRandomProcessId;
    r.massDensityGPerCm3_ = .0012;
    auto& p = r.particle_;
    p.pid_ = 22;
    p.energyGeV_ = energy;
    p.weight_ = 2.;
    p.direction_[0] = .36;
    p.direction_[1] = .48;
    p.direction_[2] = .8;
    p.positionM_[0] = 123.;
    p.positionM_[1] = -42.;
    p.positionM_[2] = 1.e6;
    p.timeS_ = 1.e-5;
    p.historyId_ = 100;
    p.stepId_ = 12;
    p.environmentNode_ = 2;
    q.lpm_.componentCount_ = 1;
    q.lpm_.fineStructureConstant_ = 1. / 137.;
    q.lpm_.components_[0] = {component, 7., 183.};
    return q;
  }
  inline void compare_device(current::ProposalNativeDeviceView const& table,
                             native_table_reference::ProposalNativeDeviceView const& old,
                             std::vector<Input> const& cases) {
    Kokkos::View<Input*, memory_space_type> in("photon vertices", cases.size());
    auto host = Kokkos::create_mirror_view(in);
    for (std::size_t i = 0; i < cases.size(); ++i) host(i) = cases[i];
    Kokkos::deep_copy(in, host);
    // Bound test memory and thread-local output while comparing each physical field.
    Kokkos::View<unsigned*, memory_space_type> mismatch("photon mismatches",
                                                        cases.size());
    for (bool reverse : {false, true}) {
      Kokkos::parallel_for(
          "production photon final-state comparison",
          Kokkos::RangePolicy<execution_space_type>(0, cases.size()),
          KOKKOS_LAMBDA(std::size_t work) {
            auto i = reverse ? in.extent(0) - work - 1 : work;
            auto a = evaluate(table, in(i));
            auto b = expected(old, in(i));
            unsigned n = a.reals_ != b.reals_ || a.integers_ != b.integers_;
            for (unsigned k = 0; k < a.reals_; ++k)
              n += !(a.real_[k] == b.real_[k] ||
                     (a.real_[k] != a.real_[k] && b.real_[k] != b.real_[k]));
            for (unsigned k = 0; k < a.integers_; ++k)
              n += a.integer_[k] != b.integer_[k];
            mismatch(i) = n;
          });
      auto out = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, mismatch);
      for (std::size_t i = 0; i < cases.size(); ++i) {
        INFO("device vertex " << i);
        REQUIRE(out(i) == 0);
      }
    }
  }
} // namespace photon_final_test

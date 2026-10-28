/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/kokkos/output/algorithms/ProfileAccumulation.hpp>
#include <corsika/kokkos/output/ProfileAtomicOperations.hpp>
#include <tests/kokkos/output/ProfileAccumulationReference.hpp>
#include <tests/kokkos/em/transport/TransportChecks.hpp>
#include <algorithm>
#include <fstream>
namespace profile_test {
  namespace current = corsika::kokkos;
  namespace reference = profile_accumulation_reference;
  struct Case {
    int kind_, pid_, limit_, variant_;
    double start_, end_, weight_, energy_;
  };
  CORSIKA_HOST_DEVICE inline current::ProfileAccumulatorView view(
      long long* hist, current::ProfileCounters* counters, int mode, double threshold) {
    return {hist,      hist + 8,  hist + 16,
            hist + 24, hist + 32, hist + 40,
            hist + 48, hist + 56, counters,
            8,         1.,        static_cast<corsika::ProfileCrossingMode>(mode),
            threshold, 1024.,     1. / 1024.,
            1024.,     1. / 1024.};
  }
  CORSIKA_HOST_DEVICE inline reference::DeviceProfileAccumulator old_view(
      long long* hist, reference::DeviceProfileCounters* counters, int mode,
      double threshold) {
    return {hist,      hist + 8,  hist + 16,
            hist + 24, hist + 32, hist + 40,
            hist + 48, hist + 56, counters,
            8,         1.,        static_cast<reference::ProfileCrossingMode>(mode),
            threshold, 1024.,     1. / 1024.,
            1024.,     1. / 1024.};
  }
  inline void compare(current::ProfileCounters const& a,
                      reference::DeviceProfileCounters const& b) {
    REQUIRE(a.steps_ == b.steps);
    REQUIRE(a.depositedSteps_ == b.deposited_steps);
    REQUIRE(a.photonCuts_ == b.photon_cuts);
    for (std::size_t i = 0; i < 8; ++i) REQUIRE(a.leptonLimits_[i] == b.lepton_limits[i]);
    REQUIRE(a.moliereTrials_ == b.moliere_trials);
    REQUIRE(a.moliereDeflections_ == b.moliere_deflections);
    REQUIRE(a.moliereZeroDeflections_ == b.moliere_zero_deflections);
    REQUIRE(a.moliereNewtonIterations_ == b.moliere_newton_iterations);
    REQUIRE(a.moliereMaxNewtonIterations_ == b.moliere_max_newton_iterations);
    REQUIRE(a.thinningHillasVertices_ == b.thinning_hillas_vertices);
    REQUIRE(a.thinningStatisticalVertices_ == b.thinning_statistical_vertices);
    REQUIRE(a.thinningParticlesDiscarded_ == b.thinning_particles_discarded);
    REQUIRE(a.fixedPointOverflows_ == b.fixed_point_overflows);
    // The frozen production oracle labels MaterialBoundary as invalid. Only
    // that diagnostic moves to its own counter; every histogram/energy stays exact.
    REQUIRE(a.invalidRecords_ + a.leptonMaterialBoundaries_ == b.invalid_records);
    REQUIRE(a.weightedMediumRestMassInput_ == b.weighted_medium_rest_mass_input);
    REQUIRE(a.weightedCutRestMassEnergy_ == b.weighted_cut_rest_mass_energy);
    REQUIRE(a.weightedObservedTotalEnergy_ == b.weighted_observed_total_energy);
    REQUIRE(a.weightedEscapedTotalEnergy_ == b.weighted_escaped_total_energy);
    REQUIRE(a.weightedUnwrittenPhotoelectricBindingEnergy_ ==
            b.weighted_unwritten_photoelectric_binding_energy);
    REQUIRE(a.weightedObservationCutOverlapEnergy_ ==
            b.weighted_observation_cut_overlap_energy);
    REQUIRE(a.weightedMassConventionCorrection_ == b.weighted_mass_convention_correction);
    REQUIRE(a.weightedNativeMuonPairCutEnergy_ == b.weighted_native_muon_pair_cut_energy);
  }
  static_assert(offsetof(current::ProfileCounters, leptonMaterialBoundaries_) ==
                sizeof(reference::DeviceProfileCounters));
  static_assert(sizeof(current::ProfileCounters) ==
                sizeof(reference::DeviceProfileCounters) + sizeof(unsigned long long));
  static_assert(alignof(current::ProfileCounters) ==
                alignof(reference::DeviceProfileCounters));
  static_assert(offsetof(current::ProfileCounters, steps_) ==
                offsetof(reference::DeviceProfileCounters, steps));
  static_assert(offsetof(current::ProfileCounters, depositedSteps_) ==
                offsetof(reference::DeviceProfileCounters, deposited_steps));
  static_assert(offsetof(current::ProfileCounters, photonCuts_) ==
                offsetof(reference::DeviceProfileCounters, photon_cuts));
  static_assert(offsetof(current::ProfileCounters, leptonLimits_) ==
                offsetof(reference::DeviceProfileCounters, lepton_limits));
  static_assert(offsetof(current::ProfileCounters, moliereTrials_) ==
                offsetof(reference::DeviceProfileCounters, moliere_trials));
  static_assert(offsetof(current::ProfileCounters, moliereDeflections_) ==
                offsetof(reference::DeviceProfileCounters, moliere_deflections));
  static_assert(offsetof(current::ProfileCounters, moliereZeroDeflections_) ==
                offsetof(reference::DeviceProfileCounters, moliere_zero_deflections));
  static_assert(offsetof(current::ProfileCounters, moliereNewtonIterations_) ==
                offsetof(reference::DeviceProfileCounters, moliere_newton_iterations));
  static_assert(offsetof(current::ProfileCounters, moliereMaxNewtonIterations_) ==
                offsetof(reference::DeviceProfileCounters,
                         moliere_max_newton_iterations));
  static_assert(offsetof(current::ProfileCounters, thinningHillasVertices_) ==
                offsetof(reference::DeviceProfileCounters, thinning_hillas_vertices));
  static_assert(offsetof(current::ProfileCounters, thinningStatisticalVertices_) ==
                offsetof(reference::DeviceProfileCounters,
                         thinning_statistical_vertices));
  static_assert(offsetof(current::ProfileCounters, thinningParticlesDiscarded_) ==
                offsetof(reference::DeviceProfileCounters, thinning_particles_discarded));
  static_assert(offsetof(current::ProfileCounters, fixedPointOverflows_) ==
                offsetof(reference::DeviceProfileCounters, fixed_point_overflows));
  static_assert(offsetof(current::ProfileCounters, invalidRecords_) ==
                offsetof(reference::DeviceProfileCounters, invalid_records));
  static_assert(offsetof(current::ProfileCounters, weightedMediumRestMassInput_) ==
                offsetof(reference::DeviceProfileCounters,
                         weighted_medium_rest_mass_input));
  static_assert(offsetof(current::ProfileCounters, weightedCutRestMassEnergy_) ==
                offsetof(reference::DeviceProfileCounters,
                         weighted_cut_rest_mass_energy));
  static_assert(offsetof(current::ProfileCounters, weightedObservedTotalEnergy_) ==
                offsetof(reference::DeviceProfileCounters,
                         weighted_observed_total_energy));
  static_assert(offsetof(current::ProfileCounters, weightedEscapedTotalEnergy_) ==
                offsetof(reference::DeviceProfileCounters,
                         weighted_escaped_total_energy));
  static_assert(offsetof(current::ProfileCounters,
                         weightedUnwrittenPhotoelectricBindingEnergy_) ==
                offsetof(reference::DeviceProfileCounters,
                         weighted_unwritten_photoelectric_binding_energy));
  static_assert(offsetof(current::ProfileCounters,
                         weightedObservationCutOverlapEnergy_) ==
                offsetof(reference::DeviceProfileCounters,
                         weighted_observation_cut_overlap_energy));
  static_assert(offsetof(current::ProfileCounters, weightedMassConventionCorrection_) ==
                offsetof(reference::DeviceProfileCounters,
                         weighted_mass_convention_correction));
  static_assert(offsetof(current::ProfileCounters, weightedNativeMuonPairCutEnergy_) ==
                offsetof(reference::DeviceProfileCounters,
                         weighted_native_muon_pair_cut_energy));
  static_assert(sizeof(current::ProfileAccumulatorView) ==
                sizeof(reference::DeviceProfileAccumulator));
  static_assert(alignof(current::ProfileAccumulatorView) ==
                alignof(reference::DeviceProfileAccumulator));
  static_assert(offsetof(current::ProfileAccumulatorView, photons_) ==
                offsetof(reference::DeviceProfileAccumulator, photons));
  static_assert(offsetof(current::ProfileAccumulatorView, electrons_) ==
                offsetof(reference::DeviceProfileAccumulator, electrons));
  static_assert(offsetof(current::ProfileAccumulatorView, positrons_) ==
                offsetof(reference::DeviceProfileAccumulator, positrons));
  static_assert(offsetof(current::ProfileAccumulatorView, muonsMinus_) ==
                offsetof(reference::DeviceProfileAccumulator, muons_minus));
  static_assert(offsetof(current::ProfileAccumulatorView, muonsPlus_) ==
                offsetof(reference::DeviceProfileAccumulator, muons_plus));
  static_assert(offsetof(current::ProfileAccumulatorView, muonParentProductions_) ==
                offsetof(reference::DeviceProfileAccumulator, muon_parent_productions));
  static_assert(offsetof(current::ProfileAccumulatorView, energyLoss_) ==
                offsetof(reference::DeviceProfileAccumulator, energy_loss));
  static_assert(offsetof(current::ProfileAccumulatorView, muonEnergyLoss_) ==
                offsetof(reference::DeviceProfileAccumulator, muon_energy_loss));
  static_assert(offsetof(current::ProfileAccumulatorView, counters_) ==
                offsetof(reference::DeviceProfileAccumulator, counters));
  static_assert(offsetof(current::ProfileAccumulatorView, bins_) ==
                offsetof(reference::DeviceProfileAccumulator, bins));
  static_assert(offsetof(current::ProfileAccumulatorView, binWidthGPerCm2_) ==
                offsetof(reference::DeviceProfileAccumulator, bin_width_g_per_cm2));
  static_assert(offsetof(current::ProfileAccumulatorView, crossingMode_) ==
                offsetof(reference::DeviceProfileAccumulator, crossing_mode));
  static_assert(offsetof(current::ProfileAccumulatorView, energyLossThresholdGPerCm2_) ==
                offsetof(reference::DeviceProfileAccumulator,
                         energy_loss_threshold_g_per_cm2));
  static_assert(offsetof(current::ProfileAccumulatorView, weightScale_) ==
                offsetof(reference::DeviceProfileAccumulator, weight_scale));
  static_assert(offsetof(current::ProfileAccumulatorView, inverseWeightScale_) ==
                offsetof(reference::DeviceProfileAccumulator, inverse_weight_scale));
  static_assert(offsetof(current::ProfileAccumulatorView, energyScale_) ==
                offsetof(reference::DeviceProfileAccumulator, energy_scale));
  static_assert(offsetof(current::ProfileAccumulatorView, inverseEnergyScale_) ==
                offsetof(reference::DeviceProfileAccumulator, inverse_energy_scale));
  namespace current_driver {
    // Test scenario shared by the independent frozen and migrated implementations.
    CORSIKA_HOST_DEVICE inline void apply(Case const& q,
                                          current::ProfileAccumulatorView const& a,
                                          current::ProfileProjectionView const& p) {
      using Atomic = current::ProfileAtomicOperations;
      switch (q.kind_) {
        case 0:
          current::accumulate_particle_profile<Atomic>(a, q.pid_, q.start_, q.end_,
                                                       q.weight_);
          break;
        case 1:
          current::accumulate_energy_profile<Atomic>(a, a.energyLoss_, q.start_, q.end_,
                                                     q.energy_ * q.weight_);
          break;
        case 2:
          current::accumulate_point_energy_profile<Atomic>(
              a, a.energyLoss_, q.end_, q.energy_ * q.weight_, q.variant_ != 0);
          break;
        case 3:
          current::accumulate_terminal_energy<Atomic>(a, q.energy_ * q.weight_,
                                                      q.variant_ != 0);
          break;
        case 4:
          current::accumulate_muon_parent_production<Atomic>(a, q.end_, q.weight_);
          break;
        case 5: {
          current::PhotonTransportRecord r{};
          r.start_.pid_ = 22;
          r.start_.historyId_ = 7;
          r.start_.positionM_[2] = q.start_;
          r.start_.weight_ = q.weight_;
          r.start_.energyGeV_ = q.energy_ + 1.;
          r.end_ = r.start_;
          r.end_.positionM_[2] = q.end_;
          r.end_.energyGeV_ = q.energy_;
          r.limit_ = static_cast<current::PhotonTransportLimit>(q.limit_);
          r.cutDepositedEnergyGeV_ = q.energy_;
          r.observationSurfaceReachedBeforeCut_ = q.variant_ % 2;
          current::accumulate_photon_profile_step<Atomic>(p, a, r);
          break;
        }
        case 6: {
          current::LeptonTransportRecord r{};
          r.start_.pid_ = q.pid_;
          r.start_.historyId_ = 7;
          r.start_.positionM_[2] = q.start_;
          r.start_.weight_ = q.weight_;
          r.start_.energyGeV_ = q.energy_ + 1.;
          r.end_ = r.start_;
          r.end_.positionM_[2] = q.end_;
          r.end_.energyGeV_ = q.energy_;
          r.limit_ = static_cast<current::LeptonTransportLimit>(q.limit_);
          r.continuousDepositedEnergyGeV_ = q.energy_ / 4.;
          r.cutDepositedEnergyGeV_ = q.energy_ / 8.;
          r.observationSurfaceReachedBeforeCut_ = q.variant_ % 2;
          r.multipleScatteringIterations_ = 3 * q.variant_;
          r.multipleScatteringApplied_ = q.variant_ % 2;
          r.multipleScatteringStatus_ =
              static_cast<std::uint32_t>(current::MoliereStatus::NoDeflection);
          if (q.variant_ == 3)
            r.interaction_.status_ = current::InteractionStatus::AtRestAnnihilation;
          if (q.variant_ == 4 || q.variant_ == 5) {
            r.start_.pid_ = 13;
            r.start_.reserved_ = current::NativeMuonPairCutFlag;
            r.start_.generation_ = 1;
            r.start_.energyGeV_ = .1;
            r.end_.energyGeV_ = .1;
            r.continuousDepositedEnergyGeV_ = q.variant_ == 4 ? 0. : .01;
            r.cutDepositedEnergyGeV_ = .1 - current::TransportMuonMassGeV;
          }
          if (q.variant_ == 6)
            current::accumulate_lepton_profile_step<Atomic, false>(p, a, r);
          else
            current::accumulate_lepton_profile_step<Atomic>(p, a, r);
          break;
        }
        case 7: {
          current::PhotonTransportRecord t[3]{};
          for (int i = 0; i < 3; ++i) {
            t[i].inputIndex_ = 2 * i;
            t[i].start_.historyId_ = 7;
            t[i].start_.weight_ = q.weight_;
            t[i].end_.energyGeV_ = q.energy_;
            t[i].interaction_.processId_ = q.pid_;
          }
          current::PhotonFinalStateRecord r{};
          r.inputIndex_ = q.variant_ == 1 ? 1 : 2;
          r.parentHistoryId_ = q.variant_ == 2 ? 8 : 7;
          r.processId_ = q.pid_;
          r.secondaryCount_ = q.limit_;
          r.energySplitFraction_ = q.start_;
          r.thinningStatus_ = q.variant_ == 5 ? 99 : q.variant_ % 3;
          r.weightedMassConventionCorrectionGeV_ = q.weight_ / 1024.;
          if (q.variant_ == 3) t[1].interaction_.processId_ = -9;
          current::accumulate_photon_profile_final_state<Atomic>(
              p, a, t, q.variant_ == 4 ? 0 : 3, r);
          break;
        }
        case 8: {
          current::LeptonTransportRecord t[3]{};
          for (int i = 0; i < 3; ++i) {
            t[i].inputIndex_ = 2 * i;
            t[i].start_.historyId_ = 7;
            t[i].start_.weight_ = q.weight_;
            t[i].start_.pid_ = q.variant_ % 2 ? 13 : 11;
            t[i].end_.positionM_[2] = q.end_;
          }
          current::LeptonFinalStateRecord r{};
          r.inputIndex_ = q.variant_ == 1 ? 1 : 2;
          r.parentHistoryId_ = q.variant_ == 2 ? 8 : 7;
          r.processId_ = q.pid_;
          r.secondaryCount_ = q.limit_;
          r.finalStateDrawId_ =
              q.variant_ == 3 ? current::AtRestAnnihilationPolarDrawId : 0;
          r.thinningStatus_ = q.variant_ == 5 ? 99 : q.variant_ % 3;
          r.weightedMassConventionCorrectionGeV_ = q.weight_ / 1024.;
          current::accumulate_lepton_profile_final_state<Atomic>(
              p, a, t, q.variant_ == 4 ? 0 : 3, r);
          break;
        }
        case 9:
          current::add_profile_fixed_point<Atomic>(a.energyLoss_, q.energy_, q.weight_,
                                                   a.counters_);
          break;
      }
    }
  } // namespace current_driver
  namespace reference_driver {
    // Test scenario shared by the independent frozen and migrated implementations.
    CORSIKA_HOST_DEVICE inline void apply(Case const& q,
                                          reference::DeviceProfileAccumulator const& a,
                                          reference::DeviceProfileProjection const& p) {
      using Atomic = reference::KokkosProfileAtomicOperations;
      switch (q.kind_) {
        case 0:
          reference::accumulateParticleProfile<Atomic>(a, q.pid_, q.start_, q.end_,
                                                       q.weight_);
          break;
        case 1:
          reference::accumulateEnergyProfile<Atomic>(a, a.energy_loss, q.start_, q.end_,
                                                     q.energy_ * q.weight_);
          break;
        case 2:
          reference::accumulatePointEnergyProfile<Atomic>(
              a, a.energy_loss, q.end_, q.energy_ * q.weight_, q.variant_ != 0);
          break;
        case 3:
          reference::accumulateTerminalEnergy<Atomic>(a, q.energy_ * q.weight_,
                                                      q.variant_ != 0);
          break;
        case 4:
          reference::accumulateMuonParentProduction<Atomic>(a, q.end_, q.weight_);
          break;
        case 5: {
          reference::PhotonTransportRecord r{};
          r.start.pid = 22;
          r.start.history_id = 7;
          r.start.position_m[2] = q.start_;
          r.start.weight = q.weight_;
          r.start.energy_GeV = q.energy_ + 1.;
          r.end = r.start;
          r.end.position_m[2] = q.end_;
          r.end.energy_GeV = q.energy_;
          r.limit = static_cast<reference::PhotonTransportLimit>(q.limit_);
          r.cut_deposited_energy_GeV = q.energy_;
          r.observation_surface_reached_before_cut = q.variant_ % 2;
          reference::accumulatePhotonProfileStep<Atomic>(p, a, r);
          break;
        }
        case 6: {
          reference::LeptonTransportRecord r{};
          r.start.pid = q.pid_;
          r.start.history_id = 7;
          r.start.position_m[2] = q.start_;
          r.start.weight = q.weight_;
          r.start.energy_GeV = q.energy_ + 1.;
          r.end = r.start;
          r.end.position_m[2] = q.end_;
          r.end.energy_GeV = q.energy_;
          r.limit = static_cast<reference::LeptonTransportLimit>(q.limit_);
          r.continuous_deposited_energy_GeV = q.energy_ / 4.;
          r.cut_deposited_energy_GeV = q.energy_ / 8.;
          r.observation_surface_reached_before_cut = q.variant_ % 2;
          r.multiple_scattering_iterations = 3 * q.variant_;
          r.multiple_scattering_applied = q.variant_ % 2;
          r.multiple_scattering_status =
              static_cast<std::uint32_t>(reference::MoliereStatus::NoDeflection);
          if (q.variant_ == 3)
            r.interaction.status = reference::EmInteractionStatus::AtRestAnnihilation;
          if (q.variant_ == 4 || q.variant_ == 5) {
            r.start.pid = 13;
            r.start.reserved = reference::NativeMuonPairCutFlag;
            r.start.generation = 1;
            r.start.energy_GeV = .1;
            r.end.energy_GeV = .1;
            r.continuous_deposited_energy_GeV = q.variant_ == 4 ? 0. : .01;
            r.cut_deposited_energy_GeV = .1 - reference::TransportMuonMassGeV;
          }
          if (q.variant_ == 6)
            reference::accumulateLeptonProfileStep<Atomic, false>(p, a, r);
          else
            reference::accumulateLeptonProfileStep<Atomic>(p, a, r);
          break;
        }
        case 7: {
          reference::PhotonTransportRecord t[3]{};
          for (int i = 0; i < 3; ++i) {
            t[i].input_index = 2 * i;
            t[i].start.history_id = 7;
            t[i].start.weight = q.weight_;
            t[i].end.energy_GeV = q.energy_;
            t[i].interaction.process_id = q.pid_;
          }
          reference::PhotonPairFinalStateRecord r{};
          r.input_index = q.variant_ == 1 ? 1 : 2;
          r.parent_history_id = q.variant_ == 2 ? 8 : 7;
          r.process_id = q.pid_;
          r.secondary_count = q.limit_;
          r.energy_split_fraction = q.start_;
          r.thinning_status = q.variant_ == 5 ? 99 : q.variant_ % 3;
          r.weighted_mass_convention_correction_GeV = q.weight_ / 1024.;
          if (q.variant_ == 3) t[1].interaction.process_id = -9;
          reference::accumulatePhotonProfileFinalState<Atomic>(
              p, a, t, q.variant_ == 4 ? 0 : 3, r);
          break;
        }
        case 8: {
          reference::LeptonTransportRecord t[3]{};
          for (int i = 0; i < 3; ++i) {
            t[i].input_index = 2 * i;
            t[i].start.history_id = 7;
            t[i].start.weight = q.weight_;
            t[i].start.pid = q.variant_ % 2 ? 13 : 11;
            t[i].end.position_m[2] = q.end_;
          }
          reference::BremsFinalStateRecord r{};
          r.input_index = q.variant_ == 1 ? 1 : 2;
          r.parent_history_id = q.variant_ == 2 ? 8 : 7;
          r.process_id = q.pid_;
          r.secondary_count = q.limit_;
          r.final_state_draw_id =
              q.variant_ == 3 ? reference::AtRestAnnihilationPolarDrawId : 0;
          r.thinning_status = q.variant_ == 5 ? 99 : q.variant_ % 3;
          r.weighted_mass_convention_correction_GeV = q.weight_ / 1024.;
          reference::accumulateLeptonProfileFinalState<Atomic>(
              p, a, t, q.variant_ == 4 ? 0 : 3, r);
          break;
        }
        case 9:
          reference::addProfileFixedPoint<Atomic>(a.energy_loss, q.energy_, q.weight_,
                                                  a.counters);
          break;
      }
    }
  } // namespace reference_driver
} // namespace profile_test

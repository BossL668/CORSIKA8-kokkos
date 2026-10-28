/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

namespace corsika::kokkos {

  CORSIKA_HOST_DEVICE inline double project_profile_grammage(
      ProfileProjectionView const& projection, double const position_m[3]) {
    auto projected_length_m = 0.;
    for (int axis = 0; axis < 3; ++axis)
      projected_length_m += (position_m[axis] - projection.axisStartPositionM_[axis]) *
                            projection.axisDirection_[axis];
    auto const fractional_bin = projected_length_m / projection.axisStepLengthM_;
    if (fractional_bin < 0.) return projection.axisGrammageGPerCm2_[0];
    auto const lower = static_cast<std::size_t>(fractional_bin);
    auto const upper = lower + 1;
    if (upper >= projection.axisSupportCount_)
      return projection.axisGrammageGPerCm2_[projection.axisSupportCount_ - 1];
    auto const fraction = fractional_bin - static_cast<double>(lower);
    return projection.axisGrammageGPerCm2_[upper] * fraction +
           projection.axisGrammageGPerCm2_[lower] * (1. - fraction);
  }
  CORSIKA_HOST_DEVICE inline ProjectedStepRecord project_photon_step(
      ProfileProjectionView const& projection, PhotonTransportRecord const& record) {
    using namespace corsika::kokkos;
    ProjectedStepRecord projected{};
    projected.historyId_ = record.start_.historyId_;
    projected.pid_ = record.start_.pid_;
    projected.processId_ = record.limit_ == PhotonTransportLimit::Interaction
                               ? record.interaction_.processId_
                               : 0;
    projected.transportLimit_ = static_cast<std::int32_t>(record.limit_);
    projected.startGrammageGPerCm2_ =
        project_profile_grammage(projection, record.start_.positionM_);
    projected.endGrammageGPerCm2_ =
        project_profile_grammage(projection, record.end_.positionM_);
    projected.endEnergyGeV_ = record.end_.energyGeV_;
    projected.depositedEnergyGeV_ = record.cutDepositedEnergyGeV_;
    projected.cutDepositedEnergyGeV_ = record.cutDepositedEnergyGeV_;
    projected.observationSurfaceReachedBeforeCut_ =
        record.observationSurfaceReachedBeforeCut_;
    projected.weight_ = record.start_.weight_;
    return projected;
  }
  CORSIKA_HOST_DEVICE inline ProjectedStepRecord project_lepton_step(
      ProfileProjectionView const& projection, LeptonTransportRecord const& record) {
    using namespace corsika::kokkos;
    ProjectedStepRecord projected{};
    projected.historyId_ = record.start_.historyId_;
    projected.pid_ = record.start_.pid_;
    projected.processId_ =
        (record.limit_ == LeptonTransportLimit::InteractionCandidate ||
         record.interaction_.status_ == InteractionStatus::AtRestAnnihilation)
            ? record.interaction_.processId_
            : 0;
    projected.transportLimit_ = static_cast<std::int32_t>(record.limit_);
    projected.reserved_ = static_cast<std::int32_t>(
        static_cast<std::uint32_t>(record.multipleScatteringApplied_) |
        (static_cast<std::uint32_t>(record.multipleScatteringStatus_) << 1) |
        (record.multipleScatteringIterations_ << 8));
    projected.nativeMuonPairCut_ =
        is_native_muon_pair_cut(record.start_) ? NativeMuonPairCutFlag : 0U;
    projected.startGrammageGPerCm2_ =
        project_profile_grammage(projection, record.start_.positionM_);
    projected.endGrammageGPerCm2_ =
        project_profile_grammage(projection, record.end_.positionM_);
    projected.endEnergyGeV_ = record.end_.energyGeV_;
    projected.depositedEnergyGeV_ =
        record.continuousDepositedEnergyGeV_ + record.cutDepositedEnergyGeV_;
    projected.cutDepositedEnergyGeV_ = record.cutDepositedEnergyGeV_;
    projected.observationSurfaceReachedBeforeCut_ =
        record.observationSurfaceReachedBeforeCut_;
    projected.weight_ = record.start_.weight_;
    return projected;
  }

} // namespace corsika::kokkos

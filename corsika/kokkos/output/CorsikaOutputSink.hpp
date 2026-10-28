/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once
#include <corsika/kokkos/data/StepRecord.hpp>
#include <corsika/kokkos/output/ProjectedStepRecord.hpp>
#include <corsika/kokkos/output/ProfileResult.hpp>
#include <corsika/kokkos/output/ObservationRecord.hpp>
#include <corsika/kokkos/output/OutputSinkStatistics.hpp>
#include <corsika/kokkos/output/OutputRecordConversion.hpp>
#include <corsika/kokkos/em/FirstInteractionRecord.hpp>
#include <corsika/kokkos/em/LeptonTransportRecord.hpp>
#include <corsika/kokkos/em/algorithms/NativeMuonPairCut.hpp>
#include <corsika/kokkos/output/RadioTrackRecord.hpp>
#include <corsika/kokkos/radio/RadioWaveforms.hpp>
#include <corsika/framework/core/Step.hpp>
#include <corsika/framework/geometry/Line.hpp>
#include <corsika/framework/geometry/StraightTrajectory.hpp>
#include <corsika/framework/process/ProcessReturn.hpp>
#include <corsika/modules/writers/FirstInteractionSnapshot.hpp>
#include <cstddef>

namespace corsika::kokkos {
  /** No-output marker used by validation routers. */
  struct NullCorsikaOutputSink {};
  /** Borrow native writers; apply each record weight once at the output boundary.
   * Records carry total GeV; native particle outputs carry kinetic energy.
   * This adapter neither advances transport nor selects cuts or thinning. */
  template <typename TEnergyLossWriter, typename TLongitudinalWriter,
            typename TProductionWriter, typename TObservationPlane,
            typename TInteractionWriter, typename TCoreas, typename TZhs>
  class CorsikaOutputSink {
  public:
    CorsikaOutputSink(CoordinateSystemPtr coordinate_system,
                      TEnergyLossWriter& energy_loss, TLongitudinalWriter& longitudinal,
                      TProductionWriter& production, TObservationPlane& observation,
                      TInteractionWriter& interaction, TCoreas& coreas, TZhs& zhs,
                      bool radio_enabled, bool gpu_radio_enabled = false);

    void onStep(StepRecord const& record);

    void onProjectedStep(ProjectedStepRecord const& record);

    void onGpuProfile(ProfileResult const& result);

    void onRadioTrack(RadioTrackRecord const& record);

    void onGpuRadioWaveforms(radio::RadioWaveforms const& waveforms,
                             std::uint64_t track_count);

    void onObservation(ObservationRecord const& record);

    void onFirstInteraction(FirstInteractionRecord const& record);

    CorsikaOutputSinkStatistics const& statistics() const noexcept;

  private:
    static void validateStep(StepRecord const& record);

    static void validateProjectedStep(ProjectedStepRecord const& record);

    CoordinateSystemPtr coordinateSystem_;
    TEnergyLossWriter& energyLoss_;
    TLongitudinalWriter& longitudinal_;
    TProductionWriter& production_;
    TObservationPlane& observation_;
    TInteractionWriter& interaction_;
    TCoreas& coreas_;
    TZhs& zhs_;
    bool radioEnabled_{};
    bool gpuRadioEnabled_{};
    CorsikaOutputSinkStatistics statistics_{};
  };
} // namespace corsika::kokkos

#include <corsika/detail/kokkos/output/CorsikaOutputSink.inl>

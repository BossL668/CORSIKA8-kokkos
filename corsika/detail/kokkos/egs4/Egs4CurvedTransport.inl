/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once
#include <corsika/detail/kokkos/egs4/Egs4CurvedFrame.inl>
#include <corsika/detail/kokkos/egs4/Egs4History.inl>

namespace corsika::kokkos::egs4 {
  struct CurvedConfig {
    bool flatOutput_{true};
    double earthRadiusCm_{}, observationHeightCm_{}, minimumCosine_{-1.};
    bool directAzimuth_{};
  };
  // Per-history observer coordinates must survive suspension and be inherited
  // by all children. They are not derivable by relabelling the surface-arc x/y.
  struct CurvedTrack {
    TrackState particle_;
    Point observerCm_;
    double wa_{1.}, wap_{};
    bool detectorLocal_{};
  };
  struct CurvedFrameStep {
    CurvedConfig config_;
    CurvedTrack previous_;
    CurvedResult* result_; // private to the calling particle/thread, never shared
    KOKKOS_INLINE_FUNCTION FrameStepResult
    operator()(TrackState state, Point advanced, Direction direction, double geometric_cm,
               double time_path_cm, double mass_MeV, double speed_cm_s) const {
      CurvedInput q;
      q.photon_ = state.pdg_ == 22;
      q.detectorLocal_ = previous_.detectorLocal_;
      q.flatOutput_ = config_.flatOutput_;
      q.previousLocal_ = state.positionCm_;
      q.previousObserver_ = previous_.observerCm_;
      q.advancedLocal_ = advanced;
      q.advancedDirection_ = direction;
      q.earthRadiusCm_ = config_.earthRadiusCm_;
      q.observationHeightCm_ = config_.observationHeightCm_;
      q.minimumCosine_ = config_.minimumCosine_;
      q.geometricStepCm_ = geometric_cm;
      q.timePathCm_ = time_path_cm;
      q.totalEnergyMeV_ = state.energyMeV_;
      q.massMeV_ = mass_MeV;
      q.speedCmS_ = speed_cm_s;
      q.directAzimuth_ = config_.directAzimuth_;
      *result_ = rebase_curved_step(q);
      return {result_->status_ == CurvedStatus::advanced, result_->local_,
              result_->direction_, result_->timeIncrementS_};
    }
  };
  struct CurvedHistoryOutcome {
    HistoryOutcome history_;
    CurvedTrack particle_, children_[2];
    CurvedResult frame_;
    bool frameApplied_{};
  };
  // Normal air-step composition only. An early CURVED angle/observation exit is
  // exposed in frame.status_ and history.error_; it is NOT a normal track and has
  // no committed loss or children. Resolving C7's label-420/498 endpoints is a
  // separate integration task. Never retry a failed call with an advanced RNG.
  template <class TRandom>
  KOKKOS_INLINE_FUNCTION CurvedHistoryOutcome
  finish_curved_history(ModelView model, HistoryPlan plan, CurvedTrack previous,
                        AirRegion region, MagneticField field, TransportSettings settings,
                        BoundaryLimit limit, CurvedConfig config, TRandom& rng) {
    CurvedHistoryOutcome r;
    r.particle_ = previous;
    CurvedFrameStep frame{config, previous, &r.frame_};
    r.history_ = finish_history(model, plan, region, field, settings, limit, rng, frame);
    if (r.history_.action_ == HistoryAction::error) return r;
    r.frameApplied_ = r.history_.hasTrack_;
    r.particle_.particle_ = r.history_.particle_;
    if (r.frameApplied_) {
      r.particle_.observerCm_ = r.frame_.observer_;
      if (!previous.detectorLocal_ || !config.flatOutput_) {
        r.particle_.wa_ = r.frame_.wa_;
        r.particle_.wap_ = r.frame_.wap_;
      }
    }
    for (int j = 0; j < r.history_.childCount_; ++j) {
      r.children_[j] = r.particle_;
      r.children_[j].particle_ = r.history_.children_[j];
    }
    return r;
  }
  static_assert(std::is_trivially_copyable_v<CurvedTrack>);
} // namespace corsika::kokkos::egs4

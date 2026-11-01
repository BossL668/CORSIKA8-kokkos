/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once
#include <corsika/detail/kokkos/egs4/Egs4C8Air.inl>

namespace corsika::kokkos::egs4::c8_adapter {
  struct AirOutput {
    bool valid_{}, hasStep_{}, hasRadio_{}, hasObservation_{};
    ::corsika::kokkos::StepRecord step_;
    ::corsika::kokkos::RadioTrackRecord radio_;
    ::corsika::kokkos::ObservationRecord observation_;
    AirDiscard discard_{AirDiscard::none};
    double discardedEnergyGeV_{}; // unweighted, separate from atmospheric deposition
    Channel channel_{Channel::invalid};
  };
  KOKKOS_INLINE_FUNCTION Point observer_to_global(Point observer, Frame const& frame) {
    auto d = to_global(frame, {observer.x_ * .01, observer.y_ * .01, observer.z_ * .01});
    return {frame.originM_.x_ + d.x_, frame.originM_.y_ + d.y_, frame.originM_.z_ + d.z_};
  }
  // Only a freshly exported position is reused. Imported Cartesian data can
  // differ by roundoff from the native observer-coordinate representation.
  KOKKOS_INLINE_FUNCTION Point output_position(AirRecord const& p,
                                               AirEnvironment const& e) {
    if ((e.overheadMask_ & overhead::coordinate_reuse) && p.exportedPosition_)
      return {p.metadata_.positionM_[0], p.metadata_.positionM_[1],
              p.metadata_.positionM_[2]};
    return observer_to_global(p.track_.observerCm_, e.observerFrame_);
  }
  // C8's existing onStep/onRadioTrack/onObservation record schema, with explicit
  // native channel metadata outside the PROPOSAL process-ID namespace.
  // C7 CoREAS AddTrack derives velocity from its pre/post Cartesian endpoints;
  // do not feed the final MSCAT-rotated continuation direction into radio.
  KOKKOS_INLINE_FUNCTION AirOutput make_air_output(AirRecord const& before,
                                                   AirRecord const& after,
                                                   AirOutcome const& outcome,
                                                   AirEnvironment const& e) {
    AirOutput output;
    if (outcome.action_ == AirAction::error) return output;
    auto const& h = outcome.history_;
    if (!finite(before.metadata_.weight_) || before.metadata_.weight_ < 0. ||
        !finite(h.continuousDepositMeV_) || h.continuousDepositMeV_ < 0. ||
        !finite(h.vertexDepositMeV_) || h.vertexDepositMeV_ < 0. ||
        !finite(outcome.discardedEnergyMeV_) || outcome.discardedEnergyMeV_ < 0.)
      return output;
    output.discard_ = outcome.discard_;
    output.discardedEnergyGeV_ = outcome.discardedEnergyMeV_ * .001;
    output.channel_ = h.channel_;
    output.hasStep_ = h.hasTrack_ || h.vertexDepositMeV_ > 0.;
    if (output.hasStep_) {
      auto start = output_position(before, e);
      auto end = h.hasTrack_ ? output_position(after, e) : start;
      auto& s = output.step_;
      s.historyId_ = before.metadata_.historyId_;
      s.stepId_ = before.metadata_.stepId_;
      s.pid_ = before.metadata_.pid_;
      s.processId_ =
          0; // geometric/deposition record, not a fictitious PROPOSAL interaction
      s.startPositionM_[0] = start.x_;
      s.startPositionM_[1] = start.y_;
      s.startPositionM_[2] = start.z_;
      s.endPositionM_[0] = end.x_;
      s.endPositionM_[1] = end.y_;
      s.endPositionM_[2] = end.z_;
      s.startTimeS_ = before.track_.particle_.timeS_;
      s.endTimeS_ = after.track_.particle_.timeS_;
      s.startEnergyGeV_ = before.track_.particle_.energyMeV_ * .001;
      s.endEnergyGeV_ = h.particle_.energyMeV_ * .001;
      s.depositedEnergyGeV_ = (h.continuousDepositMeV_ + h.vertexDepositMeV_) * .001;
      // The sink treats this field as endpoint deposition, included in total.
      s.cutDepositedEnergyGeV_ = h.vertexDepositMeV_ * .001;
      s.weight_ = before.metadata_.weight_;
      if (!valid_point(start) || !valid_point(end) || !finite(s.endTimeS_) ||
          s.endTimeS_ < s.startTimeS_)
        return output;
      Direction chord{end.x_ - start.x_, end.y_ - start.y_, end.z_ - start.z_};
      double distance = ::sqrt(dot(chord, chord));
      output.hasRadio_ =
          h.hasTrack_ && s.pid_ != 22 && distance > 0. && s.endTimeS_ > s.startTimeS_;
      if (output.hasRadio_) {
        output.radio_.step_ = s;
        double u[] = {chord.x_ / distance, chord.y_ / distance, chord.z_ / distance};
        for (int j = 0; j < 3; ++j)
          output.radio_.startDirection_[j] = output.radio_.endDirection_[j] = u[j];
      }
    }
    output.hasObservation_ = outcome.action_ == AirAction::observed;
    if (output.hasObservation_) output.observation_.particle_ = after.metadata_;
    output.valid_ = true;
    return output;
  }
  // Targets the existing CorsikaOutputSink record API. test_c8_output also runs
  // the actual C8 writers and CoREAS/ZHS through the host-facing Session.
  // The caller separately handles child production and angular/escape loss.
  template <class TSink>
  void stream_air_output(AirOutput const& output, TSink& sink) {
    if (!output.valid_) throw std::runtime_error("Invalid native EGS4 output");
    if (output.hasStep_) sink.onStep(output.step_);
    if (output.hasRadio_) sink.onRadioTrack(output.radio_);
    if (output.hasObservation_) sink.onObservation(output.observation_);
  }
} // namespace corsika::kokkos::egs4::c8_adapter

/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

namespace corsika::kokkos::radio::firn {
  inline CascadeRadioAdapter::CascadeRadioAdapter(observers_type& observers,
                                                  process_type& process)
      : observers_(observers)
      , process_(&process) {}
  inline CascadeRadioAdapter::CascadeRadioAdapter(observers_type& emptyObservers)
      : observers_(emptyObservers) {
    if (observers_.size() != 0)
      throw std::invalid_argument("inactive radio slot must be empty");
  }
  template <typename TParticle>
  inline ProcessReturn CascadeRadioAdapter::doContinuous(Step<TParticle> const& step,
                                                         bool flag) {
    if (sink_) {
      auto const pid = step.getParticlePre().getPID();
      if (process_ && captureTrack_ && (pid == Code::Electron || pid == Code::Positron)) {
        using namespace corsika::units::si;
        corsika::radio::RadioTrackSegment segment{};
        segment.pdg_ = pid == Code::Electron ? 11 : -11;
        segment.weight_ = step.getParticlePre().getWeight();
        segment.startTimeS_ = step.getTimePre() / 1_s;
        segment.endTimeS_ = step.getTimePost() / 1_s;
        segment.startKineticEnergyGeV_ = step.getEkinPre() / 1_GeV;
        segment.endKineticEnergyGeV_ = step.getEkinPost() / 1_GeV;
        auto const start = step.getPositionPre().getCoordinates(sinkCoordinates_);
        auto const end = step.getPositionPost().getCoordinates(sinkCoordinates_);
        auto const before = step.getDirectionPre().getComponents(sinkCoordinates_);
        auto const after = step.getDirectionPost().getComponents(sinkCoordinates_);
        for (int axis = 0; axis < 3; ++axis) {
          segment.startPositionM_[axis] = start[axis] / 1_m;
          segment.endPositionM_[axis] = end[axis] / 1_m;
          segment.startDirection_[axis] = before[axis].magnitude();
          segment.endDirection_[axis] = after[axis].magnitude();
        }
        sink_->append(segment);
      }
      return ProcessReturn::Ok;
    }
    return process_ ? process_->doContinuous(step, flag) : ProcessReturn::Ok;
  }
  template <typename TParticle, typename TTrack>
  inline LengthType CascadeRadioAdapter::getMaxStepLength(TParticle const&,
                                                          TTrack const&) const {
    return std::numeric_limits<double>::infinity() * corsika::units::si::meter;
  }
  inline CascadeRadioAdapter::observers_type& CascadeRadioAdapter::observerCollection() {
    return observers_;
  }
  inline bool CascadeRadioAdapter::hasTrackSink() const { return sink_ != nullptr; }
  inline void CascadeRadioAdapter::setTrackSink(corsika::radio::RadioTrackSink* sink,
                                                CoordinateSystemPtr const& coordinates,
                                                bool capture) {
    if (sink && (!coordinates || sink_))
      throw std::invalid_argument("invalid or nested firn track sink");
    sink_ = sink;
    sinkCoordinates_ = coordinates;
    captureTrack_ = capture;
  }
  inline void CascadeRadioAdapter::replayTracks(std::vector<std::string> const& paths,
                                                CoordinateSystemPtr const& coordinates) {
    if (paths.empty()) return;
    if (!process_ || sink_)
      throw std::logic_error("firn replay requires an active receiver");
    process_->prepareReplay(paths, coordinates);
  }
  inline void replay_prefix_radio(CascadeRadioAdapter& radio,
                                  std::vector<std::string> const& paths,
                                  CoordinateSystemPtr const& coordinates) {
    radio.replayTracks(paths, coordinates);
  }
} // namespace corsika::kokkos::radio::firn

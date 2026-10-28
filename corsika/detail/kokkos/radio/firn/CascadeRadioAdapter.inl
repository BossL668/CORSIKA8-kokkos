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
  inline bool CascadeRadioAdapter::hasTrackSink() const { return false; }
  inline void CascadeRadioAdapter::setTrackSink(corsika::radio::RadioTrackSink* sink,
                                                CoordinateSystemPtr const&, bool) {
    if (sink)
      throw std::invalid_argument("deferred firn radio capture is not yet connected");
  }
} // namespace corsika::kokkos::radio::firn

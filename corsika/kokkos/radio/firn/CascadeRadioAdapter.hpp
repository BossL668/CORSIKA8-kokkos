/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/kokkos/radio/firn/VariableIndexRadioProcess.hpp>
#include <corsika/kokkos/radio/firn/AnyGreenBackend.hpp>
#include <corsika/modules/radio/detectors/ObserverCollection.hpp>
#include <corsika/modules/radio/observers/TimeDomainObserver.hpp>
#include <corsika/modules/radio/RadioTrackSink.hpp>

namespace corsika::kokkos::radio::firn {
  /** Present the firn radio process through the shared cascade radio interface.
   * The first slot forwards tracks to the coherent variable-index response;
   * an empty second slot performs no calculation. Observer descriptors provide
   * geometry to the common runner, not a second set of uniform-ice waveforms.
   * Spectrum and waveform output is owned by VariableIndexRadioProcess.
   */
  class CascadeRadioAdapter : public ContinuousProcess<CascadeRadioAdapter> {
  public:
    using process_type = VariableIndexRadioProcess<AnyGreenBackend>;
    using observers_type = ObserverCollection<TimeDomainObserver>;
    CascadeRadioAdapter(observers_type& observers, process_type& process);
    explicit CascadeRadioAdapter(observers_type& emptyObservers);
    template <typename TParticle>
    ProcessReturn doContinuous(Step<TParticle> const& step, bool flag);
    template <typename TParticle, typename TTrack>
    LengthType getMaxStepLength(TParticle const&, TTrack const&) const;
    observers_type& observerCollection();
    bool hasTrackSink() const;
    void setTrackSink(corsika::radio::RadioTrackSink* sink, CoordinateSystemPtr const&,
                      bool = true);
    void replayTracks(std::vector<std::string> const&, CoordinateSystemPtr const&);

  private:
    observers_type& observers_;
    process_type* process_{};
    corsika::radio::RadioTrackSink* sink_{};
    CoordinateSystemPtr sinkCoordinates_;
    bool captureTrack_{true};
  };
  /// Found by argument-dependent lookup in the common cascade runner.
  void replay_prefix_radio(CascadeRadioAdapter&, std::vector<std::string> const&,
                           CoordinateSystemPtr const&);
} // namespace corsika::kokkos::radio::firn
#include <corsika/detail/kokkos/radio/firn/CascadeRadioAdapter.inl>

/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */
#pragma once

#include <corsika/modules/radio/RadioTrack.hpp>
#include <corsika/accelerator/em/common/Types.hpp>
#include <corsika/accelerator/em/common/TransportMass.hpp>

#include <stdexcept>

namespace corsika::accelerator::radio {

  inline gpu::em::LeptonTransportRecord makeHostRadioRecord(
      corsika::radio::RadioTrackSegment const& segment) {
    if (!gpu::em::isElectronOrPositronPid(segment.pdg)) {
      throw std::invalid_argument("only electron/positron tracks enter radio projection");
    }
    gpu::em::LeptonTransportRecord record{};
    record.start.pid = record.end.pid = segment.pdg;
    record.start.weight = record.end.weight = segment.weight;
    record.start.time_s = segment.start_time_s;
    record.end.time_s = segment.end_time_s;
    auto const mass = gpu::em::transportMassGeV(segment.pdg);
    record.start.energy_GeV = segment.start_kinetic_energy_GeV + mass;
    record.end.energy_GeV = segment.end_kinetic_energy_GeV + mass;
    for (int axis = 0; axis < 3; ++axis) {
      record.start.position_m[axis] = segment.start_position_m[axis];
      record.end.position_m[axis] = segment.end_position_m[axis];
      record.start.direction[axis] = segment.start_direction[axis];
      record.end.direction[axis] = segment.end_direction[axis];
    }
    return record;
  }

  /** The backend batches these records independently of the transport queue. */
  template <typename TBackend>
  class HostRadioCollector final : public corsika::radio::RadioTrackSink {
  public:
    explicit HostRadioCollector(TBackend& backend) : backend_(backend) {
      if (!backend_.gpuRadioEnabled()) {
        throw std::logic_error("cannot redirect scalar radio to a disabled backend");
      }
    }

    void append(corsika::radio::RadioTrackSegment const& segment) override {
      auto const record = makeHostRadioRecord(segment);
      backend_.enqueueRadioTracks(&record, 1);
    }

  private:
    TBackend& backend_;
  };

  /** Event-local routing: restored even when transport or projection throws. */
  template <typename TCoreas, typename TZhs>
  class ScopedRadioTrackSink {
  public:
    template <typename TCoordinates>
    ScopedRadioTrackSink(TCoreas& coreas, TZhs& zhs,
                         corsika::radio::RadioTrackSink& sink,
                         TCoordinates const& coordinates)
        : coreas_(coreas), zhs_(zhs) {
      if (coreas_.hasTrackSink() || zhs_.hasTrackSink()) {
        throw std::logic_error("scalar radio already has a track sink");
      }
      // The common accelerated projector handles both enabled models. Even
      // though both processes visit the step, capture its geometry only once.
      coreas_.setTrackSink(&sink, coordinates, true);
      try {
        zhs_.setTrackSink(&sink, coordinates, coreas_.observerCollection().size() == 0);
      } catch (...) {
        coreas_.setTrackSink(nullptr, {});
        throw;
      }
    }

    ~ScopedRadioTrackSink() {
      coreas_.setTrackSink(nullptr, {});
      zhs_.setTrackSink(nullptr, {});
    }

    ScopedRadioTrackSink(ScopedRadioTrackSink const&) = delete;
    ScopedRadioTrackSink& operator=(ScopedRadioTrackSink const&) = delete;

  private:
    TCoreas& coreas_;
    TZhs& zhs_;
  };

} // namespace corsika::accelerator::radio

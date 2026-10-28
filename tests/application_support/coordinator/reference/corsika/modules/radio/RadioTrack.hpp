/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <cstdint>

namespace corsika::radio {

  /** A completed scalar step, expressed in the receiver's coordinate system.
   *
   * This is a radio input, not a particle to be transported a second time.
   * Capturing it must neither change the step nor consume random numbers.
   */
  struct RadioTrackSegment {
    double start_position_m[3]{};
    double end_position_m[3]{};
    double start_direction[3]{};
    double end_direction[3]{};
    double start_time_s{};
    double end_time_s{};
    double start_kinetic_energy_GeV{};
    double end_kinetic_energy_GeV{};
    double weight{};
    std::int32_t pdg{};
  };

  /** Destination for deferred projection of a scalar electron/positron step. */
  class RadioTrackSink {
  public:
    virtual ~RadioTrackSink() = default;
    virtual void append(RadioTrackSegment const& segment) = 0;
  };

} // namespace corsika::radio

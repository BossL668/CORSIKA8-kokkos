/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <cstdint>

namespace corsika::radio {
  /** Completed scalar step in one common coordinate system. Positions are metres,
   * times seconds and energies kinetic GeV. Recording never advances particles
   * or draws random numbers. Field order follows the production contract. */
  struct RadioTrackSegment {
    double startPositionM_[3]{};
    double endPositionM_[3]{};
    double startDirection_[3]{};
    double endDirection_[3]{};
    double startTimeS_{};
    double endTimeS_{};
    double startKineticEnergyGeV_{};
    double endKineticEnergyGeV_{};
    double weight_{};
    std::int32_t pdg_{};
  };
} // namespace corsika::radio

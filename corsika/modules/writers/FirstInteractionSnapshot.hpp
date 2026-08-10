/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <vector>

#include <corsika/framework/core/ParticleProperties.hpp>
#include <corsika/framework/core/PhysicalGeometry.hpp>
#include <corsika/framework/core/PhysicalUnits.hpp>
#include <corsika/framework/geometry/Point.hpp>

namespace corsika {

  /** Lightweight, output-format-independent secondary at one interaction. */
  struct InteractionSecondarySnapshot {
    Code pid;
    HEPEnergyType total_energy;
    DirectionVector direction;
  };

  /**
   * Output-format-independent first-interaction state shared by scalar Stack
   * views and accelerator output adapters.
   */
  struct FirstInteractionSnapshot {
    Code parent_pid;
    HEPEnergyType parent_kinetic_energy;
    Point position;
    DirectionVector direction;
    TimeType time;
    std::vector<InteractionSecondarySnapshot> secondaries;
  };

} // namespace corsika

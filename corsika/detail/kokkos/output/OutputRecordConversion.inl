/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

namespace corsika::kokkos::output_detail {
  inline Code code_from_device_pid(std::int32_t pid) {
    return convert_from_PDG(static_cast<PDGCode>(pid));
  }

  inline Point point_from_array(CoordinateSystemPtr const& coordinate_system,
                                double const values_m[3]) {
    return Point{coordinate_system, values_m[0] * meter, values_m[1] * meter,
                 values_m[2] * meter};
  }

  inline DirectionVector direction_from_array(
      CoordinateSystemPtr const& coordinate_system, double const values[3]) {
    DirectionVector direction{coordinate_system, {values[0], values[1], values[2]}};
    auto const norm = direction.getNorm();
    if (!std::isfinite(norm) || !(norm > 0.)) {
      throw std::runtime_error("GPU EM output contains an invalid particle direction");
    }
    return direction.normalized();
  }

  inline RadioTrackParticle::RadioTrackParticle(Code pid, HEPEnergyType kinetic_energy,
                                                DirectionVector direction, Point position,
                                                TimeType time, double weight)
      : pid_(pid)
      , kineticEnergy_(kinetic_energy)
      , direction_(std::move(direction))
      , position_(std::move(position))
      , time_(time)
      , weight_(weight) {}

  inline Code RadioTrackParticle::getPID() const { return pid_; }

  inline HEPEnergyType RadioTrackParticle::getKineticEnergy() const {
    return kineticEnergy_;
  }

  inline DirectionVector const& RadioTrackParticle::getDirection() const {
    return direction_;
  }

  inline Point const& RadioTrackParticle::getPosition() const { return position_; }

  inline TimeType RadioTrackParticle::getTime() const { return time_; }

  inline double RadioTrackParticle::getWeight() const { return weight_; }
} // namespace corsika::kokkos::output_detail

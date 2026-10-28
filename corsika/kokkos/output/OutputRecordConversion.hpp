/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once
#include <corsika/framework/core/ParticleProperties.hpp>
#include <corsika/framework/core/PhysicalGeometry.hpp>
#include <corsika/framework/core/PhysicalUnits.hpp>
#include <corsika/framework/geometry/Point.hpp>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <utility>

namespace corsika::kokkos::output_detail {
  inline Code code_from_device_pid(std::int32_t pid);

  inline Point point_from_array(CoordinateSystemPtr const& coordinate_system,
                                double const values_m[3]);

  inline DirectionVector direction_from_array(
      CoordinateSystemPtr const& coordinate_system, double const values[3]);

  class RadioTrackParticle {
  public:
    RadioTrackParticle(Code pid, HEPEnergyType kinetic_energy, DirectionVector direction,
                       Point position, TimeType time, double weight);

    Code getPID() const;
    HEPEnergyType getKineticEnergy() const;
    DirectionVector const& getDirection() const;
    Point const& getPosition() const;
    TimeType getTime() const;
    double getWeight() const;

  private:
    Code pid_;
    HEPEnergyType kineticEnergy_;
    DirectionVector direction_;
    Point position_;
    TimeType time_;
    double weight_;
  };
} // namespace corsika::kokkos::output_detail

#include <corsika/detail/kokkos/output/OutputRecordConversion.inl>

/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/framework/core/PhysicalUnits.hpp>
#include <corsika/framework/geometry/Box.hpp>
#include <corsika/framework/geometry/SeparationPlane.hpp>
#include <corsika/framework/geometry/Sphere.hpp>
#include <corsika/media/Universe.hpp>

#include <cmath>
#include <stdexcept>
#include <typeinfo>

namespace corsika::kokkos {

  namespace shape_conversion {

    inline double finite_value(double const value) {
      if (!std::isfinite(value)) {
        throw std::invalid_argument("Shape export requires finite geometry parameters");
      }
      return value;
    }

    inline void require_coordinate_system(CoordinateSystemPtr const& coordinateSystem) {
      if (!coordinateSystem) {
        throw std::invalid_argument("Shape export requires a coordinate system");
      }
    }

    inline void copy_point(Point const& point,
                           CoordinateSystemPtr const& coordinateSystem,
                           double (&destination)[3]) {
      require_coordinate_system(point.getCoordinateSystem());
      auto const coordinates = point.getCoordinates(coordinateSystem);
      for (int component = 0; component < 3; ++component) {
        destination[component] = finite_value(coordinates[component] / meter);
      }
    }

    inline void copy_direction(DirectionVector const& direction,
                               CoordinateSystemPtr const& coordinateSystem,
                               double (&destination)[3]) {
      require_coordinate_system(direction.getCoordinateSystem());
      auto const components = direction.getComponents(coordinateSystem);
      for (int component = 0; component < 3; ++component) {
        destination[component] = finite_value(components[component].magnitude());
      }
    }

  } // namespace shape_conversion

  inline EnvironmentShape export_shape(IVolume const& volume,
                                       CoordinateSystemPtr const& coordinateSystem) {
    shape_conversion::require_coordinate_system(coordinateSystem);
    EnvironmentShape shape;
    if (typeid(volume) == typeid(Universe)) {
      auto const& universe = static_cast<Universe const&>(volume);
      shape_conversion::require_coordinate_system(universe.getCoordinateSystem());
      shape.kind_ = EnvironmentShape::Kind::Universe;
    } else if (typeid(volume) == typeid(Sphere)) {
      auto const& sphere = static_cast<Sphere const&>(volume);
      shape.kind_ = EnvironmentShape::Kind::Sphere;
      shape_conversion::copy_point(sphere.getCenter(), coordinateSystem, shape.centerM_);
      shape.radiusM_ = shape_conversion::finite_value(sphere.getRadius() / meter);
    } else if (typeid(volume) == typeid(Box)) {
      auto const& box = static_cast<Box const&>(volume);
      auto const boxSystem = box.getCoordinateSystem();
      shape_conversion::require_coordinate_system(boxSystem);
      shape.kind_ = EnvironmentShape::Kind::Box;
      // Native Box::contains uses the current frame, also after Box::rotate.
      shape_conversion::copy_point(Point(boxSystem, 0_m, 0_m, 0_m), coordinateSystem,
                                   shape.centerM_);
      shape_conversion::copy_direction(DirectionVector(boxSystem, {1., 0., 0.}),
                                       coordinateSystem, shape.axes_[0]);
      shape_conversion::copy_direction(DirectionVector(boxSystem, {0., 1., 0.}),
                                       coordinateSystem, shape.axes_[1]);
      shape_conversion::copy_direction(DirectionVector(boxSystem, {0., 0., 1.}),
                                       coordinateSystem, shape.axes_[2]);
      shape.halfExtentM_[0] = shape_conversion::finite_value(box.getX() / meter);
      shape.halfExtentM_[1] = shape_conversion::finite_value(box.getY() / meter);
      shape.halfExtentM_[2] = shape_conversion::finite_value(box.getZ() / meter);
    } else if (typeid(volume) == typeid(SeparationPlane)) {
      auto const& separation = static_cast<SeparationPlane const&>(volume);
      auto const plane = separation.getPlane();
      shape.kind_ = EnvironmentShape::Kind::SeparationPlane;
      shape_conversion::copy_point(plane.getCenter(), coordinateSystem, shape.centerM_);
      shape_conversion::copy_direction(plane.getNormal(), coordinateSystem,
                                       shape.normal_);
    } else {
      throw std::invalid_argument("Unsupported native volume in shape export");
    }
    return shape;
  }

} // namespace corsika::kokkos

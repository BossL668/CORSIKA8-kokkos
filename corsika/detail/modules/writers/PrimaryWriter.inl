/*
 * (c) Copyright 2020 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/framework/core/ParticleProperties.hpp>

namespace corsika {

  template <typename TTracking, typename TOutput>
  inline PrimaryWriter<TTracking, TOutput>::PrimaryWriter(
      ObservationPlane<TTracking, TOutput> const& obsPlane)
      : obsPlane_(obsPlane)
      , writtenThisShower_(false)
      , showerId_(0) {}

  template <typename TTracking, typename TOutput>
  inline PrimaryWriter<TTracking, TOutput>::~PrimaryWriter() {}

  template <typename TTracking, typename TOutput>
  inline void PrimaryWriter<TTracking, TOutput>::recordPrimary(
      std::tuple<Code, HEPEnergyType, DirectionVector, Point, TimeType> const&
          primaryProps) {
    auto const [pID, kineticEnergy, direction, position, time] = primaryProps;

    Vector const displacement = position - obsPlane_.getPlane().getCenter();

    auto const x = displacement.dot(obsPlane_.getXAxis());
    auto const y = displacement.dot(obsPlane_.getYAxis());
    auto const z = displacement.dot(obsPlane_.getPlane().getNormal());
    auto const nx = direction.dot(obsPlane_.getXAxis());
    auto const ny = direction.dot(obsPlane_.getYAxis());
    auto const nz = direction.dot(obsPlane_.getPlane().getNormal());

    output_.event(showerId_)["pdg"] =
        static_cast<int>(get_PDG(pID));
    output_.event(showerId_)["name"] =
        static_cast<std::string>(get_name(pID));
    output_.event(showerId_)["total_energy"] =
        (kineticEnergy + get_mass(pID)) / 1_GeV;
    output_.event(showerId_)["kinetic_energy"] =
        kineticEnergy / 1_GeV;
    output_.event(showerId_)["x"] = x / 1_m;
    output_.event(showerId_)["y"] = y / 1_m;
    output_.event(showerId_)["z"] = z / 1_m;
    output_.event(showerId_)["nx"] = static_cast<double>(nx);
    output_.event(showerId_)["ny"] = static_cast<double>(ny);
    output_.event(showerId_)["nz"] = static_cast<double>(nz);
    output_.event(showerId_)["time"] = time / 1_s;
  }

  template <typename TTracking, typename TOutput>
  inline void PrimaryWriter<TTracking, TOutput>::startOfLibrary([
      [maybe_unused]] boost::filesystem::path const& directory) {
    output_.open(directory);
    showerId_ = 0;
  }

  template <typename TTracking, typename TOutput>
  inline void PrimaryWriter<TTracking, TOutput>::endOfShower([
      [maybe_unused]] unsigned int const showerId) {
    output_.flush();
    ++showerId_;
  }

  template <typename TTracking, typename TOutput>
  inline void PrimaryWriter<TTracking, TOutput>::endOfLibrary() {}

  template <typename TTracking, typename TOutput>
  inline YAML::Node PrimaryWriter<TTracking, TOutput>::getConfig() const {
    YAML::Node node;
    node["type"] = "PrimaryParticle";

    node["units"]["length"] = "m"; // add default units for values
    node["units"]["energy"] = "GeV";
    node["units"]["time"] = "s";

    // the center of the plane
    auto const center{obsPlane_.getPlane().getCenter()};

    // save each component in its native coordinate system
    auto const center_coords{center.getCoordinates(center.getCoordinateSystem())};
    node["plane"]["center"].push_back(center_coords.getX() / 1_m);
    node["plane"]["center"].push_back(center_coords.getY() / 1_m);
    node["plane"]["center"].push_back(center_coords.getZ() / 1_m);

    auto const normal{obsPlane_.getPlane().getNormal().getComponents()};
    node["plane"]["normal"].push_back(normal.getX().magnitude());
    node["plane"]["normal"].push_back(normal.getY().magnitude());
    node["plane"]["normal"].push_back(normal.getZ().magnitude());

    auto const xAxis_coords{
        obsPlane_.getXAxis().getComponents(obsPlane_.getXAxis().getCoordinateSystem())};
    node["x-axis"].push_back(xAxis_coords.getX().magnitude());
    node["x-axis"].push_back(xAxis_coords.getY().magnitude());
    node["x-axis"].push_back(xAxis_coords.getZ().magnitude());

    auto const yAxis_coords{
        obsPlane_.getYAxis().getComponents(obsPlane_.getYAxis().getCoordinateSystem())};
    node["y-axis"].push_back(yAxis_coords.getX().magnitude());
    node["y-axis"].push_back(yAxis_coords.getY().magnitude());
    node["y-axis"].push_back(yAxis_coords.getZ().magnitude());

    return node;
  }

  template <typename TTracking, typename TOutput>
  inline YAML::Node PrimaryWriter<TTracking, TOutput>::getSummary() const {
    return output_.snapshot();
  }

} // namespace corsika

// Uniform local IGRF field, evaluated just as in c8_air_shower, but expressed
// in the terrain scene's geographic ENU axes. No CLI or YAML dependency.
#pragma once
#include <corsika/media/GeomagneticModel.hpp>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
namespace corsika::terrain {
inline std::array<double,3> magneticNwuToEnu(std::array<double,3> const& nwu) {
  return {-nwu[1],nwu[0],nwu[2]};
}
inline std::array<double,3> igrf14AirFieldEnu(CoordinateSystemPtr const& cs,
    std::filesystem::path const& coefficients,double year,double ellipsoidal_height_m,
    double latitude_deg,double longitude_deg) {
  if(!std::isfinite(year)||year<2025.||year>2030.||
     !std::isfinite(ellipsoidal_height_m)||!std::isfinite(latitude_deg)||std::abs(latitude_deg)>=90.||
     !std::isfinite(longitude_deg)||std::abs(longitude_deg)>180.)
    throw std::invalid_argument("terrain IGRF14 requires 2025..2030 and finite geographic coordinates");
  if(!std::filesystem::is_regular_file(coefficients))throw std::invalid_argument("IGRF14 coefficient file is missing");
  // Avoid labelling an older COF as IGRF14 and extrapolating it beyond its epoch.
  std::ifstream file(coefficients);std::string line;bool epoch2025=false;
  while(std::getline(file,line)) {
    std::istringstream row(line);std::string name;double epoch=0.;
    if(row>>name>>epoch)epoch2025|=name=="IGRF2025"&&epoch==2025.;
  }
  if(!epoch2025)throw std::invalid_argument("IGRF14 coefficient file lacks its 2025 reference epoch");
  GeomagneticModel model(Point(cs,0_m,0_m,0_m),coefficients.string());
  // getField returns North, West, Up, not geographic ENU.
  // Its geodetic -> geocentric conversion uses the ellipsoid radius. Do not
  // confuse this height with the ASL coordinate used by the density profile.
  auto field=model.getField(year,ellipsoidal_height_m*1_m,latitude_deg,longitude_deg);
  return magneticNwuToEnu({field.getX(cs)/1_T,field.getY(cs)/1_T,field.getZ(cs)/1_T});
}
} // namespace corsika::terrain

// Application-level YAML admission; geometry/atmosphere classes remain YAML-free.
#pragma once
#include <corsika/modules/terrain/TerrainAtmosphere.hpp>
#include <corsika/accelerator/em/common/tables/Sha256.hpp>
#include <yaml-cpp/yaml.h>
#include <filesystem>
#include <fstream>
#include <set>

namespace corsika::applications::terrain {
using namespace corsika::units::si;
inline Point point(YAML::Node const& p,CoordinateSystemPtr const& cs) {
  if(!p.IsSequence()||p.size()!=3)throw std::invalid_argument("invalid ENU triple");
  for(auto const& v:p)if(!std::isfinite(v.as<double>()))throw std::invalid_argument("nonfinite ENU coordinate");
  return Point(cs,p[0].as<double>()*1_m,p[1].as<double>()*1_m,p[2].as<double>()*1_m);
}
struct Scene {
  YAML::Node yaml;
  corsika::terrain::TerrainAtmosphereConfig atmosphere;
  std::filesystem::path mesh;
  std::string mesh_hash;
  explicit Scene(std::filesystem::path const& file):yaml(YAML::LoadFile(file.string())),atmosphere{0.} {
    if(yaml["schema_version"].as<int>()!=1||yaml["site"]["coordinate_frame"].as<std::string>()!="geographic_ENU"||
       yaml["atmosphere"]["model"].as<std::string>()!="us_standard_bk"||
       yaml["geometry"]["kind"].as<std::string>()!="terrain_mesh")
      throw std::invalid_argument("requires versioned ENU terrain/USStdBK scene");
    auto material=yaml["geometry"]["material"].as<std::string>();
    if(material=="Water"||material=="Ice") {
      atmosphere.embedded_medium=material=="Water"?Medium::WaterLiquid:Medium::WaterIce;
      atmosphere.embedded_nuclei={Code::Hydrogen,Code::Oxygen};
      atmosphere.embedded_number_fractions={2./3.,1./3.};
    } else if(material!="SiO2")
      throw std::invalid_argument("unsupported scene material: use SiO2, Water or Ice; C++ interface accepts explicit material properties");
    auto site=yaml["site"];
    double lat=site["latitude_deg"].as<double>(),lon=site["longitude_deg"].as<double>();
    if(!std::isfinite(lat)||std::abs(lat)>90.||!std::isfinite(lon)||std::abs(lon)>180.)
      throw std::invalid_argument("invalid scene geographic origin");
    atmosphere.origin_altitude_asl_m=yaml["atmosphere"]["origin_altitude_asl_m"].as<double>();
    atmosphere.sea_level_refractive_index=yaml["atmosphere"]["sea_level_refractive_index"].as<double>();
    atmosphere.rock_density_g_cm3=yaml["geometry"]["rock_density_g_cm3"].as<double>();
    atmosphere.rock_refractive_index=yaml["geometry"]["rock_refractive_index"].as<double>();
    auto h=site["origin_ellipsoidal_height_m"].as<double>(),n=site["geoid_undulation_m"].as<double>();
    if(!std::isfinite(h)||!std::isfinite(n)||std::abs(h-n-atmosphere.origin_altitude_asl_m)>1.e-6)
      throw std::invalid_argument("ASL/ellipsoidal/geoid height mismatch");
    mesh=yaml["geometry"]["mesh_path"].as<std::string>();
    if(mesh.is_relative())mesh=std::filesystem::absolute(file).parent_path()/mesh;
    auto size=std::filesystem::file_size(mesh);
    if(size>128u*1024u*1024u)throw std::invalid_argument("mesh byte budget exceeded (128 MiB)");
    std::vector<std::uint8_t> bytes(size);std::ifstream input(mesh,std::ios::binary);
    if(!input.read(reinterpret_cast<char*>(bytes.data()),bytes.size()))throw std::runtime_error("cannot read full mesh");
    mesh_hash=gpu::em::tables::toHex(gpu::em::tables::sha256(bytes));
    if(mesh_hash!=yaml["provenance"]["mesh_sha256"].as<std::string>())throw std::invalid_argument("mesh SHA-256 mismatch");
  }
  auto build(corsika::terrain::Environment& env)const {
    auto cs=env.getCoordinateSystem();corsika::terrain::buildAtmosphere(env,atmosphere);
    double padding=yaml["geometry"]["boundary_padding_m"].as<double>();
    if(!std::isfinite(padding)||padding<=0.||padding>1.e-3)throw std::invalid_argument("invalid terrain padding");
    auto info=corsika::terrain::embedTerrain(env,atmosphere,
        corsika::terrain::loadValidatedTerrain(mesh.string(),cs,padding*1_m),
        point(yaml["geometry"]["rock_reference_enu_m"],cs));
    std::set<std::string> names;
    auto observers=yaml["radio"]["observers"];
    if(!observers.IsSequence()||observers.size()==0)throw std::invalid_argument("scene needs observer positions");
    for(auto const& obs:observers) {
      if(!names.insert(obs["name"].as<std::string>()).second)throw std::invalid_argument("duplicate observer name");
      auto p=point(obs["position_enu_m"],cs);
      auto* node=env.getUniverse()->getContainingNode(p);
      if(!node||!node->hasModelProperties()||dynamic_cast<corsika::terrain::TriangularMesh const*>(&node->getVolume()))
        throw std::invalid_argument("observer outside atmosphere or inside terrain");
    }
    return info;
  }
};
} // namespace corsika::applications::terrain

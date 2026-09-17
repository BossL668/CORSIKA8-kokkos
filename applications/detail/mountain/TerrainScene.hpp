// Application-level YAML admission; geometry/atmosphere classes remain YAML-free.
#pragma once
#include <corsika/modules/terrain/TerrainAtmosphere.hpp>
#include <corsika/accelerator/em/common/tables/Sha256.hpp>
#include <yaml-cpp/yaml.h>
#include <filesystem>
#include <fstream>
#include <set>
#include "TerrainMaterialConfig.hpp"
#include <corsika/geometry/terrain/DemCoverageData.hpp>
#include <corsika/geometry/terrain/FlatTerrainExport.hpp>

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
  interfaces::MaterialModel material;
  bool legacy_material_overrides{false};
  bool dem_coverage_enabled{false};
  corsika::terrain::coverage::Data coverage;
  explicit Scene(std::filesystem::path const& file):yaml(YAML::LoadFile(file.string())),atmosphere{0.} {
    if(yaml["schema_version"].as<int>()!=1||yaml["site"]["coordinate_frame"].as<std::string>()!="geographic_ENU"||
       yaml["atmosphere"]["model"].as<std::string>()!="us_standard_bk"||
       yaml["geometry"]["kind"].as<std::string>()!="terrain_mesh")
      throw std::invalid_argument("requires versioned ENU terrain/USStdBK scene");
    auto geometry=yaml["geometry"];
    if(auto boundary=geometry["transport_boundary"]) {
      auto type=boundary["type"].as<std::string>();
      if(type!="dem_coverage"&&type!="none")throw std::invalid_argument("unknown terrain transport boundary");
      dem_coverage_enabled=type=="dem_coverage";
    }
    if(geometry["material_file"]) {
      if(geometry["material"]) throw std::invalid_argument("choose material or material_file, not both");
      std::filesystem::path path=geometry["material_file"].as<std::string>();
      if(path.is_relative()) path=std::filesystem::absolute(file).parent_path()/path;
      material=readMaterial(YAML::LoadFile(path.string()));
    } else material=readMaterial(geometry["material"]);
    auto legacy=[&](YAML::Node const& value,double& destination,double scale) {
      if(!value) return;
      double v=value.as<double>()*scale;
      // New map/file cards have one owner. Redundant equal legacy fields are
      // accepted so scenes can be migrated without any ambiguity.
      if(v!=destination && (geometry["material_file"] || geometry["material"].IsMap()))
        throw std::invalid_argument("legacy material parameter conflicts with selected material card");
      if(v!=destination) legacy_material_overrides=true;
      destination=v;
    };
    legacy(geometry["rock_density_g_cm3"],material.density_kg_m3,1000.);
    legacy(geometry["rock_refractive_index"],material.refractive_index,1.);
    if(geometry["attenuation_length_m"])
      throw std::invalid_argument("geometry.attenuation_length_m was not a radio input; use material.radio.field_attenuation_length_m");
    legacy(yaml["radio"]["rock_attenuation_length_m"],material.field_attenuation_length_m,1.);
    material.transportProperties();
    atmosphere.material=std::make_shared<interfaces::MaterialModel const>(material);
    atmosphere.embedded_medium=material.reference_medium;
    auto composition=material.composition();
    atmosphere.embedded_nuclei=composition.getComponents();
    atmosphere.embedded_number_fractions=composition.getFractions();
    auto site=yaml["site"];
    double lat=site["latitude_deg"].as<double>(),lon=site["longitude_deg"].as<double>();
    if(!std::isfinite(lat)||std::abs(lat)>90.||!std::isfinite(lon)||std::abs(lon)>180.)
      throw std::invalid_argument("invalid scene geographic origin");
    atmosphere.origin_altitude_asl_m=yaml["atmosphere"]["origin_altitude_asl_m"].as<double>();
    atmosphere.sea_level_refractive_index=yaml["atmosphere"]["sea_level_refractive_index"].as<double>();
    atmosphere.rock_density_g_cm3=material.density_kg_m3*.001;
    atmosphere.rock_refractive_index=material.refractive_index;
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
  auto build(corsika::terrain::Environment& env) {
    auto cs=env.getCoordinateSystem();corsika::terrain::buildAtmosphere(env,atmosphere);
    double padding=yaml["geometry"]["boundary_padding_m"].as<double>();
    if(!std::isfinite(padding)||padding<=0.||padding>1.e-3)throw std::invalid_argument("invalid terrain padding");
    auto info=corsika::terrain::embedTerrain(env,atmosphere,
        corsika::terrain::loadValidatedTerrain(mesh.string(),cs,padding*1_m),
        point(yaml["geometry"]["rock_reference_enu_m"],cs));
    if(dem_coverage_enabled)coverage=corsika::terrain::coverage::fromHeightfield(corsika::terrain::exportFlatTerrain(*info.mesh));
    std::set<std::string> names;
    auto observers=yaml["radio"]["observers"];
    if(!observers.IsSequence()||observers.size()==0)throw std::invalid_argument("scene needs observer positions");
    for(auto const& obs:observers) {
      if(!names.insert(obs["name"].as<std::string>()).second)throw std::invalid_argument("duplicate observer name");
      auto p=point(obs["position_enu_m"],cs);
      if(!corsika::terrain::coverage::contains(coverage.view(),{p.getX(cs)/1_m,p.getY(cs)/1_m,p.getZ(cs)/1_m}))
        throw std::invalid_argument("observer outside DEM coverage");
      auto* node=env.getUniverse()->getContainingNode(p);
      if(!node||!node->hasModelProperties()||dynamic_cast<corsika::terrain::TriangularMesh const*>(&node->getVolume()))
        throw std::invalid_argument("observer outside atmosphere or inside terrain");
    }
    return info;
  }
};
} // namespace corsika::applications::terrain

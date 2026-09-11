/* beta5 native atmosphere + geographic DEM/antenna admission probe.
 * This independent executable does not claim to simulate shower or radio. */
#include "detail/mountain/TerrainScene.hpp"
#include <CLI/CLI.hpp>
#include <iostream>
using namespace corsika;
using namespace corsika::units::si;
namespace terrainapp=corsika::applications::terrain;

int main(int argc,char** argv) {
  CLI::App app{"Validate DEM, native USStdBK atmosphere and ENU antennas (no shower)"};
  std::string configuration,output;
  app.add_option("--config",configuration)->required()->check(CLI::ExistingFile);
  app.add_option("--output",output)->required();
  CLI11_PARSE(app,argc,argv);
  try {
    if(std::filesystem::exists(output))throw std::runtime_error("refuse overwrite output");
    terrainapp::Scene scene(configuration);
    terrain::Environment env;auto cs=env.getCoordinateSystem();
    auto info=scene.build(env);
    YAML::Node result;
    result["complete"]=false;
    result["scope"]="geometry/material/antenna admission only; no shower or radio";
    result["mesh_vertices"]=info.vertices;result["mesh_triangles"]=info.triangles;
    result["mesh_sha256"]=scene.mesh_hash;
    result["minimum_altitude_asl_m"]=info.minimum_altitude_asl_m;
    result["maximum_altitude_asl_m"]=info.maximum_altitude_asl_m;
    result["origin_altitude_asl_m"]=scene.atmosphere.origin_altitude_asl_m;
    auto center=terrain::earthCenter(env,scene.atmosphere);
    for(auto const& observer:scene.yaml["radio"]["observers"]) {
      auto p=terrainapp::point(observer["position_enu_m"],cs);
      auto const* node=env.getUniverse()->getContainingNode(p);
      YAML::Node row;row["name"]=observer["name"];row["enu_m"]=observer["position_enu_m"];
      row["altitude_asl_m"]=terrain::altitude(p,center);
      row["density_kg_m3"]=node->getModelProperties().getMassDensity(p)/(1_kg/(1_m*1_m*1_m));
      row["refractive_index"]=node->getModelProperties().getRefractiveIndex(p);
      row["inside_rock"]=false;result["observers"].push_back(row);
    }
    result["complete"]=true;result["scene"]=scene.yaml;
    std::ofstream file(output);if(!file)throw std::runtime_error("cannot create output");
    file<<result<<'\n';if(!file)throw std::runtime_error("output write failure");
    std::cout<<"Validated "<<info.triangles<<" triangles, "<<result["observers"].size()<<" air observers\n";
    return 0;
  }catch(std::exception const& e){std::cerr<<e.what()<<'\n';return 1;}
}

#pragma once
#include "TerrainScene.hpp"
#include <corsika/modules/radio/interface/Types.hpp>
namespace corsika::applications::terrain {
inline corsika::radio::interface::Config makeTerrainRadioConfig(
    Scene const& scene,corsika::terrain::Environment const& env){
  namespace ri=corsika::radio::interface;
  ri::Config c;c.enabled=true;c.propagation.geometry=ri::Geometry::Mesh;
  auto input=scene.yaml["radio"];auto cs=env.getCoordinateSystem();
  if(input["algorithm"])throw std::invalid_argument("remove radio.algorithm: mountain radio always computes CoREAS and ZHS together");
  c.coreas_cherenkov_threshold=input["coreas_cherenkov_threshold"].as<double>(1.e-3);
  c.start_time_s=input["start_ns"].as<double>(-1000.)*1.e-9;
  c.sample_rate_Hz=input["sample_rate_GHz"].as<double>(1.)*1.e9;
  c.samples=input["samples"].as<std::size_t>(16384);c.moment_order=input["moment_order"].as<unsigned>(12);
  c.subdivision_frequency_Hz=input["subdivision_frequency_GHz"].as<double>(2.)*1.e9;
  c.fraunhofer_limit=input["fraunhofer_limit"].as<double>(.025);
  c.maximum_subdivision_depth=input["maximum_subdivision_depth"].as<unsigned>(12);
  c.mesh_maximum_segment_m=input["mesh_maximum_segment_m"].as<double>(.1);
  c.maximum_device_bytes=input["memory_MiB"].as<std::size_t>(256)*1024u*1024u;
  c.propagation.optical_integration_samples=input["optical_integration_samples"].as<unsigned>(64);
  auto& rock=c.propagation.media[1];rock.refractive_index=scene.atmosphere.rock_refractive_index;
  rock.attenuation_length_m=scene.material.field_attenuation_length_m;
  auto& air=c.propagation.media[0];air.refractive_index=input["uniform_air_index"].as<double>(scene.atmosphere.sea_level_refractive_index);
  auto mode=input["air_index_model"].as<std::string>("native");
  if(mode!="native"&&mode!="uniform")throw std::invalid_argument("radio air_index_model must be native or uniform");
  if(mode=="native"){
    air.refractive_index=scene.atmosphere.sea_level_refractive_index;
    auto center=corsika::terrain::earthCenter(env,scene.atmosphere);
    auto seaLevel=Point(cs,0_m,0_m,-scene.atmosphere.origin_altitude_asl_m*1_m);
    auto const* seaNode=env.getUniverse()->getContainingNode(seaLevel);
    if(!seaNode||!seaNode->hasModelProperties())throw std::invalid_argument("missing native sea-level density reference");
    auto referenceDensity=seaNode->getModelProperties().getMassDensity(seaLevel);
    double referenceDensitySI=referenceDensity/(1_kg/(1_m*1_m*1_m));
    if(!(std::isfinite(referenceDensitySI)&&referenceDensitySI>0.))throw std::invalid_argument("invalid native sea-level density reference");
    auto inverseReferenceDensity=1/referenceDensity;
    air.center_m={center.getX(cs)/1_m,center.getY(cs)/1_m,center.getZ(cs)/1_m};
    air.reference_radius_m=constants::EarthRadius::Mean/1_m;
    double maximum=0.;std::vector<double> boundaries;
    auto visit=[&](auto const& self,auto const* node)->void {
      if(auto sphere=dynamic_cast<Sphere const*>(&node->getVolume())){
        double height=sphere->getRadius()/1_m-air.reference_radius_m;
        if(node->hasModelProperties()&&std::isfinite(height)&&height>0.&&height<1.e6){maximum=std::max(maximum,height);boundaries.push_back(height);}
      }
      for(auto const& child:node->getChildNodes())self(self,child.get());
    };visit(visit,env.getUniverse().get());
    double spacing=input["index_table_step_m"].as<double>(10.);
    if(!(std::isfinite(maximum)&&maximum>0.&&maximum<1.e6&&spacing>=.1&&spacing<=100.))
      throw std::invalid_argument("invalid native radio atmosphere table domain");
    auto count=static_cast<std::size_t>(std::ceil(maximum/spacing));
    std::vector<double> heights;heights.reserve(count+1+2*boundaries.size());
    for(std::size_t i=0;i<=count;++i)heights.push_back(maximum*i/count);
    // Do not smear density-model changes across a ten-metre interpolation
    // cell. Preserve both sides of every native spherical layer boundary.
    for(double h:boundaries){
      double lo=std::max(0.,h-1.e-6);heights.push_back(lo);
      air.radial_integration_breaks_m.push_back(lo);air.radial_integration_breaks_m.push_back(h);
      if(h<maximum){heights.push_back(h+1.e-6);air.radial_integration_breaks_m.push_back(h+1.e-6);}
    }
    std::sort(air.radial_integration_breaks_m.begin(),air.radial_integration_breaks_m.end());
    std::sort(heights.begin(),heights.end());heights.erase(std::unique(heights.begin(),heights.end()),heights.end());
    for(double h:heights){
      auto p=Point(cs,0_m,0_m,(h-scene.atmosphere.origin_altitude_asl_m)*1_m);
      auto const* node=env.getUniverse()->getContainingNode(p);
      if(node&&dynamic_cast<corsika::terrain::TriangularMesh const*>(&node->getVolume()))node=node->getParent();
      // One common sea-level density reference for all layers. Querying each
      // layer's separately normalized refractive-index wrapper would reset a
      // homogeneous upper layer to n_sea, despite its very small density.
      // This independent adapter does not alter the original air model classes.
      double index=node&&node->hasModelProperties()?
        1.+(scene.atmosphere.sea_level_refractive_index-1.)*
          (node->getModelProperties().getMassDensity(p)*inverseReferenceDensity):1.;
      air.radial_index.push_back({h,index});
    }
  }
  for(auto row:input["observers"]){
    auto p=point(row["position_enu_m"],cs);auto name=row["name"].as<std::string>();
    for(unsigned char ch:name)if(ch<32)throw std::invalid_argument("control character in radio observer name");
    c.observers.push_back({name,{p.getX(cs)/1_m,p.getY(cs)/1_m,p.getZ(cs)/1_m},0});
  }
  return c;
}
} // namespace corsika::applications::terrain

/* Native USStdBK + nonconvex DEM, adapted from mountain/TerrainAtmosphere.hpp.
 * No installed/global mesh overlay; no changes to the air-shower application. */
#pragma once
#include <corsika/geometry/terrain/ValidatedTerrain.hpp>
#include <corsika/media/Environment.hpp>
#include <corsika/media/CORSIKA7Atmospheres.hpp>
#include <corsika/media/MediumPropertyModel.hpp>
#include <corsika/media/GladstoneDaleRefractiveIndex.hpp>
#include <corsika/media/HomogeneousMedium.hpp>
#include <corsika/media/UniformRefractiveIndex.hpp>
#include <corsika/media/UniformMagneticField.hpp>
#include <corsika/modules/transport/HomogeneousMaterial.hpp>

namespace corsika::terrain {
  using Interface=IRefractiveIndexModel<IMediumPropertyModel<IMagneticFieldModel<IMediumModel>>>;
  using Environment=corsika::Environment<Interface>;
  template<class T> using AirModel=GladstoneDaleRefractiveIndex<MediumPropertyModel<UniformMagneticField<T>>>;
  using RockModel=UniformRefractiveIndex<MediumPropertyModel<UniformMagneticField<HomogeneousMedium<Interface>>>>;

  struct TerrainAtmosphereConfig {
    // Orthometric/ASL reference, NOT the WGS84 ellipsoidal ENU origin height.
    double origin_altitude_asl_m;
    double sea_level_refractive_index{1.000327};
    double rock_density_g_cm3{2.65};
    double rock_refractive_index{2.};
    std::array<double,3> air_magnetic_field_enu_T{};
    // Legacy field names are retained for existing scene files. The embedded
    // solid's composition/properties are no longer inferred from its shape.
    Medium embedded_medium{Medium::SiliconDioxideFusedQuartz};
    std::vector<Code> embedded_nuclei{get_nucleus_code(28,14),Code::Oxygen};
    std::vector<double> embedded_number_fractions{1./3.,2./3.};
  };
  inline Point earthCenter(Environment const& env,TerrainAtmosphereConfig const& cfg) {
    return Point(env.getCoordinateSystem(),0_m,0_m,
                 -constants::EarthRadius::Mean-cfg.origin_altitude_asl_m*1_m);
  }
  inline double altitude(Point const& p,Point const& center) {
    return ((p-center).getNorm()-constants::EarthRadius::Mean)/1_m;
  }
  inline void buildAtmosphere(Environment& env,TerrainAtmosphereConfig const& cfg) {
    if(!std::isfinite(cfg.origin_altitude_asl_m)||cfg.origin_altitude_asl_m<0.||
       cfg.origin_altitude_asl_m>=7000.||!std::isfinite(cfg.sea_level_refractive_index)||
       cfg.sea_level_refractive_index<1.)
      throw std::invalid_argument("explicit finite ASL origin below 7 km and n0>=1 required");
    for(double b:cfg.air_magnetic_field_enu_T)
      if(!std::isfinite(b))throw std::invalid_argument("nonfinite air magnetic field");
    auto const& b=cfg.air_magnetic_field_enu_T;
    create_5layer_atmosphere<Interface,AirModel>(env,AtmosphereId::USStdBK,
        earthCenter(env,cfg),cfg.sea_level_refractive_index,
        Point(env.getCoordinateSystem(),0_m,0_m,-cfg.origin_altitude_asl_m*1_m),
        Medium::AirDry1Atm,MagneticFieldVector(env.getCoordinateSystem(),b[0]*1_T,b[1]*1_T,b[2]*1_T));
  }
  struct TerrainEmbeddingInfo {
    ClosedMesh const* mesh{}; // owned by the environment node
    double minimum_altitude_asl_m{};
    double maximum_altitude_asl_m{};
    std::size_t vertices{};
    std::size_t triangles{};
  };
  inline TerrainEmbeddingInfo embedTerrain(Environment& env,
      TerrainAtmosphereConfig const& cfg,ClosedMesh mesh,Point const& rockReference) {
    if(!(std::isfinite(cfg.rock_density_g_cm3)&&cfg.rock_density_g_cm3>0.&&
         std::isfinite(cfg.rock_refractive_index)&&cfg.rock_refractive_index>=1.))
      throw std::invalid_argument("invalid rock material");
    if(mesh.inside(rockReference)!=EInside::kInside)
      throw std::invalid_argument("rock reference must be strictly inside the actual terrain");
    auto* parent=env.getUniverse()->getContainingNode(rockReference);
    auto const* sphere=parent?dynamic_cast<Sphere const*>(&parent->getVolume()):nullptr;
    if(!sphere||!parent->getChildNodes().empty()||
       std::abs((sphere->getRadius()-constants::EarthRadius::Mean)/1_m-7000.)>1.e-6)
      throw std::invalid_argument("terrain must fit in the innermost 7 km USStdBK layer");
    TerrainEmbeddingInfo info;
    info.minimum_altitude_asl_m=std::numeric_limits<double>::infinity();
    info.maximum_altitude_asl_m=-info.minimum_altitude_asl_m;
    auto center=earthCenter(env,cfg);
    for(auto const& p:mesh.getVertices()) {
      auto h=altitude(p,center);
      if(!std::isfinite(h)||h<0.||h>=7000.-1.e-4)
        throw std::invalid_argument("terrain crosses sea level or 7 km: partition is required");
      info.minimum_altitude_asl_m=std::min(info.minimum_altitude_asl_m,h);
      info.maximum_altitude_asl_m=std::max(info.maximum_altitude_asl_m,h);
    }
    info.vertices=mesh.getVertexCount();info.triangles=mesh.getTriangleCount();
    auto rock=env.createNode<ClosedMesh>(std::move(mesh));
    interfaces::HomogeneousMaterial material{cfg.embedded_medium,NuclearComposition(cfg.embedded_nuclei,cfg.embedded_number_fractions),
        cfg.rock_density_g_cm3,cfg.rock_refractive_index,{0.,0.,0.}};
    rock->setModelProperties(material.makeModel<Interface>(env.getCoordinateSystem()));
    info.mesh=&static_cast<ClosedMesh const&>(rock->getVolume());
    parent->addChild(std::move(rock));
    return info;
  }
}

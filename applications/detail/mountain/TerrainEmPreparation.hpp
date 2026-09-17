#pragma once
#include "TerrainScene.hpp"
#include <corsika/geometry/terrain/FlatTerrainExport.hpp>
#include <corsika/accelerator/em/common/EnvironmentSnapshotBuilder.hpp>
#include <corsika/accelerator/em/common/tables/ProposalNativeTableExporter.hpp>
#include <corsika/accelerator/em/ProposalNativeRequirements.hpp>
#include <corsika/modules/terrain/TerrainEmSession.hpp>
#include <corsika/modules/transport/InterfaceMaterialPreparation.hpp>

namespace corsika::applications::terrain {
template<class Interaction,class Continuous>
std::vector<corsika::terrain::TerrainEmMaterial> prepareEmMaterials(
    Scene const& scene,corsika::terrain::Environment const& env,
    corsika::terrain::FlatTerrainData const& mesh,Interaction const& interaction,
    Continuous const& continuous,accelerator::em::AcceleratedPhysicsRequirements const& requirements,
    std::filesystem::path const& cache) {
  auto cs=env.getCoordinateSystem();
  auto* rockNode=env.getUniverse()->getContainingNode(point(scene.yaml["geometry"]["rock_reference_enu_m"],cs));
  auto rockHash=rockNode->getModelProperties().getNuclearComposition().getHash();
  auto interactions=interaction.nativeCalculatorViews();auto losses=continuous.nativeCalculatorViews();
  std::vector<corsika::terrain::TerrainEmMaterial> banks(2);
  for(std::size_t i=0;i<2;++i) {
    std::vector<proposal::NativeInteractionCalculatorView> iv;
    std::vector<proposal::NativeContinuousCalculatorView> cv;
    for(auto const& view:interactions)if((view.medium_hash==rockHash)==bool(i))iv.push_back(view);
    for(auto const& view:losses)if((view.medium_hash==rockHash)==bool(i))cv.push_back(view);
    banks[i]=interfaces::prepareEmMaterial(std::move(iv),std::move(cv),banks[i].environment,requirements,cache);
  }
  auto& air=banks[0].environment;
  air=gpu::em::makeCorsika7AtmosphereSnapshot(AtmosphereId::USStdBK,
      {0.,0.,-constants::EarthRadius::Mean/1_m-scene.atmosphere.origin_altitude_asl_m});
  // CPU's innermost atmosphere is a ball, not a ground-absorbing shell. Retain
  // its exponential density below ASL=0. The unreachable 1 mm central domain
  // is an explicit unsupported geometry, never a fictitious sea-level escape.
  air.atmosphere_layers[0].inner_radius_m=.001;
  for(int axis=0;axis<3;++axis)air.magnetic_field_T[axis]=scene.atmosphere.air_magnetic_field_enu_T[axis];
  auto& rock=banks[1].environment;
  rock.geometry=gpu::em::EnvironmentGeometry::HomogeneousConvexPolyhedron;
  rock.number_of_layers=1;rock.number_of_convex_planes=6;rock.convex_boundary_tolerance_m=1.e-8;
  rock.atmosphere_layers[0].density_model=gpu::em::DensityModel::Homogeneous;
  rock.atmosphere_layers[0].density_parameter_a=scene.atmosphere.rock_density_g_cm3;
  for(int axis=0;axis<3;++axis) rock.magnetic_field_T[axis]=scene.material.magnetic_field_T[axis];
  rock.atmosphere_layers[0].medium_id=1;rock.observation_plane_normal[2]=1.;
  auto const& box=mesh.nodes.at(0);double lo[3]={box.low.x-1.,box.low.y-1.,box.low.z-1.};
  double hi[3]={box.high.x+1.,box.high.y+1.,box.high.z+1.};
  for(int axis=0;axis<3;++axis) {
    auto& lower=rock.convex_planes[2*axis];auto& upper=rock.convex_planes[2*axis+1];
    lower={0.,0.,0.,-lo[axis]};upper={0.,0.,0.,hi[axis]};
    if(axis==0){lower.nx=-1.;upper.nx=1.;}
    if(axis==1){lower.ny=-1.;upper.ny=1.;}
    if(axis==2){lower.nz=-1.;upper.nz=1.;}
  }
  // The box is only a homogeneous integration envelope. A missing DEM exit
  // while logically in rock is an error in TerrainStepKernel, not a box escape.
  if(!gpu::em::atmosphere_detail::validEnvironment(air)||!gpu::em::atmosphere_detail::validEnvironment(rock))
    throw std::runtime_error("invalid terrain material snapshot");
  return banks;
}
}

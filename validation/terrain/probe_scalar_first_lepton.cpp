// Live scalar PROPOSAL oracle for a recorded, straight first rock step.
// This is deliberately not a second full-shower algorithm or a GPU table query.
#include "detail/mountain/TerrainScene.hpp"
#include <corsika/modules/proposal/ContinuousProcess.hpp>
#include <corsika/accelerator/em/RandomDomains.hpp>
#include <corsika/accelerator/em/common/Philox.hpp>
#include <PROPOSAL/PROPOSAL.h>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <iostream>

int main(int argc,char** argv) {
  using namespace corsika;
  using namespace corsika::units::si;
  namespace em=gpu::em;
  if(argc!=3){std::cerr<<"scene.yaml tracks.csv\n";return 2;}
  logging::set_level(logging::level::warn);
  applications::terrain::Scene scene(argv[1]);terrain::Environment env;
  auto mesh=scene.build(env);auto cs=env.getCoordinateSystem();
  if(!mesh.mesh)throw std::runtime_error("scene has no terrain mesh");
  auto& rng=RNGManager<>::getInstance();rng.registerRandomStream("proposal");rng.setSeed(67101);
  for(auto pid:{Code::Photon,Code::Electron,Code::Positron,Code::MuMinus,Code::MuPlus,Code::TauMinus,Code::TauPlus})
    set_energy_production_threshold(pid,.0005_GeV);
  proposal::ContinuousProcess<> continuous(env);
  auto* rock=env.getUniverse()->getContainingNode(
      applications::terrain::point(scene.yaml["geometry"]["rock_reference_enu_m"],cs));
  auto hash=rock->getModelProperties().getNuclearComposition().getHash();
  std::ifstream input(argv[2]);std::string line,part;std::getline(input,line);std::getline(input,line);
  std::istringstream row(line);std::vector<std::string> v;
  while(std::getline(row,part,','))v.push_back(part);
  if(v.size()!=20||v[0]!="1"||v[3]!="rock"||
     (v[1]!="11"&&v[1]!="-11")||std::stod(v[12])!=1.||
     std::stod(v[14])!=0.||std::stod(v[15])!=0.||std::stod(v[16])!=1.) {
    throw std::runtime_error("oracle is limited to the documented first vertical 1 GeV rock lepton step");
  }
  auto pid=v[1]=="11"?Code::Electron:Code::Positron;
  Point a(cs,std::stod(v[4])*1_m,std::stod(v[5])*1_m,std::stod(v[6])*1_m);
  Point b(cs,std::stod(v[7])*1_m,std::stod(v[8])*1_m,std::stod(v[9])*1_m);
  // Use the original medium's straight-track integral, as doContinuous does.
  auto delta=b-a;auto speed=constants::c; // Parameterization cancels in dX.
  StraightTrajectory track(Line(a,delta.normalized()*speed),delta.getNorm()/speed);
  double dx=rock->getModelProperties().getIntegratedGrammage(track)/1_g*1_cm*1_cm;
  auto views=continuous.nativeCalculatorViews();
  auto view=std::find_if(views.begin(),views.end(),[&](auto const& x){return x.projectile==pid&&x.medium_hash==hash;});
  if(view==views.end())throw std::runtime_error("missing live scalar rock calculator");
  // The calculator is owned by our NON-const local ContinuousProcess. Its
  // public export view is const, while the original numerical method is not.
  // This diagnostic intentionally executes that original mutable CPU API;
  // production read-only exports and their caches are not modified by a hook.
  double ef=const_cast<PROPOSAL::Displacement*>(view->displacement)->UpperLimitTrackIntegral(1000.,dx);
  PROPOSAL::EMinusDef electron;PROPOSAL::EPlusDef positron;
  auto const& definition=pid==Code::Electron?static_cast<PROPOSAL::ParticleDef const&>(electron):
      static_cast<PROPOSAL::ParticleDef const&>(positron);
  auto scattering=PROPOSAL::make_multiple_scattering(PROPOSAL::MultipleScatteringType::MoliereInterpol,definition,*view->medium);
  auto uniform=[](std::uint64_t draw){return em::uniformOpen01(em::RandomNumberKey{
      67101,0,1,0,em::ContinuousScatteringRandomProcessId,draw});};
  double u1=uniform(em::MoliereFirstAngleDrawId),u2=uniform(em::MoliereSecondAngleDrawId);
  double uphi=uniform(em::MoliereAzimuthDrawId);
  double angle=scattering->CalculateScatteringAngle2D(dx,1000.,ef,u1,u2);
  // Reproduce ContinuousProcess::scatter's original coordinate rotations.
  DirectionVector initial(cs,{0.,0.,1.}),normal(cs,{0.,-1.,0.});
  auto r1=make_rotation(cs,normal.getComponents(),angle);
  auto r2=make_rotation(r1,initial.getComponents(),uphi*2.*M_PI);
  DirectionVector final(cs,initial.getComponents(r2));final=final.normalized();
  double e_gpu=std::stod(v[13]);double direction_delta=0.;
  double same_energy_angle=scattering->CalculateScatteringAngle2D(dx,1000.,e_gpu*1000.,u1,u2);
  double recorded_angle=std::atan2(std::hypot(std::stod(v[17]),std::stod(v[18])),std::stod(v[19]));
  double d[3]={final.getX(cs),final.getY(cs),final.getZ(cs)};
  for(int i=0;i<3;++i)direction_delta=std::max(direction_delta,std::abs(d[i]-std::stod(v[17+i])));
  std::cout<<std::setprecision(17)<<"{\"pid\":"<<v[1]<<",\"grammage_g_cm2\":"<<dx
           <<",\"scalar_energy_GeV\":"<<ef/1000.<<",\"record_energy_GeV\":"<<e_gpu
           <<",\"energy_delta_GeV\":"<<ef/1000.-e_gpu<<",\"moliere_u1\":"<<u1
           <<",\"moliere_u2\":"<<u2<<",\"moliere_azimuth_u\":"<<uphi
           <<",\"scalar_angle\":"<<angle<<",\"scalar_angle_at_record_energy\":"<<same_energy_angle
           <<",\"angle_from_recorded_direction\":"<<recorded_angle
           <<",\"scalar_direction\":["<<d[0]<<','<<d[1]<<','<<d[2]
           <<"],\"max_direction_delta\":"<<direction_delta<<"}\n";
  return 0;
}

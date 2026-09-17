// Actual PROPOSAL banks + compiled InterfaceEmSession, not a mock kernel.
#include <corsika/modules/transport/InterfaceEmSession.hpp>
#include "InterfaceResidentChecks.hpp"
#include <cstdlib>
#include <corsika/modules/transport/InterfaceMaterialPreparation.hpp>
#include <corsika/modules/transport/InterfaceRegionMap.hpp>
#include <corsika/modules/terrain/TerrainAtmosphere.hpp>
#include <corsika/modules/terrain/TerrainMagneticTracking.hpp>
#include <corsika/modules/PROPOSAL.hpp>
#include <corsika/accelerator/em/common/TransportMass.hpp>
#include <iostream>
#include <iomanip>
#include <random>
using namespace corsika;
using namespace corsika::units::si;
namespace f=terrain::flat;
namespace em=gpu::em;
namespace api=interfaces;
struct NoHadronicFinalState {
  template<class... Args> void doInteraction(Args&&...) {throw std::runtime_error("test must not execute hadronic final state");}
  bool isValid(Code,Code,HEPEnergyType)const{return true;}
};
struct ScalarParticle {
  Point position;DirectionVector direction;Code pid;
  terrain::Environment::BaseNodeType const* node;
  auto getPosition()const{return position;}
  auto getDirection()const{return direction;}
  auto getEnergy()const{return 1_GeV;}
  auto getPID()const{return pid;}
  auto getCharge()const{return get_charge(pid);}
  auto getMomentum()const{return direction*sqrt(static_pow<2>(1_GeV)-static_pow<2>(get_mass(pid)));}
  auto getVelocity()const{return getMomentum()/getEnergy()*constants::c;}
  auto getNode()const{return node;}
};
void require(bool pass,char const* label){if(!pass)throw std::runtime_error(label);}
terrain::MeshData box(bool tetra) {
  if(tetra)return {{{{-1.,-1.,-1.}},{{1.,-1.,-1.}},{{0.,1.,-1.}},{{0.,0.,1.}}},
                  {{{0,2,1}},{{0,1,3}},{{1,2,3}},{{2,0,3}}}};
  return {{{{-1.,-1.,-1.}},{{1.,-1.,-1.}},{{1.,1.,-1.}},{{-1.,1.,-1.}},
           {{-1.,-1.,1.}},{{1.,-1.,1.}},{{1.,1.,1.}},{{-1.,1.,1.}}},
          {{{0,2,1}},{{0,3,2}},{{4,5,6}},{{4,6,7}},{{0,1,5}},{{0,5,4}},
           {{1,2,6}},{{1,6,5}},{{2,3,7}},{{2,7,6}},{{3,0,4}},{{3,4,7}}}};
}
em::EnvironmentSnapshot homogeneous(double density,double bz) {
  em::EnvironmentSnapshot e;
  e.number_of_layers=1;e.earth_center_m[2]=-1000.;e.observation_radius_m=500.;
  auto& l=e.atmosphere_layers[0];l.inner_radius_m=.001;l.outer_radius_m=10000.;
  l.density_model=em::DensityModel::Homogeneous;l.density_parameter_a=density;
  // Deliberately NOT a logical interface ID.
  l.medium_id=456;e.magnetic_field_T[2]=bz;
  e.observation_plane_normal[2]=1.;
  return e;
}
int main(int argc,char** argv) {
  try {
    bool const fieldOnly=argc>1&&std::string(argv[1])=="fieldonly";
    bool const same=fieldOnly||(argc>1&&std::string(argv[1])=="same");
    bool const bent=fieldOnly||(argc>1&&std::string(argv[1])=="field");
    bool const tetra=argc>1&&std::string(argv[1])=="tetra";
    bool const swapped=argc>1&&std::string(argv[1])=="swapped";
    logging::set_level(logging::level::warn);
    RNGManager<>::getInstance().registerRandomStream("proposal");
    for(auto p:{Code::Electron,Code::Positron,Code::Photon,Code::MuMinus,Code::MuPlus})
      set_energy_production_threshold(p,.5_MeV);
    terrain::Environment env;auto cs=env.getCoordinateSystem();
    auto world=env.createNode<Sphere>(Point(cs,0_m,0_m,-1000_m),10000_m);
    api::HomogeneousMaterial water{Medium::WaterLiquid,NuclearComposition({Code::Hydrogen,Code::Oxygen},{2./3.,1./3.}),1.,1.33,{0.,0.,bent?1.e-2:0.}};
    api::HomogeneousMaterial rock{Medium::SiliconDioxideFusedQuartz,NuclearComposition({get_nucleus_code(28,14),Code::Oxygen},{1./3.,2./3.}),2.65,2.,{0.,0.,bent?-1.e-2:0.}};
    world->setModelProperties(water.makeModel<terrain::Interface>(cs));
    auto* outside=world.get();env.getUniverse()->addChild(std::move(world));
    auto data=box(tetra);terrain::validateTerrainData(data);
    auto solid=env.createNode<terrain::ClosedMesh>(terrain::MeshLoader::toPoints(data,cs,1_m),data.faces);
    auto* inside=solid.get();auto const& mesh=static_cast<terrain::ClosedMesh const&>(inside->getVolume());
    auto embedded=same?water:rock;
    if(fieldOnly)embedded.magnetic_field_T[2]=-water.magnetic_field_T[2];
    solid->setModelProperties(embedded.makeModel<terrain::Interface>(cs));
    outside->addChild(std::move(solid));auto flat=terrain::exportFlatTerrain(mesh);
    api::validateCalculatorMaterialKeys(env);
    api::MaterialInterface binding{17,93,0,same&&!fieldOnly?0:1};binding.validate(same&&!fieldOnly?1:2);
    api::InterfaceRegionMap regionOf(*inside,binding);
    require(regionOf(outside)==17&&regionOf(inside)==93,"logical node routing");
    auto invalid=binding;invalid.inside_region=17;
    bool caught=false;try{invalid.validate(2);}catch(std::invalid_argument const&){caught=true;}
    require(caught,"duplicate region gate");
    invalid=binding;invalid.inside_material=5;caught=false;
    try{invalid.validate(2);}catch(std::invalid_argument const&){caught=true;}
    require(caught,"missing material gate");
    // Million analytic plane crossings with arbitrary region IDs, both signs.
    // Cube/tetra share the horizontal bottom face z=-1, at x=y=0.
    std::mt19937_64 rng(19191);std::uniform_real_distribution<double> u(.001,.2);
    for(int i=0;i<1000000;++i) {
      double d=u(rng);bool insideSide=i%2;
      f::QuadraticPath path{{{0.,0.,-1.+(insideSide?d:-d)},{0.,0.,insideSide?-1.:1.},false},{0.,0.,0.},1.};
      auto h=api::nextCrossing(flat.view(),binding,insideSide?93:17,path);
      require(h.hit.found()&&!h.invalid_region&&h.to_region==(insideSide?17:93),"analytic region crossing");
      require(std::abs(h.hit.distance-d)<1.e-14,"analytic distance");
    }
    auto invalidHit=api::nextCrossing(flat.view(),binding,99,{{{0.,0.,0.},{0.,0.,1.},false},{0.,0.,0.},1.});
    require(invalidHit.invalid_region,"invalid query region");
    NoHadronicFinalState hadronic;
    proposal::Interaction interaction(env,hadronic,hadronic,80_GeV);
    proposal::ContinuousProcess<> continuous(env);
    auto iv=interaction.nativeCalculatorViews();auto cv=continuous.nativeCalculatorViews();
    std::vector<api::EmMaterial> banks;
    auto add=[&](auto const& material,double density,double field) {
      auto hash=material.composition.getHash();std::vector<proposal::NativeInteractionCalculatorView> i;
      std::vector<proposal::NativeContinuousCalculatorView> c;
      for(auto const& v:iv)if(v.medium_hash==hash)i.push_back(v);
      for(auto const& v:cv)if(v.medium_hash==hash)c.push_back(v);
      banks.push_back(api::prepareEmMaterial(i,c,homogeneous(density,field),{1000.,.5,300.,.5},{}));
    };
    add(water,1.,water.magnetic_field_T[2]);if(!same)add(rock,2.65,rock.magnetic_field_T[2]);
    if(fieldOnly) {
      banks.push_back(banks.front());
      banks.back().environment.magnetic_field_T[2]=-water.magnetic_field_T[2];
    }
    if(swapped){std::swap(banks[0],banks[1]);binding.outside_material=1;binding.inside_material=0;}
    api::EmConfig cfg;cfg.threads=2;cfg.batch_size=64;cfg.maximum_device_bytes=256*1024*1024;
    if(auto threads=std::getenv("C8_INTERFACE_TEST_THREADS"))cfg.threads=std::stoi(threads);
    cfg.seed=81819;cfg.interface=binding;cfg.maximum_step_m=.01;
    cfg.resident_capacity=4096;
    cfg.resident_record_capacity=96; // Deliberately forces bounded output checkpoints.
    std::size_t wideFront=0;
    if(auto value=std::getenv("C8_INTERFACE_TEST_WIDE_FRONT"))wideFront=std::stoul(value);
    if(wideFront){
      cfg.resident_wavefront_capacity=wideFront;cfg.resident_capacity=262144;
      cfg.resident_record_capacity=4*wideFront;cfg.maximum_device_bytes=2ull*1024*1024*1024;
    }
    // Rejected configurations must not initialize a runtime or upload any table.
    auto saved=banks[0].environment;
    banks[0].environment.atmosphere_layers[0].density_parameter_a=-1.;caught=false;
    try{api::InterfaceEmSession bad(flat,banks,cfg);}catch(std::invalid_argument const&){caught=true;}
    require(caught,"negative snapshot density gate");banks[0].environment=saved;
    banks[0].environment.magnetic_field_T[0]=std::numeric_limits<double>::quiet_NaN();caught=false;
    try{api::InterfaceEmSession bad(flat,banks,cfg);}catch(std::invalid_argument const&){caught=true;}
    require(caught,"nonfinite snapshot field gate");banks[0].environment=saved;
    auto badConfig=cfg;badConfig.resident_capacity=std::numeric_limits<std::size_t>::max();
    interface_test::reject([&]{api::InterfaceEmSession bad(flat,banks,badConfig);});
    badConfig=cfg;badConfig.maximum_device_bytes=1024;
    interface_test::reject([&]{api::InterfaceEmSession bad(flat,banks,badConfig);});
    badConfig=cfg;badConfig.resident_record_capacity=cfg.batch_size-1;
    interface_test::reject([&]{api::InterfaceEmSession bad(flat,banks,badConfig);});
    api::InterfaceEmSession session(flat,banks,cfg);
    if(session.executionSpace()=="OpenMP")
      require(session.executionConcurrency()==cfg.threads,"OpenMP thread configuration was not applied");
    std::vector<em::EmParticleState> input;
    for(int pid:{22,11,-11})for(bool insideSide:{false,true}) {
      em::EmParticleState p;p.pid=pid;p.energy_GeV=1.;p.history_id=input.size()+1;
      p.medium_id=insideSide?93:17;p.position_m[2]=-1.+(insideSide?1.e-7:-1.e-7);
      p.direction[2]=insideSide?-1.:1.;p.weight=3.;input.push_back(p);
    }
    auto initial=input;
    auto a=session.advance(input,100),b=session.advance(input,100);
    terrain::MagneticTracking cpuTracking(mesh);
    for(std::size_t i=0;i<a.size();++i) {
      auto const& r=a[i];require(r.crossed_material&&r.outcome==api::EmOutcome::Continuation,"real step reaches interface");
      auto const& p=r.start;
      ScalarParticle scalar{Point(cs,p.position_m[0]*1_m,p.position_m[1]*1_m,p.position_m[2]*1_m),
          DirectionVector(cs,{p.direction[0],p.direction[1],p.direction[2]}),
          p.pid==22?Code::Photon:p.pid==11?Code::Electron:Code::Positron,
          p.medium_id==17?outside:inside};
      auto [track,nextNode]=cpuTracking.getTrack(scalar);
      require(regionOf(nextNode)==r.end.medium_id,"actual CPU/Kokkos target region");
      require(std::abs(track.getDuration()/1_s-(r.end.time_s-r.start.time_s))<1.e-20,"actual CPU/Kokkos crossing time");
      auto endpoint=track.getPosition(1.);
      require(std::abs(endpoint.getZ(cs)/1_m-r.end.position_m[2])<1.e-12,"actual CPU/Kokkos crossing endpoint");
      require(r.end.medium_id==binding.opposite(r.start.medium_id),"real step next region");
      require(r.end.history_id==r.start.history_id&&r.end.weight==r.start.weight&&
              r.end.step_id==r.start.step_id+1&&r.end.time_s>r.start.time_s,"crossing identity/time");
      double density=r.start.medium_id==17||same?1.:2.65;
      require(std::abs(r.grammage_g_cm2-r.distance_m*100.*density)<1.e-10,"grammage belongs to pre-crossing medium");
      require(r.end.energy_GeV<=r.start.energy_GeV&&r.deposited_GeV>=0.,"no energy injection at interface");
      require(r.end.energy_GeV==b[i].end.energy_GeV&&r.end.time_s==b[i].end.time_s,"same-backend repeat");
      input[i]=r.end;
    }
    // The next step must use the newly selected bank, not the previous material.
    auto after=session.advance(input,200);
    for(auto const& r:after) {
      require(r.has_track,"post-crossing transport record");
      double density=r.start.medium_id==17||same?1.:2.65;
      // Moliere changes the final direction, not the recorded chord grammage.
      double chord=0.;for(int k=0;k<3;++k)chord+=std::pow(r.end.position_m[k]-r.start.position_m[k],2);
      require(std::abs(r.grammage_g_cm2-std::sqrt(chord)*100.*density)<1.e-7,"post-crossing material density");
      require(binding.containsRegion(r.end.medium_id),"environment layer ID leaked into region ID");
    }
    // Strong in-solid field, far from a surface: finite magnetic-step end is
    // NOT a missing-exit error. Both sides carry independently supplied fields.
    if(bent) {
      auto p=input[2];p.medium_id=93;p.position_m[0]=0.;p.position_m[1]=0.;p.position_m[2]=0.;
      p.direction[0]=1.;p.direction[1]=p.direction[2]=0.;p.energy_GeV=.0011;p.step_id=0;
      auto r=session.advance({p},500).front();require(!r.error,"interior magnetic cap without boundary");
      // Opposite fields in two genuinely different materials bend the same
      // electron in opposite directions, before stochastic angular scattering.
      p.energy_GeV=1.;p.medium_id=17;p.position_m[2]=-2.;
      auto exterior=session.advance({p},600).front();
      p.medium_id=93;p.position_m[2]=0.;
      auto interior=session.advance({p},700).front();
      require(exterior.has_track&&interior.has_track&&exterior.path_midpoint_m[1]>0.&&
              interior.path_midpoint_m[1]<0.,"two-sided magnetic vectors not respected");
    }
    interface_test::replay(session,initial);
    if(wideFront){
      std::vector<em::EmParticleState> wide;
      for(std::size_t i=0;i<wideFront+19;++i){
        auto p=initial[i%initial.size()];p.history_id=i+1;p.parent_history_id=0;wide.push_back(p);
      }
      interface_test::cascadeReplay(session,wide,cfg.resident_record_capacity,wideFront);
    }else interface_test::cascadeReplay(session,initial,cfg.resident_record_capacity);
    std::cout<<"PASS interface variant="<<(argc>1?argv[1]:"ordinary")
      <<" geometry_checks=1000000 live_crossings=6 execution="<<session.executionSpace()
      <<" concurrency="<<session.executionConcurrency()<<'\n';
    return 0;
  }catch(std::exception const& e){std::cerr<<e.what()<<'\n';return 1;}
}

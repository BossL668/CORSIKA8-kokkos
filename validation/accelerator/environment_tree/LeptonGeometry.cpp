#include "GeometryDriver.hpp"
#include <corsika/accelerator/em/common/FullEnvironmentTreeBuilder.hpp>
#include <corsika/accelerator/em/common/EnvironmentSnapshotStorage.hpp>
#include <corsika/accelerator/em/common/ProcessCapabilities.hpp>
#include <corsika/media/Environment.hpp>
#include <corsika/media/HomogeneousMedium.hpp>
#include <corsika/media/UniformMagneticField.hpp>
#include <corsika/media/MediumPropertyModel.hpp>
#include <corsika/media/GladstoneDaleRefractiveIndex.hpp>
#include <corsika/modules/tracking/TrackingLeapFrogCurved.hpp>
#include <corsika/accelerator/em/common/RouterParticleConversion.hpp>
#include <corsika/framework/core/ScalarCascadeStepper.hpp>
#include <corsika/setup/SetupStack.hpp>
#include <nlohmann/json.hpp>
#include <iostream>
#include <algorithm>

namespace corsika {
  template<typename TDerived> struct OriginalIntersect {
    template<typename TParticle> auto nextIntersect(TParticle const&,TimeType) const;
  };
}
#include "OriginalIntersect.inl"
using namespace corsika;
using namespace corsika::units::si;
namespace em=corsika::gpu::em;
namespace et=em::environment_tree;
using Interface=IRefractiveIndexModel<IMediumPropertyModel<IMagneticFieldModel<IMediumModel>>>;
template<typename T> using Extra=GladstoneDaleRefractiveIndex<MediumPropertyModel<UniformMagneticField<T>>>;
using Node=VolumeTreeNode<Interface>;
struct Particle {
  Node const* node;Point position;Vector<hepmomentum_d> momentum;
  HEPEnergyType energy;int sign;
  auto getNode() const {return node;}
  auto const& getPosition() const {return position;}
  auto const& getMomentum() const {return momentum;}
  auto getEnergy() const {return energy;}
  auto getVelocity() const {return momentum/energy*constants::c;}
  auto getCharge() const {return sign*constants::e;}
};

static nlohmann::json identityRoundtrip() {
  RNGManager<>::getInstance().registerRandomStream("cascade");
  Environment<Interface> env;auto cs=env.getCoordinateSystem();Point origin(cs,0_m,0_m,0_m);
  NuclearComposition composition({Code::Nitrogen,Code::Oxygen},{.8,.2});
  auto outer=env.createNode<Sphere>(origin,200_m);auto inner=env.createNode<Sphere>(origin,100_m);
  auto* logical=outer.get();
  for(auto node:{outer.get(),inner.get()})
    node->setModelProperties<Extra<HomogeneousMedium<Interface>>>(1.0003,origin,
      Medium::AirDry1Atm,MagneticFieldVector(cs,0_T,0_T,0_T),1.2_kg/(1_m*1_m*1_m),composition);
  outer->addChild(std::move(inner));env.getUniverse()->addChild(std::move(outer));
  auto exported=et::exportFullEnvironmentTree(*env.getUniverse(),cs);et::SnapshotStorage storage(exported);
  auto const key=et::snapshotKey(storage.view());
  auto original=storage.models[0].magnetic_field_T[0];storage.models[0].magnetic_field_T[0]=.01;
  if(key==et::snapshotKey(storage.view()))throw std::runtime_error("session key ignores changed field");
  storage.models[0].magnetic_field_T[0]=original;
  setup::HybridStack<Environment<Interface>> input,output;
  for(unsigned i=0;i<128;++i) {
    // Represent a just-exited child whose rounded position is slightly inside
    // that child. The explicit outer logical node must survive the roundtrip.
    auto p=input.addParticle(std::make_tuple(Code::Electron,1_GeV,
      DirectionVector(cs,{0.,0.,1.}),Point(cs,0_m,0_m,std::nextafter(100.,0.)*1_m),0_s));
    p.setNode(logical);p.setWeight(3.);p.setTransportIdentity(transport::TransportIdentity{i+10,1,2,7});
    auto state=em::router_detail::toDeviceState(p,cs,p.getHistoryId(),p.getParentHistoryId(),p.getGeneration(),p.getStepId(),storage.view());
    auto imported=em::router_detail::importParticle(output,state,cs,storage.view());
    if(imported.getNode()!=logical || imported.getHistoryId()!=i+10 || imported.getWeight()!=3.)
      throw std::runtime_error("CPU/device identity roundtrip failed");
  }
  tracking_line::Tracking tracking;int unused_process=0;
  ScalarCascadeStepper<tracking_line::Tracking,int,decltype(output)> stepper(env,tracking,unused_process,output);
  stepper.setNodes(true);
  for(auto const& p:output)
    if(p.getNode()!=logical)throw std::runtime_error("node refresh overwrote explicit logical node");
  stepper.setNodes();
  for(auto const& p:output)
    if(p.getNode()==logical)throw std::runtime_error("legacy containment refresh unexpectedly changed");
  return {{"particles",128},{"passed",true},{"preserves_logical_boundary_side",true},
      {"legacy_refresh_unchanged",true},{"session_key_detects_changed_field",true}};
}

static nlohmann::json exercise(double field_strength) {
  Environment<Interface> env;auto cs=env.getCoordinateSystem();Point origin(cs,0_m,0_m,0_m);
  NuclearComposition composition({Code::Nitrogen,Code::Oxygen},{.8,.2});
  auto outer=env.createNode<Sphere>(origin,200_m);
  auto middle=env.createNode<Sphere>(origin,100_m);
  auto inner=env.createNode<Sphere>(origin,50_m);
  unsigned index=0;
  for (auto node:{outer.get(),middle.get(),inner.get()}) {
    ++index;
    double const f=field_strength;
    node->setModelProperties<Extra<HomogeneousMedium<Interface>>>(
      1.0003,origin,Medium::AirDry1Atm,
      MagneticFieldVector(cs,(index==2?f:0.)*1_T,(index==1?f:-f)*1_T,0_T),
      index*1.2_kg/(1_m*1_m*1_m),composition);
  }
  middle->addChild(std::move(inner));outer->addChild(std::move(middle));
  env.getUniverse()->addChild(std::move(outer));
  auto full=et::exportFullEnvironmentTree(*env.getUniverse(),cs);et::SnapshotStorage storage(full);
  std::vector<em::EmParticleState> input;
  for (unsigned i=0;i<256;++i) {
    em::EmParticleState p;p.pid=i%4<2?11:13;if(i%2)p.pid=-p.pid;
    p.energy_GeV=1.;p.medium_id=static_cast<int>(Medium::AirDry1Atm);
    p.position_m[0]=(i%16)-7.5;p.position_m[1]=(i/16)-7.5;p.position_m[2]=150.;
    double angle=.001*(static_cast<int>(i%5)-2);p.direction[0]=std::sin(angle);p.direction[2]=-std::cos(angle);
    p.history_id=i+1;p.environment_node=full.topology.id(env.getUniverse()->getContainingNode(
        Point(cs,p.position_m[0]*1_m,p.position_m[1]*1_m,p.position_m[2]*1_m)));
    input.push_back(p);
  }
  auto output=runLeptonSequences(storage.view(),input);
  OriginalIntersect<tracking_leapfrog_curved::Tracking> native;
  OriginalIntersect<tracking_line::Tracking> straight;
  unsigned steps=0,transitions=0,magnetic_steps=0,errors=0;
  double max_position=0.,max_direction=0.,max_energy=0.,max_column=0.,max_time=0.;
  nlohmann::json examples=nlohmann::json::array();
  auto fail=[&](unsigned i,unsigned j,std::string reason,double actual=0.,double expected=0.) {
    ++errors;if(examples.size()<12)examples.push_back({{"particle",i},{"step",j},{"reason",reason},{"actual",actual},{"expected",expected}});
  };
  for (unsigned i=0;i<input.size();++i) {
    auto state=input[i];auto const& sequence=output[i];
    if(sequence.fallback) {fail(i,sequence.count,"fallback",static_cast<int>(sequence.failure.reason));continue;}
    for (unsigned j=0;j<sequence.count;++j) {
      auto const& record=sequence.records[j];
      double mass=get_mass(convert_from_PDG(static_cast<PDGCode>(state.pid)))/1_GeV;
      double p=std::sqrt((state.energy_GeV-mass)*(state.energy_GeV+mass));
      Particle particle{full.topology.native_nodes[state.environment_node],
        Point(cs,state.position_m[0]*1_m,state.position_m[1]*1_m,state.position_m[2]*1_m),
        Vector<hepmomentum_d>(cs,state.direction[0]*p*1_GeV,state.direction[1]*p*1_GeV,state.direction[2]*p*1_GeV),
        state.energy_GeV*1_GeV,state.pid>0?-1:1};
      auto const field=particle.node->getModelProperties().getMagneticField(particle.position);
      auto const velocity=particle.getVelocity();auto const speed=velocity.getNorm();
      auto const momentumSI=constants::c*convert_HEP_to_SI<MassType::dimension_type>(particle.getMomentum().getNorm());
      auto perpendicular=field.getNorm()==0_T?particle.getMomentum().getNorm():
          (particle.momentum-particle.momentum.getParallelProjectionOnto(field)).getNorm();
      auto rg=field.getNorm()==0_T?std::numeric_limits<double>::infinity()*1_m:
          convert_HEP_to_SI<MassType::dimension_type>(perpendicular)*constants::c/abs(particle.getCharge())/field.getNorm();
      bool linear=field.getNorm()==0_T || perpendicular<1_eV || rg>1.e9_m;
      auto limit=linear?std::numeric_limits<double>::infinity()*1_s:2*std::cos(.2)*std::sin(.2)*rg/speed;
      auto [time,next]=linear?straight.nextIntersect(particle,limit):native.nextIntersect(particle,limit);
      auto k=particle.getCharge()/momentumSI*speed;
      LeapFrogTrajectory track(particle.position,velocity,
          linear?MagneticFieldVector(cs,0_T,0_T,0_T):field,k,time);
      auto endpoint=track.getPosition(1);auto xyz=endpoint.getCoordinates(cs);
      auto direction=track.getDirection(1).getComponents(cs);
      // Continuous loss sees the chord (not the start-tangent competitor).
      auto chord=endpoint-particle.position;
      StraightTrajectory line(Line(particle.position,chord/time),time);
      double column=particle.node->getModelProperties().getIntegratedGrammage(line)/(1_g/(1_cm*1_cm));
      double final_energy=state.energy_GeV-column/1.e6;
      if(record.start.environment_node!=state.environment_node || record.end.environment_node!=full.topology.id(next))
        fail(i,j,"node",record.end.environment_node,full.topology.id(next));
      if(record.end.step_id!=j+1) fail(i,j,"step ID",record.end.step_id,j+1);
      for (int a=0;a<3;++a) {
        double error=std::abs(record.end.position_m[a]-xyz[a]/1_m);max_position=std::max(max_position,error);
        if(error>1.e-7) fail(i,j,"position",record.end.position_m[a],xyz[a]/1_m);
        double dir_error=std::abs(record.end.direction[a]-direction[a].magnitude());max_direction=std::max(max_direction,dir_error);
        if(dir_error>1.e-9)fail(i,j,"direction",record.end.direction[a],direction[a].magnitude());
        state.position_m[a]=xyz[a]/1_m;state.direction[a]=direction[a].magnitude();
      }
      max_column=std::max(max_column,std::abs(record.traversed_grammage_g_per_cm2-column));
      max_energy=std::max(max_energy,std::abs(record.end.energy_GeV-final_energy));
      max_time=std::max(max_time,std::abs(record.end.time_s-(state.time_s+time/1_s)));
      if(std::abs(record.traversed_grammage_g_per_cm2-column)>1.e-7)fail(i,j,"grammage");
      if(std::abs(record.end.energy_GeV-final_energy)>1.e-10)fail(i,j,"linear fixture loss");
      if(std::abs(record.end.time_s-(state.time_s+time/1_s))>1.e-13)fail(i,j,"time");
      if(next!=particle.node)++transitions;else ++magnetic_steps;
      ++steps;state.energy_GeV=final_energy;state.time_s+=time/1_s;state.environment_node=full.topology.id(next);
      if(!next || !next->hasModelProperties()) {
        if(j+1!=sequence.count || record.limit!=em::LeptonTransportLimit::EscapedEnvironment)fail(i,j,"escape not terminal");
        break;
      }
      if(record.end.medium_id!=static_cast<int>(next->getModelProperties().getMedium()))fail(i,j,"medium");
    }
    if(sequence.count==0 || sequence.records[sequence.count-1].limit!=em::LeptonTransportLimit::EscapedEnvironment)
      fail(i,sequence.count,"did not escape in 32 steps");
  }
  return {{"scenario",field_strength==0.?"zero_field":"node_specific_fields"},{"field_strength_T",field_strength},{"particles",input.size()},
    {"steps",steps},{"transitions",transitions},{"magnetic_steps",magnetic_steps},{"errors",errors},{"examples",examples},
    {"max_position_error_m",max_position},{"max_direction_error",max_direction},
    {"max_energy_error_GeV",max_energy},{"max_grammage_error_g_cm2",max_column},{"max_time_error_s",max_time}};
}
int main(int argc,char** argv) {
  initializeDevice(argc,argv);
  nlohmann::json report;report["backend"]=deviceName();
  report["scope"]="actual lepton transport and native C8 tracking; synthetic linear range isolates geometry; not full-shower physics acceptance";
  report["tests"]={exercise(.003),exercise(.02),exercise(0.)};
  report["identity_roundtrip"]=identityRoundtrip();
  bool good=true;for(auto const& t:report["tests"])good=good && t["errors"]==0;
  report["passed"]=good;std::cout<<report.dump(2)<<'\n';finalizeDevice();return good?0:2;
}

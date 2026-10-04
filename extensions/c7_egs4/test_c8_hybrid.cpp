#include "Egs4C8Router.hpp"
#include <corsika/framework/core/HybridCascade.hpp>
#include <corsika/framework/process/ProcessSequence.hpp>
#include <corsika/accelerator/em/common/EnvironmentSnapshotBuilder.hpp>
#include <corsika/media/UniformMagneticField.hpp>
#include <corsika/modules/pythia8/Decay.hpp>
#include <corsika/output/DummyOutputManager.hpp>
#include <corsika/setup/SetupStack.hpp>
#include <corsika/setup/SetupTrajectory.hpp>
#include <iomanip>
#include <iostream>
#include <set>

namespace {
using namespace corsika;
namespace app=c7_egs4::application;
void check(bool b,char const* message){if(!b)throw std::runtime_error(message);}
using MediumInterface=IMagneticFieldModel<IMediumModel>;
using Env=Environment<MediumInterface>;
using Stack=setup::HybridStack<Env>;
struct SecondaryAudit:SecondariesProcess<SecondaryAudit> {
  std::uint64_t count{};std::set<std::uint64_t> ids;
  template<class View>void doSecondaries(View& view) {
    for(auto&& p:view) {
      check(p.getPID()==Code::Photon||p.getPID()==Code::Electron||p.getPID()==Code::Positron,"Unexpected pi0 daughter");
      check(p.getGeneration()==1&&p.getParentHistoryId()!=0&&ids.insert(p.getHistoryId()).second,"C8 secondary ancestry lost");
      ++count;
    }
  }
};
struct RejectUnexpectedHost {
  bool canHandle(app::HostRequest const&)const{return false;}
  std::size_t handle(Stack&,app::HostRequest const&)const{throw std::runtime_error("Unexpected sub-threshold host vertex");}
};
struct Result {
  app::RunStatistics native;
  std::uint64_t scalar_steps{},secondary_count{},max_history{};
  double initial{},deposit{},observed{},discarded{};
  std::vector<double> records;
};
Result run(app::Session& session,bool host_reference,std::size_t batch) {
  auto& random=RNGManager<>::getInstance();random.setSeed(5731);
  // setSeed() changes the key, not the buffered draws/counter. Reconstruct
  // both engines for identical Pythia inputs in every host/device replay.
  random.getRandomStream("cascade")=default_prng_type(5731,0);
  random.getRandomStream("pythia")=default_prng_type(5731,1);
  Env env;auto cs=env.getCoordinateSystem();auto R=constants::EarthRadius::Mean;
  Point center{cs,0_m,0_m,0_m};MagneticFieldVector field{cs,{2.e-5_T,1.e-5_T,-4.e-5_T}};
  create_5layer_atmosphere<MediumInterface,UniformMagneticField>(env,AtmosphereId::LinsleyUSStd,center,field);
  auto snapshot=gpu::em::makeCorsika7AtmosphereSnapshot(AtmosphereId::LinsleyUSStd,
      {0.,0.,0.},3,(R+1100_m)/1_m,{2.e-5,1.e-5,-4.e-5},.2);
  app::Configuration configuration;configuration.environment=snapshot;configuration.earth_radius_m=R/1_m;
  configuration.electron_mass_MeV=get_mass(Code::Electron)/1_MeV;
  configuration.seed=731;configuration.shower_id=129;configuration.maximum_waves=200000;
  configuration.resolve_rare_vertices=true;
  Result result;double mass=configuration.electron_mass_MeV*.001;
  app::OutputCallbacks output;
  output.step=[&](auto const& r){
    result.deposit+=r.deposited_energy_GeV*r.weight;result.max_history=std::max(result.max_history,r.history_id);
    result.records.push_back(r.end_energy_GeV);result.records.push_back(r.deposited_energy_GeV);
  };
  output.radio=[&](auto const& r){check(r.step.pid!=22&&r.step.end_time_s>r.step.start_time_s,"Invalid hybrid radio track");};
  output.observation=[&](auto const& r){auto p=r.particle;
    result.observed+=p.weight*(p.energy_GeV+(p.pid==11?-mass:p.pid==-11?mass:0.));};
  output.discarded=[&](double energy,unsigned){result.discarded+=energy;};
  RejectUnexpectedHost host;auto region=[](auto const* node){if(!node||!node->hasModelProperties())throw std::runtime_error("Missing C8 medium");return 3;};
  app::Router<Stack,RejectUnexpectedHost,decltype(region)> router(session,configuration,output,cs,host,region,batch,host_reference);
  pythia8::Decay decay;decay.setHandleDecay(Code::Pi0);SecondaryAudit audit;
  auto sequence=make_sequence(decay,audit);Stack stack;setup::Tracking tracking(.2);DummyOutputManager manager;
  for(double height:{4000.1,4100.1}) {
    auto p=stack.addParticle(std::make_tuple(Code::Pi0,5_MeV,DirectionVector{cs,{.3,.4,-std::sqrt(.75)}},
      Point{cs,10_m,20_m,R+height*1_m},0_ns));p.setWeight(1.25);result.initial+=p.getEnergy()/1_GeV*p.getWeight();
  }
  HybridCascade<setup::Tracking,decltype(sequence),DummyOutputManager,Stack,decltype(router)> cascade(env,tracking,sequence,manager,stack,router);
  cascade.forceDecay();cascade.run(); // second pion follows actual C8 tracking/lifetime scheduling
  result.native=router.statistics();result.secondary_count=audit.count;
  for(auto const& [pdg,count]:cascade.timingStatistics().scalar_steps_by_pdg) {
    check(pdg==111,"An EM particle escaped routing into the scalar C8 path");result.scalar_steps+=count;
  }
  check(stack.getEntries()==0&&!router.pending(),"C8 hybrid/EM queues did not drain");
  check(result.scalar_steps>=2&&audit.count>=4&&result.native.injections==audit.count,"Real C8 decay-to-EM handoff not exercised");
  check(result.native.children>0&&result.native.steps>0&&result.native.radio_tracks>0&&result.native.host_requests==0,"Incomplete native EM branching/output");
  check(result.max_history>audit.count+2,"Native children did not use shared C8 history allocation");
  check(std::abs(result.initial-result.deposit-result.observed-result.discarded)<1.e-8,"Hybrid decay/EM energy ledger mismatch");
  std::cout<<std::defaultfloat<<std::setprecision(17)<<"execution_space="<<result.native.execution_space<<" batch="<<batch<<" scalar_pi0_steps="<<result.scalar_steps
    <<" c8_decay_daughters="<<audit.count<<" native_injections="<<result.native.injections<<" native_waves="<<result.native.waves
    <<" native_children="<<result.native.children<<" native_steps="<<result.native.steps<<" radio_tracks="<<result.native.radio_tracks
    <<" initial_GeV="<<result.initial<<" deposit_GeV="<<result.deposit<<" observed_GeV="<<result.observed<<" discarded_GeV="<<result.discarded<<'\n';
  return result;
}
}
int main(int argc,char** argv) {
  try {
    if(argc!=2)return 2;logging::set_level(logging::level::err);
    auto& rng=RNGManager<>::getInstance();rng.registerRandomStream("cascade");rng.registerRandomStream("pythia");
    app::Session session(argv[1]);std::cout<<std::setprecision(17);
    for(std::size_t batch:{1,8}) {
      auto host=run(session,true,batch),device=run(session,false,batch);
      check(host.native.children==device.native.children&&host.native.steps==device.native.steps&&host.records.size()==device.records.size(),
        "Hybrid native host/device branching differs");
      double error=0.;for(std::size_t i=0;i<host.records.size();++i)error=std::max(error,std::abs(host.records[i]-device.records[i]));
      check(error<1.e-8,"Hybrid native host/device energy output differs");
      std::cout<<"batch="<<batch<<" host_device_max_record_energy_difference_GeV="<<error<<'\n';
    }
    std::cout<<"scope=real C8 HybridCascade/HybridStack/Pythia8 pi0 decay into native C++ EGS4 EM transport; no PROPOSAL or Fortran EM calls; not high-energy hadronic production/thinning/multicard acceptance\n";
    return 0;
  }catch(std::exception const& e){std::cerr<<e.what()<<'\n';return 1;}
}

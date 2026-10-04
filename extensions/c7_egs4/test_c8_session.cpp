#include "Egs4C8Session.hpp"
#include "AirTestFixture.hpp"
#include <Kokkos_Core.hpp>
#include <iostream>
#include <stdexcept>
#include <set>
#include <iomanip>

namespace {
void check(bool condition,char const* message){if(!condition)throw std::runtime_error(message);}
struct WriteSeven {
  Kokkos::View<int*> value;
  KOKKOS_INLINE_FUNCTION void operator()(int)const{value(0)=7;}
};
template<class F>void mustReject(F&& action,char const* message) {
  bool rejected=false;try{action();}catch(std::exception const&){rejected=true;}
  check(rejected,message);
}
void streamingContracts(char const* table_path,bool host_reference) {
  namespace app=c7_egs4::application;namespace air=c7_egs4::c8_adapter;
  app::Session session(table_path);
  app::Configuration c;c.environment=c7_egs4::air_test::environment();c.earth_radius_m=6.371315e6;
  auto env=air::makeAirEnvironment(c.environment,c.earth_radius_m,10.,air::AirConvention::c7_egs4_four_exponentials);
  auto p=c7_egs4::air_test::input(env,22,.2,4000.,8); // below photon cut: one local deposit
  std::uint64_t steps=0;double deposit=0.;app::OutputCallbacks output;
  output.step=[&](auto const& r){++steps;deposit+=r.deposited_energy_GeV*r.weight;};
  output.radio=[](auto const&){throw std::runtime_error("Subcut photon emitted radio");};
  output.observation=[](auto const&){throw std::runtime_error("Subcut photon reached observation");};
  output.discarded=[](double,unsigned){throw std::runtime_error("Subcut photon discarded");};
  auto invalid_policy=c;invalid_policy.native_photonuclear_vertices=true;
  mustReject([&]{session.begin(invalid_policy,output,host_reference);},"Session accepted native PIGEN without rare resolution");
  mustReject([&]{session.run(invalid_policy,{},output,host_reference);},"Standalone session silently ignored PIGEN policy");
  invalid_policy.resolve_rare_vertices=true;invalid_policy.rho_mass_MeV=0.;
  mustReject([&]{session.begin(invalid_policy,output,host_reference);},"Session accepted missing C7 vector-meson masses");
  invalid_policy=c;invalid_policy.resolve_rare_vertices=true;invalid_policy.native_photonuclear_vertices=true;
  invalid_policy.native_prompt_rho_decay=true;
  mustReject([&]{session.begin(invalid_policy,output,host_reference);},"Prompt rho accepted without a genealogy consumer");
  mustReject([&]{session.run(invalid_policy,{},output,host_reference);},"Standalone rho silently lost genealogy");
  invalid_policy.native_prompt_rho_decay=false;invalid_policy.native_prompt_resonance_decay=true;
  mustReject([&]{session.begin(invalid_policy,output,host_reference);},"Prompt omega/phi accepted without genealogy callback");
  session.begin(c,output,host_reference);
  mustReject([&]{session.begin(c,output,host_reference);},"Stream allowed double begin");
  mustReject([&]{session.run(c,{},output,host_reference);},"Standalone run overlapped active stream");
  session.append({p});mustReject([&]{session.finishEMQueue();},"Nonempty stream finished");
  mustReject([&]{session.append({p});},"Stream reimported same history");
  auto invalid=p;invalid.history_id=4;invalid.medium_id=99;
  mustReject([&]{session.append({invalid});},"Stream accepted unsupported medium");
  check(session.activeParticles()==1,"Rejected append changed live queue");
  auto first=session.advance(100,2);check(first.host_requests.empty()&&first.active_particles==0,"Subcut photon did not terminate");
  p.history_id=2;session.append({p}); // older original CPU history after a newer GPU reservation
  session.advance(102,2);auto stats=session.finishEMQueue();
  check(stats.injections==2&&stats.waves==2&&stats.steps==2&&steps==2&&std::abs(deposit-.0005)<1.e-15&&
    stats.target_rest_energy_GeV==0.&&first.target_rest_energy_GeV==0.,
    "Interleaved injection lost/duplicated energy or output");
  mustReject([&]{session.activeParticles();},"Finished stream remains active");
  // Failure after a callback begins must poison the stream: never replay
  // part of a wave or report success after externally emitted output.
  output.step=[](auto const&){throw std::runtime_error("Injected writer failure");};
  session.begin(c,output,host_reference);session.append({p});
  mustReject([&]{session.advance(100,2);},"Writer failure was swallowed");
  mustReject([&]{session.advance(102,2);},"Failed wave was replayed");
  mustReject([&]{session.append({p});},"Failed stream accepted more work");
  mustReject([&]{session.finishEMQueue();},"Failed stream reported success");
  std::cout<<"native_incremental_contracts="<<(host_reference?"host_reference":"default_backend")<<" passed\n";
}
void persistentContracts(char const* table_path,bool host_reference) {
  namespace app=c7_egs4::application;namespace air=c7_egs4::c8_adapter;
  app::Session session(table_path);app::Configuration c;
  c.environment=c7_egs4::air_test::environment();c.earth_radius_m=6.371315e6;
  c.retain_across_showers=true;c.reuse_queue_workspace=true;
  auto env=air::makeAirEnvironment(c.environment,c.earth_radius_m,10.,air::AirConvention::c7_egs4_four_exponentials);
  auto p=c7_egs4::air_test::input(env,22,.2,4000.,8);
  std::uint64_t steps=0;app::OutputCallbacks output;
  output.step=[&](auto const&){++steps;};output.radio=[](auto const&){};
  output.observation=[](auto const&){};output.discarded=[](double,unsigned){};
  mustReject([&]{session.resume(1,1,output);},"Uninitialized session resumed");
  for(int event=0;event<3;++event) {
    steps=0;
    if(event)session.resume(event==1?732:731,1,output);else session.begin(c,output,host_reference);
    check(!session.hasRetainedShower(),"Retained stream still present while active");
    session.append({p}); // same original history ID is legal in a new event
    mustReject([&]{session.resume(1,1,output);},"Active event reset was allowed");
    session.advance(100,2);auto stats=session.finishEMQueue();
    check(session.hasRetainedShower()&&steps==1&&stats.injections==1&&stats.steps==1&&stats.waves==1,
      "Persistent event retained cumulative counters or callbacks");
    mustReject([&]{session.begin(c,output,host_reference);},"Retained configuration silently changed");
  }
  std::cout<<"native_persistent_contracts="<<(host_reference?"host":"default")<<" passed\n";
}
// A bounded physical-rate cohort exercises the host-facing Session payload,
// not just the device queue. Returned hadrons are energy-accounted sinks in
// this fixture, NOT a completed hadronic shower or a host transport model.
void streamingNuclear(char const* table_path,bool host_reference,bool rho=false,bool resonance=false) {
  namespace app=c7_egs4::application;namespace air=c7_egs4::c8_adapter;
  app::Session session(table_path);app::Configuration c;
  c.environment=c7_egs4::air_test::environment();c.earth_radius_m=6.371315e6;
  c.seed=971;c.shower_id=712;c.queue_capacity=32768;c.maximum_waves=20000;
  c.electron_total_cut_MeV=50.;c.resolve_rare_vertices=true;c.native_photonuclear_vertices=true;
  c.native_prompt_rho_decay=rho;
  c.native_prompt_resonance_decay=resonance;
  // Large primary cohort but no energetic secondary-lepton cascade: this is
  // a physical-rate callback/ledger control, not the shower-physics sample.
  if(resonance)c.electron_total_cut_MeV=10000.;
  auto env=air::makeAirEnvironment(c.environment,c.earth_radius_m,10.,air::AirConvention::c7_egs4_four_exponentials);
  // Higher precision for the test ledger prevents O(N) accumulation error
  // from obscuring per-vertex energy checks in the enlarged callback cohort.
  long double initial=0.,deposit=0.,observed=0.,discarded=0.,exported=0.;double target=0.;std::uint64_t count=0,generated=0;
  std::set<std::uint64_t> exported_ids,rho_ids,rho_children,photon_children,transported_ids;app::OutputCallbacks output;
  int resonance_count=0;
  output.step=[&](auto const& r){deposit+=r.deposited_energy_GeV*r.weight;transported_ids.insert(r.history_id);};
  output.radio=[](auto const&){};
  output.observation=[&](auto const& r){auto p=r.particle;double m=c.electron_mass_MeV*.001;
    observed+=p.weight*(p.energy_GeV+(p.pid==11?-m:p.pid==-11?m:0.));};
  output.discarded=[&](double energy,unsigned){discarded+=energy;};
  output.prompt_decay=[&](app::PromptDecayRecord const& r) {
    check(((rho&&r.parent.pid==113)||(resonance&&(r.parent.pid==223||r.parent.pid==333)))&&
      rho_ids.insert(r.parent.history_id).second,"Duplicate/invalid transient meson record");
    if(r.parent.pid!=113)++resonance_count;
    double sum=0.;
    check(r.count==2||(resonance&&r.count==3),"Meson genealogy has wrong multiplicity");
    for(int j=0;j<r.count;++j) {
      auto const& p=r.daughters[j];
      check(rho_children.insert(p.history_id).second&&p.parent_history_id==r.parent.history_id&&
        p.generation==r.parent.generation+1&&p.weight==r.parent.weight&&p.time_s==r.parent.time_s,
        "Session lost prompt decay ancestry/weight/time");
      sum+=p.energy_GeV;
      if(p.pid==22)photon_children.insert(p.history_id);
    }
    check(std::abs(sum-r.parent.energy_GeV)<1.e-10,"Session prompt rho energy closure failed");
  };
  session.begin(c,output,host_reference);
  std::vector<air::C8Particle> input;
  for(int i=0;i<(resonance?8192:512);++i) {
    auto p=c7_egs4::air_test::input(env,22,resonance?10000.:rho?3000.:1400.,10000.+i,i);initial+=p.energy_GeV*p.weight;input.push_back(p);
  }
  session.append(input);std::uint64_t first=10000;
  while(session.activeParticles()) {
    auto reserved=(resonance?5:rho?4:3)*session.activeParticles();auto wave=session.advance(first,reserved);first+=reserved;
    target+=wave.target_rest_energy_GeV;
    for(auto const& r:wave.host_requests) {
      ++count;auto p=r.particle;exported+=p.energy_GeV*p.weight;
      check(exported_ids.insert(p.history_id).second,"Session duplicated a host request");
      check(!rho_ids.count(p.history_id),"Prompt rho was counted twice as a host energy sink");
      if(rho_ids.count(p.parent_history_id))check(rho_children.count(p.history_id),"Host rho daughter lacks its decay record");
      if(rho)check(p.pid!=113,"Native prompt rho still requested host decay");
      if(resonance)check(p.pid!=223&&p.pid!=333&&!photon_children.count(p.history_id),"Prompt resonance/photon exported twice");
      check(r.random_key.history_id==p.history_id&&r.hadron_random_key.history_id==p.history_id&&
        r.random_key.process_id!=r.hadron_random_key.process_id,"Session lost per-history random keys");
      check(r.kind!=app::HostRequestKind::photonuclear,"Session ignored native photonuclear selection");
      if(r.kind==app::HostRequestKind::hadron_transport||r.kind==app::HostRequestKind::c7_vector_meson_decay) {
        ++generated;check(r.photonuclear_branch>=1&&r.photonuclear_branch<=3&&
          (r.target_pdg==2212||r.target_pdg==2112)&&p.pid!=22,"Session lost generated nuclear metadata");
      }
      if(r.kind==app::HostRequestKind::photonuclear_many_hadrons)
        check(r.photonuclear_branch==4&&p.pid==22,"Session lost selected many-hadron request");
    }
  }
  auto stats=session.finishEMQueue();double closure=initial+target-deposit-observed-discarded-exported;
  std::cout<<std::setprecision(17)<<"native_session_photonuclear="<<(host_reference?"host_reference":"default_backend")
    <<" primaries="<<input.size()<<" waves="<<stats.waves<<" host_requests="<<count<<" generated_hadrons="<<generated
    <<" prompt_decays="<<stats.prompt_decays<<" omega_phi_decays="<<resonance_count<<" reinjected_photons="<<photon_children.size()
    <<" target_rest_energy_GeV="<<target<<" energy_closure_GeV="<<closure<<'\n';
  check(generated>0&&stats.host_requests==count&&stats.target_rest_energy_GeV==target&&target>0.,
    "Native Session nuclear payload/target ledger not exercised");
  check(std::abs(closure)<1.e-8,"Native Session EM/host/medium energy ledger differs");
  check(stats.prompt_decays==rho_ids.size()&&(!rho||stats.prompt_decays>0),"Session did not exercise/record prompt rho decay");
  if(resonance)check(resonance_count>0,"Session omega/phi prompt dispatch not exercised");
  for(auto id:rho_children)check(photon_children.count(id)?transported_ids.count(id):exported_ids.count(id),
    "Recorded prompt daughter never reached its native/host destination");
}
}
int main(int argc,char** argv) {
  try {
    if(argc!=3)return 2;
    bool external=std::string(argv[2])=="external";
    if(external)Kokkos::initialize();
    using Session=c7_egs4::application::Session;
    auto first=std::make_unique<Session>(argv[1]);
    auto second=std::make_unique<Session>(argv[1]);
    // Query only: this must not reserve 90% of the test GPU or alter histories.
#ifdef KOKKOS_ENABLE_CUDA
    if constexpr(std::is_same_v<Kokkos::DefaultExecutionSpace,Kokkos::Cuda>) {
      auto budget=second->gpuMemoryBudget(.9);
      check(budget.fraction==.9&&budget.queue_capacity>=8192&&budget.free_bytes<=budget.total_bytes&&
        std::size_t(budget.queue_capacity)*budget.bytes_per_history<=budget.working_bytes,"CUDA memory plan invalid");
      std::cout<<"native_gpu_budget fraction="<<budget.fraction<<" capacity="<<budget.queue_capacity
        <<" bytes_per_history="<<budget.bytes_per_history<<'\n';
    }
#else
    mustReject([&]{second->gpuMemoryBudget(.9);},"CPU session accepted a GPU budget");
#endif
    first.reset();
    check(Kokkos::is_initialized()&&!Kokkos::is_finalized(),"First session finalized a live shared runtime");
    // Allocation and a real kernel must still work after the initial owner dies.
    {
      Kokkos::View<int*> value("live_session_test",1);
      Kokkos::parallel_for("live_session_test",1,WriteSeven{value});
      auto host=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},value);
      check(host(0)==7,"Runtime no longer executes kernels");
    }
    streamingContracts(argv[1],true);streamingContracts(argv[1],false);
    persistentContracts(argv[1],true);persistentContracts(argv[1],false);
    streamingNuclear(argv[1],true);streamingNuclear(argv[1],false);
    streamingNuclear(argv[1],true,true);streamingNuclear(argv[1],false,true);
    streamingNuclear(argv[1],true,true,true);streamingNuclear(argv[1],false,true,true);
    second.reset();
    if(external) {
      check(Kokkos::is_initialized()&&!Kokkos::is_finalized(),"Session finalized an application-owned runtime");
      // Simulate a caller incorrectly ending its runtime while a session exists.
      auto third=std::make_unique<Session>(argv[1]);
      Kokkos::finalize();bool rejected=false;
      c7_egs4::application::Configuration configuration;
      try{third->run(configuration, {}, {});}catch(std::runtime_error const&){rejected=true;}
      check(rejected,"Session accepted use after external finalization");
      third.reset(); // Must not double-finalize.
    }else check(Kokkos::is_finalized(),"Last owned session did not finalize runtime");
    bool rejected=false;
    try{Session after(argv[1]);}catch(std::runtime_error const&){rejected=true;}
    check(rejected,"Session attempted unsupported Kokkos reinitialization");
    std::cout<<"native_session_runtime="<<(external?"application_owned":"shared_session_owned")<<" passed\n";
    return 0;
  }catch(std::exception const& error){std::cerr<<error.what()<<'\n';return 1;}
}

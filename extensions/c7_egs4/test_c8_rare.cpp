#include "Egs4C8AirWavefront.hpp"
#include "AirTestFixture.hpp"
#include <algorithm>
#include <iomanip>
#include <iostream>

namespace {
using namespace c7_egs4;
using namespace c7_egs4::c8_adapter;
bool splitKernels=false; // Repeat the same independent oracle in both queue modes.
void check(bool value,char const* message){if(!value)throw std::runtime_error(message);}
RareMaterial material(){return {.51099895,105.6583755,139.57039,938.27208816,14.543,{.7847,.2105,.0048}};}
PhotonuclearPolicy nuclearPolicy(bool enabled=true){return {enabled,
  {938.27208816,939.56542052,134.9768,139.57039,1019.461,782.65,775.26},makeTransverseMomentumTable()};}
ChannelThresholds thresholds(bool suppressed=false){return {.51099895,152.,suppressed?10000.:152.,422.};}
AirEnvironment environment(){return makeAirEnvironment(air_test::environment(),6.371315e6,10.,AirConvention::c7_egs4_four_exponentials);}
// Branch coefficients alone are FORCED for adapter/control-path coverage.
// Physical sampling agreement is tested independently against original C7 by
// test_rare.cpp; these are not physical shower rates or a shower comparison.
Tables forcedTables(Tables t,int mode) {
  for(auto& row:t.photons)for(int j=0;j<4;++j){row.c[2+2*j]=(mode==1&&j==0)?0.f:1.f;row.c[3+2*j]=0.f;}
  for(auto& row:t.electrons)for(int j=8;j<18;j+=2){row.c[j]=1.f;row.c[j+1]=0.f;}
  return t;
}
template<class Exec>std::vector<AirRecord> prepare(AirWavefront<Exec>& queue,ModelView model,double remaining=0.) {
  auto r=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},queue.activeRecords());
  std::vector<AirRecord> result;
  for(std::size_t i=0;i<r.extent(0);++i) {
    auto& p=r(i).track.particle;
    if(p.pdg==22)p.photon_clock={remaining,true};
    else p.clock={remaining,electronQuery(model.tables,p.energy_MeV,model.thresholds.mass_MeV,p.pdg==11?-1:1).rate_per_cm,true};
    result.push_back(r(i));
  }
  Kokkos::deep_copy(queue.activeRecords(),r);return result;
}
template<class Exec>void exercise(Tables const& original,int mode,bool suppressed=false,bool transported=false,bool nuclear=false,bool rho=false) {
  auto tables=forcedTables(original,mode);auto env=environment();auto limits=thresholds(suppressed);
  auto model=modelView(tables,limits);TransportSettings settings{1.,.5,.0625};
  auto policy=nuclearPolicy(nuclear);policy.prompt_rho_decay=rho;
  AirWavefront<Exec> queue(tables,limits,env,settings,256,true,material(),policy,{},splitKernels,splitKernels);
  std::vector<C8Particle> input;
  double photon_energy[]={200.,1400.,3000.,10000.},lepton_energy[]={1000.,10000.,1000000.,10000.};
  // The transported ELNUCL 32-input cohort happens to produce only omega,
  // not rho. Expand deterministic coverage rather than bypassing RHOGEN.
  for(int i=0;i<(rho?128:32);++i) {
    double energy=nuclear?(mode==2?lepton_energy[i%4]:photon_energy[i%4]):10000.;
    input.push_back(air_test::input(env,mode==2?(i%2?11:-11):22,energy,10000.+i,i));
  }
  queue.beginShower(input,771,921,1000);auto reference=prepare(queue,model,transported?1.e-5:0.);
  auto batch=queue.advance();auto next_id=std::uint64_t(1000);
  auto transfers=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},batch.host);
  auto after=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},batch.after);
  auto outcomes=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},batch.outcomes);
  auto output=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},batch.outputs);
  auto live=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},queue.activeRecords());
  double difference=0.,closure=0.,target_energy=0.,rho_momentum=0.;int count=0,branches[5]{},host_kinds[5]{},decays=0;
  for(std::size_t i=0;i<reference.size();++i) {
    auto before=reference[i];auto& parent=reference[i];
    auto history=advanceAirRecord(model,parent,env,settings);
    check(history.action==AirAction::needs_host_channel,"Forced native rare channel was not selected");
    auto native_output=makeAirOutput(before,parent,history,env);
    auto expected=resolveRareAirRecord(model,parent,history,env,settings,material(),policy);
    check(expected.valid&&assignHostHistoryIds(expected,parent.metadata,next_id),"Host rare resolution failed");
    next_id+=expected.created;auto got=transfers(i);count+=got.count;
    check(got.valid&&got.count==expected.count&&got.created==expected.created&&got.status==expected.status,"Rare transfer disposition differs");
    check(got.has_rho_decay==expected.has_rho_decay,"Prompt rho disposition differs");
    if(got.has_rho_decay) {
      ++decays;auto a=expected.rho_parent,b=got.rho_parent;
      check(rho&&a.pid==113&&b.pid==113&&a.history_id==b.history_id&&b.parent_history_id==before.metadata.history_id&&
        b.generation==before.metadata.generation+1&&got.created==4&&got.count==3,"Transient rho genealogy lost");
      check(b.weight==before.metadata.weight&&b.time_s==got.vertices[0].particle.time_s,"Prompt rho weight/time differs");
      check(std::abs(got.vertices[0].particle.energy_GeV+got.vertices[1].particle.energy_GeV-b.energy_GeV)<1.e-10,
        "Rho decay energy is missing or double counted");
      double mass=policy.masses.rho_MeV*.001;
      double momentum=std::sqrt((b.energy_GeV-mass)*(b.energy_GeV+mass));
      for(int k=0;k<3;++k) {
        double sum=0.;
        for(int j=0;j<2;++j) {
          auto child=got.vertices[j].particle;
          double m=(child.pid==13||child.pid==-13?material().muon_mass_MeV:policy.masses.charged_pion_MeV)*.001;
          sum+=std::sqrt((child.energy_GeV-m)*(child.energy_GeV+m))*child.direction[k];
          check(std::abs(child.position_m[k]-b.position_m[k])<1.e-8,"Prompt decay moved the collision point");
        }
        rho_momentum=std::max(rho_momentum,std::abs(sum-momentum*b.direction[k])/b.energy_GeV);
      }
      difference=std::max(difference,std::abs(a.energy_GeV-b.energy_GeV));
      for(int k=0;k<3;++k)difference=std::max(difference,std::abs(a.direction[k]-b.direction[k]));
    }
    check(history.action==outcomes(i).action&&parent.random.key.draw_id==after(i).random.key.draw_id,"Rare RNG/action differs");
    check(parent.hadron_random.key.draw_id==after(i).hadron_random.key.draw_id,"Hadronic RNG differs");
    check(output(i).has_step==native_output.has_step&&output(i).has_radio==native_output.has_radio,"Vertex invented a radio segment");
    if(transported)check(output(i).has_radio&&output(i).step.end_energy_GeV>after(i).metadata.energy_GeV&&
      std::abs(output(i).step.end_energy_GeV-native_output.step.end_energy_GeV)<1.e-10,
      "Stochastic ELNUCL energy transfer altered the preceding radio segment");
    double energy=outcomes(i).history.continuous_deposit_MeV*.001;
    if(history.action==AirAction::active) {
      check(live(i).metadata.history_id==before.metadata.history_id&&live(i).track.particle.pending_channel==Channel::invalid,
        "Residual lepton lost its identity or has a pending vertex");
      check(!live(i).track.particle.clock.initialized&&!live(i).track.particle.photon_clock.initialized,"Lepton retained the old interaction clock");
      check(live(i).random.key.draw_id==after(i).random.key.draw_id,"Residual RNG was reset");
      check(live(i).hadron_random.key.draw_id==after(i).hadron_random.key.draw_id,"Residual hadronic RNG was reset");
      energy+=live(i).metadata.energy_GeV;
      difference=std::max(difference,std::abs(live(i).metadata.energy_GeV-parent.metadata.energy_GeV));
      if(suppressed)check(live(i).metadata.energy_GeV==before.metadata.energy_GeV&&got.count==0&&
        live(i).random.key.draw_id==2002,"ELNUCL 1000-trial unchanged-lepton policy differs");
    }
    for(int j=0;j<got.count;++j) {
      auto a=expected.vertices[j],b=got.vertices[j];auto x=a.particle,y=b.particle;energy+=y.energy_GeV;
      ++host_kinds[int(b.kind)];if(j==0)++branches[b.photonuclear_branch];
      check(a.kind==b.kind&&a.source==b.source&&a.new_history==b.new_history&&a.rho_daughter==b.rho_daughter&&a.virtual_photon==b.virtual_photon&&
        a.target_atomic_number==b.target_atomic_number&&x.pid==y.pid&&x.history_id==y.history_id&&
        x.parent_history_id==y.parent_history_id&&x.generation==y.generation&&x.step_id==y.step_id,"Host request metadata differs");
      check(a.photonuclear_branch==b.photonuclear_branch&&a.target_pdg==b.target_pdg&&
        a.random.key.draw_id==b.random.key.draw_id&&a.hadron_random.key.draw_id==b.hadron_random.key.draw_id,
        "Host vertex lost branch or continued RNG");
      if(b.new_history)check(b.random.key.history_id==y.history_id&&b.hadron_random.key.history_id==y.history_id&&
        b.random.key.process_id==NativeEgs4RandomDomain&&b.hadron_random.key.process_id==NativeEgs4HadronRandomDomain&&
        b.random.key.draw_id==0&&b.hadron_random.key.draw_id==0,"New host histories share the parent's RNG stream");
      check(y.weight==before.metadata.weight&&std::abs(y.time_s-parent.metadata.time_s)<1.e-14&&y.medium_id==before.metadata.medium_id,
        "Host request lost weight/time/material");
      if(b.rho_daughter)check(got.has_rho_decay&&y.parent_history_id==got.rho_parent.history_id&&
        y.generation==got.rho_parent.generation+1&&y.history_id!=got.rho_parent.history_id&&
        (y.pid==211||y.pid==-211||y.pid==13||y.pid==-13),"Rho daughter ancestry/PID differs");
      double norm=0.;for(int k=0;k<3;++k) {
        check(std::abs(y.position_m[k]-parent.metadata.position_m[k])<1.e-8,"Host vertex moved location");
        difference=std::max(difference,std::abs(x.direction[k]-y.direction[k]));norm+=y.direction[k]*y.direction[k];
      }
      check(std::abs(norm-1.)<1.e-8,"Invalid host direction");
      difference=std::max(difference,std::abs(x.energy_GeV-y.energy_GeV));
      if(mode==2)check(b.virtual_photon&&b.new_history,"Virtual nuclear vertex lost origin or new identity");
      if(!nuclear&&mode==2)check(b.kind==HostVertexKind::photonuclear,"Virtual photon entered ordinary EM transport");
      if(!nuclear&&mode==1)check(!b.new_history&&!b.virtual_photon&&y.history_id==before.metadata.history_id,"Real photon lost identity");
      if(nuclear) {
        check(b.kind!=HostVertexKind::photonuclear,"Native PIGEN policy returned an unselected nuclear vertex");
        if(b.photonuclear_branch==4) {
          check(b.kind==HostVertexKind::photonuclear_many_hadrons&&y.pid==22&&got.count==1&&got.target_rest_energy_MeV==0.,
            "Many-hadron request invented particles/target energy");
          check(b.target_pdg==1000070140||b.target_pdg==1000080160||b.target_pdg==1000180400,
            "SDPM air target not selected before host handoff");
          if(mode==1)check(!b.new_history&&y.history_id==before.metadata.history_id&&
            b.random.key.draw_id==after(i).random.key.draw_id,"Selected real-photon host request lost its continued RNG");
        }else {
          check(b.new_history&&y.pid!=22&&got.target_rest_energy_MeV>0.,"Native nuclear vertex was not realized");
          check(got.count==(b.photonuclear_branch==2||got.has_rho_decay?3:2),"Native pion/meson/recoil multiplicity lost");
          if(b.kind==HostVertexKind::c7_vector_meson_decay)check(y.pid==113||y.pid==223||y.pid==333,"Non-meson requested C7 vector decay");
          if(rho)check(y.pid!=113,"Prompt rho still escaped as an unresolved host decay");
        }
      }
    }
    target_energy+=got.target_rest_energy_MeV*.001;
    check(got.target_rest_energy_MeV==expected.target_rest_energy_MeV,"Target rest energy differs");
    closure=std::max(closure,std::abs(energy-before.metadata.energy_GeV-got.target_rest_energy_MeV*.001));
  }
  check(next_id==queue.nextHistoryId()&&int(next_id-1000)==batch.created,"Host child IDs overlap or are missing");
  check(queue.activeCount()==(mode==2?input.size():0),"Muon/virtual photon leaked into the EM queue");
  check(closure<1.e-10&&difference<1.e-9,"Rare adapter energy/direction comparison failed");
  if(nuclear&&mode==1)for(int branch=1;branch<=4;++branch)check(branches[branch]>0,"Missing native photonuclear branch coverage");
  if(rho)check(decays>0&&rho_momentum<1.e-7,"Prompt rho path/rotated momentum closure failed");
  std::cout<<"forced_mode="<<mode<<" suppressed="<<suppressed<<" transported="<<transported<<" native_nuclear="<<nuclear<<" primaries="<<input.size()<<" host_requests="<<count
    <<" retained_leptons="<<queue.activeCount()<<" allocated_children="<<batch.created
    <<" prompt_rho_decays="<<decays<<" rho_momentum_over_energy="<<rho_momentum
    <<" max_energy_closure_GeV="<<closure<<" max_host_device_difference="<<difference<<" target_rest_energy_GeV="<<target_energy<<" branches=";
  for(auto n:branches)std::cout<<n<<',';std::cout<<" host_kinds=";for(auto n:host_kinds)std::cout<<n<<',';std::cout<<'\n';
}
template<class Exec>void failures(Tables const& original) {
  auto tables=forcedTables(original,0);auto env=environment();auto limits=thresholds();auto model=modelView(tables,limits);
  auto p=air_test::input(env,22,10000.,10000.,0);bool failed=false;
  try{AirWavefront<Exec> bad(tables,limits,env,{1.,.5,.0625},8,true,{});}catch(std::invalid_argument const&){failed=true;}
  check(failed,"Missing rare material accepted");
  AirWavefront<Exec> queue(tables,limits,env,{1.,.5,.0625},8,true,material(),{},{},splitKernels,splitKernels);
  queue.beginShower({p},7,9,~std::uint64_t{0});auto before=prepare(queue,model);
  for(int attempt=0;attempt<2;++attempt) {
    failed=false;try{queue.advance();}catch(std::length_error const&){failed=true;}
    auto current=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},queue.activeRecords());
    check(failed&&queue.nextHistoryId()==~std::uint64_t{0}&&queue.activeCount()==1&&current(0).random.key.draw_id==0,
      "Rare child-ID overflow changed live queue/RNG");
  }
  queue.beginShower({p},7,9,1000);prepare(queue,model);
  auto state=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},queue.activeRecords());
  state(0).random.key.draw_id=~std::uint64_t{0}-1;Kokkos::deep_copy(queue.activeRecords(),state);
  failed=false;try{queue.advance();}catch(std::runtime_error const&){failed=true;}
  auto current=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},queue.activeRecords());
  check(failed&&queue.nextHistoryId()==1000&&queue.activeCount()==1&&current(0).random.key.draw_id==~std::uint64_t{0}-1,
    "Rare random failure consumed live state");
}
template<class Exec>void photonuclearFailures(Tables const& original) {
  auto tables=forcedTables(original,1);auto env=environment();auto limits=thresholds();auto model=modelView(tables,limits);
  auto policy=nuclearPolicy();auto p=air_test::input(env,22,1400.,10000.,0);bool failed=false;
  try{AirWavefront<Exec> bad(tables,limits,env,{1.,.5,.0625},8,false,material(),policy);}
  catch(std::invalid_argument const&){failed=true;}
  check(failed,"Native nuclear policy allowed unresolved rare vertices");
  auto wrong=policy;wrong.masses.charged_pion_MeV+=1.;failed=false;
  try{AirWavefront<Exec> bad(tables,limits,env,{1.,.5,.0625},8,true,material(),wrong);}
  catch(std::invalid_argument const&){failed=true;}
  check(failed,"Inconsistent ELNUCL/PIGEN pion masses accepted");
  AirWavefront<Exec> queue(tables,limits,env,{1.,.5,.0625},8,true,material(),policy,{},splitKernels,splitKernels);
  queue.beginShower({p},7,9,1000);prepare(queue,model);
  // At 1400 MeV C7 selects the three-output PIGEN2 branch. A failed
  // reservation must be retryable with BOTH RNG streams unchanged.
  for(int attempt=0;attempt<2;++attempt) {
    failed=false;try{queue.advance(1000,2);}catch(std::length_error const&){failed=true;}
    auto live=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},queue.activeRecords());
    check(failed&&queue.activeCount()==1&&queue.nextHistoryId()==1000&&
      live(0).random.key.draw_id==0&&live(0).hadron_random.key.draw_id==0,
      "Three-child reservation failure changed parent/RNG/IDs");
  }
  auto wave=queue.advance(1000,3);auto out=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},wave.host);
  check(wave.created==3&&out(0).count==3&&queue.nextHistoryId()==1003&&!queue.activeCount(),
    "Three-child retry did not preserve the native PIGEN2 vertex");
  for(int j=0;j<3;++j)check(out(0).vertices[j].particle.history_id==std::uint64_t(1000+j),"Nuclear child IDs overlap");
  queue.beginShower({p},7,9,1000);prepare(queue,model);
  auto saved=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},queue.activeRecords());
  saved(0).hadron_random.key.draw_id=~std::uint64_t{0};Kokkos::deep_copy(queue.activeRecords(),saved);
  failed=false;try{queue.advance(1000,3);}catch(std::runtime_error const&){failed=true;}
  auto live=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},queue.activeRecords());
  check(failed&&queue.activeCount()==1&&queue.nextHistoryId()==1000&&live(0).random.key.draw_id==0&&
    live(0).hadron_random.key.draw_id==~std::uint64_t{0},"PTRANS RNG exhaustion committed part of the nuclear vertex");
  std::cout<<"native_photonuclear_transaction_contracts=passed\n";
}
template<class Exec>void rhoContracts(Tables const& original) {
  auto tables=forcedTables(original,1);auto env=environment();auto limits=thresholds();auto model=modelView(tables,limits);
  auto policy=nuclearPolicy();policy.prompt_rho_decay=true;TransportSettings settings{1.,.5,.0625};
  auto bad=policy;bad.enabled=false;bool failed=false;
  try{AirWavefront<Exec> q(tables,limits,env,settings,8,true,material(),bad);}catch(std::invalid_argument const&){failed=true;}
  check(failed,"Rho decay enabled without native nuclear generation");
  AirWavefront<Exec> queue(tables,limits,env,settings,8,true,material(),policy,{},splitKernels,splitKernels);
  C8Particle p;AirRecord before;bool found=false;
  // Select a real native RHOGEN->rho input, without editing a generated PID
  // or bypassing PIGEN. Forced nuclear rates only select the vertex category.
  for(int i=0;i<64&&!found;++i) {
    p=air_test::input(env,22,10000.,10000.,i);queue.beginShower({p},771,921,1000);
    before=prepare(queue,model)[0];auto candidate=before;auto outcome=advanceAirRecord(model,candidate,env,settings);
    auto transfer=resolveRareAirRecord(model,candidate,outcome,env,settings,material(),nuclearPolicy());
    found=transfer.valid&&transfer.count==2&&transfer.vertices[0].particle.pid==113;
  }
  check(found,"No deterministic rho input found");
  // Force the rare muon branch by selecting a known location on the existing
  // Philox stream, not by changing the decay kernel or its branching ratio.
  auto scan=before.hadron_random;std::uint64_t selected=0;found=false;
  for(int i=0;i<1000000&&!found;++i) {
    selected=scan.key.draw_id;double u;check(scan.next(u),"Cannot scan hadronic test stream");found=u>=.999955;
  }
  check(found,"Rare rho muon branch not covered");
  auto reset=[&](std::uint64_t draw) {
    queue.beginShower({p},771,921,1000);prepare(queue,model);
    auto h=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},queue.activeRecords());
    h(0).hadron_random.key.draw_id=draw;Kokkos::deep_copy(queue.activeRecords(),h);
  };
  reset(selected);
  for(int attempt=0;attempt<2;++attempt) {
    failed=false;try{queue.advance(1000,3);}catch(std::length_error const&){failed=true;}
    auto live=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},queue.activeRecords());
    check(failed&&queue.activeCount()==1&&queue.nextHistoryId()==1000&&live(0).random.key.draw_id==0&&
      live(0).hadron_random.key.draw_id==selected,"Four-ID rho failure consumed live state");
  }
  auto batch=queue.advance(1000,4);auto h=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},batch.host);auto t=h(0);
  check(!queue.activeCount()&&batch.created==4&&queue.nextHistoryId()==1004&&t.has_rho_decay&&t.count==3,
    "Rho four-ID retry failed");
  check(t.rho_parent.history_id==1000&&t.rho_parent.parent_history_id==p.history_id,"Rho intermediate ancestry failed");
  for(int j=0;j<3;++j) {
    auto v=t.vertices[j];check(v.particle.history_id==std::uint64_t(1001+j),"Prompt decay has duplicate/missing IDs");
    check(v.particle.parent_history_id==(j<2?1000:p.history_id),"Recoil/rho daughter parent mixed up");
    check(v.particle.generation==p.generation+(j<2?2:1),"Rho/recoil generation mismatch");
  }
  check(t.vertices[0].kind==HostVertexKind::muon_transport&&t.vertices[0].particle.pid==13&&
    t.vertices[1].kind==HostVertexKind::muon_transport&&t.vertices[1].particle.pid==-13,"Rare rho muons not exported");
  check(t.vertices[0].polarization_cosine==-t.vertices[1].polarization_cosine&&
    std::abs(t.vertices[1].polarization_azimuth-t.vertices[0].polarization_azimuth-std::acos(-1.))<1.e-14,
    "Rho muon polarization lost");
  reset(~std::uint64_t{0}-1);failed=false;
  try{queue.advance(1000,4);}catch(std::runtime_error const&){failed=true;}
  auto live=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},queue.activeRecords());
  check(failed&&queue.activeCount()==1&&queue.nextHistoryId()==1000&&live(0).random.key.draw_id==0&&
    live(0).hadron_random.key.draw_id==~std::uint64_t{0}-1,"Prompt rho random failure committed a partial decay");
  p.generation=~std::uint32_t{0}-1;reset(selected);failed=false;
  try{queue.advance(1000,4);}catch(std::runtime_error const&){failed=true;}
  check(failed&&queue.activeCount()==1&&queue.nextHistoryId()==1000,"Two-generation rho overflow committed the queue");
  std::cout<<"native_prompt_rho_contracts=passed rare_muon_stream_offset="<<selected<<'\n';
}
template<class Exec>void resonanceContracts(Tables const& original) {
  auto tables=forcedTables(original,1);auto env=environment();auto limits=thresholds();auto model=modelView(tables,limits);
  auto policy=nuclearPolicy();policy.prompt_resonance_decay=true;
  policy.resonance={782.65,1019.461,139.57039,134.9768,105.6583755,493.677,497.611,497.611,547.862};
  TransportSettings settings{1.,.5,.0625};
  auto bad=policy;bad.resonance.omega_MeV+=1.;bool failed=false;
  try{AirWavefront<Exec> q(tables,limits,env,settings,8,true,material(),bad);}catch(std::invalid_argument const&){failed=true;}
  check(failed,"Inconsistent RHOGEN/RESDEC masses accepted");
  AirWavefront<Exec> queue(tables,limits,env,settings,8,true,material(),policy,{},splitKernels,splitKernels);
  int modes=0,native_photons=0;double max_error=0.;
  for(int pid:{223,333}) {
    C8Particle primary;AirRecord before;RareAirTransfer undecayed;bool found=false;
    // Keep actual native RHOGEN kinematics. Only rates are forced; no
    // post-generation PID editing or artificial resonance daughters.
    for(int i=0;i<1000&&!found;++i) {
      primary=air_test::input(env,22,10000.,10000.,i);queue.beginShower({primary},771,921,10000);
      before=prepare(queue,model)[0];auto candidate=before;auto result=advanceAirRecord(model,candidate,env,settings);
      undecayed=resolveRareAirRecord(model,candidate,result,env,settings,material(),nuclearPolicy());
      found=undecayed.valid&&undecayed.count==2&&undecayed.vertices[0].particle.pid==pid;
    }
    check(found,"Missing RHOGEN omega/phi input");
    double omega[]{0.,.8996252,.9843332,.9997739,.9999090,1.};
    double phi[]{0.,.4901808,.8330066,.9865765,.9996981,.9999857,1.};
    auto boundaries=pid==223?omega:phi;int count=pid==223?5:6;
    for(int mode=0;mode<count;++mode) {
      auto scan=before.hadron_random;std::uint64_t selected=0;found=false;
      for(int j=0;j<2000000&&!found;++j) {
        selected=scan.key.draw_id;double u;check(scan.next(u),"Cannot scan resonance test stream");
        found=u>boundaries[mode]&&u<=boundaries[mode+1];
      }
      check(found,"Missing RESDEC conditional channel coverage");
      auto reset=[&](std::uint64_t draw) {
        queue.beginShower({primary},771,921,10000);prepare(queue,model);
        auto h=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},queue.activeRecords());
        h(0).hadron_random.key.draw_id=draw;Kokkos::deep_copy(queue.activeRecords(),h);
      };
      auto candidate=before;candidate.hadron_random.key.draw_id=selected;
      auto outcome=advanceAirRecord(model,candidate,env,settings);
      auto expected=resolveRareAirRecord(model,candidate,outcome,env,settings,material(),policy);
      check(expected.valid&&expected.has_resonance_decay&&assignHostHistoryIds(expected,candidate.metadata,10000),
        "Host prompt RESDEC composition failed");
      reset(selected);failed=false;
      try{queue.advance(10000,expected.created-1);}catch(std::length_error const&){failed=true;}
      auto unchanged=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},queue.activeRecords());
      check(failed&&queue.activeCount()==1&&queue.nextHistoryId()==10000&&unchanged(0).random.key.draw_id==0&&
        unchanged(0).hadron_random.key.draw_id==selected,"RESDEC ID failure partially committed vertex");
      auto batch=queue.advance(10000,expected.created);
      auto h=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},batch.host);auto t=h(0);
      auto after=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},batch.after);
      auto active=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},queue.activeRecords());
      check(t.has_resonance_decay&&!t.has_rho_decay&&t.count==expected.count&&t.created==expected.created&&
        t.resonance_children==expected.resonance_children&&t.resonance_parent.pid==pid&&
        t.resonance_parent.history_id==10000&&batch.created==t.created,"RESDEC disposition/genealogy lost");
      check(after(0).random.key.draw_id==candidate.random.key.draw_id&&
        after(0).hadron_random.key.draw_id==candidate.hadron_random.key.draw_id,"RESDEC random continuation differs");
      double sum=0.,daughter_sum=0.;int photons=0;
      for(int j=0;j<t.count;++j) {
        auto a=expected.vertices[j],b=t.vertices[j];auto p=b.particle;
        check(a.kind==b.kind&&a.particle.pid==p.pid&&p.history_id==std::uint64_t(10001+j)&&
          b.resonance_daughter==(j<t.resonance_children)&&p.parent_history_id==(j<t.resonance_children?10000:primary.history_id)&&
          p.generation==primary.generation+(j<t.resonance_children?2:1),"RESDEC daughter/recoil IDs mixed");
        check(b.random.key.draw_id==0&&b.hadron_random.key.draw_id==0&&b.random.key.history_id==p.history_id,
          "RESDEC daughter RNG not independent");
        max_error=std::max(max_error,std::abs(a.particle.energy_GeV-p.energy_GeV));
        for(int k=0;k<3;++k)max_error=std::max(max_error,std::abs(a.particle.direction[k]-p.direction[k]));
        sum+=p.energy_GeV;if(j<t.resonance_children)daughter_sum+=p.energy_GeV;
        if(b.kind==HostVertexKind::native_em) {
          check(p.pid==22&&std::size_t(photons)<active.extent(0),"Native decay photon not reinjected");
          auto r=active(photons++);
          check(r.metadata.history_id==p.history_id&&r.metadata.parent_history_id==10000&&
            r.track.particle.pdg==22&&!r.track.particle.clock.initialized&&!r.track.particle.photon_clock.initialized&&
            r.track.particle.pending_channel==Channel::invalid&&r.random.key.history_id==p.history_id,
            "Reinjected decay photon lost clocks/ancestry");
          C8Particle exported=r.metadata;check(exportAirParticle(r.track,env,exported),"Reinjected photon frame invalid");
          for(int k=0;k<3;++k)check(std::abs(exported.direction[k]-p.direction[k])<1.e-10&&
            std::abs(exported.position_m[k]-p.position_m[k])<1.e-8,"Reinjected photon has wrong frame");
        }
      }
      check(queue.activeCount()==std::size_t(photons)&&std::abs(sum-primary.energy_GeV-t.target_rest_energy_MeV*.001)<1.e-10&&
        std::abs(daughter_sum-t.resonance_parent.energy_GeV)<1.e-10,"RESDEC energy duplicated/lost");
      if(photons) {auto wave=queue.advance();check(wave.before.extent(0)==std::size_t(photons),"Decay photon never advanced");}
      native_photons+=photons;++modes;
      reset(~std::uint64_t{0}-1);failed=false;
      try{queue.advance(10000,5);}catch(std::runtime_error const&){failed=true;}
      unchanged=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},queue.activeRecords());
      check(failed&&queue.activeCount()==1&&queue.nextHistoryId()==10000&&unchanged(0).random.key.draw_id==0&&
        unchanged(0).hadron_random.key.draw_id==~std::uint64_t{0}-1,"RESDEC RNG failure committed state");
    }
  }
  check(modes==11&&native_photons==3&&max_error<1.e-9,"Incomplete resident RESDEC channel coverage");
  std::cout<<"native_prompt_resonance_contracts=passed channels="<<modes<<" reinjected_photons="<<native_photons
    <<" max_host_device_difference="<<max_error<<'\n';
}
}
int main(int argc,char** argv) {
  try {
    if(argc!=2)return 2;Kokkos::ScopeGuard guard(argc,argv);auto tables=readAirTables(argv[1]);
    using Exec=Kokkos::DefaultExecutionSpace;
    std::cout<<std::setprecision(17)<<"execution_space="<<Exec::name()<<'\n';
    for(bool split:{false,true}) {
    splitKernels=split;std::cout<<"split_rare_kernels="<<split<<'\n';
    for(int mode=0;mode<3;++mode)exercise<Exec>(tables,mode);
    exercise<Exec>(tables,2,true);exercise<Exec>(tables,2,false,true);
    exercise<Exec>(tables,1,false,false,true);exercise<Exec>(tables,2,false,false,true);
    exercise<Exec>(tables,2,false,true,true);failures<Exec>(tables);photonuclearFailures<Exec>(tables);
    exercise<Exec>(tables,1,false,false,true,true);exercise<Exec>(tables,2,false,false,true,true);
    exercise<Exec>(tables,2,false,true,true,true);rhoContracts<Exec>(tables);resonanceContracts<Exec>(tables);
    }
    std::cout<<"scope=native rare/PIGEN/prompt rho/omega/phi vertices integrated into C8-carrier wavefront; forced rates only for control coverage, no complete hadronic transport or high-energy generator claim\n";
    return 0;
  }catch(std::exception const& e){std::cerr<<e.what()<<'\n';return 1;}
}

#include "Egs4C8AirWavefront.hpp"
#include "AirTestFixture.hpp"
#include <algorithm>
#include <iomanip>
#include <iostream>

namespace {
using namespace c7_egs4;
using namespace c7_egs4::c8_adapter;
void check(bool b,char const* message){if(!b)throw std::runtime_error(message);}
ChannelThresholds thresholds(){return {.51099895,140.,140.,422.};}
AirEnvironment environment(){return makeAirEnvironment(air_test::environment(),6.371315e6,10.,AirConvention::c7_egs4_four_exponentials);}
struct Sink {
  int steps{},radio{},observations{},point_deposits{};
  double deposit{},observed_energy{};
  void onStep(::corsika::gpu::em::EmStepRecord const& r) {
    ++steps;deposit+=1000.*r.deposited_energy_GeV*r.weight;
    check(r.process_id==0&&r.cut_deposited_energy_GeV<=r.deposited_energy_GeV,"Invalid native deposition/process record");
    check(r.end_time_s>=r.start_time_s,"Backwards output time");
    if(r.end_time_s==r.start_time_s) {
      ++point_deposits;
      for(int j=0;j<3;++j)check(r.start_position_m[j]==r.end_position_m[j],"Point deposit has fictitious path");
    }
  }
  void onRadioTrack(::corsika::gpu::em::RadioTrackRecord const& r) {
    ++radio;check(r.step.pid!=22&&r.step.end_time_s>r.step.start_time_s,"Invalid radio species/time");
    double length=0.;for(int j=0;j<3;++j)length+=std::pow(r.step.end_position_m[j]-r.step.start_position_m[j],2);
    length=std::sqrt(length);check(length>0.,"Zero-length radio track");
    for(int j=0;j<3;++j) {
      auto u=(r.step.end_position_m[j]-r.step.start_position_m[j])/length;
      check(std::abs(r.start_direction[j]-u)<1.e-12&&r.start_direction[j]==r.end_direction[j],"Radio did not use endpoint chord");
    }
  }
  void onObservation(::corsika::gpu::em::ObservationRecord const& r) {
    ++observations;auto p=r.particle;
    check(r.status==::corsika::gpu::em::ObservationStatus::ReachedObservationSurface,"Wrong observation status");
    double energy=p.energy_GeV*1000.;
    observed_energy+=(p.pid==11?energy-thresholds().mass_MeV:p.pid==-11?energy+thresholds().mass_MeV:energy)*p.weight;
  }
};
void hash(std::uint64_t& x,std::uint64_t y){x=(x^y)*1099511628211ULL;}
struct Totals {
  int waves{},histories{},maximum_queue{},layer_changes{},discarded{},handoffs{};
  Sink sink;double initial{},escaped{},max_local_closure{},energy_error{},position_error{};
  std::uint64_t control_hash{1469598103934665603ULL};
};
AirRecord referenceChild(AirRecord parent,CurvedTrack track,AirEnvironment const& env,std::uint64_t id) {
  parent.track=track;auto& p=parent.metadata;
  check(exportAirParticle(track,env,p),"Reference child export failed");
  p.parent_history_id=p.history_id;p.history_id=id;p.step_id=0;++p.generation;
  parent.random.key={parent.random.key.seed,parent.random.key.shower_id,id,0,NativeEgs4RandomDomain,0};
  parent.hadron_random.key={parent.hadron_random.key.seed,parent.hadron_random.key.shower_id,id,0,NativeEgs4HadronRandomDomain,0};return parent;
}
template<class Exec> Totals cascade(Tables const& tables,double stepfc,double energy,bool reuse=false) {
  auto env=environment();auto model=modelView(tables,thresholds());TransportSettings settings{1.,.5,stepfc};
  AirWavefront<Exec> queue(tables,thresholds(),env,settings,2048,false,{}, {}, {},reuse);
  std::vector<C8Particle> primaries;
  for(int pdg:{11,-11,22})for(double height:{1100.001,10000.1})for(int i=0;i<2;++i)
    primaries.push_back(air_test::input(env,pdg,energy,height,primaries.size()));
  queue.beginShower(primaries,731,129,primaries.size()+1);
  std::vector<AirRecord> reference;Totals total;
  for(auto p:primaries) {
    AirRecord r;check(importAirRecord(p,env,731,129,r),"Import primary");reference.push_back(r);
    total.initial+=terminationEnergy(r.track.particle,thresholds().mass_MeV)*p.weight;
  }
  auto next_id=std::uint64_t(primaries.size()+1);
  while(queue.activeCount()) {
    check(++total.waves<20000,"Native atmospheric cascade did not terminate");
    total.maximum_queue=std::max(total.maximum_queue,int(queue.activeCount()));
    check(reference.size()==queue.activeCount(),"Air queue length differs");
    typename AirWavefront<Exec>::Batch batch;
    try{batch=queue.advance();}catch(std::exception const&) {
      for(auto r:reference) {
        auto result=advanceAirRecord(model,r,env,settings);
        if(result.action==AirAction::error)std::cerr<<"error_history="<<r.metadata.history_id<<" species="<<r.metadata.pid
          <<" E="<<r.track.particle.energy_MeV<<" region="<<r.region<<" z="<<r.track.particle.position_cm.z
          <<" code="<<int(result.history.error)<<'\n';
      }
      throw;
    }
    auto results=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},batch.outcomes);
    auto records=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},batch.after);
    auto outputs=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},batch.outputs);
    std::vector<AirRecord> next;int created=0;
    for(std::size_t i=0;i<reference.size();++i) {
      auto before=reference[i];auto r=advanceAirRecord(model,reference[i],env,settings);auto d=results(i);auto record=records(i);
      check(r.action==d.action&&r.history.child_count==d.history.child_count&&r.history.channel==d.history.channel&&
        reference[i].random.key.draw_id==record.random.key.draw_id,"Air cascade control/RNG differs");
      check(record.region==reference[i].region&&record.metadata.history_id==reference[i].metadata.history_id&&
        record.metadata.parent_history_id==reference[i].metadata.parent_history_id&&
        record.metadata.generation==reference[i].metadata.generation&&record.metadata.step_id==reference[i].metadata.step_id,
        "Air metadata/region differs");
      auto const& h=d.history;
      double accounted=h.continuous_deposit_MeV+h.vertex_deposit_MeV+d.discarded_energy_MeV;
      if(d.action==AirAction::replaced)for(int j=0;j<h.child_count;++j)accounted+=terminationEnergy(h.children[j],thresholds().mass_MeV);
      if(d.action==AirAction::active||d.action==AirAction::observed||d.action==AirAction::needs_host_channel)
        accounted+=terminationEnergy(record.track.particle,thresholds().mass_MeV);
      total.max_local_closure=std::max(total.max_local_closure,std::abs(accounted-terminationEnergy(before.track.particle,thresholds().mass_MeV)));
      total.energy_error=std::max(total.energy_error,std::abs(reference[i].track.particle.energy_MeV-record.track.particle.energy_MeV));
      for(int j=0;j<3;++j)total.position_error=std::max(total.position_error,std::abs(reference[i].metadata.position_m[j]-record.metadata.position_m[j]));
      total.layer_changes+=record.region!=before.region;total.discarded+=d.action==AirAction::discarded;
      total.handoffs+=d.action==AirAction::needs_host_channel;
      total.escaped+=d.discarded_energy_MeV*record.metadata.weight;
      streamAirOutput(outputs(i),total.sink);
      hash(total.control_hash,std::uint64_t(d.action));hash(total.control_hash,std::uint64_t(h.channel));
      hash(total.control_hash,record.random.key.draw_id);hash(total.control_hash,record.metadata.history_id);
      check(d.action!=AirAction::error&&d.action!=AirAction::needs_host_channel,"Unexpected low-energy handoff/error");
      if(r.action==AirAction::active)next.push_back(reference[i]);
      if(r.action==AirAction::replaced)for(int j=0;j<r.history.child_count;++j)
        next.push_back(referenceChild(reference[i],r.children[j],env,next_id+created++));
    }
    next_id+=created;check(next_id==queue.nextHistoryId()&&created==batch.created,"Air child ID prefix differs");
    reference=std::move(next);
  }
  total.histories=int(next_id-1);
  check(total.sink.steps>0&&total.sink.radio>0&&total.sink.observations>0&&total.histories>int(primaries.size()),"Missing branch/output coverage");
  check(total.max_local_closure<1.e-8&&std::abs(total.initial-total.sink.deposit-total.sink.observed_energy-total.escaped)<1.e-7,"Atmospheric cascade energy balance");
  check(total.energy_error<1.e-7&&total.position_error<.01,"Atmospheric host/device mismatch");
  check(!queue.advance().outputs.extent(0),"Empty queue emitted output");return total;
}
template<class Exec>void prefixContracts() {
  using Queue=AirWavefront<Exec>;
  for(int n:{0,1,255,256,257,4099}) {
    typename Queue::Outcomes outcomes("prefix_outcomes",n);
    typename Queue::HostTransfers transfers("prefix_transfers",n);
    typename Queue::ThinResults thinning("prefix_thinning",n);
    typename Queue::Offsets live("prefix_live",n),children("prefix_children",n);
    auto r=Kokkos::create_mirror_view(outcomes);auto h=Kokkos::create_mirror_view(transfers);
    auto t=Kokkos::create_mirror_view(thinning);
    std::vector<int> expected_live,expected_children;int total_live=0,total_children=0;
    for(int i=0;i<n;++i) {
      expected_live.push_back(total_live);expected_children.push_back(total_children);
      r(i).action=i%4==0?AirAction::active:i%4==1?AirAction::replaced:AirAction::needs_host_channel;
      if(r(i).action==AirAction::active)++total_live;
      if(r(i).action==AirAction::replaced) {
        r(i).history.child_count=2;t(i).keep_mask=(i/4)%4;total_children+=2;
        total_live+=(t(i).keep_mask&1U)+((t(i).keep_mask>>1)&1U);
      }
      if(i%4>=2) {
        h(i).created=5;h(i).count=2;total_children+=5;
        h(i).vertices[0].kind=HostVertexKind::native_em;
        h(i).vertices[1].kind=HostVertexKind::muon_transport;++total_live;
      }
    }
    Kokkos::deep_copy(outcomes,r);Kokkos::deep_copy(transfers,h);Kokkos::deep_copy(thinning,t);
    typename Queue::PrefixCounts result;
    Kokkos::parallel_scan("test_combined_offsets",Kokkos::RangePolicy<Exec>(0,n),
      typename Queue::CombinedOffset{outcomes,transfers,live,children,thinning},result);
    auto l=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},live);
    auto c=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},children);
    check(result.live==total_live&&result.created==total_children,"Combined scan total differs");
    for(int i=0;i<n;++i)check(l(i)==expected_live[i]&&c(i)==expected_children[i],
      "Combined scan changed stable live/history offsets");
  }
}
template<class Exec>void contracts(Tables const& tables,bool reuse) {
  auto env=environment();AirWavefront<Exec> small(tables,thresholds(),env,{1.,.5,1.},1,false,{},{},{},reuse);
  auto p=air_test::input(env,-11,.8,10000.,0);small.beginShower({p},1,2,2);
  for(int i=0;i<2;++i) {
    bool failed=false;try{small.advance();}catch(std::length_error const&){failed=true;}
    check(failed&&small.activeCount()==1&&small.nextHistoryId()==2,"Air overflow not atomic");
    auto saved=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},small.activeRecords());
    check(saved(0).random.key.draw_id==0&&saved(0).metadata.step_id==4,"Air overflow changed RNG/step");
  }
  // Queue limits are storage guards, not physics/RNG inputs. Raising the
  // limit for Fe must preserve the same committed vertex and daughter IDs.
  AirWavefront<Exec> exact(tables,thresholds(),env,{1.,.5,1.},2,false,{},{},{},reuse);
  AirWavefront<Exec> enlarged(tables,thresholds(),env,{1.,.5,1.},1048576,false,{},{},{},reuse);
  exact.beginShower({p},1,2,2);enlarged.beginShower({p},1,2,2);
  auto a=exact.advance(),bigger=enlarged.advance();
  check(a.created==bigger.created&&exact.nextHistoryId()==enlarged.nextHistoryId(),"Capacity changed child IDs");
  auto ra=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},exact.activeRecords());
  auto rb=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},enlarged.activeRecords());
  check(ra.extent(0)==rb.extent(0),"Capacity changed retained daughter count");
  for(std::size_t i=0;i<ra.extent(0);++i) {
    auto const& x=ra(i);auto const& y=rb(i);
    check(x.metadata.history_id==y.metadata.history_id&&x.metadata.weight==y.metadata.weight&&
      x.random.key.history_id==y.random.key.history_id&&x.random.key.draw_id==y.random.key.draw_id&&
      x.track.particle.energy_MeV==y.track.particle.energy_MeV&&
      x.track.particle.direction.x==y.track.particle.direction.x&&
      x.track.particle.direction.y==y.track.particle.direction.y&&
      x.track.particle.direction.z==y.track.particle.direction.z,"Capacity changed daughter physics/RNG");
  }
  bool rejected=false;try{p.medium_id=99;small.beginShower({p},1,2,2);}catch(std::invalid_argument const&){rejected=true;}
  check(rejected&&small.activeCount()==1,"Bad beginShower changed queue");
  small.beginShower({},1,2,2);check(!small.activeCount()&&!small.advance().created,"Empty beginShower failed");
  AirWavefront<Exec> queue(tables,thresholds(),env,{1.,.5,1.},4,false,{},{},{},reuse);
  p=air_test::input(env,22,.2,10000.,0);queue.beginShower({p},1,2,2);auto b=queue.advance();
  auto o=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},b.outputs);Sink sink;streamAirOutput(o(0),sink);
  check(sink.point_deposits==1&&sink.radio==0&&std::abs(sink.deposit-.25)<1.e-14,"Subcut photon deposition units/weight");
  check(!queue.activeCount(),"Subcut photon not removed");
  p=air_test::input(env,-11,.8,10000.,0);p.history_id=~std::uint64_t{0}-1;
  queue.beginShower({p},1,2,~std::uint64_t{0});rejected=false;
  try{queue.advance();}catch(std::length_error const&){rejected=true;}
  check(rejected&&queue.activeCount()==1&&queue.nextHistoryId()==~std::uint64_t{0},"ID exhaustion not atomic");
  p.history_id=1;rejected=false;
  try{queue.beginShower({p,p},1,2,3);}catch(std::invalid_argument const&){rejected=true;}
  check(rejected&&queue.nextHistoryId()==~std::uint64_t{0},"Duplicate beginShower not atomic");
  // The real C8 stack may inject an older original history after a newer
  // child-ID reservation. New input must not reset any resident continuation.
  p=air_test::input(env,11,100.,10000.,9);queue.beginShower({p},1,2,11);
  queue.advance(100,2);
  check(queue.activeCount()>0,"Missing resident continuation fixture");
  auto prior=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},queue.activeRecords());
  auto injected=air_test::input(env,22,.2,4000.,1);auto count=queue.activeCount();
  queue.append({injected});check(queue.activeCount()==count+1,"Incremental import dropped resident histories");
  auto joined=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},queue.activeRecords());
  for(std::size_t i=0;i<count;++i) {
    check(prior(i).random.key.draw_id==joined(i).random.key.draw_id&&
      prior(i).hadron_random.key.history_id==joined(i).hadron_random.key.history_id&&
      prior(i).hadron_random.key.draw_id==joined(i).hadron_random.key.draw_id&&
      prior(i).metadata.history_id==joined(i).metadata.history_id&&
      prior(i).metadata.step_id==joined(i).metadata.step_id&&
      prior(i).track.particle.energy_MeV==joined(i).track.particle.energy_MeV&&
      prior(i).track.particle.clock.remaining_mfp==joined(i).track.particle.clock.remaining_mfp&&
      prior(i).track.particle.clock.sampled_rate_per_cm==joined(i).track.particle.clock.sampled_rate_per_cm&&
      prior(i).track.particle.clock.initialized==joined(i).track.particle.clock.initialized,
      "Incremental import reset resident history/RNG/transport");
  }
  auto next=queue.nextHistoryId();rejected=false;
  try{queue.append({p});}catch(std::invalid_argument const&){rejected=true;}
  check(rejected&&queue.activeCount()==count+1&&queue.nextHistoryId()==next,"Duplicate append changed native queue");
  rejected=false;try{queue.advance(next-1,2);}catch(std::invalid_argument const&){rejected=true;}
  check(rejected&&queue.nextHistoryId()==next,"Overlapping external reservation accepted");
  // Deterministic stopped-positron branching needs exactly two new histories.
  p=air_test::input(env,-11,.8,10000.,9);queue.beginShower({p},1,2,11);rejected=false;
  try{queue.advance(100,1);}catch(std::length_error const&){rejected=true;}
  check(rejected&&queue.nextHistoryId()==11&&queue.activeCount()==1,"Reservation exhaustion changed queue");
  auto frozen=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},queue.activeRecords());
  check(frozen(0).random.key.draw_id==0,"Reservation exhaustion consumed live randomness");
  auto branch=queue.advance(100,2);check(branch.created==2&&queue.nextHistoryId()==102,"External child IDs not used");
  injected.history_id=100;rejected=false;
  try{queue.append({injected});}catch(std::invalid_argument const&){rejected=true;}
  check(rejected&&queue.activeCount()==2,"Native child ID reimported as CPU history");
  injected.history_id=2;queue.append({injected});check(queue.activeCount()==3,"Older CPU history rejected after child allocation");
}
template<class Exec>void appendContracts(Tables const& tables,bool reuse) {
  auto env=environment();AirWavefront<Exec> queue(tables,thresholds(),env,{1.,.5,1.},32,false,{},{},{},reuse);
  queue.beginShower({},731,129,1);std::vector<C8Particle> input;
  for(int i=0;i<8;++i)input.push_back(air_test::input(env,11,10.,10000.,i));
  queue.append(input);
  auto p=air_test::input(env,22,10.,10000.,8);queue.append({p});
  // Reuse mode grew 8 ->16 and still has tail room; append without moving the
  // resident histories or copying their optical clocks/random states.
  auto const* address=queue.activeRecords().data();auto copied=queue.appendCopiedActiveRecords();
  p.history_id=10;queue.append({p});
  if(reuse)check(queue.activeRecords().data()==address&&queue.appendCopiedActiveRecords()==copied,
    "Resident append failed to use existing tail capacity");
  auto next_id=queue.nextHistoryId();p.history_id=11;
  bool rejected=false;try{queue.append({p,input[0]});}catch(std::invalid_argument const&){rejected=true;}
  check(rejected&&queue.activeCount()==10&&queue.nextHistoryId()==next_id,"Invalid append partially committed");
  queue.append({p}); // ID11 from the rejected batch must not have been reserved.
  auto state=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},queue.activeRecords());
  for(int i=0;i<11;++i)check(state(i).metadata.history_id==std::uint64_t(i+1)&&state(i).random.key.draw_id==0,
    "Incremental append changed order or consumed random numbers");
  check(queue.appendCalls()==4&&queue.appendRecords()==11,"Failed append counted as successful injection");
  // Capacity failure must not alter prior state or consume an incoming ID.
  auto too_many=std::vector<C8Particle>(22,p);rejected=false;
  try{queue.append(too_many);}catch(std::length_error const&){rejected=true;}
  check(rejected&&queue.activeCount()==11&&queue.nextHistoryId()==12,"Append overflow not atomic");
}
void run(char const* path) {
  using Exec=Kokkos::DefaultExecutionSpace;auto tables=readAirTables(path);
  prefixContracts<Exec>();contracts<Exec>(tables,false);contracts<Exec>(tables,true);
  appendContracts<Exec>(tables,false);appendContracts<Exec>(tables,true);
  std::cout<<std::setprecision(17)<<"execution_space="<<Exec::name()<<'\n';
  for(double energy:{10.,100.})for(double stepfc:{1.,.0625}) {
    auto r=cascade<Exec>(tables,stepfc,energy);
    auto reused=cascade<Exec>(tables,stepfc,energy,true);
    check(r.control_hash==reused.control_hash&&r.histories==reused.histories&&r.waves==reused.waves&&
      r.sink.deposit==reused.sink.deposit&&r.sink.steps==reused.sink.steps&&r.sink.radio==reused.sink.radio&&
      r.sink.observed_energy==reused.sink.observed_energy&&r.escaped==reused.escaped,
      "Persistent workspace changed history/RNG/output");
    std::cout<<"energy_MeV="<<energy<<" stepfc="<<stepfc<<" primaries=12 waves="<<r.waves<<" histories="<<r.histories<<" max_queue="<<r.maximum_queue
      <<" layers="<<r.layer_changes<<" tracks="<<r.sink.steps<<" radio_tracks="<<r.sink.radio
      <<" observations="<<r.sink.observations<<" point_deposits="<<r.sink.point_deposits
      <<" discarded="<<r.discarded<<" host_handoffs="<<r.handoffs
      <<" initial_MeV="<<r.initial<<" deposit_MeV="<<r.sink.deposit<<" observed_MeV="<<r.sink.observed_energy<<" discarded_MeV="<<r.escaped
      <<" max_local_closure_MeV="<<r.max_local_closure<<" max_energy_host_device_MeV="<<r.energy_error
      <<" max_position_host_device_m="<<r.position_error<<" control_hash="<<r.control_hash<<'\n';
  }
  std::cout<<"scope=small complete native EM cascades in C8-schema atmosphere with output-record consumers; not production shower, thinning, radio-field calculation or timing benchmark\n";
}
}
int main(int argc,char** argv) {
  try{if(argc!=2)return 2;Kokkos::ScopeGuard guard(argc,argv);run(argv[1]);return 0;}
  catch(std::exception const& e){std::cerr<<e.what()<<'\n';return 1;}
}

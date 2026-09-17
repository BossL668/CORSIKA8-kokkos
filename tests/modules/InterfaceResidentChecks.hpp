// Real physics replay: compare resident FIFO evolution to the old host FIFO,
// using identical per-wavefront history reservations on the same backend.
#pragma once
#include <corsika/modules/transport/InterfaceEmSession.hpp>
#include <cstring>
#include <deque>
#include <iostream>

namespace interface_test {
namespace api= corsika::interfaces;
using Particle=corsika::gpu::em::EmParticleState;
inline void check(bool ok,char const* why) {if(!ok)throw std::runtime_error(why);}
inline bool sameBits(double a,double b) {return std::memcmp(&a,&b,sizeof(double))==0;}
inline bool sameParticle(Particle const& a,Particle const& b) {
  if(a.pid!=b.pid||a.medium_id!=b.medium_id||a.generation!=b.generation||a.reserved!=b.reserved||
     a.history_id!=b.history_id||a.parent_history_id!=b.parent_history_id||a.step_id!=b.step_id||
     !sameBits(a.energy_GeV,b.energy_GeV)||!sameBits(a.time_s,b.time_s)||!sameBits(a.weight,b.weight))return false;
  for(int k=0;k<3;++k)
    if(!sameBits(a.position_m[k],b.position_m[k])||!sameBits(a.direction[k],b.direction[k]))return false;
  return true;
}
inline void sameRecord(api::EmStep const& a,api::EmStep const& b) {
  check(a.outcome==b.outcome&&a.child_count==b.child_count&&a.has_track==b.has_track&&
        a.crossed_material==b.crossed_material&&a.error==b.error&&a.process_id==b.process_id,
        "resident discrete outcome differs from reference");
  check(sameParticle(a.start,b.start)&&sameParticle(a.end,b.end),"resident state/history bits differ");
  for(std::uint32_t i=0;i<a.child_count;++i)
    check(sameParticle(a.children[i],b.children[i]),"resident secondary bits differ");
  check(sameBits(a.distance_m,b.distance_m)&&sameBits(a.grammage_g_cm2,b.grammage_g_cm2)&&
        sameBits(a.deposited_GeV,b.deposited_GeV)&&sameBits(a.binding_energy_GeV,b.binding_energy_GeV)&&
        sameBits(a.unthinned_secondary_total_GeV,b.unthinned_secondary_total_GeV)&&
        sameBits(a.weighted_thinning_delta_GeV,b.weighted_thinning_delta_GeV),"resident energy/path bits differ");
  for(int k=0;k<3;++k)check(sameBits(a.path_midpoint_m[k],b.path_midpoint_m[k]),"resident midpoint bits differ");
  if(a.outcome==api::EmOutcome::Fallback)
    check(sameParticle(a.fallback.particle,b.fallback.particle)&&a.fallback.reason==b.fallback.reason&&
          a.fallback.medium_hash==b.fallback.medium_hash&&a.fallback.interaction_hash==b.fallback.interaction_hash,
          "resident selected-fallback identity differs");
}
template<class F> void reject(F&& action) {
  bool caught=false;try{action();}catch(std::exception const&){caught=true;}
  check(caught,"invalid resident operation accepted");
}
inline void replay(api::InterfaceEmSession& session,std::vector<Particle> primaries) {
  auto const bytes=session.deviceBytes(),peakBytes=session.projectedPeakDeviceBytes();
  auto bad=primaries;bad.back().medium_id=-123;
  reject([&]{session.submit(bad);});
  check(session.pendingParticles()==0&&session.residentStatistics().uploaded_particles==0,
        "rejected CPU admission partially uploaded particles");
  // Exercise a partially filled prefix and dynamic injection into a nonempty queue.
  for(auto& p:primaries)p.energy_GeV=.05;
  session.submit(primaries);
  std::deque<Particle> reference(primaries.begin(),primaries.end());
  reject([&]{session.advanceResident(UINT64_MAX-1);});
  check(session.pendingParticles()==primaries.size(),"history overflow consumed resident particles");
  std::uint64_t nextHistory=1000,steps=0,children=0,crossings=0,fallbacks=0;
  std::size_t injected=primaries.size(),round=0;
  while(!reference.empty()) {
    // The terrain accuracy cap resolves much shorter magnetic steps in the
    // deliberately strong 0.01 T fields. Keep comparing every record to the
    // independent FIFO through natural absorption; allow the extra wavefronts.
    check(++round<1000000,"resident reference replay did not drain");
    if(round==3) {
      auto p=primaries.front();p.history_id=900;p.parent_history_id=899;
      session.submit({p});reference.push_back(p);++injected;
    }
    auto count=session.nextResidentBatchSize();
    check(count>0&&session.pendingParticles()==reference.size(),"resident FIFO size differs");
    std::vector<Particle> input;
    for(std::size_t i=0;i<count;++i){input.push_back(reference.front());reference.pop_front();}
    auto expected=session.advance(input,nextHistory);
    auto actual=session.advanceResident(nextHistory);
    nextHistory+=3*count;
    check(actual.size()==count,"resident prefix length differs");
    for(std::size_t i=0;i<count;++i) {
      sameRecord(actual[i],expected[i]);
      auto const& r=expected[i];++steps;crossings+=r.crossed_material;
      if(r.outcome==api::EmOutcome::Continuation&&r.end.weight>0.)reference.push_back(r.end);
      if(r.outcome==api::EmOutcome::Children)
        for(std::uint32_t c=0;c<r.child_count;++c)
          if(r.children[c].weight>0.){reference.push_back(r.children[c]);++children;}
      if(r.outcome==api::EmOutcome::Fallback)++fallbacks;
    }
    check(session.pendingParticles()==reference.size(),"resident continuation/drop differs");
    check(session.deviceBytes()==bytes&&session.projectedPeakDeviceBytes()==peakBytes,
          "resident workspace grew during replay");
  }
  check(session.advanceResident(1).empty(),"empty resident queue launched work");
  auto const& stats=session.residentStatistics();
  check(stats.uploaded_particles==injected&&stats.advanced_particles==steps&&steps>10*injected,
        "surviving particles were uploaded again or not advanced");
  check(children>0&&crossings>=primaries.size(),"live replay did not exercise children and interfaces");
  check(stats.wavefronts==round,"resident wavefront accounting differs");
  std::cout<<"resident replay PASS steps="<<steps<<" wavefronts="<<round<<" children="<<children
           <<" crossings="<<crossings<<" selected_fallbacks="<<fallbacks<<" CPU_uploads="<<injected<<'\n';
}
} // namespace interface_test

namespace interface_test {
inline void cascadeReplay(api::InterfaceEmSession& session,std::vector<Particle> primaries,
                          std::size_t recordCapacity,std::size_t frontCapacity=64) {
  for(auto& p:primaries)p.energy_GeV=.05;
  std::deque<Particle> fifo(primaries.begin(),primaries.end());
  std::vector<api::EmStep> expected;
  std::uint64_t firstHistory=1000;
  for(auto const& p:primaries)firstHistory=std::max(firstHistory,p.history_id+1);
  std::uint64_t next=firstHistory,referenceWaves=0;
  // Independent host FIFO oracle uses only the stateless reference API.
  while(!fifo.empty()) {
    check(++referenceWaves<1000000,"cascade oracle did not drain");
    auto count=std::min(frontCapacity,fifo.size());
    std::vector<Particle> front;
    for(std::size_t i=0;i<count;++i){front.push_back(fifo.front());fifo.pop_front();}
    auto records=session.advance(front,next);next+=3*count;
    for(auto const& r:records) {
      expected.push_back(r);
      if(r.outcome==api::EmOutcome::Continuation&&r.end.weight>0.)fifo.push_back(r.end);
      if(r.outcome==api::EmOutcome::Children)
        for(std::uint32_t c=0;c<r.child_count;++c)
          if(r.children[c].weight>0.)fifo.push_back(r.children[c]);
    }
  }
  auto before=session.residentStatistics();
  auto bytes=session.deviceBytes(),peak=session.projectedPeakDeviceBytes();
  session.submit(primaries);
  reject([&]{session.runResidentCascade({});});
  reject([&]{session.runResidentCascade([](std::size_t){return std::uint64_t{0};});});
  reject([&]{session.runResidentCascade([&](std::size_t){session.submit({});return std::uint64_t{1000};});});
  check(session.pendingParticles()==primaries.size(),"invalid cascade admission consumed input");
  next=firstHistory;
  auto reserve=[&](std::size_t n){auto first=next;next+=n;return first;};
  std::size_t offset=0,waves=0,calls=0,capacityStops=0;
  while(session.pendingParticles()) {
    // First call exercises an explicit wavefront checkpoint. Later calls must
    // span multiple fronts and stop safely at the bounded output ledger.
    auto result=session.runResidentCascade(reserve,1,calls?1024:1);
    check(result.wavefronts>0&&result.records.size()<=recordCapacity,"unbounded/empty cascade result");
    if(!calls)check(result.wavefronts==1,"wavefront limit was ignored");
    if(result.checkpoint==api::ResidentCheckpoint::RecordCapacity)++capacityStops;
    for(auto const& record:result.records) {
      check(offset<expected.size(),"cascade produced extra records");
      sameRecord(record,expected[offset++]);
    }
    waves+=result.wavefronts;++calls;
    check(session.deviceBytes()==bytes&&session.projectedPeakDeviceBytes()==peak,"cascade allocation grew");
  }
  auto after=session.residentStatistics();
  check(offset==expected.size()&&waves==referenceWaves,"cascade dropped records/fronts");
  check(after.cascade_calls-before.cascade_calls==calls&&
        after.control_downloads-before.control_downloads==waves&&
        after.record_downloads-before.record_downloads==calls&&
        after.downloaded_records-before.downloaded_records==offset,"cascade transfer accounting differs");
  check(after.uploaded_particles-before.uploaded_particles==primaries.size(),"cascade reuploaded successors");
  check(capacityStops>0&&calls<waves&&after.maximum_call_wavefronts>1,"multi-wavefront/ledger checkpoint not exercised");
  check(session.runResidentCascade(reserve).records.empty(),"empty cascade returned records");
  std::cout<<"multi-wavefront cascade PASS steps="<<offset<<" waves="<<waves<<" calls="<<calls
           <<" record_capacity_checkpoints="<<capacityStops<<" maximum_call_wavefronts="<<after.maximum_call_wavefronts<<'\n';
}
} // namespace interface_test

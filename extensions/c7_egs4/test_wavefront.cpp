#include "Egs4Wavefront.hpp"
#include <algorithm>
#include <iomanip>
#include <iostream>

namespace {
using namespace c7_egs4;
using namespace c7_egs4::c8_adapter;
void check(bool b,char const* text){if(!b)throw std::runtime_error(text);}
struct FreeAir {
  KOKKOS_INLINE_FUNCTION BoundaryLimit operator()(TrackState const&,double distance)const{return {distance,BoundaryKind::none};}
};
struct BoundaryCap {
  KOKKOS_INLINE_FUNCTION BoundaryLimit operator()(TrackState const&,double distance)const{return {.25*distance,BoundaryKind::observation};}
};
double available(TrackState p,double mass){return p.pdg==11?p.energy_MeV-mass:(p.pdg==-11?p.energy_MeV+mass:p.energy_MeV);}
Frame frame(){return {{0.,0.,0.},{1.,0.,0.},{0.,-1.,0.},{0.,0.,-1.}};}
AirRegion region(){return {.001225,0.,0.};} // homogeneous AIR, not a shower atmosphere
TransportSettings settings(double stepfc){return {1.,.5,stepfc};}
C8Particle particle(int pdg,double energy,std::uint64_t id) {
  C8Particle p;p.pid=pdg;p.energy_GeV=energy*.001;p.direction[2]=-1.;p.history_id=id;p.medium_id=7;return p;
}
struct Totals {
  int waves{},maximum_queue{},histories{},tracks{},vertices{},boundaries{},handoffs{};
  double deposit{},initial{},max_closure{};
  std::uint64_t control_hash{1469598103934665603ULL};
};
void hash(std::uint64_t& x,std::uint64_t y){x=(x^y)*1099511628211ULL;}

template<class Exec>Totals cascade(Tables const& tables,double stepfc,int per_species) {
  ChannelThresholds thresholds{.51099895,140.,140.,422.};auto host_model=modelView(tables,thresholds);
  LocalWavefront<Exec> queue(tables,thresholds,frame(),region(),{},settings(stepfc),7,2048);
  std::vector<C8Particle> primaries;
  for(int pdg:{11,-11,22})for(int i=0;i<per_species;++i)primaries.push_back(particle(pdg,10.,primaries.size()+1));
  queue.load(primaries,731,129,primaries.size()+1);
  std::vector<NativeRecord> reference;
  Totals total;
  for(auto const& p:primaries) {
    NativeRecord r;check(importRecord(p,frame(),731,129,r),"Import primary");reference.push_back(r);
    total.initial+=available(r.state,thresholds.mass_MeV)*r.metadata.weight;
  }
  auto first_id=std::uint64_t(primaries.size()+1);
  while(queue.activeCount()) {
    check(++total.waves<10000,"Native cascade failed to terminate");
    total.maximum_queue=std::max(total.maximum_queue,int(queue.activeCount()));
    check(queue.activeCount()==reference.size(),"Host/device active count mismatch");
    auto batch=queue.advance(FreeAir{});
    auto outcomes=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},batch.outcomes);
    auto records=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},batch.records);
    std::vector<NativeRecord> next;int created=0;
    for(std::size_t i=0;i<reference.size();++i) {
      auto before=reference[i];auto r=advanceRecord(host_model,reference[i],frame(),region(),{},settings(stepfc),FreeAir{});
      auto d=outcomes(i);auto record=records(i);
      check(r.action==d.action&&r.child_count==d.child_count&&r.channel==d.channel&&
        reference[i].random.key.draw_id==record.random.key.draw_id,"Cascade control/RNG differs");
      check(d.action==HistoryAction::active||d.action==HistoryAction::replaced,"Unexpected native cascade handoff/error");
      check(std::abs(r.particle.energy_MeV-d.particle.energy_MeV)<1.e-9,"Cascade energy differs");
      check(reference[i].metadata.history_id==record.metadata.history_id,"Queue history order differs");
      double outgoing=d.continuous_deposit_MeV+d.vertex_deposit_MeV;
      if(d.action==HistoryAction::replaced)for(int j=0;j<d.child_count;++j)outgoing+=available(d.children[j],thresholds.mass_MeV);
      else outgoing+=available(d.particle,thresholds.mass_MeV);
      total.max_closure=std::max(total.max_closure,std::abs(outgoing-available(before.state,thresholds.mass_MeV)));
      total.deposit+=(d.continuous_deposit_MeV+d.vertex_deposit_MeV)*record.metadata.weight;
      total.tracks+=d.has_track;total.vertices+=d.action==HistoryAction::replaced;
      hash(total.control_hash,std::uint64_t(d.action));hash(total.control_hash,std::uint64_t(d.channel));
      hash(total.control_hash,record.random.key.draw_id);hash(total.control_hash,record.metadata.history_id);
      if(r.action==HistoryAction::active)next.push_back(reference[i]);
      else for(int j=0;j<r.child_count;++j) {
        NativeRecord child;check(childRecord(reference[i],r.children[j],frame(),first_id+created++,child),"Create reference child");
        next.push_back(child);
      }
    }
    first_id+=created;check(first_id==queue.nextHistoryId(),"Queue ID prefix allocation differs");
    reference=std::move(next);
  }
  total.histories=int(first_id-1);
  check(total.tracks>0&&total.vertices>int(primaries.size())&&total.histories>int(primaries.size()),"No branching cascade");
  check(total.max_closure<1.e-8&&std::abs(total.deposit-total.initial)<1.e-7,"Cascade energy balance");
  check(queue.advance(FreeAir{}).next_count==0,"Empty queue should remain empty");
  return total;
}

template<class Exec>void contracts(Tables const& tables) {
  ChannelThresholds thresholds{.51099895,140.,140.,422.};
  // A cut positron must produce two photons; overflow must not lose either,
  // nor commit consumed RNG or history IDs. Retrying yields the same failure.
  LocalWavefront<Exec> small(tables,thresholds,frame(),region(),{},settings(1.),7,1);
  small.load({particle(-11,.8,1)},1,2,2);
  for(int retry=0;retry<2;++retry) {
    bool failed=false;try{small.advance(FreeAir{});}catch(std::runtime_error const& e){failed=std::string(e.what()).find("overflow")!=std::string::npos;}
    check(failed&&small.activeCount()==1&&small.nextHistoryId()==2,"Queue overflow was not atomic");
    auto state=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},small.activeRecords());
    check(state(0).random.key.draw_id==0&&state(0).state.energy_MeV==.8,"Failed wave changed active history");
  }
  LocalWavefront<Exec> queue(tables,thresholds,frame(),region(),{},settings(1.),7,8);
  queue.load({particle(22,10.,1)},1,2,2);auto batch=queue.advance(BoundaryCap{});
  auto outcomes=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},batch.outcomes);
  check(queue.activeCount()==0&&outcomes(0).action==HistoryAction::boundary&&
    outcomes(0).particle.photon_clock.initialized,"Observation boundary not handed to caller with optical clock");
  auto records=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},batch.records);
  auto saved=records(0);queue.resume({saved});
  auto resumed=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},queue.activeRecords());
  check(queue.activeCount()==1&&resumed(0).random.key.draw_id==saved.random.key.draw_id&&
    resumed(0).state.photon_clock.remaining_mfp==saved.state.photon_clock.remaining_mfp,"Resume reset clock/RNG");
  bool duplicate=false;try{queue.resume({saved});}catch(std::runtime_error const&){duplicate=true;}
  check(duplicate&&queue.activeCount()==1,"Duplicate resumed history accepted");
  bool invalid=false;auto p=particle(11,10.,1);p.medium_id=99;
  try{queue.load({p},1,2,2);}catch(std::runtime_error const&){invalid=true;}
  check(invalid,"Unsupported material silently treated as AIR");
  invalid=false;try{queue.load({particle(11,10.,1),particle(22,10.,1)},1,2,2);}catch(std::runtime_error const&){invalid=true;}
  check(invalid,"Duplicate primary history ID accepted");
}
void run(std::string const& path) {
  using Exec=Kokkos::DefaultExecutionSpace;auto tables=readAirTables(path);contracts<Exec>(tables);
  std::cout<<std::setprecision(17)<<"execution_space="<<Exec::name()<<"\n";
  for(double stepfc:{1.,.0625}) {
    auto r=cascade<Exec>(tables,stepfc,4);
    std::cout<<"stepfc="<<stepfc<<" primaries=12 waves="<<r.waves<<" histories="<<r.histories
      <<" tracks="<<r.tracks<<" max_queue="<<r.maximum_queue<<" terminal_vertices="<<r.vertices
      <<" initial_MeV="<<r.initial<<" deposit_MeV="<<r.deposit<<" max_local_energy_closure_MeV="<<r.max_closure
      <<" control_hash="<<r.control_hash<<'\n';
  }
  std::cout<<"scope=complete small branching histories in homogeneous air, no thinning, no nuclear channels at these energies; not full C8 shower integration or a timing benchmark\n";
}
}
int main(int argc,char** argv) {
  try {
    if(argc!=2){std::cerr<<"Usage: test_egs4_wavefront EGSDAT\n";return 2;}
    Kokkos::ScopeGuard guard(argc,argv);run(argv[1]);return 0;
  }catch(std::exception const& e){std::cerr<<e.what()<<'\n';return 1;}
}

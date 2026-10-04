#include "Egs4C8Resonance.hpp"
#include "Egs4C8AirWavefront.hpp"
#include "AirTestFixture.hpp"
#include <iostream>
#include <iomanip>
#include <algorithm>

namespace {
using namespace c7_egs4;using namespace c7_egs4::c8_adapter;
void check(bool ok,char const* message){if(!ok)throw std::runtime_error(message);}
ResonanceMasses masses(){return {782.65,1019.461,139.57039,134.9768,105.6583755,493.677,497.611,497.611,547.862};}
struct Input {HostVertex request;Frame frame;std::uint64_t first;};
struct Batch {
  Kokkos::View<Input*> input;Kokkos::View<ResonanceTransfer*> output;ResonanceMasses mass;
  KOKKOS_INLINE_FUNCTION void operator()(int i)const {
    auto q=input(i);output(i)=resolveResonanceRequest(q.request,q.frame,mass,q.first,3);
  }
};
struct DeviceContracts {
  Kokkos::View<Input*> input;ResonanceMasses mass;
  KOKKOS_INLINE_FUNCTION void operator()(int i,int& errors)const {
    auto q=input(i);auto bad=q.request;
    errors+=resolveResonanceRequest(q.request,q.frame,mass,q.first,1).valid;
    errors+=resolveResonanceRequest(q.request,q.frame,mass,q.request.particle.history_id,3).valid;
    errors+=resolveResonanceRequest(q.request,q.frame,mass,~std::uint64_t{0}-1,3).valid;
    bad.hadron_random.key.draw_id=~std::uint64_t{0};
    auto exhausted=resolveResonanceRequest(bad,q.frame,mass,q.first,3);
    errors+=exhausted.valid||exhausted.status!=InteractionStatus::random_failure;
    errors+=bad.hadron_random.key.draw_id!=~std::uint64_t{0};
    bad=q.request;bad.particle.generation=~std::uint32_t{0};
    errors+=resolveResonanceRequest(bad,q.frame,mass,q.first,3).valid;
    bad=q.request;bad.particle.pid=113;errors+=resolveResonanceRequest(bad,q.frame,mass,q.first,3).valid;
    auto f=q.frame;f.z={-f.z.x,-f.z.y,-f.z.z};
    errors+=resolveResonanceRequest(q.request,f,mass,q.first,3).valid;
  }
};
std::vector<Input> inputs(AirEnvironment const& env) {
  std::vector<Input> out;
  // Find Philox counters selecting each channel; conditional coverage, not a
  // statement about physical frequencies. Keep the actual native RNG provider.
  double omega[]{0.,.8996252,.9843332,.9997739,.9999090,1.};
  double phi[]{0.,.4901808,.8330066,.9865765,.9996981,.9999857,1.};
  for(int id:{223,333})for(int branch=0;branch<(id==223?5:6);++branch) {
    auto thresholds=id==223?omega:phi;CounterStream r;
    r.key={819,13,101,0,NativeEgs4HadronRandomDomain,0};
    std::uint64_t offset=0;bool found=false;
    for(;offset<2000000;++offset){double u;check(r.next(u),"Philox failed");
      if(u>thresholds[branch]&&u<=thresholds[branch+1]){found=true;break;}}
    check(found,"Did not cover resonance channel");
    for(int geometry=0;geometry<6;++geometry) {
      auto p=air_test::input(env,22,2000.,4000.+1000.*geometry,100);
      p.position_m[0]+=10000.*geometry;p.position_m[1]-=2000.*geometry;
      AirRecord record;check(importAirRecord(p,env,819,13,record),"Cannot construct C7 frame");
      if(geometry%2){record.track.detector_local=true;record.track.particle.direction={.3,.4,std::sqrt(.75)};}
      HostVertex request;request.kind=HostVertexKind::c7_vector_meson_decay;
      check(exportHostSecondary(record,{2000.,record.track.particle.direction,id},env,request.particle),"Cannot export meson");
      request.random=record.random;request.hadron_random=r;request.hadron_random.key.draw_id=offset;
      request.particle.generation=3;request.particle.step_id=0;
      auto frame=decayFrame(record,env);
      auto back=toLocal(frame,{request.particle.direction[0],request.particle.direction[1],request.particle.direction[2]});
      auto original=record.track.particle.direction;
      check(std::max({std::abs(back.x-original.x),std::abs(back.y-original.y),std::abs(back.z-original.z)})<1.e-12,
        "Recorded frame does not recover the original C7 meson direction");
      out.push_back({request,frame,10000+out.size()*3});
    }
  }
  return out;
}
void run(std::string const& path) {
  using Exec=Kokkos::DefaultExecutionSpace;
  auto env=makeAirEnvironment(air_test::environment(),6.371315e6,10.,AirConvention::c7_egs4_four_exponentials);
  auto all=inputs(env);auto m=masses();
  Kokkos::View<Input*> in("meson_request",all.size());Kokkos::View<ResonanceTransfer*> out("meson_reply",all.size());
  auto h=Kokkos::create_mirror_view(in);for(std::size_t i=0;i<all.size();++i)h(i)=all[i];Kokkos::deep_copy(in,h);
  int errors=0;Kokkos::parallel_reduce("native_c7_resonance_failure_contracts",all.size(),DeviceContracts{in,m},errors);
  check(errors==0,"Device resonance failure contracts");
  Kokkos::parallel_for("native_c7_resonance_carrier_bridge",all.size(),Batch{in,out,m});Exec().fence();
  auto got=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},out);
  double ediff=0.,ddiff=0.,closure=0.;int channels[12]{},destinations[3]{};std::vector<C8Particle> photons;
  for(std::size_t i=0;i<all.size();++i) {
    auto q=all[i];auto a=resolveResonanceRequest(q.request,q.frame,m,q.first,3),b=got(i);auto p=q.request.particle;
    check(a.valid&&b.valid&&a.count==b.count&&a.branch==b.branch,"Native decay bridge failed");
    check(a.parent_hadron_continuation.key.draw_id==b.parent_hadron_continuation.key.draw_id,"Decay RNG continuation differs");
    ++channels[int(b.branch)];double total=0.;
    for(int j=0;j<b.count;++j) {
      auto x=a.products[j],y=b.products[j];auto c=y.particle;total+=c.energy_GeV;++destinations[int(y.destination)];
      check(c.history_id==q.first+j&&c.parent_history_id==p.history_id&&c.generation==4&&c.step_id==0,
        "Decay ancestry lost");
      check(c.pid==x.particle.pid&&y.destination==x.destination&&c.medium_id==p.medium_id&&
        c.weight==p.weight&&c.time_s==p.time_s,"Decay metadata lost");
      check(y.random.key.history_id==c.history_id&&y.hadron_random.key.history_id==c.history_id&&
        y.random.key.draw_id==0&&y.hadron_random.key.draw_id==0,"New child RNG shared with parent");
      ediff=std::max(ediff,std::abs(x.particle.energy_GeV-c.energy_GeV));
      for(int k=0;k<3;++k){check(c.position_m[k]==p.position_m[k],"Decay moved position");
        ddiff=std::max(ddiff,std::abs(x.particle.direction[k]-c.direction[k]));}
      check(y.polarization_cosine==x.polarization_cosine&&y.polarization_azimuth==x.polarization_azimuth,"Polarization changed");
      if(y.destination==DecayDestination::native_em){check(c.pid==22,"Non-photon returned to EM");photons.push_back(c);}
    }
    closure=std::max(closure,std::abs(total-p.energy_GeV));
    for(auto count:{0ULL,1ULL})check(!resolveResonanceRequest(q.request,q.frame,m,q.first,count).valid,"Insufficient IDs accepted");
    check(!resolveResonanceRequest(q.request,q.frame,m,p.history_id,3).valid,"Parent ID reused");
    auto bad=q.request;bad.hadron_random.key.draw_id=~std::uint64_t{0};
    auto failed=resolveResonanceRequest(bad,q.frame,m,q.first,3);
    check(!failed.valid&&failed.status==InteractionStatus::random_failure&&bad.hadron_random.key.draw_id==~std::uint64_t{0},"RNG failure changed request");
    bad=q.request;bad.particle.generation=~std::uint32_t{0};check(!resolveResonanceRequest(bad,q.frame,m,q.first,3).valid,"Generation overflow accepted");
    auto f=q.frame;f.y=f.x;check(!resolveResonanceRequest(q.request,f,m,q.first,3).valid,"Wrong handed frame accepted");
  }
  for(int j=1;j<=11;++j)check(channels[j]==6,"Channel/frame coverage differs");
  check(ediff<1.e-11&&ddiff<1.e-10&&closure<1.e-12&&!photons.empty(),"Carrier bridge numerical check");
  // Actual resident queue ingestion, not an exported photon counted as a sink.
  // Only a first transport wave here; hadrons and muons are not fully tracked.
  auto tables=readAirTables(path);AirWavefront<Exec> queue(tables,{.51099895,152.,152.,422.},env,{1.,.5,.0625},256);
  queue.beginShower({},819,13,1);queue.append(photons);
  check(queue.activeCount()==photons.size(),"Decay photon injection lost children");
  auto before=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},queue.activeRecords());
  for(std::size_t i=0;i<photons.size();++i)check(before(i).metadata.history_id==photons[i].history_id&&
    !before(i).track.particle.photon_clock.initialized,"Injected photon reused nuclear free path");
  auto batch=queue.advance();check(batch.before.extent(0)==photons.size(),"Photons did not advance");
  std::cout<<std::setprecision(17)<<"execution_space="<<Exec::name()<<" decay_requests="<<all.size()<<" channels=11"
    <<" native_photons="<<photons.size()<<" muons="<<destinations[1]<<" hadrons="<<destinations[2]
    <<" max_energy_difference_GeV="<<ediff<<" max_direction_difference="<<ddiff<<" energy_closure_GeV="<<closure
    <<" photon_first_wave=passed\nscope=explicit-frame C8 carrier bridge and photon reinjection; not automatic Session dispatch, cuts/thinning or full hadronic shower\n";
}
}
int main(int argc,char** argv){try{if(argc!=2)return 2;Kokkos::ScopeGuard guard(argc,argv);run(argv[1]);return 0;}
  catch(std::exception const& e){std::cerr<<e.what()<<'\n';return 1;}}

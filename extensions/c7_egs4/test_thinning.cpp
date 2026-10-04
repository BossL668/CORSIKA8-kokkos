#include "Egs4C8AirWavefront.hpp"
#include "AirTestFixture.hpp"
#include <iostream>
#include <iomanip>
namespace {
using namespace c7_egs4;using namespace c7_egs4::c8_adapter;
void check(bool value,char const* message){if(!value)throw std::runtime_error(message);}
template<class View>struct SampleVertices {
  View sums;
  KOKKOS_INLINE_FUNCTION void operator()(int i)const {
    AirRecord p;p.metadata.weight=1.;p.track.particle.energy_MeV=10.;p.random.key={731,17,std::uint64_t(i+1),0,NativeEgs4RandomDomain,29};
    AirOutcome r;r.action=AirAction::replaced;r.history.child_count=2;r.history.channel=Channel::compton;
    r.children[0].particle.energy_MeV=3.;r.children[1].particle.energy_MeV=7.;
    auto h=thinAirVertex({1.,10.,1,1},p,r),s=thinAirVertex({1.,1.5,1,1},p,r);
    sums(i,0)=h.first_weight;sums(i,1)=h.second_weight;sums(i,2)=s.first_weight;sums(i,3)=s.second_weight;
  }
};
template<class Exec>void vertices() {
  constexpr int n=262144;
  Kokkos::View<double**,typename Exec::memory_space> sums("thinning_samples",n,4);
  Kokkos::parallel_for("thinning_vertices",Kokkos::RangePolicy<Exec>(0,n),SampleVertices<decltype(sums)>{sums});
  auto h=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},sums);
  double average[4]{};for(int i=0;i<n;++i)for(int j=0;j<4;++j)average[j]+=h(i,j)/n;
  for(double a:average)check(std::abs(a-1.)<.015,"Thinning biased daughter population");
  std::cout<<Exec::name()<<" samples="<<n<<" mean_weights=";
  for(double a:average)std::cout<<a<<' ';std::cout<<'\n';
  AirRecord p;p.metadata.weight=2.;p.track.particle.energy_MeV=10.;p.random.key={731,17,2,0,NativeEgs4RandomDomain,29};
  AirOutcome r;r.action=AirAction::replaced;r.history.child_count=2;r.history.channel=Channel::invalid;
  check(thinAirVertex({1.,10.,1,1},p,r).status==ThinningStatus::NotApplied,"Thinned cut-positron photons");
  r.history.channel=Channel::compton;r.children[0].particle.energy_MeV=3.;r.children[1].particle.energy_MeV=7.;
  auto original=p.random.key.draw_id;
  check(thinAirVertex({},p,r).keep_mask==3&&p.random.key.draw_id==original,"Disabled thinning changed RNG/children");
  check(thinAirVertex({.001,10.,1,1},p,r).status==ThinningStatus::NotApplied,"Threshold not respected");
  check(thinAirVertex({1.,2.,1,1},p,r).status==ThinningStatus::NotApplied,"Weight cap not respected");
  bool bad=false;try{validateThinning({1.,2.,1,0});}catch(std::invalid_argument const&){bad=true;}
  check(bad,"Unsupported zero-weight multithinning was silently accepted");
}
template<class Exec>std::uint64_t cascade(Tables const& tables) {
  auto env=makeAirEnvironment(air_test::environment(),6.371315e6,10.,AirConvention::c7_egs4_four_exponentials);
  AirWavefront<Exec> queue(tables,{.51099895,152.,152.,422.},env,{1.,.5,.0625},4096,false,{}, {},{.1,8.,1,1});
  std::vector<C8Particle> roots;
  for(int i=0;i<48;++i){auto p=air_test::input(env,i%3==0?22:i%3==1?11:-11,30.,5000.,i);p.weight=1.;roots.push_back(p);}
  queue.beginShower(roots,731,129,100);int waves=0,thinned=0,removed=0;std::uint64_t hash=1469598103934665603ULL;
  while(queue.activeCount()) {
    check(++waves<20000,"Thinned cascade did not terminate");
    auto b=queue.advance();
    auto t=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},b.thinning);
    auto outcomes=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},b.outcomes);
    auto records=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},b.after);
    auto output=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},b.outputs);
    auto next=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},queue.activeRecords());
    int children=0,retained=0;
    for(std::size_t i=0;i<t.extent(0);++i) {
      auto d=t(i);hash=(hash^d.keep_mask)*1099511628211ULL;
      if(outcomes(i).action==AirAction::replaced){children+=outcomes(i).history.child_count;
        for(int j=0;j<outcomes(i).history.child_count;++j)retained+=(d.keep_mask>>j)&1U;}
      else retained+=outcomes(i).action==AirAction::active;
      if(d.status!=ThinningStatus::NotApplied){++thinned;removed+=2-((d.keep_mask&1U)!=0)-((d.keep_mask&2U)!=0);}
      if(output(i).has_step)check(output(i).step.weight==records(i).metadata.weight,"Parent deposition weight changed");
      if(output(i).has_radio)check(output(i).radio.step.weight==records(i).metadata.weight,"Parent radio weight changed");
    }
    check(children==b.created&&retained==int(next.extent(0)),"Thinned compaction/physical history count mismatch");
    for(std::size_t i=0;i<next.extent(0);++i)check(next(i).metadata.weight>0.&&next(i).metadata.weight<=8.*(1.+1.e-14),"Invalid retained weight");
  }
  check(thinned>100&&removed>100,"Missing thinning integration coverage");
  std::cout<<Exec::name()<<" waves="<<waves<<" vertices="<<thinned<<" removed="<<removed<<" control_hash="<<hash<<'\n';return hash;
}
}
int main(int argc,char** argv){Kokkos::ScopeGuard scope(argc,argv);try {
  auto tables=readAirTables(argv[1]);vertices<Kokkos::DefaultHostExecutionSpace>();vertices<Kokkos::DefaultExecutionSpace>();
  auto a=cascade<Kokkos::DefaultHostExecutionSpace>(tables),b=cascade<Kokkos::DefaultExecutionSpace>(tables);
  check(a==b,"Host/device thinning control changed");return 0;
}catch(std::exception const& e){std::cerr<<e.what()<<'\n';return 1;}}

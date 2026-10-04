#include "Egs4Transport.hpp"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>

namespace {
using namespace c7_egs4;
constexpr double mass=.51099895;
struct Input {double energy;std::int64_t seed;};
struct Answer {double v[22]{};};
struct Stream {
  std::int64_t seed;int used{};
  KOKKOS_INLINE_FUNCTION bool next(double& x) {
    if(used>=100000)return false;
    seed=(seed*48271)%2147483647;x=double(seed)/2147483647.;++used;return true;
  }
};
void check(bool b,char const* m){if(!b)throw std::runtime_error(m);}
KOKKOS_INLINE_FUNCTION int iq(int pdg){return pdg==22?1:pdg==-11?2:3;}
template<int Mode>KOKKOS_INLINE_FUNCTION Answer evaluate(ModelView model,Input q,int index) {
  Answer a;Stream rng{q.seed};int pdg=index%2?11:-11;
  if constexpr(Mode<2) {
    auto c=electronQuery(model.tables,q.energy,mass,pdg==11?-1:1);
    TrackState s{pdg,q.energy,0.,{10.,-20.,-1.e6},{.3,.4,::sqrt(.75)},
      {.1,c.rate_per_cm,true}};
    AirRegion region{.001225,1.25e-6,10.};MagneticField field{5.e-5*2.99792458,.6,.8,.2};
    TransportSettings settings{mass+.5,.5,Mode==0?1.:.0625};
    auto p=prepareElectronSegment(model,s,region,field,settings);
    auto r=finishElectronSegment(model,s,region,field,settings,p,{p.proposed_cm*.8,BoundaryKind::none},rng);
    if(p.status!=Status::success||(r.status!=TransportStatus::advanced&&r.status!=TransportStatus::collision_due&&r.status!=TransportStatus::energy_cut))a.v[1]=-1.;
    double vals[]{p.unclipped_cm,p.proposed_cm,r.geometric_step_cm,r.loss_path_cm,r.time_path_cm,
      r.deposit_MeV,r.particle.energy_MeV,r.particle.position_cm.x,r.particle.position_cm.y,r.particle.position_cm.z,
      r.particle.direction.x,r.particle.direction.y,r.particle.direction.z,r.particle.time_s,
      r.scattering_angle,r.bend_angle,r.particle.clock.remaining_mfp,p.local.loss_MeV_per_cm};
    for(int j=0;j<18;++j)a.v[4+j]=vals[j];
  }else if constexpr(Mode==2) {
    auto r=sampleScattering(model.tables.medium,q.energy,mass,.01/model.tables.medium.rho,rng);
    a.v[1]=r.status==InteractionStatus::success?double(r.below_scattering_threshold):-1.;a.v[4]=r.angle;
  }else {
    auto r=sampleRadiative(model.radiative,{q.energy,mass,.001225,{.3,.4,::sqrt(.75)},Mode==4?22:pdg},rng);
    a.v[1]=r.status==InteractionStatus::success?r.count:-1.;
    for(int j=0;j<r.count;++j) {
      auto p=r.particle[j];a.v[2+j]=iq(p.pdg);a.v[4+4*j]=p.energy_MeV;
      a.v[5+4*j]=p.direction.x;a.v[6+4*j]=p.direction.y;a.v[7+4*j]=p.direction.z;
    }
  }
  a.v[0]=rng.used;return a;
}
template<int Mode,class Exec>struct Batch {
  ModelView model;Kokkos::View<Input*,typename Exec::memory_space> in;
  Kokkos::View<Answer*,typename Exec::memory_space> out;
  KOKKOS_INLINE_FUNCTION void operator()(int i)const{out(i)=evaluate<Mode>(model,in(i),i);}
};
template<int Mode>void run(std::string table,std::string inputs,std::string reference,std::string csv,bool scalar) {
  using Exec=Kokkos::DefaultExecutionSpace;
  check(!std::filesystem::exists(csv),"Refusing to overwrite timing");
  auto tables=readAirTables(table);ChannelThresholds thresholds{mass,152.,152.,422.};
  auto host=modelView(tables,thresholds);DeviceModel<Exec> device(tables,thresholds);
  std::ifstream f(inputs);std::size_t n=0;f>>n;check(n>0&&n<=1000000,"Invalid input size");
  std::vector<Input> input(n);for(auto& q:input)f>>q.energy>>q.seed;
  check(bool(f),"Truncated inputs");
  std::vector<Answer> output(n),expected(n);
  std::ifstream rf(reference,std::ios::binary);rf.read(reinterpret_cast<char*>(expected.data()),n*sizeof(Answer));
  check(bool(rf)&&rf.peek()==std::char_traits<char>::eof(),"Wrong reference size");
  Kokkos::View<Input*,typename Exec::memory_space> in("inputs",n);
  Kokkos::View<Answer*,typename Exec::memory_space> out("outputs",n);auto h=Kokkos::create_mirror_view(in);
  for(std::size_t i=0;i<n;++i)h(i)=input[i];Kokkos::deep_copy(in,h);Exec().fence();
  auto launch=[&]{
    if(scalar){for(std::size_t i=0;i<n;++i)output[i]=evaluate<Mode>(host,input[i],i);std::atomic_signal_fence(std::memory_order_seq_cst);}
    else Kokkos::parallel_for("native_C7_comparison",Kokkos::RangePolicy<Exec>(0,n),Batch<Mode,Exec>{device.view(),in,out});
  };
  std::ofstream file(csv);file<<std::setprecision(17)<<"round,seconds_per_batch,checksum\n";
  for(int round=-2;round<7;++round){
    auto t0=std::chrono::steady_clock::now();for(int rep=0;rep<5;++rep)launch();Exec().fence();
    double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count()/5.;
    if(!scalar){auto m=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},out);for(std::size_t i=0;i<n;++i)output[i]=m(i);}
    double checksum=0.;for(auto a:output)for(auto x:a.v)checksum+=x;
    if(round>=0)file<<round<<','<<seconds<<','<<checksum/n<<'\n';
  }
  double maxnorm=0.,maxdir=0.,maxenergy=0.;std::size_t controls=0,errors=0;
  for(std::size_t i=0;i<n;++i)for(int j=0;j<22;++j){
    double x=output[i].v[j],y=expected[i].v[j],delta=std::abs(x-y);
    if(j<4){controls+=x!=y;continue;}
    maxnorm=std::max(maxnorm,delta/(1.+std::abs(y)));
    bool angle=Mode<2?(j>=14&&j<=16)||j==18||j==19:Mode==2?j==4:(j>=4&&j<12&&(j-4)%4!=0);
    bool energy=Mode<2?j==10:Mode>2&&(j==4||j==8);
    if(angle)maxdir=std::max(maxdir,delta);
    if(energy)maxenergy=std::max(maxenergy,delta/std::max(1.e-20,std::abs(y)));
    errors+=!std::isfinite(x)||delta>1.e-8*(1.+std::abs(y));
  }
  std::cout<<std::setprecision(17)<<"backend="<<(scalar?"scalar":Exec::name())<<" N="<<n<<" mode="<<Mode
    <<" control_mismatches="<<controls<<" numeric_mismatches="<<errors<<" max_normalized_difference="<<maxnorm
    <<" max_direction_angle_absolute="<<maxdir<<" max_final_energy_relative="<<maxenergy<<'\n';
  check(controls==0&&errors==0,"Original C7 comparison failed");check(bool(file),"Failed output");
}
}
int main(int argc,char** argv){
  try{if(argc!=7){std::cerr<<"TABLE INPUT REFERENCE OUTPUT MODE scalar|default\n";return 2;}
    std::string t=argv[1],i=argv[2],r=argv[3],o=argv[4],b=argv[6];int mode=std::stoi(argv[5]);
    check(b=="scalar"||b=="default","Invalid backend");Kokkos::ScopeGuard guard(argc,argv);
#define RUN(M) case M:run<M>(t,i,r,o,b=="scalar");break
    switch(mode){RUN(0);RUN(1);RUN(2);RUN(3);RUN(4);default:throw std::runtime_error("Invalid mode");}
#undef RUN
    return 0;
  }catch(std::exception const& e){std::cerr<<e.what()<<'\n';return 1;}
}

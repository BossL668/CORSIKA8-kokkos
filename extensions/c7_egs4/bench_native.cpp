#include "Egs4C8Adapter.hpp"
#include "Egs4ResonanceDecay.hpp"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>

namespace {
using namespace c7_egs4;using namespace c7_egs4::c8_adapter;
using Clock=std::chrono::steady_clock;
void check(bool ok,char const* message){if(!ok)throw std::runtime_error(message);}
struct Input {TrackState state;std::uint64_t id;};
struct Answer {double v[12]{};int status{},count{};std::uint64_t draws{};};
constexpr char const* names[]{"table_lookup","local_step_loss","multiple_scattering","electron_segment",
  "compton_moller_bhabha","brems_pair","photoelectric","omega_phi_decay"};
KOKKOS_INLINE_FUNCTION void pack(Answer& a,Secondary const* p,int n) {
  a.count=n;
  for(int j=0;j<n;++j){a.v[4*j]=p[j].energy_MeV;a.v[4*j+1]=p[j].direction.x;
    a.v[4*j+2]=p[j].direction.y;a.v[4*j+3]=p[j].direction.z;}
}
template<int Module>KOKKOS_INLINE_FUNCTION Answer evaluate(ModelView model,Input q) {
  Answer out;auto s=q.state;double mass=model.thresholds.mass_MeV;
  CounterStream rng;rng.key={20261003,77,q.id,0,NativeEgs4RandomDomain,0};
  AirRegion region{.001225,1.25e-6,10.};MagneticField field{5.e-5*2.99792458,.6,.8,.2};
  TransportSettings settings{mass+.5,.5,.0625};
  if constexpr(Module==0) {
    auto e=electronQuery(model.tables,s.energy_MeV,mass,s.pdg==11?-1:1);
    auto p=photonQuery(model.tables,s.energy_MeV);
    out.status=(e.status==Status::success&&p.status==Status::success)?0:1;
    out.v[0]=e.rate_per_cm;out.v[1]=e.loss_MeV_per_cm;out.v[2]=e.maximum_step_cm;
    for(int j=0;j<3;++j)out.v[3+j]=e.branches[j];out.v[6]=p.mean_free_path_cm;
    for(int j=0;j<4;++j)out.v[7+j]=p.branches[j];
  }else if constexpr(Module==1) {
    auto e=electronQuery(model.tables,s.energy_MeV,mass,s.pdg==11?-1:1);
    auto p=proposeLocalStep(model.tables,e,s.energy_MeV,mass,settings.electron_cut_total_MeV,
      region.reference_density_g_cm3,s.position_cm.z,region.sterncor,settings.stepfc,.1,1.e9);
    out.status=p.status==Status::success?0:1;out.v[0]=p.loss_MeV_per_cm;out.v[1]=p.rate_per_cm;
    out.v[2]=p.true_step_cm;out.v[3]=p.projected_step_cm;
    out.v[4]=continuousLoss(p.loss_MeV_per_cm,p.true_step_cm); // local only, not full path correction
  }else if constexpr(Module==2) {
    auto r=sampleScattering(model.tables.medium,s.energy_MeV,mass,.01/model.tables.medium.rho,rng);
    out.status=r.status==InteractionStatus::success?0:1;out.v[0]=r.angle;out.v[1]=r.trials;out.v[2]=r.below_scattering_threshold;
  }else if constexpr(Module==3) {
    auto p=prepareElectronSegment(model,s,region,field,settings);
    auto r=finishElectronSegment(model,s,region,field,settings,p,{p.proposed_cm*.8,BoundaryKind::none},rng);
    out.status=r.status==TransportStatus::invalid||r.status==TransportStatus::random_failure||r.status==TransportStatus::sampling_failure?1:0;
    out.v[0]=r.particle.energy_MeV;out.v[1]=r.deposit_MeV;out.v[2]=r.particle.position_cm.x;
    out.v[3]=r.particle.position_cm.y;out.v[4]=r.particle.position_cm.z;out.v[5]=r.particle.direction.x;
    out.v[6]=r.particle.direction.y;out.v[7]=r.particle.direction.z;out.v[8]=r.particle.time_s;
    out.v[9]=r.particle.clock.remaining_mfp;out.v[10]=r.scattering_angle;out.v[11]=r.bend_angle;
  }else if constexpr(Module==4||Module==5) {
    Channel channel;int pdg;
    if constexpr(Module==4) {
      auto index=q.id%3;channel=index==0?Channel::compton:index==1?Channel::moller:Channel::bhabha;
      pdg=index==0?22:index==1?11:-11;
    }else {channel=q.id%2?Channel::bremsstrahlung:Channel::pair_production;pdg=q.id%2?11:22;}
    auto r=sampleAtVertex(model,channel,{pdg,s.energy_MeV,.001225,s.direction},rng);
    out.status=r.disposition==Disposition::replace_parent||r.disposition==Disposition::keep_parent?0:1;
    pack(out,r.local.secondaries.particle,r.local.secondaries.count);
    out.count+=r.disposition==Disposition::keep_parent?10:0;
  }else if constexpr(Module==6) {
    auto r=photoelectric(1.+s.energy_MeV*.001,mass,model.tables.medium.binding_energy_MeV,s.direction);
    out.status=r.secondaries.status==InteractionStatus::success?0:1;pack(out,r.secondaries.particle,r.secondaries.count);
    out.v[10]=r.deposit_MeV;
  }else {
    ResonanceMasses masses{782.65,1019.461,139.57039,134.9768,105.6583755,493.677,497.611,497.611,547.862};
    rng.key.process_id=NativeEgs4HadronRandomDomain;
    auto r=sampleResonanceDecay({2000.+s.energy_MeV,q.id%2?223:333,s.direction},masses,rng);
    out.status=r.status==InteractionStatus::success?0:1;pack(out,r.particle,r.count);
  }
  out.draws=rng.key.draw_id;return out;
}
template<int Module,class Exec>struct Batch {
  ModelView model;Kokkos::View<Input*,typename Exec::memory_space> input;
  Kokkos::View<Answer*,typename Exec::memory_space> output;
  KOKKOS_INLINE_FUNCTION void operator()(int i)const{output(i)=evaluate<Module>(model,input(i));}
};
template<int Module>void scalarLoop(ModelView model,std::vector<Input> const& in,std::vector<Answer>& out) {
  for(std::size_t i=0;i<in.size();++i)out[i]=evaluate<Module>(model,in[i]);
  std::atomic_signal_fence(std::memory_order_seq_cst);
}
template<int Module,class Exec>void measure(ModelView host_model,ModelView device_model,
    std::vector<Input> const& input,Kokkos::View<Input*,typename Exec::memory_space> in,
    Kokkos::View<Answer*,typename Exec::memory_space> out,bool scalar,int rounds,int repeats,std::ofstream& csv) {
  std::vector<Answer> host(input.size()),reference(input.size());scalarLoop<Module>(host_model,input,reference);
  auto launch=[&] {
    if(scalar)scalarLoop<Module>(host_model,input,host);
    else Kokkos::parallel_for(names[Module],Kokkos::RangePolicy<Exec>(0,input.size()),Batch<Module,Exec>{device_model,in,out});
  };
  for(int j=0;j<3;++j)launch();Exec().fence();
  std::vector<double> times;
  for(int round=0;round<rounds;++round) {
    auto start=Clock::now();for(int r=0;r<repeats;++r)launch();Exec().fence();
    double elapsed=std::chrono::duration<double>(Clock::now()-start).count()/repeats;times.push_back(elapsed);
    csv<<(scalar?"scalar":Exec::name())<<','<<Exec().concurrency()<<','<<names[Module]<<','<<input.size()<<','
      <<round<<','<<repeats<<','<<elapsed<<'\n';
  }
  if(!scalar){auto mirror=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},out);
    for(std::size_t i=0;i<input.size();++i)host[i]=mirror(i);}
  double delta=0.,checksum=0.;std::size_t mismatches=0;
  for(std::size_t i=0;i<input.size();++i) {
    auto a=reference[i],b=host[i];mismatches+=a.status!=0||b.status!=0||a.count!=b.count||a.draws!=b.draws;
    for(int j=0;j<12;++j){check(std::isfinite(b.v[j]),"Nonfinite timing result");
      delta=std::max(delta,std::abs(a.v[j]-b.v[j])/(1.+std::abs(a.v[j])));checksum+=b.v[j]/input.size();}
  }
  check(mismatches==0&&delta<1.e-8,"Timing result changed native physics/RNG");
  std::sort(times.begin(),times.end());auto median=times[times.size()/2];
  std::cout<<"module="<<names[Module]<<" N="<<input.size()<<" median_ms="<<median*1000.
    <<" ns_per_input="<<median*1.e9/input.size()<<" normalized_difference="<<delta
    <<" control_mismatches="<<mismatches<<" checksum="<<checksum<<'\n';
}
void run(std::string const& table_path,bool scalar,std::size_t n,int rounds,int repeats,std::string const& destination) {
  using Exec=Kokkos::DefaultExecutionSpace;
  check(n>=1&&n<=1000000&&rounds>=3&&rounds<=31&&repeats>=1&&repeats<=100,"Invalid benchmark size");
  check(!std::filesystem::exists(destination),"Refusing to overwrite a timing result");
  auto start=Clock::now();auto tables=readAirTables(table_path);
  double read_s=std::chrono::duration<double>(Clock::now()-start).count();ChannelThresholds thresholds{.51099895,152.,152.,422.};
  auto model=modelView(tables,thresholds);start=Clock::now();DeviceModel<Exec> device(tables,thresholds);Exec().fence();
  double upload_s=std::chrono::duration<double>(Clock::now()-start).count();
  std::vector<Input> input(n);
  for(std::size_t i=0;i<n;++i) {
    auto& q=input[i];q.id=i+1;q.state.pdg=i%2?11:-11;
    q.state.energy_MeV=.51099895+std::pow(10.,5.*(i%1024)/1023.);
    q.state.direction={.3,.4,std::sqrt(.75)};q.state.position_cm={10.,-20.,-1.e6};
    q.state.clock={.1,electronQuery(model.tables,q.state.energy_MeV,.51099895,q.state.pdg==11?-1:1).rate_per_cm,true};
  }
  Kokkos::View<Input*,typename Exec::memory_space> in("bench_inputs",n);
  Kokkos::View<Answer*,typename Exec::memory_space> out("bench_outputs",n);auto h=Kokkos::create_mirror_view(in);
  for(std::size_t i=0;i<n;++i)h(i)=input[i];Kokkos::deep_copy(in,h);Exec().fence();
  std::ofstream csv(destination);check(bool(csv),"Cannot open timing output");csv<<std::setprecision(17);
  csv<<"backend,execution_concurrency,module,inputs,round,repeats,seconds_per_batch\n";
  std::cout<<std::setprecision(17)<<"backend="<<(scalar?"scalar":Exec::name())<<" execution_concurrency="<<Exec().concurrency()
    <<" table_read_ms="<<read_s*1000.<<" device_model_init_ms="<<upload_s*1000.
    <<" table_rows="<<tables.electrons.size()<<','<<tables.photons.size()<<'\n';
#define MEASURE(M) measure<M,Exec>(model,device.view(),input,in,out,scalar,rounds,repeats,csv)
  MEASURE(0);MEASURE(1);MEASURE(2);MEASURE(3);MEASURE(4);MEASURE(5);MEASURE(6);MEASURE(7);
#undef MEASURE
  check(bool(csv),"Failed timing output");
  std::cout<<"scope=synthetic native C++ kernels; setup/transfers/IO outside timed region; GPU launch+fence included; not C7 Fortran or C8 PROPOSAL timings, not full shower; 1024 energy bins with distinct Philox histories, not independent showers\n";
}
}
int main(int argc,char** argv) {
  try{if(argc!=8){std::cerr<<"EGSDAT scalar|default N rounds repeats output.csv --run\n";return 2;}
    std::string path=argv[1],mode=argv[2],output=argv[6];
    check(mode=="scalar"||mode=="default","Invalid benchmark mode");check(std::string(argv[7])=="--run","Missing run acknowledgement");
    auto n=std::stoull(argv[3]);int rounds=std::stoi(argv[4]),repeats=std::stoi(argv[5]);
    Kokkos::ScopeGuard guard(argc,argv);run(path,mode=="scalar",n,rounds,repeats,output);return 0;
  }catch(std::exception const& e){std::cerr<<e.what()<<'\n';return 1;}
}

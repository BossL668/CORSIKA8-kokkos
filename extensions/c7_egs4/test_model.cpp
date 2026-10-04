#include "Egs4Model.hpp"
#include <algorithm>
#include <cstdint>
#include <iostream>
#include <random>

namespace {
using namespace c7_egs4;
void check(bool b,char const* msg){if(!b)throw std::runtime_error(msg);}
struct Stream {
  std::int64_t seed;int used{};
  KOKKOS_INLINE_FUNCTION bool next(double& u){seed=(seed*48271)%2147483647;u=double(seed)/2147483647.;++used;return true;}
};
struct Case{Channel channel;VertexInput input;std::int64_t seed;bool select;};
struct Answer{VertexOutcome outcome;int used;};
KOKKOS_INLINE_FUNCTION Answer evaluate(ModelView model,Case q) {
  Stream rng{q.seed};auto out=q.select?selectAndSample(model,q.input,rng):sampleAtVertex(model,q.channel,q.input,rng);
  return {out,rng.used};
}
template<class Exec>struct Batch {
  ModelView model;
  Kokkos::View<Case*,typename Exec::memory_space> input;
  Kokkos::View<Answer*,typename Exec::memory_space> output;
  KOKKOS_INLINE_FUNCTION void operator()(int i)const{output(i)=evaluate(model,input(i));}
};
void run(std::string const& path) {
  using Exec=Kokkos::DefaultExecutionSpace;auto tables=readAirTables(path);
  ChannelThresholds thresholds{.51099895,140.,140.,422.};
  auto host=modelView(tables,thresholds);DeviceModel<Exec> device(tables,thresholds);
  std::vector<Case> cases;std::mt19937_64 rng(2026100305);
  auto add=[&](Channel c,int pdg,double E,bool select) {
    for(int j=0;j<128;++j)cases.push_back({c,{pdg,E,.001225,{.3,.4,std::sqrt(.75)}},
                                         std::int64_t(rng()%2147483646)+1,select});
  };
  for(auto c:{Channel::bremsstrahlung,Channel::moller,Channel::bhabha,Channel::annihilation,
              Channel::pair_production,Channel::compton,Channel::photoelectric,
              Channel::electronuclear,Channel::photonuclear,Channel::muon_pair,Channel::continue_transport}) {
    int pdg=(c==Channel::bremsstrahlung||c==Channel::moller||c==Channel::electronuclear)?11:
      (c==Channel::bhabha||c==Channel::annihilation)?-11:22;
    add(c,pdg,1000.,false);
  }
  add(Channel::bremsstrahlung,11,1.e15,false); // must preserve the parent after LPM suppression
  for(int pdg:{11,-11,22})for(double E:{1.1,10.,1.e3,1.e8})add(Channel::invalid,pdg,E,true);
  Kokkos::View<Case*,typename Exec::memory_space> input("model_inputs",cases.size());
  Kokkos::View<Answer*,typename Exec::memory_space> output("model_outputs",cases.size());
  auto h=Kokkos::create_mirror_view(input);for(std::size_t i=0;i<cases.size();++i)h(i)=cases[i];
  Kokkos::deep_copy(input,h);
  Kokkos::parallel_for("C7_CPP_MODEL_DISPATCH",Kokkos::RangePolicy<Exec>(0,cases.size()),Batch<Exec>{device.view(),input,output});
  Exec().fence();auto result=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},output);
  int kept=0,returned=0,replaced=0;double energy_error=0.,direction_error=0.;
  for(std::size_t i=0;i<cases.size();++i) {
    auto q=cases[i];auto reference=evaluate(host,q),actual=result(i);
    auto a=actual.outcome,b=reference.outcome;
    check(a.disposition!=Disposition::error,"Model dispatch error");
    check(a.disposition==b.disposition&&a.channel==b.channel&&actual.used==reference.used,"Dispatch or RNG mismatch");
    check(a.local.secondaries.status==b.local.secondaries.status&&a.local.secondaries.count==b.local.secondaries.count,
          "Output state mismatch");
    check(a.local.deposit_MeV==b.local.deposit_MeV,"Deposition mismatch");
    if(a.disposition==Disposition::needs_host_channel) {
      ++returned;check(a.local.secondaries.count==0&&a.local.secondaries.status==InteractionStatus::host_required,
                       "Unimplemented host channel produced children");
      if(!q.select)check(actual.used==0,"Preselected host channel consumed randoms");
    } else if(a.disposition==Disposition::keep_parent) {
      ++kept;check(a.local.secondaries.count==0,"Suppressed interaction replaced parent");
    } else ++replaced;
    for(int j=0;j<a.local.secondaries.count;++j) {
      auto x=a.local.secondaries.particle[j],y=b.local.secondaries.particle[j];
      check(x.pdg==y.pdg,"Child identity mismatch");
      energy_error=std::max(energy_error,std::abs(x.energy_MeV-y.energy_MeV)/y.energy_MeV);
      direction_error=std::max({direction_error,std::abs(x.direction.x-y.direction.x),
        std::abs(x.direction.y-y.direction.y),std::abs(x.direction.z-y.direction.z)});
    }
  }
  check(energy_error<1.e-10&&direction_error<1.e-9,"Host/device model differences");
  check(kept>128&&returned>=384&&replaced>0,"Missing dispatch coverage");
  Stream bad{1};VertexInput photon{22,1000.,.001,{0.,0.,1.}};
  check(sampleAtVertex(host,Channel::moller,photon,bad).disposition==Disposition::error&&bad.used==0,
        "Wrong species accepted by dispatch");
  auto invalid=photon;invalid.density_g_cm3=-1.;
  check(selectAndSample(host,invalid,bad).disposition==Disposition::error&&bad.used==0,
        "Invalid input consumed randomness");
  // Different per-instance configuration must not leak into the first model.
  auto other=tables;other.medium.binding_energy_MeV=double(.1f);
  DeviceModel<Exec> second(other,thresholds);
  auto other_view=modelView(other,thresholds);
  Stream r1{1},r2{1};
  auto p1=sampleAtVertex(host,Channel::photoelectric,photon,r1),p2=sampleAtVertex(other_view,Channel::photoelectric,photon,r2);
  check(p1.local.deposit_MeV!=p2.local.deposit_MeV,"Model instances share mutable binding energy");
  check(second.view().tables.medium.binding_energy_MeV!=device.view().tables.medium.binding_energy_MeV,
        "Device model configuration leaked");
  std::cout<<"execution_space="<<Exec::name()<<"\nlocal_model_dispatch_inputs="<<cases.size()
    <<"\nreplaced="<<replaced<<"\nkept="<<kept<<"\nexplicit_host_channel_requests="<<returned
    <<"\nmax_energy_relative_cpu_device="<<energy_error<<"\nmax_direction_absolute_cpu_device="<<direction_error
    <<"\nscope=composition of independently tested kernels; not full atmosphere or shower\n";
}
}
int main(int argc,char** argv) {
  try {
    if(argc!=2){std::cerr<<"Usage: test_egs4_model EGSDAT\n";return 2;}
    Kokkos::ScopeGuard guard(argc,argv);run(argv[1]);return 0;
  }catch(std::exception const& e){std::cerr<<e.what()<<'\n';return 1;}
}

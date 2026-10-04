#include <Kokkos_Core.hpp>
#include "StableMuonIonization.hpp"
#include <array>
#include <iomanip>
#include <iostream>
#include <vector>
using namespace c7_egs4::application;
struct Query {double energy,loss;};
struct Result {double legacy_cosine,legacy_sine;MuonIonizationAngles stable;};
constexpr double M=.1056583745,m=.0005109989461;
KOKKOS_INLINE_FUNCTION Result calculate(Query q) {
  double ef=q.energy*(1.-q.loss/q.energy);
  double pi=::sqrt((q.energy+M)*(q.energy-M)),pf=::sqrt((ef+M)*(ef-M));
  double c=((q.energy+m)*ef-q.energy*m-M*M)/(pi*pf);
  double bounded=c>1.?1.:(c<-1.?-1.:c),s2=(1.-bounded)*(1.+bounded);
  return {c,::sqrt(s2>0.?s2:0.),stableMuonIonizationAngles(q.energy,q.loss,M,m)};
}
int main(int argc,char** argv) {
  Kokkos::ScopeGuard guard(argc,argv);
  std::vector<Query> q;
  // First real audited muon ionization, followed by a broad dynamic range.
  // The handoff stores C8 transport total energy: subtract its generated
  // electron mass, not PROPOSAL's slightly different parametrization mass.
  q.push_back({99999.996598077531,.0011626688902335001-5.109989e-4});
  for(double e:{.2,1.,10.,1000.,100000.,1.e8}) {
    double tm=2.*m*(e-M)*(e+M)/(M*M+2.*e*m+m*m);
    for(double fraction:{1.e-12,1.e-9,1.e-6,.001,.1,.5,.99})q.push_back({e,tm*fraction});
  }
  Kokkos::View<Query*> input("input",q.size());auto host=Kokkos::create_mirror_view(input);
  for(std::size_t i=0;i<q.size();++i)host(i)=q[i];Kokkos::deep_copy(input,host);
  Kokkos::View<Result*> output("output",q.size());
  Kokkos::parallel_for("muon_angles",q.size(),KOKKOS_LAMBDA(int i){output(i)=calculate(input(i));});
  auto result=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),output);
  double worst=0.;int zero=0;
  std::cout<<std::setprecision(17)<<"{\"execution_space\":\""<<Kokkos::DefaultExecutionSpace::name()<<"\",\"samples\":[";
  for(std::size_t i=0;i<q.size();++i) {
    long double e=q[i].energy,t=q[i].loss,mu=M,el=m;
    long double p2=(e-mu)*(e+mu),pf=std::sqrt((e-t-mu)*(e-t+mu));
    long double transverse=std::sqrt(2*el*t-t*t*(mu*mu+2*e*el+el*el)/p2);
    double reference=double(transverse/pf),relative=std::abs(result(i).stable.muon_sine/reference-1.);
    if(!std::isfinite(relative)||relative>1.e-10)throw std::runtime_error("Stable angle disagrees with long-double momentum conservation");
    worst=std::max(worst,relative);zero+=result(i).legacy_sine==0.;
    if(i)std::cout<<',';
    std::cout<<"{\"energy\":"<<q[i].energy<<",\"loss\":"<<q[i].loss<<",\"legacy_cosine\":"<<result(i).legacy_cosine
      <<",\"legacy_sine\":"<<result(i).legacy_sine<<",\"stable_sine\":"<<result(i).stable.muon_sine
      <<",\"reference_sine\":"<<reference<<'}';
  }
  std::cout<<"],\"max_relative_sine_error\":"<<worst<<",\"legacy_zero_angles\":"<<zero<<"}\n";
}

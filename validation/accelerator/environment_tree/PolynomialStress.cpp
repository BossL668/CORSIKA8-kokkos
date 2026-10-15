#include <corsika/accelerator/em/common/EnvironmentStableQuartic.hpp>
#include <Kokkos_Core.hpp>
#include <nlohmann/json.hpp>
#include <random>
#include <vector>
#include <algorithm>
#include <iostream>
namespace p=corsika::gpu::em::environment_tree::polynomial;
struct Input{double b,c,d,expected[4];int count;};
int main(){
 std::mt19937_64 rng(20261015);std::uniform_real_distribution<double> u(.1,1.),sg(-1.,1.);
 std::vector<Input> cases;
 // Exact tangencies, repeated roots, a zero root and no real intersection.
 cases.push_back({-2,0,1,{-1,1,0,0},2});
 cases.push_back({0,0,0,{0,0,0,0},1});
 cases.push_back({0,0,1,{0,0,0,0},0});
 cases.push_back({-2,0,0,{-std::sqrt(2.),0,std::sqrt(2.),0},3});
 cases.push_back({-6,8,-3,{-3,1,0,0},2});
 cases.push_back({-6,-8,-3,{-1,3,0,0},2});
 // Actual near-horizontal CPU regressions: tiny positive resolvent root,
 // p<0,r<0. Previously Cardano rounded that root to zero/negative.
 cases.push_back({-12617755126.204693,93475.28779592746,-9.348251154305234e16,{-112361.73542985525,112361.7354224557,0,0},2});
 cases.push_back({-45859813171.72974,2022803.313820246,-8.740579080740667e18,{-214591.75431680618,214591.7542730599,0,0},2});
 cases.push_back({-136605593059.19147,8379065.095819603,-7.311172879370212e19,{-370322.4480604438,370322.44799958117,0,0},2});
 cases.push_back({-300522391899.2113,38191909.466465585,-3.210234311979433e20,{-549169.2268444281,549169.2267182369,0,0},2});
 for(int scale=-6;scale<=9;++scale)for(int gap=1;gap<=12;++gap){
  double b=std::pow(10.,scale),a=b*std::pow(10.,-gap);
  cases.push_back({-(a*a+b*b),0,a*a*b*b,{-b,-a,a,b},4});
 }
 for(int exponent=-6;exponent<=9;++exponent)for(int i=0;i<100;++i){
  double s=std::pow(10.,exponent),a=(.2+u(rng))*s,b=(2+u(rng))*s;
  // Biquadratic, with two and four well-separated real roots.
  cases.push_back({b*b-a*a,0,-a*a*b*b,{-a,a,0,0},2});
  cases.push_back({-(a*a+b*b),0,a*a*b*b,{-b,-a,a,b},4});
  // Two real roots and a complex-conjugate pair, whose total sum is zero.
  double c=-(.2+u(rng))*s,d=(2+u(rng))*s,re=-(c+d)/2,im=(.3+u(rng))*s;
  cases.push_back({im*im-3*re*re+c*d,2*re*(re*re+im*im-c*d),(re*re+im*im)*c*d,{c,d,0,0},2});
  // Four distinct roots with sum zero; reject accidental near-coincidence.
  double r[4]{(-3-u(rng))*s,(-.1-u(rng))*s,(1+u(rng))*s,0};r[3]=-r[0]-r[1]-r[2];
  std::sort(r,r+4);if(r[2]-r[1]>.02*s&&r[3]-r[2]>.02*s){
   double B=0,C=0,D=1;for(int j=0;j<4;++j){D*=r[j];for(int k=j+1;k<4;++k){B+=r[j]*r[k];for(int l=k+1;l<4;++l)C-=r[j]*r[k]*r[l];}}
   cases.push_back({B,C,D,{r[0],r[1],r[2],r[3]},4});
  }
 }
 Kokkos::initialize(Kokkos::InitializationSettings{}.set_num_threads(1));
 nlohmann::json report;{
 using Ex=Kokkos::DefaultExecutionSpace;
 Kokkos::View<Input*,Ex> in("inputs",cases.size());Kokkos::View<p::Roots*,Ex> out("roots",cases.size());
 auto h=Kokkos::create_mirror_view(in);for(unsigned i=0;i<cases.size();++i)h(i)=cases[i];Kokkos::deep_copy(in,h);
 Kokkos::parallel_for("stable quartic property tests",Kokkos::RangePolicy<Ex>(0,cases.size()),KOKKOS_LAMBDA(int i){auto x=in(i);out(i)=p::stableQuartic(x.b,x.c,x.d);});Ex{}.fence();
 auto roots=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},out);int bad=0;double worst=0;
 nlohmann::json examples=nlohmann::json::array();
 for(unsigned i=0;i<cases.size();++i){auto r=roots(i);auto x=cases[i];std::sort(r.values,r.values+r.size);
  r.size=std::unique(r.values,r.values+r.size)-r.values;
  double err=0;bool failure=(int)r.size!=x.count;
  if(!failure)for(int j=0;j<x.count;++j)err=std::max(err,std::abs(r.values[j]-x.expected[j])/std::max(1.e-30,std::abs(x.expected[j])));
  worst=std::max(worst,err);failure|=err>2.e-9;
  if(failure){++bad;if(examples.size()<16)examples.push_back({{"i",i},{"p",x.b},{"q",x.c},{"r",x.d},{"expected",std::vector<double>(x.expected,x.expected+x.count)},
   {"actual",std::vector<double>(r.values,r.values+r.size)},{"relative",err}});}
 }
 report={{"cases",cases.size()},{"failed",bad},{"max_relative_error",worst},{"examples",examples}};
 }Kokkos::finalize();std::cout<<report.dump(2)<<'\n';return report["failed"]==0?0:1;
}

#include "Egs4NuclearTarget.hpp"
#include <fstream>
#include <iostream>
#include <iomanip>
#include <vector>

namespace {
using namespace c7_egs4;
void check(bool v,char const* m){if(!v)throw std::runtime_error(m);}
struct Input {double composition[3],uniform;};
struct Tape {
  double value;int used{};
  KOKKOS_INLINE_FUNCTION bool next(double& x){if(used++)return false;x=value;return true;}
};
struct Result {NuclearTarget target;int used;};
KOKKOS_INLINE_FUNCTION Result evaluate(Input q) {
  Tape r{q.uniform};auto target=sampleManyHadronTarget(q.composition,r);return {target,r.used};
}
struct Evaluate {
  Kokkos::View<Input*> input;Kokkos::View<Result*> output;
  KOKKOS_INLINE_FUNCTION void operator()(int i)const{output(i)=evaluate(input(i));}
};
std::vector<Input> cases() {
  std::vector<Input> out;
  for(auto composition:std::vector<std::vector<double>>{{.7847,.2105,.0048},{1.,0.,0.},{0.,1.,0.},{0.,0.,1.},{1.,1.,1.}}) {
    for(int j=0;j<4096;++j)out.push_back({{composition[0],composition[1],composition[2]},(j+.5)/4096.});
    double a=composition[0]*11.04019,b=a+composition[1]*12.46663,sum=b+composition[2]*28.69952;
    for(double u:{a/sum,b/sum})for(double x:{std::nextafter(u,0.),u,std::nextafter(u,1.)})
      if(x>0.&&x<1.)out.push_back({{composition[0],composition[1],composition[2]},x});
  }
  return out;
}
struct Contracts {
  KOKKOS_INLINE_FUNCTION void operator()(int,int& errors)const {
    double empty[]{0.,0.,0.},bad[]{-.1,1.,0.},good[]{1.,0.,0.};Tape r{.5};
    errors+=sampleManyHadronTarget(empty,r).status!=InteractionStatus::invalid_input||r.used;
    errors+=sampleManyHadronTarget(bad,r).status!=InteractionStatus::invalid_input||r.used;
    Tape exhausted{.5,1};
    errors+=sampleManyHadronTarget(good,exhausted).status!=InteractionStatus::random_failure;
  }
};
}
int main(int argc,char** argv) {
  try {
    auto input=cases();
    if(argc==3&&std::string(argv[1])=="--write-input") {
      std::ofstream f(argv[2]);f<<std::setprecision(17)<<input.size()<<'\n';
      for(auto q:input)f<<q.composition[0]<<' '<<q.composition[1]<<' '<<q.composition[2]<<' '<<q.uniform<<'\n';
      check(bool(f),"Cannot write target fixture");return 0;
    }
    if(argc!=3||std::string(argv[1])!="--compare")return 2;
    std::ifstream reference(argv[2]);std::vector<int> targets;
    for(std::size_t i=0;i<input.size();++i){int a,used;reference>>a>>used;check(bool(reference)&&used==1,"Bad SDPM target oracle");targets.push_back(a);}
    std::string extra;check(!(reference>>extra),"Extra target reference rows");
    Kokkos::ScopeGuard guard(argc,argv);using Exec=Kokkos::DefaultExecutionSpace;
    Kokkos::View<Input*> q("target_inputs",input.size());auto h=Kokkos::create_mirror_view(q);
    for(std::size_t i=0;i<input.size();++i)h(i)=input[i];Kokkos::deep_copy(q,h);
    Kokkos::View<Result*> out("target_results",input.size());
    Kokkos::parallel_for("sdpm_target",input.size(),Evaluate{q,out});
    auto got=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},out);
    int mismatches=0,counts[3]{};
    for(std::size_t i=0;i<input.size();++i) {
      auto a=evaluate(input[i]),b=got(i);
      mismatches+=a.target.mass_number!=targets[i]||b.target.mass_number!=targets[i]||
        a.used!=1||b.used!=1||b.target.status!=InteractionStatus::success||a.target.pdg!=b.target.pdg;
      if(i<4096)++counts[b.target.mass_number==14?0:b.target.mass_number==16?1:2];
    }
    int errors=0;Kokkos::parallel_reduce("target_contracts",1,Contracts{},errors);
    std::cout<<"execution_space="<<Exec::name()<<" inputs="<<input.size()<<" oracle_control_mismatches="<<mismatches
      <<" air_N_O_Ar_counts="<<counts[0]<<','<<counts[1]<<','<<counts[2]<<" contracts="<<errors<<'\n';
    check(!mismatches&&!errors,"SDPM target mismatch");return 0;
  }catch(std::exception const& e){std::cerr<<e.what()<<'\n';return 1;}
}

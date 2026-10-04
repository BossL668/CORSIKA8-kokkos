#include "Egs4Interactions.hpp"
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>

namespace {
using namespace c7_egs4;
constexpr int TapeLength = 256;
void check(bool ok, std::string const& message) {
  if (!ok) throw std::runtime_error(message);
}
struct Case { CollisionInput input; double tape[TapeLength]; };
struct Tape {
  double const* values;
  int count, used{};
  KOKKOS_INLINE_FUNCTION bool next(double& x) {
    if (used == count) return false;
    x = values[used++]; return true;
  }
};
struct Answer { Secondaries state; int used{}; };
KOKKOS_INLINE_FUNCTION Answer evaluate(Case const& q) {
  Tape t{q.tape,TapeLength};
  auto r = sampleCollision(q.input,t);
  return {r,t.used};
}
std::vector<Case> cases() {
  std::mt19937_64 random(20261003);
  std::vector<Case> result;
  Direction directions[]{{0.,0.,1.},{0.,0.,-1.},{.6,0.,-.8},
                         {.3,.4,std::sqrt(.75)},{1.e-11,0.,1.},{1.e-9,0.,-1.}};
  for (int process=0; process<3; ++process)
    for (double kinetic : {.81,1.,2.,10.,1000.,100000.,1.e8})
      for (auto dir : directions) for (int i=0; i<64; ++i) {
        Case c{};
        // Two explicit masses check that no C8 mass is hard-coded in the port.
        double m = i % 2 ? .51099895 : .511;
        c.input = {static_cast<Collision>(process),kinetic+(process?m:0.),m,.4,dir};
        for (auto& u : c.tape) u = double(random() >> 11) * 0x1p-53;
        result.push_back(c);
      }
  return result;
}
void writeCases(std::string const& path) {
  auto all = cases();
  std::ofstream f(path);
  check(bool(f),"Cannot write fixture inputs");
  f << std::setprecision(17) << all.size() << '\n';
  for (auto const& c : all) {
    auto q=c.input;
    f << int(q.process) << ' ' << q.total_MeV << ' ' << q.mass_MeV << ' '
      << q.secondary_kinetic_cut_MeV << ' ' << q.direction.x << ' '
      << q.direction.y << ' ' << q.direction.z << ' ' << TapeLength << '\n';
    for (auto u : c.tape) f << u << ' ';
    f << '\n';
  }
  check(bool(f),"Failed to write complete fixture inputs");
}
template<class Exec> struct EvaluateBatch {
  Kokkos::View<Case*,typename Exec::memory_space> input;
  Kokkos::View<Answer*,typename Exec::memory_space> output;
  KOKKOS_INLINE_FUNCTION void operator()(int i) const { output(i)=evaluate(input(i)); }
};
int pdg(int iq) {
  check(iq>=1 && iq<=3,"C7 reference particle code invalid");
  return iq==1 ? 22 : iq==2 ? -11 : 11;
}
struct Difference { double energy_relative{}, direction_absolute{}; };
void compare(Answer const& a, Answer const& ref, Difference& worst, std::string const& context) {
  check(a.state.status==InteractionStatus::success && a.state.count==2,
        context+": C++ sampler failed");
  check(a.used==ref.used,context+": uniform-consumption mismatch");
  for (int j=0;j<2;++j) {
    auto x=a.state.particle[j], y=ref.state.particle[j];
    check(x.pdg==y.pdg,context+": particle identity/order mismatch");
    double de=std::abs(x.energy_MeV-y.energy_MeV)/std::max(1.e-20,std::abs(y.energy_MeV));
    double dq=std::max({std::abs(x.direction.x-y.direction.x),
                       std::abs(x.direction.y-y.direction.y),std::abs(x.direction.z-y.direction.z)});
    check(std::isfinite(de) && std::isfinite(dq),context+": nonfinite output");
    worst.energy_relative=std::max(worst.energy_relative,de);
    worst.direction_absolute=std::max(worst.direction_absolute,dq);
    check(de<1.e-10 && dq<5.e-8,context+": arithmetic mismatch");
  }
}
void edgeCases() {
  double values[]{.1,.2,.3,.4,.5,.6};
  CollisionInput q{Collision::compton,1.,.511,.4,{0.,0.,1.}};
  Tape empty{values,0};
  check(sampleCollision(q,empty).status==InteractionStatus::random_failure,"Empty tape accepted");
  q.process=Collision::moller; q.total_MeV=.511+.8;
  Tape rng{values,6};
  check(sampleCollision(q,rng).status==InteractionStatus::invalid_input && rng.used==0,
        "Moller threshold handling");
  q.process=Collision::bhabha; q.total_MeV=.511+.4;
  check(sampleCollision(q,rng).status==InteractionStatus::invalid_input,"Bhabha threshold handling");
  q.process=Collision::compton; q.total_MeV=-1.;
  check(sampleCollision(q,rng).status==InteractionStatus::invalid_input,"Negative energy accepted");
  q.total_MeV=1.; q.direction={2.,0.,0.};
  check(sampleCollision(q,rng).status==InteractionStatus::invalid_input,"Nonunit direction accepted");
  q.direction={0.,0.,1.};
  check(sampleCollision(q,rng,0).status==InteractionStatus::invalid_input,"Invalid loop bound accepted");
  double bad[]{1.}; Tape invalid{bad,1};
  check(sampleCollision(q,invalid).status==InteractionStatus::random_failure,"Invalid uniform accepted");
  // Force rejection and confirm no biased accepted value / silent fallback.
  double reject[]{.1,.5,.5,.999999}; Tape limited{reject,4};
  check(sampleCollision(q,limited,1).status==InteractionStatus::trial_limit,"Trial limit fallback");
}
void run(std::string const& reference_path) {
  using Exec=Kokkos::DefaultExecutionSpace;
  edgeCases(); auto all=cases();
  Kokkos::View<Case*,typename Exec::memory_space> inputs("c7_inputs",all.size());
  Kokkos::View<Answer*,typename Exec::memory_space> answers("cpp_results",all.size());
  auto h=Kokkos::create_mirror_view(inputs);
  for (std::size_t i=0;i<all.size();++i) h(i)=all[i];
  Kokkos::deep_copy(inputs,h);
  Kokkos::parallel_for("C7_CPP_COMPT_MOLLER_BHABHA",Kokkos::RangePolicy<Exec>(0,all.size()),
                      EvaluateBatch<Exec>{inputs,answers});
  Exec().fence();
  auto actual=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},answers);
  std::ifstream f(reference_path); check(bool(f),"Cannot open external C7 reference");
  Difference host[3]{},device[3]{};
  double max_energy_balance=0.,max_momentum_balance=0.,max_norm_error=0.;
  int rejected_trials=0,bhabha_electron_first=0,compton_photon_first=0;
  for (std::size_t i=0;i<all.size();++i) {
    Answer oracle{}; int iq1,iq2;
    f >> oracle.state.count >> oracle.used >> iq1 >> iq2;
    oracle.state.status=InteractionStatus::success;
    for (auto& p : oracle.state.particle)
      f >> p.energy_MeV >> p.direction.x >> p.direction.y >> p.direction.z;
    check(bool(f) && oracle.state.count==2,"Incomplete external reference");
    oracle.state.particle[0].pdg=pdg(iq1); oracle.state.particle[1].pdg=pdg(iq2);
    auto q=all[i].input; int process=int(q.process);
    std::string context="case "+std::to_string(i)+" process "+std::to_string(process);
    auto cpu=evaluate(all[i]);
    compare(cpu,oracle,host[process],context+" host");
    compare(actual(i),oracle,device[process],context+" device");
    auto const& s=actual(i).state;
    rejected_trials += s.trials>1;
    bhabha_electron_first += process==2 && s.particle[0].pdg==11;
    compton_photon_first += process==0 && s.particle[0].pdg==22;
    double energy_out=0.; Direction momentum{};
    double momentum_in=process==0?q.total_MeV:std::sqrt((q.total_MeV-q.mass_MeV)*(q.total_MeV+q.mass_MeV));
    for (auto p:s.particle) {
      energy_out+=p.energy_MeV;
      check(p.energy_MeV>=(p.pdg==22?0.:q.mass_MeV),"Secondary below rest mass");
      double pm=p.pdg==22?p.energy_MeV:std::sqrt(std::max(0.,(p.energy_MeV-q.mass_MeV)*(p.energy_MeV+q.mass_MeV)));
      momentum.x+=pm*p.direction.x; momentum.y+=pm*p.direction.y; momentum.z+=pm*p.direction.z;
      max_norm_error=std::max(max_norm_error,std::abs(p.direction.x*p.direction.x+p.direction.y*p.direction.y+p.direction.z*p.direction.z-1.));
    }
    max_energy_balance=std::max(max_energy_balance,std::abs(energy_out/(q.total_MeV+q.mass_MeV)-1.));
    double dp=std::max({std::abs(momentum.x-momentum_in*q.direction.x),
                        std::abs(momentum.y-momentum_in*q.direction.y),
                        std::abs(momentum.z-momentum_in*q.direction.z)})/momentum_in;
    max_momentum_balance=std::max(max_momentum_balance,dp);
    check(s.particle[0].energy_MeV>=s.particle[1].energy_MeV,"Secondary order");
  }
  std::string extra; check(!(f>>extra),"Extra reference rows");
  check(rejected_trials>100 && bhabha_electron_first>0 && compton_photon_first>0,"Missing branch coverage");
  std::cout<<std::setprecision(17)<<"execution_space="<<Exec::name()<<"\nsynthetic_collisions="<<all.size()<<'\n';
  for(int p=0;p<3;++p)
    std::cout<<"process="<<p<<" host_C7_energy_relative="<<host[p].energy_relative
             <<" host_C7_direction_absolute="<<host[p].direction_absolute
             <<" device_C7_energy_relative="<<device[p].energy_relative
             <<" device_C7_direction_absolute="<<device[p].direction_absolute<<'\n';
  std::cout<<"max_energy_balance_relative="<<max_energy_balance
           <<"\nmax_momentum_balance_relative="<<max_momentum_balance
           <<"\nmax_direction_norm_error="<<max_norm_error
           <<"\ncollisions_with_rejection="<<rejected_trials
           <<"\nbhabha_electron_sorted_first="<<bhabha_electron_first
           <<"\ncompton_photon_sorted_first="<<compton_photon_first
           <<"\nruntime_models=C++ only; external Fortran reference is a separate fixture generator\n";
  // Very forward high-energy angles use sqrt(1-cos^2) in the ORIGINAL C7
  // arithmetic. Allow its cancellation error (~sqrt(machine epsilon)), while
  // the separate oracle test above still constrains the rewrite itself.
  check(max_energy_balance<1.e-12 && max_momentum_balance<5.e-8 && max_norm_error<1.e-7,
        "Kinematic identity failure");
}
}
int main(int argc,char** argv) {
  try {
    if(argc==3 && std::string(argv[1])=="--make-cases") { writeCases(argv[2]); return 0; }
    if(argc!=2) { std::cerr<<"Usage: test_egs4_interactions REFERENCE or --make-cases INPUT\n"; return 2; }
    Kokkos::ScopeGuard guard(argc,argv); run(argv[1]); return 0;
  } catch(std::exception const& e) { std::cerr<<e.what()<<'\n'; return 1; }
}

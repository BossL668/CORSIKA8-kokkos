// Diagnostic seed prefilter only: real cascade Philox stream and exponential draws.
// Assumes two draws in air followed by two in rock. Full runs must verify depth.
// No shower final states, channel selection, or decay decisions are forced.
#include <corsika/detail/framework/random/random_iterator/detail/Engine.hpp>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
int main(int argc,char** argv) {
  if(argc!=2) return 2;
  constexpr double energy=1e8, length=1387.4258149368304, rho=2.65;
  double l=std::log(std::log10(energy)+1.826);
  double cc=std::pow(10.,-17.31-6.406*l+1.431*l*l-17.91/l);
  double nc=std::pow(10.,-17.31-6.448*l+1.431*l*l-18.61/l);
  double lambda=1.6605402e-24/(cc+nc); // exact project's constants::u in grams
  std::ofstream out(argv[1]); if(!out) return 3;
  out<<std::setprecision(17)<<"seed,air_draw_g_cm2,rock_draw_g_cm2,depth_m\n";
  std::uint64_t hits=0;
  for(std::uint64_t seed=1;seed<=1000000;++seed) {
    random_iterator::detail::philox rng(seed,0);
    std::exponential_distribution<double> exp(1.);
    double air=lambda*exp(rng); (void)exp(rng);
    double rock=lambda*exp(rng); (void)exp(rng);
    // Conservative air-column bound; actual air flight is only 50 m.
    if(air>100. && rock<length*rho*100.) {
      ++hits; out<<seed<<','<<air<<','<<rock<<','<<rock/(rho*100.)<<'\n';
    }
  }
  std::cout<<std::setprecision(17)<<"trials=1000000 energy_GeV="<<energy
    <<" hits="<<hits<<" lambda_g_cm2="<<lambda<<" first_chord_probability="
    <<-std::expm1(-length*rho*100./lambda)<<'\n';
}

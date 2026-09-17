// PSR-only comparison of original and expanded FLUKA material setup.
#include "../../../applications/detail/mountain/MaterialFlukaInteraction.hpp"
#include <corsika/modules/terrain/TerrainAtmosphere.hpp>
#include <fstream>
#include <iomanip>
#include <csignal>
#include <unistd.h>
using namespace corsika;
using namespace corsika::units::si;
namespace t=corsika::applications::terrain;
template<class Model> void probe(Model& model,std::set<Code> const& targets,std::string const& output) {
  std::signal(SIGALRM,SIG_IGN); ::alarm(0);
  terrain::Environment env; auto cs=env.getCoordinateSystem();
  std::ofstream file(output); if(!file) throw std::runtime_error("cannot create FLUKA oracle output");
  file<<std::setprecision(17);
  for(auto target:targets) for(auto pid:{Code::Proton,Code::Neutron,Code::PiPlus,Code::PiMinus})
    for(double energy:{1.,10.,50.}) {
      auto E=get_mass(pid)+energy*1_GeV;
      FourMomentum projectile(E,MomentumVector(cs,0_GeV,0_GeV,calculate_momentum(E,get_mass(pid))));
      FourMomentum stationary(get_mass(target),MomentumVector(cs,0_GeV,0_GeV,0_GeV));
      auto xs=model.getCrossSection(pid,target,projectile,stationary)/1_mb;
      if(!std::isfinite(xs) || xs<=0.) throw std::runtime_error("invalid FLUKA cross section");
      file<<"XS "<<static_cast<int>(pid)<<' '<<get_nucleus_Z(target)<<' '<<energy<<' '<<xs<<'\n';
    }
  auto target=get_nucleus_code(16,8); auto E=10_GeV+Proton::mass;
  FourMomentum projectile(E,MomentumVector(cs,0_GeV,0_GeV,calculate_momentum(E,Proton::mass)));
  FourMomentum stationary(get_mass(target),MomentumVector(cs,0_GeV,0_GeV,0_GeV));
  for(int event=0;event<3;++event) {
    auto particles=model.generateFinalState(Code::Proton,target,projectile,stationary);
    file<<"EVENT "<<event<<' '<<particles.size()<<'\n';
    for(auto const& particle:particles) {
      auto const& [pid,ekin,direction]=particle;
      auto v=direction.getComponents().getEigenVector();
      file<<"PARTICLE "<<static_cast<int>(pid)<<' '<<ekin/1_GeV<<' '<<v[0]<<' '<<v[1]<<' '<<v[2]<<'\n';
    }
  }
}
int main(int argc,char** argv) {
  try {
    if(argc!=3) throw std::runtime_error("FlukaMaterialOracle stock|expanded OUTPUT");
    logging::set_level(logging::level::warn);
    auto& rng=RNGManager<>::getInstance(); rng.registerRandomStream("fluka"); rng.setSeed(1731);
    std::set<Code> targets;
    for(auto za:{std::pair{1,1},std::pair{6,12},std::pair{7,14},std::pair{8,16},std::pair{11,23},
                 std::pair{12,24},std::pair{13,27},std::pair{14,28},std::pair{18,40},std::pair{26,56}})
      targets.insert(get_nucleus_code(za.second,za.first));
    if(std::string(argv[1])=="stock") {corsika::fluka::InteractionModel model(targets); probe(model,targets,argv[2]);}
    else if(std::string(argv[1])=="expanded") {t::fluka_detail::ExpandedInteractionModel model(targets); probe(model,targets,argv[2]);}
    else throw std::runtime_error("unknown FLUKA oracle mode");
    return 0;
  } catch(std::exception const& e) {std::cerr<<e.what()<<'\n'; return 1;}
}

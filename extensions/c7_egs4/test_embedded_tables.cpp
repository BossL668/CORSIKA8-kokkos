#include "EmbeddedTables.hpp"
#include "Egs4C8Session.hpp"
#include "AirTestFixture.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <stdexcept>

int main(int argc,char** argv) {
  try {
    if(argc!=2)throw std::invalid_argument("Expected reference EGSDAT path");
    auto a=c7_egs4::readAirTables(argv[1]);
    std::ifstream file(argv[1]);std::ostringstream bytes;bytes<<file.rdbuf();
    std::istringstream input(bytes.str());auto b=c7_egs4::readAirTables(input);
    if(a.electrons.size()!=b.electrons.size()||a.photons.size()!=b.photons.size()||
       a.brems_pair_coefficients!=b.brems_pair_coefficients)throw std::runtime_error("Table shape/coefficients differ");
    for(std::size_t i=0;i<a.electrons.size();++i)for(int j=0;j<20;++j)
      if(a.electrons[i].c[j]!=b.electrons[i].c[j])throw std::runtime_error("Electron coefficients differ");
    for(std::size_t i=0;i<a.photons.size();++i)for(int j=0;j<10;++j)
      if(a.photons[i].c[j]!=b.photons[i].c[j])throw std::runtime_error("Photon coefficients differ");
    namespace app=c7_egs4::application;
    auto embedded=app::makeEmbeddedSession();app::Session external(argv[1]);
    app::Configuration config;config.environment=c7_egs4::air_test::environment();config.earth_radius_m=6.371315e6;
    config.seed=1931;config.first_child_id=100;config.maximum_waves=100000;config.queue_capacity=65536;
    auto env=c7_egs4::c8_adapter::makeAirEnvironment(config.environment,config.earth_radius_m,10.,
      c7_egs4::c8_adapter::AirConvention::c7_egs4_four_exponentials);
    for(int pid:{11,-11,22}) {
      auto particle=c7_egs4::air_test::input(env,pid,20.,4000.,8);
      auto run=[&](auto& session) {
        std::vector<double> trace;app::OutputCallbacks out;
        out.step=[&](auto const& s){trace.push_back(s.deposited_energy_GeV);trace.push_back(s.weight);};
        out.radio=[&](auto const& r){trace.push_back(r.step.weight);};
        out.observation=[&](auto const& r){trace.push_back(r.particle.weight);};
        out.discarded=[&](double e,unsigned why){trace.push_back(e);trace.push_back(why);};
        auto result=session.run(config,{particle},out);
        trace.push_back(result.steps);trace.push_back(result.children);return trace;
      };
      if(run(*embedded)!=run(external))throw std::runtime_error("Embedded/external EM transport differs");
    }
    std::cout<<"Embedded/file tables and seeded transport are identical; SHA256 "<<app::embeddedTableSha256()<<'\n';
  }catch(std::exception const& e){std::cerr<<e.what()<<'\n';return 1;}
}

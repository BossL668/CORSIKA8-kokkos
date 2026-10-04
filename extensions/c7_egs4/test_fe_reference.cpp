#include "Egs4FeReference.hpp"
#include <filesystem>
#include <iostream>
#include <cstdlib>
int main(){try {
  char path[]="/tmp/egs4-fe-settings-XXXXXX";if(!mkdtemp(path))return 1;
  std::string antenna=std::string(path)+"/antennas.txt";
  {std::ofstream out(antenna);for(int i=0;i<110;++i)out<<i<<" 0 1100\n";}
  CLI::App cli; c7_egs4::application::FeReference r;r.options(cli);
  int pdg=2212;double e=10.,h=5000.,t=0.,w=1.;
  cli.add_option("--pdg",pdg);cli.add_option("--energy-GeV",e);cli.add_option("--height-m",h);
  cli.add_option("--thin-threshold-GeV",t);cli.add_option("--thin-max-weight",w);
  std::vector<std::string> args{antenna,"--antenna-file","--fe-reference"};cli.parse(args);
  r.defaults(cli,pdg,e,h,t,w);
  if(pdg!=1000260560||e!=215400.||h!=112750.||std::abs(t-.2154)>1e-15||w!=2.154)
    throw std::runtime_error("Incorrect frozen Fe500 preset");
  {std::ofstream out(antenna);out<<"0 0 1100\n";}
  bool rejected=false;try{r.positions();}catch(std::exception const&){rejected=true;}
  if(!rejected)throw std::runtime_error("Incomplete antenna set accepted");
  std::cout<<"PASS Fe215.4 TeV preset, thinning, 110-observer validation\n";
  return 0;
}catch(std::exception const& e){std::cerr<<e.what()<<'\n';return 1;}}

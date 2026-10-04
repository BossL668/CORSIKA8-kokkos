#pragma once
#include <CLI/CLI.hpp>
#include <fstream>
#include <sstream>
#include <array>
#include <cmath>
#include <vector>

namespace c7_egs4::application {
// Frozen Fe500 INPUT convention from runtime_fe_baseline_v1/INPUT_ADAPTER.patch.
// It does not pretend that EGS4 and PROPOSAL implement identical EM physics.
struct FeReference {
  bool enabled{};
  std::string antennas;
  void options(CLI::App& cli) {
    cli.add_flag("--fe-reference",enabled,"Fe500 matched geometry/models: 215.4 TeV, 60 deg, QGSJet-III/FLUKA");
    cli.add_option("--antenna-file",antennas,"Frozen 110-observer NWU antenna file");
  }
  void defaults(CLI::App const& cli,int& pdg,double& energy,double& height,double& thin,double& weight) const {
    if(!enabled) {
      if(!antennas.empty())throw std::invalid_argument("--antenna-file requires --fe-reference");
      return;
    }
    if(!cli.count("--pdg"))pdg=1000260560;
    if(!cli.count("--energy-GeV"))energy=215400.;
    if(!cli.count("--height-m"))height=112750.;
    if(!cli.count("--thin-threshold-GeV"))thin=energy*1.e-6;
    if(!cli.count("--thin-max-weight"))weight=2.154;
    (void)positions(); // fail before any model/output/worker is created
  }
  std::vector<std::array<double,3>> positions()const {
    std::ifstream file(antennas);if(!file)throw std::invalid_argument("Cannot read frozen Fe antennas");
    std::vector<std::array<double,3>> out;std::string line;
    while(std::getline(file,line)) {
      line=line.substr(0,line.find('#'));std::istringstream row(line);row>>std::ws;if(row.eof())continue;
      std::array<double,3> p;std::string extra;
      if(!(row>>p[0]>>p[1]>>p[2]) || row>>extra ||
         !std::isfinite(p[0])||!std::isfinite(p[1])||p[2]!=1100.)
        throw std::invalid_argument("Malformed Fe500 antenna (expected finite NWU, altitude 1100 m)");
      out.push_back(p);
    }
    if(out.size()!=110)throw std::invalid_argument("Fe500 requires exactly 110 antennas");
    return out;
  }
};
}

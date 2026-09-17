// Diagnostic arithmetic only: no RNG, no particle mutation, no stored histories.
#pragma once
#include <corsika/framework/process/SecondariesProcess.hpp>
#include <corsika/modules/thinning/EMThinning.hpp>
#include <corsika/modules/ParticleCut.hpp>
#include <corsika/modules/terrain/TerrainEmSession.hpp>
#include <yaml-cpp/yaml.h>
#include <map>

namespace corsika::applications::terrain {
struct EnergyLedger {
  bool enabled{};
  long double cpuContinuous{},deviceContinuous{},deviceCutTotal{},deviceCutRest{},
      deviceDeposit{},cpuReaction{},deviceReaction{},cpuThinning{},deviceThinning{},worldEscaped{},domainEscaped{};
  std::map<int,long double> cpuReactionByPid,deviceReactionByProcess,deviceThinningByProcess;
  std::uint64_t cpuVertices{},deviceVertices{};
  template<class View> static long double weightedEnergy(View const& v) {
    long double value=0.;for(auto const& p:v)value+=static_cast<long double>(p.getWeight())*(p.getEnergy()/1_GeV);
    return value;
  }
  template<class View> void recordCpuVertex(View const& v) {
    if(!enabled)return;
    auto p=v.getProjectile();auto delta=weightedEnergy(v)-static_cast<long double>(p.getWeight())*(p.getEnergy()/1_GeV);
    cpuReaction+=delta;cpuReactionByPid[static_cast<int>(get_PDG(p.getPID()))]+=delta;++cpuVertices;
  }
  void recordDevice(corsika::terrain::TerrainEmStep const& r) {
    if(!enabled)return;
    using Outcome=corsika::terrain::TerrainEmOutcome;
    auto const w=static_cast<long double>(r.start.weight);
    deviceContinuous+=w*(r.start.energy_GeV-r.end.energy_GeV);
    deviceDeposit+=w*r.deposited_GeV;
    if(r.outcome==Outcome::Children) {
      deviceReaction+=r.unthinned_secondary_total_GeV-w*r.end.energy_GeV;
      deviceReactionByProcess[r.process_id]+=r.unthinned_secondary_total_GeV-w*r.end.energy_GeV;
      deviceThinning+=r.weighted_thinning_delta_GeV;
      deviceThinningByProcess[r.process_id]+=r.weighted_thinning_delta_GeV;++deviceVertices;
    }
    if(r.outcome==Outcome::Cut) {
      deviceCutTotal+=w*r.end.energy_GeV;
      if(r.start.pid==11||r.start.pid==-11)deviceCutRest+=w*(get_mass(Code::Electron)/1_GeV);
    }
    if(r.outcome==Outcome::Escape)worldEscaped+=w*r.end.energy_GeV;
  }
  YAML::Node summary(double initial,double deposited,double window,ParticleCutStatistics const& cut)const {
    YAML::Node n;n["enabled"]=enabled;if(!enabled)return n;
    auto put=[&](char const* key,long double value){n[key]=static_cast<double>(value);};
    long double cpuCutTotal=cut.weighted_kinetic_energy_GeV+cut.weighted_rest_mass_energy_GeV;
    long double reactions=cpuReaction+deviceReaction,thinning=cpuThinning+deviceThinning;
    // A tracked particle disappears at a cut with its TOTAL energy; the writer
    // deposits only its kinetic energy. Separately, a generator may introduce
    // target rest energy or leave untracked target states. Do not conflate
    // those signed changes with the stochastic thinning jump or with heat.
    long double removed=cpuContinuous+deviceContinuous+cpuCutTotal+deviceCutTotal;
    long double sinksNotDeposited=removed-deposited;
    long double offset=deposited+window+worldEscaped+domainEscaped-initial;
    put("domain_escape_GeV",domainEscaped);
    put("initial_GeV",initial);put("deposited_GeV",deposited);put("window_survivor_GeV",window);
    put("world_escape_GeV",worldEscaped);put("cpu_continuous_removed_GeV",cpuContinuous);
    put("device_continuous_removed_GeV",deviceContinuous);put("cpu_cut_total_GeV",cpuCutTotal);
    put("cpu_cut_rest_GeV",cut.weighted_rest_mass_energy_GeV);put("device_cut_total_GeV",deviceCutTotal);
    put("device_cut_rest_GeV",deviceCutRest);put("device_deposit_GeV",deviceDeposit);
    put("cpu_unthinned_generator_balance_GeV",cpuReaction);put("device_unthinned_generator_balance_GeV",deviceReaction);
    put("cpu_thinning_jump_GeV",cpuThinning);put("device_thinning_jump_GeV",deviceThinning);
    put("removed_but_not_in_deposit_GeV",sinksNotDeposited);
    put("removed_minus_deposit_minus_cut_rest_GeV",sinksNotDeposited-cut.weighted_rest_mass_energy_GeV-deviceCutRest);
    put("recorded_offset_GeV",offset);put("predicted_offset_GeV",reactions+thinning-sinksNotDeposited);
    put("unexplained_GeV",offset-(reactions+thinning-sinksNotDeposited));
    put("unexplained_over_initial",(offset-(reactions+thinning-sinksNotDeposited))/initial);
    for(auto [pid,value]:cpuReactionByPid)n["cpu_generator_balance_by_parent_pdg"][pid]=static_cast<double>(value);
    for(auto [pid,value]:deviceReactionByProcess)n["device_generator_balance_by_process"][pid]=static_cast<double>(value);
    for(auto [pid,value]:deviceThinningByProcess)n["device_thinning_jump_by_process"][pid]=static_cast<double>(value);
    n["cpu_vertices"]=cpuVertices;n["device_vertices"]=deviceVertices;
    n["sign_convention"]="after-before for generator/thinning; positive removed energy; total energies in GeV";
    n["generator_balance_is_not_a_pure_numerical_error"]=true;
    n["scope"]="algebraic transport ledger; generator balance includes target exchange and untracked generator states";
    return n;
  }
};

class AuditedEmThinning:public SecondariesProcess<AuditedEmThinning> {
 public:
  AuditedEmThinning(HEPEnergyType threshold,double maximum,bool erase,EnergyLedger& ledger)
      :native_(threshold,maximum,erase),ledger_(ledger){}
  template<class View> void doSecondaries(View& v) {
    auto before=ledger_.enabled?EnergyLedger::weightedEnergy(v):0.L;
    native_.doSecondaries(v);
    if(ledger_.enabled)ledger_.cpuThinning+=EnergyLedger::weightedEnergy(v)-before;
  }
 private:EMThinning native_;EnergyLedger& ledger_;
};
} // namespace corsika::applications::terrain

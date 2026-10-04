#include "Egs4C8HostHandler.hpp"
#include <corsika/accelerator/em/common/EnvironmentSnapshotBuilder.hpp>
#include <corsika/media/UniformMagneticField.hpp>
#include <corsika/modules/sophia/InteractionModel.hpp>
#include <corsika/modules/sibyll/HadronInteractionModel.hpp>
#include <corsika/modules/proposal/HadronicPhotonModel.hpp>
#include <corsika/setup/SetupStack.hpp>
#include <corsika/setup/SetupC7trackedParticles.hpp>
#include <iostream>
#include <iomanip>
#include <set>

namespace {
using namespace corsika;
namespace app=c7_egs4::application;
using Env=Environment<IMagneticFieldModel<IMediumModel>>;
using HostStack=setup::HybridStack<Env>;
void check(bool value,char const* message){if(!value)throw std::runtime_error(message);}
struct Audit {
  std::uint64_t calls{},electromagnetic{},count{};
  template<class View>void doSecondaries(View& view) {
    ++calls;std::set<std::uint64_t> ids;
    auto parent=view.getProjectile();
    for(auto&& p:view) {
      check(ids.insert(p.getHistoryId()).second&&p.getHistoryId()!=parent.getHistoryId(),"Duplicate secondary identity");
      check(p.getParentHistoryId()==parent.getHistoryId()&&p.getGeneration()==parent.getGeneration()+1,"Lost photonuclear ancestry");
      check(p.getTime()==parent.getTime()&&(p.getPosition()-parent.getPosition()).getNorm()==0_m&&p.getWeight()==parent.getWeight(),"Lost photonuclear vertex/weight");
      check(std::isfinite(p.getEnergy()/1_GeV)&&p.getEnergy()>=p.getMass(),"Invalid hadronic secondary energy");
      if(p.getPID()==Code::Photon||p.getPID()==Code::Electron||p.getPID()==Code::Positron)++electromagnetic;
      ++count;
    }
  }
};
struct Arrival {
  int calls{};
  template<class Particle>bool operator()(Particle& p,app::HostRequest const& r) {
    ++calls;check(p.getPDG()==static_cast<PDGCode>(r.particle.pid)&&p.getHistoryId()==r.particle.history_id&&
      p.getParentHistoryId()==r.particle.parent_history_id&&p.getGeneration()==r.particle.generation&&
      p.getStepId()==r.particle.step_id,"Native host identity changed");
    check(std::abs(p.getEnergy()/1_GeV-r.particle.energy_GeV)<1.e-14&&p.getWeight()==r.particle.weight,"Native host energy/weight changed");
    check(p.getNode()&&p.getNode()->hasModelProperties(),"No host medium");
    return true;
  }
};
app::HostRequest request(int pdg,std::uint64_t id,double energy) {
  app::HostRequest r;r.kind=pdg==22?app::HostRequestKind::photonuclear_many_hadrons:
    std::abs(pdg)==13?app::HostRequestKind::muon_transport:app::HostRequestKind::hadron_transport;
  auto& p=r.particle;p.pid=pdg;p.energy_GeV=energy;p.direction[0]=.3;p.direction[1]=.4;p.direction[2]=-std::sqrt(.75);
  p.position_m[0]=10.;p.position_m[1]=20.;p.position_m[2]=constants::EarthRadius::Mean/1_m+4000.1;
  p.time_s=7.e-9;p.weight=1.25;p.history_id=id;p.parent_history_id=1;p.generation=2;p.step_id=3;
  if(pdg==22){r.photonuclear_branch=4;r.target_pdg=1000070140;r.target_atomic_number=7;}
  return r;
}
}
int main() {
  try {
    logging::set_level(logging::level::err);auto& rng=RNGManager<>::getInstance();
    for(auto name:{"sophia","sibyll","proposal"})rng.registerRandomStream(name);
    rng.setSeed(48271);
    Env env;auto cs=env.getCoordinateSystem();Point center{cs,0_m,0_m,0_m};
    create_5layer_atmosphere<IMagneticFieldModel<IMediumModel>,UniformMagneticField>(env,
      AtmosphereId::LinsleyUSStd,center,MagneticFieldVector{cs,{2.e-5_T,1.e-5_T,-4.e-5_T}});
    corsika::sophia::InteractionModel le;sibyll::HadronInteractionModel he(setup::C7trackedParticles);
    proposal::HadronicPhotonModel photons(le,he,80_GeV);Audit audit;Arrival arrival;HostStack stack;
    app::HostHandler<HostStack,Env,decltype(photons),Audit,Arrival> handler(env,cs,photons,audit,arrival);
    for(int pdg:{13,-13,211,-211,2212}) {
      auto r=request(pdg,stack.reserveTransportHistoryIds(1),2.);
      check(handler.canHandle(r)&&handler.handle(stack,r)==1,"Native non-EM import failed");
      check(!handler.canHandle(r),"Repeated host request accepted");
    }
    for(double energy:{.7,5.,120.}) {
      auto r=request(22,stack.reserveTransportHistoryIds(1),energy);auto n=stack.getEntries();
      auto count=handler.handle(stack,r);check(count>0&&stack.getEntries()==n+count,"Photonuclear parent not consumed exactly once");
    }
    auto bad=request(11,stack.reserveTransportHistoryIds(1),1.);bad.kind=app::HostRequestKind::hadron_transport;
    check(!handler.canHandle(bad),"EM particle accepted for scalar transport");
    bad=request(22,stack.reserveTransportHistoryIds(1),5.);bad.target_pdg=0;
    check(!handler.canHandle(bad),"Missing selected target accepted");
    bad.target_pdg=1000070140;bad.target_atomic_number=8;
    check(!handler.canHandle(bad),"Inconsistent selected target accepted");
    auto const& s=handler.statistics();auto const& ledger=photons.energyLedgerStatistics();
    check(arrival.calls==5&&s.transported_muons==2&&s.transported_hadrons==3&&s.selected_photonuclear==3,"Host routing coverage incomplete");
    check(audit.calls==3&&s.generated_secondaries==audit.count&&ledger.low_energy_interactions==2&&ledger.high_energy_interactions==1,"C8 photonuclear dispatch mismatch");
    std::cout<<std::setprecision(17)<<"muons="<<s.transported_muons<<" hadrons="<<s.transported_hadrons
      <<" photonuclear="<<s.selected_photonuclear<<" generated="<<s.generated_secondaries
      <<" directly_generated_em="<<audit.electromagnetic<<" target_input_GeV="<<ledger.weighted_target_total_energy_GeV
      <<" generator_energy_residual_GeV="<<ledger.weighted_energy_residual_GeV<<'\n';
    std::cout<<"scope=real C8 HybridStack plus unchanged SOPHIA/SIBYLL photonuclear final states and native muon/hadron import; no C7 Fortran EM wrapper; not yet a full hadron/muon shower\n";
    return 0;
  }catch(std::exception const& e){std::cerr<<e.what()<<'\n';return 1;}
}

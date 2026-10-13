/* (c) Copyright 2026 CORSIKA Project; BSD-3-Clause. */
#include "MuonBremsDriver.hpp"
#include <corsika/accelerator/em/common/tables/ProposalNativeTableExporter.hpp>
#include <corsika/accelerator/em/common/tables/ProposalNativeAux.hpp>
#include <corsika/accelerator/em/common/TransportMass.hpp>
#include <corsika/modules/proposal/ProposalProcessBase.hpp>
#include <PROPOSAL/PROPOSAL.h>
#include <PROPOSAL/secondaries/SecondariesCalculator.h>
#include <PROPOSAL/secondaries/parametrization/bremsstrahlung/BremsEGS4Approximation.h>
#include <algorithm>
#include <iostream>
#include <stdexcept>
using namespace corsika;
namespace em=muon_test::em;
namespace nt=em::tables;
void require(bool ok,char const* message){if(!ok)throw std::runtime_error(message);}
struct Owner {
  PROPOSAL::Air medium;Code code;PROPOSAL::crosssection_list_t cross;
  std::unique_ptr<PROPOSAL::Interaction> interaction;
  std::unique_ptr<PROPOSAL::Displacement> displacement;
  std::unique_ptr<PROPOSAL::crosssection::BremsLPM> brems;
  std::unique_ptr<PROPOSAL::crosssection::PhotoPairLPM> pair;
  Owner(Code c):code(c){
    cross=proposal::make_cross_sections(c,medium,.5_MeV,true);
    interaction=PROPOSAL::make_interaction(cross,true,true);
    if(c==Code::Photon)pair=std::make_unique<PROPOSAL::crosssection::PhotoPairLPM>(proposal::particle.at(c),medium,PROPOSAL::crosssection::PhotoPairKochMotz());
    else {
      brems=std::make_unique<PROPOSAL::crosssection::BremsLPM>(proposal::particle.at(c),medium,PROPOSAL::crosssection::BremsElectronScreening());
      displacement=PROPOSAL::make_displacement(cross,true);
    }
  }
  auto view()const{return proposal::NativeInteractionCalculatorView{code,medium.GetHash(),&medium,interaction.get(),pair.get(),brems.get(),.5_MeV,proposal::particle.at(code).mass};}
  auto continuous()const{return proposal::NativeContinuousCalculatorView{code,medium.GetHash(),&medium,displacement.get(),.5_MeV,proposal::particle.at(code).mass};}
};
int main(int argc,char**argv){try{
  require(argc>1,"usage: testMuonBrems TABLES");
  logging::set_level(logging::level::warn);PROPOSAL::Logging::SetGlobalLoglevel(spdlog::level::err);
  PROPOSAL::InterpolationSettings::TABLES_PATH=argv[1];
  struct Runtime{Runtime(){muon_test::initialize();}~Runtime(){muon_test::finalize();}} runtime;
  Owner photon(Code::Photon),electron(Code::Electron),positron(Code::Positron),minus(Code::MuMinus),plus(Code::MuPlus);
  std::vector<proposal::NativeInteractionCalculatorView> views{photon.view(),electron.view(),positron.view(),minus.view(),plus.view()};
  auto tables=nt::exportProposalNativeTables(views,{electron.continuous(),positron.continuous(),minus.continuous(),plus.continuous()});
  auto aux=nt::loadOrCreateProposalNativeAux(views,"muon_aux");
  auto reload=nt::loadOrCreateProposalNativeAux(views,"muon_aux");
  require(reload.cache_hit && reload.brems_lpm.muon_e_lpm_MeV==aux.brems_lpm.muon_e_lpm_MeV,"cache roundtrip");
  auto host=nt::makeProposalNativeHostView(tables);
  unsigned long count=0;double maxV=0,maxD=0,maxE=0,maxL=0;
  for(int pid:{13,-13}){
    auto& owner=pid==13?minus:plus;
    auto factory=PROPOSAL::DefaultFactory<PROPOSAL::secondaries::Parametrization>::Create(PROPOSAL::InteractionType::Brems,proposal::particle.at(owner.code),owner.medium);
    auto* native=dynamic_cast<PROPOSAL::secondaries::BremsEGS4Approximation*>(factory.get());
    require(native,"unexpected native angular model");
    for(auto const& comp:owner.medium.GetComponents())for(double E:{1.e3,1.e4,1.e6,1.e8,1.e11})for(double vf:{.01,.3,.9}){
      auto column=nt::native_detail::findDndx(host,pid,em::BremsProcessId,comp.GetHash());
      require(column,"missing muon brems column");auto lim=nt::native_detail::kinematicLimits(host,*column,E);
      double v=lim.minimum+(lim.maximum-lim.minimum)*vf;
      if(v*E<.5)continue;
      std::vector<muon_test::Input> in;
      for(unsigned i=0;i<256;++i)in.push_back({pid,em::BremsProcessId,E,v,i%2?1.2e-3:1.e-6,(i+.5)/256,comp.GetHash(),count+i+1});
      auto device=muon_test::run(tables,aux.brems_lpm,in);
      auto cpu=muon_test::run(tables,aux.brems_lpm,in,false);
      for(unsigned i=0;i<in.size();++i){auto const& y=device[i];auto const& h=cpu[i];auto const& q=in[i];
        require(!y.fallback && !y.error && !y.replay && y.loss_status==0,"valid muon input fell back");
        require(y.suppressed==h.suppressed && y.count==h.count && y.trials==0,"device/host mismatch");
        double ref=-1.;for(auto const& r:owner.interaction->Rates(E))if(int(r.crosssection->GetInteractionType())==em::BremsProcessId && r.comp_hash==comp.GetHash())ref=r.crosssection->CalculateStochasticLoss(comp.GetHash(),E,q.quantile*r.rate);
        require(ref>0,"native selected rate missing");maxV=std::max(maxV,std::abs(y.loss-ref)/ref);
        auto lpm=owner.brems->suppression_factor(E,v,comp,q.density/owner.medium.GetMassDensity());
        maxL=std::max(maxL,std::abs(std::clamp(lpm,0.,1.)-std::clamp(y.lpm,0.,1.)));
        if(y.suppressed){require(y.count==0,"suppression emitted particles");continue;}
        require(y.count==2 && y.pids[0]==pid && y.pids[1]==22,"wrong muon brems children");
        double corrected=y.energies[0]-(em::transportMassGeV(pid)-PROPOSAL::MMU/1000.);
        maxE=std::max(maxE,std::abs(corrected+y.energies[1]-E/1000.)/(E/1000.));
        PROPOSAL::Cartesian3D dir(0.,0.,q.history%3==0?1.:-1.);if(q.history%3==2)dir=PROPOSAL::Cartesian3D(.6,0.,.8);
        std::vector<double> rnd{y.azimuth};auto d=native->CalculateDirections(dir,E,E*v,comp,rnd);
        for(unsigned j=0;j<2;++j){
          require(std::abs(y.energies[j]-h.energies[j])<1.e-10*E/1000.,"host/device energy mismatch");
          double norm=0.;for(int k=0;k<3;++k){norm+=y.directions[j][k]*y.directions[j][k];maxD=std::max({maxD,std::abs(y.directions[j][k]-(j?d.second[k]:d.first[k])),std::abs(y.directions[j][k]-h.directions[j][k])});}
          require(std::abs(norm-1.)<1.e-10,"direction norm");
        }
      }
      count+=in.size();
    }
    std::vector<muon_test::Input> bad;
    for(unsigned fault=1;fault<=8;++fault)bad.push_back({pid,em::BremsProcessId,1.e6,.1,1.e-3,.5,owner.medium.GetComponents()[0].GetHash(),99000000+fault,fault});
    for(auto const& y:muon_test::run(tables,aux.brems_lpm,bad))require(y.fallback && !y.count,"invalid input not guarded");
  }
  require(maxV<1.e-7 && maxD<1.e-10 && maxE<3.e-12 && maxL<1.1e-6,"native oracle tolerance failed");
  std::cout<<"PASS muon brems samples="<<count<<" max_relative_v="<<maxV<<" max_direction="<<maxD<<" max_relative_energy="<<maxE<<" max_LPM="<<maxL<<std::endl;
  return 0;
}catch(std::exception const& e){std::cerr<<e.what()<<std::endl;return 1;}}

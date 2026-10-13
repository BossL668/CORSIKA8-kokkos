/* (c) Copyright 2026 CORSIKA Project; BSD-3-Clause. */
#include "PhotonMuonPairDriver.hpp"
#include <corsika/accelerator/em/common/Philox.hpp>
#include <corsika/accelerator/em/common/ProcessCapabilities.hpp>
#include <corsika/accelerator/em/common/TransportMass.hpp>
#include <corsika/accelerator/em/common/NativeMuonPairCut.hpp>
#include <corsika/accelerator/em/common/tables/ProposalNativeTableExporter.hpp>
#include <corsika/accelerator/em/PhotonFinalStateRandomDomains.hpp>
#include <corsika/modules/proposal/ProposalProcessBase.hpp>
#include <PROPOSAL/PROPOSAL.h>
#include <PROPOSAL/secondaries/SecondariesCalculator.h>
#include <PROPOSAL/secondaries/parametrization/photomupairproduction/PhotoMuPairProductionBurkhardtKelnerKokoulin.h>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
using namespace corsika;
namespace em=corsika::gpu::em;
namespace nt=em::tables;
namespace test=photon_muon_test;
namespace domains=corsika::accelerator::em;
void require(bool good,char const* reason){if(!good)throw std::runtime_error(reason);}
struct Runtime {Runtime(){test::initialize();}~Runtime(){test::finalize();}};
int main(int argc,char** argv){try{
  logging::set_level(logging::level::warn);
  PROPOSAL::Logging::SetGlobalLoglevel(spdlog::level::err);
  std::filesystem::path cache=argc>1?argv[1]:"photomu_tables";
  std::filesystem::create_directories(cache);
  PROPOSAL::InterpolationSettings::TABLES_PATH=cache.string();
  unsigned n=argc>2?std::stoul(argv[2]):2048;
  require(n>=16 && n<=100000,"samples outside 16..100000");
  Runtime runtime;
  PROPOSAL::Air medium;
  auto cross=proposal::make_cross_sections(Code::Photon,medium,.5_MeV,true);
  auto interaction=PROPOSAL::make_interaction(cross,true,true);
  std::vector<proposal::NativeInteractionCalculatorView> views{
    {Code::Photon,medium.GetHash(),&medium,interaction.get(),nullptr,nullptr,.5_MeV,0.}};
  auto tables=nt::exportProposalNativeTables(views,{});
  // PROPOSAL 7.6.2's two secondary constructors both use build_dndx(hash=0).
  // Isolate BKK from any Koch--Motz secondary cache. The rate export itself
  // has a process-specific hash; never use a cached wrong-process CDF as oracle.
  auto bkkCache=cache/"BKK_secondary_reference_v1";
  std::filesystem::create_directories(bkkCache);
  PROPOSAL::InterpolationSettings::TABLES_PATH=bkkCache.string();
  auto factory=PROPOSAL::DefaultFactory<PROPOSAL::secondaries::Parametrization>::Create(
    PROPOSAL::InteractionType::PhotoMuPair,PROPOSAL::GammaDef(),medium);
  auto* native=dynamic_cast<PROPOSAL::secondaries::PhotoMuPairProductionBurkhardtKelnerKokoulin*>(factory.get());
  require(native,"unexpected native secondary model");
  require(tables.constants.muon_mass_MeV==PROPOSAL::MMU,"wrong exported muon mass");
  require(em::gpuProcessCapability(22,em::PhotonMuonPairProcessId)==em::GpuProcessCapability::PhotonMuonPair,"still CPU-only");
  for(double q:{0.,1.e-12,.5,1.-1.e-12,1.})
    require(!em::proposalNativeSelectionRequiresReplay(22,em::PhotonMuonPairProcessId,q),"v=1 replay is unnecessary");
  std::ofstream csv("cells.csv");csv<<std::setprecision(17);
  csv<<"Z,energy_MeV,N,native_subrest_cut,x_abs_error,probe_abs_error,energy_rel_error,direction_abs_error,norm_error,host_device_x_error,host_fixture_s,device_fixture_s\n";
  std::ofstream bad("native_unphysical.csv");bad<<std::setprecision(17);
  bad<<"Z,energy_MeV,history,uniform,native_x,ported_x,mu_minus_MeV,mu_plus_MeV\n";
  std::uint64_t checked=0,unphysical=0;double maxX=0.,maxP=0.,maxE=0.,maxD=0.,maxH=0.;
  std::vector<em::EmParticleState> cutChildren;
  double nativeSignedDeposit=0.;
  // PROPOSAL refines the table's lower edge above the nominal BKK threshold.
  // Below that edge its rate is zero: no such physical vertex is selected.
  auto hostView=nt::makeProposalNativeHostView(tables);
  for(auto const& comp:medium.GetComponents()) {
    auto column=nt::native_detail::findDndx(hostView,22,em::PhotonMuonPairProcessId,comp.GetHash());
    require(column,"missing photomu rate column");
    std::cout<<"native table Z="<<comp.GetNucCharge()<<" lower="<<column->spline.energy_axis.low
      <<" upper="<<column->spline.energy_axis.high<<" rate_lower="<<column->lower_energy_limit_MeV<<std::endl;
  for(double E:{column->spline.energy_axis.low*1.001,3000.,1.e4,1.e5,1.e6,1.e8,1.e11}){
    std::vector<test::Input> input;
    double probes[]={0.,1.e-12,1.e-6,.1,.5,.9,1.-1.e-6,1.-1.e-12,1.};
    for(unsigned i=0;i<n;++i)input.push_back({E,probes[i%9],comp.GetHash(),checked+i+1});
    auto cpu=test::run(tables,input,false);auto gpu=test::run(tables,input,true);
    double errX=0.,errP=0.,errE=0.,errD=0.,errN=0.,errH=0.;
    unsigned unphysicalCell=0;
    for(unsigned i=0;i<n;++i){auto q=input[i];auto y=gpu.outputs[i];auto h=cpu.outputs[i];
      require(y.public_status==0 && y.public_v==1.,"public loss v changed");
      require(y.probe_status==0,"split endpoint query failed");
      em::RandomNumberKey key{26101193,73,q.history,0,unsigned(em::PhotonMuonPairProcessId),domains::PhotonMuonPairSplitDrawId};
      double u=em::uniformOpen01(key);key.draw_id=domains::PhotonMuonPairAzimuthDrawId;
      double phi=em::uniformOpen01(key);
      require(y.uniform==u,"CPU split RNG address changed");
      auto energy=(E/1000.)*1000.;
      double ref=native->Calculatex(energy,u,comp);
      double refProbe=native->Calculatex(E,q.quantile,comp);
      errX=std::max(errX,std::abs(y.x-ref));errP=std::max(errP,std::abs(y.probe-refProbe));
      errH=std::max(errH,std::abs(y.x-h.x));
      // Do not hide native interpolation pathologies by clipping energies or
      // by silently counting protected inputs as successful physical events.
      if(ref*energy<PROPOSAL::MMU || (1.-ref)*energy<PROPOSAL::MMU){
        require(!y.fallback && !h.fallback && y.count==2 && h.count==2 && !y.error,
          "native sub-rest-mass energy was not aligned");
        ++unphysicalCell;
        bad<<comp.GetNucCharge()<<','<<E<<','<<q.history<<','<<u<<','<<ref<<','<<y.x<<','
           <<(1.-ref)*energy<<','<<ref*energy<<'\n';
      }
      require(!y.fallback && !y.error && y.count==2,"valid photomu input failed");
      require(y.azimuth==phi,"CPU azimuth RNG address changed");
      PROPOSAL::Cartesian3D dir(q.history%3==2?.6:0.,0.,q.history%3==0?1.:q.history%3==1?-1.:.8);
      auto dirs=native->CalculateDirections(dir,energy,comp,phi);
      auto energies=native->CalculateEnergy(energy,ref);
      for(unsigned j=0;j<2;++j){auto const& child=y.children[j];
        int pid=j?-13:13;
        double nativeE=j?std::get<1>(energies):std::get<0>(energies);
        double expected=(nativeE/1000.-PROPOSAL::MMU/1000.)+em::transportMassGeV(pid);
        errE=std::max(errE,std::abs(child.energy_GeV-expected)/(E/1000.));
        auto const& direction=j?std::get<1>(dirs):std::get<0>(dirs);
        double norm=0.;for(int k=0;k<3;++k){
          errD=std::max(errD,std::abs(child.direction[k]-direction[k]));
          norm+=child.direction[k]*child.direction[k];}
        errN=std::max(errN,std::abs(norm-1.));
        require(child.pid==pid && child.energy_GeV>0.,"bad muon PID/energy");
        if(child.energy_GeV<=em::transportMassGeV(pid)){
          require(em::isNativeMuonPairCut(child) && y.cut_status[j]==2 && h.cut_status[j]==2,
            "native child was not cut before transport");
          require(y.cuts[j].start_grammage_g_per_cm2==y.cuts[j].end_grammage_g_per_cm2 &&
            y.cuts[j].deposited_energy_GeV==y.cuts[j].cut_deposited_energy_GeV,
            "cut created a continuous trajectory");
          nativeSignedDeposit+=test::nativeCutReference(child,y.cuts[j]);
          cutChildren.push_back(child);
        }else require(child.reserved==0,"ordinary muon was tagged for special cut");
        require(child.parent_history_id==q.history && child.history_id==1000+2*q.history+j && child.generation==1 && child.step_id==0,"bad child identity");
        require(child.time_s==1.e-6 && child.position_m[0]==3. && child.weight==1.,"vertex or weight changed");
      }
      auto sum=y.children[0].energy_GeV+y.children[1].energy_GeV-y.mass_correction;
      require(std::abs(sum-E/1000.)/(E/1000.)<3.e-14,"local energy balance failed");
    }
    std::cout<<std::setprecision(12)<<"cell E="<<E<<" Z="<<comp.GetNucCharge()<<" x="<<errX<<" probe="<<errP<<" dir="<<errD<<" native_unphysical="<<unphysicalCell<<std::endl;
    require(errX<2.e-10 && errP<2.e-10 && errH<2.e-10,"native/ported split mismatch");
    require(errE<2.e-10 && errD<1.e-11 && errN<1.e-11,"native/ported kinematics mismatch");
    csv<<comp.GetNucCharge()<<','<<E<<','<<n<<','<<unphysicalCell<<','<<errX<<','<<errP<<','<<errE<<','<<errD<<','<<errN<<','<<errH<<','<<cpu.kernel_s<<','<<gpu.kernel_s<<'\n';csv.flush();bad.flush();
    input.resize(16);test::checkBatch(tables,input);
    // Thin on both execution spaces: no unthinned particle or partial output.
    for(auto& q:input)q.thinning=1;
    auto thin=test::run(tables,input,true);auto thinCPU=test::run(tables,input,false);
    for(unsigned i=0;i<input.size();++i){auto const& y=thin.outputs[i];auto const& h=thinCPU.outputs[i];
      require(!y.fallback && !y.error && y.count==2 && y.count==h.count && y.mask==3 && y.mask==h.mask,"native non-EM thinning bypass failed");
      for(unsigned j=0;j<y.count;++j)
        require(y.children[j].pid==h.children[j].pid && y.children[j].weight==1. && h.children[j].weight==1.,"muon weight changed under EM thinning");
    }
    checked+=n;unphysical+=unphysicalCell;maxX=std::max(maxX,errX);maxP=std::max(maxP,errP);maxE=std::max(maxE,errE);maxD=std::max(maxD,errD);maxH=std::max(maxH,errH);
  }}
  auto component=medium.GetComponents()[0].GetHash();
  std::vector<test::Input> guards{{1.e3,.5,component,1,1},{1.e3,.5,component,2,2},{1.e3,.5,component,3,3},
    {100.,.5,component,4},{NAN,.5,component,5},{1.e16,.5,component,6},
    {1.e4,.5,component,7,4},{1.e4,.5,component,8,5}};
  for(auto const& y:test::run(tables,guards,true).outputs)
    require(y.fallback && y.count==0,"invalid table/input did not fail closed");
  std::cout<<std::setprecision(17)<<"DIFFERENTIAL_PASS samples="<<checked<<" guards="<<guards.size()<<" max_x="<<maxX
    <<" max_probe="<<maxP<<" max_energy="<<maxE<<" max_direction="<<maxD<<" max_host_device_x="<<maxH<<std::endl;
  if(!cutChildren.empty()){
    auto profile=test::checkCutProfile(cutChildren);
    require(std::abs(profile.weighted_native_muon_pair_cut_energy_GeV-nativeSignedDeposit)<1.e-11 &&
      std::abs(profile.weighted_deposited_energy_GeV-nativeSignedDeposit)<1.e-11,
      "device/cooperative profile lost signed native cuts");
    test::checkCutOutput(profile);
    std::cout<<"NATIVE_CUT_PASS children="<<cutChildren.size()<<" signed_deposit_GeV="<<nativeSignedDeposit
      <<" device_deposit_GeV="<<profile.weighted_deposited_energy_GeV<<std::endl;
  }
  std::cout<<"PASS native C8 alignment; subrest inputs="<<unphysical<<" / "<<checked<<std::endl;
  return 0;
}catch(std::exception const& e){std::cerr<<e.what()<<std::endl;return 1;}}

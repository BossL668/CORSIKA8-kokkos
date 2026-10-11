/* (c) Copyright 2026 CORSIKA Project; BSD-3-Clause. */
#include "PhotonPairFallbackDriver.hpp"
#include <corsika/accelerator/em/common/Philox.hpp>
#include <corsika/accelerator/em/common/PhotonPairFinalState.hpp>
#include <corsika/accelerator/em/common/ProcessCapabilities.hpp>
#include <corsika/accelerator/em/common/TransportMass.hpp>
#include <corsika/accelerator/em/PhotonFinalStateRandomDomains.hpp>
#include <corsika/accelerator/em/common/tables/ProposalNativeTableExporter.hpp>
#include <corsika/modules/proposal/ProposalProcessBase.hpp>
#include <PROPOSAL/PROPOSAL.h>
#include <PROPOSAL/secondaries/SecondariesCalculator.h>
#include <PROPOSAL/secondaries/parametrization/photopairproduction/PhotoPairProductionKochMotz.h>
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <stdexcept>

using namespace corsika;
namespace em = corsika::gpu::em;
namespace nt = em::tables;
namespace test = corsika::accelerator::em::testing;
namespace domains = corsika::accelerator::em;

namespace {
  void require(bool pass, char const* message) {
    if (!pass) throw std::runtime_error(message);
  }
  struct Runtime {
    Runtime(){test::initializePhotonPairFallbackTest();}
    ~Runtime(){test::finalizePhotonPairFallbackTest();}
  };
}

int main(int argc, char** argv) { try {
  logging::set_level(logging::level::warn);
  PROPOSAL::Logging::SetGlobalLoglevel(spdlog::level::err);
  auto cache=std::filesystem::path(argc>1?argv[1]:"pair_fallback_tables");
  std::filesystem::create_directories(cache);
  PROPOSAL::InterpolationSettings::TABLES_PATH=cache.string();
  auto n=argc>2?std::stoul(argv[2]):256u;
  require(n>0 && n<=100000,"sample count must be in [1,100000]");
  Runtime runtime;
  PROPOSAL::Air medium;
  auto cross=proposal::make_cross_sections(Code::Photon,medium,.4_MeV,true);
  auto interaction=PROPOSAL::make_interaction(cross,true,true);
  auto pair=std::make_unique<PROPOSAL::crosssection::PhotoPairLPM>(
      proposal::particle.at(Code::Photon),medium,PROPOSAL::crosssection::PhotoPairKochMotz());
  std::vector<proposal::NativeInteractionCalculatorView> views{
    {Code::Photon,medium.GetHash(),&medium,interaction.get(),pair.get(),nullptr,.4_MeV,0.}};
  auto tables=nt::exportProposalNativeTables(views,{});
  // This photon-only fixture needs no electron/muon auxiliary tables.
  nt::ProposalNativeAuxData aux{};
  auto exported=pair->ExportParameters();
  auto& lpm=aux.photon_pair_lpm;
  lpm.baseline_mass_density_g_per_cm3=exported.mass_density_g_per_cm3;
  lpm.molecular_density_per_cm3=exported.molecular_density_per_cm3;
  lpm.sum_charge=exported.sum_charge;
  lpm.e_lpm_MeV=exported.e_lpm_MeV;
  lpm.classical_electron_radius_cm=PROPOSAL::RE;
  lpm.fine_structure_constant=PROPOSAL::ALPHA;
  for(auto const& c:medium.GetComponents())
    lpm.components[lpm.component_count++]={c.GetHash(),c.GetNucCharge(),c.GetLogConstant()};
  auto factory=PROPOSAL::DefaultFactory<PROPOSAL::secondaries::Parametrization>::Create(
      PROPOSAL::InteractionType::Photopair,PROPOSAL::GammaDef(),medium);
  auto* native=dynamic_cast<PROPOSAL::secondaries::PhotoPairProductionKochMotzSauter*>(factory.get());
  require(native,"unexpected native secondary model");
  auto components=medium.GetComponents();
  auto host=nt::makeProposalNativeHostView(tables);
  std::uint64_t checked=0, fallbacks=0, normal_exhaustions=0, maximum_attempts=0;
  double maximum_cdf_error=0.,maximum_rejection_error=0.;
  for(auto const& component:components)for(double E:{1.1,2.,3.,5.,10.,1.e3,1.e7,1.e8,1.e11}) {
    auto v=nt::queryProposalNativeLossFraction(host,22,em::PhotonPairProcessId,component.GetHash(),E,.5);
    require(v.status==nt::NativeQueryStatus::Success && v.value==1.,"public photon loss must remain v=1");
    for(unsigned cap:{64u,1u,0u}) {
      std::vector<test::PhotonPairFallbackInput> inputs;
      for(unsigned i=0;i<n;++i)inputs.push_back({E,1.e-3,component.GetHash(),checked+i+1,cap});
      auto output=test::runPhotonPairFallbackTest(tables,aux,inputs);
      for(unsigned i=0;i<n;++i) {
        auto q=inputs[i]; auto y=output[i];
        require(!y.fallback && !y.error,"valid photon split failed");
        require(std::isfinite(y.x) && y.x>=em::ElectronMassMeV/E && y.x<=1-em::ElectronMassMeV/E,"unphysical split");
        bool accepted=false; double expected=0.,expected_uniform=0.; unsigned expected_draw=0;
        for(unsigned attempt=0;attempt<cap;++attempt) {
          em::RandomNumberKey key{26101117,71,q.history,0,unsigned(em::PhotonPairProcessId),domains::PhotonPairAnalyticCandidateDrawIdBase+attempt};
          auto u=em::uniformOpen01(key); key.draw_id=domains::PhotonPairAnalyticAcceptanceDrawIdBase+attempt;
          auto trial=em::photonPairFinalStateTrial(aux.photon_pair_lpm,q.component,(E/1000.)*1000.,u,em::uniformOpen01(key));
          require(trial.status==em::PhotonPairFinalStateStatus::Success,"invalid reference rejection trial");
          if(trial.accepted) {
            accepted=true;expected=trial.split_fraction;expected_uniform=u;
            expected_draw=domains::PhotonPairAnalyticCandidateDrawIdBase+attempt;
            maximum_attempts=std::max(maximum_attempts,std::uint64_t(attempt+1));break;
          }
        }
        if(!accepted) {
          em::RandomNumberKey key{26101117,71,q.history,0,unsigned(em::PhotonPairProcessId),domains::PhotonPairSplitDrawId};
          expected_uniform=em::uniformOpen01(key);expected_draw=domains::PhotonPairSplitDrawId;
          expected=native->CalculateRho((E/1000.)*1000.,expected_uniform,component);++fallbacks;
          if(cap==64)++normal_exhaustions;
          maximum_cdf_error=std::max(maximum_cdf_error,std::abs(expected-y.x));
        } else maximum_rejection_error=std::max(maximum_rejection_error,std::abs(expected-y.x));
        require(y.draw==expected_draw && y.uniform==expected_uniform,"changed RNG or premature/delayed CDF lookup");
        require(std::abs(expected-y.x)<2.e-12,"split differs from selected reference");
        if(!y.suppressed) {
          require(y.children==2,"wrong secondary count");
          auto energy=y.energy_sum_GeV-2*(em::transportMassGeV(11)-em::ElectronMassGeV);
          require(std::abs(energy-E/1000.)/(E/1000.)<2.e-12 && y.norm_error<2.e-12,"energy/direction invariant failed");
        } else require(y.children==0,"LPM-suppressed event emitted children");
      }
      checked+=n;
    }
  }
  auto c=components.front().GetHash();
  std::vector<test::PhotonPairFallbackInput> guards{
    {100.,1.e-3,c,9000001,0,1}, {100.,1.e-3,c,9000002,64,2},
    {100.,1.e-3,0,9000003,64}, {1.e15,1.e-3,c,9000004,0},
    {1.04,1.e-3,c,9000005,0}, {1.,1.e-3,c,9000006,64},
    {100.,-1.,c,9000007,0}, {NAN,1.e-3,c,9000008,0}};
  for(auto const& y:test::runPhotonPairFallbackTest(tables,aux,guards))
    require(y.fallback && y.children==0,"invalid input did not fail closed");
  require(fallbacks>0,"CDF branch was never exercised");
  std::cout<<"PASS samples="<<checked<<" cdf="<<fallbacks<<" max_attempts="<<maximum_attempts
           <<" normal_exhaustions="<<normal_exhaustions
           <<" max_cdf_error="<<maximum_cdf_error<<" max_rejection_error="<<maximum_rejection_error
           <<" guards="<<guards.size()<<std::endl;
  return 0;
}catch(std::exception const& e){std::cerr<<e.what()<<std::endl;return 1;}}

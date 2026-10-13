/* (c) Copyright 2026 CORSIKA Project; BSD-3-Clause. */
#include "PhotonPairFallbackDriver.hpp"
#include <corsika/accelerator/em/common/Philox.hpp>
#include <corsika/accelerator/em/common/ProposalFallbackAdapter.hpp>
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
  std::uint64_t checked=0;
  double maximum_cdf_error=0.;
  for(auto const& component:components)for(double E:{1.1,2.,3.,5.,10.,1.e3,1.e7,1.e8,1.e11}) {
    auto v=nt::queryProposalNativeLossFraction(host,22,em::PhotonPairProcessId,component.GetHash(),E,.5);
    require(v.status==nt::NativeQueryStatus::Success && v.value==1.,"public photon loss must remain v=1");
    std::vector<test::PhotonPairFallbackInput> inputs;
    for(unsigned i=0;i<n;++i)inputs.push_back({E,1.e-3,component.GetHash(),checked+i+1,0});
    auto output=test::runPhotonPairFallbackTest(tables,aux,inputs);
    for(unsigned i=0;i<n;++i) {
      auto q=inputs[i]; auto y=output[i];
      require(!y.fallback && !y.error,"valid photon CDF split failed");
      require(std::isfinite(y.x) && y.x>=em::ElectronMassMeV/E && y.x<=1-em::ElectronMassMeV/E,"unphysical split");
      em::RandomNumberKey key{26101117,71,q.history,0,unsigned(em::PhotonPairProcessId),domains::PhotonPairSplitDrawId};
      auto u=em::uniformOpen01(key);
      auto expected=native->CalculateRho((E/1000.)*1000.,u,component);
      maximum_cdf_error=std::max(maximum_cdf_error,std::abs(expected-y.x));
      require(y.draw==domains::PhotonPairSplitDrawId && y.uniform==u,"photon did not use CDF quantile directly");
      require(std::abs(expected-y.x)<2.e-12,"split differs from native CDF inverse");
      if(!y.suppressed) {
        require(y.children==2,"wrong secondary count");
        auto energy=y.energy_sum_GeV-2*(em::transportMassGeV(11)-em::ElectronMassGeV);
        require(std::abs(energy-E/1000.)/(E/1000.)<2.e-12 && y.norm_error<2.e-12,"energy/direction invariant failed");
      } else require(y.children==0,"LPM-suppressed event emitted children");
    }
    checked+=n;
  }
  auto c=components.front().GetHash();
  std::vector<test::PhotonPairFallbackInput> guards{
    {100.,1.e-3,c,9000001,1}, {100.,1.e-3,c,9000002,2},
    {100.,1.e-3,0,9000003,0}, {1.e15,1.e-3,c,9000004,0},
    {1.04,1.e-3,c,9000005,0}, {1.,1.e-3,c,9000006,0},
    {100.,-1.,c,9000007,0}, {NAN,1.e-3,c,9000008,0}};
  for(auto const& y:test::runPhotonPairFallbackTest(tables,aux,guards))
    require(y.fallback && y.children==0 && y.vertex_reached==1,"invalid input did not preserve fallback vertex");

  std::vector<em::EmInteractionRecord> direct;
  for(int pid:{11,-11,13,-13})for(double E:{1000.,1.e6}) {
    em::EmInteractionRecord r{};
    r.status=em::EmInteractionStatus::Selected;r.interaction_vertex_reached=1;
    r.process_id=em::ElectronPairProcessId;r.component_hash=c;r.energy_fraction=.1;
    r.particle.pid=pid;r.particle.energy_GeV=E/1000.;r.particle.weight=1.;
    r.particle.direction[2]=1.;r.particle.history_id=direct.size()+9001000;
    r.particle.position_m[2]=5000.;r.particle.time_s=2.e-5;
    direct.push_back(r);
  }
  auto failure=direct[2]; // Actual low-energy positron envelope counterexample.
  failure.particle.pid=-11;failure.particle.energy_GeV=.096100347063277691;
  failure.energy_fraction=.077053200608839439;failure.particle.history_id=9010000;
  direct.push_back(failure);
  auto routed=test::runDirectPairCpuRoutingTest(direct);
  for(std::size_t i=0;i<direct.size();++i) {
    auto const& r=direct[i];auto const& y=routed[i];auto ev=y.event;
    require(y.fallback && !y.children && !y.rejection_trials && !y.record,"direct pair ran device sampling/output");
    require(ev.reason==em::ProposalFallbackReason::CpuOnlyProcess,"wrong direct pair routing reason");
    require(ev.interaction_vertex_reached==1 && ev.energy_fraction==r.energy_fraction &&
            ev.process_id==r.process_id && ev.component_hash==r.component_hash &&
            ev.particle.energy_GeV==r.particle.energy_GeV &&
            ev.particle.position_m[2]==r.particle.position_m[2] &&
            ev.particle.time_s==r.particle.time_s,"CPU routing changed selected vertex/loss");
    ev.medium_hash=medium.GetHash();ev.interaction_hash=1;
    require(em::permitsSpecifiedProposalCpuFinalState(ev),"CPU adapter rejected direct pair");
    auto before=ev;before.interaction_vertex_reached=0;
    require(!em::permitsSpecifiedProposalCpuFinalState(before),"CPU accepted pre-transport fallback");
    auto record=em::makeProposalInteractionRecord(ev,26101117,71);
    auto generator=PROPOSAL::DefaultFactory<PROPOSAL::secondaries::Parametrization>::Create(
        PROPOSAL::InteractionType::Epair,proposal::particle.at(record.context.projectile_id),medium);
    PROPOSAL::StochasticLoss loss(em::ElectronPairProcessId,
        record.v_loss*record.context.projectile_energy_MeV,
        PROPOSAL::Cartesian3D(0.,0.,record.context.position_cm[2]),
        PROPOSAL::Cartesian3D(0.,0.,1.),record.context.time_s,0.,
        record.context.projectile_energy_MeV,record.component_hash);
    auto rnd=em::makeProposalFinalStateRandomNumbers(ev,generator->RequiredRandomNumbers(),26101117,71);
    auto children=generator->CalculateSecondaries(loss,components.front(),rnd);
    require(children.size()==3,"CPU direct pair did not produce three particles");
    double sum=0.;
    for(auto const& child:children) {
      require(std::isfinite(child.energy) && child.energy>0.,"nonfinite CPU secondary");
      sum+=child.energy;
    }
    require(std::abs(sum-loss.parent_particle_energy)<1.e-9*sum,"CPU direct pair energy sum failed");
  }
  std::cout<<"PASS native_CDF_samples="<<checked<<" max_CDF_absolute_error="<<maximum_cdf_error
           <<" guards="<<guards.size()<<" direct_pair_CPU_cases="<<direct.size()
           <<" rejection_trials=0"<<std::endl;
  return 0;
}catch(std::exception const& e){std::cerr<<e.what()<<std::endl;return 1;}}

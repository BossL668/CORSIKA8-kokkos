// Selected-process controls: these tests do not estimate a natural mountain rate.
#include <corsika/modules/neutrino/MountainNeutrinoInteraction.hpp>
#include <corsika/modules/neutrino/PrescribedTauDecay.hpp>
#include <corsika/modules/neutrino/TransportedLeptons.hpp>
#include <corsika/setup/SetupStack.hpp>
#include <corsika/setup/SetupC7trackedParticles.hpp>
#include <corsika/modules/terrain/TerrainAtmosphere.hpp>
#include <corsika/framework/geometry/RootCoordinateSystem.hpp>
#include <catch2/catch_all.hpp>
#include <cstdlib>
#include <fstream>
#include <random>

using namespace corsika;
using namespace corsika::units::si;
using Catch::Approx;
using TestStack = setup::HybridStack<corsika::terrain::Environment>;

TEST_CASE("CTW competing CC and NC rates and million-draw selection", "[mountain][nc]") {
  using namespace neutrino;
  std::mt19937_64 rng(90019);
  std::uniform_real_distribution<double> uniform(0.,1.);
  std::ofstream csv;
  if (auto path=std::getenv("C8_NC_RATES_CSV")) {
    csv.open(path);REQUIRE(csv.good());
    csv<<"pdg,energy_GeV,cc_cm2,nc_cm2,nc_probability,nc_count,draws\n";
  }
  for (Code pid : {Code::NuTau,Code::NuTauBar}) for (double energy : {1.e4,1.e6,1.e9}) {
    double cc=ctw2011CrossSectionCm2(pid,energy,true);
    double nc=ctw2011CrossSectionCm2(pid,energy,false);
    REQUIRE(nc>0.); REQUIRE(nc<cc);
    double p=nc/(nc+cc);
    std::size_t count=0;
    constexpr std::size_t n=1000000;
    for (std::size_t i=0;i<n;++i) count+=selectWeakCurrent(cc,nc,uniform(rng))==WeakCurrent::NC;
    CHECK(std::abs(double(count)/n-p)<6.*std::sqrt(p*(1.-p)/n));
    if(csv) csv<<static_cast<int>(get_PDG(pid))<<','<<energy<<','<<cc<<','<<nc<<','<<p<<','<<count<<','<<n<<'\n';
    CHECK(selectWeakCurrent(cc,0.,.9)==WeakCurrent::CC);
    CHECK(selectWeakCurrent(0.,nc,.0)==WeakCurrent::NC);
  }
  CHECK_THROWS(selectWeakCurrent(0.,0.,.5));
  CHECK_THROWS(selectWeakCurrent(1.,1.,1.));
  CHECK_THROWS(selectWeakCurrent(-1.,2.,.5));
  auto cs=get_root_CoordinateSystem();
  FourMomentum beam{1.e6_GeV,MomentumVector(cs,{0_GeV,0_GeV,1.e6_GeV})};
  FourMomentum target{get_mass(Code::Oxygen),MomentumVector(cs,{0_GeV,0_GeV,0_GeV})};
  RNGManager<>::getInstance().registerRandomStream("pythia");
  MountainNeutrinoInteraction both(setup::C7trackedParticles,SamplingMode::NaturalCC,InteractionChannels::CCAndNC);
  MountainNeutrinoInteraction cc(setup::C7trackedParticles,SamplingMode::NaturalCC,InteractionChannels::CCOnly);
  MountainNeutrinoInteraction nc(setup::C7trackedParticles,SamplingMode::NaturalCC,InteractionChannels::NCOnly);
  CHECK(both.getCrossSection(Code::NuTau,Code::Oxygen,beam,target)/
        (cc.getCrossSection(Code::NuTau,Code::Oxygen,beam,target)+nc.getCrossSection(Code::NuTau,Code::Oxygen,beam,target))==Approx(1.));
}

TEST_CASE("NC retains same flavor and conserves charge and four-momentum", "[mountain][nc]") {
  auto& rng=RNGManager<>::getInstance();rng.registerRandomStream("pythia");rng.setSeed(909310);
  auto cs=get_root_CoordinateSystem();
  neutrino::AuditedPythiaNeutrinoFinalState gen(neutrino::withTransportedTaus(setup::C7trackedParticles));
  for (int pdg : {12,-12,14,-14,16,-16}) for (Code target : {Code::Proton,Code::Neutron,Code::Oxygen}) {
    CAPTURE(pdg,target);
    auto event=gen.generate(convert_from_PDG(static_cast<PDGCode>(pdg)),target,
        FourMomentum{1.e5_GeV,MomentumVector(cs,{0_GeV,0_GeV,1.e5_GeV})},neutrino::WeakCurrent::NC);
    CHECK(event.audit.current==neutrino::WeakCurrent::NC);
    CHECK(event.audit.selectedOutgoingLeptonPdg==pdg);
    CHECK(event.audit.finalStatePdgCounts[pdg]>=1);
    CHECK(event.audit.finalChargeE==event.audit.initialChargeE);
    CHECK(event.audit.corsikaMaxComponentRelativeResidual<1.e-4);
    CHECK(event.audit.inelasticityY>0.);CHECK(event.audit.inelasticityY<1.);
    CHECK(event.audit.q2GeV2>=25.*(1.-1.e-8));
  }
}

TEST_CASE("Prescribed tau polarization pion angular oracle both charges", "[mountain][polarization]") {
  logging::set_level(logging::level::warn);
  auto& rng=RNGManager<>::getInstance();rng.registerRandomStream("pythia");rng.setSeed(909320);
  auto cs=get_root_CoordinateSystem();
  std::ofstream csv;
  if(auto path=std::getenv("C8_TAU_POLARIZATION_CSV")) {
    csv.open(path);REQUIRE(csv.good());csv<<"pdg,tau_minus_P,cos_theta_pi,nu_energy_fraction\n";
  }
  for(double pol : {-1.,0.,1.}) {
    corsika::pythia8::Decay decay;
    neutrino::configurePrescribedTauDecay(decay,pol,neutrino::TauDecayChannels::PionNeutrinoControl);
    for(Code pid : {Code::TauMinus,Code::TauPlus}) {
      CAPTURE(pid,pol);
      double mean=0.,mean2=0.;constexpr int n=10000;
      for(int i=0;i<n;++i) {
        TestStack stack;
        auto parent=stack.addParticle(std::make_tuple(pid,10_GeV-get_mass(pid),
            DirectionVector(cs,{0.,0.,1.}),Point(cs,0_m,0_m,0_m),0_s));
        COMBoost boost(parent.getMomentum(),parent.getMass());
        TestStack::stack_view_type view(parent);
        CHECK(decay.getLifetime(parent)/get_lifetime(pid)==Approx(parent.getEnergy()/parent.getMass()));
        decay.doDecay(view);
        REQUIRE(view.getSize()==2);
        double cosine=2.,nuFraction=0.;HEPEnergyType sum=0_GeV;
        for(auto const& child : view) {
          sum+=child.getEnergy();
          if(std::abs(static_cast<int>(get_PDG(child.getPID())))==211) {
            auto rest=boost.toCoM(FourMomentum{child.getEnergy(),child.getMomentum()});
            auto p=rest.getSpaceLikeComponents();
            cosine=p.getComponents(boost.getRotatedCS())[2]/p.getNorm();
          } else {
            CHECK(child.getPID()==(pid==Code::TauMinus?Code::NuTau:Code::NuTauBar));
            nuFraction=child.getEnergy()/parent.getEnergy();
          }
        }
        REQUIRE(cosine>=-1.);REQUIRE(cosine<=1.);
        CHECK(std::abs((sum-parent.getEnergy())/parent.getEnergy())<1.e-4);
        mean+=cosine;mean2+=cosine*cosine;
        if(csv) csv<<static_cast<int>(get_PDG(pid))<<','<<pol<<','<<cosine<<','<<nuFraction<<'\n';
      }
      // dP/dcos(theta_pi)=(1+P_tau- cos(theta_pi))/2. CP flips
      // both physical tau+ helicity and pion analyzing power, not this slope.
      CHECK(mean/n==Approx(pol/3.).margin(6.*std::sqrt((1./3.-pol*pol/9.)/n)));
      CHECK(mean2/n==Approx(1./3.).margin(.025));
    }
  }
}

TEST_CASE("Actual SecondaryView CC tau decay NC CC regeneration chain", "[mountain][regeneration]") {
  // Conditional vertices exercise reinsertion/history/energy transfer, NOT
  // probabilities. No energy reset, forced decay time or density enhancement.
  auto& rng=RNGManager<>::getInstance();rng.registerRandomStream("pythia");rng.setSeed(909330);
  auto cs=get_root_CoordinateSystem();
  auto stable=neutrino::withTransportedTaus(setup::C7trackedParticles);
  std::ofstream csv;
  if (auto path=std::getenv("C8_REGENERATION_CSV")) {
    csv.open(path);REQUIRE(csv.good());
    csv<<"primary_pdg,cycle,stage,history_id,parent_id,energy_GeV\n";
  }
  neutrino::MountainNeutrinoInteraction cc(stable,neutrino::SamplingMode::NaturalCC);
  neutrino::MountainNeutrinoInteraction nc(stable,neutrino::SamplingMode::NaturalCC,neutrino::InteractionChannels::NCOnly);
  corsika::pythia8::Decay decay;
  neutrino::configurePrescribedTauDecay(decay,-1.);
  FourMomentum target{get_mass(Code::Oxygen),MomentumVector(cs,{0_GeV,0_GeV,0_GeV})};
  for(Code primary : {Code::NuTau,Code::NuTauBar}) {
    TestStack stack;
    auto active=stack.addParticle(std::make_tuple(primary,1.e8_GeV,
        DirectionVector(cs,{0.,0.,1.}),Point(cs,0_m,0_m,0_m),0_s));
    for(int cycle=0;cycle<2;++cycle) {
      CAPTURE(primary,cycle);
      auto neutrinoId=active.getHistoryId();
      auto before=active.getEnergy();
      auto record=[&](char const* stage,auto const& p) {
        if(csv) csv<<static_cast<int>(get_PDG(primary))<<','<<cycle<<','<<stage<<','
            <<p.getHistoryId()<<','<<p.getParentHistoryId()<<','<<p.getEnergy()/1_GeV<<'\n';
      };
      record("incoming_nu",active);
      TestStack::stack_view_type ccView(active);
      cc.doInteraction(ccView,active.getPID(),Code::Oxygen,
          FourMomentum{active.getEnergy(),active.getMomentum()},target);
      auto tau=ccView.begin();
      while(tau!=ccView.end() && tau.getPID()!=(primary==Code::NuTau?Code::TauMinus:Code::TauPlus)) ++tau;
      REQUIRE(tau!=ccView.end());
      REQUIRE(tau.getParentHistoryId()==neutrinoId);
      // View iterators refer to the underlying stack through getIndexFromIterator.
      auto tauStack=stack.begin();
      while(tauStack!=stack.end() && tauStack.getHistoryId()!=tau.getHistoryId()) ++tauStack;
      REQUIRE(tauStack!=stack.end());
      auto tauId=tauStack.getHistoryId();
      record("CC_tau",tauStack);
      TestStack::stack_view_type decayView(tauStack);decay.doDecay(decayView);
      auto regenerated=decayView.begin();
      while(regenerated!=decayView.end() && regenerated.getPID()!=primary) ++regenerated;
      REQUIRE(regenerated!=decayView.end());
      REQUIRE(regenerated.getParentHistoryId()==tauId);
      REQUIRE(regenerated.getEnergy()<before);
      REQUIRE(regenerated.getEnergy()>=1.e4_GeV);
      record("decay_nu",regenerated);
      auto next=stack.begin();
      while(next!=stack.end() && next.getHistoryId()!=regenerated.getHistoryId()) ++next;
      REQUIRE(next!=stack.end());
      REQUIRE(cc.getCrossSection(primary,Code::Oxygen,FourMomentum{next.getEnergy(),next.getMomentum()},target)>0_nb);
      TestStack::stack_view_type ncView(next);
      nc.doInteraction(ncView,primary,Code::Oxygen,FourMomentum{next.getEnergy(),next.getMomentum()},target);
      auto outgoing=ncView.begin();
      while(outgoing!=ncView.end() && outgoing.getPID()!=primary) ++outgoing;
      REQUIRE(outgoing!=ncView.end());
      REQUIRE(outgoing.getEnergy()<next.getEnergy());
      active=stack.begin();
      while(active!=stack.end() && active.getHistoryId()!=outgoing.getHistoryId()) ++active;
      REQUIRE(active!=stack.end());
      record("NC_nu",active);
    }
  }
  CHECK(cc.records().size()==4);CHECK(nc.records().size()==4);
}

TEST_CASE("All tau decay channels preserve flavor charge energy and branching", "[mountain][polarization][regeneration]") {
  logging::set_level(logging::level::warn);
  auto& rng=RNGManager<>::getInstance();rng.registerRandomStream("pythia");rng.setSeed(909350);
  auto cs=get_root_CoordinateSystem();
  std::ofstream csv;
  if(auto path=std::getenv("C8_TAU_BRANCHING_CSV")) {
    csv.open(path);REQUIRE(csv.good());
    csv<<"pdg,tau_minus_P,n,expected_e_BR,observed_e_BR,expected_mu_BR,observed_mu_BR,max_energy_residual,max_momentum_residual\n";
  }
  for(double pol : {-1.,0.,1.}) {
    corsika::pythia8::Decay decay;neutrino::configurePrescribedTauDecay(decay,pol);
    auto entry=decay.particleData.particleDataEntryPtr(15);
    double total=0.,brE=0.,brMu=0.;
    for(int j=0;j<entry->sizeChannels();++j) {
      auto const& channel=entry->channel(j);
      if(!channel.onMode()) continue;
      total+=channel.bRatio();
      for(int k=0;k<channel.multiplicity();++k) {
        if(std::abs(channel.product(k))==11) brE+=channel.bRatio();
        if(std::abs(channel.product(k))==13) brMu+=channel.bRatio();
      }
    }
    brE/=total;brMu/=total;
    REQUIRE(brE>.1);REQUIRE(brMu>.1);
    for(Code pid : {Code::TauMinus,Code::TauPlus}) {
      CAPTURE(pid,pol);
      constexpr int n=5000;int electrons=0,muons=0;double maxE=0.,maxP=0.;
      for(int i=0;i<n;++i) {
        TestStack stack;
        auto parent=stack.addParticle(std::make_tuple(pid,100_GeV-get_mass(pid),
            DirectionVector(cs,{1./3.,2./3.,2./3.}),Point(cs,0_m,0_m,0_m),0_s));
        TestStack::stack_view_type view(parent);decay.doDecay(view);
        bool electron=false,muon=false;
        for(int index:decay.event[1].daughterList()) {
          electron|=std::abs(decay.event[index].id())==11;
          muon|=std::abs(decay.event[index].id())==13;
        }
        electrons+=electron;muons+=muon;
        int tauFlavor=0,charge=0;HEPEnergyType sumE=0_GeV;
        MomentumVector sumP(cs,{0_GeV,0_GeV,0_GeV});
        for(auto const& child:view) {
          sumE+=child.getEnergy();sumP+=child.getMomentum();charge+=get_charge_number(child.getPID());
          tauFlavor+=child.getPID()==Code::NuTau;tauFlavor-=child.getPID()==Code::NuTauBar;
        }
        CHECK(charge==get_charge_number(pid));
        CHECK(tauFlavor==(pid==Code::TauMinus?1:-1));
        maxE=std::max(maxE,std::abs((sumE-parent.getEnergy())/parent.getEnergy()));
        maxP=std::max(maxP,(sumP-parent.getMomentum()).getNorm()/parent.getEnergy());
      }
      CHECK(maxE<1.e-3);CHECK(maxP<1.e-3);
      CHECK(double(electrons)/n==Approx(brE).margin(6.*std::sqrt(brE*(1.-brE)/n)));
      CHECK(double(muons)/n==Approx(brMu).margin(6.*std::sqrt(brMu*(1.-brMu)/n)));
      if(csv) csv<<static_cast<int>(get_PDG(pid))<<','<<pol<<','<<n<<','<<brE<<','<<double(electrons)/n<<','<<brMu<<','<<double(muons)/n<<','<<maxE<<','<<maxP<<'\n';
    }
  }
}

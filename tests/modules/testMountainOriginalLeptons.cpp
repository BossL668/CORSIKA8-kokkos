// Compare the mountain assembly with actual original CORSIKA modules.
// This does not certify a low-energy weak model or event-derived CC spin.
#include <corsika/modules/neutrino/TransportedLeptonDecay.hpp>
#include <corsika/modules/neutrino/MountainNeutrinoInteraction.hpp>
#include <corsika/modules/neutrino/TransportedLeptons.hpp>
#include <corsika/modules/pythia8/NeutrinoInteraction.hpp>
#include <corsika/modules/terrain/TerrainAtmosphere.hpp>
#include <corsika/setup/SetupStack.hpp>
#include <corsika/setup/SetupC7trackedParticles.hpp>
#include <corsika/framework/geometry/RootCoordinateSystem.hpp>
#include <catch2/catch_all.hpp>
#include <array>
#include <cstdlib>
#include <fstream>

using namespace corsika;
using namespace corsika::units::si;
using Catch::Approx;
using LeptonTestStack = setup::HybridStack<terrain::Environment>;
extern "C" void rmarut_(int*,int*,int*);

namespace {
void initStreams() {
  logging::set_level(logging::level::warn);
  auto& r = RNGManager<>::getInstance();
  r.registerRandomStream("pythia");
  r.registerRandomStream("tauola");
}
std::array<int,3> ranmarState() {
  std::array<int,3> result{};rmarut_(&result[0],&result[1],&result[2]);return result;
}
void restoreRanmar(std::array<int,3> state) {
  Tauolapp::rmarin_(&state[0],&state[1],&state[2]);
}
auto makeParent(LeptonTestStack& stack, Code pid, double energy, DirectionVector direction) {
  auto cs = direction.getCoordinateSystem();
  return stack.addParticle(std::make_tuple(pid, energy*1_GeV-get_mass(pid),
      direction, Point(cs,1_m,2_m,3_m),7_ns));
}
void compareViews(LeptonTestStack::stack_view_type const& a, LeptonTestStack::stack_view_type const& b) {
  REQUIRE(a.getSize() == b.getSize());
  auto p=a.begin(), q=b.begin();
  for (; p!=a.end(); ++p, ++q) {
    CHECK(p.getPID()==q.getPID());
    CHECK(p.getEnergy()==q.getEnergy());
    CHECK((p.getMomentum()-q.getMomentum()).getNorm()==0_GeV);
    CHECK(p.getTime()==q.getTime());
    CHECK(p.getHistoryId()==q.getHistoryId());
    CHECK(p.getParentHistoryId()==q.getParentHistoryId());
    CHECK(p.getWeight()==q.getWeight());
  }
}
} // namespace

TEST_CASE("Mountain TAUOLA mass scope restores global state, including exceptions",
          "[mountain][original][tauola][mass-scope]") {
  initStreams();
  neutrino::TransportedLeptonDecay decay;
  auto snapshot=[] {
    auto const& m=Tauolapp::parmas_;
    return std::array<float,10>{m.amtau,m.amnuta,m.amell,m.amnue,m.ammu,
        m.amnumu,m.ampiz,m.ampi,m.amk,m.amkz};
  };
  auto before=snapshot();
  {
    neutrino::ScopedTauolaMassConvention outer;
    CHECK(Tauolapp::parmas_.amnuta==0.f);
    CHECK(Tauolapp::parmas_.amtau==static_cast<float>(get_mass(Code::TauMinus)/1_GeV));
    auto aligned=snapshot();
    {neutrino::ScopedTauolaMassConvention inner;CHECK(snapshot()==aligned);}
    CHECK(snapshot()==aligned);
  }
  CHECK(snapshot()==before);
  CHECK_THROWS_AS([&] {
    neutrino::ScopedTauolaMassConvention scope;
    throw std::runtime_error("diagnostic exception");
  }(),std::runtime_error);
  CHECK(snapshot()==before);
}

TEST_CASE("Original neutrino cross section is a forced-vertex helper, not attenuation",
          "[mountain][original]") {
  initStreams();
  auto cs=get_root_CoordinateSystem();
  pythia8::NeutrinoInteraction original(setup::C7trackedParticles,true,true);
  neutrino::MountainNeutrinoInteraction physical(setup::C7trackedParticles,
      neutrino::SamplingMode::NaturalCC,neutrino::InteractionChannels::CCAndNC);
  for (Code pid : {Code::NuE,Code::NuMuBar,Code::NuTau,Code::NuTauBar})
    for (double e : {1.e4,1.e8}) {
      FourMomentum beam{e*1_GeV,MomentumVector(cs,{0_GeV,0_GeV,e*1_GeV})};
      FourMomentum target{get_mass(Code::Oxygen),MomentumVector(cs,{0_GeV,0_GeV,0_GeV})};
      CHECK(original.getCrossSection(pid,Code::Oxygen,beam,target)==4_nb);
      double sigma = neutrino::ctw2011CrossSectionCm2(pid,e,true)
                   + neutrino::ctw2011CrossSectionCm2(pid,e,false);
      CHECK(physical.getCrossSection(pid,Code::Oxygen,beam,target)/1_nb==Approx(16.*sigma/1.e-33));
    }
}

TEST_CASE("Missing low-energy neutrino physics is counted once or explicitly rejected",
          "[mountain][original][domain]") {
  neutrino::NeutrinoModelDomain audit;
  audit.observe(Code::NuTau,9999.,2.,41);
  audit.observe(Code::NuTau,9999.,2.,41); // a rock/air boundary, not a new particle
  audit.observe(Code::NuTauBar,1.,1.,42);
  audit.observe(Code::NuE,1.e4,1.,43);
  audit.observe(Code::Electron,1.,1.,44);
  CHECK(audit.count()==2);
  CHECK(audit.weightedEnergyGeV()==19999.);
  CHECK(audit.species().at(16).count==1);
  neutrino::NeutrinoModelDomain strict(true);
  CHECK_THROWS_AS(strict.check(Code::NuTau,9999.),std::runtime_error);
  CHECK_THROWS_AS(strict.check(Code::NuTau,1.01e12),std::runtime_error);
  CHECK_NOTHROW(strict.check(Code::NuTau,1.e4));
  CHECK_NOTHROW(strict.check(Code::NuTau,1.e12));
  CHECK_THROWS(audit.observe(Code::NuTau,-1.,1.,50));
  CHECK_THROWS(audit.observe(Code::NuTau,1.,-1.,50));
}

TEST_CASE("Mass-aligned mountain tau matches mass-aligned original TAUOLA daughters and draws",
          "[mountain][original][tauola]") {
  initStreams();
  auto& rng=RNGManager<>::getInstance(); rng.setSeed(909401);
  auto cs=get_root_CoordinateSystem();
  for (auto h : {tauola::Helicity::LeftHanded,tauola::Helicity::Unpolarized,
                  tauola::Helicity::RightHanded}) {
    neutrino::TransportedLeptonDecayConfig config; config.helicity=h;
    neutrino::TransportedLeptonDecay adapter(config);
    tauola::Decay original(h);
    for (Code pid : {Code::TauMinus,Code::TauPlus})
      for (double e : {10.,100.,900.})
        for (auto axis : {std::array<double,3>{0.,0.,1.}, {1.,0.,0.}, {.36,.48,-.8}})
          for (int sample=0; sample<10; ++sample) {
            CAPTURE(pid,e,axis,sample,h);
            DirectionVector d(cs,{axis[0],axis[1],axis[2]});
            LeptonTestStack a,b;auto pa=makeParent(a,pid,e,d), pb=makeParent(b,pid,e,d);
            CHECK(adapter.getLifetime(pa)==original.getLifetime(pb));
            CHECK(adapter.getLifetime(pa)==get_lifetime(pid)*(pa.getEnergy()/pa.getMass()));
            LeptonTestStack::stack_view_type va(pa),vb(pb);
            auto saved=rng.getRandomStream("tauola");
            auto fortranBefore=ranmarState();
            adapter.doDecay(va);
            auto after=rng.getRandomStream("tauola");
            auto fortranAfter=ranmarState();
            rng.getRandomStream("tauola")=saved;
            restoreRanmar(fortranBefore);
            {
              neutrino::ScopedTauolaMassConvention masses;
              original.doDecay(vb);
            }
            CHECK(ranmarState()==fortranAfter);
            std::stringstream expected,actual;
            expected<<after;actual<<rng.getRandomStream("tauola");
            CHECK(expected.str()==actual.str());
            compareViews(va,vb);
            int charge=0, nNu=0;HEPEnergyType sum=0_GeV;
            MomentumVector psum(cs,{0_GeV,0_GeV,0_GeV});
            for (auto const& daughter : va) {
              charge+=get_charge_number(daughter.getPID());sum+=daughter.getEnergy();
              psum+=daughter.getMomentum();
              nNu+=daughter.getPID()==(pid==Code::TauMinus?Code::NuTau:Code::NuTauBar);
            }
            CHECK(charge==get_charge_number(pid)); CHECK(nNu==1);
            CHECK(std::abs((sum-pa.getEnergy())/pa.getEnergy())<2.e-6);
            CHECK((psum-pa.getMomentum()).getNorm()/pa.getEnergy()<2.e-6);
          }
  }
}

TEST_CASE("Conditioned TAUOLA remains finite and conserves four-momentum at extreme energy",
          "[mountain][original][tauola]") {
  initStreams();auto& rng=RNGManager<>::getInstance();rng.setSeed(909405);
  auto cs=get_root_CoordinateSystem();
  neutrino::TransportedLeptonDecay decay;
  std::ofstream energyCsv;
  if(auto path=std::getenv("C8_TAUOLA_ENERGY_CSV")) {
    energyCsv.open(path);REQUIRE(energyCsv.good());energyCsv.precision(17);
    energyCsv<<"pdg,energy_GeV,sample,energy_relative_residual,momentum_relative_residual\n";
  }
  for(Code pid : {Code::TauMinus,Code::TauPlus}) for(double e : {1.e4,1.e8,1.e12})
    for(auto axis : {std::array<double,3>{0.,0.,1.}, {1.,0.,0.}, {.36,.48,-.8}})
      for(int i=0;i<100;++i) {
        CAPTURE(pid,e,axis,i);
        LeptonTestStack stack;auto parent=makeParent(stack,pid,e,DirectionVector(cs,{axis[0],axis[1],axis[2]}));
        LeptonTestStack::stack_view_type view(parent);decay.doDecay(view);
        int charge=0,nNu=0;HEPEnergyType sum=0_GeV;
        MomentumVector psum(cs,{0_GeV,0_GeV,0_GeV});
        for(auto const& child:view) {
          REQUIRE(std::isfinite(child.getEnergy()/1_GeV));REQUIRE(child.getEnergy()>0_GeV);
          charge+=get_charge_number(child.getPID());sum+=child.getEnergy();psum+=child.getMomentum();
          nNu+=child.getPID()==(pid==Code::TauMinus?Code::NuTau:Code::NuTauBar);
        }
        CHECK(charge==get_charge_number(pid));CHECK(nNu==1);
        if(energyCsv)energyCsv<<static_cast<int>(get_PDG(pid))<<','<<e<<','<<i<<','
          <<(sum-parent.getEnergy())/parent.getEnergy()<<','
          <<(psum-parent.getMomentum()).getNorm()/parent.getEnergy()<<'\n';
        CHECK(std::abs((sum-parent.getEnergy())/parent.getEnergy())<2.e-6);
        CHECK((psum-parent.getMomentum()).getNorm()/parent.getEnergy()<2.e-6);
      }
}

TEST_CASE("Non-tau Pythia and explicit legacy tau path retain the stock random stream",
          "[mountain][original]") {
  initStreams();
  auto& rng=RNGManager<>::getInstance();rng.setSeed(909402);
  auto cs=get_root_CoordinateSystem();
  for (auto model : {neutrino::TauDecayModel::Tauola,neutrino::TauDecayModel::Pythia}) {
    neutrino::TransportedLeptonDecayConfig c; c.model=model;
    neutrino::TransportedLeptonDecay adapter(c);pythia8::Decay original;
    for (Code pid : {Code::MuMinus,Code::MuPlus,Code::PiPlus,Code::TauMinus,Code::TauPlus}) {
      if (model==neutrino::TauDecayModel::Tauola && neutrino::TransportedLeptonDecay::isTau(pid)) continue;
      for (int i=0;i<20;++i) {
        LeptonTestStack a,b;DirectionVector d(cs,{.36,.48,-.8});
        auto pa=makeParent(a,pid,10.,d), pb=makeParent(b,pid,10.,d);
        CHECK(adapter.getLifetime(pa)==original.getLifetime(pb));
        LeptonTestStack::stack_view_type va(pa),vb(pb);
        auto saved=rng.getRandomStream("pythia");
        adapter.doDecay(va);auto after=rng.getRandomStream("pythia");
        rng.getRandomStream("pythia")=saved;original.doDecay(vb);
        compareViews(va,vb);
        std::stringstream expected,actual;expected<<after;actual<<rng.getRandomStream("pythia");
        CHECK(expected.str()==actual.str());
      }
    }
  }
  neutrino::TransportedLeptonDecayConfig invalid;
  invalid.prescribedPythiaPolarization=-1.;
  CHECK_THROWS_AS(neutrino::TransportedLeptonDecay(invalid),std::invalid_argument);
}

TEST_CASE("CC TAUOLA NC regeneration uses original decays and retains neutrino histories",
          "[mountain][original][regeneration]") {
  initStreams();
  auto& rng=RNGManager<>::getInstance();rng.setSeed(909403);
  auto cs=get_root_CoordinateSystem();
  auto stable=neutrino::withTransportedTaus(setup::C7trackedParticles);
  neutrino::MountainNeutrinoInteraction cc(stable,neutrino::SamplingMode::NaturalCC);
  neutrino::MountainNeutrinoInteraction nc(stable,neutrino::SamplingMode::NaturalCC,
      neutrino::InteractionChannels::NCOnly);
  neutrino::TransportedLeptonDecay decay;
  FourMomentum target{get_mass(Code::Oxygen),MomentumVector(cs,{0_GeV,0_GeV,0_GeV})};
  std::ofstream csv;
  if (auto path=std::getenv("C8_ORIGINAL_REGENERATION_CSV")) {
    csv.open(path);REQUIRE(csv.good());csv<<"primary_pdg,cycle,stage,history,parent,energy_GeV\n";
  }
  for (Code pid : {Code::NuTau,Code::NuTauBar}) {
    LeptonTestStack stack;auto active=makeParent(stack,pid,1.e9,DirectionVector(cs,{.36,.48,-.8}));
    auto record=[&](int cycle,char const* stage,auto const& p) {
      if(csv) csv<<static_cast<int>(get_PDG(pid))<<','<<cycle<<','<<stage<<','
                 <<p.getHistoryId()<<','<<p.getParentHistoryId()<<','<<p.getEnergy()/1_GeV<<'\n';
    };
    auto byId=[&](std::uint64_t id) {
      auto p=stack.begin();while(p!=stack.end() && p.getHistoryId()!=id) ++p;
      REQUIRE(p!=stack.end());return p;
    };
    for(int cycle=0;cycle<3;++cycle) {
      CAPTURE(pid,cycle);
      auto parentId=active.getHistoryId(); auto before=active.getEnergy();
      REQUIRE(before>=1.e4_GeV);record(cycle,"incoming_nu",active);
      LeptonTestStack::stack_view_type ccView(active);
      cc.doInteraction(ccView,pid,Code::Oxygen,FourMomentum{before,active.getMomentum()},target);
      auto t=ccView.begin();while(t!=ccView.end() && !neutrino::TransportedLeptonDecay::isTau(t.getPID())) ++t;
      REQUIRE(t!=ccView.end());CHECK(t.getParentHistoryId()==parentId);
      auto tau=byId(t.getHistoryId());record(cycle,"CC_tau",tau);
      LeptonTestStack::stack_view_type dv(tau);decay.doDecay(dv);
      auto n=dv.begin();while(n!=dv.end() && n.getPID()!=pid) ++n;
      REQUIRE(n!=dv.end());CHECK(n.getParentHistoryId()==tau.getHistoryId());
      auto regenerated=byId(n.getHistoryId());record(cycle,"decay_nu",regenerated);
      REQUIRE(regenerated.getEnergy()>=1.e4_GeV);
      CHECK(regenerated.getEnergy()<before);
      CHECK(nc.getCrossSection(pid,Code::Oxygen,
          FourMomentum{regenerated.getEnergy(),regenerated.getMomentum()},target)>0_nb);
      LeptonTestStack::stack_view_type nv(regenerated);
      nc.doInteraction(nv,pid,Code::Oxygen,
          FourMomentum{regenerated.getEnergy(),regenerated.getMomentum()},target);
      auto next=nv.begin();while(next!=nv.end() && next.getPID()!=pid) ++next;
      REQUIRE(next!=nv.end());CHECK(next.getParentHistoryId()==regenerated.getHistoryId());
      active=byId(next.getHistoryId());record(cycle,"NC_nu",active);
    }
  }
  CHECK(cc.records().size()==6);CHECK(nc.records().size()==6);
}

TEST_CASE("Original TAUOLA helicity pion angular oracle and conditioned-frame continuity",
          "[.tauolapion][mountain]") {
  // A separate executable invocation is required: TAUOLA's channel selection
  // is global. Never impose this pi-only diagnostic on an application run.
  initStreams();REQUIRE_FALSE(Tauolapp::Tauola::getIsTauolaIni());
  Tauolapp::Tauola::setSameParticleDecayMode(3);
  Tauolapp::Tauola::setOppositeParticleDecayMode(3);
  RNGManager<>::getInstance().setSeed(909410);
  auto cs=get_root_CoordinateSystem();
  std::ofstream csv;
  if(auto path=std::getenv("C8_TAUOLA_PION_CSV")) {
    csv.open(path);REQUIRE(csv.good());csv<<"pdg,helicity,energy_GeV,axis,cos_theta_pi,nu_fraction\n";
  }
  for(auto helicity : {tauola::Helicity::LeftHanded,tauola::Helicity::Unpolarized,tauola::Helicity::RightHanded}) {
    neutrino::TransportedLeptonDecayConfig config;config.helicity=helicity;
    neutrino::TransportedLeptonDecay decay(config);
    double h=helicity==tauola::Helicity::LeftHanded?-1.:(helicity==tauola::Helicity::RightHanded?1.:0.);
    for(Code pid:{Code::TauMinus,Code::TauPlus}) for(double e:{10.,1000.})
      for(int axis=0;axis<2;++axis) {
        DirectionVector d(cs,{0.,0.,1.});
        if(axis) d=DirectionVector(cs,{.36,.48,-.8});
        double sum=0.,sum2=0.;constexpr int n=3000;
        for(int i=0;i<n;++i) {
          LeptonTestStack stack;auto parent=makeParent(stack,pid,e,d);
          COMBoost boost(parent.getMomentum(),parent.getMass());
          LeptonTestStack::stack_view_type view(parent);decay.doDecay(view);
          REQUIRE(view.getSize()==2);
          double cosine=2.,fraction=0.;
          for(auto const& child:view) {
            if(std::abs(static_cast<int>(get_PDG(child.getPID())))==211) {
              auto rest=boost.toCoM(FourMomentum{child.getEnergy(),child.getMomentum()});
              auto p=rest.getSpaceLikeComponents();cosine=p.getComponents(boost.getRotatedCS())[2]/p.getNorm();
            } else {CHECK(child.getPID()==(pid==Code::TauMinus?Code::NuTau:Code::NuTauBar));fraction=child.getEnergy()/parent.getEnergy();}
          }
          REQUIRE(std::isfinite(cosine));sum+=cosine;sum2+=cosine*cosine;
          if(csv)csv<<static_cast<int>(get_PDG(pid))<<','<<h<<','<<e<<','<<axis<<','<<cosine<<','<<fraction<<'\n';
        }
        CAPTURE(pid,e,axis,h,sum/n,sum2/n);
        // For the upstream charge-conjugate helicity convention the two pi
        // analyzing powers yield the same slope (1+h*cos(theta))/2.
        CHECK(sum/n==Approx(h/3.).margin(6.*std::sqrt((1./3.-h*h/9.)/n)));
        CHECK(sum2/n==Approx(1./3.).margin(.04));
      }
  }
}

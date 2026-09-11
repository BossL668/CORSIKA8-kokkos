// TAUOLA angular oracle and unmodified-path regression, not a CC spin oracle.
#include <corsika/modules/neutrino/TransportedLeptonDecay.hpp>
#include <corsika/modules/neutrino/NeutrinoPhysicsCapabilities.hpp>
#include <corsika/modules/terrain/TerrainAtmosphere.hpp>
#include <corsika/setup/SetupStack.hpp>
#include <corsika/framework/geometry/RootCoordinateSystem.hpp>
#include <catch2/catch_all.hpp>
#include <array>
#include <cstdlib>
#include <fstream>
#include <sstream>

using namespace corsika;
using namespace corsika::units::si;
using SpinTestStack = setup::HybridStack<terrain::Environment>;
extern "C" void rmarut_(int*,int*,int*);

namespace {
void streams() {
  logging::set_level(logging::level::warn);
  auto& rng=RNGManager<>::getInstance();
  rng.registerRandomStream("pythia");rng.registerRandomStream("tauola");
}
auto parent(SpinTestStack& stack, Code pid, double energy, DirectionVector d) {
  return stack.addParticle(std::make_tuple(pid,energy*1_GeV-get_mass(pid),d,
      Point(d.getCoordinateSystem(),1_m,2_m,3_m),7_ns));
}
}

TEST_CASE("Unavailable weak and spin models fail admission", "[mountain][spin]") {
  using namespace neutrino;
  CHECK_NOTHROW(validateNeutrinoPhysicsRequirements({}));
  CHECK_THROWS_AS(validateNeutrinoPhysicsRequirements({true,false,false}),std::invalid_argument);
  CHECK_THROWS_AS(validateNeutrinoPhysicsRequirements({false,true,false}),std::invalid_argument);
  CHECK_THROWS_AS(validateNeutrinoPhysicsRequirements({false,false,true}),std::invalid_argument);
  CHECK(tauChargeConjugatePolarization(Code::TauPlus,.6)==-.6);
  CHECK_THROWS_AS(tauChargeConjugatePolarization(Code::Electron,0.),std::invalid_argument);
  CHECK_THROWS_AS(tauChargeConjugatePolarization(Code::TauMinus,NAN),std::invalid_argument);
  CHECK_THROWS_AS(tauChargeConjugatePolarization(Code::TauMinus,1.01),std::invalid_argument);
}

TEST_CASE("Longitudinal endpoints preserve original TAUOLA daughters and RNG", "[mountain][spin]") {
  streams();auto& rng=RNGManager<>::getInstance();rng.setSeed(909501);
  auto cs=get_root_CoordinateSystem();
  for(double h : {-1.,0.,1.}) {
    neutrino::TransportedLeptonDecayConfig legacy,explicitP;
    legacy.helicity=h<0?tauola::Helicity::LeftHanded:
        (h>0?tauola::Helicity::RightHanded:tauola::Helicity::Unpolarized);
    explicitP.prescribedTauolaPolarization=h;
    neutrino::TransportedLeptonDecay a(legacy),b(explicitP);
    for(Code pid : {Code::TauMinus,Code::TauPlus}) for(double e : {10.,1000.,1.e8})
      for(int sample=0;sample<20;++sample) {
        SpinTestStack sa,sb;DirectionVector d(cs,{.36,.48,-.8});
        auto pa=parent(sa,pid,e,d),pb=parent(sb,pid,e,d);
        SpinTestStack::stack_view_type va(pa),vb(pb);
        auto saved=rng.getRandomStream("tauola");std::array<int,3> f{};
        rmarut_(&f[0],&f[1],&f[2]);
        a.doDecay(va);auto after=rng.getRandomStream("tauola");
        std::array<int,3> fend{};rmarut_(&fend[0],&fend[1],&fend[2]);
        rng.getRandomStream("tauola")=saved;Tauolapp::rmarin_(&f[0],&f[1],&f[2]);
        b.doDecay(vb);REQUIRE(va.getSize()==vb.getSize());
        auto i=va.begin(),j=vb.begin();
        for(;i!=va.end();++i,++j) {
          CHECK(i.getPID()==j.getPID());CHECK(i.getEnergy()==j.getEnergy());
          CHECK((i.getMomentum()-j.getMomentum()).getNorm()==0_GeV);
          CHECK(i.getHistoryId()==j.getHistoryId());
        }
        std::stringstream x,y;x<<after;y<<rng.getRandomStream("tauola");CHECK(x.str()==y.str());
        std::array<int,3> fafter{};rmarut_(&fafter[0],&fafter[1],&fafter[2]);CHECK(fend==fafter);
        REQUIRE(b.lastTauolaPolarization());CHECK_FALSE(b.lastTauolaPolarization()->mixtureDraw);
        CHECK(b.lastTauolaPolarization()->physicalPolarization==(pid==Code::TauMinus?h:-h));
      }
  }
}

TEST_CASE("Continuous longitudinal TAUOLA polarization obeys pion angular law", "[.spinangular][mountain]") {
  // Isolated invocation: no pi-only settings can leak into another run.
  streams();REQUIRE_FALSE(Tauolapp::Tauola::getIsTauolaIni());
  Tauolapp::Tauola::setSameParticleDecayMode(3);
  Tauolapp::Tauola::setOppositeParticleDecayMode(3);
  RNGManager<>::getInstance().setSeed(909502);
  auto cs=get_root_CoordinateSystem();
  neutrino::TransportedLeptonDecay decay;
  std::ofstream csv;
  if(auto path=std::getenv("C8_TAU_SPIN_MATRIX_CSV")) {
    csv.open(path);REQUIRE(csv.good());
    csv<<"pdg,P_physical,energy_GeV,axis,samples,mean_cos,expected_cos,sem,second_moment,right_fraction\n";
  }
  constexpr int n=3000;
  for(double h : {-1.,-.5,0.,.5,1.})
    for(Code pid : {Code::TauMinus,Code::TauPlus})
      for(double e : {10.,1000.,1.e8}) for(int axis=0;axis<2;++axis) {
        double sum=0.,sum2=0.;int right=0;
        double p=pid==Code::TauMinus?h:-h;
        for(int sample=0;sample<n;++sample) {
          DirectionVector d(cs,{0.,0.,1.});
          if(axis) d=DirectionVector(cs,{.36,.48,-.8});
          SpinTestStack stack;auto pa=parent(stack,pid,e,d);SpinTestStack::stack_view_type v(pa);
          COMBoost boost(pa.getMomentum(),pa.getMass());
          decay.doDecayWithLongitudinalPolarization(v,p);
          REQUIRE(v.getSize()==2);double cosine=2.;HEPEnergyType energy=0_GeV;
          int charge=0;
          for(auto const& child:v) {
            energy+=child.getEnergy();charge+=get_charge_number(child.getPID());
            if(std::abs(static_cast<int>(get_PDG(child.getPID())))==211) {
              // Energy-fraction two-body oracle avoids subtracting E and p
              // at gamma~1e8. It does not use the TAUOLA spin implementation.
              double const m=pa.getMass()/1_GeV, mpi=child.getMass()/1_GeV;
              double const estar=(m*m+mpi*mpi)/(2.*m), pstar=(m*m-mpi*mpi)/(2.*m);
              double const beta=pa.getMomentum().getNorm()/pa.getEnergy();
              cosine=((child.getEnergy()/pa.getEnergy())*m-estar)/(beta*pstar);
            } else CHECK(child.getPID()==(pid==Code::TauMinus?Code::NuTau:Code::NuTauBar));
          }
          CHECK(charge==get_charge_number(pid));CHECK(std::abs(energy/pa.getEnergy()-1.)<1.e-3);
          REQUIRE(std::isfinite(cosine));REQUIRE(std::abs(cosine)<=1.+1.e-3);
          sum+=cosine;sum2+=cosine*cosine;
          right+=decay.lastTauolaPolarization()->sampledHelicity==1.;
        }
        double sem=std::sqrt((1./3.-h*h/9.)/n);
        CAPTURE(pid,e,axis,h,sum/n,sem);
        // CP-conjugate pion analyzing powers: <cos(theta_pi)>=h/3.
        CHECK(std::abs(sum/n-h/3.)<6.*sem);
        CHECK(std::abs(sum2/n-1./3.)<.04);
        if(h!=0.) CHECK(std::abs(double(right)/n-(1.+h)/2.)<.04);
        if(csv) csv<<static_cast<int>(get_PDG(pid))<<','<<p<<','<<e<<','<<axis<<','<<n<<','<<sum/n<<','<<h/3.<<','<<sem<<','<<sum2/n<<','<<double(right)/n<<'\n';
      }
}

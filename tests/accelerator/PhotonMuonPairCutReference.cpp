/* Native C8 stack/ParticleCut oracle and signed output regression. */
#include "PhotonMuonPairDriver.hpp"
#include <corsika/accelerator/em/common/CorsikaOutputSink.hpp>
#include <corsika/modules/ParticleCut.hpp>
#include <corsika/modules/thinning/EMThinning.hpp>
#include <tests/common/SetupStack.hpp>
#include <yaml-cpp/yaml.h>
#include <cmath>
#include <stdexcept>
namespace photon_muon_test {
using namespace corsika;
namespace {
void near(double a,double b){if(!std::isfinite(a)||std::abs(a-b)>1.e-11)
  throw std::runtime_error("native signed cut bookkeeping mismatch");}
struct CutWriter : WriterOff {
  double sum{};
  void write(Point const&,Code,HEPEnergyType e){sum+=e/1_GeV;}
  template<class... T>void write(T const&...){}
  template<class... T>void writeProjected(T const&...){}
  std::size_t getNBins() const{return 2;}
  std::size_t GetNBins() const{return 2;}
  YAML::Node getConfig() const{YAML::Node n;n["bin-size"]=10.;return n;}
  void addBin(std::size_t,HEPEnergyType e){sum+=e/1_GeV;}
  void addBin(std::size_t,Code,double){}
  void addElectromagneticBin(std::size_t,HEPEnergyType e){sum+=e/1_GeV;}
};
using Sink=em::CorsikaOutputSink<CutWriter,CutWriter,CutWriter,CutWriter,CutWriter,CutWriter,CutWriter>;
}
double nativeCutReference(em::EmParticleState const& child,em::ProjectedEmStepRecord const& projected){
  static bool const registered=[] {
    RNGManager<>::getInstance().registerRandomStream("thinning");return true;
  }();
  (void)registered;
  DummyEnvironment env;auto cs=env.getCoordinateSystem();
  corsika::test::Stack stack;
  auto parent=stack.addParticle(std::make_tuple(Code::Photon,10_GeV,
    DirectionVector(cs,{0.,0.,1.}),Point(cs,3_m,0_m,0_m),1_us));
  corsika::test::StackView view(parent);
  auto projectile=view.getProjectile();
  auto pid=child.pid==13?Code::MuMinus:Code::MuPlus;
  auto kinetic=(child.energy_GeV-em::transportMassGeV(child.pid))*1_GeV;
  auto p=projectile.addSecondary(std::make_tuple(pid,kinetic,DirectionVector(cs,{0.,0.,1.})));
  p.setWeight(child.weight);
  near(p.getKineticEnergy()/1_GeV,kinetic/1_GeV);
  // Actual native thinning must leave a non-EM two-body final state alone.
  auto partner=projectile.addSecondary(std::make_tuple(
    pid==Code::MuMinus?Code::MuPlus:Code::MuMinus,9_GeV,DirectionVector(cs,{0.,0.,1.})));
  partner.setWeight(child.weight);
  EMThinning thin(20_GeV,1000.,true);thin.doSecondaries(view);
  if(view.getEntries()!=2)throw std::runtime_error("native EMThinning removed a muon");
  for(auto const& mu:view)near(mu.getWeight(),child.weight);
  ParticleCut<CutWriter> cut(.5_MeV,.5_MeV,.3_GeV,.3_GeV,.3_GeV,false);
  cut.doSecondaries(view);
  if(view.getEntries()!=1 || cut.statistics().particles!=1)
    throw std::runtime_error("native C8 did not cut sub-rest-mass child");
  auto expected=kinetic/1_GeV*child.weight;
  near(cut.sum,expected);near(cut.statistics().weighted_kinetic_energy_GeV,expected);
  CutWriter energy,other;Sink sink(cs,energy,other,other,other,other,other,other,false);
  sink.onProjectedStep(projected);near(energy.sum,expected);
  em::EmStepRecord full{};full.pid=child.pid;full.weight=child.weight;
  full.start_time_s=full.end_time_s=child.time_s;
  full.start_energy_GeV=full.end_energy_GeV=child.energy_GeV;
  for(int k=0;k<3;++k)full.start_position_m[k]=full.end_position_m[k]=child.position_m[k];
  full.cut_deposited_energy_GeV=full.deposited_energy_GeV=kinetic/1_GeV;
  full.native_muon_pair_cut=em::NativeMuonPairCutFlag;
  sink.onStep(full);near(energy.sum,2*expected);
  full.native_muon_pair_cut=0;
  if(expected<0.){bool rejected=false;try{sink.onStep(full);}catch(std::runtime_error const&){rejected=true;}
    if(!rejected)throw std::runtime_error("untagged negative deposit accepted");}
  return expected;
}
void checkCutOutput(em::GpuProfileResult const& profile){
  DummyEnvironment env;CutWriter energy,other;
  Sink sink(env.getCoordinateSystem(),energy,other,other,other,other,other,other,false);
  sink.onGpuProfile(profile);near(energy.sum,profile.weighted_native_muon_pair_cut_energy_GeV);
  auto invalid=profile;invalid.weighted_native_muon_pair_cut_energy_GeV=0.;
  if(profile.weighted_deposited_energy_GeV<0.){
    bool rejected=false;try{sink.onGpuProfile(invalid);}catch(std::runtime_error const&){rejected=true;}
    if(!rejected)throw std::runtime_error("negative profile without native ledger accepted");
  }
}
}

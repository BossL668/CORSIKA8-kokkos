#pragma once
#include "TerrainEnergyLedger.hpp"
#include <corsika/modules/terrain/TerrainAtmosphere.hpp>
#include <corsika/framework/process/ContinuousProcess.hpp>
#include <corsika/framework/core/Step.hpp>
#include <corsika/framework/process/BoundaryCrossingProcess.hpp>
#include <corsika/output/BaseOutput.hpp>
#include <fstream>
#include <iomanip>
#include <map>
#include <corsika/modules/terrain/TerrainEmSession.hpp>
#include <corsika/geometry/terrain/TerrainTrajectoryAudit.hpp>
#include <corsika/modules/neutrino/NeutrinoModelDomain.hpp>

namespace corsika::applications::terrain {
// Streaming, bounded-memory CPU and accelerated terrain track diagnostics.
// Never absorbs particles at rock/air interfaces and never repairs their node.
class ShowerOutput: public BaseOutput,public ContinuousProcess<ShowerOutput>,public BoundaryCrossingProcess<ShowerOutput>,public SecondariesProcess<ShowerOutput> {
 public:
  EnergyLedger ledger;
  explicit ShowerOutput(corsika::terrain::Environment const& env,
      corsika::terrain::TriangularMesh const* mesh=nullptr,std::size_t rowLimit=200000,
      double maxStepM=HUGE_VAL,double windowS=HUGE_VAL,bool requireNeutrinoCoverage=false,
      std::uint64_t stepLimit=2000000)
      :env_(env),mesh_(mesh),rowLimit_(rowLimit),stepLimit_(stepLimit),maxStepM_(maxStepM),windowS_(windowS),
       neutrinoDomain_(requireNeutrinoCoverage){}
  void startOfLibrary(boost::filesystem::path const& directory)override {
    directory_=directory;tracks_.open((directory/"tracks.csv").string());
    if(!tracks_)throw std::runtime_error("cannot create terrain tracks");
    tracks_<<std::setprecision(17)<<"step,pdg,weight,medium,x0_m,y0_m,z0_m,x1_m,y1_m,z1_m,t0_s,t1_s,E0_GeV,E1_GeV,nx0,ny0,nz0,nx1,ny1,nz1,history_id,parent_history_id\n";
    survivors_.open((directory/"window_survivors.csv").string());
    deposits_.open((directory/"deposits.csv").string());
    if(!survivors_||!deposits_)throw std::runtime_error("cannot create terrain energy diagnostics");
    survivors_<<std::setprecision(17)<<"pdg,weight,time_s,x_m,y_m,z_m,total_GeV,history_id\n";
    deposits_<<std::setprecision(17)<<"pdg,x0_m,y0_m,z0_m,x1_m,y1_m,z1_m,weighted_deposited_GeV\n";
    setInit(true);
  }
  void startOfShower(unsigned int)override{if(started_)throw std::logic_error("terrain reference supports one shower per output");started_=true;}
  void endOfShower(unsigned int)override{tracks_.flush();}
  void endOfLibrary()override{tracks_.close();survivors_.close();deposits_.close();}
  YAML::Node getConfig()const override {
    YAML::Node n;n["scope"]="native five-layer atmosphere + DEM transport audit; air magnetic field; zero field in rock; radio disabled";
    n["material_validation"]="trajectory midpoint (quadratic leapfrog in magnetic field); 10 nm interface ambiguity band";
    n["energy_ledger"]="deposition only; hadronic rest-mass closure not certified";return n;
  }
  YAML::Node getSummary()const override {
    YAML::Node n;n["steps"]=steps_;n["rock_steps"]=rock_;n["air_steps"]=air_;
    n["material_mismatches"]=mismatches_;n["surface_ambiguities"]=surface_;
    n["rock_to_air"]=exits_;n["air_to_rock"]=entries_;n["world_escapes"]=escapes_;
    n["deposited_energy_GeV"]=deposition_;
    n["accelerated_steps"]=accelerated_;n["device_binding_energy_GeV"]=binding_;
    n["csv_row_limit"]=rowLimit_;n["csv_truncated"]=steps_>rowLimit_;
    n["finite_window_survivors"]=windowSurvivors_;
    n["finite_window_escaped_total_GeV"]=windowEscaped_;
    n["finite_window_escaped_neutrino_GeV"]=windowNeutrino_;
    n["finite_window_semantics"]="remaining total energy is escape, not deposition; distinct from world escapes";
    n["deposition_rows"]=depositRows_;n["deposition_csv_truncated"]=depositRows_>rowLimit_;
    auto domain=n["neutrino_model_domain"];
    domain["required"]=neutrinoDomain_.strict();
    domain["uncovered_transported_histories"]=neutrinoDomain_.count();
    domain["weighted_energy_at_first_uncovered_step_GeV"]=neutrinoDomain_.weightedEnergyGeV();
    domain["all_observed_neutrinos_in_CTW_domain"]=neutrinoDomain_.count()==0;
    domain["not_a_deposition_or_interaction_probability"]=true;
    for(auto const& [pdg,row]:neutrinoDomain_.species()) {
      domain["by_pdg"][pdg]["histories"]=row.count;
      domain["by_pdg"][pdg]["weighted_energy_GeV"]=row.weightedEnergyGeV;
    }
    return n;
  }
  template<class P,class T> LengthType getMaxStepLength(P const& p,T const&)const {
    neutrinoDomain_.check(p.getPID(),p.getEnergy()/1_GeV);
    double cap=is_neutrino(p.getPID())?HUGE_VAL:maxStepM_;
    if(std::isfinite(windowS_)) {
      double speed=(p.getMomentum().getNorm()/p.getEnergy())*constants::c/(1_m/1_s);
      cap=std::min(cap,std::max(1.e-9,(windowS_-p.getTime()/1_s)*speed));
    }
    return cap*1_m;
  }
  template<class View> void doSecondaries(View& view) {ledger.recordCpuVertex(view);}
  template<class P> ProcessReturn doContinuous(Step<P> const& step,bool) {
    if(ledger.enabled)ledger.cpuContinuous+=static_cast<long double>(step.getParticlePre().getWeight())*
        ((step.getEkinPre()-step.getEkinPost())/1_GeV);
    if(++steps_>stepLimit_)throw std::runtime_error("terrain reference step budget exceeded (output incomplete)");
    auto const& p=step.getParticlePre();auto* logical=p.getNode();
    neutrinoDomain_.observe(p.getPID(),p.getEnergy()/1_GeV,p.getWeight(),p.getHistoryId());
    auto middle=step.getPositionPre()+.5*step.getDisplacement();
    // Step stores only endpoints, not the original trajectory. Reconstruct its
    // quadratic midpoint using the unchanged pre-step velocity and flight time.
    // The assembled continuous processes change energy/direction, not position/time.
    if(p.getCharge()!=0*constants::e&&logical->getModelProperties().getMagneticField(p.getPosition()).getNorm()!=0_T) {
      auto cs=env_.getCoordinateSystem();auto a=step.getPositionPre(),b=step.getPositionPost();
      auto delta=p.getVelocity()*step.getDiffT();
      auto m=corsika::terrain::flat::quadraticMidpoint(
          {a.getX(cs)/1_m,a.getY(cs)/1_m,a.getZ(cs)/1_m},
          {b.getX(cs)/1_m,b.getY(cs)/1_m,b.getZ(cs)/1_m},
          {delta.getX(cs)/1_m,delta.getY(cs)/1_m,delta.getZ(cs)/1_m});
      middle=Point(cs,m.x*1_m,m.y*1_m,m.z*1_m);
    }
    auto* actual=env_.getUniverse()->getContainingNode(middle);
    auto isRock=[](auto* node){return node&&dynamic_cast<corsika::terrain::TriangularMesh const*>(&node->getVolume());};
    bool rock=isRock(logical);if(rock)++rock_;else ++air_;
    if(rock!=bool(isRock(actual))) {
      // Preserve the mountain audit's interface ambiguity accounting.
      auto* node=rock?logical:actual;
      auto& mesh=static_cast<corsika::terrain::TriangularMesh const&>(node->getVolume());
      double near=std::numeric_limits<double>::infinity();
      for(double sign:{-1.,1.}) {auto h=mesh.intersectRay(middle,sign*step.getDirectionPre());if(h.hit)near=std::min(near,h.distance/1_m);}
      if(near>1.e-8){++mismatches_;throw std::runtime_error("terrain material mismatch; do not accept output");}
      ++surface_;
    }
    auto cs=env_.getCoordinateSystem();auto a=step.getPositionPre(),b=step.getPositionPost();
    auto d0=step.getDirectionPre(),d1=step.getDirectionPost();
    if(steps_<=rowLimit_)tracks_<<steps_<<','<<static_cast<int>(get_PDG(p.getPID()))<<','<<p.getWeight()<<','<<(rock?"rock":"air")<<','
      <<a.getX(cs)/1_m<<','<<a.getY(cs)/1_m<<','<<a.getZ(cs)/1_m<<','<<b.getX(cs)/1_m<<','<<b.getY(cs)/1_m<<','<<b.getZ(cs)/1_m<<','
      <<step.getTimePre()/1_s<<','<<step.getTimePost()/1_s<<','<<(step.getEkinPre()+get_mass(p.getPID()))/1_GeV<<','<<(step.getEkinPost()+get_mass(p.getPID()))/1_GeV<<','
      <<static_cast<double>(d0.getX(cs))<<','<<static_cast<double>(d0.getY(cs))<<','<<static_cast<double>(d0.getZ(cs))<<','
      <<static_cast<double>(d1.getX(cs))<<','<<static_cast<double>(d1.getY(cs))<<','<<static_cast<double>(d1.getZ(cs))<<','
      <<p.getHistoryId()<<','<<p.getParentHistoryId()<<'\n';
    if(!tracks_)throw std::runtime_error("terrain track output failed");
    // Below-cut particles are left to stock ParticleCut: never double count
    // their energy as both a cut deposit and a finite-window survivor.
    auto kinetic=step.getEkinPost();auto perParticle=is_nucleus(p.getPID())?kinetic/get_nucleus_A(p.getPID()):kinetic;
    if(std::isfinite(windowS_)&&step.getTimePost()/1_s>=windowS_-1.e-15&&
        perParticle>=get_kinetic_energy_propagation_threshold(p.getPID())) {
      recordWindow(static_cast<int>(get_PDG(p.getPID())),p.getWeight(),step.getTimePost()/1_s,
                   b,(kinetic+get_mass(p.getPID()))/1_GeV,p.getHistoryId());
      return ProcessReturn::ParticleAbsorbed;
    }
    return ProcessReturn::Ok;
  }
  void recordDeviceStep(corsika::terrain::TerrainEmStep const& r) {
    ledger.recordDevice(r);
    if(++steps_>stepLimit_)throw std::runtime_error("terrain validation step budget exceeded");
    ++accelerated_;bool rock=r.start.medium_id==1;if(rock)++rock_;else ++air_;
    auto cs=env_.getCoordinateSystem();auto const& a=r.start;auto const& b=r.end;
    if(r.crossed_material && (b.medium_id!=1-a.medium_id||b.history_id!=a.history_id||
        b.parent_history_id!=a.parent_history_id||b.generation!=a.generation||
        b.step_id!=a.step_id+1||b.weight!=a.weight||b.time_s<a.time_s||b.energy_GeV>a.energy_GeV))
      throw std::runtime_error("material transition violated particle identity/state continuity");
    Point middle(cs,r.path_midpoint_m[0]*1_m,r.path_midpoint_m[1]*1_m,r.path_midpoint_m[2]*1_m);
    auto* actual=env_.getUniverse()->getContainingNode(middle);
    bool actualRock=actual&&dynamic_cast<corsika::terrain::TriangularMesh const*>(&actual->getVolume());
    if(rock!=actualRock) {
      // Surface points have ambiguous geometric ownership; only admit the same
      // 10 nm band as the independent CPU audit, never a long wrong-material step.
      auto* mesh=mesh_;
      if(!mesh)throw std::runtime_error("device terrain logical material has no matching mesh");
      DirectionVector direction(cs,{a.direction[0],a.direction[1],a.direction[2]});
      double near=HUGE_VAL;for(double sign:{-1.,1.}){auto h=mesh->intersectRay(middle,sign*direction);if(h.hit)near=std::min(near,h.distance/1_m);}
      if(near>1.e-8){++mismatches_;throw std::runtime_error("device terrain step crossed an unprocessed material boundary");}
      ++surface_;
    }
    if(r.crossed_material){if(rock)++exits_;else ++entries_;}
    if(r.outcome==corsika::terrain::TerrainEmOutcome::Escape)++escapes_;
    if(!std::isfinite(r.deposited_GeV)||r.deposited_GeV<-1.e-12)throw std::runtime_error("invalid device terrain deposition");
    deposition_+=r.deposited_GeV*a.weight;binding_+=r.binding_energy_GeV*a.weight;
    recordDeposit(a.pid,a.position_m,b.position_m,r.deposited_GeV*a.weight);
    if(steps_<=rowLimit_) {
      tracks_<<steps_<<','<<a.pid<<','<<a.weight<<','<<(rock?"rock":"air");
      for(double x:a.position_m)tracks_<<','<<x;
      for(double x:b.position_m)tracks_<<','<<x;
      tracks_<<','<<a.time_s<<','<<b.time_s<<','<<a.energy_GeV<<','<<b.energy_GeV;
      for(double x:a.direction)tracks_<<','<<x;
      for(double x:b.direction)tracks_<<','<<x;
      tracks_<<','<<a.history_id<<','<<a.parent_history_id<<'\n';
    }
    if(!tracks_)throw std::runtime_error("device terrain track output failed");
  }
  template<class P> ProcessReturn doBoundaryCrossing(P& p,typename P::node_type const& from,typename P::node_type const& to) {
    bool a=dynamic_cast<corsika::terrain::TriangularMesh const*>(&from.getVolume());
    bool b=dynamic_cast<corsika::terrain::TriangularMesh const*>(&to.getVolume());
    if(a&&!b)++exits_;
    if(!a&&b)++entries_;
    if(&to==env_.getUniverse().get()) {
      ++escapes_;
      if(ledger.enabled)ledger.worldEscaped+=static_cast<long double>(p.getWeight())*(p.getEnergy()/1_GeV);
    }
    return ProcessReturn::Ok;
  }
  void recordDeviceWindow(corsika::terrain::TerrainEmStep const& r) {
    auto const& p=r.end;auto cs=env_.getCoordinateSystem();
    recordWindow(p.pid,p.weight,p.time_s,Point(cs,p.position_m[0]*1_m,p.position_m[1]*1_m,p.position_m[2]*1_m),p.energy_GeV,p.history_id);
  }
  void write(Point const& p,Code pid,HEPEnergyType value) {write(p,p,pid,value);}
  YAML::Node energySummary(double initial,ParticleCutStatistics const& cut)const {
    return ledger.summary(initial,deposition_,windowEscaped_,cut);
  }
  void write(Point const& a,Point const& b,Code pid,HEPEnergyType value) {
    double v=value/1_GeV;if(!std::isfinite(v)||v<-1.e-12)throw std::runtime_error("invalid terrain energy deposit");deposition_+=v;
    auto cs=env_.getCoordinateSystem();
    double x[3]{a.getX(cs)/1_m,a.getY(cs)/1_m,a.getZ(cs)/1_m};
    double y[3]{b.getX(cs)/1_m,b.getY(cs)/1_m,b.getZ(cs)/1_m};
    recordDeposit(static_cast<int>(get_PDG(pid)),x,y,v);
  }
 private:
  void recordDeposit(int pid,double const* a,double const* b,double value) {
    if(value==0.)return;
    if(++depositRows_<=rowLimit_) {
      deposits_<<pid;for(int i=0;i<3;++i)deposits_<<','<<a[i];
      for(int i=0;i<3;++i)deposits_<<','<<b[i];
      deposits_<<','<<value<<'\n';
    }
    if(!deposits_)throw std::runtime_error("terrain deposition stream failed");
  }
  void recordWindow(int pid,double weight,double time,Point const& x,double total,std::uint64_t history) {
    ++windowSurvivors_;windowEscaped_+=weight*total;
    if(std::abs(pid)==12||std::abs(pid)==14||std::abs(pid)==16)windowNeutrino_+=weight*total;
    auto cs=env_.getCoordinateSystem();
    survivors_<<pid<<','<<weight<<','<<time<<','<<x.getX(cs)/1_m<<','<<x.getY(cs)/1_m<<','<<x.getZ(cs)/1_m<<','<<total<<','<<history<<'\n';
    if(!survivors_)throw std::runtime_error("terrain survivor stream failed");
  }
  corsika::terrain::Environment const& env_;corsika::terrain::TriangularMesh const* mesh_{};
  boost::filesystem::path directory_;std::ofstream tracks_;
  bool started_{};std::uint64_t steps_{},rock_{},air_{},mismatches_{},surface_{},exits_{},entries_{},escapes_{};
  double deposition_{},binding_{};std::uint64_t accelerated_{};
  std::size_t rowLimit_;
  std::uint64_t stepLimit_;
  double maxStepM_,windowS_,windowEscaped_{},windowNeutrino_{};
  std::uint64_t windowSurvivors_{},depositRows_{};
  std::ofstream survivors_,deposits_;
  neutrino::NeutrinoModelDomain neutrinoDomain_;
};
}

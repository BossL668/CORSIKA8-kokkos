/* Independent finite-volume diagnostics; does not alter air-shower writers. */
#pragma once

#include <corsika/framework/core/Step.hpp>
#include <corsika/framework/geometry/ConvexPolyhedron.hpp>
#include <corsika/framework/process/ContinuousProcess.hpp>
#include <corsika/output/BaseOutput.hpp>
#include <corsika/accelerator/em/common/Types.hpp>
#include <corsika/accelerator/radio/common/Types.hpp>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <functional>
#include <iomanip>
#include <map>
#include <limits>
#include <stdexcept>
#include <vector>

namespace corsika::applications::mountain {

class Diagnostics : public BaseOutput, public ContinuousProcess<Diagnostics> {
 public:
  Diagnostics(ConvexPolyhedron const& volume, Point origin,
              DirectionVector direction, double extent_m, std::size_t bins=200)
      : volume_(volume), origin_(origin), direction_(direction),
        extent_(extent_m), bins_(bins), deposit_(bins, 0.) {}

  void startOfLibrary(boost::filesystem::path const& directory) override {
    directory_=directory;
    escapes_.open((directory/"escaped_particles.csv").string());
    if (!escapes_) throw std::runtime_error("cannot create escape output");
    escapes_ << std::setprecision(17)
             << "pdg,total_energy_GeV,weight,x_m,y_m,z_m,time_s,nx,ny,nz\n";
    setInit(true);
  }
  void startOfShower(unsigned int) override {
    if (started_) throw std::runtime_error("mountain pilot requires one shower per output");
    started_=true;
  }
  void endOfShower(unsigned int) override { escapes_.flush(); }
  void endOfLibrary() override {
    std::ofstream out((directory_/"profile.csv").string());
    if (!out) throw std::runtime_error("cannot create mountain profile");
    out << std::setprecision(17) << "depth_m,deposited_energy_GeV";
    for (auto const& row:counts_) out << ",N_pdg_" << row.first;
    out << '\n';
    for (std::size_t i=0;i<bins_;++i) {
      out << depth(i) << ',' << deposit_[i];
      for (auto const& row:counts_) out << ',' << row.second[i];
      out << '\n';
    }
    escapes_.close();
  }
  YAML::Node getConfig() const override {
    YAML::Node n;
    n["geometry"]="finite_convex_homogeneous_sio2";
    n["outside_transport"]="absorbing escape boundary; no exterior atmosphere";
    n["profile"]="weighted crossings of planes normal to primary direction";
    n["energy_closure"]="not certified: nuclear rest-mass exchange requires a separate ledger";
    return n;
  }
  YAML::Node getSummary() const override {
    YAML::Node n;
    n["cpu_steps"]=cpu_steps_; n["accelerated_steps"]=gpu_steps_;
    n["escaped_particles"]=escaped_; n["escaped_weighted_total_energy_GeV"]=escape_energy_;
    n["deposited_energy_GeV"]=deposited_; n["boundary_violations"]=boundary_violations_;
    return n;
  }
  template<class P,class T> LengthType getMaxStepLength(P const&,T const&) const {
    return std::numeric_limits<double>::infinity()*1_m;
  }
  template<class P> ProcessReturn doContinuous(Step<P> const& step,bool) {
    ++cpu_steps_;
    auto const& p=step.getParticlePre();
    recordSegment(step.getPositionPre(),step.getPositionPost(),p.getPID(),p.getWeight());
    auto const pos=step.getPositionPost();
    auto const xyz=coordinates(pos);
    auto const d=step.getDirectionPre().getComponents(volume_.getCoordinateSystem());
    bool reached=false;
    for (auto const& face:volume_.exportPlanes()) {
      auto const s=face.nx*xyz[0]+face.ny*xyz[1]+face.nz*xyz[2]-face.offset_m;
      auto const outward=face.nx*d[0]+face.ny*d[1]+face.nz*d[2];
      if (s>=-1e-8 && outward>0.) reached=true;
    }
    if (reached) {
      // All continuous processes still execute after ParticleAbsorbed. Cut has
      // priority at a coincident boundary, matching the device termination.
      if (!endpoint_cut || !endpoint_cut(p.getPID(),step.getEkinPost(),step.getTimePost()))
        escape(p.getPID(),(step.getEkinPost()+get_mass(p.getPID()))/1_GeV,
             p.getWeight(),pos,step.getTimePost(),step.getDirectionPost());
      return ProcessReturn::ParticleAbsorbed;
    }
    return ProcessReturn::Ok;
  }
  void write(Point const& p,Code,HEPEnergyType energy) {
    auto const value=energy/1_GeV;
    finiteNonnegative(value);
    deposited_+=value;
    auto i=bin(project(p));
    if (i>=0 && i<static_cast<int>(bins_)) deposit_[i]+=value;
  }
  void write(Point const& a,Point const& b,Code pid,HEPEnergyType energy) {
    auto const value=energy/1_GeV; finiteNonnegative(value);
    auto x=project(a),y=project(b);
    if (std::abs(x-y)<1e-14) { write(b,pid,energy); return; }
    deposited_+=value;
    if(x>y) std::swap(x,y);
    for(std::size_t i=0;i<bins_;++i) {
      auto lo=-extent_+i*width(),hi=lo+width();
      deposit_[i]+=value*std::max(0.,std::min(y,hi)-std::max(x,lo))/(y-x);
    }
  }
  void onStep(gpu::em::EmStepRecord const& r) {
    ++gpu_steps_;
    auto pid=convert_from_PDG(static_cast<PDGCode>(r.pid));
    Point a=point(r.start_position_m),b=point(r.end_position_m);
    recordSegment(a,b,pid,r.weight);
    write(a,b,pid,(r.deposited_energy_GeV-r.cut_deposited_energy_GeV)*r.weight*1_GeV);
    write(b,pid,r.cut_deposited_energy_GeV*r.weight*1_GeV);
  }
  void onObservation(gpu::em::ObservationRecord const& r) {
    if(r.status!=gpu::em::ObservationStatus::EscapedEnvironment) return;
    auto const& p=r.particle;
    escape(convert_from_PDG(static_cast<PDGCode>(p.pid)),p.energy_GeV,p.weight,
           point(p.position_m),p.time_s*1_s,
           DirectionVector(volume_.getCoordinateSystem(),{p.direction[0],p.direction[1],p.direction[2]}));
  }
  void onFirstInteraction(gpu::em::GpuFirstInteractionSnapshot const&) {}
  void onProjectedStep(gpu::em::ProjectedEmStepRecord const&) {
    throw std::logic_error("mountain diagnostics require full step records");
  }
  void onGpuProfile(gpu::em::GpuProfileResult const&) {
    throw std::logic_error("mountain diagnostics do not accept atmospheric projected profiles");
  }
  std::function<void(gpu::em::RadioTrackRecord const&)> radio_track;
  std::function<bool(Code,HEPEnergyType,TimeType)> endpoint_cut;
  std::function<void(gpu::radio::GpuRadioWaveforms const&)> radio_merge;
  void onRadioTrack(gpu::em::RadioTrackRecord const& r) {
    if(radio_track) radio_track(r);
  }
  void onGpuRadioWaveforms(gpu::radio::GpuRadioWaveforms const& waves,std::uint64_t) {
    if(!radio_merge) throw std::logic_error("unexpected GPU radio output");
    radio_merge(waves);
  }
 private:
  void finiteNonnegative(double v) const {
    if(!std::isfinite(v)||v<-1e-12) throw std::runtime_error("invalid energy deposit");
  }
  std::array<double,3> coordinates(Point const& p) const {
    auto const cs=volume_.getCoordinateSystem();
    return {p.getX(cs)/1_m,p.getY(cs)/1_m,p.getZ(cs)/1_m};
  }
  Point point(double const p[3]) const {
    return {volume_.getCoordinateSystem(),p[0]*1_m,p[1]*1_m,p[2]*1_m};
  }
  double project(Point const& p) const { return (p-origin_).dot(direction_)/1_m; }
  double width() const {return 2*extent_/bins_;}
  double depth(std::size_t i) const {return -extent_+(i+0.5)*width();}
  int bin(double x) const{return static_cast<int>(std::floor((x+extent_)/width()));}
  void recordSegment(Point const& a,Point const& b,Code pid,double weight) {
    if(!std::isfinite(weight)||weight<0) throw std::runtime_error("invalid particle weight");
    auto const xyz=coordinates(b);
    for(auto v:xyz) if(!std::isfinite(v)) throw std::runtime_error("nonfinite mountain track");
    for(auto const& f:volume_.exportPlanes())
      if(f.nx*xyz[0]+f.ny*xyz[1]+f.nz*xyz[2]-f.offset_m>1e-6) {
        ++boundary_violations_; throw std::runtime_error("particle transported outside mountain");
      }
    auto& row=counts_[static_cast<int>(get_PDG(pid))];
    if(row.empty())row.resize(bins_);
    auto x=project(a),y=project(b);if(x>y)std::swap(x,y);
    for(std::size_t i=0;i<bins_;++i)if(x<=depth(i)&&depth(i)<y)row[i]+=weight;
  }
  void escape(Code pid,double energy,double weight,Point const& p,TimeType t,DirectionVector const& d) {
    finiteNonnegative(energy);++escaped_;escape_energy_+=energy*weight;
    auto x=coordinates(p);auto n=d.getComponents(volume_.getCoordinateSystem());
    escapes_<<static_cast<int>(get_PDG(pid))<<','<<energy<<','<<weight<<','
      <<x[0]<<','<<x[1]<<','<<x[2]<<','<<t/1_s<<','<<n[0]<<','<<n[1]<<','<<n[2]<<'\n';
    if(!escapes_)throw std::runtime_error("failed to write escape record");
  }
  ConvexPolyhedron const& volume_;Point origin_;DirectionVector direction_;
  double extent_;std::size_t bins_;std::vector<double> deposit_;
  std::map<int,std::vector<double>> counts_;
  boost::filesystem::path directory_;std::ofstream escapes_;
  bool started_{};std::uint64_t cpu_steps_{},gpu_steps_{},escaped_{},boundary_violations_{};
  double deposited_{},escape_energy_{};
};
} // namespace corsika::applications::mountain

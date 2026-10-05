#pragma once
#include <corsika/modules/egs4/Session.hpp>
#include <corsika/detail/modules/egs4/Egs4ProposalMuonBackend.hpp>
#include <corsika/framework/geometry/PhysicalGeometry.hpp>
#include <corsika/accelerator/em/common/RouterParticleConversion.hpp>
#include <stdexcept>
#include <utility>

namespace c7_egs4::application {
// Actual HybridCascade external-router API. The C8 stack owns the global
// history-ID allocator; Session owns resident native transport/RNG state.
// HostHandler must explicitly consume transport/decay/selected-nuclear requests. This is
// not a PROPOSAL fallback and never silently re-routes an EM particle to C8.
template<class Stack,class HostHandler,class RegionOf,class MuonRouter=NoAcceleratedMuons>class Router {
  Session& session_;
  ::corsika::CoordinateSystemPtr coordinates_;
  HostHandler& host_;
  RegionOf region_of_;
  std::size_t batch_size_;
  std::uint64_t children_per_particle_;
  std::vector<::corsika::gpu::em::EmParticleState> staged_;
  RunStatistics statistics_;
  bool ended_{};
  MuonRouter* muons_{};
  ProposalMuonBackend* muon_backend_{};
public:
  Router(Session& session,Configuration const& configuration,OutputCallbacks const& output,
      ::corsika::CoordinateSystemPtr coordinates,HostHandler& host,RegionOf region_of,
      std::size_t batch_size,bool host_reference=false,MuonRouter* muons=nullptr,
      ProposalMuonBackend* muon_backend=nullptr):session_(session),coordinates_(std::move(coordinates)),
      host_(host),region_of_(std::move(region_of)),batch_size_(batch_size),
      children_per_particle_(configuration.native_prompt_resonance_decay?5:configuration.native_prompt_rho_decay?4:configuration.native_photonuclear_vertices?3:2),
      muons_(muons),muon_backend_(muon_backend) {
    if(bool(muons_)!=bool(muon_backend_))throw std::invalid_argument("Muon router/backend must be supplied together");
    if(!coordinates_||!batch_size||batch_size>std::size_t(configuration.queue_capacity/2))
      throw std::invalid_argument("Invalid native EGS4 router batch/coordinate system");
    if(configuration.resume_retained_shower)session_.resume(configuration.seed,configuration.shower_id,output);
    else session_.begin(configuration,output,host_reference);
  }
  template<class Particle>bool canRoute(Particle const& p,std::uint64_t step)const {
    using ::corsika::Code;
    return p.getPID()==Code::Electron||p.getPID()==Code::Positron||p.getPID()==Code::Photon||
      (muons_&&muons_->canRoute(p,step));
  }
  template<class Particle>void stage(Particle const& particle,::corsika::transport::HistoryId id,
      ::corsika::transport::HistoryId parent,::corsika::transport::Generation generation,
      ::corsika::transport::StepId step) {
    if(ended_)throw std::logic_error("Invalid native EGS4 stage");
    if(!::corsika::is_em(particle.getPID())) {
      if(!muons_)throw std::logic_error("Non-EM input without muon router");
      muons_->stage(particle,id,parent,generation,step);return;
    }
    auto p=::corsika::gpu::em::router_detail::toDeviceState(particle,coordinates_,id,parent,generation,step);
    if(muon_backend_)muon_backend_->auditScalarHandoff(p);
    p.medium_id=region_of_(particle.getNode());staged_.push_back(p);
  }
  bool pending()const{return !ended_&&(!staged_.empty()||session_.activeParticles()!=0||
    (muons_&&muons_->pending())||(muon_backend_&&muon_backend_->hasElectromagnetic()));}
  template<class H,class S>bool consumeForcedDecay(H id,S step){return muons_&&muons_->consumeForcedDecay(id,step);}
  bool readyForScalarInterleave()const {
    // Accumulate scalar-produced muons with the EM front; a muon-only front
    // is flushed when the scalar stack drains, not after every individual root.
    return !ended_&&staged_.size()+session_.activeParticles()>=batch_size_;
  }
  std::size_t advanceOneWavefrontAndReturn(Stack& stack) {
    if(ended_)throw std::logic_error("Native EGS4 router already ended");
    std::size_t returned=0;
    if(muons_&&muons_->pending())returned+=muons_->advanceOneWavefrontAndReturn(stack);
    if(muon_backend_)for(auto const& p:muon_backend_->takeElectromagnetic())staged_.push_back(p);
    if(!staged_.empty()){session_.append(staged_);staged_.clear();}
    auto active=session_.activeParticles();if(!active)return returned;
    auto reserved=children_per_particle_*active;
    auto first=stack.reserveTransportHistoryIds(reserved);
    auto wave=session_.advance(first,reserved);
    for(auto const& request:wave.host_requests)if(!host_.canHandle(request))
      throw std::runtime_error("Native EGS4 host request has no explicit consumer; shower incomplete");
    for(auto const& request:wave.host_requests)returned+=host_.handle(stack,request);
    return returned;
  }
  void endOfShower() {
    if(ended_||pending())throw std::logic_error("Native EGS4 router did not drain");
    if(muons_)muons_->endOfShower();
    statistics_=session_.finishEMQueue();ended_=true;
  }
  RunStatistics const& statistics()const {
    if(!ended_)throw std::logic_error("Native EGS4 statistics requested before router end");
    return statistics_;
  }
};
} // namespace c7_egs4::application

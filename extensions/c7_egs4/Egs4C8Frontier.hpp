#pragma once
#include "Egs4C8Router.hpp"
#include <applications/detail/air_shower_multigpu/Frontier.hpp>
#include <memory>

namespace c7_egs4::application {
// Experiment-only adapter around the existing C8 frontier format/feed. The
// Prefix captures unstarted e-/e+/gamma, and optionally PROPOSAL muons.
// No in-flight optical clock is serialized or reset; CPU-muon reference
// mode retains the original EM-only frontier.
template<class Stack,class HostHandler,class RegionOf,class MuonRouter=NoAcceleratedMuons>class FrontierRouter {
  using Base=Router<Stack,HostHandler,RegionOf,MuonRouter>;
  using Input=::corsika::applications::multigpu::FrontierInput;
  std::unique_ptr<Base> base_;
  std::unique_ptr<Input> input_;
  ::corsika::CoordinateSystemPtr cs_;
  std::ofstream capture_;
  RunStatistics prefix_stats_;
  unsigned worker_{};
  std::size_t batch_{};
  bool prefix_{},ended_{};
  bool capture_muons_{};
  void checkNamespace(Stack& stack)const {
    // The existing stack has no non-mutating next-ID API; reserve a guard ID.
    // Gaps are deliberate and never reused (zero-length reservations are invalid).
    if(worker_&&stack.reserveTransportHistoryIds(1)>=
       (worker_+1)*::corsika::applications::multigpu::historyStride)
      throw std::overflow_error("Native EGS4 worker history namespace exhausted");
  }
public:
  FrontierRouter(Session& session,Configuration const& cfg,OutputCallbacks const& callbacks,
      ::corsika::CoordinateSystemPtr cs,HostHandler& host,RegionOf region,Stack& stack,
      bool host_reference,std::string const& capture,std::string const& frontier,unsigned worker,std::size_t batch=64,
      MuonRouter* muons=nullptr,ProposalMuonBackend* muon_backend=nullptr,bool capture_muons=false):
    cs_(cs),worker_(worker),batch_(batch),prefix_(!capture.empty()),capture_muons_(capture_muons) {
    if(!batch_||batch_>std::size_t(cfg.queue_capacity/2))throw std::invalid_argument("Invalid native frontier batch size");
    if(prefix_&&!frontier.empty())throw std::invalid_argument("Cannot capture and import a frontier together");
    if(prefix_) {
      capture_.exceptions(std::ios::badbit|std::ios::failbit);
      capture_.open(capture);capture_<<"C8_STATIC_FRONTIER_V1\n"<<std::setprecision(17);
      prefix_stats_.execution_space=capture_muons_?"C8 CPU prefix; EM and muons captured before transport":"C8 CPU prefix; EM captured before transport";
    }else {
      if(!frontier.empty()) {
        input_=std::make_unique<Input>(frontier);input_->reserveNamespace(stack,worker);
      }else if(worker)throw std::invalid_argument("Worker ID requires a frontier");
      base_=std::make_unique<Base>(session,cfg,callbacks,cs,host,region,batch_,host_reference,muons,muon_backend);
    }
  }
  template<class Particle,class Step>bool canRoute(Particle const& p,Step step)const {
    return prefix_?(::corsika::is_em(p.getPID())||(capture_muons_&&::corsika::is_muon(p.getPID()))):base_->canRoute(p,step);
  }
  template<class H,class S>bool consumeForcedDecay(H id,S step){return !prefix_&&base_->consumeForcedDecay(id,step);}
  template<class Particle,class H,class G,class S>void stage(Particle const& p,H id,H parent,G generation,S step) {
    if(ended_)throw std::logic_error("Invalid native frontier stage");
    if(prefix_) {
      if(step!=0)throw std::runtime_error("Native prefix cannot serialize an in-flight EM particle");
      ::corsika::applications::multigpu::write(capture_,
        ::corsika::gpu::em::router_detail::toDeviceState(p,cs_,id,parent,generation,step));
      ++prefix_stats_.injections;
    }else base_->stage(p,id,parent,generation,step);
  }
  bool pending()const {return !ended_&&!prefix_&&(base_->pending()||(input_&&input_->pending()));}
  bool readyForScalarInterleave()const {return !ended_&&!prefix_&&base_->readyForScalarInterleave();}
  std::size_t advanceOneWavefrontAndReturn(Stack& stack) {
    if(ended_||prefix_)throw std::logic_error("Cannot advance captured/closed native frontier");
    checkNamespace(stack);
    if(base_->pending()) {auto count=base_->advanceOneWavefrontAndReturn(stack);checkNamespace(stack);return count;}
    if(input_&&input_->pending()) {
      auto count=input_->refill(stack,cs_,batch_);
      // Imported input is a new native history, not a continuation checkpoint.
      for(auto const& p:stack)if((!::corsika::is_em(p.getPID())&&!(capture_muons_&&::corsika::is_muon(p.getPID())))||p.getStepId()!=0)
        throw std::invalid_argument("EGS4 frontier must contain unstarted EM roots, not in-flight states");
      return count;
    }
    return 0;
  }
  void endOfShower() {
    if(ended_||pending())throw std::logic_error("Native frontier did not drain");
    if(prefix_){capture_.flush();capture_.close();}else base_->endOfShower();
    ended_=true;
  }
  RunStatistics const& statistics()const {
    if(!ended_)throw std::logic_error("Native frontier statistics before completion");
    return prefix_?prefix_stats_:base_->statistics();
  }
  std::size_t importedRoots()const {return input_?input_->count():0;}
};
} // namespace c7_egs4::application

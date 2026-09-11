/* Bounded HybridCascade router. Caller supplies node->region mapping, selected
 * CPU fallback and output callbacks. No geometry/material names, YAML or radio.
 * CPU owns the stack; synchronous records return before it is edited. */
#pragma once
#include <corsika/modules/transport/InterfaceEmSession.hpp>
#include <corsika/modules/terrain/TerrainParticleAdmission.hpp>
#include <corsika/accelerator/em/common/RouterParticleConversion.hpp>
#include <deque>
#include <sstream>
#include <iomanip>

namespace corsika::interfaces {
struct InterfaceRouterStatistics {
  std::size_t peak{};
  std::uint64_t batches{},advanced{},fallbacks{},cuts{},escaped{},windowEscaped{},canonicalized{};
};
template<class Stack,class Fallback,class Output,class RegionOf> class InterfaceEmRouter {
 public:
  InterfaceEmRouter(InterfaceEmSession& session,CoordinateSystemPtr cs,
      Fallback& fallback,Output& output,std::size_t batch,RegionOf regionOf)
      :session_(session),cs_(cs),fallback_(fallback),output_(output),batch_(batch),regionOf_(std::move(regionOf)) {
    if(!batch_)throw std::invalid_argument("zero interface router batch");
  }
  template<class P> bool canRoute(P const& p,std::uint64_t)const {
    return p.getPID()==Code::Photon||p.getPID()==Code::Electron||p.getPID()==Code::Positron;
  }
  template<class P> void stage(P const& p,transport::HistoryId h,transport::HistoryId parent,
      transport::Generation generation,transport::StepId step) {
    auto state=gpu::em::router_detail::toDeviceState(p,cs_,h,parent,generation,step);
    if(terrain::canonicalizeCpuDirection(state))++stats_.canonicalized;
    state.medium_id=regionOf_(p.getNode());enqueue(state);
  }
  bool pending()const{return !queue_.empty();}
  bool readyForScalarInterleave()const{return queue_.size()>=batch_;}
  std::size_t advanceOneWavefrontAndReturn(Stack& stack) {
    std::vector<gpu::em::EmParticleState> input;
    while(!queue_.empty()&&input.size()<batch_){input.push_back(queue_.front());queue_.pop_front();}
    if(input.empty())return 0;
    auto first=stack.reserveTransportHistoryIds(3*input.size());
    auto records=session_.advance(input,first);std::size_t returned=0;++stats_.batches;
    for(auto const& r:records) {
      ++stats_.advanced;if(r.has_track)output_.recordDeviceStep(r);
      switch(r.outcome) {
        case EmOutcome::Continuation:enqueue(r.end);break;
        case EmOutcome::Children:
          for(std::uint32_t i=0;i<r.child_count;++i)enqueue(r.children[i]);
          break;
        case EmOutcome::Cut:++stats_.cuts;break;
        case EmOutcome::Escape:++stats_.escaped;break;
        case EmOutcome::WindowEscape:output_.recordDeviceWindow(r);++stats_.windowEscaped;break;
        case EmOutcome::Fallback:
          if(!r.has_track||!fallback_.canHandle(r.fallback)) {
            std::ostringstream error;error<<std::setprecision(17)
              <<"unsupported interface fallback before a valid vertex: reason="
              <<static_cast<int>(r.fallback.reason)<<", status="<<r.fallback.diagnostic_status
              <<", pid="<<r.start.pid<<", E_GeV="<<r.start.energy_GeV
              <<", region="<<r.start.medium_id<<", history="<<r.start.history_id
              <<", step="<<r.start.step_id<<", has_track="<<r.has_track;
            throw std::runtime_error(error.str());
          }
          fallback_.handle(stack,r.fallback);++stats_.fallbacks;++returned;break;
        default:throw std::runtime_error("unhandled interface wavefront outcome");
      }
    }
    return returned;
  }
  InterfaceRouterStatistics const& statistics()const{return stats_;}
  std::size_t pendingParticles()const{return queue_.size();}
 protected:
  InterfaceEmSession& session_;
 private:
  void enqueue(gpu::em::EmParticleState const& p) {
    if(!std::isfinite(p.energy_GeV)||p.energy_GeV<0.||!std::isfinite(p.weight)||p.weight<0.)
      throw std::runtime_error("invalid interface secondary");
    if(p.weight==0.)return;
    if(queue_.size()>=200000)throw std::runtime_error("interface host-front limit exceeded; output incomplete");
    queue_.push_back(p);stats_.peak=std::max(stats_.peak,queue_.size());
  }
  CoordinateSystemPtr cs_;Fallback& fallback_;Output& output_;
  std::size_t batch_;RegionOf regionOf_;std::deque<gpu::em::EmParticleState> queue_;
  InterfaceRouterStatistics stats_;
};
} // namespace corsika::interfaces

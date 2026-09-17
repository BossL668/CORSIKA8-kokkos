/* HybridCascade adapter for a single-execution-space resident interface FIFO.
 * CPU owns the scalar stack. Only newly routed CPU particles are staged here;
 * survivors and electromagnetic children never enter this host staging vector. */
#pragma once
#include <corsika/modules/transport/InterfaceEmRouter.hpp>

namespace corsika::interfaces {
template<class Stack, class Fallback, class Output, class RegionOf>
class ResidentInterfaceEmRouter {
 public:
  ResidentInterfaceEmRouter(InterfaceEmSession& session, CoordinateSystemPtr cs,
      Fallback& fallback, Output& output, std::size_t batch, RegionOf regionOf)
      : session_(session), cs_(cs), fallback_(fallback), output_(output),
        batch_(batch), regionOf_(std::move(regionOf)) {
    if (!batch_ || !session_.residentCapacity())
      throw std::invalid_argument("resident interface router requires a bounded device queue");
    staging_.reserve(batch_);
  }
  template<class P> bool canRoute(P const& p, std::uint64_t) const {
    return p.getPID()==Code::Photon || p.getPID()==Code::Electron || p.getPID()==Code::Positron;
  }
  template<class P> void stage(P const& p, transport::HistoryId history,
      transport::HistoryId parent, transport::Generation generation, transport::StepId step) {
    auto state=gpu::em::router_detail::toDeviceState(p,cs_,history,parent,generation,step);
    if (terrain::canonicalizeCpuDirection(state)) ++stats_.canonicalized;
    state.medium_id=regionOf_(p.getNode());
    if (pendingParticles()>=session_.residentCapacity())
      throw std::length_error("CPU routing exceeds resident interface capacity");
    staging_.push_back(state);
    stats_.peak=std::max(stats_.peak,staging_.size());
    if (staging_.size()==batch_) flush();
  }
  bool pending() const { return pendingParticles()!=0; }
  bool readyForScalarInterleave() const { return pendingParticles()>=batch_; }
  std::size_t pendingParticles() const { return staging_.size()+session_.pendingParticles(); }
  InterfaceRouterStatistics const& statistics() const { return stats_; }

  std::size_t advanceOneWavefrontAndReturn(Stack& stack) {
    flush();
    // Preserve the scalar scheduler's existing interleave point: when CPU work
    // remains, yield as soon as the device front falls below one full batch.
    auto result=session_.runResidentCascade(
        [&](std::size_t slots){return stack.reserveTransportHistoryIds(slots);},
        stack.isEmpty()?1:batch_);
    std::size_t returned=0;
    stats_.batches+=result.wavefronts;
    // Optional output-only preparation. Its guard retains cached audit results
    // through the ordered callbacks and releases them even if a callback fails.
    [[maybe_unused]] auto outputBatch=prepareOutput(output_,result.records,0);
    for (auto const& r:result.records) {
      ++stats_.advanced;
      if (r.has_track) output_.recordDeviceStep(r);
      switch (r.outcome) {
        case EmOutcome::Continuation:
        case EmOutcome::Children: break; // Already retained on the execution space.
        case EmOutcome::Cut: ++stats_.cuts; break;
        case EmOutcome::Escape: ++stats_.escaped; break;
        case EmOutcome::WindowEscape:
          output_.recordDeviceWindow(r); ++stats_.windowEscaped; break;
        case EmOutcome::DomainEscape:
          output_.recordDeviceDomain(r); ++stats_.domainEscaped; break;
        case EmOutcome::Fallback:
          if (!r.has_track || !fallback_.canHandle(r.fallback)) {
            std::ostringstream error;
            error<<"unsupported resident interface fallback: reason="
                 <<static_cast<int>(r.fallback.reason)<<", pid="<<r.start.pid
                 <<", region="<<r.start.medium_id<<", history="<<r.start.history_id
                 <<", step="<<r.start.step_id<<", has_track="<<r.has_track;
            throw std::runtime_error(error.str());
          }
          fallback_.handle(stack,r.fallback); ++stats_.fallbacks; ++returned; break;
        default: throw std::runtime_error("unhandled resident interface outcome");
      }
    }
    return returned;
  }
 protected:
  InterfaceEmSession& session_;
 private:
  struct NoOutputBatch {};
  template<class O,class R>
  static auto prepareOutput(O& output,R const& records,int)
      -> decltype(output.prepareDeviceBatch(records)) {
    return output.prepareDeviceBatch(records);
  }
  template<class O,class R>
  static NoOutputBatch prepareOutput(O&,R const&,long) { return {}; }
  void flush() {
    if (staging_.empty()) return;
    session_.submit(staging_);
    staging_.clear();
  }
  CoordinateSystemPtr cs_;
  Fallback& fallback_;
  Output& output_;
  std::size_t batch_;
  RegionOf regionOf_;
  std::vector<gpu::em::EmParticleState> staging_;
  InterfaceRouterStatistics stats_;
};
} // namespace corsika::interfaces

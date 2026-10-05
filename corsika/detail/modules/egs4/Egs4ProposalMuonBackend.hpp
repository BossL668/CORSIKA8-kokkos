#pragma once
#include <corsika/detail/modules/egs4/Egs4MuonAudit.hpp>
#include <corsika/accelerator/em/IAcceleratedEmBackend.hpp>
#include <algorithm>
#include <stdexcept>
#include <utility>

namespace c7_egs4::application {
// Host-boundary adapter only: all muon physics remains in the existing C8
// backend. A single wave is essential: its freshly produced e+/e-/gamma must
// reach EGS4 BEFORE another PROPOSAL transport step. Cross-species retention
// must be disabled in the underlying backend's initialization config.
class ProposalMuonBackend final : public ::corsika::accelerator::em::IAcceleratedEmBackend {
  using Base=::corsika::accelerator::em::IAcceleratedEmBackend;
  using Particle=::corsika::gpu::em::EmParticleState;
  Base& backend_;
  MuonAudit audit_;
  std::vector<Particle> electromagnetic_;
  std::uint64_t waves_{},steps_{},transferred_{};
  static bool muon(Particle const& p){return p.pid==13||p.pid==-13;}
  static bool em(Particle const& p){return p.pid==11||p.pid==-11||p.pid==22;}
  void separate(std::vector<Particle>& particles) {
    std::vector<Particle> retained;retained.reserve(particles.size());
    for(auto const& p:particles) {
      if(muon(p))retained.push_back(p);
      else if(em(p)) {
        if(p.step_id!=0)throw std::logic_error("PROPOSAL transported an EGS4 daughter before handoff");
        audit_.particle("direct_em_handoff",p);
        electromagnetic_.push_back(p);++transferred_;
      }else throw std::logic_error("Unexpected species in PROPOSAL muon output");
    }
    particles=std::move(retained);
  }
  void requireEmptyDeviceQueues()const {
    if(backend_.pendingPhotonCount()||backend_.pendingLeptonCount())
      throw std::logic_error("Muon-only adapter requires resident_cross_species=false");
  }
public:
  explicit ProposalMuonBackend(Base& backend):backend_(backend) {
    if(!backend.capabilities().muon_transport)throw std::invalid_argument("PROPOSAL muon transport unavailable");
    if(backend.gpuRadioEnabled())throw std::invalid_argument("Muon adapter must not own EM radio output");
    if(backend.gpuProfileEnabled())throw std::invalid_argument("Muon adapter currently streams full steps to the C8 writers");
    requireEmptyDeviceQueues();
  }
  std::vector<Particle> takeElectromagnetic(){return std::exchange(electromagnetic_,{});}
  bool hasElectromagnetic()const{return !electromagnetic_.empty();}
  std::uint64_t muonWaves()const{return waves_;}
  std::uint64_t muonSteps()const{return steps_;}
  std::uint64_t emTransfers()const{return transferred_;}
  void auditScalarHandoff(Particle const& p){audit_.particle("scalar_em_handoff",p);}
  void beginShower(::corsika::accelerator::em::AcceleratedEmShowerConfig const& c)override {
    if(hasElectromagnetic())throw std::logic_error("Undrained muon-to-EGS4 handoff");
    backend_.beginShower(c);waves_=steps_=transferred_=0;requireEmptyDeviceQueues();
  }
  bool canTransport(Particle const& p)const override{return muon(p)&&backend_.canTransport(p);}
  bool hasProposalTable()const override{return backend_.hasProposalTable();}
  // The EM router's small-batch CPU expansion cannot grow a muon population.
  // Disable that scheduling heuristic here, not any physical fallback.
  std::size_t minimumBatchSize()const override{return 1;}
  std::size_t maximumResidentPhotonBatchSize()const override{return backend_.maximumResidentPhotonBatchSize();}
  std::size_t maximumResidentLeptonBatchSize()const override{return backend_.maximumResidentLeptonBatchSize();}
  std::size_t maximumResidentInputBatchSize()const override{return backend_.maximumResidentInputBatchSize();}
  std::size_t pendingPhotonCount()const noexcept override{return backend_.pendingPhotonCount();}
  std::size_t pendingLeptonCount()const noexcept override{return backend_.pendingLeptonCount();}
  ::corsika::gpu::em::ResidentPhotonCascadeResult runPhotonWavefront(
      std::vector<Particle> const&,std::uint64_t,std::size_t,std::size_t)override {
    throw std::logic_error("Photons belong to EGS4, not the PROPOSAL muon backend");
  }
  ::corsika::gpu::em::ResidentLeptonCascadeResult runLeptonWavefront(
      std::vector<Particle> const& input,std::uint64_t first,std::size_t waves,
      std::uint64_t limit,std::size_t minimum)override {
    requireEmptyDeviceQueues();
    if(!waves)throw std::invalid_argument("Muon wave count must be positive");
    for(auto const& p:input)if(!muon(p))throw std::invalid_argument("Only muons may enter PROPOSAL transport");
    for(auto const& p:input)audit_.particle("muon_input",p);
    auto result=backend_.runLeptonWavefront(input,first,1,limit,minimum);
    audit_.wave(result);
    requireEmptyDeviceQueues();
    if(result.wavefronts>1)throw std::logic_error("Muon backend exceeded the single-wave handoff boundary");
    for(auto const& r:result.step_records)if(!muon(r.start))
      throw std::logic_error("Non-muon PROPOSAL step in EGS4 transport");
    separate(result.generated_photons);separate(result.remaining_leptons);separate(result.cpu_spill_particles);
    result.completed=result.remaining_leptons.empty();
    waves_+=result.wavefronts;steps_+=result.transport_records;
    return result;
  }
  ::corsika::accelerator::em::BackendCapabilities capabilities()const override {
    auto c=backend_.capabilities();c.photon_transport=c.electron_transport=c.positron_transport=false;return c;
  }
  ::corsika::accelerator::em::AcceleratedEmStatistics const& statistics()const override{return backend_.statistics();}
  bool gpuProfileEnabled()const noexcept override{return backend_.gpuProfileEnabled();}
  ::corsika::gpu::em::GpuProfileResult downloadProfile()override{return backend_.downloadProfile();}
  bool gpuRadioEnabled()const noexcept override{return false;}
  ::corsika::gpu::radio::GpuRadioWaveforms downloadRadioWaveforms()override{
    throw std::logic_error("EGS4 owns radio finalization");
  }
  std::optional<::corsika::gpu::em::GpuFirstInteractionSnapshot> downloadFirstInteractionSnapshot()override {
    return backend_.downloadFirstInteractionSnapshot();
  }
};
// Optional hook lets the frozen CPU-muon configuration use the same native
// router without constructing any additional PROPOSAL device resources.
struct NoAcceleratedMuons {
  template<class P,class S>bool canRoute(P const&,S){return false;}
  template<class... Args>void stage(Args&&...){throw std::logic_error("Muon router disabled");}
  bool pending()const{return false;}
  bool readyForScalarInterleave()const{return false;}
  template<class S>std::size_t advanceOneWavefrontAndReturn(S&){return 0;}
  template<class H,class S>bool consumeForcedDecay(H,S){return false;}
  void endOfShower(){}
};
template<class Sink>struct MuonOutputSink {
  Sink& sink;
  explicit MuonOutputSink(Sink& output):sink(output){}
  template<class T>void onStep(T const& r){sink.onStep(r);}
  template<class T>void onObservation(T const& r){sink.onObservation(r);}
  template<class T>void onRadioTrack(T const& r){sink.onRadioTrack(r);}
  // This harness has no first-interaction/production diagnostic writer.
  template<class T>void onFirstInteraction(T const&){}
  template<class T>void onProjectedStep(T const&){throw std::logic_error("Muon projected output was not enabled");}
  template<class T>void onGpuProfile(T const&){throw std::logic_error("Muon resident profile was not enabled");}
  template<class T>void onGpuRadioWaveforms(T const&,std::uint64_t){throw std::logic_error("EGS4 owns radio finalization");}
};
} // namespace c7_egs4::application

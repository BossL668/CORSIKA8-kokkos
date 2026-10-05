#include "Egs4C8Session.hpp"
#include "Egs4C8AirWavefront.hpp"
#include "Egs4C8DeviceRadio.hpp"
#include "Egs4C8DeviceProfile.hpp"
#include "Egs4C8HostOutput.hpp"
#include <type_traits>
#ifdef KOKKOS_ENABLE_CUDA
#include <cuda_runtime_api.h>
#endif

namespace c7_egs4::application {
namespace {
struct RuntimeLease {
  bool owns_runtime{};
  RuntimeLease() {
    if(Kokkos::is_finalized())throw std::runtime_error("Cannot create EGS4 session after Kokkos finalization");
    owns_runtime=!Kokkos::is_initialized();if(owns_runtime)Kokkos::initialize();
  }
  ~RuntimeLease(){if(owns_runtime&&Kokkos::is_initialized()&&!Kokkos::is_finalized())Kokkos::finalize();}
};
// Sessions are managed on the application's main host thread. Share runtime
// ownership so destroying the first session cannot invalidate another one.
std::shared_ptr<RuntimeLease> acquireRuntime() {
  if(Kokkos::is_finalized())throw std::runtime_error("Cannot create EGS4 session after Kokkos finalization");
  static std::weak_ptr<RuntimeLease> live;
  auto lease=live.lock();
  if(!lease){lease=std::make_shared<RuntimeLease>();live=lease;}
  return lease;
}
RareMaterial rareMaterial(Configuration const& c) {
  return {c.electron_mass_MeV,c.muon_mass_MeV,c.charged_pion_mass_MeV,c.proton_mass_MeV,c.average_atomic_weight,
    {c.air_composition[0],c.air_composition[1],c.air_composition[2]}};
}
c8_adapter::PhotonuclearPolicy photonuclearPolicy(Configuration const& c) {
  return {c.native_photonuclear_vertices,
    {c.proton_mass_MeV,c.neutron_mass_MeV,c.neutral_pion_mass_MeV,c.charged_pion_mass_MeV,
     c.phi_mass_MeV,c.omega_mass_MeV,c.rho_mass_MeV},makeTransverseMomentumTable(),c.native_prompt_rho_decay,
     c.native_prompt_resonance_decay,{c.omega_mass_MeV,c.phi_mass_MeV,c.charged_pion_mass_MeV,
       c.neutral_pion_mass_MeV,c.muon_mass_MeV,c.charged_kaon_mass_MeV,c.long_kaon_mass_MeV,c.short_kaon_mass_MeV,c.eta_mass_MeV}};
}
struct Stream {
  bool failed{};
  virtual ~Stream()=default;
  virtual void append(std::vector<::corsika::gpu::em::EmParticleState> const&)=0;
  virtual std::size_t active()const=0;
  virtual WaveResult advance(std::uint64_t,std::uint64_t)=0;
  virtual RunStatistics statistics()const=0;
  virtual void finish()=0;
  virtual void restart(std::uint64_t,std::uint64_t,OutputCallbacks const&)=0;
};
template<class Exec>struct NativeStream final:Stream {
  c8_adapter::AirWavefront<Exec> queue;
  OutputCallbacks output;
  DeviceRadio<Exec> radio;
  DeviceProfile<Exec> profile;
  HostOutput<Exec> host_output;
  RunStatistics stats;
  int maximum_waves;
  bool needs_prompt_decay;
  NativeStream(Tables const& tables,Configuration const& c,OutputCallbacks const& callbacks):
    queue(tables,{c.electron_mass_MeV,c.electronuclear_threshold_MeV,c.photonuclear_threshold_MeV,c.muon_pair_threshold_MeV},
      c8_adapter::makeAirEnvironment(c.environment,c.earth_radius_m,c.sterncor,c8_adapter::AirConvention::c7_egs4_four_exponentials),
      {c.electron_total_cut_MeV,c.photon_cut_MeV,c.stepfc},c.queue_capacity,c.resolve_rare_vertices,
      rareMaterial(c),photonuclearPolicy(c),c.thinning,c.reuse_queue_workspace,c.split_rare_kernels),output(callbacks),radio(c.radio,output),
      profile(c.profile,output),maximum_waves(c.maximum_waves),
      needs_prompt_decay(c.native_prompt_rho_decay||c.native_prompt_resonance_decay) {
    if(maximum_waves<1||!output.step||!output.radio||!output.observation||!output.discarded||
       ((c.native_prompt_rho_decay||c.native_prompt_resonance_decay)&&!output.prompt_decay))
      throw std::invalid_argument("Native EGS4 stream requires all output consumers and bounded waves");
    queue.beginShower({},c.seed,c.shower_id,1);stats.execution_space=Exec::name();
  }
  void append(std::vector<::corsika::gpu::em::EmParticleState> const& input)override {
    queue.append(input);stats.injections+=input.size();
  }
  std::size_t active()const override{return queue.activeCount();}
  RunStatistics statistics()const override{return stats;}
  void restart(std::uint64_t seed,std::uint64_t shower,OutputCallbacks const& next)override {
    if(!next.step||!next.radio||!next.observation||!next.discarded||(needs_prompt_decay&&!next.prompt_decay)||
       (radio.enabled()&&!next.radio_waveforms)||(profile.enabled()&&!next.profile))
      throw std::invalid_argument("Persistent native EGS4 event is missing an output consumer");
    queue.resetForShower(seed,shower);output=next;stats={};stats.execution_space=Exec::name();
    radio.reset();profile.reset();
  }
  void finish()override{
    profile.finish(stats);radio.finish(stats);
    stats.queue_workspace_allocations=queue.workspaceAllocations();stats.queue_workspace_reuses=queue.workspaceReuses();
    stats.queue_processed_records=queue.processedRecords();stats.queue_peak_active=queue.peakActive();
    stats.queue_advance_wall_ms=queue.advanceWallMilliseconds();
    stats.queue_append_calls=queue.appendCalls();stats.queue_append_records=queue.appendRecords();
    stats.queue_append_copied_active_records=queue.appendCopiedActiveRecords();
    stats.queue_append_wall_ms=queue.appendWallMilliseconds();
    stats.queue_split_rare_waves=queue.splitRareWaves();stats.queue_split_rare_records=queue.splitRareRecords();
    output={}; // never retain references to the completed event's writers
  }
  WaveResult advance(std::uint64_t first,std::uint64_t count)override {
    if(!active())return {};
    if(stats.waves>=std::uint64_t(maximum_waves))throw std::runtime_error("Native EGS4 stream wave limit reached");
    auto batch=queue.advance(first,count);++stats.waves;stats.children+=batch.created;
    radio.accumulate(batch.outputs,stats);
    profile.accumulate(batch.outputs);
    auto records=host_output.collect(batch,profile.enabled(),radio.enabled(),stats);
    WaveResult result;result.active_particles=active();
    for(std::size_t i=0;i<records.extent(0);++i) {
      auto const& transfer=records(i).transfer;
      result.target_rest_energy_GeV+=transfer.target_rest_energy_MeV*.001*records(i).weight;
      for(int j=0;j<transfer.count;++j) {
        auto const& source=transfer.vertices[j];HostRequest request;
        switch(source.kind) {
          case c8_adapter::HostVertexKind::muon_transport: request.kind=HostRequestKind::muon_transport;break;
          case c8_adapter::HostVertexKind::photonuclear: request.kind=HostRequestKind::photonuclear;break;
          case c8_adapter::HostVertexKind::hadron_transport: request.kind=HostRequestKind::hadron_transport;break;
          case c8_adapter::HostVertexKind::c7_vector_meson_decay: request.kind=HostRequestKind::c7_vector_meson_decay;break;
          case c8_adapter::HostVertexKind::photonuclear_many_hadrons: request.kind=HostRequestKind::photonuclear_many_hadrons;break;
          case c8_adapter::HostVertexKind::native_em: continue; // already resident in native queue
        }
        request.particle=source.particle;request.virtual_photon=source.virtual_photon;
        request.target_atomic_number=source.target_atomic_number;
        request.polarization_cosine=source.polarization_cosine;request.polarization_azimuth=source.polarization_azimuth;
        request.photonuclear_branch=source.photonuclear_branch;request.target_pdg=source.target_pdg;
        request.random_key=source.random.key;request.hadron_random_key=source.hadron_random.key;
        result.host_requests.push_back(request);
      }
    }
    for(std::size_t i=0;i<records.extent(0);++i) {
      auto const& r=records(i).output;
      auto const& transfer=records(i).transfer;
      if(transfer.has_rho_decay||transfer.has_resonance_decay) {
        PromptDecayRecord decay;decay.parent=transfer.has_rho_decay?transfer.rho_parent:transfer.resonance_parent;
        decay.count=transfer.has_rho_decay?2:transfer.resonance_children;
        for(int j=0;j<decay.count;++j) {
          decay.daughters[j]=transfer.vertices[j].particle;
          decay.polarization_cosine[j]=transfer.vertices[j].polarization_cosine;
          decay.polarization_azimuth[j]=transfer.vertices[j].polarization_azimuth;
        }
        output.prompt_decay(decay);++stats.prompt_decays;
      }
      if(r.has_step&&!profile.enabled())output.step(r.step);
      if(r.has_radio&&!radio.enabled())output.radio(r.radio);
      if(r.has_observation)output.observation(r.observation);
      if(r.discard!=c8_adapter::AirDiscard::none) {
        output.discarded(r.discarded_energy_GeV*records(i).weight,static_cast<unsigned>(r.discard));
      }
    }
    stats.host_requests+=result.host_requests.size();stats.target_rest_energy_GeV+=result.target_rest_energy_GeV;return result;
  }
};
}
struct Session::Impl {
  Tables tables;
  std::shared_ptr<RuntimeLease> runtime;
  std::unique_ptr<Stream> stream;
  std::unique_ptr<Stream> retained;
  bool retain_across_showers{};
  explicit Impl(std::string const& path):tables(readAirTables(path)),runtime(acquireRuntime()){}
  explicit Impl(Tables value):tables(std::move(value)),runtime(acquireRuntime()){}
};
Session::Session(std::string const& path):impl_(std::make_unique<Impl>(path)){}
Session::Session(Tables tables):impl_(std::make_unique<Impl>(std::move(tables))){}
Session::~Session()=default;
GpuMemoryBudget Session::gpuMemoryBudget(double fraction,::corsika::gpu::radio::GpuRadioConfig const& radio)const {
  if(impl_->stream||!Kokkos::is_initialized()||Kokkos::is_finalized())
    throw std::logic_error("Plan native EGS4 GPU memory before starting a live session");
#ifdef KOKKOS_ENABLE_CUDA
  if constexpr(std::is_same_v<Kokkos::DefaultExecutionSpace,Kokkos::Cuda>) {
    std::size_t free=0,total=0;
    auto error=cudaMemGetInfo(&free,&total);
    if(error!=cudaSuccess)throw std::runtime_error(std::string("CUDA memory query: ")+cudaGetErrorString(error));
    using namespace c8_adapter;
    // Upper bound at capacity, including temporary next-generation storage.
    constexpr auto payload=3*sizeof(AirRecord)+sizeof(AirOutcome)+sizeof(AirOutput)+
      sizeof(RareAirTransfer)+sizeof(ThinningResult)+3*sizeof(int)+sizeof(HostOutputRecord);
    using Acc=::corsika::accelerator::radio::kokkos_detail::KokkosRadioAccumulator<Kokkos::Cuda>;
    // Retained radio buffers can overlap their replacements when a batch grows.
    auto per_history=payload+(radio.enabled?2*sizeof(::corsika::gpu::em::LeptonTransportRecord)+
      2*sizeof(::corsika::accelerator::radio::detail::RadioTrackKinematics):0);
    // Retained workspaces can overlap their replacements during growth.
    return planGpuMemory(fraction,total,free,2*per_history,
      512ULL*1024*1024+Acc::projectedDeviceBytes(radio));
  }
#endif
  throw std::invalid_argument("GPU memory fraction requires the CUDA execution space");
}
template<class Exec>RunStatistics runCascade(Tables const& tables,Configuration const& c,
    std::vector<::corsika::gpu::em::EmParticleState> const& input,OutputCallbacks const& output) {
  namespace na=c8_adapter;
  if(c.maximum_waves<1||!output.step||!output.radio||!output.observation||!output.discarded||
     ((c.native_prompt_rho_decay||c.native_prompt_resonance_decay)&&!output.prompt_decay))
    throw std::invalid_argument("Native EGS4 session requires bounded waves and all output consumers");
  auto env=na::makeAirEnvironment(c.environment,c.earth_radius_m,c.sterncor,na::AirConvention::c7_egs4_four_exponentials);
  ChannelThresholds thresholds{c.electron_mass_MeV,c.electronuclear_threshold_MeV,c.photonuclear_threshold_MeV,c.muon_pair_threshold_MeV};
  na::AirWavefront<Exec> queue(tables,thresholds,env,
      {c.electron_total_cut_MeV,c.photon_cut_MeV,c.stepfc},c.queue_capacity,
      c.resolve_rare_vertices,rareMaterial(c),photonuclearPolicy(c),c.thinning,c.reuse_queue_workspace,c.split_rare_kernels);
  queue.beginShower(input,c.seed,c.shower_id,c.first_child_id);RunStatistics stats;stats.execution_space=Exec::name();
  DeviceRadio<Exec> radio(c.radio,output);
  DeviceProfile<Exec> profile(c.profile,output);HostOutput<Exec> host_output;
  stats.injections=input.size();
  while(queue.activeCount()) {
    if(++stats.waves>std::uint64_t(c.maximum_waves))throw std::runtime_error("Native EGS4 wave limit reached; shower incomplete");
    auto batch=queue.advance();stats.children+=batch.created;
    radio.accumulate(batch.outputs,stats);
    profile.accumulate(batch.outputs);
    auto records=host_output.collect(batch,profile.enabled(),radio.enabled(),stats);
    // Refuse the unfinished rare-channel integration before emitting this wave.
    for(std::size_t i=0;i<records.extent(0);++i)if(records(i).transfer.count)
      throw std::runtime_error("Native EGS4 rare channel needs a same-model handler; no PROPOSAL substitution or completed-shower claim");
    for(std::size_t i=0;i<records.extent(0);++i) {
      auto const& r=records(i).output;
      if(r.has_step&&!profile.enabled())output.step(r.step);
      if(r.has_radio&&!radio.enabled())output.radio(r.radio);
      if(r.has_observation)output.observation(r.observation);
      if(r.discard!=na::AirDiscard::none) {
        output.discarded(r.discarded_energy_GeV*records(i).weight,static_cast<unsigned>(r.discard));
      }
    }
  }
  profile.finish(stats);radio.finish(stats);
  stats.queue_workspace_allocations=queue.workspaceAllocations();stats.queue_workspace_reuses=queue.workspaceReuses();
  stats.queue_processed_records=queue.processedRecords();stats.queue_peak_active=queue.peakActive();
  stats.queue_advance_wall_ms=queue.advanceWallMilliseconds();
  stats.queue_append_calls=queue.appendCalls();stats.queue_append_records=queue.appendRecords();
  stats.queue_append_copied_active_records=queue.appendCopiedActiveRecords();
  stats.queue_append_wall_ms=queue.appendWallMilliseconds();
  stats.queue_split_rare_waves=queue.splitRareWaves();stats.queue_split_rare_records=queue.splitRareRecords();
  return stats;
}
RunStatistics Session::run(Configuration const& c,std::vector<::corsika::gpu::em::EmParticleState> const& input,
    OutputCallbacks const& callbacks,bool host_reference) {
  if(!Kokkos::is_initialized()||Kokkos::is_finalized())
    throw std::runtime_error("Native EGS4 session requires a live Kokkos runtime");
  if(impl_->stream||impl_->retained)throw std::logic_error("Finish/release the incremental EGS4 session before a standalone run");
  if(host_reference)return runCascade<Kokkos::DefaultHostExecutionSpace>(impl_->tables,c,input,callbacks);
  return runCascade<Kokkos::DefaultExecutionSpace>(impl_->tables,c,input,callbacks);
}
void Session::begin(Configuration const& c,OutputCallbacks const& output,bool host_reference) {
  if(!Kokkos::is_initialized()||Kokkos::is_finalized())throw std::runtime_error("Native EGS4 runtime is not live");
  if(impl_->stream||impl_->retained)throw std::logic_error("Native EGS4 stream already started/retained; use resume with unchanged configuration");
  if(host_reference)impl_->stream=std::make_unique<NativeStream<Kokkos::DefaultHostExecutionSpace>>(impl_->tables,c,output);
  else impl_->stream=std::make_unique<NativeStream<Kokkos::DefaultExecutionSpace>>(impl_->tables,c,output);
  impl_->retain_across_showers=c.retain_across_showers;
}
bool Session::hasRetainedShower()const{return bool(impl_->retained);}
void Session::resume(std::uint64_t seed,std::uint64_t shower,OutputCallbacks const& output) {
  if(impl_->stream||!impl_->retained||impl_->retained->failed||!Kokkos::is_initialized()||Kokkos::is_finalized())
    throw std::logic_error("No completed native EGS4 state available for reuse");
  impl_->stream=std::move(impl_->retained);
  try{impl_->stream->restart(seed,shower,output);}catch(...){impl_->stream->failed=true;throw;}
}
void Session::append(std::vector<::corsika::gpu::em::EmParticleState> const& input) {
  if(!impl_->stream||impl_->stream->failed||!Kokkos::is_initialized()||Kokkos::is_finalized())
    throw std::logic_error("No usable native EGS4 stream");
  impl_->stream->append(input); // rejected injection leaves the stream unchanged
}
std::size_t Session::activeParticles()const {
  if(!impl_->stream||impl_->stream->failed||!Kokkos::is_initialized()||Kokkos::is_finalized())
    throw std::logic_error("No usable native EGS4 stream");
  return impl_->stream->active();
}
RunStatistics Session::progressStatistics()const {
  if(!impl_->stream||impl_->stream->failed)throw std::logic_error("No usable native EGS4 stream");
  return impl_->stream->statistics();
}
WaveResult Session::advance(std::uint64_t first,std::uint64_t count) {
  if(!impl_->stream||impl_->stream->failed||!Kokkos::is_initialized()||Kokkos::is_finalized())
    throw std::logic_error("No usable native EGS4 stream");
  try{return impl_->stream->advance(first,count);}
  catch(...){impl_->stream->failed=true;throw;} // partial output callbacks cannot be replayed
}
RunStatistics Session::finishEMQueue() {
  if(!impl_->stream||impl_->stream->failed||!Kokkos::is_initialized()||Kokkos::is_finalized()||impl_->stream->active())
    throw std::logic_error("Native EGS4 queue is absent, failed or still active");
  try{impl_->stream->finish();}catch(...){impl_->stream->failed=true;throw;}
  auto result=impl_->stream->statistics();
  if(impl_->retain_across_showers)impl_->retained=std::move(impl_->stream);
  else impl_->stream.reset();
  return result;
}
} // namespace c7_egs4::application

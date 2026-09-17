// Independent material-interface execution owner. Air backends do not link this library.
#include <Kokkos_Core.hpp>
#include <corsika/modules/transport/InterfaceEmSession.hpp>
#include <corsika/modules/transport/detail/InterfaceEmStep.hpp>
#include <corsika/modules/transport/kokkos/KokkosInterfaceQueue.hpp>
#include <corsika/modules/transport/kokkos/ExecutionSpace.hpp>
#include <corsika/modules/radio/interface/KokkosAccumulator.hpp>
#include <corsika/geometry/terrain/TerrainMagneticAccuracy.hpp>
#include <corsika/geometry/terrain/KokkosDemCoverage.hpp>
namespace corsika::interfaces {
namespace kd = accelerator::em::kokkos_detail;
using Space = kokkos::ExecutionSpace;
using Memory = Space::memory_space;
using terrain::FlatTerrainData;
using terrain::KokkosTerrainSession;
#ifdef C8_TERRAIN_STEP_AUDIT
namespace audit = terrain::audit;
#endif
class InterfaceEmSession::Impl {
 public:
  // Declared first, destroyed last: Views must be released before finalize().
  Kokkos::ScopeGuard runtime;
  Space execution;
  kokkos::InterfaceQueue<Space> resident;
  ResidentStatistics resident_stats;
  KokkosTerrainSession<Space> terrain;
  terrain::coverage::Data host_coverage;
  terrain::coverage::KokkosData<Space> coverage;
  std::vector<kd::KokkosProposalNativeTable<Space>> tables;
  std::vector<kd::KokkosMoliereInterpolation<Space>> moliere;
  Kokkos::View<kd::KokkosPhysicsContextData*,Memory> materials;
  Kokkos::View<em::EmParticleState*,Memory> input;
  Kokkos::View<EmStep*,Memory> output;
  decltype(input)::HostMirror host_input;
  decltype(output)::HostMirror host_output;
  decltype(output) resident_records;
  decltype(output)::HostMirror host_resident_records;
  std::size_t capacity{},bytes{};
  bool cascade_running{};
  double maximum_step_m,transport_window_s;
  bool energy_ledger{};
  interfaces::MaterialInterface interface;
  std::vector<std::vector<em::tables::NativeInteractionIdentity>> identities;
  using Radio=corsika::radio::interface::KokkosAccumulator<Space,true>;
  std::unique_ptr<Radio> radio;
  Radio::Tracks radio_tracks;
#ifdef C8_TERRAIN_STEP_AUDIT
  // Bounded diagnostic mirrors only. Never present in production builds.
  FlatTerrainData audit_mesh;
  std::vector<EmMaterial> audit_banks;
  std::vector<em::MoliereInterpolationTable> audit_moliere;
  Kokkos::View<kd::KokkosPhysicsContextData*,Kokkos::HostSpace> audit_materials;
  Kokkos::View<audit::Record*,Memory> audit_device;
  Kokkos::View<audit::Record*,Kokkos::HostSpace> audit_host,audit_reference;
  Kokkos::View<EmStep*,Kokkos::HostSpace> audit_result;
  audit::Writer audit_writer;
#endif
  void validateParticle(em::EmParticleState const& p, bool allowZeroWeight=false) const {
    double norm=0.;
    for(int a=0;a<3;++a) {
      if(!std::isfinite(p.position_m[a])||!std::isfinite(p.direction[a]))
        throw std::invalid_argument("nonfinite interface particle");
      norm+=p.direction[a]*p.direction[a];
    }
    if((p.pid!=22&&p.pid!=11&&p.pid!=-11)||!interface.containsRegion(p.medium_id)||p.history_id==0||
        p.step_id==UINT64_MAX||!std::isfinite(p.energy_GeV)||p.energy_GeV<0.||
        !std::isfinite(p.time_s)||!std::isfinite(p.weight)||p.weight<0.||
        (!allowZeroWeight&&p.weight==0.)||std::abs(norm-1.)>1.e-12)
      throw std::invalid_argument("invalid routed interface particle");
  }
  static void validateHistory(std::size_t count, std::uint64_t first) {
    if(first==0 || first>UINT64_MAX-3*count)
      throw std::invalid_argument("interface child-history range overflow");
  }
  static std::size_t workCapacity(EmConfig const& c) {
    return c.resident_capacity?std::max(c.batch_size,c.resident_wavefront_capacity):c.batch_size;
  }
  void finishRecord(EmStep& r) const {
    if(auto error=detail::recordError(r,interface))
      throw std::runtime_error("interface device record error "+std::to_string(error));
    if(r.outcome==EmOutcome::Fallback) {
      r.fallback.particle.medium_id=r.start.medium_id;
      auto const& ids=identities[interface.material(r.start.medium_id)];
      auto found=std::find_if(ids.begin(),ids.end(),[&](auto const& id){return id.pdg_id==r.fallback.particle.pid;});
      if(found==ids.end())throw std::runtime_error("missing actual interface fallback calculator identity");
      r.fallback.medium_hash=found->medium_hash;r.fallback.interaction_hash=found->interaction_hash;
    }
  }
  static Kokkos::InitializationSettings settings(EmConfig const& c,
      FlatTerrainData const& mesh,std::vector<EmMaterial> const& banks) {
    if(Kokkos::is_initialized())throw std::logic_error("interface transport requires its own independent runtime");
    auto work=workCapacity(c);
    if(c.threads<1||c.device<0||c.batch_size<1||c.batch_size>65536||
       c.resident_wavefront_capacity>65536||c.maximum_device_bytes<1024 ||
       c.resident_capacity > (std::size_t{1} << 24))
      throw std::invalid_argument("invalid interface runtime settings");
    if(c.resident_capacity && (c.resident_record_capacity<work ||
        c.resident_record_capacity>(std::size_t{1}<<20)))
      throw std::invalid_argument("invalid interface resident record capacity");
    if(!(c.maximum_step_m>0.)||!(c.transport_window_s>0.)||
        (std::isfinite(c.transport_window_s)&&c.transport_window_s>1.e-3))
      throw std::invalid_argument("invalid terrain diagnostic step/window");
    if (!std::isfinite(c.emcut_MeV) || !(c.emcut_MeV > 0.))
      throw std::invalid_argument("invalid interface transport cut");
    c.interface.validate(banks.size());
    for(auto const& bank:banks) {
      if(!em::atmosphere_detail::validEnvironment(bank.environment))
        throw std::invalid_argument("unsupported interface material density/geometry/field snapshot");
      for(double b:bank.environment.magnetic_field_T)
        if(!std::isfinite(b))throw std::invalid_argument("nonfinite interface magnetic field");
    }
    auto projected=mesh.bytes()+c.coverage.bytes()+1024+work*(sizeof(em::EmParticleState)+sizeof(EmStep))+
        banks.size()*(sizeof(kd::KokkosPhysicsContextData)+kd::KokkosMoliereInterpolation<Space>::projectedDeviceBytes());
    for(auto const& bank:banks)projected+=kd::KokkosProposalNativeTable<Space>::projectedDeviceBytes(bank.table);
    projected += kokkos::InterfaceQueue<Space>::peakWorkspaceBytes(c.resident_capacity,work);
    if(c.resident_capacity) projected+=c.resident_record_capacity*sizeof(EmStep);
    if(c.radio.enabled)projected+=Radio::projectedBytes(c.radio,work,mesh.nodes.size())+
      work*sizeof(corsika::radio::interface::Track);
    if(projected>c.maximum_device_bytes)throw std::runtime_error(
        "interface pre-initialization memory gate requires "+std::to_string(projected)+" bytes");
    Kokkos::InitializationSettings s;s.set_num_threads(c.threads);s.set_device_id(c.device);return s;
  }
  Impl(FlatTerrainData const& mesh,std::vector<EmMaterial> const& banks,EmConfig const& c)
      :runtime(settings(c,mesh,banks)),resident(c.resident_capacity,workCapacity(c),execution),
       terrain(mesh,1),host_coverage(c.coverage),coverage(c.coverage),capacity(workCapacity(c)),
       maximum_step_m(c.maximum_step_m),transport_window_s(c.transport_window_s),energy_ledger(c.energy_ledger),interface(c.interface),
       identities(banks.size()) {
    tables=decltype(tables)(banks.size());moliere=decltype(moliere)(banks.size());
    bytes=terrain.geometryBytes()+terrain.workspaceBytes()+c.coverage.bytes()+capacity*(sizeof(em::EmParticleState)+sizeof(EmStep))+
        banks.size()*sizeof(kd::KokkosPhysicsContextData);
    for(auto const& bank:banks) {
      for(double b:bank.environment.magnetic_field_T)
        if(!std::isfinite(b))throw std::invalid_argument("nonfinite terrain magnetic field");
      bytes+=kd::KokkosProposalNativeTable<Space>::projectedDeviceBytes(bank.table);
    }
    if(bytes+banks.size()*kd::KokkosMoliereInterpolation<Space>::projectedDeviceBytes()>c.maximum_device_bytes)
      throw std::runtime_error("interface immutable data exceed the validated memory budget");
    materials=decltype(materials)("terrain_material_contexts",banks.size());
    auto host=Kokkos::create_mirror_view(materials);
    for(std::size_t i=0;i<banks.size();++i) {
      identities[i]=banks[i].table.interaction_identities;
      tables[i].initialize(banks[i].table,c.maximum_device_bytes);
      auto path=banks[i].auxiliary.cache_file;if(!path.empty())path+=".moliere-initial-v1.c8cache";
      moliere[i].initialize(banks[i].auxiliary.electron_moliere,path);
      bytes+=moliere[i].deviceBytes();
      auto& d=host(i);d.physics.proposal_native=tables[i].deviceView();d.physics.physics_source=1;
      d.physics.em_transport_cut_MeV=c.emcut_MeV;d.environment=banks[i].environment;
      d.environment.maximum_magnetic_deflection_rad=std::min(
          d.environment.maximum_magnetic_deflection_rad,terrain::MaximumQuadraticMagneticDeflection);
      d.electron_moliere=banks[i].auxiliary.electron_moliere;
      d.moliere_interpolation=moliere[i].deviceView();d.photon_pair_lpm=banks[i].auxiliary.photon_pair_lpm;
      d.brems_lpm=banks[i].auxiliary.brems_lpm;d.thinning=c.thinning;d.random_seed=c.seed;d.shower_id=c.shower_id;
    }
    bytes += kokkos::InterfaceQueue<Space>::workspaceBytes(c.resident_capacity,capacity);
    if(c.resident_capacity) bytes+=c.resident_record_capacity*sizeof(EmStep);
    if(c.radio.enabled)bytes+=Radio::projectedBytes(c.radio,capacity,mesh.nodes.size())+
      capacity*sizeof(corsika::radio::interface::Track);
    if(bytes + c.resident_capacity*sizeof(em::EmParticleState)>c.maximum_device_bytes)throw std::runtime_error("interface memory budget exceeded");
    Kokkos::deep_copy(materials,host);
    input=decltype(input)("terrain_step_input",capacity);output=decltype(output)("terrain_step_output",capacity);
    host_input=Kokkos::create_mirror_view(input);host_output=Kokkos::create_mirror_view(output);
    if(c.resident_capacity) {
      resident_records=decltype(output)("interface_resident_records",c.resident_record_capacity);
      host_resident_records=Kokkos::create_mirror_view(resident_records);
    }
    if(c.radio.enabled){
      radio=std::make_unique<Radio>(c.radio,terrain.deviceView(),mesh.triangles.size(),capacity,execution);
      radio_tracks=Radio::Tracks("interface_radio_transport_tracks",capacity);
    }
    execution.fence("terrain material initialization");
#ifdef C8_TERRAIN_STEP_AUDIT
    audit_mesh=mesh;audit_banks=banks;
    audit_materials=decltype(audit_materials)("audit_materials",banks.size());
    audit_moliere.resize(banks.size());
    for(std::size_t i=0;i<banks.size();++i) {
      auto& d=audit_materials(i);d=host(i);
      d.physics.proposal_native=em::tables::makeProposalNativeHostView(audit_banks[i].table);
      auto path=banks[i].auxiliary.cache_file;if(!path.empty())path+=".moliere-initial-v1.c8cache";
      audit_moliere[i]=em::loadOrMakeMoliereInterpolationTable(banks[i].auxiliary.electron_moliere,path);
      d.moliere_interpolation=em::makeMoliereInterpolationView(
          audit_moliere[i].polynomials.data(),audit_moliere[i].initial_guess_delta.data());
    }
    audit_device=decltype(audit_device)("audit_device",capacity);
    audit_host=decltype(audit_host)("audit_host",capacity);
    audit_reference=decltype(audit_reference)("audit_reference",capacity);
    audit_result=decltype(audit_result)("audit_result",capacity);
#endif
  }
};
InterfaceEmSession::InterfaceEmSession(FlatTerrainData const& m,std::vector<EmMaterial> const& b,EmConfig const& c)
    :impl_(std::make_unique<Impl>(m,b,c)){}
InterfaceEmSession::~InterfaceEmSession()=default;
std::size_t InterfaceEmSession::deviceBytes()const{return impl_->bytes;}
std::string InterfaceEmSession::executionSpace()const{return Space::name();}
int InterfaceEmSession::executionConcurrency()const{return impl_->execution.concurrency();}
std::size_t InterfaceEmSession::projectedPeakDeviceBytes()const {
  return impl_->bytes+impl_->resident.capacity()*sizeof(em::EmParticleState);
}
std::size_t InterfaceEmSession::residentCapacity()const{return impl_->resident.capacity();}
std::size_t InterfaceEmSession::pendingParticles()const{return impl_->resident.size();}
std::size_t InterfaceEmSession::nextResidentBatchSize()const {
  return std::min(impl_->capacity,impl_->resident.size());
}
ResidentStatistics const& InterfaceEmSession::residentStatistics()const{return impl_->resident_stats;}
std::vector<EmStep> InterfaceEmSession::advance(std::vector<em::EmParticleState> const& particles,std::uint64_t first) {
  auto& d=*impl_;
  if(d.cascade_running)throw std::logic_error("reentrant interface cascade");
  if(particles.empty())return {};
  if(particles.size()>d.capacity)throw std::length_error("interface reference batch exceeds capacity");
  Impl::validateHistory(particles.size(),first);
  for(auto const& p:particles)d.validateParticle(p);
  for(std::size_t i=0;i<particles.size();++i)d.host_input(i)=particles[i];
  auto range=std::make_pair(std::size_t{0},particles.size());
  Kokkos::deep_copy(d.execution,Kokkos::subview(d.input,range),Kokkos::subview(d.host_input,range));
  return execute(particles.size(),first);
}
void InterfaceEmSession::submit(std::vector<em::EmParticleState> const& particles) {
  auto& d=*impl_;
  if(d.cascade_running)throw std::logic_error("reentrant interface cascade");
  if(!d.resident.capacity())throw std::logic_error("interface resident queue is disabled");
  if(particles.size()>d.resident.capacity()-d.resident.size())
    throw std::length_error("CPU injection exceeds interface resident capacity");
  // Admission is atomic, including when the input spans multiple staging batches.
  for(auto const& p:particles)d.validateParticle(p);
  for(std::size_t begin=0;begin<particles.size();begin+=d.capacity) {
    auto count=std::min(d.capacity,particles.size()-begin);
    for(std::size_t i=0;i<count;++i)d.host_input(i)=particles[begin+i];
    auto range=std::make_pair(std::size_t{0},count);
    Kokkos::deep_copy(d.execution,Kokkos::subview(d.input,range),Kokkos::subview(d.host_input,range));
    d.resident.append(d.input,count,d.execution);
    // Staging may be aliased in HostSpace; finish before the next host write.
    d.execution.fence("interface CPU injection complete");
    d.resident_stats.uploaded_particles+=count;
    ++d.resident_stats.upload_batches;
  }
  d.resident_stats.peak_pending_particles=std::max(d.resident_stats.peak_pending_particles,d.resident.size());
}
std::vector<EmStep> InterfaceEmSession::advanceResident(std::uint64_t first) {
  auto& d=*impl_;
  if(d.cascade_running)throw std::logic_error("reentrant interface cascade");
  if(!d.resident.capacity())throw std::logic_error("interface resident queue is disabled");
  auto count=nextResidentBatchSize();if(!count)return {};
  Impl::validateHistory(count,first);
  d.resident.copyPrefix(d.input,count,d.execution);
  auto records=execute(count,first);
  auto appended=d.resident.commit(d.output,count,d.execution);
  d.execution.fence("interface resident queue commit complete");
  ++d.resident_stats.wavefronts;
  d.resident_stats.advanced_particles+=count;
  d.resident_stats.device_enqueued_particles+=appended;
  d.resident_stats.peak_pending_particles=std::max(d.resident_stats.peak_pending_particles,d.resident.size());
  return records;
}
ResidentCascadeResult InterfaceEmSession::runResidentCascade(
    std::function<std::uint64_t(std::size_t)> const& reserve,
    std::size_t minimum_pending,std::size_t maximum_wavefronts) {
  auto& d=*impl_;
  if(d.cascade_running)throw std::logic_error("reentrant interface cascade");
  if(!d.resident.capacity() || !reserve || !minimum_pending || !maximum_wavefronts)
    throw std::invalid_argument("invalid resident cascade controls");
  ResidentCascadeResult result;
  if(!d.resident.size())return result;
  d.cascade_running=true;
  std::size_t buffered=0;
  try {
    while(d.resident.size()) {
      if(result.wavefronts && d.resident.size()<minimum_pending) {
        result.checkpoint=ResidentCheckpoint::ScalarInterleave;break;
      }
      if(result.wavefronts==maximum_wavefronts) {
        result.checkpoint=ResidentCheckpoint::WavefrontLimit;break;
      }
      auto count=nextResidentBatchSize();
      if(count>d.resident_records.extent(0)-buffered) {
        result.checkpoint=ResidentCheckpoint::RecordCapacity;break;
      }
      auto first=reserve(3*count);
      Impl::validateHistory(count,first);
      d.resident.copyPrefix(d.input,count,d.execution);
#ifdef C8_TERRAIN_STEP_AUDIT
      // Explicit audit binaries retain their per-wavefront CPU shadow replay.
      // The production branch never downloads these complete step records.
      execute(count,first);
#else
      launch(count,first);
#endif
      auto control=d.resident.commitControlled(d.output,count,d.interface,d.execution);
#ifndef C8_TERRAIN_STEP_AUDIT
      if(d.radio)d.radio->accumulate(d.radio_tracks,count,d.execution);
#endif
      ++d.resident_stats.control_downloads;
      Kokkos::deep_copy(d.execution,
          Kokkos::subview(d.resident_records,std::make_pair(buffered,buffered+count)),
          Kokkos::subview(d.output,std::make_pair(std::size_t{0},count)));
      buffered+=count;
      ++result.wavefronts;
      ++d.resident_stats.wavefronts;
      unsigned sizeBin=0;for(auto n=count;n>1;n>>=1)++sizeBin;
      ++d.resident_stats.wavefront_size_log2[sizeBin];
      d.resident_stats.advanced_particles+=count;
      d.resident_stats.device_enqueued_particles+=control.successors;
      d.resident_stats.peak_pending_particles=std::max(d.resident_stats.peak_pending_particles,d.resident.size());
      if(control.fallbacks) {
        result.checkpoint=ResidentCheckpoint::CpuFallback;break;
      }
    }
    if(buffered) {
      auto range=std::make_pair(std::size_t{0},buffered);
      Kokkos::deep_copy(d.execution,Kokkos::subview(d.host_resident_records,range),
          Kokkos::subview(d.resident_records,range));
      d.execution.fence("interface resident cascade output checkpoint");
      result.records.reserve(buffered);
      for(std::size_t i=0;i<buffered;++i) {
        auto r=d.host_resident_records(i);d.finishRecord(r);result.records.push_back(r);
      }
      ++d.resident_stats.record_downloads;
      d.resident_stats.downloaded_records+=buffered;
    }
    ++d.resident_stats.cascade_calls;
    ++d.resident_stats.checkpoint_counts[static_cast<std::size_t>(result.checkpoint)];
    d.resident_stats.maximum_call_wavefronts=std::max(d.resident_stats.maximum_call_wavefronts,result.wavefronts);
    d.resident_stats.peak_buffered_records=std::max(d.resident_stats.peak_buffered_records,buffered);
    d.cascade_running=false;
    return result;
  } catch(...) {
    // Keep all backing Views alive until queued D2D operations finish.
    d.execution.fence("unwind interface resident cascade");
    d.cascade_running=false;
    throw;
  }
}
void InterfaceEmSession::launch(std::size_t count,std::uint64_t first) {
  auto& d=*impl_;
  detail::InterfaceStepKernel<Memory> kernel{d.terrain.deviceView(),d.materials,d.input,d.output,first,
      d.maximum_step_m,d.transport_window_s,d.energy_ledger,d.interface};
  kernel.radio_tracks=d.radio_tracks;
  kernel.coverage=d.coverage.view();
#ifdef C8_TERRAIN_STEP_AUDIT
  kernel.audit=d.audit_device;
#endif
  Kokkos::parallel_for("interface_material_em_step",Kokkos::RangePolicy<Space>(d.execution,0,count),kernel);
}
std::vector<EmStep> InterfaceEmSession::execute(std::size_t count,std::uint64_t first) {
  auto& d=*impl_;
  launch(count,first);
  auto range=std::make_pair(std::size_t{0},count);
  Kokkos::deep_copy(d.execution,Kokkos::subview(d.host_output,range),Kokkos::subview(d.output,range));
  d.execution.fence("interface step records complete");
#ifdef C8_TERRAIN_STEP_AUDIT
  // Resident inputs originate on the device; shadow replay needs an explicit mirror.
  Kokkos::deep_copy(d.host_input,d.input);
  Kokkos::deep_copy(d.audit_reference,d.audit_device);
  detail::InterfaceStepKernel<Kokkos::HostSpace> shadow{
      d.audit_mesh.view(),d.audit_materials,d.host_input,d.audit_result,first,
      d.maximum_step_m,d.transport_window_s,d.energy_ledger,d.interface,d.audit_host,{}};
  shadow.coverage=d.host_coverage.view();
  for(int aligned=0;aligned<2;++aligned) {
    if(aligned)shadow.replay=d.audit_reference;
    for(std::size_t i=0;i<count;++i) {
      shadow(i); // Same input, history key, immutable banks, exact production functor.
      d.audit_writer.compare(aligned?"aligned-stages":"same-input",i,d.host_input(i),
          d.audit_host(i),d.audit_reference(i),d.audit_result(i),d.host_output(i));
    }
  }
  d.audit_writer.endBatch();
#endif
  std::vector<EmStep> result;result.reserve(count);
  for(std::size_t i=0;i<count;++i) {
    auto r=d.host_output(i);
    d.finishRecord(r);
    result.push_back(r);
  }
  if(d.radio)d.radio->accumulate(d.radio_tracks,count,d.execution);
  return result;
}
bool InterfaceEmSession::radioEnabled()const{return bool(impl_->radio);}
void InterfaceEmSession::accumulateRadioCpu(std::vector<corsika::radio::interface::Track> const& tracks){
  auto& d=*impl_;if(d.cascade_running)throw std::logic_error("reentrant interface radio submission");
  if(!d.radio)throw std::logic_error("interface radio is disabled");
  d.radio->accumulateCpu(tracks,d.execution);
}
corsika::radio::interface::PairedResult InterfaceEmSession::finishRadio(){
  auto& d=*impl_;if(d.cascade_running||d.resident.size())throw std::logic_error("interface radio finalized before cascade drain");
  if(!d.radio)throw std::logic_error("interface radio is disabled");
  return d.radio->download(d.execution);
}
}

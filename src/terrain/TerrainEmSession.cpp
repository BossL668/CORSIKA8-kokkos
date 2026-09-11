// Device translation unit: no host CORSIKA geometry, YAML, or PROPOSAL calculators.
#include <Kokkos_Core.hpp>
#include <corsika/modules/terrain/TerrainEmSession.hpp>
#include <corsika/geometry/terrain/KokkosTerrainSession.hpp>
#include <corsika/geometry/terrain/TerrainCurvedBoundary.hpp>
#include <corsika/geometry/terrain/TerrainTrajectoryAudit.hpp>
#include <corsika/accelerator/em/kokkos/KokkosProposalNativeTable.hpp>
#include <corsika/accelerator/em/kokkos/KokkosMoliereInterpolation.hpp>
#include <corsika/accelerator/em/kokkos/KokkosPhysicsContext.hpp>
#include <corsika/accelerator/em/detail/InteractionSelection.hpp>
#include <corsika/accelerator/em/detail/PhotonTransportStep.hpp>
#include <corsika/accelerator/em/detail/LeptonTransportStep.hpp>
#include <corsika/accelerator/em/detail/MoliereStep.hpp>
#include <corsika/accelerator/em/detail/LeptonVertexSelection.hpp>
#include <corsika/accelerator/em/detail/PhotonFinalStateStep.hpp>
#include <corsika/accelerator/em/detail/LeptonFinalStateStep.hpp>
#include <algorithm>
#ifdef C8_TERRAIN_STEP_AUDIT
#include "TerrainStepAudit.hpp"
#define C8_TERRAIN_AUDIT_STAGE(member, value, mask) \
  audit(index).member=value; audit(index).stages|=mask; \
  if(replay.data())value=replay(index).member
#else
#define C8_TERRAIN_AUDIT_STAGE(member, value, mask)
#endif

namespace corsika::terrain {
namespace kd = accelerator::em::kokkos_detail;
namespace physics = accelerator::em::detail;
using Space = Kokkos::DefaultExecutionSpace;
using Memory = Space::memory_space;

template<class Storage=Memory> struct TerrainStepKernel {
  flat::View terrain;
  Kokkos::View<kd::KokkosPhysicsContextData*, Storage> materials;
  Kokkos::View<em::EmParticleState*, Storage> input;
  Kokkos::View<TerrainEmStep*, Storage> output;
  std::uint64_t first_history;
  double maximum_step_m{HUGE_VAL}, transport_window_s{HUGE_VAL};
  bool energy_ledger{false};
  interfaces::MaterialInterface interface;
#ifdef C8_TERRAIN_STEP_AUDIT
  Kokkos::View<audit::Record*,Storage> audit{};
  Kokkos::View<audit::Record*,Storage> replay{};
#endif

  KOKKOS_INLINE_FUNCTION void recordMidpoint(TerrainEmStep& out,bool bent)const {
    auto const& a=out.start;auto const& b=out.end;
    flat::Vec3 middle{.5*(a.position_m[0]+b.position_m[0]),
                      .5*(a.position_m[1]+b.position_m[1]),
                      .5*(a.position_m[2]+b.position_m[2])};
    if(bent)middle=flat::quadraticMidpoint(
        {a.position_m[0],a.position_m[1],a.position_m[2]},
        {b.position_m[0],b.position_m[1],b.position_m[2]},
        {out.distance_m*a.direction[0],out.distance_m*a.direction[1],out.distance_m*a.direction[2]});
    out.path_midpoint_m[0]=middle.x;out.path_midpoint_m[1]=middle.y;out.path_midpoint_m[2]=middle.z;
  }

  template<class Final>
  KOKKOS_INLINE_FUNCTION void finish(TerrainEmStep& out, Final const& final) const {
    if (final.error) {out.error=final.error; return;}
    if (final.has_record) {
      out.outcome=TerrainEmOutcome::Children;out.child_count=final.secondary_count;
      for(std::uint32_t c=0;c<out.child_count;++c)out.children[c]=final.secondaries[c];
    } else if(final.has_fallback) {
      out.outcome=TerrainEmOutcome::Fallback;out.fallback=final.fallback;
    } else if(final.has_suppression) {
      out.outcome=TerrainEmOutcome::Continuation;out.end=final.suppression.particle;
    } else if(final.has_continuation) {
      out.outcome=TerrainEmOutcome::Continuation;out.end=final.continuation.particle;
    } else out.error=100;
  }

  KOKKOS_INLINE_FUNCTION void operator()(std::size_t index) const {
    auto& out=output(index);out=TerrainEmStep{};
#ifdef C8_TERRAIN_STEP_AUDIT
    audit(index)=terrain::audit::Record{};
#endif
    auto const start=input(index);out.start=start;out.end=start;
    if(!interface.containsRegion(start.medium_id)) {out.error=101;return;}
    auto const& bank=materials(interface.material(start.medium_id));
    auto selected=physics::selectDiscreteInteraction(bank.physics,start,index,bank.random_seed,bank.shower_id);
    C8_TERRAIN_AUDIT_STAGE(selection,selected,1);
    auto const deferred_photon_loss = start.pid==22 && selected.fallback_flag &&
        selected.fallback.reason==em::ProposalFallbackReason::NativeSelectionReplay;
    if(selected.fallback_flag&&!deferred_photon_loss) {
      out.outcome=TerrainEmOutcome::Fallback;out.fallback=selected.fallback;return;
    }
    // Photon energy is unchanged in flight: v is not needed to compete the
    // sampled interaction distance with geometry. Complete the SAME selection
    // on CPU only if that interaction actually happens, at the transported
    // vertex. Never create a rare final state at the pre-step starting point.
    if(deferred_photon_loss)selected.interaction.status=em::EmInteractionStatus::Selected;
    flat::BoundaryQuery query;
    query.origin={start.position_m[0],start.position_m[1],start.position_m[2]};
    query.direction={start.direction[0],start.direction[1],start.direction[2]};
    query.logically_inside=start.medium_id==interface.inside_region;
    flat::QuadraticPath path{query,{0.,0.,0.},HUGE_VAL};
    if(start.pid==11||start.pid==-11) {
      // Boundary curvature must use the SAME transport mass as transportLepton,
      // not the slightly different PROPOSAL cross-section/final-state mass.
      auto mass=em::tables::queryContinuousTransportMass(bank.physics,start.pid);
      if(mass.status!=em::tables::TableLookupStatus::Success) {out.error=104;return;}
      double mass_GeV=mass.value/1000.;
      double charge=start.pid>0?-1.:1.;
      auto const& field=bank.environment.magnetic_field_T;
      auto limit=em::maximumUniformMagneticStep(start,mass_GeV,charge,field,
          bank.environment.maximum_magnetic_deflection_rad);
      if(limit.status==em::MagneticStepStatus::Success) {
        double momentum=std::sqrt((start.energy_GeV-mass_GeV)*(start.energy_GeV+mass_GeV));
        auto q=flat::cross(query.direction,{field[0],field[1],field[2]});
        double scale=.5*charge*em::GeVPerCToTeslaMeter/momentum;
        path={query,{scale*q.x,scale*q.y,scale*q.z},limit.distance_m};
      }
    }
    auto boundary=interfaces::nextCrossing(terrain,interface,start.medium_id,path);
    // A finite magnetic step can end before any surface, on EITHER side.
    // A missing exit for an unbounded straight ray in a closed solid is invalid.
    if(query.logically_inside&&!boundary.hit.found()&&!std::isfinite(path.maximum_length_m)) {out.error=102;return;}
    em::ExternalTransportBoundary limiter;
    limiter.enabled=boundary.hit.found();limiter.distance_m=boundary.hit.distance;
    limiter.disable_observation=true; // No ground absorption in the terrain scene.
    limiter.allow_unbounded_inward_grammage=true;
    // Diagnostic caps compete using the same arclength parameter as geometry.
    // They are NOT material surfaces: reaching one must not flip medium_id.
    double diagnostic_distance=maximum_step_m;
    if(std::isfinite(transport_window_s)) {
      double speed=em::SpeedOfLightMPerS;
      if(start.pid!=22) {
        auto mass=em::tables::queryContinuousTransportMass(bank.physics,start.pid);
        if(mass.status!=em::tables::TableLookupStatus::Success){out.error=104;return;}
        double mass_GeV=mass.value/1000.;
        double momentum2=(start.energy_GeV-mass_GeV)*(start.energy_GeV+mass_GeV);
        speed*=std::sqrt(momentum2>0.?momentum2:0.)/start.energy_GeV;
      }
      double remaining=(transport_window_s-start.time_s)*speed;
      remaining=remaining>1.e-9?remaining:1.e-9;
      diagnostic_distance=remaining<diagnostic_distance?remaining:diagnostic_distance;
    }
    bool const diagnostic_limiter=diagnostic_distance<limiter.distance_m;
    if(diagnostic_limiter){limiter.enabled=true;limiter.distance_m=diagnostic_distance;}
    C8_TERRAIN_AUDIT_STAGE(boundary,limiter,2);
    if(start.pid==22) {
      auto result=physics::transportPhoton(bank.environment,selected.interaction,limiter);
      C8_TERRAIN_AUDIT_STAGE(photon,result,32);
      if(result.fallback_flag) {out.outcome=TerrainEmOutcome::Fallback;out.fallback=result.fallback;return;}
      auto const& step=result.record;out.end=step.end;out.has_track=1;
      out.distance_m=step.distance_m;out.grammage_g_cm2=step.traversed_grammage_g_per_cm2;
      recordMidpoint(out,false);
      out.deposited_GeV=step.cut_deposited_energy_GeV;
      if(step.limit==em::PhotonTransportLimit::ParticleCut)out.outcome=TerrainEmOutcome::Cut;
      else if(step.limit==em::PhotonTransportLimit::EscapedEnvironment)out.outcome=TerrainEmOutcome::Escape;
      else if(std::isfinite(transport_window_s)&&step.end.time_s>=transport_window_s-1.e-15)
        out.outcome=TerrainEmOutcome::WindowEscape;
      else if(step.limit==em::PhotonTransportLimit::Interaction) {
        out.process_id=step.interaction.process_id;
        if(deferred_photon_loss) {
          out.outcome=TerrainEmOutcome::Fallback;out.fallback=selected.fallback;
          out.fallback.particle=step.end;return;
        }
        auto const classification=physics::classifyPhotonFinalState(bank.physics,bank.photon_pair_lpm,bank.thinning,
            step.interaction,bank.random_seed,bank.shower_id);
        auto const final=physics::materializePhotonFinalState(step.interaction,classification,3*index,first_history);
#ifdef C8_TERRAIN_STEP_AUDIT
        if(final.has_record){audit(index).photon_final=final.record;audit(index).stages|=64;}
#endif
        finish(out,final);
        if(energy_ledger&&final.has_record) {
          auto raw=classification;
          raw.parameters.thinning_keep_mask=0x3U;
          raw.parameters.thinning_first_weight=step.end.weight;
          raw.parameters.thinning_second_weight=step.end.weight;
          auto const unthinned=physics::materializePhotonFinalState(step.interaction,raw,3*index,first_history);
          if(unthinned.error){out.error=unthinned.error;return;}
          double after=0.;
          for(std::uint32_t c=0;c<unthinned.secondary_count;++c)
            out.unthinned_secondary_total_GeV+=unthinned.secondaries[c].weight*unthinned.secondaries[c].energy_GeV;
          for(std::uint32_t c=0;c<final.secondary_count;++c)after+=final.secondaries[c].weight*final.secondaries[c].energy_GeV;
          out.weighted_thinning_delta_GeV=after-out.unthinned_secondary_total_GeV;
        }
        if(final.has_record&&out.process_id==em::PhotoelectricProcessId)
          out.binding_energy_GeV=step.end.energy_GeV*(1.-final.record.energy_split_fraction);
      } else {
        out.outcome=TerrainEmOutcome::Continuation;
        if(step.limit==em::PhotonTransportLimit::MaterialBoundary&&!diagnostic_limiter)out.crossed_material=1;
      }
    } else if(start.pid==11||start.pid==-11) {
      em::LeptonTransportRecord step;
      auto state=physics::transportLepton(bank.physics,true,bank.environment,selected.interaction,step,out.fallback,limiter);
      C8_TERRAIN_AUDIT_STAGE(transport_state,state,0);
      C8_TERRAIN_AUDIT_STAGE(transport,step,4);
      state=physics::applyMoliereScatteringStage<em::MaxMoliereComponents>(bank.electron_moliere,bank.muon_moliere,
          bank.moliere_interpolation,false,bank.random_seed,bank.shower_id,step,out.fallback,state);
      C8_TERRAIN_AUDIT_STAGE(scattering_state,state,0);
      C8_TERRAIN_AUDIT_STAGE(scattering,step,8);
      if(state) {out.outcome=TerrainEmOutcome::Fallback;return;}
      out.end=step.end;out.has_track=1;out.distance_m=step.distance_m;
      recordMidpoint(out,step.magnetic_bending_applied!=0);
      out.grammage_g_cm2=step.traversed_grammage_g_per_cm2;
      out.deposited_GeV=step.continuous_deposited_energy_GeV+step.cut_deposited_energy_GeV;
      if(step.limit==em::LeptonTransportLimit::ParticleCut)out.outcome=TerrainEmOutcome::Cut;
      else if(step.limit==em::LeptonTransportLimit::EscapedEnvironment)out.outcome=TerrainEmOutcome::Escape;
      else if(std::isfinite(transport_window_s)&&step.end.time_s>=transport_window_s-1.e-15)
        out.outcome=TerrainEmOutcome::WindowEscape;
      else if(step.limit==em::LeptonTransportLimit::InteractionCandidate) {
        auto vertex=physics::selectLeptonVertex(bank.physics,step.interaction,bank.random_seed,bank.shower_id);
        C8_TERRAIN_AUDIT_STAGE(vertex,vertex,16);
        if(vertex.fallback_flag) {out.outcome=TerrainEmOutcome::Fallback;out.fallback=vertex.fallback;return;}
        if(vertex.continuation_flag) {out.outcome=TerrainEmOutcome::Continuation;out.end=vertex.record.particle;return;}
        out.process_id=vertex.record.process_id;
        auto const classification=physics::classifyLeptonFinalState(bank.brems_lpm,bank.thinning,vertex.record,
            bank.random_seed,bank.shower_id);
        auto const final=physics::materializeLeptonFinalState(vertex.record,classification,3*index,first_history,
            bank.brems_lpm.lepton_mass_MeV/1000.);
#ifdef C8_TERRAIN_STEP_AUDIT
        if(final.has_record){audit(index).lepton_final=final.record;audit(index).stages|=128;}
#endif
        finish(out,final);
        if(energy_ledger&&final.has_record) {
          auto raw=classification;
          raw.parameters.thinning_keep_mask=0x3U;
          raw.parameters.thinning_first_weight=step.end.weight;
          raw.parameters.thinning_second_weight=step.end.weight;
          auto const unthinned=physics::materializeLeptonFinalState(vertex.record,raw,3*index,first_history,
              bank.brems_lpm.lepton_mass_MeV/1000.);
          if(unthinned.error){out.error=unthinned.error;return;}
          double after=0.;
          for(std::uint32_t c=0;c<unthinned.secondary_count;++c)
            out.unthinned_secondary_total_GeV+=unthinned.secondaries[c].weight*unthinned.secondaries[c].energy_GeV;
          for(std::uint32_t c=0;c<final.secondary_count;++c)after+=final.secondaries[c].weight*final.secondaries[c].energy_GeV;
          out.weighted_thinning_delta_GeV=after-out.unthinned_secondary_total_GeV;
        }
      } else {
        out.outcome=TerrainEmOutcome::Continuation;
        if(step.limit==em::LeptonTransportLimit::MaterialBoundary&&!diagnostic_limiter)out.crossed_material=1;
      }
    } else out.error=103;
    if(out.crossed_material)out.end.medium_id=boundary.to_region;
  }
};

class TerrainEmSession::Impl {
 public:
  // Declared first, destroyed last: Views must be released before finalize().
  Kokkos::ScopeGuard runtime;
  Space execution;
  KokkosTerrainSession<Space> terrain;
  std::vector<kd::KokkosProposalNativeTable<Space>> tables;
  std::vector<kd::KokkosMoliereInterpolation<Space>> moliere;
  Kokkos::View<kd::KokkosPhysicsContextData*,Memory> materials;
  Kokkos::View<em::EmParticleState*,Memory> input;
  Kokkos::View<TerrainEmStep*,Memory> output;
  decltype(input)::HostMirror host_input;
  decltype(output)::HostMirror host_output;
  std::size_t capacity{},bytes{};
  double maximum_step_m,transport_window_s;
  bool energy_ledger{};
  interfaces::MaterialInterface interface;
  std::vector<std::vector<em::tables::NativeInteractionIdentity>> identities;
#ifdef C8_TERRAIN_STEP_AUDIT
  // Bounded diagnostic mirrors only. Never present in production builds.
  FlatTerrainData audit_mesh;
  std::vector<TerrainEmMaterial> audit_banks;
  std::vector<em::MoliereInterpolationTable> audit_moliere;
  Kokkos::View<kd::KokkosPhysicsContextData*,Kokkos::HostSpace> audit_materials;
  Kokkos::View<audit::Record*,Memory> audit_device;
  Kokkos::View<audit::Record*,Kokkos::HostSpace> audit_host,audit_reference;
  Kokkos::View<TerrainEmStep*,Kokkos::HostSpace> audit_result;
  audit::Writer audit_writer;
#endif
  static Kokkos::InitializationSettings settings(TerrainEmConfig const& c,
      FlatTerrainData const& mesh,std::vector<TerrainEmMaterial> const& banks) {
    if(Kokkos::is_initialized())throw std::logic_error("terrain requires its own independent runtime");
    if(c.threads<1||c.device<0||c.batch_size<1||c.batch_size>4096||c.maximum_device_bytes<1024)
      throw std::invalid_argument("invalid bounded terrain runtime settings");
    if(!(c.maximum_step_m>0.)||!(c.transport_window_s>0.)||
        (std::isfinite(c.transport_window_s)&&c.transport_window_s>1.e-3))
      throw std::invalid_argument("invalid terrain diagnostic step/window");
    c.interface.validate(banks.size());
    for(auto const& bank:banks) {
      if(!em::atmosphere_detail::validEnvironment(bank.environment))
        throw std::invalid_argument("unsupported interface material density/geometry/field snapshot");
      for(double b:bank.environment.magnetic_field_T)
        if(!std::isfinite(b))throw std::invalid_argument("nonfinite interface magnetic field");
    }
    auto projected=mesh.bytes()+1024+c.batch_size*(sizeof(em::EmParticleState)+sizeof(TerrainEmStep))+
        banks.size()*(sizeof(kd::KokkosPhysicsContextData)+kd::KokkosMoliereInterpolation<Space>::projectedDeviceBytes());
    for(auto const& bank:banks)projected+=kd::KokkosProposalNativeTable<Space>::projectedDeviceBytes(bank.table);
    if(projected>c.maximum_device_bytes)throw std::runtime_error(
        "terrain pre-initialization memory gate requires "+std::to_string(projected)+" bytes");
    Kokkos::InitializationSettings s;s.set_num_threads(c.threads);s.set_device_id(c.device);return s;
  }
  Impl(FlatTerrainData const& mesh,std::vector<TerrainEmMaterial> const& banks,TerrainEmConfig const& c)
      :runtime(settings(c,mesh,banks)),terrain(mesh,1),capacity(c.batch_size),
       maximum_step_m(c.maximum_step_m),transport_window_s(c.transport_window_s),energy_ledger(c.energy_ledger),interface(c.interface),
       identities(banks.size()) {
    tables=decltype(tables)(banks.size());moliere=decltype(moliere)(banks.size());
    bytes=terrain.geometryBytes()+terrain.workspaceBytes()+capacity*(sizeof(em::EmParticleState)+sizeof(TerrainEmStep))+
        banks.size()*sizeof(kd::KokkosPhysicsContextData);
    for(auto const& bank:banks) {
      for(double b:bank.environment.magnetic_field_T)
        if(!std::isfinite(b))throw std::invalid_argument("nonfinite terrain magnetic field");
      bytes+=kd::KokkosProposalNativeTable<Space>::projectedDeviceBytes(bank.table);
    }
    if(bytes+banks.size()*kd::KokkosMoliereInterpolation<Space>::projectedDeviceBytes()>c.maximum_device_bytes)
      throw std::runtime_error("terrain immutable data exceed the validated memory budget");
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
      d.electron_moliere=banks[i].auxiliary.electron_moliere;
      d.moliere_interpolation=moliere[i].deviceView();d.photon_pair_lpm=banks[i].auxiliary.photon_pair_lpm;
      d.brems_lpm=banks[i].auxiliary.brems_lpm;d.thinning=c.thinning;d.random_seed=c.seed;d.shower_id=c.shower_id;
    }
    if(bytes>c.maximum_device_bytes)throw std::runtime_error("terrain memory budget exceeded");
    Kokkos::deep_copy(materials,host);
    input=decltype(input)("terrain_step_input",capacity);output=decltype(output)("terrain_step_output",capacity);
    host_input=Kokkos::create_mirror_view(input);host_output=Kokkos::create_mirror_view(output);
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
TerrainEmSession::TerrainEmSession(FlatTerrainData const& m,std::vector<TerrainEmMaterial> const& b,TerrainEmConfig const& c)
    :impl_(std::make_unique<Impl>(m,b,c)){}
TerrainEmSession::~TerrainEmSession()=default;
std::size_t TerrainEmSession::deviceBytes()const{return impl_->bytes;}
std::string TerrainEmSession::executionSpace()const{return Space::name();}
std::vector<TerrainEmStep> TerrainEmSession::advance(std::vector<em::EmParticleState> const& particles,std::uint64_t first) {
  auto& d=*impl_;if(particles.empty())return {};
  if(particles.size()>d.capacity||first==0||first>UINT64_MAX-3*particles.size())
    throw std::invalid_argument("terrain wavefront/history capacity exceeded");
  for(std::size_t i=0;i<particles.size();++i) {
    auto const& p=particles[i];double norm=0.;
    for(int a=0;a<3;++a) {
      if(!std::isfinite(p.position_m[a])||!std::isfinite(p.direction[a]))
        throw std::invalid_argument("nonfinite terrain particle");
      norm+=p.direction[a]*p.direction[a];
    }
    if((p.pid!=22&&p.pid!=11&&p.pid!=-11)||!d.interface.containsRegion(p.medium_id)||p.history_id==0||
        p.step_id==UINT64_MAX||!std::isfinite(p.energy_GeV)||p.energy_GeV<0.||
        !std::isfinite(p.time_s)||!std::isfinite(p.weight)||p.weight<=0.||std::abs(norm-1.)>1.e-12)
      throw std::invalid_argument("invalid routed terrain particle");
    d.host_input(i)=p;
  }
  Kokkos::deep_copy(d.execution,d.input,d.host_input);
  TerrainStepKernel<> kernel{d.terrain.deviceView(),d.materials,d.input,d.output,first,
      d.maximum_step_m,d.transport_window_s,d.energy_ledger,d.interface};
#ifdef C8_TERRAIN_STEP_AUDIT
  kernel.audit=d.audit_device;
#endif
  Kokkos::parallel_for("terrain_material_em_step",Kokkos::RangePolicy<Space>(d.execution,0,particles.size()),kernel);
  Kokkos::deep_copy(d.execution,d.host_output,d.output);d.execution.fence("terrain step and records complete");
#ifdef C8_TERRAIN_STEP_AUDIT
  Kokkos::deep_copy(d.audit_reference,d.audit_device);
  TerrainStepKernel<Kokkos::HostSpace> shadow{
      d.audit_mesh.view(),d.audit_materials,d.host_input,d.audit_result,first,
      d.maximum_step_m,d.transport_window_s,d.energy_ledger,d.interface,d.audit_host,{}};
  for(int aligned=0;aligned<2;++aligned) {
    if(aligned)shadow.replay=d.audit_reference;
    for(std::size_t i=0;i<particles.size();++i) {
      shadow(i); // Same input, history key, immutable banks, exact production functor.
      d.audit_writer.compare(aligned?"aligned-stages":"same-input",i,particles[i],
          d.audit_host(i),d.audit_reference(i),d.audit_result(i),d.host_output(i));
    }
  }
  d.audit_writer.endBatch();
#endif
  std::vector<TerrainEmStep> result;result.reserve(particles.size());
  for(std::size_t i=0;i<particles.size();++i) {
    auto r=d.host_output(i);
    if(r.error||r.outcome==TerrainEmOutcome::Error)throw std::runtime_error("terrain device error "+std::to_string(r.error));
    if(r.has_track&&(!std::isfinite(r.distance_m)||r.distance_m<0.||
        !std::isfinite(r.grammage_g_cm2)||r.grammage_g_cm2<0.))
      throw std::runtime_error("invalid finite terrain transport distance/grammage");
    // Restore logical region after a native atmosphere-layer transition. Such
    // transitions do NOT cross this mesh. No position, time, energy or RNG edits.
    r.end.medium_id=r.crossed_material?d.interface.opposite(r.start.medium_id):r.start.medium_id;
    for(std::uint32_t c=0;c<r.child_count;++c)r.children[c].medium_id=r.start.medium_id;
    if(r.outcome==TerrainEmOutcome::Fallback) {
      r.fallback.particle.medium_id=r.start.medium_id;
      auto const& identities=d.identities[d.interface.material(r.start.medium_id)];
      auto found=std::find_if(identities.begin(),identities.end(),[&](auto const& id){return id.pdg_id==r.fallback.particle.pid;});
      if(found==identities.end())throw std::runtime_error("missing actual terrain fallback calculator identity");
      r.fallback.medium_hash=found->medium_hash;r.fallback.interaction_hash=found->interaction_hash;
    }
    result.push_back(r);
  }
  return result;
}
}

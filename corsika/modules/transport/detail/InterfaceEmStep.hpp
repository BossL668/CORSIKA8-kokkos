// Two-sided EM step: shared beta5 physics with caller-supplied material/field banks.
#pragma once
#include <Kokkos_Core.hpp>
#include <corsika/modules/transport/InterfaceEmTypes.hpp>
#include <corsika/geometry/terrain/KokkosTerrainSession.hpp>
#include <corsika/geometry/terrain/TerrainCurvedBoundary.hpp>
#include <corsika/geometry/terrain/TerrainTrajectoryAudit.hpp>
#include <corsika/accelerator/em/kokkos/KokkosProposalNativeTable.hpp>
#include <corsika/accelerator/em/kokkos/KokkosMoliereInterpolation.hpp>
#include <corsika/accelerator/em/kokkos/KokkosPhysicsContext.hpp>
#include <corsika/accelerator/em/detail/InteractionSelection.hpp>
#include <corsika/accelerator/em/detail/PhotonTransportStep.hpp>
#include <corsika/accelerator/em/detail/LeptonTransportStep.hpp>
#include <corsika/modules/transport/detail/InterfaceMagneticPolicy.hpp>
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

namespace corsika::interfaces::detail {
namespace kd = accelerator::em::kokkos_detail;
namespace physics = accelerator::em::detail;
namespace flat = terrain::flat;
#ifdef C8_TERRAIN_STEP_AUDIT
namespace audit = terrain::audit;
#endif
template<class Storage> struct InterfaceStepKernel {
  flat::View terrain;
  Kokkos::View<kd::KokkosPhysicsContextData*, Storage> materials;
  Kokkos::View<em::EmParticleState*, Storage> input;
  Kokkos::View<EmStep*, Storage> output;
  std::uint64_t first_history;
  double maximum_step_m{HUGE_VAL}, transport_window_s{HUGE_VAL};
  bool energy_ledger{false};
  interfaces::MaterialInterface interface;
#ifdef C8_TERRAIN_STEP_AUDIT
  Kokkos::View<audit::Record*,Storage> audit{};
  Kokkos::View<audit::Record*,Storage> replay{};
#endif
  Kokkos::View<corsika::radio::interface::Track*,Storage> radio_tracks;
  terrain::coverage::View coverage;

  KOKKOS_INLINE_FUNCTION void recordMidpoint(EmStep& out,bool bent)const {
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
  KOKKOS_INLINE_FUNCTION void finish(EmStep& out, Final const& final) const {
    if (final.error) {out.error=final.error; return;}
    if (final.has_record) {
      out.outcome=EmOutcome::Children;out.child_count=final.secondary_count;
      for(std::uint32_t c=0;c<out.child_count;++c)out.children[c]=final.secondaries[c];
    } else if(final.has_fallback) {
      out.outcome=EmOutcome::Fallback;out.fallback=final.fallback;
    } else if(final.has_suppression) {
      out.outcome=EmOutcome::Continuation;out.end=final.suppression.particle;
    } else if(final.has_continuation) {
      out.outcome=EmOutcome::Continuation;out.end=final.continuation.particle;
    } else out.error=100;
  }

  KOKKOS_INLINE_FUNCTION void transport(std::size_t index) const {
    auto& out=output(index);out=EmStep{};
#ifdef C8_TERRAIN_STEP_AUDIT
    audit(index)=terrain::audit::Record{};
#endif
    auto const start=input(index);out.start=start;out.end=start;
    if(!terrain::coverage::contains(coverage,{start.position_m[0],start.position_m[1],start.position_m[2]})) {out.error=106;return;}
    if(!interface.containsRegion(start.medium_id)) {out.error=101;return;}
    auto const& bank=materials(interface.material(start.medium_id));
    auto selected=physics::selectDiscreteInteraction(bank.physics,start,index,bank.random_seed,bank.shower_id);
    C8_TERRAIN_AUDIT_STAGE(selection,selected,1);
    if(selected.fallback_flag) {
      out.outcome=EmOutcome::Fallback;out.fallback=selected.fallback;return;
    }
    // Deferred photon selected-loss completion is owned by the shared
    // selector/transport/final-state pipeline, as in the air resident front.
    flat::BoundaryQuery query;
    query.origin={start.position_m[0],start.position_m[1],start.position_m[2]};
    query.direction={start.direction[0],start.direction[1],start.direction[2]};
    query.logically_inside=start.medium_id==interface.inside_region;
    flat::QuadraticPath path{query,{0.,0.,0.},HUGE_VAL};
    bool linear_field_override=false;
    if(start.pid==11||start.pid==-11) {
      // Boundary curvature must use the SAME transport mass as transportLepton,
      // not the slightly different PROPOSAL cross-section/final-state mass.
      auto mass=em::tables::queryContinuousTransportMass(bank.physics,start.pid);
      if(mass.status!=em::tables::TableLookupStatus::Success) {out.error=104;return;}
      double mass_GeV=mass.value/1000.;
      double charge=start.pid>0?-1.:1.;
      auto const& field=bank.environment.magnetic_field_T;
      auto limit=maximumInterfaceMagneticStep(start,mass_GeV,charge,field,
          bank.environment.maximum_magnetic_deflection_rad);
      if(limit.status==em::MagneticStepStatus::InvalidInput||limit.status==em::MagneticStepStatus::NonFiniteResult){out.error=105;return;}
      linear_field_override=limit.status==em::MagneticStepStatus::Linear&&
          (field[0]!=0.||field[1]!=0.||field[2]!=0.);
      if(limit.status==em::MagneticStepStatus::Success) {
        double momentum=std::sqrt((start.energy_GeV-mass_GeV)*(start.energy_GeV+mass_GeV));
        auto q=flat::cross(query.direction,{field[0],field[1],field[2]});
        double scale=.5*charge*em::GeVPerCToTeslaMeter/momentum;
        path={query,{scale*q.x,scale*q.y,scale*q.z},limit.distance_m};
      }
    }
    auto boundary=interfaces::nextCrossing(terrain,interface,start.medium_id,path);
    auto domain=terrain::coverage::nextExit(coverage,path);
    // A finite magnetic step can end before any surface, on EITHER side.
    // A missing exit for an unbounded straight ray in a closed solid is invalid.
    if(query.logically_inside&&!boundary.hit.found()&&!std::isfinite(path.maximum_length_m)) {out.error=102;return;}
    em::ExternalTransportBoundary limiter;
    limiter.enabled=boundary.hit.found();limiter.distance_m=boundary.hit.distance;
    limiter.disable_observation=true; // No ground absorption in the terrain scene.
    limiter.allow_unbounded_inward_grammage=true;
    // Diagnostic caps compete using the same arclength parameter as geometry.
    // They are NOT material surfaces: reaching one must not flip medium_id.
    // The stable interface cap also competes inside the reused legacy
    // transport step. A cap endpoint is a continuation, never a material flip.
    double diagnostic_distance=path.maximum_length_m<maximum_step_m?path.maximum_length_m:maximum_step_m;
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
    // Ties at artificial DEM sides belong to the computational boundary.
    // Use the earlier endpoint, never propagate past the material solver's hit.
    bool const domain_limiter=domain.found()&&domain.distance<=limiter.distance_m+1.e-8;
    if(domain_limiter){limiter.enabled=true;limiter.distance_m=limiter.distance_m<domain.distance?limiter.distance_m:domain.distance;out.domain_edge=domain.edge;}
    C8_TERRAIN_AUDIT_STAGE(boundary,limiter,2);
    if(start.pid==22) {
      auto result=physics::transportPhoton(bank.environment,selected.interaction,limiter);
      C8_TERRAIN_AUDIT_STAGE(photon,result,32);
      if(result.fallback_flag) {out.outcome=EmOutcome::Fallback;out.fallback=result.fallback;return;}
      auto const& step=result.record;out.end=step.end;out.has_track=1;
      out.distance_m=step.distance_m;out.grammage_g_cm2=step.traversed_grammage_g_per_cm2;
      recordMidpoint(out,false);
      out.deposited_GeV=step.cut_deposited_energy_GeV;
      if(step.limit==em::PhotonTransportLimit::ParticleCut)out.outcome=EmOutcome::Cut;
      else if(step.limit==em::PhotonTransportLimit::EscapedEnvironment)out.outcome=EmOutcome::Escape;
      else if(domain_limiter&&step.limit==em::PhotonTransportLimit::MaterialBoundary)out.outcome=EmOutcome::DomainEscape;
      else if(std::isfinite(transport_window_s)&&step.end.time_s>=transport_window_s-1.e-15)
        out.outcome=EmOutcome::WindowEscape;
      else if(step.limit==em::PhotonTransportLimit::Interaction) {
        out.process_id=step.interaction.process_id;
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
        out.outcome=EmOutcome::Continuation;
        if(step.limit==em::PhotonTransportLimit::MaterialBoundary&&!diagnostic_limiter)out.crossed_material=1;
      }
    } else if(start.pid==11||start.pid==-11) {
      em::LeptonTransportRecord step;
      std::uint32_t state;
      if(linear_field_override){
        // Apply the same linear decision to legacy geometry AND propagation.
        // The caller's material/field bank is immutable; air modules are unchanged.
        auto environment=bank.environment;
        for(unsigned axis=0;axis<3;++axis)environment.magnetic_field_T[axis]=0.;
        state=physics::transportLepton(bank.physics,true,environment,selected.interaction,step,out.fallback,limiter);
      }else state=physics::transportLepton(bank.physics,true,bank.environment,selected.interaction,step,out.fallback,limiter);
      C8_TERRAIN_AUDIT_STAGE(transport_state,state,0);
      C8_TERRAIN_AUDIT_STAGE(transport,step,4);
      state=physics::applyMoliereScatteringStage<em::MaxMoliereComponents>(bank.electron_moliere,bank.muon_moliere,
          bank.moliere_interpolation,false,bank.random_seed,bank.shower_id,step,out.fallback,state);
      C8_TERRAIN_AUDIT_STAGE(scattering_state,state,0);
      C8_TERRAIN_AUDIT_STAGE(scattering,step,8);
      if(state) {out.outcome=EmOutcome::Fallback;return;}
      out.end=step.end;out.has_track=1;out.distance_m=step.distance_m;
      // Capture the completed physical segment before vertex selection or
      // continuation/material bookkeeping changes the composite output state.
      if(radio_tracks.data()){
        auto const& a=out.start;auto const& b=step.end;
        radio_tracks(index)={{a.position_m[0],a.position_m[1],a.position_m[2]},
          {b.position_m[0],b.position_m[1],b.position_m[2]},a.time_s,b.time_s,
          a.pid==11?-1.:1.,a.weight,a.history_id,a.step_id,
          std::uint32_t(a.medium_id==interface.inside_region),1,1.e-9};
      }
      recordMidpoint(out,step.magnetic_bending_applied!=0);
      out.grammage_g_cm2=step.traversed_grammage_g_per_cm2;
      out.deposited_GeV=step.continuous_deposited_energy_GeV+step.cut_deposited_energy_GeV;
      if(step.limit==em::LeptonTransportLimit::ParticleCut)out.outcome=EmOutcome::Cut;
      else if(step.limit==em::LeptonTransportLimit::EscapedEnvironment)out.outcome=EmOutcome::Escape;
      else if(domain_limiter&&step.limit==em::LeptonTransportLimit::MaterialBoundary)out.outcome=EmOutcome::DomainEscape;
      else if(std::isfinite(transport_window_s)&&step.end.time_s>=transport_window_s-1.e-15)
        out.outcome=EmOutcome::WindowEscape;
      else if(step.limit==em::LeptonTransportLimit::InteractionCandidate) {
        auto vertex=physics::selectLeptonVertex(bank.physics,step.interaction,bank.random_seed,bank.shower_id);
        C8_TERRAIN_AUDIT_STAGE(vertex,vertex,16);
        if(vertex.fallback_flag) {out.outcome=EmOutcome::Fallback;out.fallback=vertex.fallback;return;}
        if(vertex.continuation_flag) {out.outcome=EmOutcome::Continuation;out.end=vertex.record.particle;return;}
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
        out.outcome=EmOutcome::Continuation;
        if(step.limit==em::LeptonTransportLimit::MaterialBoundary&&!diagnostic_limiter)out.crossed_material=1;
      }
    } else out.error=103;
    if(out.crossed_material)out.end.medium_id=boundary.to_region;
  }
  KOKKOS_INLINE_FUNCTION void operator()(std::size_t index) const {
    if(radio_tracks.data())radio_tracks(index)=corsika::radio::interface::Track{};
    transport(index);
    auto& r = output(index);
    if (r.child_count > 3) { r.error = 105; return; }
    r.end.medium_id = r.crossed_material ? interface.opposite(r.start.medium_id) : r.start.medium_id;
    for (std::uint32_t c = 0; c < r.child_count; ++c)
      r.children[c].medium_id = r.start.medium_id;
    r.fallback.particle.medium_id = r.start.medium_id;
  }

};

} // namespace corsika::interfaces::detail
#undef C8_TERRAIN_AUDIT_STAGE

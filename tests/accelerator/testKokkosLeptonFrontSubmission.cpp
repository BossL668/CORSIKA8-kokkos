/* Isolated frozen-source regression using real PROPOSAL native air splines. */
#include <Kokkos_Core.hpp>
#include <corsika/accelerator/em/KokkosRuntime.hpp>
#include <corsika/accelerator/em/kokkos/KokkosLeptonFrontSubmission.hpp>
#include <corsika/accelerator/em/kokkos/KokkosProposalNativeTable.hpp>
#include <corsika/accelerator/em/kokkos/KokkosMoliereInterpolation.hpp>
#include <corsika/accelerator/em/common/ConvexEnvironmentSnapshot.hpp>
#include "CooperativeLeptonFixture.hpp"
#include "ResidentLeptonCascadeBefore.hpp"
#include <iostream>
#include <thread>
#include <chrono>
#include <set>
#ifdef C8_COOPERATIVE_CUPTI
#include "CooperativeActivityTrace.hpp"
#endif
namespace kd = corsika::accelerator::em::kokkos_detail;
namespace em = corsika::gpu::em;
namespace nt = em::tables;
namespace rd = corsika::accelerator::radio::detail;
namespace rk = corsika::accelerator::radio::kokkos_detail;
namespace ed = corsika::accelerator::em::detail;
#include "CooperativeProjectionFixture.hpp"
void require(bool value,char const* message) { if(!value) throw std::runtime_error(message); }
template<class F> void rejects(F&& f) {
  bool rejected=false;try{f();}catch(std::exception const&){rejected=true;}
  require(rejected,"invalid state was accepted");
}
void compare(em::EmParticleState const& a, em::EmParticleState const& b) {
#define SAME(field) require(a.field == b.field, "particle first divergence: " #field)
  SAME(pid); SAME(medium_id); SAME(generation); SAME(reserved);
  SAME(energy_GeV); SAME(time_s); SAME(weight); SAME(history_id);
  SAME(parent_history_id); SAME(step_id);
  for (int i=0; i<3; ++i) { SAME(position_m[i]); SAME(direction[i]); }
#undef SAME
}
void compare(em::EmInteractionRecord const& a, em::EmInteractionRecord const& b) {
  compare(a.particle,b.particle);
#define SAME(field) require(a.field == b.field, "interaction first divergence: " #field)
  SAME(input_index); SAME(process_id); SAME(status); SAME(component_hash);
  SAME(total_rate_cm2_per_g); SAME(vertex_total_rate_cm2_per_g);
  SAME(interaction_grammage_g_per_cm2); SAME(mass_density_g_per_cm3);
  SAME(particle_mass_GeV); SAME(decay_distance_m); SAME(decay_uniform); SAME(decay_draw_id);
  SAME(energy_fraction); SAME(distance_uniform); SAME(process_uniform); SAME(loss_quantile);
  SAME(distance_draw_id); SAME(process_draw_id); SAME(loss_draw_id);
  SAME(proposal_selection_uniform); SAME(process_random_process_id);
  SAME(proposal_selection_random_process_id); SAME(proposal_selection_draw_id);
#undef SAME
}
void compare(em::PhotonTransportRecord const& a, em::PhotonTransportRecord const& b) {
  compare(a.start,b.start); compare(a.end,b.end); compare(a.interaction,b.interaction);
#define SAME(field) require(a.field == b.field, "transport first divergence: " #field)
  SAME(input_index); SAME(limit); SAME(start_layer_index); SAME(end_layer_index);
  SAME(distance_m); SAME(traversed_grammage_g_per_cm2); SAME(start_density_g_per_cm3);
  SAME(end_density_g_per_cm3); SAME(limiting_radius_m); SAME(cut_deposited_energy_GeV);
  SAME(observation_surface_reached_before_cut);
#undef SAME
}
void compare(em::PhotonFinalStateRecord const& a, em::PhotonFinalStateRecord const& b) {
#define SAME(field) require(a.field == b.field, "final-state first divergence: " #field)
  SAME(input_index); SAME(parent_history_id); SAME(secondary_offset); SAME(secondary_count);
  SAME(process_id); SAME(energy_split_fraction); SAME(split_uniform); SAME(azimuth_uniform);
  SAME(electron_polar_uniform); SAME(positron_polar_uniform); SAME(lpm_survival_probability);
  SAME(lpm_uniform); SAME(split_draw_id); SAME(azimuth_draw_id); SAME(electron_polar_draw_id);
  SAME(positron_polar_draw_id); SAME(lpm_draw_id); SAME(thinning_status); SAME(thinning_keep_mask);
  SAME(thinning_first_uniform); SAME(thinning_second_uniform);
  SAME(thinning_first_draw_id); SAME(thinning_second_draw_id);
  SAME(weighted_mass_convention_correction_GeV);
#undef SAME
}
void compare(em::ProposalFallbackEvent const& a, em::ProposalFallbackEvent const& b) {
  compare(a.particle,b.particle);
#define SAME(field) require(a.field == b.field, "fallback first divergence: " #field)
  SAME(input_index); SAME(process_id); SAME(reason); SAME(diagnostic_status); SAME(diagnostic_reserved);
  SAME(diagnostic_value0); SAME(diagnostic_value1); SAME(diagnostic_value2);
  SAME(component_hash); SAME(medium_hash); SAME(interaction_hash); SAME(energy_fraction);
  SAME(selection_uniform); SAME(loss_quantile); SAME(final_state_uniform);
  SAME(outer_acceptance_uniform); SAME(random_process_id); SAME(outer_acceptance_random_process_id);
  SAME(random_draw_id); SAME(outer_acceptance_draw_id); SAME(final_state_draw_id);
#undef SAME
}
void compare(em::ObservationRecord const& a, em::ObservationRecord const& b) {
  compare(a.particle,b.particle);
  require(a.status == b.status && a.reserved == b.reserved,"observation first divergence");
}
void compare(em::GpuFirstInteractionSnapshot const& a, em::GpuFirstInteractionSnapshot const& b) {
  require(a.process_id == b.process_id && a.secondary_count == b.secondary_count,"first vertex differs");
  compare(a.parent_at_vertex,b.parent_at_vertex);
  for(unsigned i=0;i<a.secondary_count;++i) compare(a.secondaries[i],b.secondaries[i]);
}
void compare(em::LeptonTransportRecord const& a, em::LeptonTransportRecord const& b) {
  compare(a.interaction,b.interaction);
  compare(a.start,b.start);
  compare(a.end,b.end);
  require(a.input_index==b.input_index,"LeptonTransportRecord first divergence: input_index");
  require(a.limit==b.limit,"LeptonTransportRecord first divergence: limit");
  require(a.start_layer_index==b.start_layer_index,"LeptonTransportRecord first divergence: start_layer_index");
  require(a.end_layer_index==b.end_layer_index,"LeptonTransportRecord first divergence: end_layer_index");
  require(a.distance_m==b.distance_m,"LeptonTransportRecord first divergence: distance_m");
  require(a.traversed_grammage_g_per_cm2==b.traversed_grammage_g_per_cm2,"LeptonTransportRecord first divergence: traversed_grammage_g_per_cm2");
  require(a.continuous_step_grammage_g_per_cm2==b.continuous_step_grammage_g_per_cm2,"LeptonTransportRecord first divergence: continuous_step_grammage_g_per_cm2");
  require(a.start_density_g_per_cm3==b.start_density_g_per_cm3,"LeptonTransportRecord first divergence: start_density_g_per_cm3");
  require(a.end_density_g_per_cm3==b.end_density_g_per_cm3,"LeptonTransportRecord first divergence: end_density_g_per_cm3");
  require(a.limiting_radius_m==b.limiting_radius_m,"LeptonTransportRecord first divergence: limiting_radius_m");
  require(a.continuous_deposited_energy_GeV==b.continuous_deposited_energy_GeV,"LeptonTransportRecord first divergence: continuous_deposited_energy_GeV");
  require(a.cut_deposited_energy_GeV==b.cut_deposited_energy_GeV,"LeptonTransportRecord first divergence: cut_deposited_energy_GeV");
  require(a.observation_surface_reached_before_cut==b.observation_surface_reached_before_cut,"LeptonTransportRecord first divergence: observation_surface_reached_before_cut");
  require(a.multiple_scattering_applied==b.multiple_scattering_applied,"LeptonTransportRecord first divergence: multiple_scattering_applied");
  require(a.multiple_scattering_status==b.multiple_scattering_status,"LeptonTransportRecord first divergence: multiple_scattering_status");
  require(a.multiple_scattering_iterations==b.multiple_scattering_iterations,"LeptonTransportRecord first divergence: multiple_scattering_iterations");
  require(a.multiple_scattering_angle_rad==b.multiple_scattering_angle_rad,"LeptonTransportRecord first divergence: multiple_scattering_angle_rad");
  require(a.multiple_scattering_first_uniform==b.multiple_scattering_first_uniform,"LeptonTransportRecord first divergence: multiple_scattering_first_uniform");
  require(a.multiple_scattering_second_uniform==b.multiple_scattering_second_uniform,"LeptonTransportRecord first divergence: multiple_scattering_second_uniform");
  require(a.multiple_scattering_azimuth_uniform==b.multiple_scattering_azimuth_uniform,"LeptonTransportRecord first divergence: multiple_scattering_azimuth_uniform");
  require(a.magnetic_bending_applied==b.magnetic_bending_applied,"LeptonTransportRecord first divergence: magnetic_bending_applied");
  require(a.magnetic_step_status==b.magnetic_step_status,"LeptonTransportRecord first divergence: magnetic_step_status");
  require(a.magnetic_step_limit_m==b.magnetic_step_limit_m,"LeptonTransportRecord first divergence: magnetic_step_limit_m");
  require(a.magnetic_gyroradius_m==b.magnetic_gyroradius_m,"LeptonTransportRecord first divergence: magnetic_gyroradius_m");
  require(a.magnetic_bend_parameter==b.magnetic_bend_parameter,"LeptonTransportRecord first divergence: magnetic_bend_parameter");
  require(a.magnetic_chord_length_m==b.magnetic_chord_length_m,"LeptonTransportRecord first divergence: magnetic_chord_length_m");
}
void compare(em::BremsFinalStateRecord const& a, em::BremsFinalStateRecord const& b) {
  require(a.input_index==b.input_index,"BremsFinalStateRecord first divergence: input_index");
  require(a.parent_history_id==b.parent_history_id,"BremsFinalStateRecord first divergence: parent_history_id");
  require(a.secondary_offset==b.secondary_offset,"BremsFinalStateRecord first divergence: secondary_offset");
  require(a.secondary_count==b.secondary_count,"BremsFinalStateRecord first divergence: secondary_count");
  require(a.process_id==b.process_id,"BremsFinalStateRecord first divergence: process_id");
  require(a.photon_energy_fraction==b.photon_energy_fraction,"BremsFinalStateRecord first divergence: photon_energy_fraction");
  require(a.final_state_uniform==b.final_state_uniform,"BremsFinalStateRecord first divergence: final_state_uniform");
  require(a.azimuth_uniform==b.azimuth_uniform,"BremsFinalStateRecord first divergence: azimuth_uniform");
  require(a.auxiliary_uniform==b.auxiliary_uniform,"BremsFinalStateRecord first divergence: auxiliary_uniform");
  require(a.lpm_survival_probability==b.lpm_survival_probability,"BremsFinalStateRecord first divergence: lpm_survival_probability");
  require(a.lpm_uniform==b.lpm_uniform,"BremsFinalStateRecord first divergence: lpm_uniform");
  require(a.final_state_draw_id==b.final_state_draw_id,"BremsFinalStateRecord first divergence: final_state_draw_id");
  require(a.azimuth_draw_id==b.azimuth_draw_id,"BremsFinalStateRecord first divergence: azimuth_draw_id");
  require(a.auxiliary_draw_id==b.auxiliary_draw_id,"BremsFinalStateRecord first divergence: auxiliary_draw_id");
  require(a.lpm_draw_id==b.lpm_draw_id,"BremsFinalStateRecord first divergence: lpm_draw_id");
  require(a.thinning_status==b.thinning_status,"BremsFinalStateRecord first divergence: thinning_status");
  require(a.thinning_keep_mask==b.thinning_keep_mask,"BremsFinalStateRecord first divergence: thinning_keep_mask");
  require(a.thinning_first_uniform==b.thinning_first_uniform,"BremsFinalStateRecord first divergence: thinning_first_uniform");
  require(a.thinning_second_uniform==b.thinning_second_uniform,"BremsFinalStateRecord first divergence: thinning_second_uniform");
  require(a.thinning_first_draw_id==b.thinning_first_draw_id,"BremsFinalStateRecord first divergence: thinning_first_draw_id");
  require(a.thinning_second_draw_id==b.thinning_second_draw_id,"BremsFinalStateRecord first divergence: thinning_second_draw_id");
  require(a.weighted_mass_convention_correction_GeV==b.weighted_mass_convention_correction_GeV,"BremsFinalStateRecord first divergence: weighted_mass_convention_correction_GeV");
}
template<class T> void compareVectors(std::vector<T> const& a, std::vector<T> const& b) {
  require(a.size() == b.size(), "record count mismatch");
  for (std::size_t i=0;i<a.size();++i) compare(a[i],b[i]);
}
template<class V, class Ex> auto download(V const& data, std::size_t count, Ex const& ex) {
  auto slice = Kokkos::subview(data, std::make_pair(std::size_t{0}, count));
  auto mirror = Kokkos::create_mirror_view(slice);
  Kokkos::deep_copy(ex, mirror, slice); ex.fence("diagnostic photon result download");
  std::vector<typename V::non_const_value_type> result(count);
  for (std::size_t i=0;i<count;++i) result[i] = mirror(i);
  return result;
}
template<class Ex> struct PackNext {
  kd::ParticleSoARawView source;
  Kokkos::View<em::EmParticleState*, typename Ex::memory_space> target;
  KOKKOS_INLINE_FUNCTION void operator()(std::size_t i) const { target(i) = source.load(i); }
};
template<class Workspace, class Ex> auto downloadNext(
    Workspace const& workspace, std::size_t count, Ex const& ex) {
  Kokkos::View<em::EmParticleState*, typename Ex::memory_space> temp("front_test_next",count);
  Kokkos::parallel_for("test_pack_photon_next",Kokkos::RangePolicy<Ex>(ex,0,count),
                      PackNext<Ex>{workspace.queue.next().rawDeviceView(),temp});
  return download(temp,count,ex);
}


template<class Ex> struct Fixture {
  Ex ex{};
  kd::KokkosProposalNativeTable<Ex> table;
  kd::KokkosMoliereInterpolation<Ex> moliere;
  kd::KokkosPhysicsContext<Ex> context;
  explicit Fixture(CooperativeLeptonFixture const& h) {
    table.initialize(h.tables);
    auto path=h.auxiliary.cache_file;
    if(!path.empty()) path += ".moliere-initial-v1.c8cache";
    moliere.initialize(h.auxiliary.electron_moliere,path);
    em::tables::NativePhysicsView physics{};
    physics.proposal_native=table.deviceView();
    physics.em_transport_cut_MeV=.4;
    physics.muon_transport_cut_MeV=300.;
    struct Box {
      std::vector<corsika::geometry_detail::ConvexPlane> planes;
      auto const& exportPlanes() const { return planes; }
    } box{{{1.,0.,0.,1000.},{-1.,0.,0.,1000.},{0.,1.,0.,1000.},
           {0.,-1.,0.,1000.},{0.,0.,1.,1000.},{0.,0.,-1.,1000.}}};
    auto env=em::makeHomogeneousConvexSnapshot(.001,box);
    auto const& a=h.auxiliary;
    context.initialize(physics,env,a.electron_moliere,a.muon_moliere,
        moliere.deviceView(),a.has_muon_moliere!=0,a.photon_pair_lpm,a.brems_lpm,
        {},{},26091022,0,ex);
  }
};
std::vector<em::EmParticleState> inputs(std::size_t n,unsigned seed) {
  std::vector<em::EmParticleState> v(n);
  for(std::size_t i=0;i<n;++i) {
    auto& p=v[i];p.pid=std::array<int,4>{11,-11,13,-13}[i%4];
    p.energy_GeV=1.+10.*(i%199); p.direction[0]=1.;p.weight=1.;
    p.history_id=1+i+seed*100000ULL;p.generation=1;
    if(i%7==0) p.energy_GeV=(std::abs(p.pid)==11 ? .0007:.2);
    if(i%11==0) p.energy_GeV=1.e18;
    if(i%13==0) p.position_m[0]=999.99999;
    if(i%17==0) p.time_s=.011;
  }
  return v;
}
struct Output {
  std::vector<em::LeptonTransportRecord> steps;
  std::vector<em::BremsFinalStateRecord> records;
  std::vector<em::EmInteractionRecord> interactions;
  std::vector<em::EmParticleState> photons, remaining, decays;
  std::vector<em::ProposalFallbackEvent> fallbacks;
  std::vector<em::ObservationRecord> observations;
  std::vector<std::uint64_t> statistics;
  std::optional<em::GpuFirstInteractionSnapshot> first;
};
template<class T> void append(std::vector<T>& a,std::vector<T> const& b) {
  a.insert(a.end(),b.begin(),b.end());
}
template<class Frame> Output collect(Frame& frame) {
  Output o;
  frame.consume([&](auto const& c, auto const& w, auto const& ex) {
    auto const& t=c.totals.values;
    o.steps=download(w.steps,t[kd::LeptonStepOffset],ex);
    o.records=download(w.records,t[kd::LeptonRecordOffset],ex);
    o.photons=download(w.photons,t[kd::LeptonPhotonOffset],ex);
    o.decays=download(w.decays,t[kd::LeptonDecayOffset],ex);
    o.observations=download(w.observations,t[kd::LeptonObservationOffset],ex);
    auto n=t[kd::LeptonSelectionFallbackOffset]+t[kd::LeptonTransportFallbackOffset]+
           t[kd::LeptonVertexFallbackOffset]+t[kd::LeptonFinalStateFallbackOffset];
    o.fallbacks=download(w.fallbacks,n,ex);
    for(auto const& v:download(w.vertices,c.interaction_count,ex))
      if(v.interaction_flag) o.interactions.push_back(v.record);
    o.statistics=download(w.call_statistics,w.call_statistics.extent(0),ex);
    auto candidates=download(w.first_interaction_candidates,1,ex);
    require(candidates[0]<=1,"duplicate first-interaction candidate");
    if(candidates[0]) o.first=download(w.first_snapshots,1,ex)[0];
  });
  return o;
}
void compare(Output const& a,em::ResidentLeptonCascadeResult const& b) {
  compareVectors(a.steps,b.step_records);compareVectors(a.records,b.final_state_records);
  compareVectors(a.interactions,b.interaction_records);compareVectors(a.photons,b.generated_photons);
  compareVectors(a.remaining,b.remaining_leptons);compareVectors(a.decays,b.decay_candidates);
  compareVectors(a.fallbacks,b.fallback_events);compareVectors(a.observations,b.observations);
}
template<class Ex> void validate(Fixture<Ex>& f,std::size_t n,std::size_t rounds,
                                bool low_positrons=false) {
  auto in=inputs(n,31);
  if(low_positrons)
    for(std::size_t i=0;i<in.size();++i) {
      auto& p=in[i];p.pid=-11;p.energy_GeV=.0015+.0001*(i%10);
      p.time_s=0.;p.position_m[0]=0.;
    }
  if(n>1) in[1].generation=0; // The only eligible first-primary snapshot.
  std::optional<em::GpuFirstInteractionSnapshot> first,first_after;
  kd::KokkosResidentLeptonWorkspace<Ex> baseline_workspace;
  auto before=kd::runResidentLeptonCascadeBefore<Ex>(f.context.deviceView(),em::ElectronMassGeV,
      true,in,1000000000,rounds,2000000000,1,first,false,{},nullptr,nullptr,
      &baseline_workspace,nullptr,nullptr,0,std::numeric_limits<std::size_t>::max(),f.ex,true);
  auto after=kd::runResidentLeptonCascade<Ex>(f.context.deviceView(),em::ElectronMassGeV,
      true,in,1000000000,rounds,2000000000,1,first_after,false,{},nullptr,nullptr,
      nullptr,nullptr,nullptr,0,std::numeric_limits<std::size_t>::max(),f.ex,true);
  Output synchronous{after.step_records,after.final_state_records,after.interaction_records,
      after.generated_photons,after.remaining_leptons,after.decay_candidates,after.fallback_events,
      after.observations,{}};
  compare(synchronous,before);
  kd::KokkosLeptonFrontSubmission<Ex> frame(std::max<std::size_t>(n*3,1),
      f.context.deviceView(),em::ElectronMassGeV,true,f.ex);
  rejects([&]{frame.submit();});
  frame.prepare(in,1000000000,2000000000,false,true);
  rejects([&]{frame.prepare(in,1000000000,2000000000);});
  Output all;
  std::size_t fronts=0;
  do {
    frame.submit();
    rejects([&]{frame.submit();});
    while(!frame.poll()) std::this_thread::yield();
    rejects([&]{collect(frame);});
    frame.submitAccumulation();
    rejects([&]{frame.submitAccumulation();});
    while(!frame.poll()) std::this_thread::yield();
    auto one=collect(frame);
    rejects([&]{collect(frame);});
    append(all.steps,one.steps);append(all.records,one.records);
    append(all.interactions,one.interactions);append(all.photons,one.photons);
    append(all.decays,one.decays);append(all.fallbacks,one.fallbacks);
    append(all.observations,one.observations);
    all.statistics=std::move(one.statistics);
    all.first=one.first;
    ++fronts;
  } while(fronts<rounds && frame.continueResident());
  all.remaining=frame.takeRemaining();
  require(frame.secondaryHistoryIdsUsed()==before.secondary_history_ids_used,
          "cooperative history reservation accounting differs");
  rejects([&]{frame.takeRemaining();});
  compare(all,before);
  require(all.first.has_value()==first.has_value() && first_after.has_value()==first.has_value(),
          "first interaction presence differs");
  if(first) {compare(*all.first,*first);compare(*first_after,*first);}
  if(n) require(all.statistics==download(baseline_workspace.call_statistics,
      baseline_workspace.call_statistics.extent(0),f.ex),"statistics first divergence");
  auto bytes=frame.deviceBytes();
  for(int repeat=0;repeat<32;++repeat) {
    frame.prepare(in,1000000000,2000000000);frame.submit();
    while(!frame.poll()) std::this_thread::yield();
    frame.submitAccumulation();while(!frame.poll()) std::this_thread::yield();
    (void)collect(frame);(void)frame.takeRemaining();
    require(frame.deviceBytes()==bytes,"lepton workspace grew across reuse");
  }
  std::set<int> processes;
  for(auto const& r:all.records) processes.insert(r.process_id);
  if(low_positrons) require(processes.count(em::AnnihilationProcessId),
                            "low-energy test did not exercise positron annihilation");
  std::cout<<"endpoint="<<Ex::name()<<" sources="<<n<<" fronts="<<fronts
      <<" low_positrons="<<low_positrons
      <<" steps="<<all.steps.size()<<" finals="<<all.records.size()
      <<" fallbacks="<<all.fallbacks.size()<<" remaining="<<all.remaining.size()
      <<" workspace_bytes="<<bytes<<" processes=";
  for(auto p:processes) std::cout<<p<<",";
  std::cout<<" exact=true reuse=32\n"<<std::flush;
}

template<class Ex> void validateFrozenProjection(Fixture<Ex>& f) {
  Accumulators<Ex> original, split;
  auto in=inputs(4096,63);
  for(auto& p:in) if(p.energy_GeV>1.e10) p.time_s=0.;
  std::optional<em::GpuFirstInteractionSnapshot> first;
  auto before=kd::runResidentLeptonCascadeBefore<Ex>(f.context.deviceView(),em::ElectronMassGeV,
      true,in,5000000000ULL,1,6000000000ULL,1,first,false,original.projection,
      &original.profile,&original.radio,nullptr,nullptr,nullptr,0,
      std::numeric_limits<std::size_t>::max(),f.ex,true);
  kd::KokkosLeptonFrontSubmission<Ex> frame(in.size(),f.context.deviceView(),
      em::ElectronMassGeV,true,f.ex,0,split.projection,&split.profile,&split.radio);
  frame.prepare(in,5000000000ULL,6000000000ULL);frame.submit();
  while(!frame.poll())std::this_thread::yield();
  frame.submitAccumulation();while(!frame.poll())std::this_thread::yield();
  auto actual=collect(frame);actual.remaining=frame.takeRemaining();
  compareVectors(actual.interactions,before.interaction_records);
  compareVectors(actual.photons,before.generated_photons);
  compareVectors(actual.remaining,before.remaining_leptons);
  compareVectors(actual.fallbacks,before.fallback_events);
  compareVectors(actual.observations,before.observations);
  compareVectors(actual.decays,before.decay_candidates);
  require(before.step_records.empty() && before.final_state_records.empty(),
          "resident reference unexpectedly returned already-accumulated records");
  auto pr=original.profile.downloadFixed("frozen",ed::CooperativeEndpoint::Cuda,original.ex);
  auto ps=split.profile.downloadFixed("frozen",ed::CooperativeEndpoint::Cuda,split.ex);
  auto rr=original.radio.downloadFixed("frozen",ed::CooperativeEndpoint::Cuda,original.ex);
  auto rs=split.radio.downloadFixed("frozen",ed::CooperativeEndpoint::Cuda,split.ex);
  require(pr.histograms==ps.histograms && pr.counters.steps==ps.counters.steps &&
          pr.counters.deposited_steps==ps.counters.deposited_steps,
          "frozen profile differs from asynchronous accumulation");
  require(rr.coreas==rs.coreas && rr.zhs==rs.zhs &&
          rr.counters.valid_tracks==rs.counters.valid_tracks &&
          rr.counters.coreas_contributions==rs.counters.coreas_contributions &&
          rr.counters.zhs_contributions==rs.counters.zhs_contributions,
          "frozen CoREAS/ZHS differs from asynchronous accumulation");
  std::cout<<"endpoint="<<Ex::name()<<" frozen_profile_radio=exact\n"<<std::flush;
}

template<class Ex> void validateFailureLifecycle(Fixture<Ex>& f) {
  using Frame=kd::KokkosLeptonFrontSubmission<Ex>;
  auto in=inputs(32,55);
  Frame limited(32,f.context.deviceView(),em::ElectronMassGeV,true,f.ex);
  rejects([&]{limited.prepare(in,9000000000ULL,9000000001ULL);});
  auto bad=in;bad[0].pid=22;
  rejects([&]{limited.prepare(bad,9000000000ULL,9001000000ULL);});
  limited.prepare(in,9000000000ULL,9001000000ULL);
  limited.submit();while(!limited.poll())std::this_thread::yield();
  limited.submitAccumulation();while(!limited.poll())std::this_thread::yield();
  auto one=collect(limited);
  if(limited.remainingCount())
    rejects([&]{limited.prepare(in,9000000000ULL,9001000000ULL);});
  auto remaining=limited.takeRemaining();
  require(remaining.size()==limited.remainingCount(),"checkpoint lost particles");
  rejects([&]{limited.continueResident();});

  for(bool after_accumulation:{false,true}) {
    // Destruction must drain only this frame's execution, including an
    // unpolled submission. No dangling pinned control/output allocations.
    Frame abandoned(32,f.context.deviceView(),em::ElectronMassGeV,true,f.ex);
    abandoned.prepare(in,9000000000ULL,9001000000ULL);abandoned.submit();
    if(after_accumulation) {
      while(!abandoned.poll())std::this_thread::yield();
      abandoned.submitAccumulation();
    }
  }
  limited.prepare(in,9000000000ULL,9001000000ULL);limited.submit();
  while(!limited.poll())std::this_thread::yield();
  limited.submitAccumulation();while(!limited.poll())std::this_thread::yield();
  rejects([&]{limited.consume([&](auto const&,auto const&,auto const&){
    rejects([&]{collect(limited);}); // Reentrant output commit is forbidden.
    throw std::runtime_error("injected lepton output failure");
  });});
  require(limited.phase()==Frame::Phase::Failed,"output failure did not poison frame");
  rejects([&]{limited.submit();});rejects([&]{limited.takeRemaining();});
  rejects([&]{limited.prepare(in,9000000000ULL,9001000000ULL);});
  std::cout<<"endpoint="<<Ex::name()<<" failure_lifecycle=pass\n"<<std::flush;
}

#if defined(CORSIKA8_KOKKOS_BACKEND_CUDA_OPENMP)
void validateOverlap(CooperativeLeptonFixture const& host) {
  Fixture<Kokkos::Cuda> gpu(host);Fixture<Kokkos::OpenMP> cpu(host);
  Accumulators<Kokkos::Cuda> ga;Accumulators<Kokkos::OpenMP> ca;
  auto gi=inputs(16384,41),ci=inputs(2048,73);
  // Out-of-table inputs belong to the fallback test, not to this fixture's
  // 1e6 GeV deposition budget. Do not also turn a 1e18 GeV fallback into a
  // time-cut deposit (the separate kernel regression covers that case).
  for(auto* input:{&gi,&ci})
    for(auto& p:*input) if(p.energy_GeV>1.e10) p.time_s=0.;
  kd::KokkosLeptonFrontSubmission<Kokkos::Cuda> gf(gi.size(),gpu.context.deviceView(),
      em::ElectronMassGeV,true,gpu.ex,0,ga.projection,&ga.profile,&ga.radio);
  kd::KokkosLeptonFrontSubmission<Kokkos::OpenMP> cf(ci.size(),cpu.context.deviceView(),
      em::ElectronMassGeV,true,cpu.ex,0,ca.projection,&ca.profile,&ca.radio);
  auto finish=[](auto& f) {
    while(!f.poll())std::this_thread::yield();
    f.submitAccumulation();
    while(!f.poll())std::this_thread::yield();
    auto result=collect(f);result.remaining=f.takeRemaining();return result;
  };
  auto warm=[&](auto& f,auto const& input,std::uint64_t start) {
    f.prepare(input,start,start+1000000000);f.submit();return finish(f);
  };
  auto expected_g=warm(gf,gi,2000000000);auto expected_c=warm(cf,ci,3000000000);
  auto rg=ga.radio.downloadFixed("overlap",ed::CooperativeEndpoint::Cuda,ga.ex);
  auto rc=ca.radio.downloadFixed("overlap",ed::CooperativeEndpoint::OpenMP,ca.ex);
  for(auto const* r:{&rg,&rc})
    for(auto const* v:{&r->coreas,&r->zhs})
      require(std::any_of(v->begin(),v->end(),[](auto x){return x!=0;}),
              "real lepton radio test must have nonzero signals on both endpoints");
  auto pg=ga.profile.downloadFixed("overlap",ed::CooperativeEndpoint::Cuda,ga.ex);
  auto pc=ca.profile.downloadFixed("overlap",ed::CooperativeEndpoint::OpenMP,ca.ex);
  ga.radio.reset(ga.ex);ca.radio.reset(ca.ex);
  ga.profile.reset(1.e6,1.e6,ga.ex);ca.profile.reset(1.e6,1.e6,ca.ex);
  gf.prepare(gi,2000000000,3000000000);cf.prepare(ci,3000000000,4000000000);
#ifdef C8_COOPERATIVE_CUPTI
  c8_overlap_trace::start("enqueueResidentLeptonFront");
  gf.submit();
  auto host_start=c8_overlap_trace::timestamp();
  cf.submit();
  auto host_end=c8_overlap_trace::timestamp();
  auto actual_c=finish(cf);
  auto actual_g=finish(gf);
  c8_overlap_trace::finish(host_start,host_end);
#else
  gf.submit();cf.submit();auto actual_c=finish(cf);auto actual_g=finish(gf);
#endif
  auto same=[](Output const& a,Output const& b) {
    compareVectors(a.steps,b.steps);compareVectors(a.records,b.records);
    compareVectors(a.interactions,b.interactions);compareVectors(a.photons,b.photons);
    compareVectors(a.remaining,b.remaining);compareVectors(a.decays,b.decays);
    compareVectors(a.observations,b.observations);compareVectors(a.fallbacks,b.fallbacks);
    require(a.statistics==b.statistics,"overlap changes lepton statistics");
  };
  same(actual_g,expected_g);same(actual_c,expected_c);
  auto ag=ga.radio.downloadFixed("overlap",ed::CooperativeEndpoint::Cuda,ga.ex);
  auto ac=ca.radio.downloadFixed("overlap",ed::CooperativeEndpoint::OpenMP,ca.ex);
  require(ag.coreas==rg.coreas && ag.zhs==rg.zhs && ac.coreas==rc.coreas && ac.zhs==rc.zhs,
          "overlap changes endpoint radio integers");
  rd::CooperativeRadioMerge radio_merge("overlap",radioConfig());
  radio_merge.commit(ag);radio_merge.commit(ac);auto radio_result=radio_merge.take();
  auto mg=ga.profile.downloadFixed("overlap",ed::CooperativeEndpoint::Cuda,ga.ex);
  auto mc=ca.profile.downloadFixed("overlap",ed::CooperativeEndpoint::OpenMP,ca.ex);
  require(mg.histograms==pg.histograms && mc.histograms==pc.histograms,
          "overlap changes endpoint profile integers");
  ed::CooperativeProfileMerge profile_merge("overlap",profileConfig());
  profile_merge.commit(mg);profile_merge.commit(mc);auto profile_result=profile_merge.take();
  require(radio_result.counters.valid_tracks==rg.counters.valid_tracks+rc.counters.valid_tracks,
          "dual radio track ledger mismatch");
  require(profile_result.counters.steps==pg.counters.steps+pc.counters.steps,
          "dual profile track ledger mismatch");
  std::cout<<"real_lepton_overlap endpoints=2 physics_equal=true profile_equal=true radio_equal=true"
           <<" cuda_tracks="<<rg.counters.valid_tracks<<" openmp_tracks="<<rc.counters.valid_tracks
           <<" cuda_front_bytes="<<gf.deviceBytes()<<" openmp_front_bytes="<<cf.deviceBytes()
           <<"\n"<<std::flush;
}
#endif

int main(int argc,char** argv) {
  try {
    if(argc!=2) throw std::invalid_argument("expected existing PROPOSAL cache directory");
    auto host=loadCooperativeLeptonFixture(argv[1]);
    std::cout<<"native_columns="<<host.tables.dndx_columns.size()
             <<" aux_cache_hit="<<host.auxiliary.cache_hit<<"\n"<<std::flush;
    corsika::accelerator::em::KokkosRuntimeConfig config;
#if defined(CORSIKA8_KOKKOS_BACKEND_CUDA_OPENMP)
    config.execution_backend="cuda";config.threads=4;config.cooperative_owner=true;
#elif defined(KOKKOS_ENABLE_CUDA)
    config.execution_backend="cuda";
#else
    config.execution_backend="openmp";config.threads=4;
#endif
    corsika::accelerator::em::KokkosRuntime runtime(config);
#if defined(KOKKOS_ENABLE_CUDA)
    {Fixture<Kokkos::Cuda> fixture(host);
     for(auto n:{0u,1u,257u,4096u}) validate(fixture,n,1);
     validate(fixture,257,8);validate(fixture,4096,4,true);
     validateFrozenProjection(fixture);
     validateFailureLifecycle(fixture);}
#endif
#if defined(KOKKOS_ENABLE_OPENMP)
    {Fixture<Kokkos::OpenMP> fixture(host);
     for(auto n:{0u,1u,257u,4096u}) validate(fixture,n,1);
     validate(fixture,257,8);validate(fixture,4096,4,true);
     validateFrozenProjection(fixture);
     validateFailureLifecycle(fixture);}
#endif
#if defined(CORSIKA8_KOKKOS_BACKEND_CUDA_OPENMP)
    validateOverlap(host);
#endif
    return 0;
  } catch(std::exception const& e) {std::cerr<<e.what()<<"\n";return 1;}
}

/* (c) Copyright 2026 CORSIKA Project; distributed under the BSD-3-Clause license. */
#include <Kokkos_Core.hpp>
#include <corsika/accelerator/em/KokkosRuntime.hpp>
#include <corsika/accelerator/em/kokkos/KokkosPhotonFrontSubmission.hpp>
#include <corsika/accelerator/em/common/ConvexEnvironmentSnapshot.hpp>
#include "ResidentPhotonCascadeBefore.hpp" // frozen source, generated before the refactor
#include <chrono>
#include <iostream>
#include <thread>
#ifdef C8_COOPERATIVE_CUPTI
#include "CooperativeActivityTrace.hpp"
#endif

namespace kd = corsika::accelerator::em::kokkos_detail;
namespace em = corsika::gpu::em;
namespace nt = em::tables;

void require(bool value, char const* message) {
  if (!value) throw std::runtime_error(message);
}
template<class F> void rejects(F&& f) {
  bool rejected = false;
  try { f(); } catch (std::exception const&) { rejected = true; }
  require(rejected, "invalid state was accepted");
}

// Analytic native-spline fixture, NOT a medium/PROPOSAL accuracy oracle.
// Actual production selection, Compton, tracking, cuts, scans, materialization
// and history-keyed random draws are exercised without rebuilding any tables.
template<class Ex> struct Fixture {
  using Mem = typename Ex::memory_space;
  Kokkos::View<nt::NativeDndxColumn*, Mem> dndx{"front_dndx", 1};
  Kokkos::View<nt::NativeTotalRateColumn*, Mem> total{"front_total", 1};
  Kokkos::View<std::uint32_t*, Mem> indices{"front_indices", 1};
  Kokkos::View<double*, Mem> values{"front_values", 4}, dx{"front_dx", 4},
      dy{"front_dy", 4}, dxy{"front_dxy", 4}, cubic{"front_cubic", 2},
      dcubic{"front_dcubic", 2};
  Kokkos::View<kd::KokkosPhysicsContextData, Mem> context{"front_physics"};
  Ex execution{};
  int process_id;
  explicit Fixture(Ex ex = {}, int process = em::ComptonProcessId)
      : execution(ex), process_id(process) {
    auto h = Kokkos::create_mirror_view(dndx);
    auto& c = h(0);
    c = {};
    c.pdg_id = 22; c.process_id = process;
    c.component_hash = 7; c.component_nuclear_charge = 7.;
    c.component_atomic_mass = 14.;
    c.loss_transform = nt::NativeLossTransform::OneMinusLog;
    c.kinematic_model = nt::NativeKinematicModel::Compton;
    if(process==em::PhotonPairProcessId)
      c.kinematic_model=nt::NativeKinematicModel::PhotonPairFullLoss;
    if(process==em::PhotoelectricProcessId)
      c.kinematic_model=nt::NativeKinematicModel::OnlyStochastic;
    c.lower_energy_limit_MeV = 1.;
    c.spline.energy_axis = {nt::NativeAxisType::Linear, 2, 1., 1.e6, 999999.};
    c.spline.loss_axis = {nt::NativeAxisType::Linear, 2, 0., 1., 1.};
    c.spline.rows = 2; c.spline.columns = 2;
    Kokkos::deep_copy(execution, dndx, h);
    auto ht = Kokkos::create_mirror_view(total);
    ht(0) = {}; ht(0).pdg_id = 22; ht(0).lower_energy_limit_MeV = 1.;
    ht(0).spline.axis = c.spline.energy_axis;
    ht(0).spline.coefficient_count = 2;
    Kokkos::deep_copy(execution, total, ht);
    auto hv = Kokkos::create_mirror_view(values);
    // CubicInterpolation index = energy_row + rows * loss_column.
    hv(0) = 0.; hv(1) = 0.; hv(2) = 1.; hv(3) = 1.;
    Kokkos::deep_copy(execution, values, hv);
    Kokkos::deep_copy(execution, indices, std::uint32_t{0});
    Kokkos::deep_copy(execution, dx, 0.); Kokkos::deep_copy(execution, dy, 1.);
    Kokkos::deep_copy(execution, dxy, 0.);
    Kokkos::deep_copy(execution, cubic, 1.); Kokkos::deep_copy(execution, dcubic, 0.);
    auto hc = Kokkos::create_mirror_view(context);
    hc() = {};
    auto& native = hc().physics.proposal_native;
    native.dndx_columns = dndx.data(); native.dndx_column_count = 1;
    native.total_rate_columns = total.data(); native.total_rate_column_count = 1;
    native.selection_column_indices = indices.data(); native.selection_column_index_count = 1;
    native.bicubic_values = values.data(); native.bicubic_derivative_energy = dx.data();
    native.bicubic_derivative_loss = dy.data(); native.bicubic_mixed_derivative = dxy.data();
    native.bicubic_coefficient_count = 4;
    native.cubic_values = cubic.data(); native.cubic_node_derivatives = dcubic.data();
    native.cubic_coefficient_count = 2;
    native.constants.electron_mass_MeV = em::ElectronMassGeV * 1000.;
    hc().physics.em_transport_cut_MeV = 1.;
    struct Box {
      std::vector<corsika::geometry_detail::ConvexPlane> planes;
      auto const& exportPlanes() const { return planes; }
    };
    Box box{{{1.,0.,0.,100.},{-1.,0.,0.,100.},{0.,1.,0.,100.},
             {0.,-1.,0.,100.},{0.,0.,1.,100.},{0.,0.,-1.,100.}}};
    hc().environment = em::makeHomogeneousConvexSnapshot(.001, box);
    hc().random_seed = 26091021; hc().shower_id = 0;
    // Positive, internally consistent diagnostic auxiliary parameters. These
    // are not an export of a physical air medium and are never used by apps.
    auto& lpm=hc().photon_pair_lpm;
    lpm.baseline_mass_density_g_per_cm3=.001; lpm.molecular_density_per_cm3=2.e19;
    lpm.sum_charge=7.; lpm.e_lpm_MeV=1.e15;
    lpm.classical_electron_radius_cm=2.8179403262e-13;
    lpm.fine_structure_constant=1./137.035999084;
    lpm.component_count=1;
    lpm.components[0].component_hash=7; lpm.components[0].nuclear_charge=7.;
    lpm.components[0].radiation_log_constant=183.;
    Kokkos::deep_copy(execution, context, hc);
    execution.fence("prepare analytic front fixture");
  }
};

std::vector<em::EmParticleState> inputs(std::size_t count, unsigned seed) {
  std::vector<em::EmParticleState> result(count);
  for (std::size_t i = 0; i < count; ++i) {
    auto& p = result[i]; p.pid = 22; p.energy_GeV = .01 + .01 * (i % 99);
    p.direction[0] = 1.; p.weight = 1.;
    p.history_id = 1 + i + seed * 100000ULL;
    p.generation = 1; // no shared generation-zero capture for this multi-primary fixture
    if (i % 7 == 0) p.energy_GeV = .0005; // cut
    if (i % 11 == 0) p.energy_GeV = 2000.; // out-of-range fallback
    if (i % 13 == 0) p.position_m[0] = 99.9999; // escaped volume
    if (i % 17 == 0) p.time_s = .011; // lifetime cut
  }
  return result;
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

struct FrontOutput {
  std::vector<em::PhotonTransportRecord> steps;
  std::vector<em::EmParticleState> charged, next;
  std::vector<em::PhotonFinalStateRecord> records;
  std::vector<em::ProposalFallbackEvent> fallbacks;
  std::vector<em::ObservationRecord> observations;
};
template<class Front> FrontOutput collect(Front& front) {
  FrontOutput result;
  front.consume([&](auto const& control, auto const& w, auto const& ex) {
    auto const& t=control.totals.values;
    result.steps=download(w.steps,t[kd::PhotonStepOffset],ex);
    result.charged=download(w.charged,t[kd::PhotonChargedOffset],ex);
    result.next=downloadNext(w,t[kd::PhotonNextOffset],ex);
    result.records=download(w.records,t[kd::PhotonRecordOffset],ex);
    result.fallbacks=download(w.fallbacks,t[kd::PhotonSelectionFallbackOffset]+
        t[kd::PhotonTransportFallbackOffset]+t[kd::PhotonFinalStateFallbackOffset],ex);
    result.observations=download(w.observations,t[kd::PhotonObservationOffset],ex);
  });
  return result;
}
void compare(FrontOutput const& a, FrontOutput const& b) {
  compareVectors(a.steps,b.steps); compareVectors(a.charged,b.charged);
  compareVectors(a.next,b.next); compareVectors(a.records,b.records);
  compareVectors(a.fallbacks,b.fallbacks); compareVectors(a.observations,b.observations);
}

template<class Ex> void validateResidentContinuation(Fixture<Ex>& fixture, unsigned seed) {
  constexpr std::size_t count=257, maximum_fronts=8;
  auto input=inputs(count,seed);
  input[0].generation=0; input[0].energy_GeV=.5;
  input[0].time_s=0.; input[0].position_m[0]=0.;
  std::optional<em::GpuFirstInteractionSnapshot> before_first, resumed_first;
  auto before=kd::runResidentPhotonCascadeBefore<Ex>(fixture.context,input,
      1000000000,maximum_fronts,1,before_first);
  kd::KokkosPhotonFrontSubmission<Ex> frame(count,fixture.context);
  frame.prepare(input,1000000000,false,true);
  FrontOutput accumulated;
  std::size_t fronts=0;
  auto append=[](auto& target,auto const& more) { target.insert(target.end(),more.begin(),more.end()); };
  do {
    frame.submit();
    while(!frame.poll()) std::this_thread::yield();
    // Capture remains resident across the entire continuation, just as in
    // the synchronous cascade. It must not be reset on every front.
    FrontOutput one;
    frame.consume([&](auto const& control,auto const& w,auto const& ex) {
      auto const& t=control.totals.values;
      one.steps=download(w.steps,t[kd::PhotonStepOffset],ex);
      one.charged=download(w.charged,t[kd::PhotonChargedOffset],ex);
      one.next=downloadNext(w,t[kd::PhotonNextOffset],ex);
      one.records=download(w.records,t[kd::PhotonRecordOffset],ex);
      one.fallbacks=download(w.fallbacks,t[kd::PhotonSelectionFallbackOffset]+
          t[kd::PhotonTransportFallbackOffset]+t[kd::PhotonFinalStateFallbackOffset],ex);
      one.observations=download(w.observations,t[kd::PhotonObservationOffset],ex);
      auto candidates=download(w.first_interaction_candidates,1,ex);
      require(candidates[0]<=1,"duplicate first primary vertex");
      if(candidates[0]) resumed_first=download(w.first_snapshots,1,ex)[0];
    });
    append(accumulated.steps,one.steps); append(accumulated.charged,one.charged);
    append(accumulated.records,one.records); append(accumulated.fallbacks,one.fallbacks);
    append(accumulated.observations,one.observations);
    accumulated.next=std::move(one.next);
    ++fronts;
  } while(fronts<maximum_fronts && frame.continueResident());
  require(fronts==before.wavefronts,"resident wavefront count changed");
  compareVectors(before.step_records,accumulated.steps);
  compareVectors(before.electromagnetic_secondaries,accumulated.charged);
  compareVectors(before.remaining_photons,accumulated.next);
  compareVectors(before.final_state_records,accumulated.records);
  compareVectors(before.fallback_events,accumulated.fallbacks);
  compareVectors(before.observations,accumulated.observations);
  require(before_first.has_value()==resumed_first.has_value(),"first vertex missing on resume");
  if(before_first) compare(*before_first,*resumed_first);
  require(fronts>1,"continuation fixture did not exercise resume");
  std::cout<<"PASS resident continuation: "<<Ex::name()<<" fronts="<<fronts
           <<" seed="<<seed<<" steps="<<accumulated.steps.size()<<'\n';

  // Abort during a real CUDA submission: destruction must drain just this
  // instance before releasing its private buffers. A subsequent frame is safe.
  {
    kd::KokkosPhotonFrontSubmission<Ex> abandoned(count,fixture.context);
    abandoned.prepare(input,1000000000); abandoned.submit();
  }
  frame.prepare(input,1000000000); frame.submit();
  while(!frame.poll()) std::this_thread::yield();
  rejects([&]{frame.consume([](auto const&,auto const&,auto const&){
    throw std::runtime_error("injected consumer failure");
  });});
  require(frame.phase()==kd::KokkosPhotonFrontSubmission<Ex>::Phase::Failed,
          "consumer failure did not poison frame");
  rejects([&]{frame.prepare(input,1000000000);});
}

template<class Ex> void validate(Fixture<Ex>& fixture, std::size_t count, unsigned seed) {
  auto input = inputs(count,seed);
  std::optional<em::GpuFirstInteractionSnapshot> old_first, new_first;
  auto before = kd::runResidentPhotonCascadeBefore<Ex>(fixture.context,input,
      1000000000,1,1,old_first);
  auto after = kd::runResidentPhotonCascade<Ex>(fixture.context,input,
      1000000000,1,1,new_first);
  compareVectors(before.step_records,after.step_records);
  compareVectors(before.electromagnetic_secondaries,after.electromagnetic_secondaries);
  compareVectors(before.remaining_photons,after.remaining_photons);
  compareVectors(before.final_state_records,after.final_state_records);
  compareVectors(before.fallback_events,after.fallback_events);
  compareVectors(before.observations,after.observations);
  require(old_first.has_value() == new_first.has_value(), "first vertex availability changed");
  if (old_first) compare(*old_first,*new_first);
  require(before.process_statistics.compton_final_states == after.process_statistics.compton_final_states,
          "synchronous Compton counter changed");
  if (count >= 256) require(before.process_statistics.gpu_final_states > 0,
                           "fixture did not execute real final-state kernels");

  kd::KokkosPhotonFrontSubmission<Ex> front(std::max<std::size_t>(count,1), fixture.context);
  rejects([&]{ front.submit(); }); rejects([&]{ front.poll(); });
  front.prepare(input,1000000000);
  rejects([&]{ front.prepare(input,1000000000); });
  front.submit();
  rejects([&]{ front.submit(); });
  rejects([&]{ front.consume([](auto const&,auto const&,auto const&){}); });
  while (!front.poll()) std::this_thread::yield();
  front.consume([&](auto const& control, auto const& workspace, auto const& ex) {
    auto const& t = control.totals.values;
    compareVectors(before.step_records, download(workspace.steps,t[kd::PhotonStepOffset],ex));
    compareVectors(before.electromagnetic_secondaries,
                   download(workspace.charged,t[kd::PhotonChargedOffset],ex));
    compareVectors(before.remaining_photons,downloadNext(workspace,t[kd::PhotonNextOffset],ex));
    compareVectors(before.final_state_records,download(workspace.records,t[kd::PhotonRecordOffset],ex));
    compareVectors(before.fallback_events,download(workspace.fallbacks,
        t[kd::PhotonSelectionFallbackOffset]+t[kd::PhotonTransportFallbackOffset]+
        t[kd::PhotonFinalStateFallbackOffset],ex));
    compareVectors(before.observations,download(workspace.observations,t[kd::PhotonObservationOffset],ex));
    require(before.fallback_events.size() == t[kd::PhotonSelectionFallbackOffset]+
            t[kd::PhotonTransportFallbackOffset]+t[kd::PhotonFinalStateFallbackOffset],
            "fallback count changed");
    require(before.observations.size() == t[kd::PhotonObservationOffset],"observation count changed");
    rejects([&]{front.prepare(input,1000000000);}); // callback cannot overwrite its own input
  });
  rejects([&]{ front.consume([](auto const&,auto const&,auto const&){}); });
  auto bytes = front.deviceBytes();
  for (unsigned repeat=0;repeat<32;++repeat) {
    front.prepare(input,1000000000); front.submit();
    while (!front.poll()) std::this_thread::yield();
    front.consume([](auto const&,auto const&,auto const&){});
    require(front.deviceBytes() == bytes, "retained workspace grew on identical inputs");
  }
  if (count) rejects([&]{front.prepare(input,std::numeric_limits<std::uint64_t>::max());});
  std::cout << "PASS exact synchronous/async photon front: " << Ex::name()
            << " inputs=" << count << " seed=" << seed << " repeat=32 bytes=" << bytes
            << " process=" << fixture.process_id
            << " final_states=" << before.process_statistics.gpu_final_states << '\n';
}

int main(int argc, char** argv) {
  try {
    corsika::accelerator::em::KokkosRuntimeConfig config;
#if defined(CORSIKA8_KOKKOS_BACKEND_CUDA_OPENMP)
    config.execution_backend = "cuda"; config.cooperative_owner = true; config.threads = 4;
#elif defined(CORSIKA8_KOKKOS_BACKEND_CUDA)
    config.execution_backend = "cuda";
#else
    config.execution_backend = "openmp"; config.threads = 4;
#endif
    corsika::accelerator::em::KokkosRuntime runtime(config);
    Fixture<Kokkos::DefaultExecutionSpace> fixture;
    for (auto count : {std::size_t{0}, std::size_t{1}, std::size_t{257}, std::size_t{4096}})
      validate(fixture,count,31);
    validateResidentContinuation(fixture,31);
    validateResidentContinuation(fixture,73);
    for(auto process : {em::PhotonPairProcessId,em::PhotoelectricProcessId}) {
      Fixture<Kokkos::DefaultExecutionSpace> other({},process);
      validate(other,257,31);
    }
#if defined(CORSIKA8_KOKKOS_BACKEND_CUDA_OPENMP)
    Fixture<Kokkos::OpenMP> host;
    for (auto count : {std::size_t{0}, std::size_t{1}, std::size_t{257}, std::size_t{4096}})
      validate(host,count,31);
    validateResidentContinuation(host,31);
    validateResidentContinuation(host,73);
    for(auto process : {em::PhotonPairProcessId,em::PhotoelectricProcessId}) {
      Fixture<Kokkos::OpenMP> other({},process);
      validate(other,257,31);
    }
    // Both endpoints have staged storage before either is submitted. The
    // main thread enqueues CUDA first, then executes a short real OpenMP
    // photon front. The GPU queue is NOT downloaded between these calls.
    kd::KokkosPhotonFrontSubmission<Kokkos::Cuda> device_front(16384,fixture.context);
    // Within the planned 256--2048 short-batch range. On a shared GPU the
    // initial 256-particle CPU batch can finish before the GPU is scheduled;
    // retain that diagnostic through an explicit option, never fake overlap.
    std::size_t const host_batch = argc==2 && std::string(argv[1])=="--host-batch=256" ? 256 : 2048;
    if(argc>2 || (argc==2 && std::string(argv[1])!="--host-batch=256"))
      throw std::invalid_argument("expected no arguments or --host-batch=256");
    kd::KokkosPhotonFrontSubmission<Kokkos::OpenMP> host_front(host_batch,host.context);
    auto gpu_input = inputs(16384,41), cpu_input = inputs(host_batch,42);
    FrontOutput expected_device, expected_host;
    for (unsigned warm=0;warm<2;++warm) {
      device_front.prepare(gpu_input,2000000000);
      host_front.prepare(cpu_input,3000000000);
      device_front.submit();
      while (!device_front.poll()) std::this_thread::yield();
      host_front.submit(); // synchronous reference: deliberately no overlap
      require(host_front.poll(),"OpenMP front not complete after synchronous submit");
      expected_device = collect(device_front);
      expected_host = collect(host_front);
    }
    device_front.prepare(gpu_input,2000000000);
    host_front.prepare(cpu_input,3000000000);
#ifdef C8_COOPERATIVE_CUPTI
    c8_overlap_trace::start("enqueueResidentPhotonFront");
#endif
    device_front.submit();
#ifdef C8_COOPERATIVE_CUPTI
    auto const host_start = c8_overlap_trace::timestamp();
#endif
    host_front.submit();
#ifdef C8_COOPERATIVE_CUPTI
    auto const host_end = c8_overlap_trace::timestamp();
#endif
    while (!device_front.poll()) std::this_thread::yield();
    require(host_front.poll(),"overlapping OpenMP front control failed");
    compare(expected_device,collect(device_front));
    compare(expected_host,collect(host_front));
#ifdef C8_COOPERATIVE_CUPTI
    c8_overlap_trace::finish(host_start,host_end);
#endif
    std::cout << "PASS real photon CUDA submission + OpenMP short front, host_batch="
              << host_batch << '\n';
#endif
    return 0;
  } catch(std::exception const& e) { std::cerr << e.what() << '\n'; return 1; }
}

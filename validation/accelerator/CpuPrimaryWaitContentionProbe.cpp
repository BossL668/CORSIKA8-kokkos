// Opt-in validation executable, not installed or used by a shower application.
// Fixed OpenMP inputs isolate CUDA submitter contention from scheduling/tree changes.
// Build with build_cpu_primary_isolation.py --source THIS_FILE against a v5 stage.
#include <corsika/accelerator/em/detail/IndependentEndpointDriver.hpp>
#include <corsika/accelerator/em/detail/CooperativeDriverAffinity.hpp>
#include <corsika/accelerator/em/detail/KokkosBackendInstance.hpp>
#include <corsika/accelerator/em/common/ConvexEnvironmentSnapshot.hpp>
#include <corsika/accelerator/em/common/tables/Sha256.hpp>
#include "CooperativeLeptonFixture.hpp"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <future>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <sys/resource.h>
#include <time.h>
#include <type_traits>

namespace em = corsika::gpu::em;
namespace acc = corsika::accelerator::em;
using Backend = acc::detail::KokkosBackendInstance;
using Clock = std::chrono::steady_clock;
using Json = nlohmann::ordered_json;

namespace {
void require(bool ok, char const* why) {
  if (!ok) throw std::runtime_error(why);
}
template<class F> void rejects(F&& f, char const* why) {
  bool rejected = false;
  try { f(); } catch (std::logic_error const&) { rejected = true; }
  require(rejected, why);
}
double seconds(Clock::time_point a, Clock::time_point b) {
  return std::chrono::duration<double>(b - a).count();
}
double threadSeconds() {
  timespec value{};
  require(clock_gettime(CLOCK_THREAD_CPUTIME_ID, &value) == 0,
          "thread CPU clock unavailable");
  return value.tv_sec + value.tv_nsec * 1.e-9;
}
double processSeconds() {
  rusage value{};
  require(getrusage(RUSAGE_SELF, &value) == 0, "process CPU accounting unavailable");
  return value.ru_utime.tv_sec + value.ru_stime.tv_sec +
         (value.ru_utime.tv_usec + value.ru_stime.tv_usec) * 1.e-6;
}
std::vector<int> allowedCpus() {
  cpu_set_t allowed;
  require(sched_getaffinity(0,sizeof(allowed),&allowed)==0,"read diagnostic thread affinity");
  std::vector<int> ids;
  for(int cpu=0;cpu<CPU_SETSIZE;++cpu) if(CPU_ISSET(cpu,&allowed)) ids.push_back(cpu);
  return ids;
}
std::uint64_t procBytes(char const* path, char const* key) {
  std::ifstream file(path);
  std::string line;
  while (std::getline(file, line)) {
    std::istringstream fields(line);
    std::string field;
    std::uint64_t kib{};
    if ((fields >> field >> kib) && field == key) return kib * 1024;
  }
  throw std::runtime_error(std::string("missing memory accounting: ") + key);
}
void memoryGate() {
  require(procBytes("/proc/meminfo", "MemAvailable:") >= (4ULL << 30),
          "less than 4 GiB system memory available; stop diagnostic");
}
template<class T> void append(std::vector<std::uint8_t>& bytes, T const& value) {
  static_assert(std::is_arithmetic_v<T>);
  if constexpr (std::is_floating_point_v<T>)
    require(std::isfinite(value), "nonfinite physics output");
  auto p = reinterpret_cast<std::uint8_t const*>(&value);
  bytes.insert(bytes.end(), p, p + sizeof(T));
}
template<class T> void vectorBytes(std::vector<std::uint8_t>& bytes,
                                   std::vector<T> const& values) {
  append(bytes, static_cast<std::uint64_t>(values.size()));
  for (auto const& value : values) append(bytes, value);
}
void particleBytes(std::vector<std::uint8_t>& b, em::EmParticleState const& p) {
  append(b,p.pid); append(b,p.medium_id); append(b,p.generation); append(b,p.reserved);
  append(b,p.energy_GeV);
  for (auto x:p.position_m) append(b,x);
  for (auto x:p.direction) append(b,x);
  append(b,p.time_s); append(b,p.weight); append(b,p.history_id);
  append(b,p.parent_history_id); append(b,p.step_id);
}
std::string digest(std::vector<std::uint8_t> const& b) {
  return em::tables::toHex(em::tables::sha256(b));
}
struct Input {
  std::vector<em::EmParticleState> photons, leptons;
  long double initial_energy{};
  std::uint64_t next_history{};
  std::string sha256;
};
Input makeInput(std::size_t count, double energy_MeV, std::uint64_t history_base) {
  Input input;
  std::vector<std::uint8_t> bytes;
  for (std::size_t i=0; i<count; ++i) {
    em::EmParticleState p{};
    p.pid = i%3 == 0 ? 22 : (i%3 == 1 ? 11 : -11);
    p.energy_GeV = (energy_MeV + .1*(i%10))*.001;
    p.history_id = history_base+i+1;
    p.direction[0] = 1.; p.weight = 1.; p.generation = 1;
    input.initial_energy += p.energy_GeV;
    particleBytes(bytes,p);
    (p.pid==22 ? input.photons : input.leptons).push_back(p);
  }
  input.next_history = history_base+1000000;
  input.sha256 = digest(bytes);
  return input;
}
// No Stack or PROPOSAL calculator is touched by this driver. Backend, queues,
// history leases and output ownership belong exclusively to its calling thread.
struct Work {
  Clock::time_point start, transport_end, end, output_start, output_end;
  double thread_cpu_seconds{}, transport_thread_cpu_seconds{};
  double output_thread_cpu_seconds{};
  double photon_seconds{}, lepton_seconds{};
  std::uint64_t photon_calls{}, lepton_calls{}, rounds{};
  std::uint64_t photon_steps{}, lepton_steps{};
  std::size_t photon_wave_limit{}, lepton_wave_limit{}, minimum_resident_batch{};
  std::vector<std::uint8_t> call_bytes;
  std::vector<em::EmParticleState> terminal;
  std::vector<em::ProposalFallbackEvent> fallback;
  long double initial_energy{}, returned_energy{};
  em::GpuProfileResult profile;
  corsika::gpu::radio::GpuRadioWaveforms radio;
  acc::detail::FixedProfileSnapshot fixed_profile;
  corsika::accelerator::radio::detail::FixedRadioSnapshot fixed_radio;
  em::GpuEmStatistics statistics;
  acc::detail::CooperativeCudaWaitStatistics waits;
  std::string input_sha256;
};
Work drive(Backend& backend, Input input, bool device) {
  Work out;
  // Match the standalone CPU and CPU-primary auxiliary CUDA call shapes.
  // One call remains a safe, complete endpoint call; this diagnostic does not
  // recreate packet scheduling or dynamically choose how much work to run.
  out.photon_wave_limit=16;
  out.lepton_wave_limit=device?8:1024;
  out.minimum_resident_batch=device?1:4096;
  out.initial_energy = input.initial_energy;
  out.input_sha256 = input.sha256;
  out.start = Clock::now();
  auto const thread_start = threadSeconds();
  auto receive = [&](auto const& result) {
    require(result.cpu_spill_particles.empty(), "bounded fixture unexpectedly spilled");
    for (auto const& q:result.observations) out.terminal.push_back(q.particle);
    for (auto const& q:result.fallback_events) {
      out.terminal.push_back(q.particle);
      out.fallback.push_back(q);
      out.returned_energy += static_cast<long double>(q.particle.weight)*q.particle.energy_GeV;
    }
  };
  while (!input.photons.empty() || !input.leptons.empty() ||
         backend.pendingPhotonCount() || backend.pendingLeptonCount()) {
    require(++out.rounds<100000, "fixed-input fixture did not drain");
    // Wall timeout is only a failure guard. Never truncate/select the workload.
    require(seconds(out.start,Clock::now())<600., "fixed-input fixture exceeded 600 s");
    if (out.rounds%64 == 0) memoryGate();
    auto advance = [&](bool photon) {
      auto& queue = photon ? input.photons : input.leptons;
      auto capacity = photon ? backend.maximumResidentPhotonBatchSize() :
                               backend.maximumResidentLeptonBatchSize();
      auto pending = std::min(capacity,photon ? backend.pendingPhotonCount() :
                                               backend.pendingLeptonCount());
      auto take = std::min(capacity-pending,queue.size());
      auto n = pending+take;
      if (!n) return;
      std::vector<em::EmParticleState> batch(queue.begin(),queue.begin()+take);
      queue.erase(queue.begin(),queue.begin()+take);
      auto const lease = photon ? 32*n :
        std::max<std::uint64_t>(3*n,std::min<std::uint64_t>(1U<<20,192*n));
      auto first = input.next_history;
      require(lease<=UINT64_MAX-first, "fixed-input history lease overflow");
      input.next_history += lease;
      append(out.call_bytes,photon); append(out.call_bytes,std::uint64_t(n));
      append(out.call_bytes,first); append(out.call_bytes,std::uint64_t(lease));
      append(out.call_bytes,std::uint64_t(photon?out.photon_wave_limit:out.lepton_wave_limit));
      append(out.call_bytes,std::uint64_t(out.minimum_resident_batch));
      auto t = Clock::now();
      if (photon) {
        auto result = backend.runPhotonWavefront(
          batch,first,out.photon_wave_limit,out.minimum_resident_batch);
        out.photon_seconds += seconds(t,Clock::now()); ++out.photon_calls;
        out.photon_steps += result.transport_records;
        receive(result);
        queue.insert(queue.end(),result.remaining_photons.begin(),result.remaining_photons.end());
        input.leptons.insert(input.leptons.end(),result.electromagnetic_secondaries.begin(),
                             result.electromagnetic_secondaries.end());
      } else {
        auto result = backend.runLeptonWavefront(
          batch,first,out.lepton_wave_limit,input.next_history,out.minimum_resident_batch);
        out.lepton_seconds += seconds(t,Clock::now()); ++out.lepton_calls;
        out.lepton_steps += result.transport_records;
        receive(result);
        require(result.decay_candidates.empty(), "electron fixture produced muon decay");
        queue.insert(queue.end(),result.remaining_leptons.begin(),result.remaining_leptons.end());
        input.photons.insert(input.photons.end(),result.generated_photons.begin(),
                             result.generated_photons.end());
      }
    };
    advance(true); advance(false);
  }
  out.transport_end = Clock::now();
  out.transport_thread_cpu_seconds = threadSeconds()-thread_start;
  out.waits = backend.cooperativeWaitStatistics();
  out.end = Clock::now();
  out.thread_cpu_seconds = threadSeconds()-thread_start;
  return out;
}
// Called only after BOTH transport jobs end. Download on the proper endpoint
// thread, but never let final profile/radio transfer or hashing create extra
// contention while measuring the other endpoint's transport.
void collect(Backend& backend, Work& out, bool radio, bool device) {
  out.output_start = Clock::now();
  auto thread_start=threadSeconds();
  auto endpoint = device ? acc::detail::CooperativeEndpoint::Cuda :
                           acc::detail::CooperativeEndpoint::OpenMP;
  // Fixed and ordinary downloads consume the same accumulator. Follow the
  // real coordinator: take ONE fixed snapshot, then decode its host copy.
  // These two hashes are not independent validation of two download paths.
  out.fixed_profile = backend.downloadFixedProfile("fixed-wait-contention",endpoint);
  out.profile = acc::detail::decodeFixedProfile(out.fixed_profile);
  if (radio) {
    out.fixed_radio = backend.downloadFixedRadio("fixed-wait-contention",endpoint);
    out.radio = corsika::accelerator::radio::detail::decodeFixedRadio(out.fixed_radio);
  }
  out.statistics = backend.statistics();
  out.output_end = Clock::now();
  out.output_thread_cpu_seconds = threadSeconds()-thread_start;
}
Json audit(Work& w, bool radio) {
  std::sort(w.terminal.begin(),w.terminal.end(),[](auto const& a,auto const& b) {
    return a.history_id<b.history_id;
  });
  for (std::size_t i=1;i<w.terminal.size();++i)
    require(w.terminal[i-1].history_id!=w.terminal[i].history_id,"duplicate terminal history");
  std::sort(w.fallback.begin(),w.fallback.end(),[](auto const& a,auto const& b) {
    return a.particle.history_id<b.particle.history_id;
  });
  auto const& p = w.profile;
  require(p.steps && !p.fixed_point_overflows && !p.invalid_records, "invalid profile");
  acc::detail::validateFixedProfile(w.fixed_profile);
  std::vector<std::uint8_t> profile, wave, terminal, fallback;
  vectorBytes(profile,w.fixed_profile.histograms);
  auto const& c=w.fixed_profile.counters;
#define PC(field) append(profile,c.field)
  PC(steps); PC(deposited_steps); PC(photon_cuts);
  for (auto v:c.lepton_limits) append(profile,v);
  PC(moliere_trials); PC(moliere_deflections); PC(moliere_zero_deflections);
  PC(moliere_newton_iterations); PC(moliere_max_newton_iterations);
  PC(thinning_hillas_vertices); PC(thinning_statistical_vertices);
  PC(thinning_particles_discarded); PC(fixed_point_overflows); PC(invalid_records);
  PC(weighted_medium_rest_mass_input); PC(weighted_cut_rest_mass_energy);
  PC(weighted_observed_total_energy); PC(weighted_escaped_total_energy);
  PC(weighted_unwritten_photoelectric_binding_energy);
  PC(weighted_observation_cut_overlap_energy); PC(weighted_mass_convention_correction);
#undef PC
  append(profile,w.fixed_profile.weight_units); append(profile,w.fixed_profile.energy_units);
  for (auto const* v:{&p.photons,&p.electrons,&p.positrons,&p.muons_minus,&p.muons_plus,
                      &p.muon_parent_productions,&p.energy_loss_GeV,&p.muon_energy_loss_GeV})
    vectorBytes(profile,*v);
  for (auto const& q:w.terminal) particleBytes(terminal,q);
  Json fallback_reasons=Json::object();
  for (auto const& f:w.fallback) {
    particleBytes(fallback,f.particle);
#define FC(field) append(fallback,f.field)
    FC(input_index); FC(process_id); append(fallback,static_cast<std::int32_t>(f.reason));
    FC(diagnostic_status); FC(diagnostic_reserved); FC(diagnostic_value0);
    FC(diagnostic_value1); FC(diagnostic_value2); FC(component_hash); FC(medium_hash);
    FC(interaction_hash); FC(energy_fraction); FC(selection_uniform); FC(loss_quantile);
    FC(final_state_uniform); FC(outer_acceptance_uniform); FC(random_process_id);
    FC(outer_acceptance_random_process_id); FC(random_draw_id); FC(outer_acceptance_draw_id);
    FC(final_state_draw_id);
#undef FC
    auto key=std::to_string(static_cast<std::int32_t>(f.reason));
    auto n=fallback_reasons.value(key,std::uint64_t{}); fallback_reasons[key]=n+1;
  }
  if (radio) {
    corsika::accelerator::radio::detail::validateFixedRadio(w.fixed_radio);
    vectorBytes(wave,w.fixed_radio.coreas); vectorBytes(wave,w.fixed_radio.zhs);
    auto const& r=w.fixed_radio.counters;
    append(wave,r.coreas_contributions); append(wave,r.zhs_contributions);
    append(wave,r.zhs_subtracks); append(wave,r.valid_tracks); append(wave,r.fixed_point_overflows);
    require(r.coreas_contributions && r.zhs_contributions,"missing real CoREAS/ZHS work");
    for (auto const* set:{&w.radio.coreas,&w.radio.zhs}) for (auto const& trace:*set) {
      vectorBytes(wave,trace.x); vectorBytes(wave,trace.y); vectorBytes(wave,trace.z);
    }
  }
  auto source=w.initial_energy+p.weighted_medium_rest_mass_input_GeV+
              p.weighted_mass_convention_correction_GeV;
  auto sink=w.returned_energy+p.weighted_deposited_energy_GeV+p.weighted_cut_rest_mass_energy_GeV+
    p.weighted_observed_total_energy_GeV+p.weighted_escaped_total_energy_GeV+
    p.weighted_unwritten_photoelectric_binding_energy_GeV-p.weighted_observation_cut_overlap_energy_GeV;
  auto residual=static_cast<double>(std::abs(source-sink)/source);
  require(residual<=1.e-4,"accelerator-boundary energy flux not closed");
  auto transport=seconds(w.start,w.transport_end);
  auto calls=w.photon_seconds+w.lepton_seconds;
  auto const& s=w.statistics;
  Json counts=Json::object();
#define COUNT(field) counts[#field]=s.field
  COUNT(wavefronts); COUNT(particles_enqueued); COUNT(particles_advanced);
  COUNT(particles_produced); COUNT(proposal_fallbacks); COUNT(interactions_selected);
  COUNT(gpu_final_states); COUNT(physical_secondaries_generated);
  COUNT(photon_pair_final_states); COUNT(brems_final_states); COUNT(compton_final_states);
  COUNT(photoelectric_final_states); COUNT(annihilation_final_states);
  COUNT(ionization_final_states); COUNT(electron_pair_final_states);
  COUNT(photon_pair_lpm_trials); COUNT(photon_pair_lpm_suppressions);
  COUNT(brems_lpm_trials); COUNT(brems_lpm_suppressions);
  COUNT(electron_pair_lpm_trials); COUNT(electron_pair_lpm_suppressions);
  COUNT(electron_pair_rejection_trials); COUNT(electron_pair_rejection_fallbacks);
  COUNT(electron_pair_envelope_violations);
  COUNT(thinning_hillas_vertices); COUNT(thinning_statistical_vertices);
  COUNT(thinning_particles_discarded);
  COUNT(photon_transport_interactions); COUNT(photon_transport_boundaries);
  COUNT(photon_transport_observations); COUNT(photon_transport_escapes); COUNT(photon_transport_cuts);
  COUNT(lepton_transport_interaction_candidates); COUNT(lepton_transport_continuous_steps);
  COUNT(lepton_transport_cuts); COUNT(lepton_transport_boundaries);
  COUNT(lepton_transport_observations); COUNT(lepton_transport_escapes);
  COUNT(lepton_transport_magnetic_steps); COUNT(lepton_transport_decay_candidates);
  COUNT(lepton_vertex_interactions_selected); COUNT(lepton_vertex_no_interaction_continuations);
  COUNT(moliere_trials); COUNT(moliere_deflections); COUNT(moliere_zero_deflections);
  COUNT(moliere_newton_iterations); COUNT(moliere_max_newton_iterations);
  COUNT(native_newton_iterations); COUNT(native_bisection_iterations); COUNT(native_inverse_failures);
  COUNT(physical_photon_wavefronts); COUNT(physical_lepton_wavefronts); COUNT(queue_overflows);
#undef COUNT
  counts["photon_transport_records"]=w.photon_steps;
  counts["lepton_transport_records"]=w.lepton_steps;
  auto count_text=counts.dump();
  Json hashes={{"input",w.input_sha256},{"profile_integer_and_float",digest(profile)},
    {"radio_integer_and_float",digest(wave)},{"terminal_particles",digest(terminal)},
    {"fallback_with_random_provenance",digest(fallback)},{"call_sequence",digest(w.call_bytes)},
    {"physical_counts",digest(std::vector<std::uint8_t>(count_text.begin(),count_text.end()))},
    {"table",em::tables::toHex(s.native_table_hash)},
    {"auxiliary",em::tables::toHex(s.auxiliary_cache_hash)}};
  return {{"hashes",hashes},{"physical_counts",counts},
    {"call_shape",{{"photon_wave_limit",w.photon_wave_limit},
                   {"lepton_wave_limit",w.lepton_wave_limit},
                   {"minimum_resident_batch",w.minimum_resident_batch},
                   {"one_call","one safe complete backend endpoint call, not a scheduler packet"}}},
    {"transport_seconds",transport},
    {"output_download_seconds",seconds(w.output_start,w.output_end)},
    {"output_driver_thread_cpu_seconds",w.output_thread_cpu_seconds},
    {"driver_seconds",seconds(w.start,w.end)},{"driver_thread_cpu_seconds",w.thread_cpu_seconds},
    {"transport_driver_thread_cpu_seconds",w.transport_thread_cpu_seconds},
    {"photon_backend_call_seconds",w.photon_seconds},{"lepton_backend_call_seconds",w.lepton_seconds},
    {"photon_calls",w.photon_calls},{"lepton_calls",w.lepton_calls},
    {"photon_wavefronts",s.physical_photon_wavefronts},{"lepton_wavefronts",s.physical_lepton_wavefronts},
    {"profile_steps",p.steps},{"photon_transport_records",w.photon_steps},
    {"lepton_transport_records",w.lepton_steps},
    {"transport_records_per_second",(w.photon_steps+w.lepton_steps)/transport},
    {"transport_records_per_backend_call_second",(w.photon_steps+w.lepton_steps)/calls},
    {"terminal_particles",w.terminal.size()},
    {"fallback_count",w.fallback.size()},{"fallback_reasons",fallback_reasons},
    {"energy_source_GeV",double(source)},{"energy_sink_GeV",double(sink)},
    {"fallback_boundary_flux_GeV",double(w.returned_energy)},
    {"accelerator_boundary_energy_relative_residual",residual},
    {"full_shower_energy_coverage",false},{"profile_overflows",p.fixed_point_overflows},
    {"profile_invalid_records",p.invalid_records},{"coreas_contributions",s.radio.coreas_contributions},
    {"zhs_contributions",s.radio.zhs_contributions},{"radio_overflows",s.radio.fixed_point_overflows},
    {"workspace_bytes",s.physical_workspace_bytes},{"blocking_wait_enabled",w.waits.blocking_enabled},
    {"blocking_wait_calls",w.waits.blocking_calls},{"blocking_wait_host_seconds",w.waits.blocking_host_seconds}};
}
double median(std::vector<double> values) {
  require(values.size()==5,"expected exactly five measured repeats");
  std::sort(values.begin(),values.end()); return values[2];
}
} // namespace

int main(int argc, char** argv) {
  try {
    require(argc==7,"CACHE THREADS CPU_ROOTS GPU_ROOTS ENERGY_MEV RADIO_0_OR_1");
    auto threads=std::stoi(argv[2]); auto cpu_count=std::stoull(argv[3]);
    auto gpu_count=std::stoull(argv[4]); auto energy=std::stod(argv[5]); auto radio=std::stoi(argv[6]);
    require(threads>=1 && threads<=256 && cpu_count>=3 && cpu_count<=65536 &&
            gpu_count>=3 && gpu_count<=65536 && std::isfinite(energy) &&
            energy>=1.5 && energy<=100. && (radio==0 || radio==1),"unbounded fixture parameters");
    require(std::max(cpu_count,gpu_count)*(energy+.9)*.001<=512.,
            "fixed fixture initial energy exceeds bounded 512 GeV per endpoint");
    memoryGate();
    // Match the production driver's pre-runtime capture and full permitted
    // OpenMP partition restoration; inheriting the bound master's sole CPU
    // would manufacture a stronger contention problem than production has.
    auto restore_driver_affinity=acc::detail::captureCooperativeDriverAffinity();
    auto main_allowed_before=allowedCpus();
    // Reuses the existing warmed real-PROPOSAL test fixture, not a new sampler.
    // The caller must provide its existing PROPOSAL/auxiliary cache environment.
    auto fixture=loadCooperativeLeptonFixture(argv[1]);
    acc::KokkosRuntimeConfig owner_config;
    owner_config.execution_backend="cuda"; owner_config.threads=threads;
    owner_config.cooperative_owner=true;
    acc::KokkosRuntime owner(owner_config);
    auto runtime=owner_config; runtime.cooperative_owner=false;
    runtime.runtime_lease=owner.shareCooperativeLifetime(); runtime.execution_backend="openmp";
    auto cpu=acc::detail::makeOpenMPBackendInstance(runtime);
    runtime.execution_backend="cuda";
    auto gpu=acc::detail::makeCudaBackendInstance(runtime);
    // Destruct before either backend/runtime, including on an exception.
    acc::detail::IndependentEndpointDriver gpu_driver;
    auto affinity_future=gpu_driver.submit([restore_driver_affinity] {
      restore_driver_affinity(); return allowedCpus();
    });
    auto driver_allowed=affinity_future.get(); gpu_driver.waitIdle();
    struct Box {
      std::vector<corsika::geometry_detail::ConvexPlane> planes;
      auto const& exportPlanes() const {return planes;}
    } box{{{1,0,0,1000},{-1,0,0,1000},{0,1,0,1000},{0,-1,0,1000},{0,0,1,1000},{0,0,-1,1000}}};
    auto environment=em::makeHomogeneousConvexSnapshot(.001,box);
    em::GpuEmConfig config;
    config.random_seed=26091022; config.shower_id=0;
    config.em_transport_cut_MeV=.4; config.muon_transport_cut_MeV=300.;
    config.min_batch_size=4096; config.resident_batch_limit=8192;
    config.memory_fraction=.10; config.resident_cross_species=true;
    auto& p=config.profile_projection;
    p.enabled=true; p.accumulate_on_device=true; p.axis_direction[0]=1.;
    p.axis_step_length_m=1.; p.output_bin_count=1000; p.output_bin_width_g_per_cm2=.1;
    p.fixed_point_weight_limit=1.e8; p.fixed_point_energy_limit_GeV=1.e8;
    for (int i=0;i<=1000;++i) p.axis_grammage_g_per_cm2.push_back(i*.1);
    auto& r=config.radio;
    r.enabled=radio; r.deterministic=true; r.track_diagnostics=true;
    r.propagation.homogeneous_refractive_index=1.0003;
    if(r.enabled) for(int i=0;i<4;++i) {
      corsika::gpu::radio::RadioObserverSnapshot o;
      o.position_m[0]=20.+i*5.; o.position_m[1]=10.; o.start_time_s=0.;
      o.sample_rate_Hz=1.e9; o.number_of_bins=4096; o.duration_s=4095.e-9;
      r.coreas_observers.push_back(o); r.zhs_observers.push_back(o);
    }
    auto init=Clock::now();
    cpu->initialize(environment,fixture.tables,fixture.auxiliary,config);
    cpu->prepareCooperativeWorkspace();
    gpu->initialize(environment,fixture.tables,fixture.auxiliary,config);
    gpu->prepareCooperativeWorkspace();
    require(cpu->runtimeInfo().host_threads==threads && cpu->runtimeInfo().openmp &&
            !cpu->runtimeInfo().gpu && gpu->runtimeInfo().gpu,"wrong endpoint configuration");
    require(!gpu->cooperativeWaitStatistics().blocking_enabled &&
            !gpu->cooperativeWaitStatistics().blocking_calls,"blocking wait unexpectedly default");
    rejects([&]{cpu->setCooperativeBlockingWait(true);},"CPU accepted CUDA blocking mode");
    gpu->setCooperativeProgress([]{return false;});
    rejects([&]{gpu->setCooperativeBlockingWait(true);},"progress/blocking conflict accepted");
    gpu->setCooperativeProgress({}); gpu->setCooperativeBlockingWait(true);
    rejects([&]{gpu->setCooperativeProgress([]{return false;});},"blocking/progress conflict accepted");
    gpu->setCooperativeBlockingWait(false);
    auto cpu_input=makeInput(cpu_count,energy,0);
    auto gpu_input=makeInput(gpu_count,energy,1000000000000ULL);
    Json report={{"schema",3},{"scope","fixed real-EM workload contention, not a hadronic shower speedup"},
      {"threads",threads},{"cpu_roots",cpu_count},{"gpu_roots",gpu_count},{"energy_MeV",energy},
      {"radio",bool(radio)},{"warmups_per_mode",1},{"measured_repeats_per_mode",5},
      {"resident_capacity_limit",8192},{"gpu_memory_fraction",.10},
      {"call_shapes",{{"cpu",{{"photon_wave_limit",16},{"lepton_wave_limit",1024},
                                {"minimum_resident_batch",4096}}},
                       {"gpu",{{"photon_wave_limit",16},{"lepton_wave_limit",8},
                                {"minimum_resident_batch",1}}}}},
      {"main_allowed_cpus_before_runtime",main_allowed_before},
      {"main_allowed_cpus_after_runtime",allowedCpus()},
      {"gpu_driver_allowed_cpus",driver_allowed},
      {"initialization_seconds",seconds(init,Clock::now())},
      {"limitations",Json::array({"Endpoint wall-window overlap is not a kernel activity timeline.",
        "Decoded arrays derive from the same single integer snapshot, not an independent ordinary-download path test.",
        "CPU main-thread CPU time excludes OpenMP worker CPU time; joint process CPU time includes both ends.",
        "Fallback is audited as outgoing accelerator flux, not transported by scalar PROPOSAL here.",
        "No CPU/GPU cross-hardware bitwise shower or 1% statistical equivalence claim.",
        "Floating track-diagnostic reduction sums are not included in deterministic output hashes."})},
      {"runs",Json::array()}};
    std::array<std::string,3> modes{{"helper-off","default-fence","blocking-event"}};
    Json expected_cpu,expected_gpu;
    std::map<std::string,std::vector<double>> transport_times,helper_cpu_times;
    bool first=true;
    for (unsigned repeat=0;repeat<=5;++repeat) for (unsigned order=0;order<3;++order) {
      auto mode=modes[(order+repeat)%3]; bool helper=mode!="helper-off";
      bool blocking=mode=="blocking-event";
      memoryGate();
      if(!first) {
        bool previous=gpu->cooperativeWaitStatistics().blocking_enabled;
        cpu->beginShower(em::makeGpuEmShowerConfig(config));
        gpu->beginShower(em::makeGpuEmShowerConfig(config));
        auto reset=gpu->cooperativeWaitStatistics();
        require(reset.blocking_enabled==previous && !reset.blocking_calls &&
                reset.blocking_host_seconds==0.,"wait mode/reset lifecycle changed");
      }
      first=false;
      gpu->setCooperativeBlockingWait(blocking);
      // Copies are bounded and occur before the measurement/start barrier.
      auto host_roots=cpu_input,device_roots=gpu_input;
      std::future<Work> device_future;
      std::promise<void> start,ready;
      auto start_future=start.get_future().share(); auto ready_future=ready.get_future();
      if(helper) device_future=gpu_driver.submit([&, roots=std::move(device_roots)]() mutable {
        ready.set_value(); start_future.get();
        return drive(*gpu,std::move(roots),true);
      });
      if(helper && ready_future.wait_for(std::chrono::seconds(10))!=std::future_status::ready) {
        // Release before unwinding: driver destruction must never wait forever.
        start.set_value(); throw std::runtime_error("CUDA driver failed to reach start barrier");
      }
      auto process_start=processSeconds(); auto origin=Clock::now();
      start.set_value();
      Work host;
      try {host=drive(*cpu,std::move(host_roots),false);}
      catch(...) {if(helper){device_future.wait();gpu_driver.waitIdle();}throw;}
      Work device;
      if(helper) {device=device_future.get();gpu_driver.waitIdle();}
      auto joint_end=Clock::now(); auto process_cpu=processSeconds()-process_start;
      collect(*cpu,host,bool(radio),false);
      if(helper) {
        auto output_future=gpu_driver.submit([&] {collect(*gpu,device,bool(radio),true);});
        output_future.get(); gpu_driver.waitIdle();
      }
      auto host_row=audit(host,bool(radio));
      if(expected_cpu.is_null()) expected_cpu=host_row["hashes"];
      require(host_row["hashes"]==expected_cpu,"same OpenMP input changed outputs/call sequence across modes");
      Json row={{"repeat",repeat},{"warmup",repeat==0},{"mode",mode},{"order",order},
        {"cpu",host_row},{"joint_wall_seconds",seconds(origin,joint_end)},
        {"joint_process_cpu_seconds",process_cpu},{"rss_bytes",procBytes("/proc/self/status","VmRSS:")},
        {"available_memory_bytes",procBytes("/proc/meminfo","MemAvailable:")}};
      if(helper) {
        auto device_row=audit(device,bool(radio));
        if(expected_gpu.is_null()) expected_gpu=device_row["hashes"];
        require(device_row["hashes"]==expected_gpu,"CUDA wait mode changed fixed-input outputs/call sequence");
        require(device.waits.blocking_enabled==blocking &&
                (blocking ? device.waits.blocking_calls>0 : device.waits.blocking_calls==0),
                "actual resident waits did not use requested mode");
        rejects([&]{gpu->setCooperativeBlockingWait(!blocking);},"wait mode changed after transport");
        row["gpu"]=device_row;
        auto overlap=std::max(0.,seconds(std::max(host.start,device.start),
                                        std::min(host.transport_end,device.transport_end)));
        row["transport_host_call_window_overlap_seconds"]=overlap;
        row["cpu_transport_fraction_overlapping_gpu_host_window"]=
          overlap/seconds(host.start,host.transport_end);
        row["gpu_start_offset_seconds"]=seconds(origin,device.start);
        row["gpu_transport_end_offset_seconds"]=seconds(origin,device.transport_end);
        if(repeat) helper_cpu_times[mode].push_back(device.thread_cpu_seconds);
      } else {
        require(!gpu->statistics().physical_photon_wavefronts &&
                !gpu->statistics().physical_lepton_wavefronts &&
                !gpu->cooperativeWaitStatistics().blocking_calls,"disabled helper performed work");
        row["gpu"]=nullptr;
      }
      row["cpu_start_offset_seconds"]=seconds(origin,host.start);
      row["cpu_transport_end_offset_seconds"]=seconds(origin,host.transport_end);
      if(repeat) transport_times[mode].push_back(seconds(host.start,host.transport_end));
      report["runs"].push_back(row);
      std::cout<<"CONTENTION_REPEAT "<<row.dump()<<std::endl;
    }
    for(auto const& mode:modes) {
      auto value=median(transport_times.at(mode));
      report["medians"][mode]={{"cpu_transport_seconds",value},
        {"cpu_time_ratio_to_helper_off",value/median(transport_times.at("helper-off"))}};
      if(mode!="helper-off")report["medians"][mode]["gpu_driver_thread_cpu_seconds"]=
        median(helper_cpu_times.at(mode));
    }
    report["same_cpu_workload_and_outputs_exact"]=true;
    report["same_gpu_workload_and_outputs_exact"]=true;
    report["wait_mode_and_lifecycle_tests_passed"]=true;
    report["complete"]=true;
    std::cout<<"CONTENTION_RESULT "<<report.dump()<<std::endl;
    return 0;
  }catch(std::exception const& e){std::cerr<<e.what()<<std::endl;return 1;}
}

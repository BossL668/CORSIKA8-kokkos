// Diagnostic executable only: links FROZEN backend objects, never installed.
// Fixed input and identical host call sequence isolate runtime/workspace costs.
// This is NOT a hadronic shower or an auxiliary-GPU speed-up benchmark.
#include <corsika/accelerator/em/detail/KokkosBackendInstance.hpp>
#include <corsika/accelerator/em/common/ConvexEnvironmentSnapshot.hpp>
#include <corsika/accelerator/em/common/tables/Sha256.hpp>
#include "CooperativeLeptonFixture.hpp"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <chrono>
#include <cstring>
#include <iostream>
#include <set>

namespace em = corsika::gpu::em;
namespace acc = corsika::accelerator::em;
using Clock = std::chrono::steady_clock;
using Json = nlohmann::ordered_json;
void require(bool ok, char const* why) { if (!ok) throw std::runtime_error(why); }
double seconds(Clock::time_point a, Clock::time_point b) {
  return std::chrono::duration<double>(b-a).count();
}
template<class T> void append(std::vector<std::uint8_t>& bytes, T const& x) {
  static_assert(std::is_arithmetic_v<T>);
  if constexpr(std::is_floating_point_v<T>) require(std::isfinite(x), "nonfinite output");
  auto p = reinterpret_cast<std::uint8_t const*>(&x);
  bytes.insert(bytes.end(), p, p+sizeof(T));
}
void particleBytes(std::vector<std::uint8_t>& b, em::EmParticleState const& p) {
  append(b,p.pid); append(b,p.medium_id); append(b,p.generation); append(b,p.reserved);
  append(b,p.energy_GeV); for(auto x:p.position_m) append(b,x);
  for(auto x:p.direction) append(b,x);
  append(b,p.time_s); append(b,p.weight); append(b,p.history_id);
  append(b,p.parent_history_id); append(b,p.step_id);
}
template<class T> void vectorBytes(std::vector<std::uint8_t>& b, std::vector<T> const& v) {
  append(b,static_cast<std::uint64_t>(v.size())); for(auto x:v) append(b,x);
}
std::string digest(std::vector<std::uint8_t> const& b) {
  return em::tables::toHex(em::tables::sha256(b));
}
int main(int argc, char** argv) {
 try {
  require(argc==8, "CACHE MODE THREADS INPUT_COUNT ENERGY_MEV RADIO_0_OR_1 REPEATS");
  std::string mode=argv[2];
  require(mode=="direct" || mode=="prepared" || mode=="borrowed" || mode=="idle-cuda" || mode=="host-pump",
          "unsupported isolation mode");
  int threads=std::stoi(argv[3]), count=std::stoi(argv[4]), repeats=std::stoi(argv[7]);
  double energy=std::stod(argv[5]); int radio=std::stoi(argv[6]);
  require(threads>0 && count>0 && count<=65536 && repeats>=2 && repeats<=16 &&
          std::isfinite(energy) && energy>=1.5 && energy<=1000. && (radio==0 || radio==1),
          "unbounded or invalid fixture parameters");
  // Below the existing primary reserve, not a new scheduler switch. Explicitly
  // reject any accidental auxiliary transport after the real pump drains.
  bool pump=mode=="host-pump";
  if(pump)require(count<=1024 && energy==1.5,"host-pump fixture must stay below primary reserve");
  auto fixture=loadCooperativeLeptonFixture(argv[1]);
  acc::KokkosRuntimeConfig runtime; runtime.execution_backend="openmp"; runtime.threads=threads;
  // Lifetime order matters: borrowed backends die before their owner.
  std::unique_ptr<acc::KokkosRuntime> owner;
  if(mode=="borrowed" || mode=="idle-cuda") {
    auto own=runtime; own.execution_backend="cuda"; own.cooperative_owner=true;
    owner=std::make_unique<acc::KokkosRuntime>(own);
    runtime.runtime_lease=owner->shareCooperativeLifetime();
  }
  if(pump)runtime.execution_backend="openmp-cuda";
  auto cpu=pump?acc::detail::makeCooperativeBackendInstance(runtime):
                acc::detail::makeOpenMPBackendInstance(runtime);
  std::unique_ptr<acc::detail::KokkosBackendInstance> gpu;
  struct Box {
    std::vector<corsika::geometry_detail::ConvexPlane> planes;
    auto const& exportPlanes() const { return planes; }
  } box{{{1,0,0,1000},{-1,0,0,1000},{0,1,0,1000},{0,-1,0,1000},
         {0,0,1,1000},{0,0,-1,1000}}};
  auto environment=em::makeHomogeneousConvexSnapshot(.001,box);
  em::GpuEmConfig config;
  config.random_seed=26091022; config.shower_id=0;
  config.em_transport_cut_MeV=.4; config.muon_transport_cut_MeV=300.;
  config.min_batch_size=4096; config.resident_batch_limit=65536;
  config.memory_fraction=.70; config.resident_cross_species=true;
  auto& p=config.profile_projection;
  p.enabled=true; p.accumulate_on_device=true; p.axis_direction[0]=1.;
  p.axis_step_length_m=1.; p.output_bin_count=1000; p.output_bin_width_g_per_cm2=.1;
  p.fixed_point_weight_limit=1.e8; p.fixed_point_energy_limit_GeV=1.e8;
  for(int i=0;i<=1000;++i)p.axis_grammage_g_per_cm2.push_back(i*.1);
  auto& r=config.radio; r.enabled=radio; r.deterministic=true; r.track_diagnostics=true;
  r.propagation.homogeneous_refractive_index=1.0003;
  if(r.enabled)for(int i=0;i<4;++i) {
    corsika::gpu::radio::RadioObserverSnapshot o;
    o.position_m[0]=20.+i*5.; o.position_m[1]=10.;
    o.start_time_s=0.; o.sample_rate_Hz=1.e9; o.number_of_bins=4096; o.duration_s=4095.e-9;
    r.coreas_observers.push_back(o); r.zhs_observers.push_back(o);
  }
  auto init=Clock::now();
  cpu->initialize(environment,fixture.tables,fixture.auxiliary,config);
  if(mode!="direct" && !pump)cpu->prepareCooperativeWorkspace();
  if(mode=="idle-cuda") {
    runtime.execution_backend="cuda";
    gpu=acc::detail::makeCudaBackendInstance(runtime);
    gpu->initialize(environment,fixture.tables,fixture.auxiliary,config);
    gpu->prepareCooperativeWorkspace();
  }
  require(cpu->runtimeInfo().host_threads==threads && cpu->runtimeInfo().openmp &&
          (pump || !cpu->runtimeInfo().gpu), "not the requested OpenMP endpoint");
  Json report={{"schema",1},{"mode",mode},{"threads",threads},{"input_count",count},
    {"energy_MeV",energy},{"radio",bool(radio)},{"init_seconds",seconds(init,Clock::now())},
    {"scope",pump?"real CPU-primary pump, below reserve, zero GPU transport required":
                  "fixed EM inputs, same synchronous driver, not the cooperative pump"},
    {"digest_schema",2},
    {"gpu_transport_steps",0},{"runs",Json::array()}};
  std::string expected;
  for(int event=0;event<repeats;++event) {
    if(event)cpu->beginShower(em::makeGpuEmShowerConfig(config));
    std::vector<em::EmParticleState> photons,leptons(count),terminal;
    std::vector<std::uint8_t> input_bytes;
    long double initial=0,returned=0;
    for(int i=0;i<count;++i) {
      auto& q=leptons[i]; q.pid=i%2?11:-11; q.energy_GeV=(energy+.1*(i%10))*.001;
      q.history_id=i+1; q.direction[0]=1.; q.weight=1.; q.generation=1;
      initial+=q.energy_GeV; particleBytes(input_bytes,q);
    }
    auto input_hash=digest(input_bytes);
    std::uint64_t next=1000000, rounds=0, photon_calls=0, lepton_calls=0;
    double photon_s=0,lepton_s=0;
    std::vector<std::uint8_t> call_bytes;
    auto receive=[&](auto const& result) {
      require(result.cpu_spill_particles.empty(),"fixture spill requires separate consumer");
      for(auto const& q:result.observations)terminal.push_back(q.particle);
      for(auto const& q:result.fallback_events) {
        terminal.push_back(q.particle);
        returned+=static_cast<long double>(q.particle.weight)*q.particle.energy_GeV;
      }
    };
    auto begin=Clock::now();
    if(pump) {
      acc::detail::SubshowerCallbacks cb;
      std::uint64_t last_first=0;
      cb.reserve_histories=[&](std::uint64_t lease) {
        require(lease<=UINT64_MAX-next,"history overflow");
        last_first=next; next+=lease; return last_first;
      };
      auto record=[&](bool photon,auto const& result) {
        append(call_bytes,photon); append(call_bytes,static_cast<std::uint64_t>(result.input_particles));
        append(call_bytes,last_first); receive(result);
      };
      cb.photons=[&](em::ResidentPhotonCascadeResult&& result,double ms) {
        record(true,result); ++photon_calls; photon_s+=ms*.001;
        require(result.remaining_photons.empty() && result.electromagnetic_secondaries.empty(),"pump lost descendant ownership");
      };
      cb.leptons=[&](em::ResidentLeptonCascadeResult&& result,double ms) {
        record(false,result); ++lepton_calls; lepton_s+=ms*.001;
        require(result.decay_candidates.empty() && result.remaining_leptons.empty() &&
                result.generated_photons.empty(),"pump lost descendant ownership");
      };
      do {
        require(++rounds<100000,"pump did not drain");
        auto accepted=cpu->advanceIndependentSubshowers(leptons,cb);
        require(accepted<=leptons.size(),"invalid acceptance count");
        leptons.erase(leptons.begin(),leptons.begin()+accepted);
      } while(!leptons.empty() || cpu->pendingPhotonCount() || cpu->pendingLeptonCount());
      auto const& c=cpu->statistics().cooperative;
      require(c.cuda_input_particles==0 && c.subshower_cuda_submissions==0 &&
              c.subshower_cuda_commits==0,"host-only fixture unexpectedly assigned GPU work");
    } else while(!leptons.empty() || !photons.empty() || cpu->pendingPhotonCount() || cpu->pendingLeptonCount()) {
      require(++rounds<100000,"fixture did not drain");
      auto drive=[&](bool photon) {
        auto& queue=photon?photons:leptons;
        auto capacity=photon?cpu->maximumResidentPhotonBatchSize():cpu->maximumResidentLeptonBatchSize();
        auto pending=std::min(capacity,photon?cpu->pendingPhotonCount():cpu->pendingLeptonCount());
        auto take=std::min(capacity-pending,queue.size()), n=take+pending;
        if(!n)return;
        std::vector<em::EmParticleState> in(queue.begin(),queue.begin()+take);
        queue.erase(queue.begin(),queue.begin()+take);
        std::uint64_t lease=photon?32*n:std::max<std::uint64_t>(3*n,std::min<std::uint64_t>(1U<<20,192*n));
        auto first=next; require(lease<=UINT64_MAX-next,"history overflow"); next+=lease;
        append(call_bytes,photon); append(call_bytes,static_cast<std::uint64_t>(n));
        append(call_bytes,first);
        auto t=Clock::now();
        if(photon) {
          auto result=cpu->runPhotonWavefront(in,first,16,4096);
          photon_s+=seconds(t,Clock::now()); ++photon_calls; receive(result);
          queue.insert(queue.end(),result.remaining_photons.begin(),result.remaining_photons.end());
          leptons.insert(leptons.end(),result.electromagnetic_secondaries.begin(),result.electromagnetic_secondaries.end());
        } else {
          auto result=cpu->runLeptonWavefront(in,first,1024,next,4096);
          lepton_s+=seconds(t,Clock::now()); ++lepton_calls; receive(result);
          require(result.decay_candidates.empty(),"electron fixture has muon decay");
          queue.insert(queue.end(),result.remaining_leptons.begin(),result.remaining_leptons.end());
          photons.insert(photons.end(),result.generated_photons.begin(),result.generated_photons.end());
        }
      };
      drive(true); drive(false);
    }
    auto elapsed=seconds(begin,Clock::now());
    std::sort(terminal.begin(),terminal.end(),[](auto const& a,auto const& b){return a.history_id<b.history_id;});
    for(std::size_t i=1;i<terminal.size();++i)require(terminal[i-1].history_id!=terminal[i].history_id,"duplicate terminal history");
    auto profile=cpu->downloadProfile();
    require(!profile.fixed_point_overflows && !profile.invalid_records,"invalid profile");
    std::vector<std::uint8_t> physics;
    vectorBytes(physics,profile.photons); vectorBytes(physics,profile.electrons);
    vectorBytes(physics,profile.positrons); vectorBytes(physics,profile.muons_minus);
    vectorBytes(physics,profile.muons_plus); vectorBytes(physics,profile.muon_parent_productions);
    vectorBytes(physics,profile.energy_loss_GeV); vectorBytes(physics,profile.muon_energy_loss_GeV);
    for(auto const& q:terminal)particleBytes(physics,q);
    append(physics,profile.steps); append(physics,profile.deposited_steps);
    append(physics,profile.particle_cuts);
    if(r.enabled) {
      auto wave=cpu->downloadRadioWaveforms();
      for(auto const* set:{&wave.coreas,&wave.zhs})for(auto const& w:*set) {
        vectorBytes(physics,w.x); vectorBytes(physics,w.y); vectorBytes(physics,w.z);
      }
      auto const& s=cpu->statistics().radio;
      require(!s.fixed_point_overflows && s.coreas_contributions && s.zhs_contributions,"missing/invalid radio");
    }
    auto source=initial+profile.weighted_medium_rest_mass_input_GeV+profile.weighted_mass_convention_correction_GeV;
    auto sink=returned+profile.weighted_deposited_energy_GeV+profile.weighted_cut_rest_mass_energy_GeV+
      profile.weighted_observed_total_energy_GeV+profile.weighted_escaped_total_energy_GeV+
      profile.weighted_unwritten_photoelectric_binding_energy_GeV-profile.weighted_observation_cut_overlap_energy_GeV;
    auto residual=double(std::abs(source-sink)/source);
    require(residual<=1.e-4,"accelerator boundary energy flux not closed");
    auto hash=digest(physics),call_hash=digest(call_bytes);
    if(event)require(hash+call_hash==expected,"same fixed-input repeat changed physics/call sequence");
    expected=hash+call_hash;
    if(gpu)require(!gpu->statistics().physical_photon_wavefronts &&
                   !gpu->statistics().physical_lepton_wavefronts,"idle auxiliary performed transport");
    auto const& stats=cpu->statistics();
    Json row={{"repeat",event},{"warm",event>0},{"driver_seconds",elapsed},
      {"photon_seconds",photon_s},{"lepton_seconds",lepton_s},{"photon_calls",photon_calls},
      {"lepton_calls",lepton_calls},{"steps",profile.steps},{"photon_waves",stats.physical_photon_wavefronts},
      {"lepton_waves",stats.physical_lepton_wavefronts},{"input_sha256",input_hash},
      {"physics_arrays_terminal_sha256",hash},{"call_sequence_sha256",call_hash},
      {"energy_relative_residual",residual},{"terminal_count",terminal.size()}};
    report["runs"].push_back(row); std::cout<<"ISOLATION_REPEAT "<<row.dump()<<std::endl;
  }
  report["complete"]=true; std::cout<<"ISOLATION_RESULT "<<report.dump()<<std::endl;
  return 0;
 } catch(std::exception const& e) {std::cerr<<e.what()<<std::endl; return 1;}
}

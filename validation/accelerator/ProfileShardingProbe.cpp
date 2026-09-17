/* (c) Copyright 2026 CORSIKA Project; BSD-3-Clause license. */
// Diagnostic, not a shower backend. Reuses the exact profile-step equations.
// Compare bounded host accumulator sharding on an identical synthetic ledger.
#include <corsika/accelerator/em/common/PhotonPairKinematics.hpp>
#include <corsika/accelerator/em/detail/ProfileAccumulationStep.hpp>
#include <corsika/accelerator/em/detail/CooperativeProfileMerge.hpp>
#include <omp.h>
#include <algorithm>
#include <chrono>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace em = corsika::gpu::em;
namespace detail = corsika::accelerator::em::detail;

struct HostAtomic {
  template<class T> static T add(T* address, T value) {
    return __atomic_fetch_add(address, value, __ATOMIC_RELAXED);
  }
  static void maximum(unsigned long long* address, unsigned long long value) {
    auto old = __atomic_load_n(address, __ATOMIC_RELAXED);
    while (old < value && !__atomic_compare_exchange_n(
        address, &old, value, true, __ATOMIC_RELAXED, __ATOMIC_RELAXED)) {}
  }
};
constexpr std::size_t bins = 128;
struct alignas(64) Ledger {
  em::detail::DeviceProfileCounters counters{};
  std::vector<long long> histograms = std::vector<long long>(8*bins, 0);
  em::detail::DeviceProfileAccumulator view() {
    em::detail::DeviceProfileAccumulator a{};
    a.photons=histograms.data(); a.electrons=a.photons+bins;
    a.positrons=a.electrons+bins; a.muons_minus=a.positrons+bins;
    a.muons_plus=a.muons_minus+bins; a.muon_parent_productions=a.muons_plus+bins;
    a.energy_loss=a.muon_parent_productions+bins; a.muon_energy_loss=a.energy_loss+bins;
    a.counters=&counters; a.bins=bins; a.bin_width_g_per_cm2=10.;
    a.energy_loss_threshold_g_per_cm2=.01;
    a.weight_scale=0x1p62/1.e12; a.inverse_weight_scale=1./a.weight_scale;
    a.energy_scale=0x1p62/1.e14; a.inverse_energy_scale=1./a.energy_scale;
    return a;
  }
};
void compare(Ledger const& a, Ledger const& b) {
  if (a.histograms!=b.histograms) throw std::runtime_error("Histogram changed");
#define CHECK(f) if(a.counters.f!=b.counters.f) throw std::runtime_error("Counter changed: " #f)
  CHECK(steps); CHECK(deposited_steps); CHECK(photon_cuts);
  for(unsigned i=0;i<8;++i) CHECK(lepton_limits[i]);
  CHECK(moliere_trials); CHECK(moliere_deflections); CHECK(moliere_zero_deflections);
  CHECK(moliere_newton_iterations); CHECK(moliere_max_newton_iterations);
  CHECK(thinning_hillas_vertices); CHECK(thinning_statistical_vertices);
  CHECK(thinning_particles_discarded); CHECK(fixed_point_overflows); CHECK(invalid_records);
  CHECK(weighted_medium_rest_mass_input); CHECK(weighted_cut_rest_mass_energy);
  CHECK(weighted_observed_total_energy); CHECK(weighted_escaped_total_energy);
  CHECK(weighted_unwritten_photoelectric_binding_energy);
  CHECK(weighted_observation_cut_overlap_energy); CHECK(weighted_mass_convention_correction);
#undef CHECK
}
int main(int argc, char** argv) try {
  if(argc!=2) throw std::invalid_argument("usage: profile-sharding-probe THREADS");
  int const threads=std::stoi(argv[1]);
  if(threads<1 || threads>256) throw std::invalid_argument("threads out of range");
  omp_set_dynamic(0); omp_set_num_threads(threads);
  constexpr std::size_t count=16384, repeats=16;
  std::vector<em::LeptonTransportRecord> steps(count);
  for(std::size_t i=0;i<count;++i) {
    auto& s=steps[i]; s.start.pid=(i%4==0?13:(i%4==1?-13:(i%4==2?11:-11)));
    s.start.weight=.5+.25*(i%5); s.start.energy_GeV=1.+.01*(i%20);
    s.start.position_m[2]=400.+.03125*(i%512);
    s.end=s.start; s.end.position_m[2]+=.25+.0625*(i%8);
    s.end.energy_GeV-=.001; s.continuous_deposited_energy_GeV=.001;
    s.limit=static_cast<em::LeptonTransportLimit>(i%8);
    if(s.limit==em::LeptonTransportLimit::ParticleCut) s.cut_deposited_energy_GeV=.01;
    s.observation_surface_reached_before_cut=(i%16==0);
    s.multiple_scattering_applied=i%2;
    s.multiple_scattering_status=static_cast<unsigned>(em::MoliereStatus::NoDeflection);
    s.multiple_scattering_iterations=i%12;
  }
  double grammage[]{0.,1280.};
  em::detail::DeviceProfileProjection projection{};
  projection.axis_direction[2]=1.; projection.axis_step_length_m=1280.;
  projection.axis_grammage_g_per_cm2=grammage; projection.axis_support_count=2;
  Ledger oracle;
  // Serial oracle invokes the very same public templated accumulation function.
  for(std::size_t r=0;r<repeats;++r)
    for(auto const& step:steps)
      detail::accumulateLeptonProfileStep<HostAtomic>(projection,oracle.view(),step);
  if(oracle.counters.fixed_point_overflows || oracle.counters.invalid_records)
    throw std::runtime_error("Invalid synthetic ledger");
  std::cout << "{\"synthetic_profile_only\":true,\"shower_speedup_claim\":false,\"threads\":"
            << threads << ",\"records_per_run\":" << count*repeats << ",\"measurements\":[";
  bool first=true;
  for(int trial=-1;trial<5;++trial) { // warm-up excluded; alternate the order
    std::vector<unsigned> order{1,4,16,32,64,128,256};
    if(trial%2==1) std::reverse(order.begin(),order.end());
    for(auto shards:order) {
      std::vector<Ledger> ledgers(shards);
      int actual_threads=0;
      auto start=std::chrono::steady_clock::now();
#pragma omp parallel
      {
#pragma omp single
        actual_threads=omp_get_num_threads();
        auto a=ledgers[static_cast<unsigned>(omp_get_thread_num())%shards].view();
        for(std::size_t r=0;r<repeats;++r) {
#pragma omp for schedule(static)
          for(std::size_t i=0;i<count;++i)
            detail::accumulateLeptonProfileStep<HostAtomic>(projection,a,steps[i]);
        }
      }
      auto const kernel=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
      if(actual_threads!=threads) throw std::runtime_error("Incorrect thread count");
      Ledger merged;
      for(auto const& ledger:ledgers) {
        merged.counters=detail::mergeProfileCounters(merged.counters,ledger.counters);
        for(std::size_t i=0;i<8*bins;++i) {
          auto value=ledger.histograms[i]; auto& out=merged.histograms[i];
          if((value>0 && out>std::numeric_limits<long long>::max()-value) ||
             (value<0 && out<std::numeric_limits<long long>::min()-value))
            throw std::overflow_error("Shard merge overflow");
          out+=value;
        }
      }
      auto const total=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
      compare(oracle,merged);
      if(trial>=0) {
        if(!first) std::cout << ',';
        first=false;
        std::cout << "{\"trial\":"<<trial<<",\"shards\":"<<shards<<",\"kernel_s\":"<<kernel
                  <<",\"including_merge_s\":"<<total<<",\"exact\":true}";
      }
    }
  }
  std::cout << "]}\n";
  return 0;
} catch(std::exception const& error) {
  std::cerr << error.what() << '\n'; return 1;
}

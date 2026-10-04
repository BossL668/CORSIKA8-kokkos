#pragma once
#include "Egs4C8AirWavefront.hpp"
#include "Egs4C8Session.hpp"

namespace c7_egs4::application {
// Only exceptional/CPU-consumed records cross the device boundary. Scan in
// input order to keep host callbacks, RNG/ID ownership and generator order.
struct HostOutputRecord {
  c8_adapter::AirOutput output;
  c8_adapter::RareAirTransfer transfer;
  c8_adapter::AirAction action{};
  double weight{};
};
struct WaveCounts {std::uint64_t steps{},radio{},observations{},discards{},hillas{},statistical{},removed{},unresolved{},needed{};};
template<class Batch>struct CountWave {
  using value_type=WaveCounts;Batch batch;bool sparse{},profile_device{},radio_device{};
  KOKKOS_INLINE_FUNCTION void init(value_type& c)const{c={};}
  KOKKOS_INLINE_FUNCTION void join(value_type& a,value_type const& b)const {
    a.steps+=b.steps;a.radio+=b.radio;a.observations+=b.observations;a.discards+=b.discards;
    a.hillas+=b.hillas;a.statistical+=b.statistical;a.removed+=b.removed;a.unresolved+=b.unresolved;a.needed+=b.needed;
  }
  KOKKOS_INLINE_FUNCTION void operator()(int i,value_type& c)const {
    auto const& r=batch.outputs(i);auto const& t=batch.thinning(i);
    c.steps+=r.has_step;c.radio+=r.has_radio;c.observations+=r.has_observation;
    c.discards+=r.discard!=c8_adapter::AirDiscard::none;
    c.hillas+=t.status==c8_adapter::ThinningStatus::Hillas;
    c.statistical+=t.status==c8_adapter::ThinningStatus::Statistical;
    if(t.status!=c8_adapter::ThinningStatus::NotApplied)c.removed+=2-((t.keep_mask&1U)!=0)-((t.keep_mask&2U)!=0);
    c.unresolved+=batch.outcomes(i).action==c8_adapter::AirAction::needs_host_channel;
    if(sparse) {
      auto const& h=batch.host(i);
      c.needed+=(!profile_device&&r.has_step)||(!radio_device&&r.has_radio)||r.has_observation||
        r.discard!=c8_adapter::AirDiscard::none||h.count||h.target_rest_energy_MeV!=0.||h.has_rho_decay||h.has_resonance_decay;
    }
  }
};
template<class Batch,class Records>struct SelectHostOutput {
  Batch batch;Records records;bool profile_device,radio_device;
  KOKKOS_INLINE_FUNCTION void operator()(int i,int& count,bool final)const {
    auto const& r=batch.outputs(i);auto const& h=batch.host(i);
    bool needed=(!profile_device&&r.has_step)||(!radio_device&&r.has_radio)||r.has_observation||
      r.discard!=c8_adapter::AirDiscard::none||h.count||h.target_rest_energy_MeV!=0.||h.has_rho_decay||h.has_resonance_decay;
    if(!needed)return;
    if(final)records(count)={r,h,batch.outcomes(i).action,batch.before(i).metadata.weight};
    ++count;
  }
};
template<class Exec>class HostOutput {
  using Records=Kokkos::View<HostOutputRecord*,typename Exec::memory_space>;
  Records records_;bool sparse_{(overhead::options()&overhead::sparse_host)!=0};
public:
  template<class Batch>auto collect(Batch const& batch,bool profile_device,bool radio_device,RunStatistics& stats) {
    int const n=int(batch.outputs.extent(0));WaveCounts counts;
    Kokkos::parallel_reduce("egs4_output_counts",Kokkos::RangePolicy<Exec>(0,n),CountWave<Batch>{batch,sparse_,profile_device,radio_device},counts);
    if(counts.unresolved)throw std::runtime_error("Native EGS4 unresolved rare channel; enable native rare vertices");
    stats.steps+=counts.steps;stats.radio_tracks+=counts.radio;stats.observations+=counts.observations;stats.discards+=counts.discards;
    stats.thinning_hillas+=counts.hillas;stats.thinning_statistical+=counts.statistical;stats.thinning_removed+=counts.removed;
    int count=0;
    if(!sparse_||counts.needed) {
    if(records_.extent(0)<std::size_t(n))records_=Records("egs4_compact_host_output",n);
    Kokkos::parallel_scan("egs4_compact_host_output",Kokkos::RangePolicy<Exec>(0,n),
      SelectHostOutput<Batch,Records>{batch,records_,profile_device,radio_device},count);
    }
    stats.host_output_records+=count;stats.host_output_bytes+=count*sizeof(HostOutputRecord)+sizeof(WaveCounts)+sizeof(int);
    return Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},
      Kokkos::subview(records_,std::make_pair(std::size_t(0),std::size_t(count))));
  }
};
} // namespace c7_egs4::application

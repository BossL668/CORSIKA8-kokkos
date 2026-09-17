/* Independent device CoREAS/ZHS accumulation. Paired mode shares source buffers
 * and immutable optical data while retaining separate kernels and moments.
 * The interval source
 * and cell-moment recurrence follow mountain's TerrainIntervalZHS and
 * FiniteIntervalMoments. Geometry, optical tables, tracks and moments stay on
 * the selected execution space. No air-radio implementation is modified. */
#pragma once
#include <corsika/modules/radio/interface/CoreasEndpoints.hpp>
#include <limits>
#include <type_traits>
#include <utility>

namespace corsika::radio::interface {
inline constexpr std::size_t DeviceTrackBufferSize=8192;
inline std::size_t projectedRadioBytes(Config const& c,std::size_t track_capacity,bool device_buffering=true) {
  if(!c.enabled)return 0;
  validatePropagation(c.propagation);
  if((c.algorithm!=Algorithm::ZHS&&c.algorithm!=Algorithm::CoREAS)||
     (c.algorithm==Algorithm::CoREAS&&c.moment_order==0)||
     !std::isfinite(c.coreas_cherenkov_threshold)||c.coreas_cherenkov_threshold<=0.||c.coreas_cherenkov_threshold>.1)
    throw std::invalid_argument("invalid interface radio algorithm/Cherenkov threshold");
  if(c.observers.empty()||c.observers.size()>256||c.samples<64||c.samples>(1u<<20)||c.samples%2||
     c.moment_order>20||!std::isfinite(c.start_time_s)||!std::isfinite(c.sample_rate_Hz)||
     !(c.sample_rate_Hz>0.&&c.sample_rate_Hz<=32.e9)||
     !(c.subdivision_frequency_Hz>0.&&c.subdivision_frequency_Hz<=128.e9)||
     !(c.fraunhofer_limit>0.&&c.fraunhofer_limit<=1.)||
     !std::isfinite(c.mesh_maximum_segment_m)||c.mesh_maximum_segment_m<=0.||
     c.maximum_subdivision_depth>20||!track_capacity||track_capacity>(1u<<20)||
     c.host_track_buffer_capacity>(1u<<20))
    throw std::invalid_argument("invalid interface radio waveform/subdivision configuration");
  for(auto const& o:c.observers)
    if(!detail::finite(o.position_m)||o.region>1)throw std::invalid_argument("invalid interface radio observer");
  auto values=c.samples*c.observers.size()*6*(c.moment_order+1);
  // Conservative padding for observer alignment, counters and view metadata.
  auto bytes=c.propagation.coverage.bytes()+values*sizeof(double)+track_capacity*sizeof(Track)+c.observers.size()*32+4096;
  if(c.algorithm==Algorithm::CoREAS)bytes+=values*sizeof(double);
  auto buffered=device_buffering?DeviceTrackBufferSize:c.host_track_buffer_capacity;
  bytes+=2*buffered*sizeof(Track);
  for(auto const& m:c.propagation.media)bytes+=m.radial_index.size()*sizeof(IndexSample)+m.radial_integration_breaks_m.size()*sizeof(double);
  if(bytes>c.maximum_device_bytes)throw std::length_error("interface radio device memory budget exceeded");
  return bytes;
}
template<class Space,bool Paired=false> class KokkosAccumulator {
 public:
  using Memory=typename Space::memory_space;
  using Tracks=Kokkos::View<Track*,Memory>;
  using Download=std::conditional_t<Paired,PairedResult,Result>;
  static constexpr bool deviceBuffering=!std::is_same<Memory,Kokkos::HostSpace>::value;
  static std::size_t projectedBytes(Config const& c,std::size_t capacity,std::size_t mesh_nodes=0) {
    auto selected=primaryConfig(c);
    auto bytes=projectedRadioBytes(selected,capacity,deviceBuffering);
    if(c.enabled&&c.propagation.geometry==Geometry::Mesh&&c.propagation.transmission_bvh)
      bytes+=mesh_nodes*sizeof(detail::TransmissionNormalBounds);
    if constexpr(Paired){
      if(c.enabled)bytes+=c.samples*c.observers.size()*6*(c.moment_order+1)*sizeof(double);
    }
    if(bytes>c.maximum_device_bytes)throw std::length_error("combined interface radio device memory budget exceeded");
    return bytes;
  }
  KokkosAccumulator(Config const& c,terrain::flat::View mesh,std::size_t faces,
                    std::size_t capacity,Space const& execution)
      : config_(primaryConfig(c)),bytes_(projectedBytes(c,capacity,mesh.node_count)),propagation_(c.propagation,mesh,faces,execution) {
    if(!c.enabled)throw std::invalid_argument("construct radio accumulator only when enabled");
    observers_=decltype(observers_)("interface_radio_observers",c.observers.size());
    auto host=Kokkos::create_mirror_view(observers_);
    for(std::size_t i=0;i<c.observers.size();++i)host(i)={c.observers[i].position_m,c.observers[i].region};
    Kokkos::deep_copy(execution,observers_,host);
    moments_=decltype(moments_)("interface_radio_moments",c.samples*c.observers.size()*6*(c.moment_order+1));
    if(config_.algorithm==Algorithm::CoREAS)
      regularized_=decltype(regularized_)("interface_radio_coreas_regularized_moments",moments_.extent(0));
    counters_=decltype(counters_)("interface_radio_counters");
    if constexpr(Paired){
      zhs_moments_=decltype(zhs_moments_)("interface_radio_zhs_moments",moments_.extent(0));
      zhs_counters_=decltype(zhs_counters_)("interface_radio_zhs_counters");
    }
    cpu_input_=Tracks("interface_radio_cpu_tracks",capacity);cpu_mirror_=Kokkos::create_mirror_view(cpu_input_);
    if(auto buffered=bufferCapacity()){
      device_pending_=Tracks("interface_radio_device_pending",buffered);
      cpu_pending_=Tracks("interface_radio_cpu_pending",buffered);
    }
    execution.fence("interface radio accumulator initialization");
  }
  void accumulate(Tracks const& tracks,std::size_t count,Space const& execution,bool cpu=false) {
    if(sealed_)throw std::logic_error("interface radio is finalized");
    if(count>tracks.extent(0))throw std::length_error("interface radio input extent exceeded");
    if(!count)return;
    if(!bufferCapacity()){launch(tracks,count,execution,cpu);}
    else {
      // The transport cadence remains independent of radio throughput. Copy
      // completed segments within the execution space, and launch wider batches.
      // Separate ledgers preserve CPU/device source accounting across interleaving.
      auto& pending=cpu?cpu_pending_:device_pending_;
      auto& used=cpu?cpu_buffered_:device_buffered_;
      for(std::size_t first=0;first<count;){
        auto n=std::min(count-first,pending.extent(0)-used);
        Kokkos::deep_copy(execution,
          Kokkos::subview(pending,std::make_pair(used,used+n)),
          Kokkos::subview(tracks,std::make_pair(first,first+n)));
        used+=n;first+=n;
        if(used==pending.extent(0)){launch(pending,used,execution,cpu);used=0;}
      }
    }
  }
  void accumulateCpu(std::vector<Track> const& tracks,Space const& execution) {
    if(sealed_)throw std::logic_error("interface radio is finalized");
    for(std::size_t first=0;first<tracks.size();){
      auto count=std::min(tracks.size()-first,cpu_input_.extent(0));
      for(std::size_t i=0;i<count;++i)cpu_mirror_(i)=tracks[first+i];
      auto range=std::make_pair(std::size_t{0},count);
      Kokkos::deep_copy(execution,Kokkos::subview(cpu_input_,range),Kokkos::subview(cpu_mirror_,range));
      // Reusable pageable staging cannot be overwritten before its upload.
      execution.fence("interface radio CPU source upload");
      accumulate(cpu_input_,count,execution,true);first+=count;
    }
  }
  Download download(Space const& execution) {
    if(sealed_)throw std::logic_error("interface radio downloaded twice");
    if(bufferCapacity()){
      launch(device_pending_,device_buffered_,execution,false);
      launch(cpu_pending_,cpu_buffered_,execution,true);
      device_buffered_=cpu_buffered_=0;
    }
    // Finalization consumes the grids. Release each grid as soon as its host
    // result is complete, instead of keeping three device grids plus all three
    // result vectors alive. This matters on OpenMP, where both use host RAM.
    // Seal before consuming any storage: a failed download cannot be retried.
    sealed_=true;
    auto first=downloadAlgorithm(config_,moments_,regularized_,counters_,execution);
    if constexpr(Paired){
      auto zhsConfig=config_;zhsConfig.algorithm=Algorithm::ZHS;
      Moments noRegularized;
      auto second=downloadAlgorithm(zhsConfig,zhs_moments_,noRegularized,zhs_counters_,execution);
      return {std::move(first),std::move(second),bytes_};
    }else{return first;}
  }
  std::size_t bytes()const{return bytes_;}
 private:
  using Moments=Kokkos::View<double*,Memory>;
  using Counters=Kokkos::View<detail::RadioCounters,Memory>;
  std::size_t bufferCapacity()const {
    return deviceBuffering?DeviceTrackBufferSize:config_.host_track_buffer_capacity;
  }
  static Config primaryConfig(Config c) {
    if constexpr(Paired)c.algorithm=Algorithm::CoREAS;
    return c;
  }
  Result downloadAlgorithm(Config const& config,Moments& moments,
      Moments& regularized,Counters const& deviceCounters,Space const& execution) {
    auto counters=Kokkos::create_mirror_view(deviceCounters);Kokkos::deep_copy(execution,counters,deviceCounters);
    execution.fence("interface radio final counters");auto c=counters();
    if(c.error)throw std::runtime_error("interface radio incomplete: device error "+std::to_string(c.error));
    auto host=Kokkos::create_mirror_view(moments);Kokkos::deep_copy(execution,host,moments);
    execution.fence("interface radio final moment download");
    Result result;result.config=config;result.moments.resize(host.extent(0));
    for(std::size_t i=0;i<host.extent(0);++i){
      if(!std::isfinite(host(i)))throw std::runtime_error("nonfinite interface radio moment");
      result.moments[i]=host(i);
    }
    host={};moments={};
    if(config.algorithm==Algorithm::CoREAS){
      auto h=Kokkos::create_mirror_view(regularized);Kokkos::deep_copy(execution,h,regularized);
      execution.fence("interface CoREAS final regularized moment download");
      result.regularized_moments.resize(h.extent(0));
      for(std::size_t i=0;i<h.extent(0);++i){
        if(!std::isfinite(h(i)))throw std::runtime_error("nonfinite interface CoREAS regularized moment");
        result.regularized_moments[i]=h(i);
      }
      h={};regularized={};
    }
    auto& s=result.statistics;s.device_tracks=c.device_tracks;s.cpu_tracks=c.cpu_tracks;s.track_observer_pairs=c.pairs;
    s.leaves=c.leaves;s.paths=c.paths;s.direct_paths=c.direct;s.transmitted_paths=c.transmitted;
    s.blocked_paths=c.blocked;s.rejected_faces=c.missing;s.wavefronts=calls_;s.downloads=1;s.device_bytes=bytes_;
    s.endpoint_contributions=c.endpoint_contributions;s.regularized_pairs=c.regularized_pairs;s.boundary_endpoints=c.boundary_endpoints;
    s.roundoff_limited_tracks=c.roundoff_limited_tracks;s.outside_coverage_paths=c.outside_coverage;
    return result;
  }
  void launch(Tracks const& tracks,std::size_t count,Space const& execution,bool cpu) {
    if(!count)return;
    detail::MomentGrid grid{config_.start_time_s,config_.sample_rate_Hz,config_.samples,config_.observers.size()*6,config_.moment_order};
    if(config_.algorithm==Algorithm::CoREAS)
      Kokkos::parallel_for("interface_radio_coreas_endpoints",Kokkos::RangePolicy<Space>(execution,0,count*observers_.extent(0)),
        detail::CoreasEndpointKernel<Space>{tracks,observers_,moments_,regularized_,counters_,propagation_.view(),grid,
          config_.subdivision_frequency_Hz,config_.fraunhofer_limit,config_.mesh_maximum_segment_m,config_.maximum_subdivision_depth,cpu,
          config_.coreas_cherenkov_threshold});
    else Kokkos::parallel_for("interface_radio_zhs_intervals",Kokkos::RangePolicy<Space>(execution,0,count*observers_.extent(0)),
        detail::AccumulateKernel<Space>{tracks,observers_,moments_,counters_,propagation_.view(),grid,
          config_.subdivision_frequency_Hz,config_.fraunhofer_limit,config_.mesh_maximum_segment_m,config_.maximum_subdivision_depth,cpu});
    if constexpr(Paired)
      Kokkos::parallel_for("interface_radio_zhs_intervals",Kokkos::RangePolicy<Space>(execution,0,count*observers_.extent(0)),
        detail::AccumulateKernel<Space>{tracks,observers_,zhs_moments_,zhs_counters_,propagation_.view(),grid,
          config_.subdivision_frequency_Hz,config_.fraunhofer_limit,config_.mesh_maximum_segment_m,config_.maximum_subdivision_depth,cpu});
    ++calls_;
  }
  Config config_;std::size_t bytes_{};KokkosPropagation<Space> propagation_;
  Kokkos::View<detail::DeviceObserver*,Memory> observers_;
  Kokkos::View<double*,Memory> moments_,regularized_,zhs_moments_;
  Kokkos::View<detail::RadioCounters,Memory> counters_,zhs_counters_;
  Tracks cpu_input_;typename Tracks::HostMirror cpu_mirror_;
  Tracks device_pending_,cpu_pending_;
  std::size_t device_buffered_{},cpu_buffered_{};
  std::uint64_t calls_{};bool sealed_{};
};
} // namespace corsika::radio::interface

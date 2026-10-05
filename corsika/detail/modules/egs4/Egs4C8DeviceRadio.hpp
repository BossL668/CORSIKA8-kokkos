#pragma once
#include <corsika/detail/modules/egs4/Egs4C8AirOutput.hpp>
#include <corsika/modules/egs4/Session.hpp>
#include <corsika/accelerator/radio/kokkos/KokkosRadioAccumulator.hpp>
#include <chrono>
#include <type_traits>
#ifdef KOKKOS_ENABLE_CUDA
#include <cuda_runtime_api.h>
#endif

namespace c7_egs4::application {
// Convert validated, pre-interaction segments entirely on the device. No new
// propagation convention or radio formula: reuse the production accumulator.
template<class Outputs,class Tracks>struct ExportRadioTracks {
  Outputs outputs;Tracks tracks;
  KOKKOS_INLINE_FUNCTION void operator()(std::size_t i)const {
    ::corsika::gpu::em::LeptonTransportRecord r{};
    if(outputs(i).has_radio) {
      auto const& source=outputs(i).radio;auto const& s=source.step;
      r.start.pid=r.end.pid=s.pid;r.start.weight=r.end.weight=s.weight;
      r.start.time_s=s.start_time_s;r.end.time_s=s.end_time_s;
      r.start.energy_GeV=s.start_energy_GeV;r.end.energy_GeV=s.end_energy_GeV;
      for(int j=0;j<3;++j) {
        r.start.position_m[j]=s.start_position_m[j];r.end.position_m[j]=s.end_position_m[j];
        r.start.direction[j]=source.start_direction[j];r.end.direction[j]=source.end_direction[j];
      }
    }
    tracks(i)=r; // invalid padding (pid=0) is rejected by the existing kernel
  }
};
template<class Exec>class DeviceRadio {
  using Accumulator=::corsika::accelerator::radio::kokkos_detail::KokkosRadioAccumulator<Exec>;
  Accumulator accumulator_;
  typename Accumulator::TrackInputView tracks_;
  OutputCallbacks const& output_;
  bool finished_{};bool skip_empty_{};
#ifdef KOKKOS_ENABLE_CUDA
  cudaEvent_t start_{},stop_{};
  static void cudaCheck(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
#endif
public:
  DeviceRadio(::corsika::gpu::radio::GpuRadioConfig const& config,OutputCallbacks const& output):output_(output) {
    if(config.enabled&&!output.radio_waveforms)throw std::invalid_argument("Device radio needs a waveform consumer");
    accumulator_.initialize(config);
    skip_empty_=(overhead::options()&overhead::empty_radio)&&config.coreas_observers.empty()&&config.zhs_observers.empty();
#ifdef KOKKOS_ENABLE_CUDA
    if constexpr(std::is_same_v<Exec,Kokkos::Cuda>)if(enabled()) {
      cudaCheck(cudaEventCreate(&start_));
      auto err=cudaEventCreate(&stop_);if(err!=cudaSuccess){cudaEventDestroy(start_);start_=nullptr;cudaCheck(err);}
    }
#endif
  }
  ~DeviceRadio(){
#ifdef KOKKOS_ENABLE_CUDA
    if(stop_)cudaEventDestroy(stop_);if(start_)cudaEventDestroy(start_);
#endif
  }
  DeviceRadio(DeviceRadio const&)=delete;DeviceRadio& operator=(DeviceRadio const&)=delete;
  bool enabled()const{return accumulator_.enabled();}
  void reset(){accumulator_.reset();finished_=false;}
  template<class Outputs>void accumulate(Outputs const& outputs,RunStatistics& stats) {
    if(!enabled()||!outputs.extent(0))return;
    if(finished_)throw std::logic_error("Device radio already exported");
    if(skip_empty_)return;
    auto const n=outputs.extent(0);
    if(tracks_.extent(0)<n)tracks_=typename Accumulator::TrackInputView("egs4_radio_tracks",n);
    Exec execution;auto begin=std::chrono::steady_clock::now();
#ifdef KOKKOS_ENABLE_CUDA
    if constexpr(std::is_same_v<Exec,Kokkos::Cuda>)cudaCheck(cudaEventRecord(start_,execution.cuda_stream()));
#endif
    Kokkos::parallel_for("egs4_export_resident_radio_tracks",Kokkos::RangePolicy<Exec>(execution,0,n),
      ExportRadioTracks<Outputs,typename Accumulator::TrackInputView>{outputs,tracks_});
    accumulator_.accumulateLeptonTracks(tracks_,n,execution);
#ifdef KOKKOS_ENABLE_CUDA
    if constexpr(std::is_same_v<Exec,Kokkos::Cuda>) {
      cudaCheck(cudaEventRecord(stop_,execution.cuda_stream()));cudaCheck(cudaEventSynchronize(stop_));
      float elapsed=0.;cudaCheck(cudaEventElapsedTime(&elapsed,start_,stop_));stats.radio_kernel_ms+=elapsed;
    } else execution.fence("EGS4 radio projection");
#else
    execution.fence("EGS4 radio projection");
#endif
    stats.radio_projection_wall_ms+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();
    ++stats.radio_projection_batches;stats.radio_execution_space=Exec::name();stats.radio=accumulator_.statistics();
  }
  void finish(RunStatistics& stats) {
    if(!enabled())return;
    if(finished_)throw std::logic_error("Device radio downloaded twice");
    finished_=true;auto waveforms=accumulator_.download();stats.radio=accumulator_.statistics();
    stats.radio_execution_space=Exec::name();
    if(!skip_empty_&&stats.radio.lepton_tracks!=stats.radio_tracks)throw std::runtime_error("Device radio lost/duplicated native segments");
    output_.radio_waveforms(waveforms,stats.radio.lepton_tracks);
  }
};
} // namespace c7_egs4::application

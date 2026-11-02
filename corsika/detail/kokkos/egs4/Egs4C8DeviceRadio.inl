/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once
#include <corsika/detail/kokkos/egs4/Egs4C8AirOutput.inl>
#include <corsika/kokkos/egs4/Session.hpp>
#include <corsika/kokkos/radio/RadioAccumulator.hpp>
#include <chrono>
#include <type_traits>
#ifdef KOKKOS_ENABLE_CUDA
#include <cuda_runtime_api.h>
#elif defined(KOKKOS_ENABLE_HIP)
#include <hip/hip_runtime_api.h>
#endif

namespace corsika::kokkos::egs4::application {
  // Convert validated, pre-interaction segments entirely on the device. No new
  // propagation convention or radio formula: reuse the production accumulator.
  template <class TOutputs, class TTracks>
  struct ExportRadioTracks {
    TOutputs outputs_;
    TTracks tracks_;
    KOKKOS_INLINE_FUNCTION void operator()(std::size_t i) const {
      ::corsika::kokkos::LeptonTransportRecord r{};
      if (outputs_(i).hasRadio_) {
        auto const& source = outputs_(i).radio_;
        auto const& s = source.step_;
        r.start_.pid_ = r.end_.pid_ = s.pid_;
        r.start_.weight_ = r.end_.weight_ = s.weight_;
        r.start_.timeS_ = s.startTimeS_;
        r.end_.timeS_ = s.endTimeS_;
        r.start_.energyGeV_ = s.startEnergyGeV_;
        r.end_.energyGeV_ = s.endEnergyGeV_;
        for (int j = 0; j < 3; ++j) {
          r.start_.positionM_[j] = s.startPositionM_[j];
          r.end_.positionM_[j] = s.endPositionM_[j];
          r.start_.direction_[j] = source.startDirection_[j];
          r.end_.direction_[j] = source.endDirection_[j];
        }
      }
      tracks_(i) = r; // invalid padding (pid=0) is rejected by the existing kernel
    }
  };
  template <class TExec>
  class DeviceRadio {
    using accumulator_type = ::corsika::kokkos::radio::RadioAccumulator<TExec>;
    accumulator_type accumulator_;
    typename accumulator_type::track_input_view_type tracks_;
    OutputCallbacks const& output_;
    bool finished_{};
    bool skipEmpty_{};
#ifdef KOKKOS_ENABLE_CUDA
    cudaEvent_t start_{}, stop_{};
    static void cudaCheck(cudaError_t e) {
      if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
    }
#elif defined(KOKKOS_ENABLE_HIP)
    hipEvent_t start_{}, stop_{};
    static void hipCheck(hipError_t error) {
      if (error != hipSuccess) throw std::runtime_error(hipGetErrorString(error));
    }
#endif
  public:
    DeviceRadio(::corsika::kokkos::radio::RadioConfig const& config,
                OutputCallbacks const& output)
        : output_(output) {
      if (config.enabled_ && !output.radioWaveforms_)
        throw std::invalid_argument("Device radio needs a waveform consumer");
      accumulator_.initialize(config);
      skipEmpty_ = (overhead::options() & overhead::empty_radio) &&
                   config.coreasObservers_.empty() && config.zhsObservers_.empty();
#ifdef KOKKOS_ENABLE_CUDA
      if constexpr (std::is_same_v<TExec, Kokkos::Cuda>)
        if (enabled()) {
          cudaCheck(cudaEventCreate(&start_));
          auto err = cudaEventCreate(&stop_);
          if (err != cudaSuccess) {
            cudaEventDestroy(start_);
            start_ = nullptr;
            cudaCheck(err);
          }
        }
#elif defined(KOKKOS_ENABLE_HIP)
      if constexpr (std::is_same_v<TExec, Kokkos::HIP>)
        if (enabled()) {
          hipCheck(hipEventCreate(&start_));
          auto error = hipEventCreate(&stop_);
          if (error != hipSuccess) {
            (void)hipEventDestroy(start_);
            start_ = nullptr;
            hipCheck(error);
          }
        }
#endif
    }
    ~DeviceRadio() {
#ifdef KOKKOS_ENABLE_CUDA
      if (stop_) cudaEventDestroy(stop_);
      if (start_) cudaEventDestroy(start_);
#elif defined(KOKKOS_ENABLE_HIP)
      if (stop_) (void)hipEventDestroy(stop_);
      if (start_) (void)hipEventDestroy(start_);
#endif
    }
    DeviceRadio(DeviceRadio const&) = delete;
    DeviceRadio& operator=(DeviceRadio const&) = delete;
    bool enabled() const { return accumulator_.enabled(); }
    void reset() {
      accumulator_.reset();
      finished_ = false;
    }
    template <class TOutputs>
    void accumulate(TOutputs const& outputs, RunStatistics& stats) {
      if (!enabled() || !outputs.extent(0)) return;
      if (finished_) throw std::logic_error("Device radio already exported");
      if (skipEmpty_) return;
      auto const n = outputs.extent(0);
      if (tracks_.extent(0) < n)
        tracks_ =
            typename accumulator_type::track_input_view_type("egs4_radio_tracks", n);
      TExec execution;
      auto begin = std::chrono::steady_clock::now();
#ifdef KOKKOS_ENABLE_CUDA
      if constexpr (std::is_same_v<TExec, Kokkos::Cuda>)
        cudaCheck(cudaEventRecord(start_, execution.cuda_stream()));
#elif defined(KOKKOS_ENABLE_HIP)
      if constexpr (std::is_same_v<TExec, Kokkos::HIP>)
        hipCheck(hipEventRecord(start_, execution.hip_stream()));
#endif
      Kokkos::parallel_for(
          "egs4_export_resident_radio_tracks",
          Kokkos::RangePolicy<TExec>(execution, 0, n),
          ExportRadioTracks<TOutputs, typename accumulator_type::track_input_view_type>{
              outputs, tracks_});
      accumulator_.accumulateLeptonTracks(tracks_, n, execution);
#ifdef KOKKOS_ENABLE_CUDA
      if constexpr (std::is_same_v<TExec, Kokkos::Cuda>) {
        cudaCheck(cudaEventRecord(stop_, execution.cuda_stream()));
        cudaCheck(cudaEventSynchronize(stop_));
        float elapsed = 0.;
        cudaCheck(cudaEventElapsedTime(&elapsed, start_, stop_));
        stats.radioKernelMs_ += elapsed;
      } else
        execution.fence("EGS4 radio projection");
#elif defined(KOKKOS_ENABLE_HIP)
      if constexpr (std::is_same_v<TExec, Kokkos::HIP>) {
        hipCheck(hipEventRecord(stop_, execution.hip_stream()));
        hipCheck(hipEventSynchronize(stop_));
        float elapsed = 0.;
        hipCheck(hipEventElapsedTime(&elapsed, start_, stop_));
        stats.radioKernelMs_ += elapsed;
      } else
        execution.fence("EGS4 radio projection");
#else
      execution.fence("EGS4 radio projection");
#endif
      stats.radioProjectionWallMs_ += std::chrono::duration<double, std::milli>(
                                          std::chrono::steady_clock::now() - begin)
                                          .count();
      ++stats.radioProjectionBatches_;
      stats.radioExecutionSpace_ = TExec::name();
      stats.radio_ = accumulator_.statistics();
    }
    void finish(RunStatistics& stats) {
      if (!enabled()) return;
      if (finished_) throw std::logic_error("Device radio downloaded twice");
      finished_ = true;
      auto waveforms = accumulator_.download();
      stats.radio_ = accumulator_.statistics();
      stats.radioExecutionSpace_ = TExec::name();
      if (!skipEmpty_ && stats.radio_.leptonTracks_ != stats.radioTracks_)
        throw std::runtime_error("Device radio lost/duplicated native segments");
      output_.radioWaveforms_(waveforms, stats.radio_.leptonTracks_);
    }
  };
} // namespace corsika::kokkos::egs4::application

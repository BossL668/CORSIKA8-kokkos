/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */
#pragma once

#include <Kokkos_Core.hpp>
#include <corsika/accelerator/em/kokkos/CudaCompletionTicket.hpp>
#include <type_traits>

namespace corsika::accelerator::em::kokkos_detail {

  /** One preallocated, asynchronously downloaded wavefront control POD.
   *
   * Only control data is downloaded. The source view is retained until its
   * stream event completes, including during exception unwinding. Callers
   * must not overwrite the device POD before take(). No allocation, global
   * fence, or particle-queue transfer occurs in submit()/poll()/take().
   * Construct and destroy on the runtime's coordinator thread.
   */
  template <class Control>
  class CudaWavefrontControl {
    static_assert(std::is_trivially_copyable_v<Control>);
    static_assert(std::is_standard_layout_v<Control>);
   public:
    using DeviceView = Kokkos::View<Control, Kokkos::CudaSpace>;
    enum class Phase { Idle, WaitingForControl, ResultsReady, Failed };

    CudaWavefrontControl()
        : host_(Kokkos::view_alloc(Kokkos::WithoutInitializing,
                                  "c8_cooperative_control_pinned")) {}
    ~CudaWavefrontControl() { ticket_.drainNoThrow(); }
    CudaWavefrontControl(CudaWavefrontControl const&) = delete;
    CudaWavefrontControl& operator=(CudaWavefrontControl const&) = delete;

    void submit(DeviceView const& source, Kokkos::Cuda const& execution) {
      if (phase_ != Phase::Idle || source.data() == nullptr)
        throw std::logic_error("invalid cooperative control submission");
      source_ = source;
      auto const stream = execution.cuda_stream();
      try {
        check(cudaMemcpyAsync(host_.data(), source_.data(), sizeof(Control),
                              cudaMemcpyDeviceToHost, stream));
        ticket_.record(stream);
        phase_ = Phase::WaitingForControl;
      } catch (...) {
        // Recording may have failed after enqueueing the copy. Do not free
        // its pinned destination or source allocation while it is in flight.
        cudaStreamSynchronize(stream);
        phase_ = Phase::Failed;
        throw;
      }
    }
    bool poll() {
      if (phase_ == Phase::ResultsReady) return true;
      if (phase_ != Phase::WaitingForControl)
        throw std::logic_error("no cooperative control submission to poll");
      try {
        if (!ticket_.ready()) return false;
        phase_ = Phase::ResultsReady;
        return true;
      } catch (...) { phase_ = Phase::Failed; throw; }
    }
    Control take() {
      if (phase_ != Phase::ResultsReady)
        throw std::logic_error("cooperative control was not observed ready");
      try {
        auto const value = host_();
        ticket_.consume();
        source_ = {};
        phase_ = Phase::Idle;
        return value;
      } catch (...) {
        phase_ = Phase::Failed;
        throw;
      }
    }
    Phase phase() const noexcept { return phase_; }
    void const* pinnedAddress() const noexcept { return host_.data(); }
    static constexpr std::size_t pinnedBytes() noexcept { return sizeof(Control); }

   private:
    static void check(cudaError_t status) {
      if (status != cudaSuccess)
        throw std::runtime_error(std::string("CUDA wavefront control copy: ") +
                                 cudaGetErrorString(status));
    }
    // Destruction order matters: the destructor drains the event before any
    // of the two referenced allocations can be released.
    Kokkos::View<Control, Kokkos::CudaHostPinnedSpace> host_;
    DeviceView source_;
    CudaCompletionTicket ticket_;
    Phase phase_{Phase::Idle};
  };
} // namespace corsika::accelerator::em::kokkos_detail

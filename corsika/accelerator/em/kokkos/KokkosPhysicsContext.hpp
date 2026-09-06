/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <Kokkos_Core.hpp>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <type_traits>

#include <corsika/accelerator/em/common/BremsLpm.hpp>
#include <corsika/accelerator/em/common/EmThinning.hpp>
#include <corsika/accelerator/em/common/MoliereScattering.hpp>
#include <corsika/accelerator/em/common/PhotonPairLpm.hpp>
#include <corsika/accelerator/em/common/Types.hpp>
#include <corsika/accelerator/em/common/detail/ProfileProjectionData.hpp>
#include <corsika/accelerator/em/common/tables/NativePhysicsQueries.hpp>

namespace corsika::accelerator::em::kokkos_detail {

  /**
   * Minimal trivially-copyable views used inside the resident kernels.
   *
   * A managed Kokkos::View handle carries allocation tracking and layout
   * state. Capturing many such handles can make a CUDA functor large enough
   * for Kokkos to use its constant-memory launch path. These raw views are
   * formed only after an allocation/growth boundary and remain valid until
   * that workspace is grown again.
   */
  template <class T>
  struct RawView1D {
    T* data{};

    KOKKOS_INLINE_FUNCTION T& operator()(std::size_t const index) const {
      return data[index];
    }
  };

  template <class T>
  struct RawView2D {
    T* data{};
    std::size_t stride_0{};
    std::size_t stride_1{};

    KOKKOS_INLINE_FUNCTION T& operator()(
        std::size_t const first, std::size_t const second) const {
      return data[first * stride_0 + second * stride_1];
    }
  };

  template <class T, class... Properties>
  RawView1D<T> rawDeviceView(
      Kokkos::View<T*, Properties...> const& view) noexcept {
    return {view.data()};
  }

  template <class T, std::size_t Extent, class... Properties>
  RawView2D<T> rawDeviceView(
      Kokkos::View<T*[Extent], Properties...> const& view) noexcept {
    return {view.data(), view.stride_0(), view.stride_1()};
  }

  /**
   * One immutable device-resident physics/environment record plus the small
   * shower state shared by every resident photon and lepton kernel.
   *
   * The object is intentionally pointer-free apart from the existing
   * validated PROPOSAL/Moliere/profile views. It is copied once at backend
   * initialization and once when a new shower changes its keyed RNG or
   * thinning configuration. Individual wavefront launches capture only the
   * rank-zero View handle.
   */
  struct KokkosPhysicsContextData {
    gpu::em::tables::NativePhysicsView physics{};
    gpu::em::EnvironmentSnapshot environment{};
    gpu::em::MoliereSnapshot electron_moliere{};
    gpu::em::MoliereSnapshot muon_moliere{};
    gpu::em::MoliereInterpolationView moliere_interpolation{};
    gpu::em::PhotonPairLpmSnapshot photon_pair_lpm{};
    gpu::em::BremsLpmSnapshot brems_lpm{};
    gpu::em::EmThinningConfig thinning{};
    gpu::em::detail::DeviceProfileProjection profile_projection{};
    std::uint64_t random_seed{};
    std::uint64_t shower_id{};
    std::uint32_t muon_moliere_available{};
  };

  static_assert(std::is_standard_layout_v<KokkosPhysicsContextData>);
  static_assert(std::is_trivially_copyable_v<KokkosPhysicsContextData>);
  static_assert(std::is_trivially_copyable_v<RawView1D<double>>);
  static_assert(std::is_trivially_copyable_v<RawView2D<double>>);

  template <class ExecutionSpace>
  class KokkosPhysicsContext {
  public:
    using memory_space = typename ExecutionSpace::memory_space;
    using DeviceView =
        Kokkos::View<KokkosPhysicsContextData const, memory_space>;

    static constexpr std::size_t projectedDeviceBytes() noexcept {
      return sizeof(KokkosPhysicsContextData);
    }

    void initialize(
        gpu::em::tables::NativePhysicsView const& physics,
        gpu::em::EnvironmentSnapshot const& environment,
        gpu::em::MoliereSnapshot const& electron_moliere,
        gpu::em::MoliereSnapshot const& muon_moliere,
        gpu::em::MoliereInterpolationView const& moliere_interpolation,
        bool const muon_moliere_available,
        gpu::em::PhotonPairLpmSnapshot const& photon_pair_lpm,
        gpu::em::BremsLpmSnapshot const& brems_lpm,
        gpu::em::EmThinningConfig const& thinning,
        gpu::em::detail::DeviceProfileProjection const& profile_projection,
        std::uint64_t const random_seed, std::uint64_t const shower_id,
        ExecutionSpace const& execution) {
      if (device_.data() == nullptr) {
        device_ = DeviceStorage("c8_kokkos_physics_context");
        host_ = HostStorage("c8_kokkos_host_physics_context");
      }
      auto& context = host_();
      context.physics = physics;
      context.environment = environment;
      context.electron_moliere = electron_moliere;
      context.muon_moliere = muon_moliere;
      context.moliere_interpolation = moliere_interpolation;
      context.photon_pair_lpm = photon_pair_lpm;
      context.brems_lpm = brems_lpm;
      context.thinning = thinning;
      context.profile_projection = profile_projection;
      context.random_seed = random_seed;
      context.shower_id = shower_id;
      context.muon_moliere_available = muon_moliere_available ? 1U : 0U;
      Kokkos::deep_copy(execution, device_, host_);
      execution.fence("initialize Kokkos physics context");
      initialized_ = true;
    }

    void beginShower(
        gpu::em::EmThinningConfig const& thinning,
        std::uint64_t const random_seed, std::uint64_t const shower_id,
        ExecutionSpace const& execution) {
      if (!initialized_)
        throw std::logic_error(
            "Kokkos physics context is not initialized");
      host_().thinning = thinning;
      host_().random_seed = random_seed;
      host_().shower_id = shower_id;
      Kokkos::deep_copy(execution, device_, host_);
      // The HostSpace source is persistent and updated in place.  Complete
      // the one context copy per shower before a later beginShower() can
      // overwrite it or an empty shower can destroy the session.  This is not
      // a wavefront fence and therefore does not serialize the hot pipeline.
      execution.fence("update Kokkos physics context for shower");
    }

    bool initialized() const noexcept { return initialized_; }

    DeviceView deviceView() const noexcept { return DeviceView(device_); }

    std::size_t deviceBytes() const noexcept {
      return device_.data() == nullptr ? 0 : sizeof(KokkosPhysicsContextData);
    }

  private:
    using DeviceStorage =
        Kokkos::View<KokkosPhysicsContextData, memory_space>;
    using HostStorage = Kokkos::View<KokkosPhysicsContextData, Kokkos::HostSpace>;

    DeviceStorage device_{};
    HostStorage host_{};
    bool initialized_{};
  };

  template <class ExecutionSpace>
  using KokkosPhysicsContextView =
      typename KokkosPhysicsContext<ExecutionSpace>::DeviceView;

} // namespace corsika::accelerator::em::kokkos_detail

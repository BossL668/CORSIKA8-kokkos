/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

namespace corsika::kokkos {

  template <typename TExecutionSpace>
  constexpr std::size_t PhysicsContext<TExecutionSpace>::projectedDeviceBytes() noexcept {
    return sizeof(PhysicsContextData);
  }

  template <typename TExecutionSpace>
  void PhysicsContext<TExecutionSpace>::initialize(
      InteractionPhysicsView const& physics, EnvironmentBorrowedView const& environment,
      TransportGeometryConfig const& geometry, MoliereSnapshot const& electronMoliere,
      MoliereSnapshot const& muonMoliere,
      MoliereInterpolationView const& moliereInterpolation,
      bool const muonMoliereAvailable, PhotonPairLpmSnapshot const& photonPairLpm,
      BremsLpmSnapshot const& bremsLpm, EmThinningConfig const& thinning,
      ProfileProjectionView const& profileProjection, std::uint64_t const randomSeed,
      std::uint64_t const showerId, TExecutionSpace const& execution) {
    if (device_.data() == nullptr) {
      device_ = device_storage_type("c8_kokkos_physics_context");
      host_ = host_storage_type("c8_kokkos_host_physics_context");
    }
    auto& context = host_();
    context.physics_ = physics;
    context.environment_ = environment;
    context.geometry_ = geometry;
    context.electronMoliere_ = electronMoliere;
    context.muonMoliere_ = muonMoliere;
    context.moliereInterpolation_ = moliereInterpolation;
    context.photonPairLpm_ = photonPairLpm;
    context.bremsLpm_ = bremsLpm;
    context.thinning_ = thinning;
    context.profileProjection_ = profileProjection;
    context.randomSeed_ = randomSeed;
    context.showerId_ = showerId;
    context.muonMoliereAvailable_ = muonMoliereAvailable ? 1U : 0U;
    Kokkos::deep_copy(execution, device_, host_);
    execution.fence("initialize Kokkos physics context");
    initialized_ = true;
  }

  template <typename TExecutionSpace>
  void PhysicsContext<TExecutionSpace>::beginShower(EmThinningConfig const& thinning,
                                                    std::uint64_t const randomSeed,
                                                    std::uint64_t const showerId,
                                                    TExecutionSpace const& execution) {
    if (!initialized_)
      throw std::logic_error("Kokkos physics context is not initialized");
    host_().thinning_ = thinning;
    host_().randomSeed_ = randomSeed;
    host_().showerId_ = showerId;
    Kokkos::deep_copy(execution, device_, host_);
    // The HostSpace source is persistent and updated in place.  Complete
    // the one context copy per shower before a later beginShower() can
    // overwrite it or an empty shower can destroy the session.  This is not
    // a wavefront fence and therefore does not serialize the hot pipeline.
    execution.fence("update Kokkos physics context for shower");
  }

  template <typename TExecutionSpace>
  bool PhysicsContext<TExecutionSpace>::isInitialized() const noexcept {
    return initialized_;
  }

  template <typename TExecutionSpace>
  auto PhysicsContext<TExecutionSpace>::getDeviceView() const noexcept
      -> device_view_type {
    return device_view_type(device_);
  }

  template <typename TExecutionSpace>
  std::size_t PhysicsContext<TExecutionSpace>::getDeviceBytes() const noexcept {
    return device_.data() == nullptr ? 0 : sizeof(PhysicsContextData);
  }

} // namespace corsika::kokkos

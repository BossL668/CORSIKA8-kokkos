/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <filesystem>
#include <memory>
#include <vector>

#include <corsika/accelerator/em/AcceleratedPhysicsRequirements.hpp>
#include <corsika/accelerator/em/KokkosEmBackend.hpp>
#include <corsika/modules/proposal/NativeCalculatorView.hpp>

namespace corsika::accelerator::em::detail {

  struct KokkosSessionBeginResult {
    KokkosEmBackend* backend{};
    bool reused{};
    bool muon_transport_available{};
  };

  /**
   * Run-level proposal-native session.  Kokkos is initialized once, immutable
   * spline/auxiliary data are uploaded once, and subsequent showers retain
   * those allocations while resetting only shower-local state.
   */
  class KokkosEmRunSession {
  public:
    explicit KokkosEmRunSession(
        KokkosRuntimeConfig const& = {},
        std::filesystem::path auxiliary_cache = {});
    ~KokkosEmRunSession();

    KokkosEmRunSession(KokkosEmRunSession&&) noexcept;
    KokkosEmRunSession& operator=(KokkosEmRunSession&&) noexcept;
    KokkosEmRunSession(KokkosEmRunSession const&) = delete;
    KokkosEmRunSession& operator=(KokkosEmRunSession const&) = delete;

    KokkosSessionBeginResult beginProposalNative(
        gpu::em::EnvironmentSnapshot const&, gpu::em::GpuEmConfig const&,
        AcceleratedPhysicsRequirements const&,
        std::vector<proposal::NativeInteractionCalculatorView> const&,
        std::vector<proposal::NativeContinuousCalculatorView> const&);

    /** Fail-closed compatibility entry point for generic application code. */


    bool initialized() const noexcept;
    KokkosEmBackend& backend();
    gpu::em::tables::ProposalNativeTableSet const* loadedNativeTable()
        const noexcept;

  private:
    class Impl;
    std::unique_ptr<Impl> impl_;
  };

} // namespace corsika::accelerator::em::detail

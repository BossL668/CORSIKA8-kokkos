/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once
#include <corsika/kokkos/em/PhysicsRequirements.hpp>
#include <corsika/kokkos/em/execution/EmBackend.hpp>
#include <corsika/modules/proposal/NativeCalculatorView.hpp>
#include <filesystem>
#include <memory>
#include <vector>
namespace corsika::kokkos {
  struct SessionBeginResult {
    EmBackend* backend_{};
    bool reused_{};
    bool muonTransportAvailable_{};
  };
  /** Production run-level session: export/upload once and reset shower-local state.
   * The caller retains the native calculators and host environment owners; the
   * backend owns its synchronous device upload. Environment changes are rejected
   * on reuse, as in production. This class does not reschedule particles.
   */
  class EmRunSession {
  public:
    explicit EmRunSession(RuntimeConfig const& = {},
                          std::filesystem::path auxiliaryCache = {});
    ~EmRunSession();
    EmRunSession(EmRunSession&&) noexcept;
    EmRunSession& operator=(EmRunSession&&) noexcept;
    EmRunSession(EmRunSession const&) = delete;
    EmRunSession& operator=(EmRunSession const&) = delete;
    SessionBeginResult beginProposalNative(
        EnvironmentDataView const&, TransportGeometryConfig const&, BackendConfig const&,
        PhysicsRequirements const&,
        std::vector<proposal::NativeInteractionCalculatorView> const&,
        std::vector<proposal::NativeContinuousCalculatorView> const&);
    bool isInitialized() const noexcept;
    EmBackend& getBackend();
    ProposalNativeTableSet const* getLoadedNativeTable() const noexcept;

  private:
    class Impl;
    std::unique_ptr<Impl> impl_;
  };
} // namespace corsika::kokkos

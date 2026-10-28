/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/kokkos/application/cascade/RunEnvironmentConfig.hpp>
#include <corsika/kokkos/application/BackendOptions.hpp>
#include <corsika/kokkos/em/EmRunSession.hpp>
#include <corsika/kokkos/output/EmRunOutput.hpp>
#include <memory>

namespace corsika::kokkos::application {
  /** Production application-owned configuration and run-session lifetime.
   * The underlying backend remains uninitialized until beginProposalNative.
   * This wrapper neither transports particles nor selects a scheduling policy.
   */
  class CascadeRunSession {
  public:
    CascadeRunSession(BackendOptions const&, RunEnvironmentConfig const&);
    bool isEnabled() const noexcept;
    EmRunOutput* getRunOutput() noexcept;
    bool hasBackend() const noexcept;
    EmRunSession& getRuntimeSession();

  private:
    std::unique_ptr<EmRunSession> runtimeSession_{};
    std::unique_ptr<EmRunOutput> runOutput_{};
  };
} // namespace corsika::kokkos::application

#include <corsika/detail/kokkos/application/cascade/CascadeRunSession.inl>

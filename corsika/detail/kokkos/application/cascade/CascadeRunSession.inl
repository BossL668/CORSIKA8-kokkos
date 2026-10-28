/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

namespace corsika::kokkos::application {
  inline bool CascadeRunSession::isEnabled() const noexcept {
    return runOutput_ != nullptr;
  }
  inline EmRunOutput* CascadeRunSession::getRunOutput() noexcept {
    return runOutput_.get();
  }
  inline bool CascadeRunSession::hasBackend() const noexcept {
    return runtimeSession_ && runtimeSession_->isInitialized();
  }
} // namespace corsika::kokkos::application

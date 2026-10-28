/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */
#pragma once

namespace corsika::kokkos::application {
  template <typename TRun>
  CascadeAdapter<TRun>::CascadeAdapter(TRun run)
      : run_(run) {}

  template <typename TRun>
  void CascadeAdapter<TRun>::forceInteraction() {
    forceInteraction_ = true;
  }

  template <typename TRun>
  void CascadeAdapter<TRun>::forceDecay() {
    forceDecay_ = true;
  }

  template <typename TRun>
  void CascadeAdapter<TRun>::run() {
    auto configure = [this](auto& cascade) {
      if (forceInteraction_) cascade.forceInteraction();
      if (forceDecay_) cascade.forceDecay();
    };
    run_(configure);
  }
} // namespace corsika::kokkos::application

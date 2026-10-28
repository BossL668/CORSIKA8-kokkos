/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */
#pragma once

namespace corsika::kokkos::application {
  /** Event-local facade exposing the same force/run operations as Cascade.
   * The stored callback borrows event objects that must outlive this adapter.
   * It receives a callable that applies the requested primary interaction or
   * decay flags to the selected cascade before execution.
   */
  template <typename TRun>
  class CascadeAdapter {
  public:
    /** Store the event runner; referenced event state remains caller-owned. */
    explicit CascadeAdapter(TRun run);
    /** Request a forced primary interaction when run() constructs the cascade. */
    void forceInteraction();
    /** Request a forced primary decay when run() constructs the cascade. */
    void forceDecay();
    /** Apply the primary flags and execute the stored event runner. */
    void run();

  private:
    TRun run_;
    bool forceInteraction_ = false;
    bool forceDecay_ = false;
  };
  template <typename TRun>
  CascadeAdapter(TRun)->CascadeAdapter<TRun>;
} // namespace corsika::kokkos::application

#include <corsika/detail/kokkos/application/cascade/CascadeAdapter.inl>

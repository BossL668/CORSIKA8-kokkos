/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */
#pragma once

#include <corsika/kokkos/application/DeviceSelection.hpp>

namespace corsika::kokkos::application {
  /** Select process organization before creating models or a device runtime.
   * Direct execution includes native C8, OpenMP and a single GPU. Coordinated
   * execution retains the existing multi-GPU and parallel-prefix implementation.
   */
  class ExecutionSelection {
  public:
    /** Whether the selected options require the process coordinator. */
    bool isCoordinated() const noexcept { return coordinated_; }
    /** Device selection to pass to the application's backend option validation. */
    DeviceSelection const& devices() const noexcept { return devices_; }
    /** Run the coordinator without entering the application's model setup.
     * Only valid for coordinated execution. The original command line is retained.
     * Set supportsWorkerSession only when the application handles repeated worker
     * requests through WorkerSession.
     */
    int runCoordinator(int argc, char** argv, bool supportsWorkerSession = false) const;

  private:
    ExecutionSelection(DeviceSelection, bool);
    friend ExecutionSelection select_execution(int, char**);
    DeviceSelection devices_;
    bool coordinated_;
  };

  /** Inspect execution options without initializing models, RNGs or Kokkos.
   * Parsing help does not query GPU hardware.
   * Invalid device selections are logged and reported with CascadeAborted.
   */
  ExecutionSelection select_execution(int argc, char** argv);
} // namespace corsika::kokkos::application

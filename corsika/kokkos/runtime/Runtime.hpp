/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <memory>
#include <string>

namespace corsika::kokkos {

  /**
   * Manage or borrow the single Kokkos runtime of this process.
   *
   * The first instance initializes Kokkos if necessary. Subsequent instances share
   * its lifetime; the last instance finalizes only an initialization owned here.
   * An externally initialized runtime is borrowed and is never finalized here.
   * Kokkos cannot be initialized again after finalization.
   *
   * Construct and destroy all instances on the coordinating host thread. An external
   * initialization must also belong to that thread and outlive all instances. Destroy
   * all Kokkos Views and execution-space objects before releasing the last instance.
   * Destruction on another thread is a contract violation and terminates the process.
   *
   * This class neither selects a physics backend nor schedules particles. The default
   * execution space is selected when building the Kokkos dependency.
   */
  class Runtime {
  public:
    /**
     * Initialize Kokkos, or acquire a compatible existing runtime.
     *
     * @param numThreads Host parallel-backend thread count; zero leaves the Kokkos
     *   default/environment setting unchanged, or accepts an existing runtime.
     *   This is not the number of CUDA threads.
     * @param deviceId Visible GPU index; minus one leaves the default unchanged or
     *   accepts an existing runtime. CPU-only builds reject explicit device indices.
     * @throws std::invalid_argument Invalid parameters or incompatible configuration.
     * @throws std::logic_error Reinitialization or acquisition on another host thread.
     */
    explicit Runtime(int numThreads = 0, int deviceId = -1);

    /// Release this lifetime reference; finalize only the last owned reference.
    ~Runtime();

    Runtime(Runtime const&) = delete;
    Runtime& operator=(Runtime const&) = delete;
    Runtime(Runtime&&) = delete;
    Runtime& operator=(Runtime&&) = delete;

    /// Actual host parallel-backend concurrency, including environment defaults.
    [[nodiscard]] int getNumThreads() const;

    /// Actual visible GPU index, or minus one for a CPU-only build.
    [[nodiscard]] int getDeviceId() const;

    /// Name of the build-selected default execution space.
    [[nodiscard]] std::string getExecutionSpaceName() const;

    /// Whether the shared lifetime owns initialization (rather than borrowing it).
    [[nodiscard]] bool ownsInitialization() const;

  private:
    struct State;
    std::shared_ptr<State> state_;
  };

} // namespace corsika::kokkos

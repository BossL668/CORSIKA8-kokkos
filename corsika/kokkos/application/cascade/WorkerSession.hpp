/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#ifdef CORSIKA_KOKKOS_COORDINATOR
#include <corsika/framework/random/RNGManager.hpp>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>
#include <nlohmann/json.hpp>
#include <unistd.h>
#endif

namespace corsika::kokkos::application {
  /** Internal request loop for one statically assigned GPU.
   * Model objects and the Kokkos session stay alive. Each request owns a separate
   * output library, input shard and device shower namespace. Native CPU
   * random streams continue for the lifetime of the worker. Publication of
   * done_<id> happens only after the event scope and all output streams close.
   */
  class WorkerSession {
  public:
    /** Read the coordinator's request channel, or remain inactive in a direct run. */
    WorkerSession();
    ~WorkerSession() = default;
    WorkerSession(WorkerSession const&) = delete;
    WorkerSession& operator=(WorkerSession const&) = delete;
    WorkerSession(WorkerSession&&) = delete;
    WorkerSession& operator=(WorkerSession&&) = delete;

    /** Whether this process serves requests from a resident worker pool. */
    bool isActive() const;

    /** Register event-local histogram and decay resets; retain the model RNG streams. */
    template <typename TDecay, typename THigh, typename TLow>
    void ready(TDecay& decay, THigh& high, TLow& low);
    /** Register the decay flags restored between independent output libraries. */
    template <typename TDecay>
    void ready(TDecay& decay);
    /** Publish CPU-model readiness without creating a GPU backend. */
    void ready();

    /** Close output before acknowledging a completed request. */
    template <typename TOutput>
    void close(TOutput& output);
    /** Read the next nonempty shard, preserving CPU streams and the GPU session.
     * Returns false after the final request. A direct run returns true unchanged.
     * The previous event must have left scope before calling this method.
     */
    template <typename TBackend, typename TApplication, typename TOutput>
    bool next(TBackend& backend, TApplication& application, TOutput& output, long& seed);

  private:
#ifdef CORSIKA_KOKKOS_COORDINATOR
    using clock_type = std::chrono::steady_clock;
    using json_type = nlohmann::json;

    double elapsed() const;
    void publish(std::string const& name, json_type value) const;
    template <typename TOutput>
    void finish(TOutput& output);

    std::filesystem::path control_;
    unsigned request_ = 0;
    unsigned requests_ = 0;
    double timeout_ = 0.;
    long initialSeed_ = 0;
    bool pending_ = false;
    bool libraryOpen_ = true;
    clock_type::time_point started_ = clock_type::now();
    clock_type::time_point eventStart_;
    json_type current_;
    std::function<void()> restoreDecayFlags_;
    std::function<void()> resetHistograms_;
#endif
  };
} // namespace corsika::kokkos::application

#include <corsika/detail/kokkos/application/cascade/WorkerSession.inl>

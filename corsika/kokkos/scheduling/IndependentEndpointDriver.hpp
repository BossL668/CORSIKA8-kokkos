/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */
#pragma once
#include <condition_variable>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <utility>

namespace corsika::kokkos {
  // One submitted job; initialization and finalization stay on the owner thread.
  class IndependentEndpointDriver {
  public:
    IndependentEndpointDriver();
    IndependentEndpointDriver(IndependentEndpointDriver const&) = delete;
    IndependentEndpointDriver& operator=(IndependentEndpointDriver const&) = delete;
    ~IndependentEndpointDriver();
    template <typename TFunction>
    auto submit(TFunction&& f);
    void waitIdle();

  private:
    void run();
    std::mutex mutex_;
    std::condition_variable changed_, idle_;
    std::function<void()> job_;
    bool busy_{}, stopping_{};
    std::thread thread_;
  };
} // namespace corsika::kokkos
#include <corsika/detail/kokkos/scheduling/IndependentEndpointDriver.inl>

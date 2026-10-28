/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once
#if defined(__linux__)
#include <pthread.h>
#endif

namespace corsika::kokkos {
  inline IndependentEndpointDriver::IndependentEndpointDriver()
      : thread_([this] { run(); }) {}

  inline IndependentEndpointDriver::~IndependentEndpointDriver() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stopping_ = true;
    }
    changed_.notify_one();
    thread_.join(); // also drains a submitted job during exception unwinding
  }

  template <typename TFunction>
  inline auto IndependentEndpointDriver::submit(TFunction&& f) {
    using result_type = std::invoke_result_t<TFunction>;
    auto task =
        std::make_shared<std::packaged_task<result_type()>>(std::forward<TFunction>(f));
    auto result = task->get_future();
    // Allocate before changing state: bad_alloc must not leave busy_=true
    // with no job, which would strand waitIdle during error unwinding.
    std::function<void()> pending = [task] { (*task)(); };
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (busy_ || stopping_)
        throw std::logic_error("Endpoint already has an in-flight job");
      job_.swap(pending); // noexcept; the stored slot was empty
      busy_ = true;
    }
    changed_.notify_one();
    return result;
  }

  inline void IndependentEndpointDriver::waitIdle() {
    std::unique_lock<std::mutex> lock(mutex_);
    idle_.wait(lock, [this] { return !busy_; });
  }

  inline void IndependentEndpointDriver::run() {
#if defined(__linux__)
    // Diagnostic only: distinguish the CUDA submitter from the OpenMP pool
    // in /proc and top -H. No affinity or scheduling policy is changed.
    (void)pthread_setname_np(pthread_self(), "c8-cuda-driver");
#endif
    for (;;) {
      std::function<void()> job;
      {
        std::unique_lock<std::mutex> lock(mutex_);
        changed_.wait(lock, [this] { return stopping_ || bool(job_); });
        if (!job_) return;
        job.swap(job_); // leave an explicitly empty slot, even across libraries
      }
      job(); // packaged_task transports all exceptions to the owner
      {
        std::lock_guard<std::mutex> lock(mutex_);
        job = {}; // destroy job captures before announcing idle
        busy_ = false;
      }
      idle_.notify_all();
    }
  }
} // namespace corsika::kokkos

/* Internal bounded host driver. No Kokkos initialization, physics, or shared Stack. */
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
#if defined(__linux__)
#include <pthread.h>
#endif

namespace corsika::accelerator::em::detail {
// Exactly one job in flight; no detached threads and no growing task queue.
// Initialization/finalization remain on the Kokkos runtime owner thread.
class IndependentEndpointDriver {
 public:
  IndependentEndpointDriver() : thread_([this] { run(); }) {}
  IndependentEndpointDriver(IndependentEndpointDriver const&) = delete;
  IndependentEndpointDriver& operator=(IndependentEndpointDriver const&) = delete;
  ~IndependentEndpointDriver() {
    { std::lock_guard<std::mutex> lock(mutex_); stopping_ = true; }
    changed_.notify_one();
    thread_.join(); // also drains a submitted job during exception unwinding
  }
  template<class F> auto submit(F&& f) {
    using R = std::invoke_result_t<F>;
    auto task = std::make_shared<std::packaged_task<R()>>(std::forward<F>(f));
    auto result = task->get_future();
    // Allocate before changing state: bad_alloc must not leave busy_=true
    // with no job, which would strand waitIdle during error unwinding.
    std::function<void()> pending = [task] { (*task)(); };
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (busy_ || stopping_) throw std::logic_error("Endpoint already has an in-flight job");
      job_.swap(pending); // noexcept; the stored slot was empty
      busy_ = true;
    }
    changed_.notify_one();
    return result;
  }
  void waitIdle() {
    std::unique_lock<std::mutex> lock(mutex_);
    idle_.wait(lock, [this] { return !busy_; });
  }
 private:
  void run() {
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
  std::mutex mutex_;
  std::condition_variable changed_, idle_;
  std::function<void()> job_;
  bool busy_{}, stopping_{};
  std::thread thread_;
};
} // namespace corsika::accelerator::em::detail

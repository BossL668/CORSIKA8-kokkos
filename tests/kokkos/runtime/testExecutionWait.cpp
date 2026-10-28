/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include <corsika/kokkos/runtime/ResidentExecutionWait.hpp>
#include <corsika/kokkos/runtime/Runtime.hpp>
#include <tests/kokkos/runtime/ExecutionWaitReference.hpp>

#include <catch2/catch_test_macros.hpp>
#include <atomic>
#include <chrono>
#include <thread>
#include <vector>

namespace {
  namespace cp = corsika::kokkos;
  using execution_space_type = Kokkos::DefaultExecutionSpace;

  void ensure_runtime() { static cp::Runtime runtime; }

  struct CountingExecution {
    mutable int fences_{};
    void fence(char const*) const { ++fences_; }
  };

  template <class TWait>
  std::vector<int> check_wait() {
    execution_space_type execution;
    TWait waiter;
    std::vector<int> result;
    REQUIRE_FALSE(waiter.blockingEnabled());
    REQUIRE(waiter.blockingCalls() == 0);
    REQUIRE(waiter.blockingHostSeconds() == 0.);
    Kokkos::View<int*, typename execution_space_type::memory_space> data("wait data", 32);
    for (int repeat = 0; repeat < 4; ++repeat) {
      Kokkos::parallel_for(
          "wait input", Kokkos::RangePolicy<execution_space_type>(execution, 0, 32),
          KOKKOS_LAMBDA(int i) { data(i) = 31 - i + repeat; });
      waiter.wait(execution, "wait ordinary execution");
      auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, data);
      for (int i = 0; i < 32; ++i) {
        REQUIRE(host(i) == 31 - i + repeat);
        result.push_back(host(i));
      }
    }
    int callbacks = 0;
    waiter.setProgress([&]() {
      ++callbacks;
      return true;
    });
    REQUIRE_THROWS_AS(waiter.setBlocking(true), std::logic_error);
    waiter.wait(execution, "optional cooperative execution");
    waiter.setProgress({});
    waiter.setBlocking(false);
#ifdef KOKKOS_ENABLE_CUDA
    if constexpr (std::is_same_v<execution_space_type, Kokkos::Cuda>) {
      waiter.setBlocking(true);
      REQUIRE(waiter.blockingEnabled());
      REQUIRE_THROWS_AS(waiter.setProgress([]() { return true; }), std::logic_error);
      for (int i = 0; i < 3; ++i) {
        Kokkos::deep_copy(execution, data, i);
        waiter.wait(execution, "blocking execution");
        REQUIRE(waiter.blockingCalls() == static_cast<std::uint64_t>(i + 1));
        auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, data);
        REQUIRE(host(0) == i);
      }
      REQUIRE(waiter.blockingHostSeconds() >= 0.);
      waiter.resetStatistics();
      REQUIRE(waiter.blockingEnabled());
      waiter.setBlocking(false);
    } else
#endif
    {
      REQUIRE(callbacks == 0);
      REQUIRE_THROWS_AS(waiter.setBlocking(true), std::logic_error);
    }
    waiter.resetStatistics();
    REQUIRE(waiter.blockingCalls() == 0);
    REQUIRE(waiter.blockingHostSeconds() == 0.);
    return result;
  }

#ifdef KOKKOS_ENABLE_CUDA
  // Hold a stream without CUDA API calls from its host callback. A deadline
  // prevents a failed assertion from leaving device work blocked indefinitely.
  struct StreamGate {
    cudaStream_t stream_{};
    std::atomic<bool> released_{false};
    std::atomic<bool> finished_{false};
    std::atomic<bool> timedOut_{false};
    ~StreamGate() {
      released_.store(true);
      if (stream_) cudaStreamSynchronize(stream_);
    }
    cudaError_t enqueue(cudaStream_t stream) {
      stream_ = stream;
      return cudaLaunchHostFunc(stream, &StreamGate::hold, this);
    }
    static void CUDART_CB hold(void* pointer) {
      auto& gate = *static_cast<StreamGate*>(pointer);
      auto const deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
      while (!gate.released_.load()) {
        if (std::chrono::steady_clock::now() >= deadline) {
          gate.timedOut_.store(true);
          break;
        }
        std::this_thread::yield();
      }
      gate.finished_.store(true);
    }
  };

  template <class TTicket>
  void check_ticket(cudaStream_t stream) {
    TTicket polling;
    REQUIRE_FALSE(polling.pending());
    REQUIRE_THROWS_AS(polling.ready(), std::logic_error);
    REQUIRE_THROWS_AS(polling.consume(), std::logic_error);
    REQUIRE_THROWS_AS(polling.consumeBlocking(), std::logic_error);
    REQUIRE(polling.drainNoThrow() == cudaSuccess);
    for (int i = 0; i < 3; ++i) {
      StreamGate gate;
      REQUIRE(gate.enqueue(stream) == cudaSuccess);
      polling.record(stream);
      REQUIRE(polling.pending());
      REQUIRE_THROWS_AS(polling.record(stream), std::logic_error);
      REQUIRE_FALSE(polling.ready());
      REQUIRE_THROWS_AS(polling.consume(), std::logic_error);
      gate.released_.store(true);
      REQUIRE(polling.drainNoThrow() == cudaSuccess);
      REQUIRE(gate.finished_.load());
      REQUIRE_FALSE(gate.timedOut_.load());
      REQUIRE_FALSE(polling.pending());
      polling.record(stream);
      REQUIRE(cudaStreamSynchronize(stream) == cudaSuccess);
      REQUIRE(polling.ready());
      polling.consume();
      REQUIRE_FALSE(polling.pending());
    }
    TTicket blocking(true);
    REQUIRE_THROWS_AS(blocking.consumeBlocking(), std::logic_error);
    blocking.record(stream);
    blocking.consumeBlocking();
    REQUIRE_FALSE(blocking.pending());
  }

  template <class TWait>
  void check_cooperation(Kokkos::Cuda const& execution) {
    for (bool throws : {false, true}) {
      StreamGate gate;
      TWait waiter;
      int callbacks = 0;
      waiter.setProgress([&]() {
        ++callbacks;
        REQUIRE_THROWS_AS(waiter.resetStatistics(), std::logic_error);
        REQUIRE_THROWS_AS(waiter.setBlocking(false), std::logic_error);
        gate.released_.store(true);
        if (throws) throw std::runtime_error("intentional progress failure");
        return true;
      });
      REQUIRE(gate.enqueue(execution.cuda_stream()) == cudaSuccess);
      if (throws) {
        REQUIRE_THROWS_AS(waiter.wait(execution, "exceptional progress"),
                          std::runtime_error);
      } else {
        waiter.wait(execution, "cooperative progress");
      }
      REQUIRE(callbacks > 0);
      REQUIRE(gate.finished_.load());
      REQUIRE_FALSE(gate.timedOut_.load());
      waiter.setProgress({});
      REQUIRE_NOTHROW(waiter.resetStatistics());
      REQUIRE_NOTHROW(waiter.setBlocking(true));
      REQUIRE_NOTHROW(waiter.wait(execution, "reused after progress"));
    }
  }
#endif
} // namespace

TEST_CASE("Production execution wait results and mode contracts",
          "[kokkos][runtime][wait]") {
  ensure_runtime();
  auto migrated = check_wait<cp::ResidentExecutionWait<execution_space_type>>();
  auto original =
      check_wait<execution_wait_reference::ResidentExecutionWait<execution_space_type>>();
  REQUIRE(migrated == original);
  execution_space_type execution;
  cp::wait_resident_execution(
      execution, "null waiter",
      static_cast<cp::ResidentExecutionWait<execution_space_type>*>(nullptr));
  cp::ResidentExecutionWait<execution_space_type> waiter;
  cp::wait_resident_execution(execution, "provided waiter", &waiter);
}

TEST_CASE("Production CUDA completion and exceptional drain", "[kokkos][runtime][wait]") {
  ensure_runtime();
#ifdef KOKKOS_ENABLE_CUDA
  cudaStream_t stream{};
  REQUIRE(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess);
  {
    Kokkos::Cuda execution(stream);
    check_ticket<cp::CudaCompletionTicket>(stream);
    check_ticket<execution_wait_reference::CudaCompletionTicket>(stream);
    check_cooperation<cp::ResidentExecutionWait<Kokkos::Cuda>>(execution);
    check_cooperation<execution_wait_reference::ResidentExecutionWait<Kokkos::Cuda>>(
        execution);
  }
  REQUIRE(cudaStreamDestroy(stream) == cudaSuccess);
#endif
  CountingExecution execution;
  cp::ResidentExecutionWait<CountingExecution> waiter;
  int callbacks = 0;
  waiter.setProgress([&]() {
    ++callbacks;
    return true;
  });
  cp::wait_resident_execution(execution, "host wrapper", &waiter);
  cp::wait_resident_execution(
      execution, "host null wrapper",
      static_cast<cp::ResidentExecutionWait<CountingExecution>*>(nullptr));
  REQUIRE(execution.fences_ == 2);
  REQUIRE(callbacks == 0);
}

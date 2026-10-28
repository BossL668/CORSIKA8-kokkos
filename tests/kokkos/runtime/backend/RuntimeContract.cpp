/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */
#include <corsika/kokkos/runtime/BackendRuntime.hpp>
#include <corsika/kokkos/runtime/Runtime.hpp>
#include <Kokkos_Core.hpp>
#include <array>
#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>

#ifndef CORSIKA_KOKKOS_PROJECT_REVISION
#error "BackendRuntime target must export its build revision"
#endif

std::uint64_t runtime_other_translation_unit(corsika::kokkos::BackendRuntime const&);

namespace {
  namespace cp = corsika::kokkos;
  void require(bool value, char const* message) {
    if (!value) throw std::runtime_error(message);
  }
  template <typename TException, typename TCall>
  void rejects(TCall call) {
    bool rejected = false;
    try {
      call();
    } catch (TException const&) { rejected = true; }
    require(rejected, "Expected configuration/argument rejection");
  }
  void probes(cp::BackendRuntime const& runtime) {
    for (std::size_t n : {1U, 2U, 17U, 513U, 4097U}) {
      std::uint64_t selected = 0, checksum = 0, slots = 0;
      for (std::size_t i = 0; i < n; ++i) {
        if (i % 3 != 1) {
          ++selected;
          checksum += i + 1;
        }
        slots += i % 4;
      }
      for (std::size_t chunk : {0U, 1U, 16U, 64U}) {
        auto result = runtime.runPrimitiveProbe(n, chunk);
        require(result.values_ == n && result.selected_ == selected &&
                    result.checksum_ == checksum && result.scanValid_,
                "Primitive scan/reduction result changed");
      }
      auto queue = runtime.runQueueProbe(n);
      require(queue.inputParticles_ == n && queue.retainedParticles_ == selected &&
                  queue.secondarySlots_ == slots && queue.stableOrder_ &&
                  queue.roundtripExact_,
              "Queue order, offsets or round trip changed");
    }
    for (auto dimensions : {std::array<std::size_t, 5>{1, 1, 1, 1, 1},
                            std::array<std::size_t, 5>{17, 11, 32, 8, 5},
                            std::array<std::size_t, 5>{65, 33, 64, 16, 8}}) {
      auto [tracks, observers, team, trackTile, observerTile] = dimensions;
      std::uint64_t checksum = 0;
      for (std::size_t t = 0; t < tracks; ++t)
        for (std::size_t o = 0; o < observers; ++o) checksum += (t + 1) * (o + 3);
      auto result =
          runtime.runTilingProbe(tracks, observers, team, trackTile, observerTile);
      require(result.exact_ && result.tracks_ == tracks &&
                  result.observers_ == observers && result.checksum_ == checksum,
              "Tiled matrix coverage changed");
    }
    runtime_other_translation_unit(runtime);
    rejects<std::invalid_argument>([&] { runtime.runPrimitiveProbe(0); });
    rejects<std::invalid_argument>([&] {
      runtime.runPrimitiveProbe(
          static_cast<std::size_t>(std::numeric_limits<int>::max()) + 1);
    });
    rejects<std::invalid_argument>([&] { runtime.runQueueProbe(0); });
    for (unsigned field = 0; field < 5; ++field) {
      std::array<std::size_t, 5> d{1, 1, 1, 1, 1};
      d[field] = 0;
      rejects<std::invalid_argument>(
          [&] { runtime.runTilingProbe(d[0], d[1], d[2], d[3], d[4]); });
    }
    rejects<std::invalid_argument>([&] { runtime.runTilingProbe(1, 1, 1025, 1, 1); });
  }
  void owner(std::string const& backend, int threads) {
    cp::RuntimeConfig config;
    config.executionBackend_ = backend;
    config.threads_ = threads;
    {
      cp::BackendRuntime runtime(config);
      require(Kokkos::is_initialized(), "Backend did not initialize Kokkos");
      auto const info = runtime.getInfo();
      require(info.backend_ == backend && info.concurrency_ > 0 &&
                  info.hostThreads_ == threads && !info.kokkosVersion_.empty() &&
                  !info.deviceName_.empty() && !info.compilerVersion_.empty() &&
                  info.projectRevision_ == CORSIKA_KOKKOS_PROJECT_REVISION,
              "RuntimeInfo does not describe the selected backend/build");
      require(info.gpu_ == (backend != "openmp") && info.openmp_ == (backend == "openmp"),
              "Backend device flags disagree");
      if (info.gpu_)
        require(
            info.deviceTotalMemoryBytes_ >= info.deviceFreeMemoryBytesAtInitialization_ &&
                info.deviceTotalMemoryBytes_ > 0 && !info.architecture_.empty(),
            "GPU information was not queried");
      rejects<std::logic_error>([&] { cp::BackendRuntime duplicate(config); });
      rejects<std::logic_error>([&] { runtime.shareCooperativeLifetime(); });
      {
        cp::Runtime borrowed;
        require(!borrowed.ownsInitialization(),
                "Generic Runtime did not borrow backend owner");
        require(borrowed.getNumThreads() == threads,
                "Borrowed host thread count changed");
        bool rejected = false;
        std::thread other([&] {
          try {
            cp::Runtime wrongThread;
          } catch (std::logic_error const&) { rejected = true; }
        });
        other.join();
        require(rejected, "Generic runtime accepted a different coordinating thread");
      }
      require(Kokkos::is_initialized(), "Borrower finalized backend lifetime");
      cp::BackendRuntime moved(std::move(runtime));
      probes(moved);
      runtime = std::move(moved);
      require(runtime.getInfo().backend_ == backend, "Move assignment lost backend info");
      std::cout << "INFO " << backend << ' ' << info.concurrency_ << ' '
                << info.hostThreads_ << ' ' << info.deviceTotalMemoryBytes_ << ' '
                << info.deviceFreeMemoryBytesAtInitialization_ << ' '
                << info.architecture_ << ' ' << info.kokkosVersion_ << ' '
                << info.driverVersion_ << ' ' << info.runtimeVersion_ << '\n'
                << "DEVICE " << info.deviceName_ << '\n';
    }
    require(Kokkos::is_finalized(), "Backend owner did not finalize Kokkos");
  }
  void cooperative(std::string const& backend) {
    cp::RuntimeConfig config;
    config.executionBackend_ = backend;
    config.cooperativeOwner_ = true;
    config.threads_ = 2;
#ifdef CORSIKA_KOKKOS_BACKEND_CUDA_OPENMP
    config.executionBackend_ = "cuda";
    auto owner = std::make_unique<cp::BackendRuntime>(config);
    auto lease = owner->shareCooperativeLifetime();
    cp::RuntimeConfig borrowed;
    borrowed.device_ = 0;
    borrowed.threads_ = 2;
    borrowed.executionBackend_ = "openmp";
    borrowed.runtimeLease_ = lease;
    auto incompatible = borrowed;
    incompatible.device_ = 1;
    rejects<std::invalid_argument>([&] { cp::BackendRuntime wrongDevice(incompatible); });
    incompatible = borrowed;
    incompatible.threads_ = 3;
    rejects<std::invalid_argument>(
        [&] { cp::BackendRuntime wrongThreads(incompatible); });
    incompatible = borrowed;
    incompatible.cooperativeOwner_ = true;
    rejects<std::invalid_argument>([&] { cp::BackendRuntime wrongOwner(incompatible); });
    incompatible.runtimeLease_.reset();
    bool rejected = false;
    std::thread wrongThread([&] {
      try {
        cp::BackendRuntime wrong(borrowed);
      } catch (std::invalid_argument const&) { rejected = true; }
    });
    wrongThread.join();
    require(rejected, "Cooperative borrower accepted the wrong coordinating thread");
    {
      cp::BackendRuntime peer(borrowed);
      owner.reset();
      require(Kokkos::is_initialized(), "Cooperative owner finalized a live lease");
      probes(peer);
    }
    require(Kokkos::is_initialized(), "External lease was not retained");
    borrowed.runtimeLease_.reset();
    lease.reset();
    require(Kokkos::is_finalized(), "Final cooperative lease did not finalize");
#else
    rejects<std::invalid_argument>([&] { cp::BackendRuntime unsupported(config); });
    require(!Kokkos::is_initialized(), "Rejected cooperative mode initialized Kokkos");
#endif
  }
} // namespace

int main(int argc, char** argv) {
  try {
    require(argc >= 3, "Expected scenario and backend");
    std::string scenario = argv[1], backend = argv[2];
    require(!Kokkos::is_initialized() && !Kokkos::is_finalized(),
            "Test needs a fresh process");
    if (scenario == "owner")
      owner(backend, argc > 3 ? std::stoi(argv[3]) : 1);
    else if (scenario == "generic-first") {
      {
        cp::Runtime generic(1);
        require(generic.ownsInitialization(), "Generic owner missing");
        rejects<std::logic_error>([] { cp::BackendRuntime refused; });
      }
      require(Kokkos::is_finalized(), "Generic owner did not finalize");
    } else if (scenario == "external-first") {
      Kokkos::initialize(Kokkos::InitializationSettings().set_num_threads(1));
      rejects<std::logic_error>([] { cp::BackendRuntime refused; });
      {
        cp::Runtime borrowed;
        require(!borrowed.ownsInitialization(), "External owner was replaced");
      }
      require(Kokkos::is_initialized(),
              "Generic borrower finalized external initialization");
      Kokkos::finalize();
    } else if (scenario == "unavailable") {
      cp::RuntimeConfig config;
      config.executionBackend_ = "not-compiled";
      rejects<std::invalid_argument>([&] { cp::BackendRuntime refused(config); });
      require(!Kokkos::is_initialized(), "Invalid backend initialized Kokkos");
    } else if (scenario == "gpu-threads") {
      cp::RuntimeConfig config;
      config.executionBackend_ = backend;
      config.threads_ = 2;
      rejects<std::invalid_argument>([&] { cp::BackendRuntime refused(config); });
      require(!Kokkos::is_initialized(), "Invalid GPU threads initialized Kokkos");
    } else if (scenario == "cooperative")
      cooperative(backend);
    else
      throw std::runtime_error("Unknown scenario");
    std::cout << "PASS " << scenario << '\n';
  } catch (std::exception const& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}

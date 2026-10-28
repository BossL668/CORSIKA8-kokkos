/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include <corsika/kokkos/runtime/Runtime.hpp>

#include <Kokkos_Core.hpp>

#include <exception>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <utility>

namespace corsika::kokkos {

  namespace {

    std::recursive_mutex& runtime_mutex() {
      // Also used during rollback if construction fails after initialization.
      static std::recursive_mutex mutex;
      return mutex;
    }

    void validate_request(int const numThreads, int const deviceId) {
      if (numThreads < 0 || deviceId < -1) {
        throw std::invalid_argument("Invalid Kokkos thread count or device index.");
      }
#ifdef KOKKOS_ENABLE_SERIAL
      if constexpr (std::is_same_v<Kokkos::DefaultHostExecutionSpace, Kokkos::Serial>) {
        if (numThreads > 1) {
          throw std::invalid_argument("This Kokkos host backend is serial.");
        }
      }
#endif
      if (deviceId >= 0 && deviceId >= Kokkos::num_devices()) {
        throw std::invalid_argument("Requested Kokkos device is not available.");
      }
    }

  } // namespace

  struct Runtime::State {
    std::thread::id ownerThread_ = std::this_thread::get_id();
    bool ownsInitialization_ = false;
    int numThreads_ = 0;
    int deviceId_ = -1;
    std::string executionSpaceName_;

    ~State() {
      std::lock_guard<std::recursive_mutex> const lock(runtime_mutex());
      if (std::this_thread::get_id() != ownerThread_) { std::terminate(); }
      if (ownsInitialization_) {
        if (!Kokkos::is_initialized() || Kokkos::is_finalized()) { std::terminate(); }
        Kokkos::fence("CORSIKA runtime finalization");
        Kokkos::finalize();
      }
    }

    void checkCompatibility(int const numThreads, int const deviceId) const {
      if (std::this_thread::get_id() != ownerThread_) {
        throw std::logic_error("Kokkos runtime belongs to another host thread.");
      }
      if ((numThreads != 0 && numThreads != numThreads_) ||
          (deviceId != -1 && deviceId != deviceId_)) {
        throw std::invalid_argument("Existing Kokkos runtime configuration differs.");
      }
    }
  };

  Runtime::Runtime(int const numThreads, int const deviceId) {
    std::lock_guard<std::recursive_mutex> const lock(runtime_mutex());
    static std::weak_ptr<State> sharedState;

    if (Kokkos::is_finalized()) {
      throw std::logic_error("Kokkos cannot be reinitialized after finalization.");
    }
    validate_request(numThreads, deviceId);

    if (auto existing = sharedState.lock()) {
      existing->checkCompatibility(numThreads, deviceId);
      state_ = std::move(existing);
      return;
    }

    auto state = std::make_shared<State>();
    if (!Kokkos::is_initialized()) {
      Kokkos::InitializationSettings settings;
      if (numThreads != 0) { settings.set_num_threads(numThreads); }
      if (deviceId != -1) { settings.set_device_id(deviceId); }
      Kokkos::initialize(settings);
      state->ownsInitialization_ = true;
    }

    state->numThreads_ = Kokkos::num_threads();
    state->deviceId_ = Kokkos::device_id();
    state->executionSpaceName_ = Kokkos::DefaultExecutionSpace::name();
    state->checkCompatibility(numThreads, deviceId);
    sharedState = state;
    state_ = std::move(state);
  }

  Runtime::~Runtime() {
    if (std::this_thread::get_id() != state_->ownerThread_) { std::terminate(); }
  }

  int Runtime::getNumThreads() const { return state_->numThreads_; }

  int Runtime::getDeviceId() const { return state_->deviceId_; }

  std::string Runtime::getExecutionSpaceName() const {
    return state_->executionSpaceName_;
  }

  bool Runtime::ownsInitialization() const { return state_->ownsInitialization_; }

} // namespace corsika::kokkos

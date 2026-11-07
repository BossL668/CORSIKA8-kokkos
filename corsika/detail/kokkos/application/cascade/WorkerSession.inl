/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

namespace corsika::kokkos::application {
#ifdef CORSIKA_KOKKOS_COORDINATOR
  inline double WorkerSession::elapsed() const {
    return std::chrono::duration<double>(clock_type::now() - started_).count();
  }

  inline void WorkerSession::publish(std::string const& name, json_type value) const {
    auto const path = control_ / name;
    if (std::filesystem::exists(path))
      throw std::runtime_error("Worker marker already exists: " + path.string());
    auto const temporary = path.string() + ".tmp";
    {
      std::ofstream out(temporary);
      out.exceptions(std::ios::badbit | std::ios::failbit);
      out << value.dump(2) << '\n';
    }
    std::filesystem::rename(temporary, path);
  }

  inline WorkerSession::WorkerSession() {
    if (auto const path = std::getenv("C8_WORKER_SESSION")) {
      control_ = path;
      auto const count = std::getenv("C8_WORKER_EVENTS");
      auto const timeout = std::getenv("C8_WORKER_TIMEOUT");
      if (!control_.is_absolute() || !count || !timeout)
        throw std::runtime_error("Incomplete resident worker configuration");
      requests_ = std::stoul(count);
      timeout_ = std::stod(timeout);
      if (!requests_ || !(timeout_ > 0.))
        throw std::runtime_error("Invalid resident worker limits");
    }
  }

  inline bool WorkerSession::isActive() const { return !control_.empty(); }

  template <typename TDecay, typename THigh, typename TLow>
  void WorkerSession::ready(TDecay& decay, THigh& high, TLow& low) {
    if (!isActive()) return;
    resetHistograms_ = [&high, &low]() {
      high.resetHistogram();
      low.resetHistogram();
    };
    ready(decay);
  }

  template <typename TDecay>
  void WorkerSession::ready(TDecay& decay) {
    if (!isActive()) return;
    std::map<int, bool> flags;
    auto& data = decay.particleData;
    for (int id = data.nextId(0); id != 0; id = data.nextId(id))
      flags.emplace(id, data.mayDecay(id));
    restoreDecayFlags_ = [&data, flags = std::move(flags)]() {
      for (auto const& [id, value] : flags) data.mayDecay(id, value);
    };
    ready();
  }

  inline void WorkerSession::ready() {
    if (!isActive()) return;
    auto& manager = RNGManager<>::getInstance();
    initialSeed_ = static_cast<long>(manager.getSeed());
    publish("READY.json",
            {{"pid", getpid()},
             {"models_ready", true},
             {"gpu_initialization_policy", "first-request-then-reuse"},
             {"cpu_rng_policy", "continue-worker-streams"},
             {"cpu_initial_seed", initialSeed_},
             {"gpu_rng_policy", "request-seed-and-particle-history"},
             {"unix", std::chrono::duration<double>(
                          std::chrono::system_clock::now().time_since_epoch())
                          .count()}});
  }

  template <typename TOutput>
  void WorkerSession::close(TOutput& output) {
    if (!isActive() || libraryOpen_) output.endOfLibrary();
    libraryOpen_ = false;
    finish(output);
  }

  template <typename TOutput>
  void WorkerSession::finish(TOutput& output) {
    if (!isActive() || !pending_) return;
    output.flushLibrarySummary();
    auto record = current_;
    record["pid"] = getpid();
    record["returncode"] = 0;
    record["skipped"] = false;
    record["end_s"] = elapsed();
    record["wall_s"] =
        std::chrono::duration<double>(clock_type::now() - eventStart_).count();
    publish("done_" + std::to_string(request_ - 1) + ".json", record);
    pending_ = false;
  }

  template <typename TBackend, typename TApplication, typename TOutput>
  bool WorkerSession::next(TBackend& backend, TApplication& application, TOutput& output,
                           long& seed) {
    if (!isActive()) return true;
    // The preceding CascadeEvent has left scope, including its input and radio sinks.
    close(output);
    while (request_ < requests_) {
      auto const path = control_ / ("request_" + std::to_string(request_) + ".json");
      auto const waiting = clock_type::now();
      while (!std::filesystem::exists(path)) {
        if (std::chrono::duration<double>(clock_type::now() - waiting).count() > timeout_)
          throw std::runtime_error("Timed out waiting for worker request");
        std::this_thread::sleep_for(std::chrono::milliseconds(25));
      }
      std::ifstream in(path);
      current_ = json_type::parse(in);
      if (current_.at("request") != request_)
        throw std::runtime_error("Unexpected resident worker request number");
      ++request_;
      if (current_.at("skip").get<bool>()) {
        publish("done_" + std::to_string(request_ - 1) + ".json",
                {{"request", request_ - 1},
                 {"pid", getpid()},
                 {"returncode", 0},
                 {"skipped", true}});
        continue;
      }
      eventStart_ = clock_type::now();
      current_["start_s"] = elapsed();
      seed = current_.at("seed").get<long>();
      if (seed <= 0) throw std::runtime_error("Invalid worker seed");
      auto const folder = current_.at("folder").get<std::string>();
      std::filesystem::current_path(folder);
      backend.assignWorkerInput(
          current_.at("frontier").get<std::string>(),
          current_.at("worker_id").get<unsigned>(),
          current_.at("radio_inputs").get<std::vector<std::string>>());
      // Native CPU streams, copied generator callbacks and SOPHIA's private
      // stream continue across requests, as in the native N > 1 event loop.
      // Reset event outputs and counters without rewinding those streams.
      current_["cpu_rng_policy"] = "continue-worker-streams";
      current_["cpu_initial_seed"] = initialSeed_;
      if (restoreDecayFlags_) restoreDecayFlags_();
      if (resetHistograms_) resetHistograms_();
      // Device sampling and selected CPU fallback replay retain their per-event
      // request seed. This does not reseed the native CPU RNGManager.
      application.setWorkerSeed(static_cast<std::uint64_t>(seed));
      output.reopenLibrary((std::filesystem::path(folder) / "shower").string(), seed);
      libraryOpen_ = true;
      pending_ = true;
      return true;
    }
    return false;
  }
#else
  inline WorkerSession::WorkerSession() = default;
  inline bool WorkerSession::isActive() const { return false; }
  template <typename TDecay, typename THigh, typename TLow>
  void WorkerSession::ready(TDecay&, THigh&, TLow&) {}
  template <typename TDecay>
  void WorkerSession::ready(TDecay&) {}
  inline void WorkerSession::ready() {}
  template <typename TOutput>
  void WorkerSession::close(TOutput& output) {
    output.endOfLibrary();
  }
  template <typename TBackend, typename TApplication, typename TOutput>
  bool WorkerSession::next(TBackend&, TApplication&, TOutput&, long&) {
    return true;
  }
#endif
} // namespace corsika::kokkos::application

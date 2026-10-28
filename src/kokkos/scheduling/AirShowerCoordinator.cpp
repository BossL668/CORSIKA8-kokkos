/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

// C++ process management; children exec the same air-shower binary.
#include <corsika/framework/core/Logging.hpp>

#include <corsika/kokkos/scheduling/AirShowerCoordinator.hpp>
#include <corsika/kokkos/application/DeviceSelection.hpp>
#include <CLI/CLI.hpp>
#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdlib>
#include <fcntl.h>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <sys/prctl.h>
#include <sys/wait.h>
#include <unistd.h>
extern char** environ;

namespace corsika::kokkos::coordinator {
  namespace {
    using clock_type = std::chrono::steady_clock;
    volatile std::sig_atomic_t interrupted = 0;
    void signal_handler(int signal) { interrupted = signal; }
    void require(bool ok, std::string const& message) {
      if (!ok) throw std::runtime_error(message);
    }
    std::string number(double x) {
      std::ostringstream out;
      out << std::setprecision(17) << x;
      return out.str();
    }
    json_type read_json(fs::path const& path) {
      std::ifstream in(path);
      require(bool(in), "Cannot read " + path.string());
      return json_type::parse(in);
    }
    double seconds(clock_type::time_point start) {
      return std::chrono::duration<double>(clock_type::now() - start).count();
    }
    struct Child {
      pid_t pid_;
      fs::path folder_;
      json_type record_;
      bool reaped_ = false;
      clock_type::time_point started_;
    };
    struct Processes {
      std::vector<Child> children_;
      clock_type::time_point started_ = clock_type::now();
      ~Processes() {
        // Only our own sessions; keep child PIDs unreaped until signals are sent.
        for (auto const& c : children_)
          if (!c.reaped_) kill(-c.pid_, SIGTERM);
        auto deadline = clock_type::now() + std::chrono::seconds(3);
        for (;;) {
          bool pending = false;
          for (auto& c : children_)
            if (!c.reaped_) {
              int status;
              if (waitpid(c.pid_, &status, WNOHANG) == c.pid_)
                c.reaped_ = true;
              else
                pending = true;
            }
          if (!pending || clock_type::now() >= deadline) break;
          std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
        for (auto& c : children_)
          if (!c.reaped_) {
            kill(-c.pid_, SIGKILL);
            while (waitpid(c.pid_, nullptr, 0) < 0 && errno == EINTR) {}
          }
      }
      std::size_t launch(std::vector<std::string> const& command, fs::path const& folder,
                         std::map<std::string, std::string> const& overrides) {
        check_interrupt();
        require(fs::create_directory(folder),
                "Process folder exists: " + folder.string());
        save(folder / "COMMAND.json", command);
        std::map<std::string, std::string> env;
        for (char** p = environ; *p; ++p) {
          std::string s(*p);
          auto eq = s.find('=');
          if (eq != std::string::npos) env[s.substr(0, eq)] = s.substr(eq + 1);
        }
        for (auto const& e : overrides) env[e.first] = e.second;
        std::vector<std::string> storage;
        for (auto const& e : env) storage.push_back(e.first + "=" + e.second);
        std::vector<char*> argv, envp;
        for (auto const& a : command) argv.push_back(const_cast<char*>(a.c_str()));
        for (auto& a : storage) envp.push_back(a.data());
        argv.push_back(nullptr);
        envp.push_back(nullptr);
        int log = open((folder / "run.log").c_str(),
                       O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0644);
        require(log >= 0, "Cannot create process log");
        auto parent = getpid();
        auto before = clock_type::now();
        pid_t pid = fork();
        if (pid == 0) {
          // No allocations, shell, Python, or CUDA operations between fork and exec.
          if (setsid() < 0 || chdir(folder.c_str()) < 0 || dup2(log, 1) < 0 ||
              dup2(log, 2) < 0)
            _exit(126);
          close(log);
          int input = open("/dev/null", O_RDONLY);
          if (input < 0 || dup2(input, 0) < 0) _exit(126);
          if (input != 0) close(input);
          prctl(PR_SET_PDEATHSIG, SIGTERM);
          if (getppid() != parent) _exit(126);
          execve(command.front().c_str(), argv.data(), envp.data());
          _exit(127);
        }
        close(log);
        require(pid > 0, "fork failed");
        children_.push_back(
            {pid,
             folder,
             {{"name", folder.filename().string()},
              {"pid", pid},
              {"start_s", std::chrono::duration<double>(before - started_).count()}},
             false,
             before});
        save(folder / "OWNED_PROCESS.json", {{"pid", pid}});
        return children_.size() - 1;
      }
      void wait(std::vector<std::size_t> pending, double timeout) {
        while (!pending.empty()) {
          check_interrupt();
          for (auto it = pending.begin(); it != pending.end();) {
            auto& c = children_[*it];
            int status = 0;
            auto rc = waitpid(c.pid_, &status, WNOHANG);
            if (rc < 0 && errno == EINTR) continue;
            require(rc >= 0, "waitpid failed");
            if (rc == c.pid_) {
              c.reaped_ = true;
              int code = WIFEXITED(status) ? WEXITSTATUS(status) : -WTERMSIG(status);
              c.record_["returncode"] = code;
              c.record_["end_s"] = seconds(started_);
              c.record_["wall_s"] = seconds(c.started_);
              save(c.folder_ / "PROCESS_EXIT.json", c.record_);
              require(code == 0, "Process failed: " + c.folder_.string() + " (exit " +
                                     std::to_string(code) + ")");
              it = pending.erase(it);
            } else {
              require(seconds(c.started_) <= timeout,
                      "Process deadline exceeded: " + c.folder_.string());
              ++it;
            }
          }
          if (!pending.empty())
            std::this_thread::sleep_for(std::chrono::milliseconds(25));
        }
      }
      json_type records() const {
        json_type out = json_type::array();
        for (auto const& c : children_) out.push_back(c.record_);
        return out;
      }
    };
    fs::path executable_on_path(std::string const& name) {
      std::istringstream path(std::getenv("PATH") ? std::getenv("PATH") : "");
      std::string folder;
      while (std::getline(path, folder, ':')) {
        auto p = fs::absolute((folder.empty() ? fs::path(".") : fs::path(folder)) / name);
        if (access(p.c_str(), X_OK) == 0) return p;
      }
      if (name == "nvidia-smi" && access("/usr/lib/wsl/lib/nvidia-smi", X_OK) == 0)
        return "/usr/lib/wsl/lib/nvidia-smi";
      throw std::runtime_error("Executable not found: " + name);
    }
    std::vector<std::string> resolve_devices(std::vector<std::string> const& requested,
                                             fs::path const& output) {
      Processes p;
      auto i = p.launch({executable_on_path("nvidia-smi").string(),
                         "--query-gpu=index,uuid", "--format=csv,noheader,nounits"},
                        output / "device_query", {});
      p.wait({i}, 15);
      std::ifstream in(output / "device_query/run.log");
      std::ostringstream listing;
      listing << in.rdbuf();
      return corsika::kokkos::application::resolve_gpu_ids(requested, listing.str());
    }
    void check_frontier_feed(fs::path const& folder, std::size_t expected) {
      auto feed = read_json(folder / "FRONTIER_FEED.json");
      auto maximum = feed.at("maximum_imported_batch").get<std::size_t>();
      auto const batches = feed.at("batches").get<std::size_t>();
      require(feed.at("mode") == "bounded-high-energy-first" &&
                  feed.at("roots_consumed").get<std::size_t>() == expected &&
                  (expected ? maximum > 0 && maximum <= 65536 && batches > 0
                            : maximum == 0 && batches == 0),
              "Incomplete frontier consumption: " + folder.string());
    }
    void event(CoordinatorOptions const& o, fs::path const& root, std::uint64_t seed,
               std::vector<std::string> const& devices) {
      Processes p;
      auto physics = o.physics_;
      physics.insert(physics.end(), {"-E", number(o.energy_)});
      auto common = physics;
      common.insert(common.end(), {"-N", "1", "--hadronic-workers", "1",
                                   "--hadronic-backend", "scalar"});
      if (o.emBackend_ == "kokkos-egs4")
        common.insert(common.end(), {"--egs4-stepfc", number(o.egs4Stepfc_)});
      std::map<std::string, std::string> env{{"OMP_NUM_THREADS", "1"},
                                             {"OMP_THREAD_LIMIT", "1"},
                                             {"OPENBLAS_NUM_THREADS", "1"},
                                             {"MKL_NUM_THREADS", "1"},
                                             {"ARROW_NUM_THREADS", "1"},
                                             {"ARROW_IO_THREADS", "1"},
                                             {"CUDA_VISIBLE_DEVICES", devices.front()}};
      auto launch = [&](std::string const& name, std::vector<std::string> const& options,
                        std::map<std::string, std::string> const& environment) {
        std::vector<std::string> args{o.worker_.string()};
        args.insert(args.end(), common.begin(), common.end());
        args.insert(args.end(), {"-f", (root / name / "shower").string()});
        args.insert(args.end(), options.begin(), options.end());
        return p.launch(args, root / name, environment);
      };
      try {
        save(root / "CONFIG.json", {{"command", {o.worker_.string()}},
                                    {"physics_args", physics},
                                    {"seed", seed},
                                    {"devices", devices},
                                    {"gpu_memory_fraction", o.memoryFraction_},
                                    {"em_backend", o.emBackend_},
                                    {"egs4_stepfc", o.egs4Stepfc_},
                                    {"cpu_prefix_workers", o.prefixWorkers_},
                                    {"cpu_prefix_split_energy_GeV", o.prefixSplitEnergy_},
                                    {"coordinator", "C++"},
                                    {"output", root.string()}});
        auto frontier = root / "frontier.txt";
        auto cap =
            o.frontierEnergy_ > 0 ? o.frontierEnergy_ : o.energy_ / (8 * devices.size());
        std::vector<fs::path> parts;
        std::vector<fs::path> radioSpools;
        bool const spoolPrefixRadio = o.emBackend_ == "kokkos-proposal";
        auto prefixRadioOptions = [&](std::string const& name) {
          std::vector<std::string> options;
          if (spoolPrefixRadio) {
            auto const file = root / name / "radio_tracks.bin";
            radioSpools.push_back(file);
            options = {"--radio-track-out", file.string()};
          }
          return options;
        };
        fs::path primaryPart;
        auto prefixStart = clock_type::now();
        if (o.prefixWorkers_ == 0) {
          CORSIKA_LOG_INFO("Seed {}: computing serial shower prefix", seed);
          auto options = prefixRadioOptions("prefix");
          options.insert(options.end(),
                         {"-s", std::to_string(seed), "--em-backend",
                          o.emBackend_ == "kokkos-egs4" ? "kokkos-egs4" : "proposal",
                          "--radio-backend", "cpu", "--frontier-out", frontier.string(),
                          "--frontier-max-energy", number(cap)});
          auto prefixEnv = env;
          if (spoolPrefixRadio) prefixEnv["CUDA_VISIBLE_DEVICES"] = "";
          auto prefix = launch("prefix", options, prefixEnv);
          p.wait({prefix}, o.timeout_);
          validate_outputs(root / "prefix/shower");
          primaryPart = root / "prefix/shower";
          parts.push_back(primaryPart);
        } else {
          CORSIKA_LOG_INFO("Seed {}: static CPU bootstrap + {} prefix workers", seed,
                           o.prefixWorkers_);
          auto cpuEnv = env;
          cpuEnv["CUDA_VISIBLE_DEVICES"] = "";
          auto cpuRoots = root / "cpu_frontier.txt";
          auto bootstrapOptions = prefixRadioOptions("bootstrap");
          bootstrapOptions.insert(
              bootstrapOptions.end(),
              {"-s", std::to_string(seed), "--em-backend", "proposal", "--radio-backend",
               "cpu", "--prefix-cpu-out", cpuRoots.string(), "--prefix-cpu-max-energy",
               number(o.prefixSplitEnergy_ > 0 ? o.prefixSplitEnergy_ : cap)});
          auto bootstrap = launch("bootstrap", bootstrapOptions, cpuEnv);
          p.wait({bootstrap}, o.timeout_);
          validate_outputs(root / "bootstrap/shower");
          primaryPart = root / "bootstrap/shower";
          parts.push_back(primaryPart);
          std::vector<fs::path> cpuShards;
          for (unsigned i = 0; i < o.prefixWorkers_; ++i)
            cpuShards.push_back(root / ("cpu_roots_" + std::to_string(i) + ".txt"));
          auto cpuAudit = partition(cpuRoots, cpuShards);
          save(root / "CPU_PARTITION.json", cpuAudit);
          require(cpuAudit["roots"].get<std::size_t>() > 0,
                  "Bootstrap produced no CPU subshower roots");
          std::vector<std::size_t> cpuWorkers;
          std::vector<fs::path> cpuFrontiers;
          auto historyLimit =
              (std::uint64_t(o.prefixWorkers_) + 1) * (std::uint64_t{1} << 48);
          for (unsigned i = 0; i < o.prefixWorkers_; ++i) {
            if (cpuAudit["counts"][i] == 0) continue;
            auto name = "cpu_prefix_" + std::to_string(i);
            auto file = root / (name + "_frontier.txt");
            cpuFrontiers.push_back(file);
            auto options = prefixRadioOptions(name);
            options.insert(
                options.end(),
                {"-s", std::to_string(worker_seed(seed, 1024 + i)), "--em-backend",
                 "proposal", "--radio-backend", "cpu", "--prefix-cpu-in",
                 cpuShards[i].string(), "--frontier-worker-id", std::to_string(i + 1),
                 "--frontier-out", file.string(), "--frontier-max-energy", number(cap),
                 "--frontier-history-limit", std::to_string(historyLimit)});
            cpuWorkers.push_back(launch(name, options, cpuEnv));
            parts.push_back(root / name / "shower");
          }
          p.wait(cpuWorkers, o.timeout_);
          for (unsigned i = 0; i < o.prefixWorkers_; ++i) {
            if (cpuAudit["counts"][i] == 0) continue;
            auto folder = root / ("cpu_prefix_" + std::to_string(i));
            validate_outputs(folder / "shower");
            auto feed = read_json(folder / "CPU_FRONTIER_FEED.json");
            require(feed["roots_consumed"] == cpuAudit["counts"][i],
                    "CPU prefix did not consume its complete shard");
            require(digest(cpuShards[i]) == cpuAudit["shard_sha256"][i],
                    "CPU shard changed during execution");
          }
          require(digest(cpuRoots) == cpuAudit["frontier_sha256"],
                  "CPU bootstrap frontier changed");
          save(root / "FRONTIER_JOIN.json", join_frontiers(cpuFrontiers, frontier));
        }
        double prefixPhase = seconds(prefixStart);
        std::vector<fs::path> shards;
        for (std::size_t i = 0; i < devices.size(); ++i)
          shards.push_back(root / ("roots_" + std::to_string(i) + ".txt"));
        auto audit = partition(frontier, shards);
        save(root / "PARTITION.json", audit);
        std::vector<unsigned> activeRadioWorkers;
        for (unsigned i = 0; i < devices.size(); ++i)
          if (audit["counts"][i] != 0) activeRadioWorkers.push_back(i);
        // The prefix may have completed every particle while still producing
        // radio tracks. An empty frontier worker can project those tracks without
        // injecting a synthetic primary or running any CPU radio calculations.
        if (activeRadioWorkers.empty() && !radioSpools.empty())
          activeRadioWorkers.push_back(0);
        auto const radioAudit = partition_radio_tracks(radioSpools, activeRadioWorkers);
        save(root / "RADIO_TRACK_PARTITION.json", radioAudit);
        std::set<unsigned> scheduledWorkers;
        for (unsigned i = 0; i < devices.size(); ++i)
          if (audit["counts"][i] != 0) scheduledWorkers.insert(i);
        for (auto const& spool : radioAudit.at("sources"))
          if (!spool.at("worker").is_null())
            scheduledWorkers.insert(spool.at("worker").get<unsigned>());
        std::vector<std::size_t> workers;
        CORSIKA_LOG_INFO("Seed {}: {} roots on {} GPUs", seed, audit["counts"].dump(),
                         devices.size());
        for (unsigned i = 0; i < devices.size(); ++i) {
          if (!scheduledWorkers.count(i)) continue;
          auto workerEnv = env;
          workerEnv["CUDA_VISIBLE_DEVICES"] = devices[i];
          std::vector<std::string> workerOptions{
              "-s",
              std::to_string(worker_seed(seed, i)),
              "--em-backend",
              o.emBackend_,
              "--radio-backend",
              "kokkos",
              "--kokkos-execution",
              "cuda",
              "--kokkos-num-threads",
              "1",
              "--kokkos-device",
              "0",
              "--gpu-memory-fraction",
              number(o.memoryFraction_),
              "--gpu-resident-batch-limit",
              "0",
              "--frontier-in",
              shards[i].string(),
              "--frontier-worker-id",
              std::to_string(o.prefixWorkers_ + i + 1)};
          for (auto const& spool : radioAudit.at("sources"))
            if (!spool.at("worker").is_null() && spool.at("worker").get<unsigned>() == i)
              workerOptions.insert(
                  workerOptions.end(),
                  {"--radio-track-in", spool.at("path").get<std::string>()});
          workers.push_back(
              launch("worker_" + std::to_string(i), workerOptions, workerEnv));
          parts.push_back(root / ("worker_" + std::to_string(i)) / "shower");
        }
        p.wait(workers, o.timeout_);
        // Per-process times are recorded at exit, before serial output validation.
        for (unsigned i = 0; i < devices.size(); ++i)
          if (scheduledWorkers.count(i)) {
            auto folder = root / ("worker_" + std::to_string(i));
            validate_outputs(folder / "shower");
            check_frontier_feed(folder, audit["counts"][i].get<std::size_t>());
          }
        require(digest(frontier) == audit["frontier_sha256"],
                "Frontier changed while running");
        for (auto const& spool : radioAudit.at("sources"))
          require(digest(spool.at("path").get<std::string>()) == spool.at("sha256"),
                  "Prefix radio spool changed while running");
        for (unsigned i = 0; i < shards.size(); ++i)
          require(digest(shards[i]) == audit["shard_sha256"][i],
                  "Shard changed while running");
        auto mergeStart = clock_type::now();
        auto products = merge(parts, root / "merged");
        fs::create_directory(root / "merged/primary");
        for (auto name : {"config.yaml", "summary.yaml"})
          fs::copy_file(primaryPart / "primary" / name, root / "merged/primary" / name);
        save(root / "merged/config.yaml",
             {{"creator", "CORSIKA8 native multi-GPU"},
              {"physics_args", physics},
              {"coordinator_config", (root / "CONFIG.json").string()},
              {"merge", (root / "merged/MERGE.json").string()}});
        save(root / "merged/summary.yaml",
             {{"showers", 1},
              {"seed", seed},
              {"runtime_seconds", seconds(p.started_)},
              {"experimental", true},
              {"statistical_acceptance", false},
              {"output_dirs",
               {"profile", "production_profile", "energyloss", "particles",
                "interactions", "primary", "CoREAS", "ZHS"}}});
        check_interrupt();
        auto wall = seconds(p.started_);
        save(root / "TIMING.json", {{"wall_s", wall},
                                    {"merge_s", seconds(mergeStart)},
                                    {"prefix_phase_wall_s", prefixPhase},
                                    {"cpu_prefix_workers", o.prefixWorkers_},
                                    {"parts", p.records()}});
        save(root / "COMPLETE.json",
             {{"products", products},
              {"wall_s", wall},
              {"partition_sha256", digest(root / "PARTITION.json")},
              {"process_complete", true},
              {"physics_validated_against_beta2", false},
              {"experimental", true},
              {"coordinator", "C++"}});
        CORSIKA_LOG_INFO("Seed {}: complete in {} s; {}", seed, wall,
                         (root / "merged").string());
      } catch (std::exception const& e) {
        save(root / "INCOMPLETE.json", {{"error", e.what()}, {"parts", p.records()}});
        throw;
      }
    }
  } // namespace
  void check_interrupt() {
    if (interrupted)
      throw std::runtime_error("Interrupted by signal " + std::to_string(interrupted));
  }
  void install_signal_handlers() {
    interrupted = 0;
    struct sigaction action {};
    action.sa_handler = signal_handler;
    sigemptyset(&action.sa_mask);
    sigaction(SIGINT, &action, nullptr);
    sigaction(SIGTERM, &action, nullptr);
  }
  void run(CoordinatorOptions const& options) {
    auto o = options;
    o.output_ = fs::absolute(o.output_).lexically_normal();
    o.worker_ = fs::absolute(o.worker_).lexically_normal();
    require(!o.devices_.empty() && o.devices_.size() <= 255,
            "Select 1..255 distinct GPUs");
    require(
        o.prefixWorkers_ + o.devices_.size() <= 255 &&
            (o.prefixWorkers_ == 0 || o.emBackend_ == "kokkos-proposal"),
        "CPU prefix workers require PROPOSAL and disjoint namespaces (CPU + GPU <=255)");
    require(std::isfinite(o.prefixSplitEnergy_) && o.prefixSplitEnergy_ >= 0,
            "Invalid CPU splitting energy");
    require(std::isfinite(o.energy_) && o.energy_ > 0,
            "Positive fixed primary energy required");
    require(o.seed_ > 0 && o.events_ > 0 &&
                o.seed_ <=
                    std::uint64_t(std::numeric_limits<std::int64_t>::max()) - o.events_,
            "Invalid seed/event count");
    require(std::isfinite(o.memoryFraction_) && o.memoryFraction_ > 0 &&
                o.memoryFraction_ <= .9,
            "Memory fraction must be in (0,0.9]");
    require(std::isfinite(o.frontierEnergy_) && o.frontierEnergy_ >= 0 &&
                std::isfinite(o.timeout_) && o.timeout_ > 0,
            "Invalid scheduling cap/timeout");
    require(access(o.worker_.c_str(), X_OK) == 0,
            "Worker executable unavailable: " + o.worker_.string());
    require(!fs::exists(o.output_), "Output directory exists: " + o.output_.string());
    if (o.output_.has_parent_path()) fs::create_directories(o.output_.parent_path());
    require(fs::create_directory(o.output_), "Output directory exists");
    try {
      auto devices = resolve_devices(o.devices_, o.output_);
      save(o.output_ / "DEVICE_UUIDS.json", devices);
      auto start = clock_type::now();
      if (o.events_ == 1)
        event(o, o.output_, o.seed_, devices);
      else {
        json_type events = json_type::array();
        save(o.output_ / "CAMPAIGN.json", {{"events", o.events_},
                                           {"first_seed", o.seed_},
                                           {"seed_rule", "seed + event_index"},
                                           {"devices", devices},
                                           {"coordinator", "C++"}});
        for (unsigned i = 0; i < o.events_; ++i) {
          auto seed = o.seed_ + i;
          auto folder = o.output_ / ("seed_" + std::to_string(seed));
          require(fs::create_directory(folder), "Event directory exists");
          event(o, folder, seed, devices);
          events.push_back({{"seed", seed},
                            {"wall_s", read_json(folder / "COMPLETE.json")["wall_s"]},
                            {"output", folder.string()}});
        }
        check_interrupt();
        save(o.output_ / "TIMING.json", {{"wall_s", seconds(start)}, {"events", events}});
        save(o.output_ / "COMPLETE.json", {{"process_complete", true},
                                           {"events", o.events_},
                                           {"wall_s", seconds(start)},
                                           {"coordinator", "C++"},
                                           {"physics_validated_against_beta2", false}});
      }
    } catch (std::exception const& e) {
      if (!fs::exists(o.output_ / "INCOMPLETE.json"))
        save(o.output_ / "INCOMPLETE.json", {{"error", e.what()}});
      throw;
    }
  }
  bool requested(int argc, char** argv) {
    for (int i = 1; i < argc; ++i)
      if (std::string(argv[i]) == "--prefix-workers" ||
          std::string(argv[i]).rfind("--prefix-workers=", 0) == 0)
        return true;
    return corsika::kokkos::application::parse_devices(argc, argv).hasMultipleDevices();
  }
  int main_entry(int argc, char** argv) {
    CoordinatorOptions o;
    std::string output, worker, emBackend, radioBackend, execution;
    CLI::App app{
        "CORSIKA 8: each shower on independent CUDA GPUs; native C++ coordination."};
    app.allow_extras();
    app.add_option("--device", "Physical GPU indices or UUIDs (comma or space separated)")
        ->expected(1, 255)
        ->type_size(1)
        ->delimiter(',');
    app.add_option("--devices", "Compatibility alias for --device")
        ->expected(1, 255)
        ->type_size(1)
        ->delimiter(',')
        ->group("");
    app.add_option("-f,--filename", output, "New output directory")->required();
    app.add_option("-E,--energy", o.energy_, "Fixed primary energy [GeV]")->required();
    app.add_option("-s,--seed", o.seed_, "First event seed; later seeds increment by one")
        ->default_val(1);
    app.add_option("-N,--nevent", o.events_,
                   "Sequential showers, each using all selected GPUs")
        ->default_val(1);
    app.add_option("--gpu-memory-fraction", o.memoryFraction_,
                   "Memory budget per GPU (0,0.9]")
        ->default_val(.5);
    app.add_option("--multigpu-timeout", o.timeout_, "Timeout seconds per prefix/worker")
        ->default_val(7200);
    app.add_option("--multigpu-frontier-energy", o.frontierEnergy_,
                   "Scheduling cap [GeV]; 0=automatic")
        ->default_val(0);
    app.add_option("--prefix-workers", o.prefixWorkers_,
                   "Static CPU prefix processes; 0 retains the original serial prefix")
        ->default_val(0);
    app.add_option("--prefix-split-energy", o.prefixSplitEnergy_,
                   "Static CPU split energy cap [GeV]; 0 uses the EM handoff cap")
        ->default_val(0);
    app.add_option("--multigpu-worker", worker,
                   "Optional compatible worker binary; default: this binary");
    app.add_option("--em-backend", emBackend,
                   "Multi-GPU transport backend: kokkos-proposal or kokkos-egs4");
    app.add_option("--egs4-stepfc", o.egs4Stepfc_, "C7-EGS4 step factor (default: 1)")
        ->check(CLI::PositiveNumber);
    app.add_option("--radio-backend", radioBackend, "Multi-GPU radio backend: kokkos");
    app.add_option("--kokkos-execution", execution, "Multi-GPU execution space: cuda");
    try {
      o.devices_ = corsika::kokkos::application::parse_devices(argc, argv).ids_;
      app.parse(argc, argv);
      o.output_ = output;
      require(o.devices_.size() > 1 || (o.devices_.size() == 1 && o.prefixWorkers_ > 0),
              "Select multiple GPUs or enable static CPU prefix workers");
      require((emBackend.empty() || emBackend == "kokkos-proposal" ||
               emBackend == "kokkos-egs4") &&
                  (radioBackend.empty() || radioBackend == "kokkos") &&
                  (execution.empty() || execution == "cuda"),
              "Multi-GPU requires kokkos-proposal/kokkos-egs4 transport, kokkos radio "
              "and cuda execution");
      o.emBackend_ = emBackend.empty() ? "kokkos-proposal" : emBackend;
#ifndef CORSIKA8_WITH_NATIVE_EGS4
      require(o.emBackend_ != "kokkos-egs4",
              "EGS4 is not compiled; build with CORSIKA_ENABLE_EGS4=ON");
#endif
      require(!app.count("--egs4-stepfc") || o.emBackend_ == "kokkos-egs4",
              "--egs4-stepfc requires --em-backend kokkos-egs4");
      require(o.emBackend_ != "kokkos-egs4" || !app.count("--multigpu-frontier-energy"),
              "EGS4 exports unstarted roots; --multigpu-frontier-energy applies only to "
              "PROPOSAL");
      o.worker_ = worker.empty() ? fs::canonical("/proc/self/exe") : fs::absolute(worker);
      o.physics_ = app.remaining();
      std::set<std::string> reserved{"--energy_range",
                                     "--force-interaction",
                                     "--force-decay",
                                     "--em-backend",
                                     "--radio-backend",
                                     "--kokkos-execution",
                                     "--kokkos-device",
                                     "--kokkos-num-threads",
                                     "--hadronic-workers",
                                     "--hadronic-backend",
                                     "--gpu-resident-batch-limit",
                                     "--compress"};
      for (std::size_t i = 0; i < o.physics_.size(); ++i) {
        auto key = o.physics_[i].substr(0, o.physics_[i].find('='));
        require(!reserved.count(key) && key.rfind("--frontier-", 0) != 0 &&
                    key.rfind("--prefix-", 0) != 0 &&
                    key.rfind("--radio-track-", 0) != 0 && key != "--",
                "Multi-GPU coordinator owns or does not support option: " + key);
        if (key == "--antenna-file" || key == "--gpu-aux-cache-dir" ||
            key == "--kokkos-tuning-cache") {
          auto eq = o.physics_[i].find('=');
          if (eq == std::string::npos) {
            require(i + 1 < o.physics_.size(), key + " needs a path");
            ++i;
            o.physics_[i] = fs::absolute(o.physics_[i]).string();
          } else
            o.physics_[i] =
                key + "=" + fs::absolute(o.physics_[i].substr(eq + 1)).string();
        }
      }
      if (auto data = std::getenv("CORSIKA_DATA"))
        setenv("CORSIKA_DATA", fs::absolute(data).c_str(), 1);
      install_signal_handlers();
      run(o);
      return 0;
    } catch (CLI::ParseError const& e) {
      return app.exit(e);
    } catch (std::exception const& e) {
      CORSIKA_LOG_ERROR("Multi-GPU: {}", e.what());
      return interrupted ? 128 + interrupted : 1;
    }
  }
} // namespace corsika::kokkos::coordinator

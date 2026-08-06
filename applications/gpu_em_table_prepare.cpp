/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include <CLI/CLI.hpp>

#include <PROPOSAL/version.h>

#include <corsika/gpu/em/tables/MediumConfig.hpp>
#include <corsika/gpu/em/tables/ProposalMedium.hpp>
#include <corsika/gpu/em/tables/RateTable.hpp>

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <fcntl.h>
#include <sys/file.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

namespace {

  using namespace corsika::gpu::em::tables;

  inline constexpr std::uint32_t PreparationSchemaVersion = 2;

  struct Options {
    std::filesystem::path medium_yaml;
    std::filesystem::path cache_dir;
    std::filesystem::path tablegen;
    double primary_energy_eV{};
    double energy_max_MeV{};
    double energy_margin{1.05};
    double energy_min_MeV{};
    double em_cut_MeV{0.5};
    double electron_transport_cut_MeV{};
    double muon_transport_cut_MeV{300.};
    double tolerance{1.e-3};
    double loss_tolerance{1.e-3};
    std::size_t initial_intervals{16};
    std::size_t max_points{20000};
    std::size_t loss_initial_energy_intervals{8};
    std::size_t loss_initial_quantile_intervals{8};
    std::size_t loss_max_energy_points{4096};
    std::size_t direct_loss_max_energy_points{65536};
    std::string nonmonotonic_loss_policy{"proposal-monotone"};
    std::size_t loss_max_quantile_points{2048};
    std::size_t loss_validation_samples{64};
    int lock_timeout_seconds{7200};
    bool no_muons{false};
    bool enable_epair_rho_table{false};
    bool lookup_only{false};
    bool dry_run{false};
    bool force{false};
    bool print_path_only{false};
  };

  struct Request {
    double energy_min_MeV{};
    double energy_max_MeV{};
    // User-facing CORSIKA ParticleCut/production threshold.
    double em_cut_MeV{};
    // Effective stochastic cut selected by the scalar PROPOSAL cache policy.
    double proposal_cut_MeV{};
    double electron_transport_cut_MeV{};
    double muon_transport_cut_MeV{};
    double tolerance{};
    double loss_tolerance{};
    std::size_t initial_intervals{};
    std::size_t max_points{};
    std::size_t loss_initial_energy_intervals{};
    std::size_t loss_initial_quantile_intervals{};
    std::size_t loss_max_energy_points{};
    std::size_t direct_loss_max_energy_points{};
    std::string nonmonotonic_loss_policy;
    std::size_t loss_max_quantile_points{};
    std::size_t loss_validation_samples{};
    bool include_muons{};
    bool enable_epair_rho_table{};
  };

  struct Candidate {
    std::filesystem::path path;
    RateTableSet table;
  };

  struct CliExit {
    int code{};
  };

  class FileLock {
  public:
    FileLock(std::filesystem::path const& path, int timeout_seconds) {
      descriptor_ = ::open(path.c_str(), O_CREAT | O_RDWR, 0600);
      if (descriptor_ < 0) {
        throw std::runtime_error(
            "cannot open preparation lock " + path.string() + ": " +
            std::strerror(errno));
      }
      auto const deadline = std::chrono::steady_clock::now() +
                            std::chrono::seconds(timeout_seconds);
      while (true) {
        if (::flock(descriptor_, LOCK_EX | LOCK_NB) == 0) {
          break;
        }
        if (errno != EWOULDBLOCK && errno != EAGAIN && errno != EINTR) {
          auto const message =
              "cannot acquire preparation lock " + path.string() + ": " +
              std::strerror(errno);
          ::close(descriptor_);
          descriptor_ = -1;
          throw std::runtime_error(message);
        }
        if (std::chrono::steady_clock::now() >= deadline) {
          ::close(descriptor_);
          descriptor_ = -1;
          throw std::runtime_error(
              "timed out waiting for GPU table preparation lock");
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
      }
    }

    FileLock(FileLock const&) = delete;
    FileLock& operator=(FileLock const&) = delete;

    ~FileLock() {
      if (descriptor_ >= 0) {
        ::flock(descriptor_, LOCK_UN);
        ::close(descriptor_);
      }
    }

  private:
    int descriptor_{-1};
  };

  std::string canonicalNumber(double value) {
    if (value == 0.) {
      value = 0.;
    }
    std::ostringstream stream;
    stream << std::scientific
           << std::setprecision(std::numeric_limits<double>::max_digits10)
           << value;
    return stream.str();
  }

  std::filesystem::path defaultCacheDirectory() {
    if (auto const* xdg = std::getenv("XDG_CACHE_HOME");
        xdg != nullptr && *xdg != '\0') {
      return std::filesystem::path(xdg) / "corsika8" / "gpu_em_tables";
    }
    if (auto const* user_home = std::getenv("HOME");
        user_home != nullptr && *user_home != '\0') {
      return std::filesystem::path(user_home) / ".cache" / "corsika8" /
             "gpu_em_tables";
    }
    return std::filesystem::current_path() / ".corsika8_gpu_em_tables";
  }

  std::filesystem::path defaultTablegen(char const* executable) {
    std::error_code error;
    auto const absolute =
        std::filesystem::absolute(std::filesystem::path(executable), error);
    if (!error) {
      auto const sibling = absolute.parent_path() / "gpu_em_tablegen";
      if (std::filesystem::is_regular_file(sibling)) {
        return sibling;
      }
    }
    return "gpu_em_tablegen";
  }

  Options parseOptions(int argc, char** argv) {
    Options options;
    CLI::App app{
        "Resolve or atomically generate a content-addressed CORSIKA 8 CUDA "
        "physics table from a normalized material YAML."};
    app.add_option("--medium-yaml", options.medium_yaml,
                   "Schema-v1 material YAML")
        ->required();
    app.add_option("--cache-dir", options.cache_dir,
                   "Content-addressed cache root");
    app.add_option("--tablegen", options.tablegen,
                   "gpu_em_tablegen executable; defaults to the sibling binary");
    app.add_option("--primary-energy-eV", options.primary_energy_eV,
                   "Primary total energy in eV; table upper bound is energy times margin");
    app.add_option("--energy-max-MeV", options.energy_max_MeV,
                   "Explicit table upper energy in MeV, mutually exclusive with primary energy");
    app.add_option("--energy-margin", options.energy_margin,
                   "Safety factor applied to --primary-energy-eV")
        ->check(CLI::Range(1.0, 10.0));
    app.add_option("--energy-min-MeV", options.energy_min_MeV,
                   "Table lower energy; zero selects --em-cut-MeV");
    app.add_option(
        "--em-cut-MeV", options.em_cut_MeV,
        "CORSIKA EM production/transport cut in MeV; the matching scalar "
        "PROPOSAL stochastic-table cut is resolved automatically");
    app.add_option("--electron-transport-cut-MeV",
                   options.electron_transport_cut_MeV,
                   "e-/e+ transport cut; zero selects --em-cut-MeV");
    app.add_option("--muon-transport-cut-MeV",
                   options.muon_transport_cut_MeV,
                   "mu-/mu+ transport cut in MeV");
    app.add_option("--tolerance", options.tolerance,
                   "Maximum rate interpolation error");
    app.add_option("--loss-tolerance", options.loss_tolerance,
                   "Maximum inverse-CDF interpolation error");
    app.add_option("--initial-intervals", options.initial_intervals);
    app.add_option("--max-points", options.max_points);
    app.add_option("--loss-initial-energy-intervals",
                   options.loss_initial_energy_intervals);
    app.add_option("--loss-initial-quantile-intervals",
                   options.loss_initial_quantile_intervals);
    app.add_option("--loss-max-energy-points",
                   options.loss_max_energy_points);
    app.add_option("--direct-loss-max-energy-points",
                   options.direct_loss_max_energy_points,
                   "Energy-grid budget for PROPOSAL-direct repair columns");
    app.add_option(
           "--nonmonotonic-loss-policy",
           options.nonmonotonic_loss_policy,
           "Repair policy: proposal-monotone or proposal-direct")
        ->check(CLI::IsMember(
            {"proposal-monotone", "proposal-direct"}));
    app.add_option("--loss-max-quantile-points",
                   options.loss_max_quantile_points);
    app.add_option("--loss-validation-samples",
                   options.loss_validation_samples);
    app.add_option("--lock-timeout-seconds", options.lock_timeout_seconds,
                   "Maximum wait for another process preparing the same request");
    app.add_flag("--no-muons", options.no_muons,
                 "Generate/accept only the EM contract");
    app.add_flag("--enable-epair-rho-table",
                 options.enable_epair_rho_table,
                 "Forward the experimental dense Epair rho request");
    app.add_flag("--lookup-only", options.lookup_only,
                 "Fail with exit code 2 instead of generating on a cache miss");
    app.add_flag("--dry-run", options.dry_run,
                 "Resolve and print the request without writing or generating");
    app.add_flag("--force", options.force,
                 "Regenerate the exact content-addressed request");
    app.add_flag("--print-path-only", options.print_path_only,
                 "Write only the resolved table path to stdout");
    try {
      app.parse(argc, argv);
    } catch (CLI::ParseError const& error) {
      throw CliExit{app.exit(error)};
    }

    auto const primary_supplied = options.primary_energy_eV != 0.;
    auto const maximum_supplied = options.energy_max_MeV != 0.;
    if (primary_supplied == maximum_supplied) {
      throw std::invalid_argument(
          "specify exactly one of --primary-energy-eV and --energy-max-MeV");
    }
    auto requirePositiveFinite = [](double value, char const* name) {
      if (!std::isfinite(value) || !(value > 0.)) {
        throw std::invalid_argument(
            std::string(name) + " must be finite and positive");
      }
    };
    if (primary_supplied) {
      requirePositiveFinite(options.primary_energy_eV,
                            "--primary-energy-eV");
    } else {
      requirePositiveFinite(options.energy_max_MeV, "--energy-max-MeV");
    }
    requirePositiveFinite(options.em_cut_MeV, "--em-cut-MeV");
    requirePositiveFinite(options.muon_transport_cut_MeV,
                          "--muon-transport-cut-MeV");
    requirePositiveFinite(options.tolerance, "--tolerance");
    requirePositiveFinite(options.loss_tolerance, "--loss-tolerance");
    if (options.tolerance > 1. || options.loss_tolerance > 1.) {
      throw std::invalid_argument("table tolerances must not exceed one");
    }
    if (options.energy_min_MeV < 0. ||
        options.electron_transport_cut_MeV < 0.) {
      throw std::invalid_argument(
          "energy minimum and electron transport cut must be non-negative");
    }
    if (options.lock_timeout_seconds < 0) {
      throw std::invalid_argument(
          "--lock-timeout-seconds must be non-negative");
    }
    if (options.force && (options.lookup_only || options.dry_run)) {
      throw std::invalid_argument(
          "--force cannot be combined with --lookup-only or --dry-run");
    }
    for (auto const value :
         {options.initial_intervals, options.max_points,
          options.loss_initial_energy_intervals,
          options.loss_initial_quantile_intervals,
          options.loss_max_energy_points,
          options.direct_loss_max_energy_points,
          options.loss_max_quantile_points,
          options.loss_validation_samples}) {
      if (value == 0) {
        throw std::invalid_argument(
            "adaptive table point and sample counts must be positive");
      }
    }
    if (options.cache_dir.empty()) {
      options.cache_dir = defaultCacheDirectory();
    }
    if (options.tablegen.empty()) {
      options.tablegen = defaultTablegen(argv[0]);
    }
    return options;
  }

  Request makeRequest(Options const& options) {
    auto optimizedProposalCutMeV = [](double const requested) {
      // Numeric mirror of proposal::energycut_table_values.  Keep this tool
      // independent of the full application stack while preserving the exact
      // scalar table-selection contract.
      constexpr double standard_cuts_MeV[]{
          1000., 100., 20., 10., 3., 1., 0.4, 0.25, 0.15, 0.05};
      double resolved = 0.;
      for (auto const candidate : standard_cuts_MeV) {
        if (candidate <= requested && candidate > resolved) {
          resolved = candidate;
        }
      }
      return resolved == 0. ? requested : resolved;
    };
    Request request;
    request.em_cut_MeV = options.em_cut_MeV;
    request.proposal_cut_MeV =
        optimizedProposalCutMeV(options.em_cut_MeV);
    request.energy_min_MeV =
        options.energy_min_MeV == 0.
            ? std::min(
                  request.em_cut_MeV,
                  request.proposal_cut_MeV)
            : options.energy_min_MeV;
    request.energy_max_MeV =
        options.energy_max_MeV != 0.
            ? options.energy_max_MeV
            : options.primary_energy_eV / 1.e6 * options.energy_margin;
    request.electron_transport_cut_MeV =
        options.electron_transport_cut_MeV == 0.
            ? options.em_cut_MeV
            : options.electron_transport_cut_MeV;
    request.muon_transport_cut_MeV =
        options.muon_transport_cut_MeV;
    request.tolerance = options.tolerance;
    request.loss_tolerance = options.loss_tolerance;
    request.initial_intervals = options.initial_intervals;
    request.max_points = options.max_points;
    request.loss_initial_energy_intervals =
        options.loss_initial_energy_intervals;
    request.loss_initial_quantile_intervals =
        options.loss_initial_quantile_intervals;
    request.loss_max_energy_points = options.loss_max_energy_points;
    request.direct_loss_max_energy_points =
        options.direct_loss_max_energy_points;
    request.nonmonotonic_loss_policy =
        options.nonmonotonic_loss_policy;
    request.loss_max_quantile_points = options.loss_max_quantile_points;
    request.loss_validation_samples = options.loss_validation_samples;
    request.include_muons = !options.no_muons;
    request.enable_epair_rho_table =
        options.enable_epair_rho_table;
    if (request.energy_max_MeV > MaximumGeneratedTableEnergyMeV) {
      throw std::invalid_argument(
          "resolved table upper energy " +
          canonicalNumber(request.energy_max_MeV) +
          " MeV exceeds the validated generator limit " +
          canonicalNumber(MaximumGeneratedTableEnergyMeV) +
          " MeV (1e20 eV)");
    }
    if (!(request.energy_min_MeV > 0.) ||
        !(request.energy_min_MeV <= request.proposal_cut_MeV) ||
        !(request.energy_max_MeV > request.energy_min_MeV) ||
        !std::isfinite(request.energy_max_MeV)) {
      throw std::invalid_argument(
          "resolved table energy domain is invalid");
    }
    return request;
  }

  std::string canonicalRequest(std::string const& medium_hash,
                               Request const& request) {
    std::ostringstream stream;
    stream
        << "preparation_schema=" << PreparationSchemaVersion << '\n'
        << "medium_sha256=" << medium_hash << '\n'
        << "proposal_version=" << getPROPOSALVersion() << '\n'
        << "table_format=" << RateTableFormatVersion << '\n'
        << "generator_contract=" << TableGeneratorContractVersion << '\n'
        << "energy_min_MeV=" << canonicalNumber(request.energy_min_MeV) << '\n'
        << "energy_max_MeV=" << canonicalNumber(request.energy_max_MeV) << '\n'
        << "em_cut_MeV=" << canonicalNumber(request.em_cut_MeV) << '\n'
        << "proposal_cut_MeV="
        << canonicalNumber(request.proposal_cut_MeV) << '\n'
        << "proposal_relative_v_cut="
        << canonicalNumber(ProposalRelativeVCut) << '\n'
        << "electron_transport_cut_MeV="
        << canonicalNumber(request.electron_transport_cut_MeV) << '\n'
        << "muon_transport_cut_MeV="
        << canonicalNumber(request.muon_transport_cut_MeV) << '\n'
        << "tolerance=" << canonicalNumber(request.tolerance) << '\n'
        << "loss_tolerance=" << canonicalNumber(request.loss_tolerance) << '\n'
        << "initial_intervals=" << request.initial_intervals << '\n'
        << "max_points=" << request.max_points << '\n'
        << "loss_initial_energy_intervals="
        << request.loss_initial_energy_intervals << '\n'
        << "loss_initial_quantile_intervals="
        << request.loss_initial_quantile_intervals << '\n'
        << "loss_max_energy_points=" << request.loss_max_energy_points << '\n'
        << "direct_loss_max_energy_points="
        << request.direct_loss_max_energy_points << '\n'
        << "nonmonotonic_loss_policy="
        << request.nonmonotonic_loss_policy << '\n'
        << "loss_max_quantile_points=" << request.loss_max_quantile_points
        << '\n'
        << "loss_validation_samples=" << request.loss_validation_samples
        << '\n'
        << "include_muons=" << (request.include_muons ? 1 : 0) << '\n'
        << "enable_epair_rho_table="
        << (request.enable_epair_rho_table ? 1 : 0) << '\n';
    return stream.str();
  }

  std::string requestHash(std::string const& canonical) {
    return toHex(sha256(
        reinterpret_cast<std::uint8_t const*>(canonical.data()),
        canonical.size()));
  }

  bool nearlyEqual(double left, double right, double relative = 2.e-6) {
    auto const scale = std::max({1., std::abs(left), std::abs(right)});
    return std::abs(left - right) <= relative * scale;
  }

  bool hasParticle(RateTableSet const& table, std::int32_t pdg) {
    return std::any_of(
        table.particles.begin(), table.particles.end(),
        [pdg](auto const& particle) { return particle.pdg_id == pdg; });
  }

  bool compatibleTable(RateTableSet const& table,
                       MediumConfig const& medium_config,
                       PROPOSAL::Medium const& medium,
                       Request const& request,
                       std::string* reason) {
    try {
      if (table.metadata.generator_version !=
          TableGeneratorContractVersion) {
        throw std::runtime_error(
            "table generator contract mismatch: cached=" +
            table.metadata.generator_version + ", required=" +
            TableGeneratorContractVersion);
      }
      validateCompatibility(
          table,
          RateTableRequirements{
              getPROPOSALVersion(), medium.GetName(),
              static_cast<std::uint64_t>(medium.GetHash()),
              request.proposal_cut_MeV, ProposalRelativeVCut,
              request.energy_min_MeV, request.energy_max_MeV,
              request.tolerance, request.loss_tolerance,
              makeRateTableMediumComponents(medium_config, medium)});
      for (auto const pdg : {22, 11, -11}) {
        if (!hasParticle(table, pdg)) {
          throw std::runtime_error(
              "table does not contain the complete EM particle set");
        }
      }
      for (auto const pdg : {11, -11}) {
        auto const& continuous = findContinuousEnergyTable(table, pdg);
        auto const kinetic_cut =
            continuous.minimum_total_energy_MeV - continuous.mass_MeV;
        if (!nearlyEqual(
                kinetic_cut,
                ContinuousCutSafetyFactor *
                    request.electron_transport_cut_MeV)) {
          throw std::runtime_error(
              "electron transport cut does not match");
        }
      }
      auto const has_muon_minus = hasParticle(table, 13);
      auto const has_muon_plus = hasParticle(table, -13);
      if (has_muon_minus != has_muon_plus) {
        throw std::runtime_error(
            "table contains only one muon charge state");
      }
      if (request.include_muons && !has_muon_minus) {
        throw std::runtime_error(
            "request requires muon tables");
      }
      if (request.include_muons) {
        for (auto const pdg : {13, -13}) {
          auto const& continuous = findContinuousEnergyTable(table, pdg);
          auto const kinetic_cut =
              continuous.minimum_total_energy_MeV - continuous.mass_MeV;
          if (!nearlyEqual(
                  kinetic_cut,
                  ContinuousCutSafetyFactor *
                      request.muon_transport_cut_MeV)) {
            throw std::runtime_error(
                "muon transport cut does not match");
          }
        }
      }
      if (request.enable_epair_rho_table && !table.epair_rho.enabled) {
        throw std::runtime_error(
            "request requires the dense Epair rho table");
      }
      return true;
    } catch (std::exception const& error) {
      if (reason != nullptr) {
        *reason = error.what();
      }
      return false;
    }
  }

  std::optional<Candidate> findCompatibleTable(
      std::filesystem::path const& directory,
      MediumConfig const& medium_config,
      PROPOSAL::Medium const& medium, Request const& request) {
    if (!std::filesystem::is_directory(directory)) {
      return std::nullopt;
    }
    std::optional<Candidate> best;
    std::error_code iterator_error;
    for (std::filesystem::directory_iterator iterator(directory,
                                                       iterator_error);
         !iterator_error &&
         iterator != std::filesystem::directory_iterator();
         iterator.increment(iterator_error)) {
      auto const& path = iterator->path();
      if (!iterator->is_regular_file() ||
          path.extension() != ".c8emrt") {
        continue;
      }
      try {
        auto table = readRateTable(path);
        std::string reason;
        if (!compatibleTable(table, medium_config, medium, request,
                             &reason)) {
          std::cerr << "Ignoring incompatible cached table " << path << ": "
                    << reason << '\n';
          continue;
        }
        if (!best ||
            table.metadata.energy_max_MeV <
                best->table.metadata.energy_max_MeV) {
          best = Candidate{path, std::move(table)};
        }
      } catch (std::exception const& error) {
        std::cerr << "Ignoring invalid cached table " << path << ": "
                  << error.what() << '\n';
      }
    }
    if (iterator_error) {
      throw std::runtime_error(
          "cannot scan GPU table cache: " + iterator_error.message());
    }
    return best;
  }

  int runProcess(std::vector<std::string> const& arguments,
                 bool redirect_stdout_to_stderr) {
    if (arguments.empty()) {
      throw std::invalid_argument("cannot execute an empty command");
    }
    auto const process = ::fork();
    if (process < 0) {
      throw std::runtime_error(
          "fork failed while launching gpu_em_tablegen");
    }
    if (process == 0) {
      if (redirect_stdout_to_stderr &&
          ::dup2(STDERR_FILENO, STDOUT_FILENO) < 0) {
        _exit(126);
      }
      std::vector<char*> argv;
      argv.reserve(arguments.size() + 1);
      for (auto const& argument : arguments) {
        argv.push_back(const_cast<char*>(argument.c_str()));
      }
      argv.push_back(nullptr);
      ::execvp(argv.front(), argv.data());
      std::cerr << "cannot execute " << arguments.front() << ": "
                << std::strerror(errno) << '\n';
      _exit(127);
    }
    int status = 0;
    while (::waitpid(process, &status, 0) < 0) {
      if (errno == EINTR) {
        continue;
      }
      throw std::runtime_error(
          "waitpid failed for gpu_em_tablegen");
    }
    if (WIFEXITED(status)) {
      return WEXITSTATUS(status);
    }
    if (WIFSIGNALED(status)) {
      throw std::runtime_error(
          "gpu_em_tablegen was terminated by signal " +
          std::to_string(WTERMSIG(status)));
    }
    throw std::runtime_error(
        "gpu_em_tablegen ended in an unknown process state");
  }

  std::vector<std::string> generatorCommand(
      Options const& options, Request const& request,
      std::filesystem::path const& output,
      std::filesystem::path const& proposal_cache) {
    auto number = [](double value) {
      std::ostringstream stream;
      stream << std::setprecision(std::numeric_limits<double>::max_digits10)
             << value;
      return stream.str();
    };
    std::vector<std::string> command{
        options.tablegen.string(),
        output.string(),
        "--medium-yaml",
        std::filesystem::absolute(options.medium_yaml).string(),
        "--proposal-cache",
        proposal_cache.string(),
        "--energy-min-MeV",
        number(request.energy_min_MeV),
        "--energy-max-MeV",
        number(request.energy_max_MeV),
        "--cut-MeV",
        number(request.proposal_cut_MeV),
        "--transport-cut-MeV",
        number(request.electron_transport_cut_MeV),
        "--muon-transport-cut-MeV",
        number(request.muon_transport_cut_MeV),
        "--tolerance",
        number(request.tolerance),
        "--loss-tolerance",
        number(request.loss_tolerance),
        "--initial-intervals",
        std::to_string(request.initial_intervals),
        "--max-points",
        std::to_string(request.max_points),
        "--loss-initial-energy-intervals",
        std::to_string(request.loss_initial_energy_intervals),
        "--loss-initial-quantile-intervals",
        std::to_string(request.loss_initial_quantile_intervals),
        "--loss-max-energy-points",
        std::to_string(request.loss_max_energy_points),
        "--direct-loss-max-energy-points",
        std::to_string(request.direct_loss_max_energy_points),
        "--nonmonotonic-loss-policy",
        request.nonmonotonic_loss_policy,
        "--loss-max-quantile-points",
        std::to_string(request.loss_max_quantile_points),
        "--loss-validation-samples",
        std::to_string(request.loss_validation_samples)};
    if (request.include_muons) {
      command.push_back("--include-muons");
    }
    if (request.enable_epair_rho_table) {
      command.push_back("--enable-epair-rho-table");
    }
    // A previous interrupted or rejected exact request may have left a
    // complete but incompatible file. We hold the request lock here, so
    // replacing that exact content-addressed path is deterministic and lets
    // the cache self-repair on the next invocation.
    if (std::filesystem::exists(output)) {
      command.push_back("--overwrite");
    }
    return command;
  }

  std::string utcTimestamp() {
    auto const now = std::chrono::system_clock::now();
    auto const time = std::chrono::system_clock::to_time_t(now);
    std::tm calendar{};
    gmtime_r(&time, &calendar);
    std::ostringstream stream;
    stream << std::put_time(&calendar, "%Y-%m-%dT%H:%M:%SZ");
    return stream.str();
  }

  void writeManifest(std::filesystem::path const& path,
                     Options const& options, Request const& request,
                     MediumConfig const& medium_config,
                     std::string const& medium_hash,
                     std::string const& request_hash,
                     std::filesystem::path const& table_path,
                     RateTableSet const& table) {
    YAML::Emitter output;
    output << YAML::BeginMap
           << YAML::Key << "schema_version" << YAML::Value
           << PreparationSchemaVersion
           << YAML::Key << "status" << YAML::Value << "complete"
           << YAML::Key << "created_utc" << YAML::Value << utcTimestamp()
           << YAML::Key << "medium" << YAML::Value << YAML::BeginMap
           << YAML::Key << "name" << YAML::Value << medium_config.name
           << YAML::Key << "source_yaml" << YAML::Value
           << std::filesystem::absolute(options.medium_yaml).string()
           << YAML::Key << "canonical_sha256" << YAML::Value << medium_hash
           << YAML::EndMap
           << YAML::Key << "request" << YAML::Value << YAML::BeginMap
           << YAML::Key << "sha256" << YAML::Value << request_hash
           << YAML::Key << "proposal_version" << YAML::Value
           << getPROPOSALVersion()
           << YAML::Key << "generator_contract" << YAML::Value
           << TableGeneratorContractVersion
           << YAML::Key << "table_format_version" << YAML::Value
           << RateTableFormatVersion
           << YAML::Key << "energy_min_MeV" << YAML::Value
           << request.energy_min_MeV
           << YAML::Key << "energy_max_MeV" << YAML::Value
           << request.energy_max_MeV
           << YAML::Key << "em_cut_MeV" << YAML::Value
           << request.em_cut_MeV
           << YAML::Key << "proposal_stochastic_cut_MeV" << YAML::Value
           << request.proposal_cut_MeV
           << YAML::Key << "proposal_relative_v_cut" << YAML::Value
           << ProposalRelativeVCut
           << YAML::Key << "electron_transport_cut_MeV" << YAML::Value
           << request.electron_transport_cut_MeV
           << YAML::Key << "muon_transport_cut_MeV" << YAML::Value
           << request.muon_transport_cut_MeV
           << YAML::Key << "tolerance" << YAML::Value
           << request.tolerance
           << YAML::Key << "loss_tolerance" << YAML::Value
           << request.loss_tolerance
           << YAML::Key << "initial_intervals" << YAML::Value
           << request.initial_intervals
           << YAML::Key << "max_points" << YAML::Value
           << request.max_points
           << YAML::Key << "loss_initial_energy_intervals" << YAML::Value
           << request.loss_initial_energy_intervals
           << YAML::Key << "loss_initial_quantile_intervals" << YAML::Value
           << request.loss_initial_quantile_intervals
           << YAML::Key << "loss_max_energy_points" << YAML::Value
           << request.loss_max_energy_points
           << YAML::Key << "direct_loss_max_energy_points" << YAML::Value
           << request.direct_loss_max_energy_points
           << YAML::Key << "nonmonotonic_loss_policy" << YAML::Value
           << request.nonmonotonic_loss_policy
           << YAML::Key << "loss_max_quantile_points" << YAML::Value
           << request.loss_max_quantile_points
           << YAML::Key << "loss_validation_samples" << YAML::Value
           << request.loss_validation_samples
           << YAML::Key << "include_muons" << YAML::Value
           << request.include_muons
           << YAML::Key << "enable_epair_rho_table" << YAML::Value
           << request.enable_epair_rho_table
           << YAML::EndMap
           << YAML::Key << "table" << YAML::Value << YAML::BeginMap
           << YAML::Key << "path" << YAML::Value
           << std::filesystem::absolute(table_path).string()
           << YAML::Key << "content_sha256" << YAML::Value
           << toHex(table.content_hash)
           << YAML::Key << "generator_version" << YAML::Value
           << table.metadata.generator_version
           << YAML::Key << "measured_rate_error" << YAML::Value
           << table.metadata.measured_max_relative_error
           << YAML::Key << "measured_loss_error" << YAML::Value
           << table.metadata.measured_max_loss_relative_error
           << YAML::EndMap
           << YAML::EndMap;
    if (!output.good()) {
      throw std::runtime_error(
          "cannot serialize GPU table preparation manifest");
    }
    auto temporary = path;
    temporary += ".tmp";
    try {
      std::ofstream stream(temporary, std::ios::trunc);
      if (!stream) {
        throw std::runtime_error(
            "cannot open temporary preparation manifest");
      }
      stream << output.c_str() << '\n';
      stream.close();
      if (!stream) {
        throw std::runtime_error(
            "failed while writing preparation manifest");
      }
      std::filesystem::rename(temporary, path);
    } catch (...) {
      std::error_code error;
      std::filesystem::remove(temporary, error);
      throw;
    }
  }

  void report(std::string const& status,
              std::filesystem::path const& table_path,
              std::string const& medium_hash,
              std::string const& request_hash,
              Request const& request,
              bool print_path_only) {
    if (print_path_only) {
      std::cout << std::filesystem::absolute(table_path).string() << '\n';
      return;
    }
    std::cout << "status=" << status << '\n'
              << "medium_sha256=" << medium_hash << '\n'
              << "request_sha256=" << request_hash << '\n'
              << "energy_min_MeV=" << std::setprecision(17)
              << request.energy_min_MeV << '\n'
              << "energy_max_MeV=" << std::setprecision(17)
              << request.energy_max_MeV << '\n'
              << "em_cut_MeV=" << request.em_cut_MeV << '\n'
              << "proposal_stochastic_cut_MeV="
              << request.proposal_cut_MeV << '\n'
              << "direct_loss_max_energy_points="
              << request.direct_loss_max_energy_points << '\n'
              << "nonmonotonic_loss_policy="
              << request.nonmonotonic_loss_policy << '\n'
              << "muon_transport="
              << (request.include_muons ? "enabled" : "disabled") << '\n'
              << "table="
              << std::filesystem::absolute(table_path).string() << '\n';
  }

} // namespace

int main(int argc, char** argv) {
  try {
    auto const options = parseOptions(argc, argv);
    auto const request = makeRequest(options);
    auto const medium_config = loadMediumConfig(options.medium_yaml);
    auto const medium_hash = mediumConfigHashHex(medium_config);
    auto const medium = makeProposalMedium(medium_config);
    auto const canonical_request =
        canonicalRequest(medium_hash, request);
    auto const request_hash = requestHash(canonical_request);

    auto const medium_directory =
        std::filesystem::absolute(options.cache_dir) / "tables" /
        medium_hash;
    auto const planned_table =
        medium_directory / ("table-" + request_hash + ".c8emrt");
    auto const manifest = planned_table.string() + ".manifest.yaml";

    if (!options.force) {
      auto candidate = findCompatibleTable(
          medium_directory, medium_config, medium, request);
      if (candidate) {
        report("cache_hit", candidate->path, medium_hash,
               request_hash, request, options.print_path_only);
        return EXIT_SUCCESS;
      }
    }

    if (options.dry_run) {
      report("would_generate", planned_table, medium_hash,
             request_hash, request, options.print_path_only);
      return EXIT_SUCCESS;
    }
    if (options.lookup_only) {
      if (!options.print_path_only) {
        report("cache_miss", planned_table, medium_hash,
               request_hash, request, false);
      }
      return 2;
    }

    std::filesystem::create_directories(medium_directory);
    std::filesystem::create_directories(
        std::filesystem::absolute(options.cache_dir) / "media");
    writeCanonicalMediumYaml(
        std::filesystem::absolute(options.cache_dir) / "media" /
            (medium_hash + ".yaml"),
        medium_config);

    FileLock lock(medium_directory / (request_hash + ".lock"),
                  options.lock_timeout_seconds);
    if (!options.force) {
      auto candidate = findCompatibleTable(
          medium_directory, medium_config, medium, request);
      if (candidate) {
        report("cache_hit_after_wait", candidate->path, medium_hash,
               request_hash, request, options.print_path_only);
        return EXIT_SUCCESS;
      }
    }

    auto const proposal_cache =
        std::filesystem::absolute(options.cache_dir) / "proposal" /
        medium_hash;
    std::filesystem::create_directories(proposal_cache);
    auto const command = generatorCommand(
        options, request, planned_table, proposal_cache);
    auto const generator_status =
        runProcess(command, options.print_path_only);
    if (generator_status != 0) {
      throw std::runtime_error(
          "gpu_em_tablegen failed with exit code " +
          std::to_string(generator_status));
    }

    auto table = readRateTable(planned_table);
    std::string incompatibility;
    if (!compatibleTable(table, medium_config, medium, request,
                         &incompatibility)) {
      throw std::runtime_error(
          "generated table failed preparation compatibility check: " +
          incompatibility);
    }
    writeManifest(manifest, options, request, medium_config,
                  medium_hash, request_hash, planned_table, table);
    report("generated", planned_table, medium_hash, request_hash,
           request, options.print_path_only);
    return EXIT_SUCCESS;
  } catch (CliExit const& exit) {
    return exit.code;
  } catch (std::exception const& error) {
    std::cerr << "gpu_em_table_prepare failed: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}

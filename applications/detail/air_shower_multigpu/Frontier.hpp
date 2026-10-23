/* Experimental static subshower handoff. No production-router changes. */
#pragma once
#include <corsika/accelerator/em/common/RouterParticleConversion.hpp>
#include <corsika/accelerator/em/common/EnvironmentSnapshotStorage.hpp>
#include <corsika/framework/core/HybridCascade.hpp>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>
#include <unordered_set>
#include <vector>

namespace corsika::applications::multigpu {
using State = gpu::em::EmParticleState;
using EnvironmentView = gpu::em::environment_tree::FullView;
constexpr std::uint64_t historyStride = std::uint64_t{1} << 48;
struct FrontierHeader {
  bool logical_nodes{};
  std::uint64_t environment_fingerprint{};
  std::uint64_t history_limit{historyStride};
  explicit FrontierHeader(std::string const& header) {
    if (header == "C8_STATIC_FRONTIER_V1") return;
    std::istringstream in(header);
    std::string version, extra;
    if (!(in >> version >> environment_fingerprint))
      throw std::runtime_error("Missing/unsupported frontier header");
    if (version == "C8_STATIC_FRONTIER_V3") {
      if (!(in >> history_limit) || history_limit <= historyStride ||
          history_limit > 255 * historyStride || history_limit % historyStride)
        throw std::runtime_error("Invalid frontier history limit");
    } else if (version != "C8_STATIC_FRONTIER_V2")
      throw std::runtime_error("Missing/unsupported frontier header");
    if (in >> extra) throw std::runtime_error("Trailing frontier header fields");
    logical_nodes = true;
  }
  void validateEnvironment(EnvironmentView const& tree) const {
    if (logical_nodes != (tree.topology.node_count != 0) ||
        (logical_nodes && environment_fingerprint !=
         gpu::em::environment_tree::snapshotFingerprint(tree)))
      throw std::runtime_error("Frontier environment mismatch; logical nodes cannot be reconstructed from position");
  }
};
inline bool routable(int pid) {
  return pid == 22 || pid == 11 || pid == -11 || pid == 13 || pid == -13;
}
inline void validate(State const& s, std::uint64_t history_limit = historyStride,
                     bool require_em = true) {
  double norm = 0.;
  for (int k = 0; k != 3; ++k) {
    if (!std::isfinite(s.position_m[k]) || !std::isfinite(s.direction[k]))
      throw std::runtime_error("Non-finite frontier coordinates");
    norm += s.direction[k] * s.direction[k];
  }
  if ((require_em && !routable(s.pid)) || !std::isfinite(s.energy_GeV) ||
      s.energy_GeV < get_mass(convert_from_PDG(static_cast<PDGCode>(s.pid))) / 1_GeV ||
      !std::isfinite(s.time_s) || !std::isfinite(s.weight) || s.weight < 0. ||
      std::abs(norm - 1.) > 1.e-9 || !s.history_id ||
      s.history_id >= history_limit || s.parent_history_id >= history_limit ||
      ((s.generation == 0) != (s.parent_history_id == 0)))
    throw std::runtime_error("Invalid frontier identity, direction, energy or weight");
}
// Explicit portable records, never pointers. The coordinator hashes the closed
// file before assigning every root exactly once. V2 retains the logical node ID.
inline void write(std::ostream& out, State const& s, bool logical_nodes = false,
                  std::uint64_t history_limit = historyStride) {
  validate(s, history_limit);
  out << s.pid << ' ' << s.history_id << ' ' << s.parent_history_id << ' '
      << s.generation << ' ' << s.step_id << ' ' << s.energy_GeV << ' '
      << s.weight << ' ' << s.time_s;
  for (auto x : s.position_m) out << ' ' << x;
  for (auto x : s.direction) out << ' ' << x;
  if (logical_nodes) {
    if (s.environment_node < 0) throw std::runtime_error("Missing frontier logical node");
    out << ' ' << s.environment_node;
  }
  out << '\n';
}

// Keep unstarted roots OUTSIDE the scalar stack. A router return must not make
// setNodes() re-locate a million unrelated, stationary roots. Read in bounded
// batches, preserve all physical/identity fields, and reverse only the insertion
// order so the LIFO scheduler consumes the file's largest-energy-first order.
class FrontierInput {
  std::ifstream file_;
  bool exhausted_{};
  std::unordered_set<std::uint64_t> seen_;
  std::size_t batches_{}, peak_batch_{};
  long double weighted_energy_GeV_{};
  FrontierHeader header_{"C8_STATIC_FRONTIER_V1"};
  EnvironmentView environment_{};
  std::uint64_t namespace_base_{};
public:
  explicit FrontierInput(std::string const& path) : file_(path) {
    std::string header;
    if (!std::getline(file_, header))
      throw std::runtime_error("Missing/unsupported frontier header: " + path);
    header_ = FrontierHeader(header);
  }
  void setEnvironmentTree(EnvironmentView const& tree) {
    header_.validateEnvironment(tree);
    environment_ = tree;
  }
  template<class Stack> void reserveNamespace(Stack& stack, unsigned worker) {
    if (worker < 1 || worker > 255) throw std::runtime_error("Worker ID outside [1,255]");
    auto const next = stack.reserveTransportHistoryIds(1);
    auto const base = worker * historyStride;
    if (base < header_.history_limit)
      throw std::runtime_error("Worker namespace overlaps imported CPU histories");
    namespace_base_ = base;
    if (next >= base) throw std::runtime_error("Worker history namespace collision");
    if (base - next > 1) stack.reserveTransportHistoryIds(base - next - 1);
  }
  bool pending() const { return !exhausted_; }
  std::size_t count() const { return seen_.size(); }
  std::size_t batches() const { return batches_; }
  std::size_t peakBatch() const { return peak_batch_; }
  double weightedEnergyGeV() const { return static_cast<double>(weighted_energy_GeV_); }
  template<class Stack>
  std::size_t refill(Stack& stack, CoordinateSystemPtr const& cs, std::size_t limit) {
    header_.validateEnvironment(environment_);
    if (!stack.isEmpty()) throw std::logic_error("Frontier refill requires a drained scalar stack");
    if (limit == 0 || limit > 65536) throw std::logic_error("Invalid bounded frontier batch");
    std::vector<State> batch;
    batch.reserve(limit);
    std::string line;
    while (batch.size() < limit && std::getline(file_, line)) {
      State s{};
      std::istringstream row(line);
      row >> s.pid >> s.history_id >> s.parent_history_id >> s.generation
          >> s.step_id >> s.energy_GeV >> s.weight >> s.time_s;
      for (auto& x : s.position_m) row >> x;
      for (auto& x : s.direction) row >> x;
      if (header_.logical_nodes) row >> s.environment_node;
      std::string trailing;
      if (!row || (row >> trailing)) throw std::runtime_error("Malformed frontier row");
      validate(s, header_.history_limit);
      if (namespace_base_ && (s.history_id >= namespace_base_ ||
                              s.parent_history_id >= namespace_base_))
        throw std::runtime_error("Imported root overlaps worker namespace");
      if (!seen_.insert(s.history_id).second) throw std::runtime_error("Duplicate frontier root");
      weighted_energy_GeV_ += static_cast<long double>(s.energy_GeV) * s.weight;
      if (!std::isfinite(weighted_energy_GeV_) ||
          weighted_energy_GeV_ > std::numeric_limits<double>::max())
        throw std::runtime_error("Imported frontier energy overflow");
      batch.push_back(s);
    }
    if (file_.eof()) exhausted_ = true;
    else if (!file_) throw std::runtime_error("Frontier read failed");
    for (auto it = batch.rbegin(); it != batch.rend(); ++it)
      gpu::em::router_detail::importParticle(stack, *it, cs, environment_);
    if (!batch.empty()) ++batches_;
    peak_batch_ = std::max(peak_batch_, batch.size());
    return batch.size();
  }
};

// Activated only for an imported frontier, including workers launched by the
// native --devices entry. Ordinary single-endpoint runs and mountain sessions
// do not activate it; common backend state remains unchanged.
inline thread_local FrontierInput* activeFrontierInput = nullptr;
inline thread_local std::string activeFrontierOutput;
struct ForwardRouterArguments {};
// Opt-in control: static import chunking is independent of the GPU
// dispatch threshold. No live queue is refilled until its local work drains.
inline std::size_t staticFrontierImportLimit(std::size_t original) {
  if (auto const* value = std::getenv("C8_STATIC_FRONTIER_IMPORT_BATCH")) {
    std::string text(value);
    if (text.empty() || text.size() > 5 ||
        !std::all_of(text.begin(), text.end(), [](char c) { return c >= '0' && c <= '9'; }))
      throw std::runtime_error("Invalid static frontier import batch");
    auto const count = std::stoul(text);
    if (count == 0 || count > 65536)
      throw std::runtime_error("Static frontier import batch outside [1,65536]");
    return count;
  }
  return std::min<std::size_t>(65536, std::max<std::size_t>(1, original));
}
class ScopedFrontierInput {
public:
  explicit ScopedFrontierInput(FrontierInput* input, std::string output = {}) {
    if (activeFrontierInput) throw std::logic_error("Nested frontier input");
    activeFrontierInput = input;
    activeFrontierOutput = std::move(output);
  }
  ~ScopedFrontierInput() { activeFrontierInput = nullptr; activeFrontierOutput.clear(); }
  ScopedFrontierInput(ScopedFrontierInput const&) = delete;
  ScopedFrontierInput& operator=(ScopedFrontierInput const&) = delete;
};

template<class BaseRouter>
class BufferedFrontierRouter : public BaseRouter {
  FrontierInput* source_;
  CoordinateSystemPtr cs_;
  std::size_t limit_;
public:
  void setFrontierEnvironment(EnvironmentView const& tree) {
    if (source_) source_->setEnvironmentTree(tree);
  }
  template<class... Args>
  BufferedFrontierRouter(ForwardRouterArguments, CoordinateSystemPtr const& cs,
                        std::size_t limit, Args&&... args)
      : BaseRouter(std::forward<Args>(args)...), source_(activeFrontierInput),
        cs_(cs), limit_(staticFrontierImportLimit(limit)) {}
  template<class Backend, class Env, class Fallback, class Output>
  BufferedFrontierRouter(Backend& backend, CoordinateSystemPtr const& cs,
                        Env const& env, Fallback& fallback, Output& output)
      : BaseRouter(backend, cs, env, fallback, output),
        source_(activeFrontierInput), cs_(cs),
        limit_(staticFrontierImportLimit(backend.minimumBatchSize())) {}
  bool pending() const {
    return BaseRouter::pending() || (source_ && source_->pending());
  }
  template<class Stack> std::size_t advanceOneWavefrontAndReturn(Stack& stack) {
    // Existing resident transport and rare-final-state handling drain locally.
    // Never mix new roots with an in-flight step or an unfinished fallback.
    if (BaseRouter::pending()) return BaseRouter::advanceOneWavefrontAndReturn(stack);
    if (source_ && source_->pending()) return source_->refill(stack, cs_, limit_);
    return 0;
  }
  void endOfShower() {
    if (source_ && source_->pending()) throw std::logic_error("Unconsumed frontier roots");
    BaseRouter::endOfShower();
    if (source_) {
      std::ofstream audit("FRONTIER_FEED.json");
      audit.exceptions(std::ios::badbit | std::ios::failbit);
      audit << "{\"mode\":\"bounded-high-energy-first\",\"roots_consumed\":"
            << source_->count() << ",\"batches\":" << source_->batches()
            << ",\"maximum_imported_batch\":" << source_->peakBatch()
            << ",\"weighted_energy_GeV\":" << std::setprecision(17) << source_->weightedEnergyGeV()
            << ",\"configured_import_batch\":" << limit_ << "}\n";
      CORSIKA_LOG_INFO("Bounded frontier drained: {} roots, {} batches, peak imported batch {}",
                      source_->count(), source_->batches(), source_->peakBatch());
    }
  }
};
class CaptureRouter {
  CoordinateSystemPtr cs_;
  double maximumEnergy_;
  std::ofstream file_;
  std::size_t count_{};
  EnvironmentView environment_{};
  std::uint64_t history_limit_{historyStride};
public:
  CaptureRouter(std::string const& path, CoordinateSystemPtr cs, double cap,
                EnvironmentView const& tree = {},
                std::uint64_t history_limit = historyStride)
      : cs_(cs), maximumEnergy_(cap), file_(path), environment_(tree),
        history_limit_(history_limit) {
    if (!file_) throw std::runtime_error("Cannot create frontier: " + path);
    if (history_limit != historyStride) {
      if (!tree.topology.node_count || history_limit <= historyStride ||
          history_limit > 255 * historyStride || history_limit % historyStride)
        throw std::runtime_error("Invalid CPU-prefix history namespace limit");
      file_ << "C8_STATIC_FRONTIER_V3 " << gpu::em::environment_tree::snapshotFingerprint(tree)
            << ' ' << history_limit << '\n';
    } else if (tree.topology.node_count)
      file_ << "C8_STATIC_FRONTIER_V2 " << gpu::em::environment_tree::snapshotFingerprint(tree) << '\n';
    else file_ << "C8_STATIC_FRONTIER_V1\n";
    file_ << std::setprecision(17);
    file_.exceptions(std::ios::badbit | std::ios::failbit);
  }
  template<class Particle, class Step>
  bool canRoute(Particle const& p, Step) const {
    return routable(static_cast<int>(p.getPDG())) && p.getEnergy() / 1_GeV <= maximumEnergy_;
  }
  template<class Particle, class H, class G, class S>
  void stage(Particle const& p, H history, H parent, G generation, S step) {
    // beginTransportStep increments storage but RETURNS the old (zero-based)
    // counter. No physical step happened: export the returned value unchanged.
    write(file_, gpu::em::router_detail::toDeviceState(
        p, cs_, history, parent, generation, step, environment_),
        environment_.topology.node_count != 0, history_limit_);
    ++count_;
  }
  bool pending() const { return false; }
  bool preserveLogicalNodes() const { return environment_.topology.node_count != 0; }
  template<class Stack> std::size_t advanceOneWavefrontAndReturn(Stack&) { return 0; }
  void endOfShower() {
    file_.flush();
    file_.close();
    CORSIKA_LOG_INFO("Static frontier closed: {} disjoint roots", count_);
  }
};
template<class Stack>
std::size_t importFrontier(Stack& stack, CoordinateSystemPtr cs,
                          std::string const& path, unsigned worker,
                          EnvironmentView const& tree = {}) {
  if (worker < 1 || worker > 255) throw std::runtime_error("Worker ID outside [1,255]");
  std::ifstream in(path);
  std::string line;
  if (!std::getline(in, line))
    throw std::runtime_error("Missing/unsupported frontier header: " + path);
  FrontierHeader const header(line);
  header.validateEnvironment(tree);
  std::set<std::uint64_t> seen;
  while (std::getline(in, line)) {
    State s{};
    std::istringstream row(line);
    row >> s.pid >> s.history_id >> s.parent_history_id >> s.generation
        >> s.step_id >> s.energy_GeV >> s.weight >> s.time_s;
    for (auto& x : s.position_m) row >> x;
    for (auto& x : s.direction) row >> x;
    if (header.logical_nodes) row >> s.environment_node;
    std::string trailing;
    if (!row || (row >> trailing)) throw std::runtime_error("Malformed frontier row");
    validate(s, header.history_limit);
    if (!seen.insert(s.history_id).second) throw std::runtime_error("Duplicate frontier root");
    gpu::em::router_detail::importParticle(stack, s, cs, tree);
  }
  if (!in.eof()) throw std::runtime_error("Frontier read failed");
  // Ancestors retain original IDs; children use an exclusive 48-bit range.
  auto const next = stack.reserveTransportHistoryIds(1);
  auto const base = worker * historyStride;
  if (base < header.history_limit) throw std::runtime_error("Worker namespace overlaps input");
  if (next >= base) throw std::runtime_error("Worker history namespace collision");
  if (base - next > 1) stack.reserveTransportHistoryIds(base - next - 1);
  CORSIKA_LOG_INFO("Worker {} imported {} roots; new histories start at {}", worker, seen.size(), base);
  return seen.size();
}
} // namespace corsika::applications::multigpu

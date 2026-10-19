// Opt-in static CPU subshower handoff. No changes to physics processes.
#pragma once
#include "Frontier.hpp"
#include <nlohmann/json.hpp>

namespace corsika::applications::multigpu {

// Capture only already-created secondary histories, before sampling their next
// transport step. The initial primary is never cloned or forced to interact.
class CpuBootstrapRouter {
  CoordinateSystemPtr cs_;
  EnvironmentView tree_;
  std::ofstream out_;
  std::size_t count_{};
  double maximum_energy_;
public:
  CpuBootstrapRouter(std::string const& path, CoordinateSystemPtr cs,
                     EnvironmentView const& tree,
                     double maximum_energy = std::numeric_limits<double>::infinity())
      : cs_(cs), tree_(tree), out_(path), maximum_energy_(maximum_energy) {
    if (!tree.topology.node_count || !out_)
      throw std::runtime_error("CPU bootstrap requires an exported environment and output");
    out_.exceptions(std::ios::badbit | std::ios::failbit);
    out_ << "C8_CPU_FRONTIER_V1 "
         << gpu::em::environment_tree::snapshotFingerprint(tree) << '\n'
         << std::setprecision(17);
  }
  template<class Particle, class Step>
  bool canRoute(Particle const& p, Step) const {
    return p.getGeneration() > 0 && p.getEnergy() / 1_GeV <= maximum_energy_;
  }
  template<class Particle, class H, class G, class S>
  void stage(Particle const& p, H history, H parent, G generation, S step) {
    auto s = gpu::em::router_detail::toDeviceState(p, cs_, history, parent, generation, step, tree_);
    validate(s, historyStride, false);
    if (!generation || s.environment_node < 0)
      throw std::runtime_error("CPU bootstrap must export secondaries with logical owners");
    // PDG encodes nuclear A/Z as well; retain kinetic energy separately to avoid
    // total-minus-rest cancellation when a CPU particle is reconstructed.
    out_ << s.pid << ' ' << s.history_id << ' ' << s.parent_history_id << ' '
         << s.generation << ' ' << s.step_id << ' ' << s.energy_GeV << ' '
         << s.weight << ' ' << s.time_s;
    for (auto x : s.position_m) out_ << ' ' << x;
    for (auto x : s.direction) out_ << ' ' << x;
    out_ << ' ' << s.environment_node << ' ' << p.getKineticEnergy() / 1_GeV << '\n';
    ++count_;
  }
  bool pending() const { return false; }
  bool preserveLogicalNodes() const { return true; }
  template<class Stack> std::size_t advanceOneWavefrontAndReturn(Stack&) { return 0; }
  void endOfShower() {
    out_.flush(); out_.close();
    CORSIKA_LOG_INFO("Static CPU bootstrap closed: {} disjoint roots", count_);
  }
};

template<class Stack>
std::size_t importCpuFrontier(Stack& stack, CoordinateSystemPtr cs,
                             std::string const& path, unsigned worker,
                             EnvironmentView const& tree) {
  if (!stack.isEmpty() || worker < 1 || worker > 254)
    throw std::runtime_error("Invalid CPU prefix stack or worker namespace");
  std::ifstream in(path);
  std::string line, version, extra;
  std::uint64_t fingerprint{};
  if (!std::getline(in, line)) throw std::runtime_error("Missing CPU frontier header");
  std::istringstream header(line);
  if (!(header >> version >> fingerprint) || version != "C8_CPU_FRONTIER_V1" ||
      (header >> extra) || !tree.topology.node_count ||
      fingerprint != gpu::em::environment_tree::snapshotFingerprint(tree))
    throw std::runtime_error("CPU frontier environment mismatch");
  struct Root { State state; double kinetic; };
  std::vector<Root> roots;
  std::unordered_set<std::uint64_t> seen;
  long double weighted = 0;
  while (std::getline(in, line)) {
    State s{}; double kinetic{};
    std::istringstream row(line);
    row >> s.pid >> s.history_id >> s.parent_history_id >> s.generation
        >> s.step_id >> s.energy_GeV >> s.weight >> s.time_s;
    for (auto& x : s.position_m) row >> x;
    for (auto& x : s.direction) row >> x;
    row >> s.environment_node >> kinetic;
    if (!row || (row >> extra)) throw std::runtime_error("Malformed CPU frontier row");
    validate(s, historyStride, false);
    auto mass = get_mass(convert_from_PDG(static_cast<PDGCode>(s.pid))) / 1_GeV;
    if (!s.generation || !std::isfinite(kinetic) || kinetic < 0 ||
        std::abs(kinetic + mass - s.energy_GeV) >
            8 * std::numeric_limits<double>::epsilon() * std::max(1., s.energy_GeV) ||
        !seen.insert(s.history_id).second)
      throw std::runtime_error("Invalid or duplicate CPU frontier root");
    roots.push_back({s, kinetic});
    weighted += static_cast<long double>(s.energy_GeV) * s.weight;
  }
  if (!in.eof()) throw std::runtime_error("CPU frontier read failed");
  // Input was sorted high-energy-first. Reverse insertion for the LIFO stack.
  for (auto it = roots.rbegin(); it != roots.rend(); ++it) {
    auto p = gpu::em::router_detail::importParticle(stack, it->state, cs, tree);
    p.setKineticEnergy(it->kinetic * 1_GeV);
  }
  auto next = stack.reserveTransportHistoryIds(1);
  auto base = worker * historyStride;
  if (next >= base) throw std::runtime_error("CPU prefix history namespace collision");
  if (base - next > 1) stack.reserveTransportHistoryIds(base - next - 1);
  nlohmann::json audit{{"roots_consumed", roots.size()}, {"worker_namespace", worker},
                       {"weighted_energy_GeV", double(weighted)},
                       {"environment_fingerprint", fingerprint}};
  std::ofstream report("CPU_FRONTIER_FEED.json");
  report.exceptions(std::ios::badbit | std::ios::failbit);
  report << audit.dump(2) << '\n';
  return roots.size();
}
} // namespace corsika::applications::multigpu

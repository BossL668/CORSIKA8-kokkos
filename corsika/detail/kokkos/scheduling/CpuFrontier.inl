/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */
#pragma once
namespace corsika::kokkos::frontier {
  template <typename TModelProperties>
  CpuBootstrapRouter<TModelProperties>::CpuBootstrapRouter(
      std::string const& path, CoordinateSystemPtr cs,
      EnvironmentSnapshot<TModelProperties> const& snapshot, double maximum_energy)
      : cs_(cs)
      , tree_(bind_environment_data(snapshot))
      , out_(path)
      , maximumEnergy_(maximum_energy)
      , topology_(snapshot.getTopology()) {
    auto const& tree = tree_;

    if (!tree.nodeCount_ || !out_)
      throw std::runtime_error(
          "CPU bootstrap requires an exported environment and output");
    out_.exceptions(std::ios::badbit | std::ios::failbit);
    out_ << "C8_CPU_FRONTIER_V1 " << environment_snapshot_fingerprint(tree) << '\n'
         << std::setprecision(17);
  }

  template <typename TModelProperties>
  template <typename TParticle, typename TStep>
  bool CpuBootstrapRouter<TModelProperties>::canRoute(TParticle const& p, TStep) const {
    return p.getGeneration() > 0 && p.getEnergy() / 1_GeV <= maximumEnergy_;
  }

  template <typename TModelProperties>
  template <typename TParticle, typename THistory, typename TGeneration, typename TStep>
  void CpuBootstrapRouter<TModelProperties>::stage(TParticle const& p, THistory history,
                                                   THistory parent,
                                                   TGeneration generation, TStep step) {
    auto s = particle_transfer::to_device_state(p, cs_, history, parent, generation, step,
                                                topology_);
    validate_frontier_state(s, HistoryStride, false);
    if (!generation || s.environmentNode_ < 0)
      throw std::runtime_error(
          "CPU bootstrap must export secondaries with logical owners");
    // PDG encodes nuclear A/Z as well; retain kinetic energy separately to avoid
    // total-minus-rest cancellation when a CPU particle is reconstructed.
    out_ << s.pid_ << ' ' << s.historyId_ << ' ' << s.parentHistoryId_ << ' '
         << s.generation_ << ' ' << s.stepId_ << ' ' << s.energyGeV_ << ' ' << s.weight_
         << ' ' << s.timeS_;
    for (auto x : s.positionM_) out_ << ' ' << x;
    for (auto x : s.direction_) out_ << ' ' << x;
    out_ << ' ' << s.environmentNode_ << ' ' << p.getKineticEnergy() / 1_GeV << '\n';
    ++count_;
  }

  template <typename TModelProperties>
  bool CpuBootstrapRouter<TModelProperties>::pending() const {
    return false;
  }

  template <typename TModelProperties>
  bool CpuBootstrapRouter<TModelProperties>::preserveLogicalNodes() const {
    return true;
  }

  template <typename TModelProperties>
  template <typename TStack>
  std::size_t CpuBootstrapRouter<TModelProperties>::advanceOneWavefrontAndReturn(
      TStack&) {
    return 0;
  }

  template <typename TModelProperties>
  void CpuBootstrapRouter<TModelProperties>::endOfShower() {
    out_.flush();
    out_.close();
    CORSIKA_LOG_INFO("Static CPU bootstrap closed: {} disjoint roots", count_);
  }

  template <typename TStack, typename TModelProperties>
  std::size_t import_cpu_frontier(TStack& stack, CoordinateSystemPtr cs,
                                  std::string const& path, unsigned worker,
                                  EnvironmentSnapshot<TModelProperties> const& snapshot) {
    auto const tree = bind_environment_data(snapshot);

    if (!stack.isEmpty() || worker < 1 || worker > 254)
      throw std::runtime_error("Invalid CPU prefix stack or worker namespace");
    std::ifstream in(path);
    std::string line, version, extra;
    std::uint64_t fingerprint{};
    if (!std::getline(in, line)) throw std::runtime_error("Missing CPU frontier header");
    std::istringstream header(line);
    if (!(header >> version >> fingerprint) || version != "C8_CPU_FRONTIER_V1" ||
        (header >> extra) || !tree.nodeCount_ ||
        fingerprint != environment_snapshot_fingerprint(tree))
      throw std::runtime_error("CPU frontier environment mismatch");
    struct Root {
      ParticleState state;
      double kinetic;
    };
    std::vector<Root> roots;
    std::unordered_set<std::uint64_t> seen;
    long double weighted = 0;
    while (std::getline(in, line)) {
      ParticleState s{};
      double kinetic{};
      std::istringstream row(line);
      row >> s.pid_ >> s.historyId_ >> s.parentHistoryId_ >> s.generation_ >> s.stepId_ >>
          s.energyGeV_ >> s.weight_ >> s.timeS_;
      for (auto& x : s.positionM_) row >> x;
      for (auto& x : s.direction_) row >> x;
      row >> s.environmentNode_ >> kinetic;
      if (!row || (row >> extra)) throw std::runtime_error("Malformed CPU frontier row");
      validate_frontier_state(s, HistoryStride, false);
      auto mass = get_mass(convert_from_PDG(static_cast<PDGCode>(s.pid_))) / 1_GeV;
      if (!s.generation_ || !std::isfinite(kinetic) || kinetic < 0 ||
          std::abs(kinetic + mass - s.energyGeV_) >
              8 * std::numeric_limits<double>::epsilon() * std::max(1., s.energyGeV_) ||
          !seen.insert(s.historyId_).second)
        throw std::runtime_error("Invalid or duplicate CPU frontier root");
      roots.push_back({s, kinetic});
      weighted += static_cast<long double>(s.energyGeV_) * s.weight_;
    }
    if (!in.eof()) throw std::runtime_error("CPU frontier read failed");
    // Input was sorted high-energy-first. Reverse insertion for the LIFO stack.
    for (auto it = roots.rbegin(); it != roots.rend(); ++it) {
      auto p = particle_transfer::import_particle(stack, it->state, cs,
                                                  snapshot.getTopology());
      p.setKineticEnergy(it->kinetic * 1_GeV);
    }
    auto next = stack.reserveTransportHistoryIds(1);
    auto base = worker * HistoryStride;
    if (next >= base) throw std::runtime_error("CPU prefix history namespace collision");
    if (base - next > 1) stack.reserveTransportHistoryIds(base - next - 1);
    nlohmann::json audit{{"roots_consumed", roots.size()},
                         {"worker_namespace", worker},
                         {"weighted_energy_GeV", double(weighted)},
                         {"environment_fingerprint", fingerprint}};
    std::ofstream report("CPU_FRONTIER_FEED.json");
    report.exceptions(std::ios::badbit | std::ios::failbit);
    report << audit.dump(2) << '\n';
    return roots.size();
  }

} // namespace corsika::kokkos::frontier

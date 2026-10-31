/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */
#pragma once
namespace corsika::kokkos::frontier {
  inline FrontierHeader::FrontierHeader(std::string const& header) {
    if (header == "C8_STATIC_FRONTIER_V1") return;
    std::istringstream in(header);
    std::string version, extra;
    if (!(in >> version >> environmentFingerprint_))
      throw std::runtime_error("Missing/unsupported frontier header");
    if (version == "C8_STATIC_FRONTIER_V3") {
      if (!(in >> historyLimit_) || historyLimit_ <= HistoryStride ||
          historyLimit_ > 255 * HistoryStride || historyLimit_ % HistoryStride)
        throw std::runtime_error("Invalid frontier history limit");
    } else if (version != "C8_STATIC_FRONTIER_V2")
      throw std::runtime_error("Missing/unsupported frontier header");
    if (in >> extra) throw std::runtime_error("Trailing frontier header fields");
    logicalNodes_ = true;
  }

  inline void FrontierHeader::validateEnvironment(EnvironmentDataView const& tree) const {
    if (logicalNodes_ != (tree.nodeCount_ != 0) ||
        (logicalNodes_ &&
         environmentFingerprint_ != environment_snapshot_fingerprint(tree)))
      throw std::runtime_error(
          "Frontier environment mismatch; logical nodes cannot be reconstructed from "
          "position");
  }

  inline bool is_routable(int pid) {
    return pid == 22 || pid == 11 || pid == -11 || pid == 13 || pid == -13;
  }

  template <typename TParticle>
  bool frontier_unit_direction(TParticle const& particle, CoordinateSystemPtr const& cs,
                                std::size_t& retained, double& maximumError) {
    auto const direction = particle.getDirection().getComponents(cs);
    double norm = 0.;
    for (int axis = 0; axis < 3; ++axis) {
      double const value = direction[axis].magnitude();
      norm += value * value;
    }
    if (!std::isfinite(norm) || norm <= 0.) {
      throw std::runtime_error("Zero or non-finite native frontier direction");
    }
    double const error = std::abs(norm - 1.);
    if (error <= DirectionNormSquaredTolerance) return true;
    ++retained;
    maximumError = std::max(maximumError, error);
    if (retained <= 4) {
      CORSIKA_LOG_WARN("Retain native CPU particle: pid={}, E={} GeV, norm2={:.17g}",
                       static_cast<int>(particle.getPDG()), particle.getEnergy() / 1_GeV,
                       norm);
    }
    return false;
  }

  inline void validate_frontier_state(ParticleState const& s, std::uint64_t history_limit,
                                      bool require_em) {
    double norm = 0.;
    for (int k = 0; k != 3; ++k) {
      if (!std::isfinite(s.positionM_[k]) || !std::isfinite(s.direction_[k]))
        throw std::runtime_error("Non-finite frontier coordinates");
      norm += s.direction_[k] * s.direction_[k];
    }
    if ((require_em && !is_routable(s.pid_)) || !std::isfinite(s.energyGeV_) ||
        s.energyGeV_ < get_mass(convert_from_PDG(static_cast<PDGCode>(s.pid_))) / 1_GeV ||
        !std::isfinite(s.timeS_) || !std::isfinite(s.weight_) || s.weight_ < 0. ||
        std::abs(norm - 1.) > DirectionNormSquaredTolerance || !s.historyId_ ||
        s.historyId_ >= history_limit ||
        s.parentHistoryId_ >= history_limit ||
        ((s.generation_ == 0) != (s.parentHistoryId_ == 0)))
      throw std::runtime_error("Invalid frontier identity, direction, energy or weight");
  }

  inline void write_frontier_state(std::ostream& out, ParticleState const& s,
                                   bool logical_nodes, std::uint64_t history_limit) {
    validate_frontier_state(s, history_limit);
    out << s.pid_ << ' ' << s.historyId_ << ' ' << s.parentHistoryId_ << ' '
        << s.generation_ << ' ' << s.stepId_ << ' ' << s.energyGeV_ << ' ' << s.weight_
        << ' ' << s.timeS_;
    for (auto x : s.positionM_) out << ' ' << x;
    for (auto x : s.direction_) out << ' ' << x;
    if (logical_nodes) {
      if (s.environmentNode_ < 0)
        throw std::runtime_error("Missing frontier logical node");
      out << ' ' << s.environmentNode_;
    }
    out << '\n';
  }

  inline FrontierInput::FrontierInput(std::string const& path)
      : file_(path) {
    std::string header;
    if (!std::getline(file_, header))
      throw std::runtime_error("Missing/unsupported frontier header: " + path);
    header_ = FrontierHeader(header);
  }

  inline void FrontierInput::setEnvironmentTree(EnvironmentDataView const& tree) {
    header_.validateEnvironment(tree);
    environment_ = tree;
  }

  template <typename TStack>
  void FrontierInput::reserveNamespace(TStack& stack, unsigned worker) {
    if (worker < 1 || worker > 255) throw std::runtime_error("Worker ID outside [1,255]");
    auto const next = stack.reserveTransportHistoryIds(1);
    auto const base = worker * HistoryStride;
    if (base < header_.historyLimit_)
      throw std::runtime_error("Worker namespace overlaps imported CPU histories");
    namespaceBase_ = base;
    if (next >= base) throw std::runtime_error("Worker history namespace collision");
    if (base - next > 1) stack.reserveTransportHistoryIds(base - next - 1);
  }

  inline bool FrontierInput::pending() const { return !exhausted_; }

  inline std::size_t FrontierInput::count() const { return seen_.size(); }

  inline std::size_t FrontierInput::batches() const { return batches_; }

  inline std::size_t FrontierInput::peakBatch() const { return peakBatch_; }

  inline double FrontierInput::weightedEnergyGeV() const {
    return static_cast<double>(weightedEnergyGeV_);
  }

  template <typename TStack, typename TModelProperties>
  std::size_t FrontierInput::refill(
      TStack& stack, CoordinateSystemPtr const& cs, std::size_t limit,
      EnvironmentTopology<TModelProperties> const* topology) {
    header_.validateEnvironment(environment_);
    if (!topology)
      throw std::logic_error("Frontier import requires a bound native topology");
    if (!stack.isEmpty())
      throw std::logic_error("Frontier refill requires a drained scalar stack");
    if (limit == 0 || limit > 65536)
      throw std::logic_error("Invalid bounded frontier batch");
    std::vector<ParticleState> batch;
    batch.reserve(limit);
    std::string line;
    while (batch.size() < limit && std::getline(file_, line)) {
      ParticleState s{};
      std::istringstream row(line);
      row >> s.pid_ >> s.historyId_ >> s.parentHistoryId_ >> s.generation_ >> s.stepId_ >>
          s.energyGeV_ >> s.weight_ >> s.timeS_;
      for (auto& x : s.positionM_) row >> x;
      for (auto& x : s.direction_) row >> x;
      if (header_.logicalNodes_) row >> s.environmentNode_;
      std::string trailing;
      if (!row || (row >> trailing)) throw std::runtime_error("Malformed frontier row");
      validate_frontier_state(s, header_.historyLimit_);
      if (namespaceBase_ &&
          (s.historyId_ >= namespaceBase_ || s.parentHistoryId_ >= namespaceBase_))
        throw std::runtime_error("Imported root overlaps worker namespace");
      if (!seen_.insert(s.historyId_).second)
        throw std::runtime_error("Duplicate frontier root");
      weightedEnergyGeV_ += static_cast<long double>(s.energyGeV_) * s.weight_;
      if (!std::isfinite(weightedEnergyGeV_) ||
          weightedEnergyGeV_ > std::numeric_limits<double>::max())
        throw std::runtime_error("Imported frontier energy overflow");
      batch.push_back(s);
    }
    if (file_.eof())
      exhausted_ = true;
    else if (!file_)
      throw std::runtime_error("Frontier read failed");
    for (auto it = batch.rbegin(); it != batch.rend(); ++it)
      particle_transfer::import_particle(stack, *it, cs, *topology);
    if (!batch.empty()) ++batches_;
    peakBatch_ = std::max(peakBatch_, batch.size());
    return batch.size();
  }

  inline std::size_t static_frontier_import_limit(std::size_t original) {
    if (auto const* value = std::getenv("C8_STATIC_FRONTIER_IMPORT_BATCH")) {
      std::string text(value);
      if (text.empty() || text.size() > 5 ||
          !std::all_of(text.begin(), text.end(),
                       [](char c) { return c >= '0' && c <= '9'; }))
        throw std::runtime_error("Invalid static frontier import batch");
      auto const count = std::stoul(text);
      if (count == 0 || count > 65536)
        throw std::runtime_error("Static frontier import batch outside [1,65536]");
      return count;
    }
    return std::min<std::size_t>(65536, std::max<std::size_t>(1, original));
  }

  inline ScopedFrontierInput::ScopedFrontierInput(FrontierInput* input,
                                                  std::string output) {
    if (activeFrontierInput) throw std::logic_error("Nested frontier input");
    activeFrontierInput = input;
    activeFrontierOutput = std::move(output);
  }

  inline ScopedFrontierInput::~ScopedFrontierInput() {
    activeFrontierInput = nullptr;
    activeFrontierOutput.clear();
  }

  template <typename TBaseRouter, typename TModelProperties>
  void BufferedFrontierRouter<TBaseRouter, TModelProperties>::setFrontierEnvironment(
      EnvironmentSnapshot<TModelProperties> const& snapshot) {
    if (source_) {
      source_->setEnvironmentTree(bind_environment_data(snapshot));
      topology_ = &snapshot.getTopology();
    }
  }

  template <typename TBaseRouter, typename TModelProperties>
  template <typename... TArgs>
  BufferedFrontierRouter<TBaseRouter, TModelProperties>::BufferedFrontierRouter(
      ForwardRouterArguments, CoordinateSystemPtr const& cs, std::size_t limit,
      TArgs&&... args)
      : TBaseRouter(std::forward<TArgs>(args)...)
      , source_(activeFrontierInput)
      , cs_(cs)
      , limit_(static_frontier_import_limit(limit)) {}

  template <typename TBaseRouter, typename TModelProperties>
  template <typename TBackend, typename TEnvironment, typename TFallback,
            typename TOutput>
  BufferedFrontierRouter<TBaseRouter, TModelProperties>::BufferedFrontierRouter(
      TBackend& backend, CoordinateSystemPtr const& cs, TEnvironment const& env,
      TransportGeometryConfig const& geometry,
      EnvironmentTopology<TModelProperties> const& topology, TFallback& fallback,
      TOutput& output)
      : TBaseRouter(backend, cs, env, geometry, topology, fallback, output)
      , source_(activeFrontierInput)
      , cs_(cs)
      , limit_(static_frontier_import_limit(backend.minimumBatchSize())) {}

  template <typename TBaseRouter, typename TModelProperties>
  bool BufferedFrontierRouter<TBaseRouter, TModelProperties>::pending() const {
    return TBaseRouter::pending() || (source_ && source_->pending());
  }

  template <typename TBaseRouter, typename TModelProperties>
  template <typename TStack>
  std::size_t
  BufferedFrontierRouter<TBaseRouter, TModelProperties>::advanceOneWavefrontAndReturn(
      TStack& stack) {
    // Existing resident transport and rare-final-state handling drain locally.
    // Never mix new roots with an in-flight step or an unfinished fallback.
    if (TBaseRouter::pending()) return TBaseRouter::advanceOneWavefrontAndReturn(stack);
    if (source_ && source_->pending())
      return source_->refill(stack, cs_, limit_, topology_);
    return 0;
  }

  template <typename TBaseRouter, typename TModelProperties>
  void BufferedFrontierRouter<TBaseRouter, TModelProperties>::endOfShower() {
    if (source_ && source_->pending())
      throw std::logic_error("Unconsumed frontier roots");
    TBaseRouter::endOfShower();
    if (source_) {
      std::ofstream audit("FRONTIER_FEED.json");
      audit.exceptions(std::ios::badbit | std::ios::failbit);
      audit << "{\"mode\":\"bounded-high-energy-first\",\"roots_consumed\":"
            << source_->count() << ",\"batches\":" << source_->batches()
            << ",\"maximum_imported_batch\":" << source_->peakBatch()
            << ",\"weighted_energy_GeV\":" << std::setprecision(17)
            << source_->weightedEnergyGeV() << ",\"configured_import_batch\":" << limit_
            << "}\n";
      CORSIKA_LOG_INFO(
          "Bounded frontier drained: {} roots, {} batches, peak imported batch {}",
          source_->count(), source_->batches(), source_->peakBatch());
    }
  }

  template <typename TModelProperties>
  CaptureRouter<TModelProperties>::CaptureRouter(
      std::string const& path, CoordinateSystemPtr cs, double cap,
      EnvironmentSnapshot<TModelProperties> const& snapshot, std::uint64_t history_limit)
      : cs_(cs)
      , maximumEnergy_(cap)
      , file_(path)
      , environment_(bind_environment_data(snapshot))
      , historyLimit_(history_limit)
      , topology_(snapshot.getTopology()) {
    auto const& tree = environment_;

    if (!file_) throw std::runtime_error("Cannot create frontier: " + path);
    if (history_limit != HistoryStride) {
      if (!tree.nodeCount_ || history_limit <= HistoryStride ||
          history_limit > 255 * HistoryStride || history_limit % HistoryStride)
        throw std::runtime_error("Invalid CPU-prefix history namespace limit");
      file_ << "C8_STATIC_FRONTIER_V3 " << environment_snapshot_fingerprint(tree) << ' '
            << history_limit << '\n';
    } else if (tree.nodeCount_)
      file_ << "C8_STATIC_FRONTIER_V2 " << environment_snapshot_fingerprint(tree) << '\n';
    else
      file_ << "C8_STATIC_FRONTIER_V1\n";
    file_ << std::setprecision(17);
    file_.exceptions(std::ios::badbit | std::ios::failbit);
  }

  template <typename TModelProperties>
  template <typename TParticle, typename TStep>
  bool CaptureRouter<TModelProperties>::canRoute(TParticle const& p, TStep) const {
    return is_routable(static_cast<int>(p.getPDG())) &&
           p.getEnergy() / 1_GeV <= maximumEnergy_ &&
           frontier_unit_direction(p, cs_, retainedNativeDirectionSteps_,
                                    maximumNativeDirectionError_);
  }

  template <typename TModelProperties>
  template <typename TParticle, typename THistory, typename TGeneration, typename TStep>
  void CaptureRouter<TModelProperties>::stage(TParticle const& p, THistory history,
                                              THistory parent, TGeneration generation,
                                              TStep step) {
    // beginTransportStep increments storage but RETURNS the old (zero-based)
    // counter. No physical step happened: export the returned value unchanged.
    write_frontier_state(file_,
                         particle_transfer::to_device_state(p, cs_, history, parent,
                                                            generation, step, topology_),
                         environment_.nodeCount_ != 0, historyLimit_);
    ++count_;
  }

  template <typename TModelProperties>
  bool CaptureRouter<TModelProperties>::pending() const {
    return false;
  }

  template <typename TModelProperties>
  bool CaptureRouter<TModelProperties>::preserveLogicalNodes() const {
    return environment_.nodeCount_ != 0;
  }

  template <typename TModelProperties>
  template <typename TStack>
  std::size_t CaptureRouter<TModelProperties>::advanceOneWavefrontAndReturn(TStack&) {
    return 0;
  }

  template <typename TModelProperties>
  void CaptureRouter<TModelProperties>::endOfShower() {
    file_.flush();
    file_.close();
    CORSIKA_LOG_INFO("Static frontier closed: {} disjoint roots", count_);
    if (retainedNativeDirectionSteps_) {
      CORSIKA_LOG_INFO("Native direction retention: {} CPU steps; maximum norm2 error {}",
                       retainedNativeDirectionSteps_, maximumNativeDirectionError_);
    }
  }

  template <typename TStack, typename TModelProperties>
  std::size_t import_frontier(TStack& stack, CoordinateSystemPtr cs,
                              std::string const& path, unsigned worker,
                              EnvironmentSnapshot<TModelProperties> const& snapshot) {
    auto const tree = bind_environment_data(snapshot);

    if (worker < 1 || worker > 255) throw std::runtime_error("Worker ID outside [1,255]");
    std::ifstream in(path);
    std::string line;
    if (!std::getline(in, line))
      throw std::runtime_error("Missing/unsupported frontier header: " + path);
    FrontierHeader const header(line);
    header.validateEnvironment(tree);
    std::set<std::uint64_t> seen;
    while (std::getline(in, line)) {
      ParticleState s{};
      std::istringstream row(line);
      row >> s.pid_ >> s.historyId_ >> s.parentHistoryId_ >> s.generation_ >> s.stepId_ >>
          s.energyGeV_ >> s.weight_ >> s.timeS_;
      for (auto& x : s.positionM_) row >> x;
      for (auto& x : s.direction_) row >> x;
      if (header.logicalNodes_) row >> s.environmentNode_;
      std::string trailing;
      if (!row || (row >> trailing)) throw std::runtime_error("Malformed frontier row");
      validate_frontier_state(s, header.historyLimit_);
      if (!seen.insert(s.historyId_).second)
        throw std::runtime_error("Duplicate frontier root");
      particle_transfer::import_particle(stack, s, cs, snapshot.getTopology());
    }
    if (!in.eof()) throw std::runtime_error("Frontier read failed");
    // Ancestors retain original IDs; children use an exclusive 48-bit range.
    auto const next = stack.reserveTransportHistoryIds(1);
    auto const base = worker * HistoryStride;
    if (base < header.historyLimit_)
      throw std::runtime_error("Worker namespace overlaps input");
    if (next >= base) throw std::runtime_error("Worker history namespace collision");
    if (base - next > 1) stack.reserveTransportHistoryIds(base - next - 1);
    CORSIKA_LOG_INFO("Worker {} imported {} roots; new histories start at {}", worker,
                     seen.size(), base);
    return seen.size();
  }

} // namespace corsika::kokkos::frontier

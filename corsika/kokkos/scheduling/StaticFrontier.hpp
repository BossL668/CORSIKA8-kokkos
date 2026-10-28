/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */
#pragma once
#include <corsika/kokkos/data/RouterParticleConversion.hpp>
#include <corsika/kokkos/environment/EnvironmentSnapshotData.hpp>
#include <corsika/kokkos/environment/EnvironmentSnapshotKey.hpp>
#include <corsika/kokkos/environment/TransportGeometryConfig.hpp>
#include <corsika/framework/core/Logging.hpp>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstddef>
#include <cstdint>
#include <ostream>
#include <fstream>
#include <iomanip>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace corsika::kokkos::frontier {
  inline constexpr std::uint64_t HistoryStride = std::uint64_t{1} << 48;
  struct FrontierHeader {
    bool logicalNodes_{};
    std::uint64_t environmentFingerprint_{};
    std::uint64_t historyLimit_{HistoryStride};
    explicit FrontierHeader(std::string const& header);
    void validateEnvironment(EnvironmentDataView const& tree) const;
  };
  bool is_routable(int pid);
  void validate_frontier_state(ParticleState const& s,
                               std::uint64_t history_limit = HistoryStride,
                               bool require_em = true);
  void write_frontier_state(std::ostream& out, ParticleState const& s,
                            bool logical_nodes = false,
                            std::uint64_t history_limit = HistoryStride);

  /** Host file reader for bounded static imports. No root is injected until the
   * existing resident/fallback work and scalar stack have drained. Native node
   * IDs are restored through the supplied topology, never by containment search.
   */
  class FrontierInput {
  public:
    explicit FrontierInput(std::string const& path);
    void setEnvironmentTree(EnvironmentDataView const& tree);
    template <typename TStack>
    void reserveNamespace(TStack& stack, unsigned worker);
    bool pending() const;
    std::size_t count() const;
    std::size_t batches() const;
    std::size_t peakBatch() const;
    double weightedEnergyGeV() const;
    template <typename TStack, typename TModelProperties>
    std::size_t refill(TStack& stack, CoordinateSystemPtr const& cs, std::size_t limit,
                       EnvironmentTopology<TModelProperties> const* topology);

  private:
    std::ifstream file_;
    bool exhausted_{};
    std::unordered_set<std::uint64_t> seen_;
    std::size_t batches_{}, peakBatch_{};
    long double weightedEnergyGeV_{};
    FrontierHeader header_{"C8_STATIC_FRONTIER_V1"};
    EnvironmentDataView environment_{};
    std::uint64_t namespaceBase_{};
  };
  // One TLS slot across all native environment model types, as in production.
  inline thread_local FrontierInput* activeFrontierInput = nullptr;
  inline thread_local std::string activeFrontierOutput;
  struct ForwardRouterArguments {};
  std::size_t static_frontier_import_limit(std::size_t original);
  class ScopedFrontierInput {
  public:
    explicit ScopedFrontierInput(FrontierInput* input, std::string output = {});
    ~ScopedFrontierInput();
    ScopedFrontierInput(ScopedFrontierInput const&) = delete;
    ScopedFrontierInput& operator=(ScopedFrontierInput const&) = delete;
  };

  /** Production bounded-import adaptor. Snapshot/native nodes must outlive drain.
   * setFrontierEnvironment binds the same numeric snapshot and native topology.
   * Constructors retain the base router's construction before TLS/limit capture.
   */
  template <typename TBaseRouter, typename TModelProperties>
  class BufferedFrontierRouter : public TBaseRouter {
  public:
    void setFrontierEnvironment(EnvironmentSnapshot<TModelProperties> const& snapshot);
    template <typename... TArgs>
    BufferedFrontierRouter(ForwardRouterArguments, CoordinateSystemPtr const& cs,
                           std::size_t limit, TArgs&&... args);
    template <typename TBackend, typename TEnvironment, typename TFallback,
              typename TOutput>
    BufferedFrontierRouter(TBackend& backend, CoordinateSystemPtr const& cs,
                           TEnvironment const& env,
                           TransportGeometryConfig const& geometry,
                           EnvironmentTopology<TModelProperties> const& topology,
                           TFallback& fallback, TOutput& output);
    bool pending() const;
    template <typename TStack>
    std::size_t advanceOneWavefrontAndReturn(TStack& stack);
    void endOfShower();

  private:
    FrontierInput* source_;
    CoordinateSystemPtr cs_;
    std::size_t limit_;
    EnvironmentTopology<TModelProperties> const* topology_{};
  };
  template <typename TModelProperties>
  class CaptureRouter {
  public:
    CaptureRouter(std::string const& path, CoordinateSystemPtr cs, double cap,
                  EnvironmentSnapshot<TModelProperties> const& snapshot,
                  std::uint64_t history_limit = HistoryStride);
    template <typename TParticle, typename TStep>
    bool canRoute(TParticle const& p, TStep) const;
    template <typename TParticle, typename THistory, typename TGeneration, typename TStep>
    void stage(TParticle const& p, THistory history, THistory parent,
               TGeneration generation, TStep step);
    bool pending() const;
    bool preserveLogicalNodes() const;
    template <typename TStack>
    std::size_t advanceOneWavefrontAndReturn(TStack&);
    void endOfShower();

  private:
    CoordinateSystemPtr cs_;
    double maximumEnergy_;
    std::ofstream file_;
    std::size_t count_{};
    EnvironmentDataView environment_{};
    std::uint64_t historyLimit_{HistoryStride};
    EnvironmentTopology<TModelProperties> const& topology_;
  };
  /// Original eager import order and history reservation, with mandatory native tree.
  template <typename TStack, typename TModelProperties>
  std::size_t import_frontier(TStack& stack, CoordinateSystemPtr cs,
                              std::string const& path, unsigned worker,
                              EnvironmentSnapshot<TModelProperties> const& snapshot);
} // namespace corsika::kokkos::frontier
#include <corsika/detail/kokkos/scheduling/StaticFrontier.inl>

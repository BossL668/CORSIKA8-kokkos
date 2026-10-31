/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */
#pragma once
#include <corsika/kokkos/scheduling/StaticFrontier.hpp>
#include <nlohmann/json.hpp>
namespace corsika::kokkos::frontier {
  /** Capture already-created secondary species eligible for frontier export.
   * The initial primary is neither copied nor forced to interact. Borrows the
   * snapshot/native node map until the output is closed. Non-unit directions
   * stay in the native cascade until they satisfy the export contract. This
   * router neither renormalizes particles nor samples physics.
   */
  template <typename TModelProperties>
  class CpuBootstrapRouter {
  public:
    CpuBootstrapRouter(std::string const& path, CoordinateSystemPtr cs,
                       EnvironmentSnapshot<TModelProperties> const& snapshot,
                       double maximum_energy = std::numeric_limits<double>::infinity());
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
    EnvironmentDataView tree_;
    std::ofstream out_;
    std::size_t count_{};
    mutable std::size_t retainedNativeDirectionSteps_{};
    mutable double maximumNativeDirectionError_{};
    double maximumEnergy_;
    EnvironmentTopology<TModelProperties> const& topology_;
  };
  template <typename TStack, typename TModelProperties>
  std::size_t import_cpu_frontier(TStack& stack, CoordinateSystemPtr cs,
                                  std::string const& path, unsigned worker,
                                  EnvironmentSnapshot<TModelProperties> const& snapshot);
} // namespace corsika::kokkos::frontier
#include <corsika/detail/kokkos/scheduling/CpuFrontier.inl>

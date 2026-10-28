/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/kokkos/em/execution/LeptonFrontData.hpp>
#include <corsika/kokkos/em/algorithms/LeptonVertexSelection.hpp>
#include <corsika/kokkos/em/algorithms/LeptonFinalStateStep.hpp>
#include <corsika/kokkos/em/LeptonTransportRecord.hpp>
#include <corsika/kokkos/em/models/MoliereScattering.hpp>
#include <corsika/kokkos/output/ProfileAtomicOperations.hpp>
#include <corsika/kokkos/output/HostProfileShards.hpp>
#include <corsika/kokkos/output/algorithms/ProfileAccumulation.hpp>
#include <Kokkos_Core.hpp>
#include <type_traits>
#include <cstddef>
#include <cstdint>

namespace corsika::kokkos {

  struct ResidentLeptonTransportStatisticsValue {
    std::uint64_t values_[LeptonTransportStatisticCount]{};
    std::uint64_t materialBoundaries_{};
  };
  struct ResidentLeptonInteractionStatisticsValue {
    std::uint64_t values_[LeptonInteractionStatisticCount]{};
  };
  template <class TExecutionSpace>
  struct ResidentLeptonTransportStatisticsFunctor {
    using memory_space_type = typename TExecutionSpace::memory_space;
    using value_type = ResidentLeptonTransportStatisticsValue;

    Kokkos::View<LeptonTransportRecord const*, memory_space_type> transports_;
    Kokkos::View<ResidentLeptonSourceCounts const*, memory_space_type> counts_;
    Kokkos::View<std::uint64_t*, memory_space_type> callStatistics_;

    KOKKOS_INLINE_FUNCTION void init(value_type& value) const;

    KOKKOS_INLINE_FUNCTION void join(value_type& destination,
                                     value_type const& source) const;

    KOKKOS_INLINE_FUNCTION void operator()(std::size_t const source,
                                           value_type& value) const;

    KOKKOS_INLINE_FUNCTION void final(value_type& value) const;
  };
  template <class TExecutionSpace>
  struct ResidentLeptonTransportStatisticsProfileFunctor {
    using memory_space_type = typename TExecutionSpace::memory_space;
    using value_type = ResidentLeptonTransportStatisticsValue;

    ResidentLeptonTransportStatisticsFunctor<TExecutionSpace> statistics_;
    ProfileProjectionView projection_;
    ProfileAccumulatorView accumulator_;
    Kokkos::View<LeptonTransportRecord*, memory_space_type> steps_;
    std::size_t stepCount_{};

    KOKKOS_INLINE_FUNCTION void init(value_type& value) const;

    KOKKOS_INLINE_FUNCTION void join(value_type& destination,
                                     value_type const& source) const;

    KOKKOS_INLINE_FUNCTION void final(value_type& value) const;

    KOKKOS_INLINE_FUNCTION void operator()(std::size_t const source,
                                           value_type& value) const;
  };
  template <class TExecutionSpace>
  struct ResidentLeptonInteractionStatisticsFunctor {
    using memory_space_type = typename TExecutionSpace::memory_space;
    using value_type = ResidentLeptonInteractionStatisticsValue;

    Kokkos::View<LeptonVertexSelectionOutcome const*, memory_space_type> vertices_;
    Kokkos::View<LeptonFinalStateClassification const*, memory_space_type> finalStates_;
    Kokkos::View<std::uint64_t*, memory_space_type> callStatistics_;

    KOKKOS_INLINE_FUNCTION void init(value_type& value) const;

    KOKKOS_INLINE_FUNCTION void join(value_type& destination,
                                     value_type const& source) const;

    KOKKOS_INLINE_FUNCTION void operator()(std::size_t const candidate,
                                           value_type& value) const;

    KOKKOS_INLINE_FUNCTION void final(value_type& value) const;
  };
  template <class TExecutionSpace>
  struct ResidentLeptonInteractionStatisticsProfileFunctor {
    using memory_space_type = typename TExecutionSpace::memory_space;
    using value_type = ResidentLeptonInteractionStatisticsValue;

    ResidentLeptonInteractionStatisticsFunctor<TExecutionSpace> statistics_;
    ProfileProjectionView projection_;
    ProfileAccumulatorView accumulator_;
    Kokkos::View<LeptonTransportRecord*, memory_space_type> steps_;
    std::size_t stepCount_{};
    Kokkos::View<LeptonFinalStateRecord*, memory_space_type> records_;
    std::size_t recordCount_{};

    KOKKOS_INLINE_FUNCTION void init(value_type& value) const;

    KOKKOS_INLINE_FUNCTION void join(value_type& destination,
                                     value_type const& source) const;

    KOKKOS_INLINE_FUNCTION void final(value_type& value) const;

    KOKKOS_INLINE_FUNCTION void operator()(std::size_t const interactionIndex,
                                           value_type& value) const;
  };
#ifdef KOKKOS_ENABLE_OPENMP
  struct HostShardedLeptonProfileFunctor
      : ResidentLeptonTransportStatisticsProfileFunctor<Kokkos::OpenMP> {
    using base_type = ResidentLeptonTransportStatisticsProfileFunctor<Kokkos::OpenMP>;
    HostProfileShards::Entry const* shards_{};
    std::size_t shardCount_{};

    void operator()(std::size_t const source, value_type& value) const;
  };
#endif
  /** Enqueue the production transport/profile reduction with optional host shards.
   * Input views, projection support, accumulator and shards must remain alive until
   * the supplied execution instance finishes. Call only on an open profile owner.
   * The source domain and packed step domain differ; neither may be reordered here.
   */
  template <class TExecutionSpace, class TPolicy>
  void enqueue_lepton_profile_statistics(
      TPolicy const& policy,
      ResidentLeptonTransportStatisticsProfileFunctor<TExecutionSpace> const& functor,
      [[maybe_unused]] HostProfileShards const& shards);

} // namespace corsika::kokkos

#include <corsika/detail/kokkos/em/execution/LeptonStatistics.inl>

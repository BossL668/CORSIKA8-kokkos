/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/kokkos/scheduling/HadronicBatchProtocol.hpp>
#include <corsika/framework/core/PhysicalUnits.hpp>
#include <corsika/framework/core/ParticleProperties.hpp>
#include <corsika/framework/geometry/FourVector.hpp>

#include <array>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace corsika::kokkos {

  struct PreparedHadronicInteraction {
    HadronicInteractionRequest request_{};
  };

  /**
   * One-transport-step capture point for a process-isolated hadronic interaction.
   *
   * A HybridCascade activates the context immediately around
   * ScalarCascadeStepper::advance(). A worker-capable InteractionProcess can then
   * serialize the already selected process, target, vertex four-momenta, and keyed
   * random identity instead of calling its non-thread-safe generator in the main
   * process.
   *
   * The context is thread-local only to avoid adding a worker parameter to every
   * ProcessSequence template. It never owns a stack iterator and is inactive on the
   * legacy Cascade path.
   */
  class HadronicInteractionDeferralContext {
  public:
    HadronicInteractionDeferralContext(HadronicRandomKey random_key,
                                       std::uint64_t const sequence_id);

    HadronicInteractionDeferralContext(HadronicInteractionDeferralContext const&) =
        delete;
    HadronicInteractionDeferralContext& operator=(
        HadronicInteractionDeferralContext const&) = delete;

    bool tryPrepare(HadronicWorkerModel const model, Code const projectile,
                    Code const target, FourMomentum const& projectileP4,
                    FourMomentum const& targetP4);

    bool hasPrepared() const noexcept;

    PreparedHadronicInteraction takePrepared();

    static HadronicInteractionDeferralContext* current() noexcept;

  private:
    friend class ScopedHadronicInteractionDeferral;

    static inline thread_local HadronicInteractionDeferralContext* current_ = nullptr;

    HadronicRandomKey randomKey_{};
    std::uint64_t sequenceId_{};
    std::optional<PreparedHadronicInteraction> prepared_;
  };

  class ScopedHadronicInteractionDeferral {
  public:
    explicit ScopedHadronicInteractionDeferral(
        HadronicInteractionDeferralContext& context);

    ScopedHadronicInteractionDeferral(ScopedHadronicInteractionDeferral const&) = delete;
    ScopedHadronicInteractionDeferral& operator=(
        ScopedHadronicInteractionDeferral const&) = delete;

    ~ScopedHadronicInteractionDeferral();

  private:
    HadronicInteractionDeferralContext* previous_{};
  };

  template <typename TProcess, typename = void>
  struct HasHadronicWorkerModel : std::false_type {};

  template <typename TProcess>
  struct HasHadronicWorkerModel<TProcess,
                                std::void_t<decltype(TProcess::hadronic_worker_model)>>
      : std::true_type {};

  template <typename TProcess>
  inline constexpr bool has_hadronic_worker_model_v =
      HasHadronicWorkerModel<TProcess>::value;

} // namespace corsika::kokkos

#include <corsika/detail/kokkos/scheduling/HadronicInteractionDeferral.inl>

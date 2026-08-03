/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/framework/core/HadronicBatchProtocol.hpp>
#include <corsika/framework/core/PhysicalUnits.hpp>
#include <corsika/framework/geometry/FourVector.hpp>

#include <array>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace corsika {

  struct PreparedHadronicInteraction {
    HadronicInteractionRequest request{};
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
    HadronicInteractionDeferralContext(
        HadronicRandomKey random_key,
        std::uint64_t const sequence_id)
        : random_key_(random_key)
        , sequence_id_(sequence_id) {}

    HadronicInteractionDeferralContext(
        HadronicInteractionDeferralContext const&) = delete;
    HadronicInteractionDeferralContext& operator=(
        HadronicInteractionDeferralContext const&) = delete;

    bool tryPrepare(
        HadronicWorkerModel const model,
        Code const projectile, Code const target,
        FourMomentum const& projectileP4,
        FourMomentum const& targetP4) {
      if (prepared_) {
        throw std::logic_error(
            "A scalar transport step attempted to defer more than one "
            "hadronic interaction");
      }

      HadronicInteractionRequest request;
      request.model = model;
      request.sequence_id = sequence_id_;
      request.random_key = random_key_;
      request.projectile_pdg =
          static_cast<std::int32_t>(
              get_PDG(projectile));
      request.target_pdg =
          static_cast<std::int32_t>(
              get_PDG(target));

      auto encode = [](FourMomentum const& p4) {
        auto const& vector =
            p4.getSpaceLikeComponents();
        auto const& coordinateSystem =
            vector.getCoordinateSystem();
        return std::array<double, 4>{
            p4.getTimeLikeComponent() / 1_GeV,
            vector.getX(coordinateSystem) / 1_GeV,
            vector.getY(coordinateSystem) / 1_GeV,
            vector.getZ(coordinateSystem) / 1_GeV};
      };
      request.projectile_four_momentum_GeV =
          encode(projectileP4);
      request.target_four_momentum_GeV =
          encode(targetP4);
      validateHadronicInteractionRequest(request);
      prepared_ = PreparedHadronicInteraction{
          std::move(request)};
      return true;
    }

    bool hasPrepared() const noexcept {
      return prepared_.has_value();
    }

    PreparedHadronicInteraction takePrepared() {
      if (!prepared_) {
        throw std::logic_error(
            "No deferred hadronic interaction is available");
      }
      auto result = std::move(*prepared_);
      prepared_.reset();
      return result;
    }

    static HadronicInteractionDeferralContext* current() noexcept {
      return current_;
    }

  private:
    friend class ScopedHadronicInteractionDeferral;

    static inline thread_local
        HadronicInteractionDeferralContext* current_ =
            nullptr;

    HadronicRandomKey random_key_{};
    std::uint64_t sequence_id_{};
    std::optional<PreparedHadronicInteraction>
        prepared_;
  };

  class ScopedHadronicInteractionDeferral {
  public:
    explicit ScopedHadronicInteractionDeferral(
        HadronicInteractionDeferralContext& context)
        : previous_(
              HadronicInteractionDeferralContext::current_) {
      HadronicInteractionDeferralContext::current_ =
          &context;
    }

    ScopedHadronicInteractionDeferral(
        ScopedHadronicInteractionDeferral const&) = delete;
    ScopedHadronicInteractionDeferral& operator=(
        ScopedHadronicInteractionDeferral const&) = delete;

    ~ScopedHadronicInteractionDeferral() {
      HadronicInteractionDeferralContext::current_ =
          previous_;
    }

  private:
    HadronicInteractionDeferralContext* previous_{};
  };

  template <typename TProcess, typename = void>
  struct HasHadronicWorkerModel : std::false_type {};

  template <typename TProcess>
  struct HasHadronicWorkerModel<
      TProcess,
      std::void_t<decltype(
          TProcess::hadronic_worker_model)>>
      : std::true_type {};

  template <typename TProcess>
  inline constexpr bool has_hadronic_worker_model_v =
      HasHadronicWorkerModel<TProcess>::value;

} // namespace corsika

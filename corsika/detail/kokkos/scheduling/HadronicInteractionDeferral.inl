/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once
namespace corsika::kokkos {
  inline HadronicInteractionDeferralContext::HadronicInteractionDeferralContext(
      HadronicRandomKey random_key, std::uint64_t const sequence_id)
      : randomKey_(random_key)
      , sequenceId_(sequence_id) {}
  inline bool HadronicInteractionDeferralContext::tryPrepare(
      HadronicWorkerModel const model, Code const projectile, Code const target,
      FourMomentum const& projectileP4, FourMomentum const& targetP4) {
    if (prepared_) {
      throw std::logic_error(
          "A scalar transport step attempted to defer more than one "
          "hadronic interaction");
    }

    HadronicInteractionRequest request;
    request.model_ = model;
    request.sequenceId_ = sequenceId_;
    request.randomKey_ = randomKey_;
    request.projectilePdg_ = static_cast<std::int32_t>(get_PDG(projectile));
    request.targetPdg_ = static_cast<std::int32_t>(get_PDG(target));

    auto encode = [](FourMomentum const& p4) {
      auto const& vector = p4.getSpaceLikeComponents();
      auto const& coordinateSystem = vector.getCoordinateSystem();
      return std::array<double, 4>{
          p4.getTimeLikeComponent() / 1_GeV, vector.getX(coordinateSystem) / 1_GeV,
          vector.getY(coordinateSystem) / 1_GeV, vector.getZ(coordinateSystem) / 1_GeV};
    };
    request.projectileFourMomentumGeV_ = encode(projectileP4);
    request.targetFourMomentumGeV_ = encode(targetP4);
    validate_hadronic_interaction_request(request);
    prepared_ = PreparedHadronicInteraction{std::move(request)};
    return true;
  }
  inline bool HadronicInteractionDeferralContext::hasPrepared() const noexcept {
    return prepared_.has_value();
  }
  inline PreparedHadronicInteraction HadronicInteractionDeferralContext::takePrepared() {
    if (!prepared_) {
      throw std::logic_error("No deferred hadronic interaction is available");
    }
    auto result = std::move(*prepared_);
    prepared_.reset();
    return result;
  }
  inline HadronicInteractionDeferralContext*
  HadronicInteractionDeferralContext::current() noexcept {
    return current_;
  }
  inline ScopedHadronicInteractionDeferral::ScopedHadronicInteractionDeferral(
      HadronicInteractionDeferralContext& context)
      : previous_(HadronicInteractionDeferralContext::current_) {
    HadronicInteractionDeferralContext::current_ = &context;
  }
  inline ScopedHadronicInteractionDeferral::~ScopedHadronicInteractionDeferral() {
    HadronicInteractionDeferralContext::current_ = previous_;
  }
} // namespace corsika::kokkos

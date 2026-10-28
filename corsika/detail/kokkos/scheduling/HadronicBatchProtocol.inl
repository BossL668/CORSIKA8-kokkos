/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once
namespace corsika::kokkos {
  inline std::uint64_t derive_hadronic_worker_seed(
      HadronicRandomKey const& key) noexcept {
    auto mix = [](std::uint64_t value) noexcept {
      value += 0x9e3779b97f4a7c15ULL;
      value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
      value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
      return value ^ (value >> 31U);
    };

    auto state = mix(key.seed_);
    state = mix(state ^ key.showerId_);
    state = mix(state ^ key.historyId_);
    state = mix(state ^ key.stepId_);
    state = mix(state ^ (static_cast<std::uint64_t>(key.processId_) << 32U) ^
                static_cast<std::uint64_t>(key.drawDomain_));
    return state;
  }
  inline void validate_hadronic_batch_header(HadronicBatchHeader const& header) {
    if (header.magic_ != HadronicBatchMagic) {
      throw std::invalid_argument("Invalid hadronic batch magic");
    }
    if (header.protocolVersion_ != HadronicBatchProtocolVersion) {
      throw std::invalid_argument("Unsupported hadronic batch protocol version");
    }
    if (header.endianMarker_ != HadronicBatchEndianMarker) {
      throw std::invalid_argument("Hadronic batch worker endian mismatch");
    }
  }
  inline void validate_hadronic_interaction_request(
      HadronicInteractionRequest const& request) {
    if (request.protocolVersion_ != HadronicBatchProtocolVersion) {
      throw std::invalid_argument("Unsupported hadronic interaction request version");
    }
    if (request.projectilePdg_ == 0 || request.targetPdg_ == 0) {
      throw std::invalid_argument(
          "Hadronic interaction request contains an invalid PDG code");
    }
    auto validateFourMomentum = [](std::array<double, 4> const& p4) {
      if (!std::all_of(p4.begin(), p4.end(),
                       [](double const value) { return std::isfinite(value); })) {
        throw std::invalid_argument(
            "Hadronic interaction request contains a non-finite four-momentum");
      }
      if (!(p4[0] > 0.)) {
        throw std::invalid_argument(
            "Hadronic interaction request energy must be positive");
      }
    };
    validateFourMomentum(request.projectileFourMomentumGeV_);
    validateFourMomentum(request.targetFourMomentumGeV_);
  }
} // namespace corsika::kokkos

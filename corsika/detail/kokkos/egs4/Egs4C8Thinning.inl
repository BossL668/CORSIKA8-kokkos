/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once
#include <corsika/detail/kokkos/egs4/Egs4C8Air.inl>
#include <corsika/kokkos/em/algorithms/EmThinning.hpp>

namespace corsika::kokkos::egs4::c8_adapter {
  using thinning_config_type = ::corsika::kokkos::EmThinningConfig;
  using thinning_result_type = ::corsika::kokkos::EmThinningResult;
  using thinning_status_type = ::corsika::kokkos::EmThinningStatus;
  inline constexpr std::uint32_t NativeEgs4ThinningDomain = 0x43475448U;
  // Only ordinary two-EM-child vertices, exactly the C8 EMThinning scope.
  // Cut positron annihilation (channel invalid) and nuclear decays are NOT
  // silently thinned here. Output for the preceding segment keeps parent weight.
  KOKKOS_INLINE_FUNCTION thinning_result_type thin_air_vertex(thinning_config_type config,
                                                              AirRecord const& parent,
                                                              AirOutcome const& result) {
    thinning_result_type keep{thinning_status_type::NotApplied, 3,
                              parent.metadata_.weight_, parent.metadata_.weight_};
    if (!config.enabled_ || result.action_ != AirAction::replaced ||
        result.history_.childCount_ != 2 || result.history_.channel_ == Channel::invalid)
      return keep;
    // Each physical vertex consumes EM draws. Its final draw counter is a
    // collision nonce in a separate domain, not another EM draw. No fixed draw
    // offset can collide with an arbitrarily long rejection-sampling stream.
    auto key = parent.random_.key_;
    key.processId_ = NativeEgs4ThinningDomain;
    key.stepId_ = key.drawId_;
    key.drawId_ = 0;
    double u = ::corsika::kokkos::uniform_open01(key);
    ++key.drawId_;
    double v = ::corsika::kokkos::uniform_open01(key);
    return ::corsika::kokkos::apply_em_thinning(
        config, parent.track_.particle_.energyMeV_ * .001, parent.metadata_.weight_,
        result.children_[0].particle_.energyMeV_ * .001,
        result.children_[1].particle_.energyMeV_ * .001, u, v);
  }
  inline void validate_thinning(thinning_config_type const& c) {
    if (c.enabled_ && (!std::isfinite(c.thresholdGeV_) || c.thresholdGeV_ <= 0. ||
                       !std::isfinite(c.maximumWeight_) || c.maximumWeight_ <= 0. ||
                       !c.eraseZeroWeight_))
      throw std::invalid_argument(
          "Native EGS4 thinning requires positive finite threshold/weight and removal of "
          "zero-weight histories; multithinning is not supported");
  }
} // namespace corsika::kokkos::egs4::c8_adapter

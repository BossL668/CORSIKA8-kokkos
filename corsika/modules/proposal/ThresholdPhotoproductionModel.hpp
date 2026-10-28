/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <stdexcept>
#include <tuple>

#include <corsika/framework/core/EnergyMomentumOperations.hpp>
#include <corsika/framework/core/Logging.hpp>
#include <corsika/framework/core/ParticleProperties.hpp>
#include <corsika/framework/core/PhysicalUnits.hpp>
#include <corsika/framework/geometry/FourVector.hpp>
#include <corsika/framework/random/RNGManager.hpp>
#include <corsika/framework/utility/COMBoost.hpp>
#include <corsika/modules/sophia/ThresholdKinematics.hpp>

namespace corsika::proposal {

  struct ThresholdPhotoproductionStatistics {
    std::uint64_t interactions_{};
  };

  /**
   * Close the narrow physical gap between PROPOSAL's real-photon production
   * threshold and SOPHIA's hard lower centre-of-mass limit.
   *
   * PROPOSAL 7.6.2's Rhode cross section becomes non-zero at its pion
   * production cutoff. SOPHIA then recomputes s with its internal Fortran
   * nucleon masses before applying s >= 1.1646 GeV^2. In the intervening
   * interval a selected photoproduction vertex must not be discarded. The
   * only model-independent inelastic two-body channel open throughout that
   * interval is gamma + N -> pi0 + N. This adapter samples that final state
   * isotropically in the centre-of-momentum frame and performs an exact
   * two-body Lorentz transformation back to the lab.
   *
   * It is deliberately valid only below SOPHIA's lower bound and above the
   * CORSIKA pi0+nucleon mass threshold. Any wider unsupported region remains
   * a hard error instead of silently extending this approximation.
   */
  class ThresholdPhotoproductionModel {
  public:
    bool isValid(Code projectile, Code target, HEPEnergyType sqrt_s) const;

    template <typename TSecondaryView>
    void doInteraction(TSecondaryView& view, Code projectile, Code target,
                       FourMomentum const& projectile_p4, FourMomentum const& target_p4);

    ThresholdPhotoproductionStatistics const& statistics() const noexcept;

  private:
    default_prng_type& rng_ = RNGManager<>::getInstance().getRandomStream("proposal");
    ThresholdPhotoproductionStatistics statistics_{};
  };

} // namespace corsika::proposal

#include <corsika/detail/modules/proposal/ThresholdPhotoproductionModel.inl>

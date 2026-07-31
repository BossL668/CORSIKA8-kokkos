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
    std::uint64_t interactions{};
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
    bool isValid(
        Code projectile, Code target,
        HEPEnergyType sqrt_s) const {
      if (projectile != Code::Photon ||
          (target != Code::Proton &&
           target != Code::Neutron)) {
        return false;
      }
      auto const threshold =
          get_mass(target) + get_mass(Code::Pi0);
      return sqrt_s >= threshold &&
             sqrt_s <
                 sophia::minimumCorsikaComEnergy(target);
    }

    template <typename TSecondaryView>
    void doInteraction(
        TSecondaryView& view, Code projectile,
        Code target, FourMomentum const& projectile_p4,
        FourMomentum const& target_p4) {
      auto const total_p4 = projectile_p4 + target_p4;
      auto const sqrt_s = total_p4.getNorm();
      if (!isValid(projectile, target, sqrt_s)) {
        throw std::runtime_error(
            "threshold photoproduction model received an unsupported "
            "projectile/target/energy configuration");
      }

      auto const s = total_p4.getNormSqr();
      auto const target_mass = get_mass(target);
      auto const pion_mass = get_mass(Code::Pi0);
      auto const mass_sum = target_mass + pion_mass;
      auto const mass_difference = target_mass - pion_mass;
      auto momentum_squared =
          ((s - mass_sum * mass_sum) *
           (s - mass_difference * mass_difference)) /
          (4. * s);
      auto const roundoff_scale = s;
      if (momentum_squared < 0_eV * 0_eV &&
          std::abs(momentum_squared / roundoff_scale) <=
              128. * std::numeric_limits<double>::epsilon()) {
        momentum_squared = 0_eV * 0_eV;
      }
      if (momentum_squared < 0_eV * 0_eV) {
        throw std::runtime_error(
            "threshold photoproduction has negative two-body phase space");
      }
      auto const momentum_norm = sqrt(momentum_squared);

      std::uniform_real_distribution<double> uniform(0., 1.);
      auto const cos_theta = 2. * uniform(rng_) - 1.;
      auto const sin_theta =
          std::sqrt(std::max(0., 1. - cos_theta * cos_theta));
      auto const phi = 2. * M_PI * uniform(rng_);

      COMBoost const boost(projectile_p4, target_p4);
      auto const& com_cs = boost.getRotatedCS();
      MomentumVector const pion_momentum_com(
          com_cs,
          {momentum_norm * sin_theta * std::cos(phi),
           momentum_norm * sin_theta * std::sin(phi),
           momentum_norm * cos_theta});
      MomentumVector const nucleon_momentum_com =
          -pion_momentum_com;

      auto const pion_p4_lab = boost.fromCoM(
          FourMomentum{
              sqrt(momentum_squared + pion_mass * pion_mass),
              pion_momentum_com});
      auto const nucleon_p4_lab = boost.fromCoM(
          FourMomentum{
              sqrt(momentum_squared +
                   target_mass * target_mass),
              nucleon_momentum_com});

      auto add_secondary =
          [&](Code code, FourMomentum const& p4) {
            auto const momentum =
                p4.getSpaceLikeComponents();
            auto const energy =
                sqrt(momentum.getSquaredNorm() +
                     get_mass(code) * get_mass(code));
            if (momentum.getNorm() == 0_eV) {
              throw std::runtime_error(
                  "threshold photoproduction produced a zero lab momentum");
            }
            view.addSecondary(std::make_tuple(
                code, energy - get_mass(code),
                momentum.normalized()));
          };
      add_secondary(Code::Pi0, pion_p4_lab);
      add_secondary(target, nucleon_p4_lab);

      ++statistics_.interactions;
      CORSIKA_LOG_WARN(
          "PROPOSAL near-threshold photoproduction uses the explicit "
          "gamma+N->pi0+N two-body fallback: target={}, sqrt(S)={} GeV, "
          "fallback count={}",
          target, sqrt_s / 1_GeV,
          statistics_.interactions);
    }

    ThresholdPhotoproductionStatistics const&
    statistics() const noexcept {
      return statistics_;
    }

  private:
    default_prng_type& rng_ =
        RNGManager<>::getInstance().getRandomStream("proposal");
    ThresholdPhotoproductionStatistics statistics_{};
  };

} // namespace corsika::proposal

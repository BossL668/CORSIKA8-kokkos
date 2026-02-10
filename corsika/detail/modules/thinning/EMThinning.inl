/*
 * (c) Copyright 2022 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/framework/core/PhysicalUnits.hpp>
#include <corsika/framework/process/SecondariesProcess.hpp>

namespace corsika {

  EMThinning::EMThinning(HEPEnergyType threshold, double maxWeight,
                         bool const eraseParticles)
      : threshold_{threshold}
      , maxWeight_{maxWeight}
      , eraseParticles_{eraseParticles} {}

  template <typename TStackView>
  void EMThinning::doSecondaries(TStackView& view) {
    if (view.getEntries() != 2) return;

    auto projectile = view.getProjectile();
    if (!is_em(projectile.getPID())) return;

    double const parentWeight = projectile.getWeight();
    if (parentWeight >= maxWeight_) { return; }

    auto particle1 = view.begin();
    auto particle2 = ++(view.begin());

    // skip non-EM interactions
    if (!is_em(particle1.getPID()) || !is_em(particle2.getPID())) { return; }

    /* skip high-energy interactions
     * We consider only the primary energy. A more sophisticated algorithm might
     * sort out above-threshold secondaries and apply thinning only on the subset of
     * those below the threshold.
     */
    if (projectile.getEnergy() > threshold_) { return; }

    corsika::HEPEnergyType const Esum = particle1.getEnergy() + particle2.getEnergy();

    double const p1 = particle1.getEnergy() / Esum;
    double const p2 = particle2.getEnergy() / Esum;

    // weight factors
    auto const f1 = 1 / p1;
    auto const f2 = 1 / p2;

    // max. allowed weight factor
    double const fMax = maxWeight_ / parentWeight;

    if (f1 <= fMax && f2 <= fMax) {
      /* cannot possibly reach wmax in this vertex; apply
        Hillas thinning; select one of the two */
      if (uniform_(rng_) <= p1) { // keep 1st with probability p1
        particle2.setWeight(0);
        particle1.setWeight(parentWeight * f1);
      } else { // keep 2nd
        particle1.setWeight(0);
        particle2.setWeight(parentWeight * f2);
      }
    } else { // weight limitation kicks in, do statistical thinning
      double const f1prime = std::min(f1, fMax);
      double const f2prime = std::min(f2, fMax);

      if (uniform_(rng_) * f1prime <= 1) { // keep 1st
        particle1.setWeight(parentWeight * f1prime);
      } else {
        particle1.setWeight(0);
      }
      if (uniform_(rng_) * f2prime <= 1) { // keep 2nd
        particle2.setWeight(parentWeight * f2prime);
      } else {
        particle2.setWeight(0);
      }
    }

    // erase discared particles in case of multithinning
    if (eraseParticles_) {
      for (auto& p : view) {
        if (auto const w = p.getWeight(); w == 0) { p.erase(); }
      }
    } else {
      return;
    }
  }

} // namespace corsika

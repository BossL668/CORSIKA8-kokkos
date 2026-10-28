/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

namespace corsika::proposal {

  inline NativeInteractionCalculatorView::NativeInteractionCalculatorView() {}

  inline NativeInteractionCalculatorView::NativeInteractionCalculatorView(
      Code projectile, std::size_t mediumHash, medium_reference_type medium,
      interaction_reference_type interaction, photon_pair_reference_type photonPairLpm,
      brems_reference_type bremsLpm, HEPEnergyType stochasticEnergyCut,
      double particleMassMeV)
      : projectile_(projectile)
      , mediumHash_(mediumHash)
      , medium_(medium ? &medium->get() : nullptr)
      , interaction_(interaction ? &interaction->get() : nullptr)
      , photonPairLpm_(photonPairLpm ? &photonPairLpm->get() : nullptr)
      , bremsLpm_(bremsLpm ? &bremsLpm->get() : nullptr)
      , stochasticEnergyCut_(stochasticEnergyCut)
      , particleMassMeV_(particleMassMeV) {}

  inline Code NativeInteractionCalculatorView::getProjectile() const {
    return projectile_;
  }

  inline std::size_t NativeInteractionCalculatorView::getMediumHash() const {
    return mediumHash_;
  }

  inline NativeInteractionCalculatorView::medium_reference_type
  NativeInteractionCalculatorView::getMedium() const {
    if (medium_ == nullptr) { return {}; }
    return std::cref(*medium_);
  }

  inline NativeInteractionCalculatorView::interaction_reference_type
  NativeInteractionCalculatorView::getInteraction() const {
    if (interaction_ == nullptr) { return {}; }
    return std::cref(*interaction_);
  }

  inline NativeInteractionCalculatorView::photon_pair_reference_type
  NativeInteractionCalculatorView::getPhotonPairLpm() const {
    if (photonPairLpm_ == nullptr) { return {}; }
    return std::cref(*photonPairLpm_);
  }

  inline NativeInteractionCalculatorView::brems_reference_type
  NativeInteractionCalculatorView::getBremsLpm() const {
    if (bremsLpm_ == nullptr) { return {}; }
    return std::cref(*bremsLpm_);
  }

  inline HEPEnergyType NativeInteractionCalculatorView::getStochasticEnergyCut() const {
    return stochasticEnergyCut_;
  }

  inline double NativeInteractionCalculatorView::getParticleMassMeV() const {
    return particleMassMeV_;
  }

  inline NativeContinuousCalculatorView::NativeContinuousCalculatorView() {}

  inline NativeContinuousCalculatorView::NativeContinuousCalculatorView(
      Code projectile, std::size_t mediumHash, medium_reference_type medium,
      displacement_reference_type displacement, HEPEnergyType stochasticEnergyCut,
      double particleMassMeV)
      : projectile_(projectile)
      , mediumHash_(mediumHash)
      , medium_(medium ? &medium->get() : nullptr)
      , displacement_(displacement ? &displacement->get() : nullptr)
      , stochasticEnergyCut_(stochasticEnergyCut)
      , particleMassMeV_(particleMassMeV) {}

  inline Code NativeContinuousCalculatorView::getProjectile() const {
    return projectile_;
  }

  inline std::size_t NativeContinuousCalculatorView::getMediumHash() const {
    return mediumHash_;
  }

  inline NativeContinuousCalculatorView::medium_reference_type
  NativeContinuousCalculatorView::getMedium() const {
    if (medium_ == nullptr) { return {}; }
    return std::cref(*medium_);
  }

  inline NativeContinuousCalculatorView::displacement_reference_type
  NativeContinuousCalculatorView::getDisplacement() const {
    if (displacement_ == nullptr) { return {}; }
    return std::cref(*displacement_);
  }

  inline HEPEnergyType NativeContinuousCalculatorView::getStochasticEnergyCut() const {
    return stochasticEnergyCut_;
  }

  inline double NativeContinuousCalculatorView::getParticleMassMeV() const {
    return particleMassMeV_;
  }

} // namespace corsika::proposal

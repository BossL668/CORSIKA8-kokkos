/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/framework/core/ParticleProperties.hpp>
#include <corsika/framework/core/PhysicalUnits.hpp>

#include <cstddef>
#include <functional>
#include <optional>

namespace PROPOSAL {
  class Displacement;
  class Interaction;
  class Medium;
  namespace crosssection {
    class PhotoPairLPM;
    class BremsLPM;
  } // namespace crosssection
} // namespace PROPOSAL

namespace corsika::proposal {

  /**
   * Non-owning read-only access to an existing stochastic calculator.
   *
   * References remain valid only while the scalar owner and its calculators remain
   * alive. Empty references preserve the original exporter failure contract; neither
   * construction nor these host-only accessors validate or evaluate the calculator.
   */
  class NativeInteractionCalculatorView {
  public:
    using medium_reference_type =
        std::optional<std::reference_wrapper<PROPOSAL::Medium const>>;
    using interaction_reference_type =
        std::optional<std::reference_wrapper<PROPOSAL::Interaction const>>;
    using photon_pair_reference_type =
        std::optional<std::reference_wrapper<PROPOSAL::crosssection::PhotoPairLPM const>>;
    using brems_reference_type =
        std::optional<std::reference_wrapper<PROPOSAL::crosssection::BremsLPM const>>;

    /** Construct an empty view, without allocating or constructing calculators. */
    NativeInteractionCalculatorView();

    /** Borrow calculator state without ownership transfer or input validation. */
    NativeInteractionCalculatorView(Code projectile, std::size_t mediumHash,
                                    medium_reference_type medium,
                                    interaction_reference_type interaction,
                                    photon_pair_reference_type photonPairLpm,
                                    brems_reference_type bremsLpm,
                                    HEPEnergyType stochasticEnergyCut,
                                    double particleMassMeV);

    /** Return the stored projectile value without evaluating the model. */
    [[nodiscard]] Code getProjectile() const;

    /** Return the stored mediumHash value without evaluating the model. */
    [[nodiscard]] std::size_t getMediumHash() const;

    /** Return the stored medium value without evaluating the model. */
    [[nodiscard]] medium_reference_type getMedium() const;

    /** Return the stored interaction value without evaluating the model. */
    [[nodiscard]] interaction_reference_type getInteraction() const;

    /** Return the stored photonPairLpm value without evaluating the model. */
    [[nodiscard]] photon_pair_reference_type getPhotonPairLpm() const;

    /** Return the stored bremsLpm value without evaluating the model. */
    [[nodiscard]] brems_reference_type getBremsLpm() const;

    /** Return the stored stochasticEnergyCut value without evaluating the model. */
    [[nodiscard]] HEPEnergyType getStochasticEnergyCut() const;

    /** Return the stored particleMassMeV value without evaluating the model. */
    [[nodiscard]] double getParticleMassMeV() const;

  private:
    Code projectile_{Code::Unknown};
    std::size_t mediumHash_{};
    PROPOSAL::Medium const* medium_{};
    PROPOSAL::Interaction const* interaction_{};
    PROPOSAL::crosssection::PhotoPairLPM const* photonPairLpm_{};
    PROPOSAL::crosssection::BremsLPM const* bremsLpm_{};
    HEPEnergyType stochasticEnergyCut_{};
    double particleMassMeV_{};
  };

  /**
   * Non-owning read-only access to an existing continuous-loss calculator.
   *
   * References remain valid only while the scalar owner and its calculators remain
   * alive. Empty references preserve the original exporter failure contract; neither
   * construction nor these host-only accessors validate or evaluate the calculator.
   */
  class NativeContinuousCalculatorView {
  public:
    using medium_reference_type =
        std::optional<std::reference_wrapper<PROPOSAL::Medium const>>;
    using displacement_reference_type =
        std::optional<std::reference_wrapper<PROPOSAL::Displacement const>>;

    /** Construct an empty view, without allocating or constructing calculators. */
    NativeContinuousCalculatorView();

    /** Borrow calculator state without ownership transfer or input validation. */
    NativeContinuousCalculatorView(Code projectile, std::size_t mediumHash,
                                   medium_reference_type medium,
                                   displacement_reference_type displacement,
                                   HEPEnergyType stochasticEnergyCut,
                                   double particleMassMeV);

    /** Return the stored projectile value without evaluating the model. */
    [[nodiscard]] Code getProjectile() const;

    /** Return the stored mediumHash value without evaluating the model. */
    [[nodiscard]] std::size_t getMediumHash() const;

    /** Return the stored medium value without evaluating the model. */
    [[nodiscard]] medium_reference_type getMedium() const;

    /** Return the stored displacement value without evaluating the model. */
    [[nodiscard]] displacement_reference_type getDisplacement() const;

    /** Return the stored stochasticEnergyCut value without evaluating the model. */
    [[nodiscard]] HEPEnergyType getStochasticEnergyCut() const;

    /** Return the stored particleMassMeV value without evaluating the model. */
    [[nodiscard]] double getParticleMassMeV() const;

  private:
    Code projectile_{Code::Unknown};
    std::size_t mediumHash_{};
    PROPOSAL::Medium const* medium_{};
    PROPOSAL::Displacement const* displacement_{};
    HEPEnergyType stochasticEnergyCut_{};
    double particleMassMeV_{};
  };

} // namespace corsika::proposal

#include <corsika/detail/modules/proposal/NativeCalculatorView.inl>

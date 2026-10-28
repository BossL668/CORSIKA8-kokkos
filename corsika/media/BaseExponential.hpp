/*
 * (c) Copyright 2020 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/framework/core/ParticleProperties.hpp>
#include <corsika/framework/core/PhysicalUnits.hpp>
#include <corsika/framework/geometry/Line.hpp>
#include <corsika/framework/geometry/Point.hpp>
#include <corsika/framework/geometry/BaseTrajectory.hpp>
#include <limits>

/**
 * @file corsika/media/BaseExponential.hpp
 */

namespace corsika {

  /**
   * This class provides the grammage/length conversion functionality for
   * (locally) flat exponential atmospheres.
   *
   * The density is described according to \f[ \varrho() \f]
   */
  template <typename TDerived>
  class BaseExponential {

  public:
    BaseExponential(Point const& point, LengthType const referenceHeight,
                    MassDensityType const rho0, LengthType const lambda);

    Point const& getAnchorPoint() const { return point_; }

    /// Return the stored reference density without evaluating the density law.
    [[nodiscard]] MassDensityType getReferenceDensity() const;

    /// Return the signed exponential scale length with its original units.
    [[nodiscard]] LengthType getScaleLength() const;

    /// Return the stored height at which the reference density is defined.
    [[nodiscard]] LengthType getReferenceHeight() const;

  protected:
    auto const& getImplementation() const;

    /**
     * Returns the mass density at altitude "height".
     *
     * @param height
     * @return MassDensityType
     */
    MassDensityType getMassDensity(LengthType const height) const;

    // clang-format off
    /**
     * For a (normalized) axis \f$ \vec{a} \f$, the grammage along a non-orthogonal line with (normalized)
     * direction \f$ \vec{u} \f$ is given by:
     * \f[
     *   X = \frac{\varrho_0 \lambda}{\vec{u} \cdot \vec{a}} \left( \exp\left( \vec{u} \cdot \vec{a} \frac{l}{\lambda} \right) - 1 \right) \quad \text{,}
     * \f]
     * where \f$ \varrho_0 \f$ is the density at the starting point.
     *
     * If \f$ \vec{u} \cdot \vec{a} = 0 \f$, the calculation is just like with a homogeneous density:
     * \f[
     *   X = \varrho_0 l
     * \f]
     */
    // clang-format on
    GrammageType getIntegratedGrammage(BaseTrajectory const& line,
                                       DirectionVector const& axis) const;

    // clang-format off
    /**
     * For a (normalized) axis \f$ \vec{a} \f$, the length of a non-orthogonal line with (normalized)
     * direction \f$ \vec{u} \f$ corresponding to grammage \f$ X \f$ is given by:
     * \f[
     *   l = \begin{cases}
     *   \frac{\lambda}{\vec{u} \cdot \vec{a}} \log\left(Y \right), & \text{if} & Y := 1 +
     *     \vec{u} \cdot \vec{a} \frac{X}{\rho_0 \lambda} > 0 \\
     *   \infty & \text{else} & \text{,}
     *   \end{cases}
     * \f] where \f$ \varrho_0 \f$ is the density at the starting point.
     *
     * If \f$ \vec{u} \cdot \vec{a} = 0 \f$, the calculation is just like for a homogeneous density:
     * \f[
     *   l =  \frac{X}{\varrho_0}
     * \f]
     */
    // clang-format on
    LengthType getArclengthFromGrammage(BaseTrajectory const& line,
                                        GrammageType const grammage,
                                        DirectionVector const& axis) const;

  private:
    MassDensityType const rho0_;
    LengthType const lambda_;
    InverseLengthType const invLambda_;
    Point const point_;
    LengthType const referenceHeight_;

  }; // class BaseExponential

} // namespace corsika

#include <corsika/detail/media/BaseExponential.inl>

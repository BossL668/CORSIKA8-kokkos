/*
 * (c) Copyright 2020 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include <corsika/framework/core/PhysicalUnits.hpp>
#include <corsika/framework/core/Logging.hpp>

#include <string>

namespace corsika {

  template <typename TEnvModel>
  inline ShowerAxis::ShowerAxis(Point const& pStart, Vector<length_d> const& length,
                                Environment<TEnvModel> const& env, bool const doThrow,
                                int const steps)
      : pointStart_(pStart)
      , length_(length)
      , throw_(doThrow)
      , max_length_(length_.getNorm())
      , steplength_(max_length_ / steps)
      , axis_normalized_(length / max_length_)
      , X_(steps + 1) {
    auto const* const universe = env.getUniverse().get();

    auto rho = [pStart, length, universe, doThrow](double x) {
      auto const p = pStart + length * x;
      auto const* node = universe->getContainingNode(p);
      if (!node->hasModelProperties()) {
        CORSIKA_LOG_CRITICAL(
            "Unable to construct ShowerAxis. ShowerAxis includes volume "
            "with no model properties at point {}.",
            p);
        if (doThrow) throw std::runtime_error("Unable to construct ShowerAxis.");
      }
      return node->getModelProperties().getMassDensity(p).magnitude();
    };

    double error;
    int k = 0;
    X_[0] = GrammageType::zero();
    auto sum = GrammageType::zero();

    for (int i = 1; i <= steps; ++i) {
      auto const x_prev = (i - 1.) / steps;
      auto const d_prev = max_length_ * x_prev;
      auto const x = double(i) / steps;
      auto const r = boost::math::quadrature::gauss_kronrod<double, 15>::integrate(
          rho, x_prev, x, 15, 1e-9, &error);
      auto const result =
          MassDensityType(phys::units::detail::magnitude_tag, r) * max_length_;

      sum += result;
      X_[i] = sum;

      for (; sum > k * X_binning_; ++k) {
        d_.emplace_back(d_prev + k * X_binning_ * steplength_ / result);
      }
    }

    if (!std::is_sorted(X_.cbegin(), X_.cend()) ||
        !std::is_sorted(d_.cbegin(), d_.cend())) {
      CORSIKA_LOG_ERROR("Shower axis bins are not sorted");
      throw std::runtime_error("Bad ShowerAxis construction");
    }
  }

  inline GrammageType ShowerAxis::getX(LengthType l) const {
    double const fractionalBin = l / steplength_;
    int const lower = fractionalBin; // indices of nearest X support points
    double const fraction = fractionalBin - lower;
    unsigned int const upper = lower + 1;

    if (fractionalBin < 0) {
      CORSIKA_LOG_TRACE("cannot extrapolate to points behind point of injection l={} m",
                        l / 1_m);
      if (throw_) {
        throw std::runtime_error(
            "cannot extrapolate to points behind point of injection");
      }
      return getMinimumX();
    }

    if (upper >= X_.size()) {
      CORSIKA_LOG_TRACE(
          "shower axis too short, cannot extrapolate (l / max_length_ = {} )",
          l / max_length_);
      if (throw_) {
        const std::string err = fmt::format(
            "shower axis too short, cannot extrapolate (l / max_length_ = {} )",
            l / max_length_);
        throw std::runtime_error(err.c_str());
      }
      return getMaximumX();
    }
    CORSIKA_LOG_TRACE("showerAxis::X frac={}, fractionalBin={}, lower={}, upper={}",
                      fraction, fractionalBin, lower, upper);

    if (!(0 <= fraction && fraction <= 1.)) {
      CORSIKA_LOG_ERROR("Fraction must be 0 <= {} <= 1", fraction);
      throw std::runtime_error("Fraction out of range");
    }

    CORSIKA_LOG_TRACE("ShowerAxis::getX l={} m, lower={}, fraction={}, upper={}", l / 1_m,
                      lower, fraction, upper);

    // linear interpolation between getX[lower] and X[upper]
    return X_[upper] * fraction + X_[lower] * (1 - fraction);
  }

  inline LengthType ShowerAxis::getSteplength() const { return steplength_; }

  inline LengthType ShowerAxis::getMaximumLength() const { return max_length_; }

  inline GrammageType ShowerAxis::getMaximumX() const { return *X_.rbegin(); }

  inline GrammageType ShowerAxis::getMinimumX() const { return GrammageType::zero(); }

  inline GrammageType ShowerAxis::getProjectedX(Point const& p) const {
    auto const projectedLength = (p - pointStart_).dot(axis_normalized_);
    return getX(projectedLength);
  }

  inline DirectionVector const& ShowerAxis::getDirection() const {
    return axis_normalized_;
  }

  inline Point const& ShowerAxis::getStart() const { return pointStart_; }

  inline std::vector<GrammageType> const& ShowerAxis::getGrammageSupport() const {
    return X_;
  }

} // namespace corsika

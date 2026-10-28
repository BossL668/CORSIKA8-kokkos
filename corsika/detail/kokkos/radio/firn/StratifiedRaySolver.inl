/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

namespace corsika::kokkos::radio::firn {
  inline StratifiedRaySolver::StratifiedRaySolver(VerticalOpticalProfile const& profile)
      : profile_(profile) {}

  inline PathDiagnostics StratifiedRaySolver::nearEndpoint(vector_type const& source,
                                                           vector_type const& observer,
                                                           bool wantTurning) const {
    auto const* pointer = dynamic_cast<ExponentialProfile const*>(&profile_);
    if (!pointer) return {};
    auto const parameters = pointer->parameters();
    auto const& exponential = *pointer;
    if (!(parameters.deltaIndex_ > 0.0)) return {};
    double const deepZ = std::min(source[2], observer[2]);
    double const shallowZ = std::max(source[2], observer[2]);
    if (!(shallowZ > deepZ)) return {};
    double const horizontal = detail::horizontal_distance(source, observer);
    if (!(horizontal > 0.0)) return {};
    double const nd = profile_.refractiveIndexAtZ(deepZ);
    double const ns = profile_.refractiveIndexAtZ(shallowZ);
    if (!(nd > ns)) return {};
    // Cover both sides of the ordinary endpoint crossing without deleting
    // an interval of p. A horizontal endpoint is not a ray-tube caustic.
    double lower = wantTurning ? -ExponentialEndpointData::cosineLimit_ : 0.0;
    double upper = wantTurning ? 0.0 : ExponentialEndpointData::cosineLimit_;
    if (wantTurning) {
      double const nTop = profile_.refractiveIndexAtZ(profile_.maximumSupportedZ());
      // A turning point must stay below the upper boundary of this profile.
      double const surfaceCosine =
          std::sqrt(std::max(0.0, (ns - nTop) * (ns + nTop))) / ns;
      lower = std::max(lower, -surfaceCosine);
      if (!(lower < 0.0)) return {};
    }
    auto const left = exponential.endpointRayData(lower, deepZ, shallowZ);
    auto const right = exponential.endpointRayData(upper, deepZ, shallowZ);
    if (!(left.horizontalDerivativeM_ < 0.0 && right.horizontalDerivativeM_ < 0.0) ||
        horizontal > left.horizontalM_ || horizontal < right.horizontalM_ ||
        (wantTurning && horizontal == right.horizontalM_))
      return {};
    for (int iteration = 0; iteration < 80; ++iteration) {
      double const middle = 0.5 * (lower + upper);
      if (exponential.endpointRayData(middle, deepZ, shallowZ).horizontalM_ > horizontal)
        lower = middle;
      else
        upper = middle;
    }
    double const q = 0.5 * (lower + upper);
    auto const data = exponential.endpointRayData(q, deepZ, shallowZ);
    if (!(data.horizontalDerivativeM_ < 0.0)) return {};
    PathDiagnostics result;
    result.valid_ = true;
    result.topology_ = wantTurning ? "upper_turning_endpoint" : "direct";
    result.invariantP_ = data.invariantP_;
    result.sourceRefractiveIndex_ = profile_.refractiveIndexAtZ(source[2]);
    result.observerRefractiveIndex_ = profile_.refractiveIndexAtZ(observer[2]);
    result.geometricLengthM_ = data.geometricLengthM_;
    result.opticalPathM_ = data.opticalPathM_;
    result.travelTimeS_ = data.opticalPathM_ / c_m_per_s;
    result.attenuationExponent_ = parameters.attenuationPerM_ * data.geometricLengthM_;
    result.horizontalResidualM_ = data.horizontalM_ - horizontal;
    result.fermatRelativeResidual_ = std::abs(result.horizontalResidualM_) / horizontal;
    result.geometricSpreadingPerM_ =
        std::sqrt(ns / (nd * horizontal * data.deepCosine_ *
                        std::abs(data.horizontalDerivativeM_)));
    auto const horizontalUnit = detail::horizontal_unit(source, observer);
    bool const sourceIsDeep = source[2] < observer[2];
    result.emitDirection_ =
        add(multiply(horizontalUnit, data.invariantP_ / result.sourceRefractiveIndex_),
            vector_type{0.0, 0.0, sourceIsDeep ? data.deepCosine_ : -q});
    result.receiveDirection_ =
        add(multiply(horizontalUnit, data.invariantP_ / result.observerRefractiveIndex_),
            vector_type{0.0, 0.0, sourceIsDeep ? q : -data.deepCosine_});
    return result;
  }

  inline PathDiagnostics StratifiedRaySolver::direct(vector_type const& source,
                                                     vector_type const& observer,
                                                     double frequencyHz) const {
    if (!finite(source) || !finite(observer) || !std::isfinite(frequencyHz) ||
        frequencyHz < 0.0 || source[2] < profile_.minimumSupportedZ() ||
        source[2] > profile_.maximumSupportedZ() ||
        observer[2] < profile_.minimumSupportedZ() ||
        observer[2] > profile_.maximumSupportedZ()) {
      throw std::runtime_error("invalid ray query or endpoint outside optical profile");
    }
    double const horizontal = detail::horizontal_distance(source, observer);
    double const dz = observer[2] - source[2];
    double const straightLength = norm(subtract(observer, source));
    if (!(straightLength > 0.0)) return {};

    double const nSource = profile_.refractiveIndexAtZ(source[2]);
    double const nObserver = profile_.refractiveIndexAtZ(observer[2]);
    if (std::abs(dz) < 1.0e-12) {
      double const nMiddle = profile_.refractiveIndexAtZ(source[2]);
      double const probe = std::max(1.0e-4, 1.0e-7 * std::max(1.0, std::abs(source[2])));
      double lowerZ = std::max(profile_.minimumSupportedZ(), source[2] - probe);
      double upperZ = std::min(profile_.maximumSupportedZ(), source[2] + probe);
      bool const locallyUniform =
          upperZ > lowerZ && std::abs(profile_.refractiveIndexAtZ(upperZ) -
                                      profile_.refractiveIndexAtZ(lowerZ)) < 1.0e-12;
      if (!locallyUniform) {
        PathDiagnostics result;
        result.topology_ = "same_depth_requires_turning_or_fullwave";
        return result;
      }
      PathDiagnostics result;
      result.valid_ = true;
      result.topology_ = "horizontal_uniform_layer";
      result.invariantP_ = nMiddle;
      result.sourceRefractiveIndex_ = nMiddle;
      result.observerRefractiveIndex_ = nMiddle;
      result.geometricLengthM_ = horizontal;
      result.opticalPathM_ = nMiddle * horizontal;
      result.travelTimeS_ = result.opticalPathM_ / c_m_per_s;
      result.attenuationExponent_ =
          profile_.attenuationAtZ(source[2], frequencyHz) * horizontal;
      result.geometricSpreadingPerM_ = 1.0 / horizontal;
      vector_type const tangent = detail::horizontal_unit(source, observer);
      result.emitDirection_ = tangent;
      result.receiveDirection_ = tangent;
      return result;
    }

    double const lowZ = std::min(source[2], observer[2]);
    auto const endpoint = nearEndpoint(source, observer, false);
    if (endpoint.valid_) return endpoint;
    double const highZ = std::max(source[2], observer[2]);
    double const maximumU = std::sqrt(highZ - lowZ);
    auto regularizedIntegral =
        [&](double p, std::function<double(double, double)> const& numerator) {
          // z=highZ-u^2 removes the integrable square-root endpoint
          // singularity of grazing monotonic rays.  Direct fixed-z Simpson
          // quadrature creates a spurious large horizontal range and hence
          // false direct roots in the turning region.
          return detail::simpson(
              [&](double u) {
                if (u == 0.0) return 0.0;
                double const z = highZ - u * u;
                double const n = profile_.refractiveIndexAtZ(z);
                double const radicand = std::max(n * n - p * p, 1.0e-28);
                return 2.0 * u * numerator(z, n) / std::sqrt(radicand);
              },
              0.0, maximumU, 4096);
        };
    auto horizontalIntegral = [&](double p) {
      return regularizedIntegral(p, [p](double, double) { return p; });
    };

    double lower = 0.0;
    // Conservative bracket for the generic quadrature path. Exponential
    // endpoint rays, including the direct/turning junction, are handled above.
    double upper = std::min(nSource, nObserver) * (1.0 - 1.0e-7);
    double const maximumHorizontal = horizontalIntegral(upper);
    if (horizontal >= maximumHorizontal) {
      PathDiagnostics result;
      result.topology_ = "no_monotonic_direct_root";
      result.horizontalResidualM_ = horizontal - maximumHorizontal;
      return result;
    }
    for (int iteration = 0; iteration < 100; ++iteration) {
      double const middle = 0.5 * (lower + upper);
      if (horizontalIntegral(middle) < horizontal) {
        lower = middle;
      } else {
        upper = middle;
      }
    }
    double p = lower;
    double recoveredHorizontal = horizontalIntegral(p);
    for (double const candidate : {upper, 0.5 * (lower + upper)}) {
      double const candidateHorizontal = horizontalIntegral(candidate);
      if (std::abs(candidateHorizontal - horizontal) <
          std::abs(recoveredHorizontal - horizontal)) {
        p = candidate;
        recoveredHorizontal = candidateHorizontal;
      }
    }
    auto integral = [&](std::function<double(double, double)> const& integrand) {
      return regularizedIntegral(p, integrand);
    };
    double const geometricLength = integral([](double, double n) { return n; });
    double const opticalPath = integral([](double, double n) { return n * n; });
    double const attenuation = integral(
        [&](double z, double n) { return profile_.attenuationAtZ(z, frequencyHz) * n; });

    double const signZ = dz > 0.0 ? 1.0 : -1.0;
    vector_type const horizontal_unit = detail::horizontal_unit(source, observer);
    auto tangent = [&](double n) {
      double const sine = std::clamp(p / n, 0.0, 1.0);
      double const cosine = std::sqrt(std::max(0.0, 1.0 - sine * sine));
      return add(multiply(horizontal_unit, sine), {0.0, 0.0, signZ * cosine});
    };
    vector_type const emit = tangent(nSource);
    vector_type const receive = tangent(nObserver);
    double const sinSource = p / nSource;
    double const cosSource = std::abs(emit[2]);
    double const cosObserver = std::abs(receive[2]);
    double const dHorizontalDp = detail::simpson(
        [&](double u) {
          if (u == 0.0) return 0.0;
          double const z = highZ - u * u;
          double const n = profile_.refractiveIndexAtZ(z);
          double const denominator = std::pow(std::max(n * n - p * p, 1.0e-28), 1.5);
          return 2.0 * u * n * n / denominator;
        },
        0.0, maximumU, 4096);
    double rayTubeJacobian = 0.0;
    if (horizontal > 1.0e-12 && p > 1.0e-12 && sinSource > 0.0) {
      double const dHorizontalDThetaSource = dHorizontalDp * nSource * cosSource;
      rayTubeJacobian = horizontal * cosObserver * dHorizontalDThetaSource / sinSource;
    } else {
      double const inverseIndexIntegral = std::abs(
          detail::simpson([&](double z) { return 1.0 / profile_.refractiveIndexAtZ(z); },
                          source[2], observer[2]));
      rayTubeJacobian = std::pow(nSource * inverseIndexIntegral, 2.0);
    }
    if (!(rayTubeJacobian > 0.0) || !std::isfinite(rayTubeJacobian)) {
      PathDiagnostics result;
      result.topology_ = "invalid_ray_tube_jacobian";
      return result;
    }
    double const geometricSpreading = std::sqrt(nSource / (nObserver * rayTubeJacobian));

    PathDiagnostics result;
    result.valid_ = true;
    result.topology_ = "direct";
    result.invariantP_ = p;
    result.sourceRefractiveIndex_ = nSource;
    result.observerRefractiveIndex_ = nObserver;
    result.geometricLengthM_ = geometricLength;
    result.opticalPathM_ = opticalPath;
    result.travelTimeS_ = opticalPath / c_m_per_s;
    result.attenuationExponent_ = attenuation;
    result.horizontalResidualM_ = recoveredHorizontal - horizontal;
    result.fermatRelativeResidual_ =
        std::abs(result.horizontalResidualM_) / std::max(horizontal, 1.0e-300);
    result.snellResidual_ =
        std::max(std::abs(nSource * std::hypot(emit[0], emit[1]) - p),
                 std::abs(nObserver * std::hypot(receive[0], receive[1]) - p));
    result.geometricSpreadingPerM_ = geometricSpreading;
    result.emitDirection_ = emit;
    result.receiveDirection_ = receive;
    return result;
  }

  inline std::vector<PathDiagnostics> StratifiedRaySolver::turningBranches(
      vector_type const& source, vector_type const& observer, double frequencyHz) const {
    if (!finite(source) || !finite(observer) || !std::isfinite(frequencyHz) ||
        frequencyHz < 0.0 || source[2] < profile_.minimumSupportedZ() ||
        source[2] > profile_.maximumSupportedZ() ||
        observer[2] < profile_.minimumSupportedZ() ||
        observer[2] > profile_.maximumSupportedZ()) {
      throw std::runtime_error("invalid ray query or endpoint outside optical profile");
    }
    double const horizontal = detail::horizontal_distance(source, observer);
    if (!(horizontal > 0.0)) return {};
    double const topZ = profile_.maximumSupportedZ();
    double const shallowEndpointZ = std::max(source[2], observer[2]);
    if (!(topZ > shallowEndpointZ)) return {};
    double const nTop = profile_.refractiveIndexAtZ(topZ);
    double const nSource = profile_.refractiveIndexAtZ(source[2]);
    double const nObserver = profile_.refractiveIndexAtZ(observer[2]);
    double const nUpper = std::min(nSource, nObserver);
    if (!(nUpper > nTop * (1.0 + 1.0e-12))) return {};

    auto turningZ = [&](double p) {
      double low = shallowEndpointZ;
      double high = topZ;
      if (!(profile_.refractiveIndexAtZ(low) >= p) ||
          !(profile_.refractiveIndexAtZ(high) <= p)) {
        throw std::runtime_error("profile is not monotonic toward its upper bound");
      }
      for (int iteration = 0; iteration < 100; ++iteration) {
        double const middle = 0.5 * (low + high);
        if (profile_.refractiveIndexAtZ(middle) > p)
          low = middle;
        else
          high = middle;
      }
      return 0.5 * (low + high);
    };
    auto legIntegral = [&](double endpointZ, double turnZ, double p,
                           std::function<double(double, double)> const& f) {
      double const upperU = std::sqrt(std::max(0.0, turnZ - endpointZ));
      if (!(upperU > 0.0)) return 0.0;
      double const derivativeStep = std::min(1.0e-4, 0.25 * (topZ - shallowEndpointZ));
      double const derivativeLow = turnZ - derivativeStep;
      double const derivativeHigh = std::min(topZ, turnZ + derivativeStep);
      double const nDerivative = (profile_.refractiveIndexAtZ(derivativeHigh) -
                                  profile_.refractiveIndexAtZ(derivativeLow)) /
                                 (derivativeHigh - derivativeLow);
      double const endpointLimit =
          2.0 * f(turnZ, p) /
          std::sqrt(std::max(2.0 * p * std::abs(nDerivative), 1.0e-30));
      return detail::simpson(
          [&](double u) {
            if (u == 0.0) return endpointLimit;
            double const z = turnZ - u * u;
            double const n = profile_.refractiveIndexAtZ(z);
            return 2.0 * u * f(z, n) / std::sqrt(std::max(n * n - p * p, 1.0e-30));
          },
          0.0, upperU, 2048);
    };
    auto const* exponential = dynamic_cast<ExponentialProfile const*>(&profile_);
    auto horizontalAt = [&](double p) {
      if (exponential) {
        return exponential->turningLegHorizontal(p, source[2]) +
               exponential->turningLegHorizontal(p, observer[2]);
      }
      double const zt = turningZ(p);
      auto integrand = [p](double, double) { return p; };
      return legIntegral(source[2], zt, p, integrand) +
             legIntegral(observer[2], zt, p, integrand);
    };

    double const pLow = nTop * (1.0 + 1.0e-10);
    double const pHigh = nUpper * (1.0 - 1.0e-10);
    constexpr std::size_t samples = 512;
    std::vector<double> roots;
    // A fixed sampling grid can miss two roots on opposite sides of a
    // maximum within one cell. Bracket that stationary point first.
    auto derivativeAt = [&](double p) {
      return exponential->turningLegHorizontalDerivative(p, source[2]) +
             exponential->turningLegHorizontalDerivative(p, observer[2]);
    };
    auto addRoot = [&](double low, double high) {
      double lowValue = horizontalAt(low) - horizontal;
      double highValue = horizontalAt(high) - horizontal;
      if (lowValue * highValue > 0.0) return;
      if (lowValue == 0.0)
        high = low;
      else if (highValue == 0.0)
        low = high;
      for (int iteration = 0; iteration < 90; ++iteration) {
        double const middle = 0.5 * (low + high);
        if (middle == low || middle == high) break;
        double const middleValue = horizontalAt(middle) - horizontal;
        if (lowValue * middleValue <= 0.0)
          high = middle;
        else {
          low = middle;
          lowValue = middleValue;
        }
      }
      double const root = 0.5 * (low + high);
      if (roots.empty() || std::abs(root - roots.back()) > 1.0e-14) roots.push_back(root);
    };
    double previousP = pLow;
    double previousDerivative = exponential ? derivativeAt(previousP) : 0.0;
    for (std::size_t index = 1; index <= samples; ++index) {
      double const p = pLow + (static_cast<double>(index) / samples) * (pHigh - pLow);
      double const derivative = exponential ? derivativeAt(p) : 0.0;
      if (previousDerivative > 0.0 && derivative < 0.0) {
        double low = previousP, high = p;
        for (int iteration = 0; iteration < 70; ++iteration) {
          double const middle = 0.5 * (low + high);
          if (middle == low || middle == high) break;
          if (derivativeAt(middle) > 0.0)
            low = middle;
          else
            high = middle;
        }
        double const peak = 0.5 * (low + high);
        if (std::abs(horizontalAt(peak) - horizontal) <= 1.0e-11 * horizontal) {
          throw std::runtime_error(
              "unresolved ray-tube caustic requires full-wave response");
        }
        addRoot(previousP, peak);
        addRoot(peak, p);
      } else {
        addRoot(previousP, p);
      }
      previousP = p;
      previousDerivative = derivative;
    }

    vector_type const horizontal_unit = detail::horizontal_unit(source, observer);
    std::vector<PathDiagnostics> result;
    auto const endpoint = nearEndpoint(source, observer, true);
    if (endpoint.valid_) result.push_back(endpoint);
    for (std::size_t branch = 0; branch < roots.size(); ++branch) {
      double const p = roots[branch];
      if (endpoint.valid_ &&
          p >= nUpper * std::sqrt(1.0 - ExponentialEndpointData::cosineLimit_ *
                                            ExponentialEndpointData::cosineLimit_))
        continue;
      double const zt = turningZ(p);
      auto sumLegs = [&](std::function<double(double, double)> const& f) {
        return legIntegral(source[2], zt, p, f) + legIntegral(observer[2], zt, p, f);
      };
      double const geometric = sumLegs([](double, double n) { return n; });
      double const optical = sumLegs([](double, double n) { return n * n; });
      double const attenuation = sumLegs([&](double z, double n) {
        return profile_.attenuationAtZ(z, frequencyHz) * n;
      });
      double dHorizontalDp;
      if (auto const* exponential = dynamic_cast<ExponentialProfile const*>(&profile_)) {
        dHorizontalDp = exponential->turningLegHorizontalDerivative(p, source[2]) +
                        exponential->turningLegHorizontalDerivative(p, observer[2]);
      } else {
        // Preserve support for other profiles. Keep the derivative stencil
        // strictly inside the available branch, especially near grazing.
        double const dp = std::min(1.0e-5 * p, 0.01 * std::min(p - pLow, pHigh - p));
        if (!(dp > 0.0) || p + dp == p || p - dp == p) {
          throw std::runtime_error("unresolved turning-ray derivative");
        }
        dHorizontalDp = (horizontalAt(p + dp) - horizontalAt(p - dp)) / (2.0 * dp);
      }
      double const sinSource = p / nSource;
      double const sinObserver = p / nObserver;
      double const cosSource = std::sqrt(std::max(0.0, 1.0 - sinSource * sinSource));
      double const cosObserver =
          std::sqrt(std::max(0.0, 1.0 - sinObserver * sinObserver));
      double const jacobian = horizontal * cosObserver *
                              std::abs(dHorizontalDp * nSource * cosSource) / sinSource;
      if (!(jacobian > 1.0e-12) || !std::isfinite(jacobian)) {
        throw std::runtime_error(
            "invalid turning-ray Jacobian requires full-wave response");
      }
      PathDiagnostics path;
      path.valid_ = true;
      path.topology_ = "upper_turning_" + std::to_string(branch);
      path.invariantP_ = p;
      path.sourceRefractiveIndex_ = nSource;
      path.observerRefractiveIndex_ = nObserver;
      path.geometricLengthM_ = geometric;
      path.opticalPathM_ = optical;
      path.travelTimeS_ = optical / c_m_per_s;
      path.attenuationExponent_ = attenuation;
      // exp(-i omega T): one ray-tube caustic gives +pi/2. A depth
      // turning point alone is not a point-source ray-tube caustic.
      path.phaseOffsetRad_ = dHorizontalDp > 0.0 ? 0.5 * pi : 0.0;
      path.horizontalResidualM_ = horizontalAt(p) - horizontal;
      path.fermatRelativeResidual_ = std::abs(path.horizontalResidualM_) / horizontal;
      path.snellResidual_ = std::max(std::abs(nSource * sinSource - p),
                                     std::abs(nObserver * sinObserver - p));
      path.geometricSpreadingPerM_ = std::sqrt(nSource / (nObserver * jacobian));
      path.emitDirection_ =
          add(multiply(horizontal_unit, sinSource), vector_type{0.0, 0.0, cosSource});
      path.receiveDirection_ = add(multiply(horizontal_unit, sinObserver),
                                   vector_type{0.0, 0.0, -cosObserver});
      result.push_back(path);
    }
    return result;
  }

  inline PathDiagnostics StratifiedRaySolver::surfaceReflected(
      vector_type const& source, vector_type const& observer, double frequencyHz) const {
    PathDiagnostics invalid;
    invalid.topology_ = "no_surface_reflected_root";
    if (!finite(source) || !finite(observer) || !std::isfinite(frequencyHz) ||
        frequencyHz < 0.0 || source[2] < profile_.minimumSupportedZ() ||
        source[2] > profile_.maximumSupportedZ() ||
        observer[2] < profile_.minimumSupportedZ() ||
        observer[2] > profile_.maximumSupportedZ()) {
      throw std::runtime_error("invalid ray query or endpoint outside optical profile");
    }
    double const horizontal = detail::horizontal_distance(source, observer);
    double const topZ = profile_.maximumSupportedZ();
    if (!(topZ > source[2]) || !(topZ > observer[2])) { return invalid; }
    double const nTop = profile_.refractiveIndexAtZ(topZ);
    double const nSource = profile_.refractiveIndexAtZ(source[2]);
    double const nObserver = profile_.refractiveIndexAtZ(observer[2]);
    if (horizontal == 0.0) {
      auto sumVertical = [&](auto const& f) {
        return detail::simpson(f, source[2], topZ) +
               detail::simpson(f, observer[2], topZ);
      };
      double const inverseIndexIntegral =
          sumVertical([&](double z) { return 1.0 / profile_.refractiveIndexAtZ(z); });
      PathDiagnostics path;
      path.valid_ = true;
      path.topology_ = "surface_reflected";
      path.sourceRefractiveIndex_ = nSource;
      path.observerRefractiveIndex_ = nObserver;
      path.geometricLengthM_ = 2.0 * topZ - source[2] - observer[2];
      path.opticalPathM_ =
          sumVertical([&](double z) { return profile_.refractiveIndexAtZ(z); });
      path.travelTimeS_ = path.opticalPathM_ / c_m_per_s;
      path.attenuationExponent_ =
          sumVertical([&](double z) { return profile_.attenuationAtZ(z, frequencyHz); });
      path.geometricSpreadingPerM_ =
          1.0 / (std::sqrt(nSource * nObserver) * inverseIndexIntegral);
      path.emitDirection_ = {0.0, 0.0, 1.0};
      path.receiveDirection_ = {0.0, 0.0, -1.0};
      return path;
    }
    auto legIntegral = [&](double endpointZ, double p,
                           std::function<double(double, double)> const& f) {
      double const upperU = std::sqrt(topZ - endpointZ);
      return detail::simpson(
          [&](double u) {
            double const z = topZ - u * u;
            double const n = profile_.refractiveIndexAtZ(z);
            return 2.0 * u * f(z, n) / std::sqrt(std::max(n * n - p * p, 1.0e-30));
          },
          0.0, upperU, 2048);
    };
    auto horizontalAt = [&](double p) {
      auto f = [p](double, double) { return p; };
      return legIntegral(source[2], p, f) + legIntegral(observer[2], p, f);
    };
    double low = 0.0;
    double high = nTop * (1.0 - 1.0e-11);
    if (horizontalAt(high) < horizontal) return invalid;
    for (int iteration = 0; iteration < 100; ++iteration) {
      double const middle = 0.5 * (low + high);
      if (horizontalAt(middle) < horizontal)
        low = middle;
      else
        high = middle;
    }
    double const p = 0.5 * (low + high);
    auto sumLegs = [&](std::function<double(double, double)> const& f) {
      return legIntegral(source[2], p, f) + legIntegral(observer[2], p, f);
    };
    double const geometric = sumLegs([](double, double n) { return n; });
    double const optical = sumLegs([](double, double n) { return n * n; });
    double const attenuation = sumLegs(
        [&](double z, double n) { return profile_.attenuationAtZ(z, frequencyHz) * n; });
    double const dp = std::max(1.0e-8, p * 1.0e-5);
    double const pMinus = std::max(0.0, p - dp);
    double const pPlus = std::min(nTop * (1.0 - 1.0e-11), p + dp);
    double const dHorizontalDp =
        (horizontalAt(pPlus) - horizontalAt(pMinus)) / (pPlus - pMinus);
    double const sinSource = p / nSource;
    double const sinObserver = p / nObserver;
    double const cosSource = std::sqrt(std::max(0.0, 1.0 - sinSource * sinSource));
    double const cosObserver = std::sqrt(std::max(0.0, 1.0 - sinObserver * sinObserver));
    double const jacobian = horizontal * cosObserver *
                            std::abs(dHorizontalDp * nSource * cosSource) /
                            std::max(sinSource, 1.0e-30);
    if (!(jacobian > 0.0) || !std::isfinite(jacobian)) return invalid;
    vector_type const horizontal_unit = detail::horizontal_unit(source, observer);
    PathDiagnostics path;
    path.valid_ = true;
    path.topology_ = "surface_reflected";
    path.invariantP_ = p;
    path.sourceRefractiveIndex_ = nSource;
    path.observerRefractiveIndex_ = nObserver;
    path.geometricLengthM_ = geometric;
    path.opticalPathM_ = optical;
    path.travelTimeS_ = optical / c_m_per_s;
    path.attenuationExponent_ = attenuation;
    path.horizontalResidualM_ = horizontalAt(p) - horizontal;
    path.fermatRelativeResidual_ = std::abs(path.horizontalResidualM_) / horizontal;
    path.snellResidual_ = std::max({std::abs(nSource * sinSource - p),
                                    std::abs(nObserver * sinObserver - p),
                                    std::abs(nTop * (p / nTop) - p)});
    path.geometricSpreadingPerM_ = std::sqrt(nSource / (nObserver * jacobian));
    path.emitDirection_ =
        add(multiply(horizontal_unit, sinSource), vector_type{0.0, 0.0, cosSource});
    path.receiveDirection_ =
        add(multiply(horizontal_unit, sinObserver), vector_type{0.0, 0.0, -cosObserver});
    return path;
  }
} // namespace corsika::kokkos::radio::firn

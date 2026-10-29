/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

namespace corsika::kokkos::radio::firn::detail::go {
  KOKKOS_INLINE_FUNCTION double minimum(double a, double b) { return b < a ? b : a; }
  KOKKOS_INLINE_FUNCTION double maximum(double a, double b) { return a < b ? b : a; }
  constexpr double lightSpeed = 299792458.;
  constexpr double pi = 3.141592653589793238462643383279502884;

  /** Invert the monotone horizontal range using its analytic derivative.
   * Keep the physical bracket when Newton leaves the domain. Stop at floating-point
   * resolution instead of repeating bisection after the endpoints become adjacent.
   * The path integrals and the ray families are unchanged.
   */
  template <typename TEvaluate>
  KOKKOS_INLINE_FUNCTION double solve_horizontal_range(TEvaluate const& evaluate,
                                                       double target, double upper) {
    if (target == 0.) return 0.;
    double lower = 0.;
    double p = .5 * upper;
    double best = p, error = INFINITY;
    for (int iteration = 0; iteration < 100; ++iteration) {
      auto const data = evaluate(p);
      double const residual = data.horizontalM_ - target;
      if (std::abs(residual) < error) {
        best = p;
        error = std::abs(residual);
      }
      if (residual == 0.) break;
      if (residual < 0.)
        lower = p;
      else
        upper = p;
      double const middle = lower + .5 * (upper - lower);
      if (middle == lower || middle == upper) break;
      double const next = p - residual / data.horizontalDerivativeM_;
      if (next == p) break;
      p = std::isfinite(next) && next > lower && next < upper ? next : middle;
    }
    // Check the neighbouring representable roots when a Newton update rounds away.
    for (int direction = 0; direction < 2; ++direction) {
      double const neighbour =
          std::nextafter(best, direction ? static_cast<double>(INFINITY)
                                         : -static_cast<double>(INFINITY));
      if (neighbour < lower || neighbour > upper) continue;
      double const residual = std::abs(evaluate(neighbour).horizontalM_ - target);
      if (residual < error) {
        best = neighbour;
        error = residual;
      }
    }
    return best;
  }
  KOKKOS_INLINE_FUNCTION ExponentialEndpointData
  Profile::endpointRayData(double q, double deepZM, double shallowZM) const {
    double const a = data_.deepIndex_;
    double const h = data_.scaleDepthM_;
    double const deepDeficit =
        data_.deltaIndex_ * std::exp((deepZM - data_.surfaceZM_) / h);
    double const shallowDeficit =
        data_.deltaIndex_ * std::exp((shallowZM - data_.surfaceZM_) / h);
    double const nd = a - deepDeficit, ns = a - shallowDeficit;
    double const p = ns * std::sqrt(1.0 - q * q);
    double const g2 = (a - p) * (a + p), g = std::sqrt(g2);
    double const root = std::sqrt((nd - ns) * (nd + ns) + ns * ns * q * q);
    double const deepCosine = root / nd;
    double const deepT =
        std::log((std::sqrt((a - p) * (nd + p)) + std::sqrt((a + p) * (nd - p))) /
                 std::sqrt(2.0 * p * deepDeficit));
    double const c = (a + p) * ns / (g * (ns + p));
    double const x = c * q, shallowT = std::atanh(x);
    double const integral = 2.0 * (deepT - shallowT) / g;
    double const hyperbolicDifference = std::atanh(deepCosine) - std::atanh(q);
    double const dpdq = -ns * ns * q / p;
    double const deepDerivative =
        h / g2 * (2.0 * a * a / g * deepT - (a * nd + p * p) / root);
    double const dxdq = c * (1.0 + q * dpdq * (1.0 / (a + p) + p / g2 - 1.0 / (ns + p)));
    double const shallowDerivative = 2.0 * h * a * a / (g2 * g) * dpdq * shallowT +
                                     2.0 * p * h / g * dxdq / (1.0 - x * x);
    // Signed q continues smoothly through the horizontal endpoint. Unlike
    // dX/dp, dX/dq remains finite; no angular interval is discarded.
    return {p,
            p * h * integral,
            h * (a * integral - hyperbolicDifference),
            h * (a * a * integral - a * hyperbolicDifference - (root - ns * q)),
            deepDerivative * dpdq - shallowDerivative,
            deepCosine};
  }

  KOKKOS_INLINE_FUNCTION double Profile::turningLegHorizontal(double p,
                                                              double endpointZM) const {
    double const a = data_.deepIndex_;
    double const deficit = data_.deltaIndex_ *
                           std::exp((endpointZM - data_.surfaceZM_) / data_.scaleDepthM_);
    double const n = a - deficit;
    double const gap = a - p;
    double const t =
        std::log((std::sqrt(gap * (n + p)) + std::sqrt((a + p) * (gap - deficit))) /
                 std::sqrt(2.0 * p * deficit));
    return 2.0 * data_.scaleDepthM_ * p * t / std::sqrt(gap * (a + p));
  }

  KOKKOS_INLINE_FUNCTION double Profile::turningLegHorizontalDerivative(
      double p, double endpointZM) const {
    double const deficit = data_.deltaIndex_ *
                           std::exp((endpointZM - data_.surfaceZM_) / data_.scaleDepthM_);
    double const n = data_.deepIndex_ - deficit;
    double const gap = data_.deepIndex_ - p;
    if (!(p > 0.0) || !(deficit > 0.0) || !(gap > deficit)) return NAN;
    double const g2 = gap * (data_.deepIndex_ + p);
    double const root = std::sqrt((gap - deficit) * (n + p));
    // Same stable analytic derivative as ExponentialProfile; no p stencil.
    double const t = std::log(
        (std::sqrt(gap * (n + p)) + std::sqrt((data_.deepIndex_ + p) * (gap - deficit))) /
        std::sqrt(2.0 * p * deficit));
    return data_.scaleDepthM_ / g2 *
           (2.0 * data_.deepIndex_ * data_.deepIndex_ / std::sqrt(g2) * t -
            (data_.deepIndex_ * n + p * p) / root);
  }
  KOKKOS_INLINE_FUNCTION HorizontalRange
  Profile::turningLegRange(double p, double endpointZM) const {
    double const a = data_.deepIndex_, h = data_.scaleDepthM_;
    double const deficit =
        data_.deltaIndex_ * std::exp((endpointZM - data_.surfaceZM_) / h);
    double const n = a - deficit, gap = a - p, g2 = gap * (a + p);
    double const g = std::sqrt(g2);
    double const t =
        std::log((std::sqrt(gap * (n + p)) + std::sqrt((a + p) * (gap - deficit))) /
                 std::sqrt(2. * p * deficit));
    double const derivative =
        (!(p > 0.) || !(deficit > 0.) || !(gap > deficit))
            ? NAN
            : h / g2 *
                  (2. * a * a / g * t -
                   (a * n + p * p) / std::sqrt((gap - deficit) * (n + p)));
    return {2. * h * p * t / g, derivative};
  }

  KOKKOS_INLINE_FUNCTION RayIntegrals Profile::monotonicRayData(double p, double deepZM,
                                                                double shallowZM) const {
    double const dz = shallowZM - deepZM;
    if (dz == 0.) return {};
    double const a = data_.deepIndex_, h = data_.scaleDepthM_;
    double const shallowDeficit =
        data_.deltaIndex_ * std::exp((shallowZM - data_.surfaceZM_) / h);
    double const deepDeficit =
        data_.deltaIndex_ * std::exp((deepZM - data_.surfaceZM_) / h);
    double const ns = a - shallowDeficit, nd = a - deepDeficit;
    // Form differences before subtracting the nearly equal refractive indices.
    double const deltaN = -shallowDeficit * std::expm1(-dz / h);
    double const rs = std::sqrt(maximum(0., (ns - p) * (ns + p)));
    double const rd = std::sqrt(maximum(0., (nd - p) * (nd + p)));
    double const g2 = (a - p) * (a + p), g = std::sqrt(g2);
    double const as = std::sqrt((a - p) * (ns + p));
    double const ad = std::sqrt((a - p) * (nd + p));
    double const bs = std::sqrt(maximum(0., (a + p) * (ns - p)));
    double const bd = std::sqrt(maximum(0., (a + p) * (nd - p)));
    double const deltaK = deltaN * ((a - p) / (ad + as) + (a + p) / (bd + bs));
    // J = integral_{ns}^{nd} dn / ((a-n) sqrt(n*n-p*p)).
    // log1p and dz/h avoid cancellation and an underflowing deficit ratio.
    double const j = (dz / h + 2. * std::log1p(deltaK / (as + bs))) / g;
    double const deltaR = deltaN * (nd + ns) / (rd + rs);
    double const deltaT = std::log1p((deltaN + deltaR) / (ns + rs));
    double derivative = static_cast<double>(INFINITY);
    if (rs > 0. && rd > 0.) {
      double const difference = p * p * deltaN * (nd + ns) / (rs * rd) *
                                (a / (ns * rd + nd * rs) + 1. / (rd + rs));
      derivative = h / g2 * (a * a * j + difference);
    }
    return {p * h * j, h * (a * j - deltaT), h * (a * a * j - a * deltaT - deltaR),
            derivative};
  }

  KOKKOS_INLINE_FUNCTION RayIntegrals Profile::turningLegData(double p,
                                                              double endpointZM) const {
    double const n = refractiveIndexAtZ(endpointZM);
    double const root = std::sqrt(maximum(0., (n - p) * (n + p)));
    double const horizontal = turningLegHorizontal(p, endpointZM);
    double const j = horizontal / (p * data_.scaleDepthM_);
    double const t = std::log1p((n - p + root) / p);
    double const a = data_.deepIndex_, h = data_.scaleDepthM_;
    return {horizontal, h * (a * j - t), h * (a * a * j - a * t - root),
            turningLegHorizontalDerivative(p, endpointZM)};
  }

  KOKKOS_INLINE_FUNCTION Vector add(Vector a, Vector b) {
    return {a[0] + b[0], a[1] + b[1], a[2] + b[2]};
  }
  KOKKOS_INLINE_FUNCTION Vector subtract(Vector a, Vector b) {
    return {a[0] - b[0], a[1] - b[1], a[2] - b[2]};
  }
  KOKKOS_INLINE_FUNCTION Vector multiply(Vector a, double b) {
    return {a[0] * b, a[1] * b, a[2] * b};
  }
  KOKKOS_INLINE_FUNCTION double dot(Vector a, Vector b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  }
  KOKKOS_INLINE_FUNCTION double norm(Vector a) { return std::sqrt(dot(a, a)); }
  KOKKOS_INLINE_FUNCTION bool finite(Vector a) {
    return std::isfinite(a[0]) && std::isfinite(a[1]) && std::isfinite(a[2]);
  }
  KOKKOS_INLINE_FUNCTION Vector cross(Vector a, Vector b) {
    return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0]};
  }
  KOKKOS_INLINE_FUNCTION Vector normalized(Vector a) { return multiply(a, 1. / norm(a)); }
  KOKKOS_INLINE_FUNCTION double horizontal_distance(Vector a, Vector b) {
    return std::hypot(b[0] - a[0], b[1] - a[1]);
  }
  KOKKOS_INLINE_FUNCTION Vector horizontal_unit(Vector a, Vector b) {
    double const d = horizontal_distance(a, b);
    return d > 0. ? Vector{(b[0] - a[0]) / d, (b[1] - a[1]) / d, 0.} : Vector{1., 0., 0.};
  }
  KOKKOS_INLINE_FUNCTION Path RaySolver::nearEndpoint(Vector const& source,
                                                      Vector const& observer,
                                                      bool wantTurning) const {
    auto const parameters = profile_.data_;
    auto const& exponential = profile_;
    if (!(parameters.deltaIndex_ > 0.0)) return {};
    double const deepZ = minimum(source[2], observer[2]);
    double const shallowZ = maximum(source[2], observer[2]);
    if (!(shallowZ > deepZ)) return {};
    double const horizontal = horizontal_distance(source, observer);
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
          std::sqrt(maximum(0.0, (ns - nTop) * (ns + nTop))) / ns;
      lower = maximum(lower, -surfaceCosine);
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
    Path result;
    result.valid_ = true;
    result.topology_ = wantTurning ? 2 : 0;
    result.invariantP_ = data.invariantP_;
    result.sourceRefractiveIndex_ = profile_.refractiveIndexAtZ(source[2]);
    result.observerRefractiveIndex_ = profile_.refractiveIndexAtZ(observer[2]);
    result.geometricLengthM_ = data.geometricLengthM_;
    result.opticalPathM_ = data.opticalPathM_;
    result.travelTimeS_ = data.opticalPathM_ / lightSpeed;
    result.attenuationExponent_ = parameters.attenuationPerM_ * data.geometricLengthM_;
    result.horizontalResidualM_ = data.horizontalM_ - horizontal;
    result.fermatRelativeResidual_ = std::abs(result.horizontalResidualM_) / horizontal;
    result.geometricSpreadingPerM_ =
        std::sqrt(ns / (nd * horizontal * data.deepCosine_ *
                        std::abs(data.horizontalDerivativeM_)));
    auto const horizontalUnit = go::horizontal_unit(source, observer);
    bool const sourceIsDeep = source[2] < observer[2];
    result.emitDirection_ =
        add(multiply(horizontalUnit, data.invariantP_ / result.sourceRefractiveIndex_),
            Vector{0.0, 0.0, sourceIsDeep ? data.deepCosine_ : -q});
    result.receiveDirection_ =
        add(multiply(horizontalUnit, data.invariantP_ / result.observerRefractiveIndex_),
            Vector{0.0, 0.0, sourceIsDeep ? q : -data.deepCosine_});
    return result;
  }

  KOKKOS_INLINE_FUNCTION Path RaySolver::direct(Vector const& source,
                                                Vector const& observer,
                                                double frequencyHz) const {
    if (!finite(source) || !finite(observer)) { return {}; }
    double const horizontal = horizontal_distance(source, observer);
    double const dz = observer[2] - source[2];
    double const straightLength = norm(subtract(observer, source));
    if (!(straightLength > 0.0)) return {};

    double const nSource = profile_.refractiveIndexAtZ(source[2]);
    double const nObserver = profile_.refractiveIndexAtZ(observer[2]);
    if (std::abs(dz) < 1.0e-12) {
      double const nMiddle = profile_.refractiveIndexAtZ(source[2]);
      double const probe = maximum(1.0e-4, 1.0e-7 * maximum(1.0, std::abs(source[2])));
      double lowerZ = maximum(profile_.minimumSupportedZ(), source[2] - probe);
      double upperZ = minimum(profile_.maximumSupportedZ(), source[2] + probe);
      bool const locallyUniform =
          upperZ > lowerZ && std::abs(profile_.refractiveIndexAtZ(upperZ) -
                                      profile_.refractiveIndexAtZ(lowerZ)) < 1.0e-12;
      if (!locallyUniform) {
        Path result;
        result.topology_ = 6;
        return result;
      }
      Path result;
      result.valid_ = true;
      result.topology_ = 6;
      result.invariantP_ = nMiddle;
      result.sourceRefractiveIndex_ = nMiddle;
      result.observerRefractiveIndex_ = nMiddle;
      result.geometricLengthM_ = horizontal;
      result.opticalPathM_ = nMiddle * horizontal;
      result.travelTimeS_ = result.opticalPathM_ / lightSpeed;
      result.attenuationExponent_ =
          profile_.attenuationAtZ(source[2], frequencyHz) * horizontal;
      result.geometricSpreadingPerM_ = 1.0 / horizontal;
      Vector const tangent = horizontal_unit(source, observer);
      result.emitDirection_ = tangent;
      result.receiveDirection_ = tangent;
      return result;
    }

    double const lowZ = minimum(source[2], observer[2]);
    auto const endpoint = nearEndpoint(source, observer, false);
    if (endpoint.valid_) return endpoint;
    double const highZ = maximum(source[2], observer[2]);
    auto horizontalIntegral = [&](double p) {
      return profile_.monotonicRayData(p, lowZ, highZ).horizontalM_;
    };

    // The signed-cosine endpoint calculation above covers this bracket's
    // excluded sliver without discarding any ordinary exponential-profile ray.
    double upper = minimum(nSource, nObserver) * (1.0 - 1.0e-7);
    double const maximumHorizontal = horizontalIntegral(upper);
    if (horizontal >= maximumHorizontal) {
      Path result;
      result.topology_ = 6;
      result.horizontalResidualM_ = horizontal - maximumHorizontal;
      return result;
    }
    double const p = solve_horizontal_range(
        [&](double value) { return profile_.monotonicRayData(value, lowZ, highZ); },
        horizontal, upper);
    auto const integrals = profile_.monotonicRayData(p, lowZ, highZ);
    double const recoveredHorizontal = integrals.horizontalM_;
    double const geometricLength = integrals.geometricLengthM_;
    double const opticalPath = integrals.opticalPathM_;
    double const attenuation = profile_.data_.attenuationPerM_ * geometricLength;

    double const signZ = dz > 0.0 ? 1.0 : -1.0;
    Vector const horizontal_unit = go::horizontal_unit(source, observer);
    auto tangent = [&](double n) {
      double const sine = maximum(0.0, minimum(1.0, p / n));
      double const cosine = std::sqrt(maximum(0.0, 1.0 - sine * sine));
      return add(multiply(horizontal_unit, sine), {0.0, 0.0, signZ * cosine});
    };
    Vector const emit = tangent(nSource);
    Vector const receive = tangent(nObserver);
    double const sinSource = p / nSource;
    double const cosSource = std::abs(emit[2]);
    double const cosObserver = std::abs(receive[2]);
    double const dHorizontalDp = integrals.horizontalDerivativeM_;
    double rayTubeJacobian = 0.0;
    if (horizontal > 1.0e-12 && p > 1.0e-12 && sinSource > 0.0) {
      double const dHorizontalDThetaSource = dHorizontalDp * nSource * cosSource;
      rayTubeJacobian = horizontal * cosObserver * dHorizontalDThetaSource / sinSource;
    } else {
      double const inverseIndexIntegral =
          profile_.monotonicRayData(0., lowZ, highZ).horizontalDerivativeM_;
      rayTubeJacobian = std::pow(nSource * inverseIndexIntegral, 2.0);
    }
    if (!(rayTubeJacobian > 0.0) || !std::isfinite(rayTubeJacobian)) {
      Path result;
      result.topology_ = 6;
      return result;
    }
    double const geometricSpreading = std::sqrt(nSource / (nObserver * rayTubeJacobian));

    Path result;
    result.valid_ = true;
    result.topology_ = 0;
    result.invariantP_ = p;
    result.sourceRefractiveIndex_ = nSource;
    result.observerRefractiveIndex_ = nObserver;
    result.geometricLengthM_ = geometricLength;
    result.opticalPathM_ = opticalPath;
    result.travelTimeS_ = opticalPath / lightSpeed;
    result.attenuationExponent_ = attenuation;
    result.horizontalResidualM_ = recoveredHorizontal - horizontal;
    result.fermatRelativeResidual_ =
        std::abs(result.horizontalResidualM_) / maximum(horizontal, 1.0e-300);
    result.snellResidual_ =
        maximum(std::abs(nSource * std::hypot(emit[0], emit[1]) - p),
                std::abs(nObserver * std::hypot(receive[0], receive[1]) - p));
    result.geometricSpreadingPerM_ = geometricSpreading;
    result.emitDirection_ = emit;
    result.receiveDirection_ = receive;
    return result;
  }

  KOKKOS_INLINE_FUNCTION BranchList<Path> RaySolver::turningBranches(
      Vector const& source, Vector const& observer, double) const {
    if (!finite(source) || !finite(observer)) { return {}; }
    double const horizontal = horizontal_distance(source, observer);
    if (!(horizontal > 0.0)) return {};
    double const topZ = profile_.maximumSupportedZ();
    double const shallowEndpointZ = maximum(source[2], observer[2]);
    if (!(topZ > shallowEndpointZ)) return {};
    double const nTop = profile_.refractiveIndexAtZ(topZ);
    double const nSource = profile_.refractiveIndexAtZ(source[2]);
    double const nObserver = profile_.refractiveIndexAtZ(observer[2]);
    double const nUpper = minimum(nSource, nObserver);
    if (!(nUpper > nTop * (1.0 + 1.0e-12))) return {};

    auto horizontalAt = [&](double p) {
      return profile_.turningLegHorizontal(p, source[2]) +
             profile_.turningLegHorizontal(p, observer[2]);
    };

    double const pLow = nTop * (1.0 + 1.0e-10);
    double const pHigh = nUpper * (1.0 - 1.0e-10);
    constexpr std::size_t samples = 512;
    BranchList<double> roots;
    // A fixed sampling grid can miss two roots on opposite sides of a
    // maximum within one cell. Bracket that stationary point first.
    auto derivativeAt = [&](double p) {
      return profile_.turningLegHorizontalDerivative(p, source[2]) +
             profile_.turningLegHorizontalDerivative(p, observer[2]);
    };
    auto rangeAt = [&](double p) {
      auto const first = profile_.turningLegRange(p, source[2]);
      auto const second = profile_.turningLegRange(p, observer[2]);
      return HorizontalRange{first.valueM_ + second.valueM_ - horizontal,
                             first.derivativeM_ + second.derivativeM_};
    };
    auto addRoot = [&](double low, double high, double lowValue, double highValue) {
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
    auto previousRange = rangeAt(previousP);
    for (std::size_t index = 1; index <= samples; ++index) {
      double const p = pLow + (static_cast<double>(index) / samples) * (pHigh - pLow);
      auto const range = rangeAt(p);
      if (previousRange.derivativeM_ > 0.0 && range.derivativeM_ < 0.0) {
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
        double const peakValue = horizontalAt(peak) - horizontal;
        if (std::abs(peakValue) <= 1.0e-11 * horizontal) { roots.invalid_ = true; }
        addRoot(previousP, peak, previousRange.valueM_, peakValue);
        addRoot(peak, p, peakValue, range.valueM_);
      } else {
        addRoot(previousP, p, previousRange.valueM_, range.valueM_);
      }
      previousP = p;
      previousRange = range;
    }

    Vector const horizontal_unit = go::horizontal_unit(source, observer);
    BranchList<Path> result;
    auto const endpoint = nearEndpoint(source, observer, true);
    if (endpoint.valid_) result.push_back(endpoint);
    for (int branch = 0; branch < roots.size(); ++branch) {
      double const p = roots[branch];
      if (endpoint.valid_ &&
          p >= nUpper * std::sqrt(1.0 - ExponentialEndpointData::cosineLimit_ *
                                            ExponentialEndpointData::cosineLimit_))
        continue;
      auto const sourceLeg = profile_.turningLegData(p, source[2]);
      auto const observerLeg = profile_.turningLegData(p, observer[2]);
      double const geometric =
          sourceLeg.geometricLengthM_ + observerLeg.geometricLengthM_;
      double const optical = sourceLeg.opticalPathM_ + observerLeg.opticalPathM_;
      double const attenuation = profile_.data_.attenuationPerM_ * geometric;
      double const dHorizontalDp =
          sourceLeg.horizontalDerivativeM_ + observerLeg.horizontalDerivativeM_;
      double const sinSource = p / nSource;
      double const sinObserver = p / nObserver;
      double const cosSource = std::sqrt(maximum(0.0, 1.0 - sinSource * sinSource));
      double const cosObserver = std::sqrt(maximum(0.0, 1.0 - sinObserver * sinObserver));
      double const jacobian = horizontal * cosObserver *
                              std::abs(dHorizontalDp * nSource * cosSource) / sinSource;
      if (!(jacobian > 1.0e-12) || !std::isfinite(jacobian)) {
        result.invalid_ = true;
        continue;
      }
      Path path;
      path.valid_ = true;
      path.topology_ = 2 + static_cast<int>(branch);
      path.invariantP_ = p;
      path.sourceRefractiveIndex_ = nSource;
      path.observerRefractiveIndex_ = nObserver;
      path.geometricLengthM_ = geometric;
      path.opticalPathM_ = optical;
      path.travelTimeS_ = optical / lightSpeed;
      path.attenuationExponent_ = attenuation;
      // exp(-i omega T): one ray-tube caustic gives +pi/2. A depth
      // turning point alone is not a point-source ray-tube caustic.
      path.phaseOffsetRad_ = dHorizontalDp > 0.0 ? 0.5 * pi : 0.0;
      path.horizontalResidualM_ = horizontalAt(p) - horizontal;
      path.fermatRelativeResidual_ = std::abs(path.horizontalResidualM_) / horizontal;
      path.snellResidual_ = maximum(std::abs(nSource * sinSource - p),
                                    std::abs(nObserver * sinObserver - p));
      path.geometricSpreadingPerM_ = std::sqrt(nSource / (nObserver * jacobian));
      path.emitDirection_ =
          add(multiply(horizontal_unit, sinSource), Vector{0.0, 0.0, cosSource});
      path.receiveDirection_ =
          add(multiply(horizontal_unit, sinObserver), Vector{0.0, 0.0, -cosObserver});
      result.push_back(path);
    }
    result.overflow_ = result.overflow_ || roots.overflow_;
    result.invalid_ = result.invalid_ || roots.invalid_;
    return result;
  }

  KOKKOS_INLINE_FUNCTION Path RaySolver::surfaceReflected(Vector const& source,
                                                          Vector const& observer,
                                                          double) const {
    Path invalid;
    invalid.topology_ = 6;
    if (!finite(source) || !finite(observer)) { return {}; }
    double const horizontal = horizontal_distance(source, observer);
    double const topZ = profile_.maximumSupportedZ();
    if (!(topZ > source[2]) || !(topZ > observer[2])) { return invalid; }
    double const nTop = profile_.refractiveIndexAtZ(topZ);
    double const nSource = profile_.refractiveIndexAtZ(source[2]);
    double const nObserver = profile_.refractiveIndexAtZ(observer[2]);
    if (horizontal == 0.0) {
      auto const sourceLeg = profile_.monotonicRayData(0., source[2], topZ);
      auto const observerLeg = profile_.monotonicRayData(0., observer[2], topZ);
      double const inverseIndexIntegral =
          sourceLeg.horizontalDerivativeM_ + observerLeg.horizontalDerivativeM_;
      Path path;
      path.valid_ = true;
      path.topology_ = 1;
      path.sourceRefractiveIndex_ = nSource;
      path.observerRefractiveIndex_ = nObserver;
      path.geometricLengthM_ = 2.0 * topZ - source[2] - observer[2];
      path.opticalPathM_ = sourceLeg.opticalPathM_ + observerLeg.opticalPathM_;
      path.travelTimeS_ = path.opticalPathM_ / lightSpeed;
      path.attenuationExponent_ =
          profile_.data_.attenuationPerM_ * path.geometricLengthM_;
      path.geometricSpreadingPerM_ =
          1.0 / (std::sqrt(nSource * nObserver) * inverseIndexIntegral);
      path.emitDirection_ = {0.0, 0.0, 1.0};
      path.receiveDirection_ = {0.0, 0.0, -1.0};
      return path;
    }
    auto rayData = [&](double p) {
      auto const first = profile_.monotonicRayData(p, source[2], topZ);
      auto const second = profile_.monotonicRayData(p, observer[2], topZ);
      return RayIntegrals{first.horizontalM_ + second.horizontalM_,
                          first.geometricLengthM_ + second.geometricLengthM_,
                          first.opticalPathM_ + second.opticalPathM_,
                          first.horizontalDerivativeM_ + second.horizontalDerivativeM_};
    };
    double high = nTop * (1.0 - 1.0e-11);
    if (rayData(high).horizontalM_ < horizontal) return invalid;
    double const p = solve_horizontal_range(rayData, horizontal, high);
    auto const sourceLeg = profile_.monotonicRayData(p, source[2], topZ);
    auto const observerLeg = profile_.monotonicRayData(p, observer[2], topZ);
    double const geometric = sourceLeg.geometricLengthM_ + observerLeg.geometricLengthM_;
    double const optical = sourceLeg.opticalPathM_ + observerLeg.opticalPathM_;
    double const attenuation = profile_.data_.attenuationPerM_ * geometric;
    double const dHorizontalDp =
        sourceLeg.horizontalDerivativeM_ + observerLeg.horizontalDerivativeM_;
    double const sinSource = p / nSource;
    double const sinObserver = p / nObserver;
    double const cosSource = std::sqrt(maximum(0.0, 1.0 - sinSource * sinSource));
    double const cosObserver = std::sqrt(maximum(0.0, 1.0 - sinObserver * sinObserver));
    double const jacobian = horizontal * cosObserver *
                            std::abs(dHorizontalDp * nSource * cosSource) /
                            maximum(sinSource, 1.0e-30);
    if (!(jacobian > 0.0) || !std::isfinite(jacobian)) return invalid;
    Vector const horizontal_unit = go::horizontal_unit(source, observer);
    Path path;
    path.valid_ = true;
    path.topology_ = 1;
    path.invariantP_ = p;
    path.sourceRefractiveIndex_ = nSource;
    path.observerRefractiveIndex_ = nObserver;
    path.geometricLengthM_ = geometric;
    path.opticalPathM_ = optical;
    path.travelTimeS_ = optical / lightSpeed;
    path.attenuationExponent_ = attenuation;
    path.horizontalResidualM_ =
        sourceLeg.horizontalM_ + observerLeg.horizontalM_ - horizontal;
    path.fermatRelativeResidual_ = std::abs(path.horizontalResidualM_) / horizontal;
    path.snellResidual_ = maximum(
        maximum(std::abs(nSource * sinSource - p), std::abs(nObserver * sinObserver - p)),
        std::abs(nTop * (p / nTop) - p));
    path.geometricSpreadingPerM_ = std::sqrt(nSource / (nObserver * jacobian));
    path.emitDirection_ =
        add(multiply(horizontal_unit, sinSource), Vector{0.0, 0.0, cosSource});
    path.receiveDirection_ =
        add(multiply(horizontal_unit, sinObserver), Vector{0.0, 0.0, -cosObserver});
    return path;
  }

  KOKKOS_INLINE_FUNCTION Query RaySolver::evaluate(Vector const& source,
                                                   Vector const& observer,
                                                   bool reflection, bool turning,
                                                   double foldGuard) const {
    Query result;
    if (!finite(source) || !finite(observer) ||
        source[2] < profile_.minimumSupportedZ() ||
        source[2] > profile_.maximumSupportedZ() ||
        observer[2] < profile_.minimumSupportedZ() ||
        observer[2] > profile_.maximumSupportedZ()) {
      result.status_ = Status::OutsideProfile;
      return result;
    }
    auto d = direct(source, observer);
    if (d.valid_) result.paths_.push_back(d);
    if (reflection) {
      auto r = surfaceReflected(source, observer);
      if (r.valid_) result.paths_.push_back(r);
    }
    if (turning) {
      auto t = turningBranches(source, observer);
      if (t.invalid_) {
        result.status_ = Status::InvalidGeometry;
        return result;
      }
      if (t.overflow_) {
        result.status_ = Status::BranchOverflow;
        return result;
      }
      for (int i = 1; i < t.size(); ++i) {
        auto path = t[i];
        int j = i;
        while (j > 0 && t[j - 1].travelTimeS_ > path.travelTimeS_) {
          t[j] = t[j - 1];
          --j;
        }
        t[j] = path;
      }
      if (t.size() >= 2 && t[1].travelTimeS_ - t[0].travelTimeS_ < foldGuard) {
        result.status_ = Status::FoldGuard;
        return result;
      }
      for (int i = 0; i < t.size(); ++i) {
        t[i].topology_ = i == 0 ? 2 : 3;
        result.paths_.push_back(t[i]);
      }
    }
    if (result.paths_.overflow_)
      result.status_ = Status::BranchOverflow;
    else if (result.paths_.empty())
      result.status_ = Status::NoPath;
    for (int i = 0; i < result.paths_.size(); ++i) {
      auto const& p = result.paths_[i];
      if (!std::isfinite(p.travelTimeS_) || !std::isfinite(p.geometricSpreadingPerM_) ||
          !std::isfinite(p.attenuationExponent_) || !finite(p.emitDirection_) ||
          !finite(p.receiveDirection_))
        result.status_ = Status::InvalidGeometry;
    }
    return result;
  }
} // namespace corsika::kokkos::radio::firn::detail::go

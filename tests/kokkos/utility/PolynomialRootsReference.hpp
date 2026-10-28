/*
 * (c) Copyright 2020, 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <Kokkos_Macros.hpp>
#include <cmath>
#include <limits>

#if defined(__CUDACC__)
#include <math_constants.h>
#endif

// Independent production oracle, frozen at
// 41cdc543429b82a216857ad458f83deb0f210a5f.
// The three source bodies below are copied from Git objects, not the dirty
// production checkout. Only their include dependencies, namespace and execution
// annotations are adapted. The default constructor uses an explicit empty body
// to avoid NVCC's ignored-annotation warning; its hi/lo member initializers are
// unchanged. No migrated solver or arithmetic helper is called.
namespace polynomial_roots_reference {
  KOKKOS_INLINE_FUNCTION double infinity() {
#if defined(__CUDA_ARCH__)
    return CUDART_INF;
#else
    return std::numeric_limits<double>::infinity();
#endif
  }
  KOKKOS_INLINE_FUNCTION double product(double first, double second) {
#if defined(__CUDA_ARCH__)
    return __dmul_rn(first, second);
#else
    return first * second;
#endif
  }
  KOKKOS_INLINE_FUNCTION double sum(double first, double second) {
#if defined(__CUDA_ARCH__)
    return __dadd_rn(first, second);
#else
    return first + second;
#endif
  }
  KOKKOS_INLINE_FUNCTION double difference(double first, double second) {
#if defined(__CUDA_ARCH__)
    return __dsub_rn(first, second);
#else
    return first - second;
#endif
  }
  KOKKOS_INLINE_FUNCTION double dot(double const* first, double const* second) {
    return sum(sum(product(first[0], second[0]), product(first[1], second[1])),
               product(first[2], second[2]));
  }
} // namespace polynomial_roots_reference

// Frozen source: corsika/accelerator/em/common/EnvironmentExtendedPrecision.hpp

namespace polynomial_roots_reference {
  KOKKOS_INLINE_FUNCTION bool extendedFinite(double x) {
#if defined(__CUDA_ARCH__)
    return ::isfinite(x);
#else
    return std::isfinite(x);
#endif
  }
  // Local two-double arithmetic for C8's cancellation-sensitive polynomial
  // resolvent. CUDA long double is just double. Keep the extra precision here,
  // not in particle states, model lookups or the rest of the transport kernel.
  struct Extended {
    double hi{}, lo{};
    KOKKOS_INLINE_FUNCTION Extended() {}
    KOKKOS_INLINE_FUNCTION Extended(double a)
        : hi(a) {}
    KOKKOS_INLINE_FUNCTION Extended(double a, double b)
        : hi(a)
        , lo(b) {}
    KOKKOS_INLINE_FUNCTION explicit operator double() const { return sum(hi, lo); }
  };
  KOKKOS_INLINE_FUNCTION Extended normalize(double a, double b) {
    auto s = sum(a, b);
    return {s, difference(b, difference(s, a))};
  }
  KOKKOS_INLINE_FUNCTION Extended operator+(Extended a, Extended b) {
    auto s = sum(a.hi, b.hi), v = difference(s, a.hi);
    auto e = sum(difference(a.hi, difference(s, v)), difference(b.hi, v));
    return normalize(s, sum(e, sum(a.lo, b.lo)));
  }
  KOKKOS_INLINE_FUNCTION Extended operator-(Extended a) { return {-a.hi, -a.lo}; }
  KOKKOS_INLINE_FUNCTION Extended operator-(Extended a, Extended b) { return a + (-b); }
  KOKKOS_INLINE_FUNCTION Extended operator*(Extended a, Extended b) {
    auto p = product(a.hi, b.hi), e = ::fma(a.hi, b.hi, -p);
    e = sum(e, sum(product(a.hi, b.lo), product(a.lo, b.hi)));
    e = sum(e, product(a.lo, b.lo));
    return normalize(p, e);
  }
  KOKKOS_INLINE_FUNCTION Extended operator/(Extended a, Extended b) {
    auto q = a.hi / b.hi;
    if (!extendedFinite(q)) return {q};
    auto r = a - b * q;
    auto q1 = r.hi / b.hi;
    r = r - b * q1;
    return (Extended(q) + q1) + r.hi / b.hi;
  }
  KOKKOS_INLINE_FUNCTION bool operator<(Extended a, Extended b) {
    return a.hi < b.hi || (a.hi == b.hi && a.lo < b.lo);
  }
  KOKKOS_INLINE_FUNCTION bool operator>(Extended a, Extended b) { return b < a; }
  KOKKOS_INLINE_FUNCTION bool operator==(Extended a, Extended b) {
    return a.hi == b.hi && a.lo == b.lo;
  }
  KOKKOS_INLINE_FUNCTION Extended& operator+=(Extended& a, Extended b) {
    a = a + b;
    return a;
  }
  KOKKOS_INLINE_FUNCTION Extended& operator-=(Extended& a, Extended b) {
    a = a - b;
    return a;
  }
  KOKKOS_INLINE_FUNCTION Extended& operator/=(Extended& a, Extended b) {
    a = a / b;
    return a;
  }
  KOKKOS_INLINE_FUNCTION Extended extendedSqrt(Extended x) {
    auto q = ::sqrt(x.hi);
    if (q == 0. || !extendedFinite(q)) return {q};
    return Extended(q) + (x - Extended(q) * q) / (2 * q);
  }
  KOKKOS_INLINE_FUNCTION Extended extendedCbrt(Extended x) {
    Extended q(::cbrt(x.hi));
    if (q.hi == 0. || !extendedFinite(q.hi)) return q;
    for (int i = 0; i < 2; ++i) q += (x - q * q * q) / (3 * q * q);
    return q;
  }
} // namespace polynomial_roots_reference

// Frozen source: corsika/accelerator/em/common/EnvironmentPolynomialRoots.hpp

namespace polynomial_roots_reference {
  // Arithmetic helpers for stable sphere roots and the native plane quadratic.
  // CPU uses extended precision; CUDA uses a local two-double expansion.
#if defined(__CUDA_ARCH__)
  using Real = Extended;
#else
  using Real = long double;
#endif
  struct Roots {
    double values[4]{};
    unsigned size{};
    template <typename T>
    KOKKOS_INLINE_FUNCTION void append(T x) {
      values[size++] = static_cast<double>(x);
    }
  };
  KOKKOS_INLINE_FUNCTION Real abs(Real x) { return x < 0 ? -x : x; }
  KOKKOS_INLINE_FUNCTION Real sqrt(Real x) {
#if defined(__CUDA_ARCH__)
    return extendedSqrt(x);
#else
    return std::sqrt(x);
#endif
  }
  KOKKOS_INLINE_FUNCTION Real cbrt(Real x) {
#if defined(__CUDA_ARCH__)
    return extendedCbrt(x);
#else
    return std::cbrt(x);
#endif
  }
  KOKKOS_INLINE_FUNCTION Real fma(Real a, Real b, Real c) {
#if defined(__CUDA_ARCH__)
    return a * b + c;
#else
    return std::fma(a, b, c);
#endif
  }
  KOKKOS_INLINE_FUNCTION Real acos(Real x) {
#if defined(__CUDA_ARCH__)
    auto angle = ::acos(x.hi);
    if (x.lo == 0.) return Extended(angle);
    if (x.hi == 1.) return extendedSqrt(2 * (Real(1) - x));
    if (x.hi == -1.)
      return Real(3.14159265358979323846) - extendedSqrt(2 * (Real(1) + x));
    return Extended(angle) - x.lo / ::sqrt((1 - x.hi) * (1 + x.hi));
#else
    return std::acos(x);
#endif
  }
  KOKKOS_INLINE_FUNCTION Real cos(Real x) {
#if defined(__CUDA_ARCH__)
    return Extended(::cos(x.hi)) - ::sin(x.hi) * x.lo;
#else
    return std::cos(x);
#endif
  }
  KOKKOS_INLINE_FUNCTION Roots linear(double a, double b) {
    Roots out;
    if (a != 0.) out.append(-b / a);
    return out;
  }
  KOKKOS_INLINE_FUNCTION Roots quadratic(Real a, Real b, Real c,
                                         double epsilon = 1.e-12) {
    if (abs(a) < epsilon) return linear(static_cast<double>(b), static_cast<double>(c));
    if (abs(c) < epsilon) {
      auto out = linear(static_cast<double>(a), static_cast<double>(b));
      out.append(0.);
      return out;
    }
    Real const w = a * 4 * c;
    Real const e = fma(a * 4, c, -w), f = fma(b, b, -w), radicant = f + e;
    Roots out;
    if (abs(radicant) < epsilon) {
      out.append(-b / (2 * a));
      return out;
    }
    if (radicant < 0.) return out;
    Real const x = c * 2 / (b > 0 ? -b - sqrt(radicant) : -b + sqrt(radicant));
    out.append(x);
    out.append(c / (a * x));
    return out;
  }
  KOKKOS_INLINE_FUNCTION void appendDistinct(Roots& roots, double x) {
    for (unsigned i = 0; i < roots.size; ++i)
      if (::fabs(x - roots.values[i]) <=
          8 * 2.220446049250313e-16 * ::fmax(::fabs(x), ::fabs(roots.values[i])))
        return;
    if (roots.size < 4) roots.append(x);
  }
} // namespace polynomial_roots_reference

// Frozen source: corsika/accelerator/em/common/EnvironmentStableQuartic.hpp
// Stable factorization of the curved-sphere quartic, without a residual scan
// or interval/bisection fallback.

namespace polynomial_roots_reference {
  // Stable monic quadratic, with exact degeneracies rather than a dimensional
  // absolute epsilon. Root ordering is not used until the sphere classifier.
  KOKKOS_INLINE_FUNCTION Roots stableMonicQuadratic(Real b, Real c) {
    Roots out;
    if (c == 0) {
      out.append(0.);
      if (!(b == 0)) out.append(-b);
      return out;
    }
    Real w = 4 * c, disc = fma(b, b, -w);
    if (disc < 0) return out;
    if (disc == 0) {
      out.append(-b / 2);
      return out;
    }
    Real large = -(b + (!(b < 0) ? sqrt(disc) : -sqrt(disc))) / 2;
    out.append(large);
    out.append(c / large);
    return out;
  }

  // Largest positive real root of m^3+p*m^2+(p^2/4-r)*m-q^2/8=0.
  // The p>=0,B>=0 branch is monotone on m>=0. A scale-aware upper initial
  // value avoids subtracting nearly equal quantities in Cardano's formula.
  // Four fixed Halley updates: no residual loop and no bisection.
  KOKKOS_INLINE_FUNCTION Real stableResolvent(Real p, Real q, Real r) {
    Real B = p * p / 4 - r, D = q * q / 8;
    // For p<0,r<0 and small D the only real resolvent root is tiny, whereas
    // Cardano subtracts terms O(|p|). Select this coefficient domain directly.
    // x0=D/B <= |p|/64 brackets that root in [x0,2*x0]; D<|p|*|r|/8
    // excludes a second positive crossing near -p/2. No residual trial/scan.
    if (p < 0 && r < 0 && D < (-p) * B / 64 && D < p * r / 8) {
      Real x = D / B;
      for (unsigned i = 0; i < 4; ++i) {
        Real f = ((x + p) * x + B) * x - D, df = (3 * x + 2 * p) * x + B,
             ddf = 6 * x + 2 * p;
        x -= 2 * f * df / (2 * df * df - f * ddf);
      }
      return x;
    }
    if (!(p < 0) && !(B < 0)) {
      Real x = cbrt(D);
      if (p > 0) {
        Real a = sqrt(D / p);
        if (a < x) x = a;
      }
      if (B > 0) {
        Real a = D / B;
        if (a < x) x = a;
      }
      for (unsigned i = 0; i < 4; ++i) {
        Real f = ((x + p) * x + B) * x - D, df = (3 * x + 2 * p) * x + B,
             ddf = 6 * x + 2 * p;
        x -= 2 * f * df / (2 * df * df - f * ddf);
      }
      return x;
    }
    Real Q = (3 * B - p * p) / 9, R = (p * (9 * B - 2 * p * p) + 27 * D) / 54;
    Real disc = fma(R, R, Q * Q * Q), x;
    if (!(disc < 0)) {
      Real z = cbrt(R + (!(R < 0) ? sqrt(disc) : -sqrt(disc)));
      x = (z == 0 ? Real(0) : z - Q / z) - p / 3;
    } else {
      Real arg = R / sqrt(-Q * Q * Q);
      if (arg > 1) arg = 1;
      if (arg < -1) arg = -1;
      x = 2 * sqrt(-Q) * cos(acos(arg) / 3) - p / 3;
    }
    return x;
  }

  KOKKOS_INLINE_FUNCTION Roots stableQuartic(double p, double q, double r) {
    Roots out;
    // Normalize with an exact power of two. frexp(0) gives exponent zero;
    // zero coefficients must not dominate a small nonzero polynomial's scale.
    int ep = -4096, eq = -4096, er = -4096;
    if (p != 0) {
      int e;
      ::frexp(p, &e);
      ep = e >= 0 ? (e + 1) / 2 : e / 2;
    }
    if (q != 0) {
      int e;
      ::frexp(q, &e);
      eq = e >= 0 ? (e + 2) / 3 : e / 3;
    }
    if (r != 0) {
      int e;
      ::frexp(r, &e);
      er = e >= 0 ? (e + 3) / 4 : e / 4;
    }
    int exponent = ep > eq ? ep : eq;
    exponent = exponent > er ? exponent : er;
    if (exponent == -4096) {
      out.append(0.);
      return out;
    }
    Real P = ::ldexp(p, -2 * exponent), Q = ::ldexp(q, -3 * exponent),
         R = ::ldexp(r, -4 * exponent);
    double scale = ::ldexp(1., exponent);
    if (q == 0) {
      auto y = stableMonicQuadratic(P, R);
      for (unsigned i = 0; i < y.size; ++i) {
        if (y.values[i] < 0) continue;
        if (y.values[i] == 0)
          appendDistinct(out, 0.);
        else {
          double x = static_cast<double>(sqrt(Real(y.values[i]))) * scale;
          appendDistinct(out, -x);
          appendDistinct(out, x);
        }
      }
    } else {
      Real m = stableResolvent(P, Q, R);
      if (!(m > 0)) {
        out.append(infinity());
        return out;
      } // explicit failure, not silent no-root
      Real s = sqrt(2 * m), A = P / 2 + m, B = Q / (2 * s), u, v;
      // uv=R. Compute the large constant without cancellation, then the small
      // one by division. Do not compute A-B when A and B are nearly equal.
      if (!(abs(A + B) < abs(A - B))) {
        v = A + B;
        u = R / v;
      } else {
        u = A - B;
        v = R / u;
      }
      auto first = stableMonicQuadratic(s, u), second = stableMonicQuadratic(-s, v);
      for (unsigned i = 0; i < first.size; ++i)
        appendDistinct(out, first.values[i] * scale);
      for (unsigned i = 0; i < second.size; ++i)
        appendDistinct(out, second.values[i] * scale);
    }
    // Native intersection classification expects its real roots in the
    // factorization order, except the unordered outside classifier retains the
    // last exit. Ascending order preserves the two-real-root entry/exit case.
    for (unsigned i = 1; i < out.size; ++i) {
      double v = out.values[i];
      unsigned j = i;
      while (j && v < out.values[j - 1]) {
        out.values[j] = out.values[j - 1];
        --j;
      }
      out.values[j] = v;
    }
    return out;
  }
} // namespace polynomial_roots_reference

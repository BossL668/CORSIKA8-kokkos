/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 * Distributed under the 3-clause BSD license; see LICENSE.
 */
#pragma once
#include <corsika/accelerator/em/common/EnvironmentTreeSnapshot.hpp>

namespace corsika::gpu::em::environment_tree::polynomial {
  C8_ACCELERATOR_INLINE_FUNCTION inline bool extendedFinite(double x) {
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
    double hi{},lo{};
    C8_ACCELERATOR_INLINE_FUNCTION Extended()=default;
    C8_ACCELERATOR_INLINE_FUNCTION Extended(double a):hi(a) {}
    C8_ACCELERATOR_INLINE_FUNCTION Extended(double a,double b):hi(a),lo(b) {}
    C8_ACCELERATOR_INLINE_FUNCTION explicit operator double() const {return sum(hi,lo);}
  };
  C8_ACCELERATOR_INLINE_FUNCTION inline Extended normalize(double a,double b) {
    auto s=sum(a,b);return {s,difference(b,difference(s,a))};
  }
  C8_ACCELERATOR_INLINE_FUNCTION inline Extended operator+(Extended a,Extended b) {
    auto s=sum(a.hi,b.hi),v=difference(s,a.hi);
    auto e=sum(difference(a.hi,difference(s,v)),difference(b.hi,v));
    return normalize(s,sum(e,sum(a.lo,b.lo)));
  }
  C8_ACCELERATOR_INLINE_FUNCTION inline Extended operator-(Extended a) {return {-a.hi,-a.lo};}
  C8_ACCELERATOR_INLINE_FUNCTION inline Extended operator-(Extended a,Extended b) {return a+(-b);}
  C8_ACCELERATOR_INLINE_FUNCTION inline Extended operator*(Extended a,Extended b) {
    auto p=product(a.hi,b.hi),e=::fma(a.hi,b.hi,-p);
    e=sum(e,sum(product(a.hi,b.lo),product(a.lo,b.hi)));
    e=sum(e,product(a.lo,b.lo));return normalize(p,e);
  }
  C8_ACCELERATOR_INLINE_FUNCTION inline Extended operator/(Extended a,Extended b) {
    auto q=a.hi/b.hi;
    if (!extendedFinite(q)) return {q};
    auto r=a-b*q;auto q1=r.hi/b.hi;r=r-b*q1;
    return (Extended(q)+q1)+r.hi/b.hi;
  }
  C8_ACCELERATOR_INLINE_FUNCTION inline bool operator<(Extended a,Extended b) {
    return a.hi<b.hi || (a.hi==b.hi && a.lo<b.lo);
  }
  C8_ACCELERATOR_INLINE_FUNCTION inline bool operator>(Extended a,Extended b) {return b<a;}
  C8_ACCELERATOR_INLINE_FUNCTION inline bool operator==(Extended a,Extended b) {return a.hi==b.hi && a.lo==b.lo;}
  C8_ACCELERATOR_INLINE_FUNCTION inline Extended& operator+=(Extended& a,Extended b) {a=a+b;return a;}
  C8_ACCELERATOR_INLINE_FUNCTION inline Extended& operator-=(Extended& a,Extended b) {a=a-b;return a;}
  C8_ACCELERATOR_INLINE_FUNCTION inline Extended& operator/=(Extended& a,Extended b) {a=a/b;return a;}
  C8_ACCELERATOR_INLINE_FUNCTION inline Extended extendedSqrt(Extended x) {
    auto q=::sqrt(x.hi);
    if (q==0. || !extendedFinite(q)) return {q};
    return Extended(q)+(x-Extended(q)*q)/(2*q);
  }
  C8_ACCELERATOR_INLINE_FUNCTION inline Extended extendedCbrt(Extended x) {
    Extended q(::cbrt(x.hi));if (q.hi==0. || !extendedFinite(q.hi)) return q;
    for (int i=0;i<2;++i) q+=(x-q*q*q)/(3*q*q);
    return q;
  }
} // namespace corsika::gpu::em::environment_tree::polynomial

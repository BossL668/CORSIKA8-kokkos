/*
 * (c) Copyright 2020, 2026 CORSIKA Project, corsika-project@lists.kit.edu
 * Distributed under the 3-clause BSD license; see LICENSE.
 */
#pragma once
#include <corsika/accelerator/em/common/EnvironmentTreeSnapshot.hpp>
#include <corsika/accelerator/em/common/EnvironmentExtendedPrecision.hpp>

namespace corsika::gpu::em::environment_tree::polynomial {
  // Arithmetic helpers for stable sphere roots and the native plane quadratic.
  // CPU uses extended precision; CUDA uses a local two-double expansion.
#if defined(__CUDA_ARCH__)
  using Real=Extended;
#else
  using Real=long double;
#endif
  struct Roots {
    double values[4]{};
    unsigned size{};
    template <typename T>
    C8_ACCELERATOR_INLINE_FUNCTION void append(T x) { values[size++]=static_cast<double>(x); }
  };
  C8_ACCELERATOR_INLINE_FUNCTION inline Real abs(Real x) { return x<0?-x:x; }
  C8_ACCELERATOR_INLINE_FUNCTION inline Real sqrt(Real x) {
#if defined(__CUDA_ARCH__)
    return extendedSqrt(x);
#else
    return std::sqrt(x);
#endif
  }
  C8_ACCELERATOR_INLINE_FUNCTION inline Real cbrt(Real x) {
#if defined(__CUDA_ARCH__)
    return extendedCbrt(x);
#else
    return std::cbrt(x);
#endif
  }
  C8_ACCELERATOR_INLINE_FUNCTION inline Real fma(Real a,Real b,Real c) {
#if defined(__CUDA_ARCH__)
    return a*b+c;
#else
    return std::fma(a,b,c);
#endif
  }
  C8_ACCELERATOR_INLINE_FUNCTION inline Real acos(Real x) {
#if defined(__CUDA_ARCH__)
    auto angle=::acos(x.hi);
    if (x.lo==0.) return Extended(angle);
    if (x.hi==1.) return extendedSqrt(2*(Real(1)-x));
    if (x.hi==-1.) return Real(3.14159265358979323846)-extendedSqrt(2*(Real(1)+x));
    return Extended(angle)-x.lo/::sqrt((1-x.hi)*(1+x.hi));
#else
    return std::acos(x);
#endif
  }
  C8_ACCELERATOR_INLINE_FUNCTION inline Real cos(Real x) {
#if defined(__CUDA_ARCH__)
    return Extended(::cos(x.hi))-::sin(x.hi)*x.lo;
#else
    return std::cos(x);
#endif
  }
  C8_ACCELERATOR_INLINE_FUNCTION inline Roots linear(double a,double b) {
    Roots out;
    if (a!=0.) out.append(-b/a);
    return out;
  }
  C8_ACCELERATOR_INLINE_FUNCTION inline Roots quadratic(Real a,Real b,Real c,
                                                       double epsilon=1.e-12) {
    if (abs(a)<epsilon) return linear(static_cast<double>(b),static_cast<double>(c));
    if (abs(c)<epsilon) { auto out=linear(static_cast<double>(a),static_cast<double>(b));out.append(0.);return out; }
    Real const w=a*4*c;
    Real const e=fma(a*4,c,-w),f=fma(b,b,-w),radicant=f+e;
    Roots out;
    if (abs(radicant)<epsilon) {out.append(-b/(2*a));return out;}
    if (radicant<0.) return out;
    Real const x=c*2/(b>0?-b-sqrt(radicant):-b+sqrt(radicant));
    out.append(x);out.append(c/(a*x));return out;
  }
  C8_ACCELERATOR_INLINE_FUNCTION inline void appendDistinct(Roots& roots, double x) {
    for(unsigned i=0;i<roots.size;++i)
      if(::fabs(x-roots.values[i])<=8*2.220446049250313e-16*
          ::fmax(::fabs(x),::fabs(roots.values[i])))return;
    if(roots.size<4)roots.append(x);
  }
} // namespace corsika::gpu::em::environment_tree::polynomial

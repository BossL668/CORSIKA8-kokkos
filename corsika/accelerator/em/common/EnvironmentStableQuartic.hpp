/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 * Distributed under the 3-clause BSD license; see LICENSE.
 * Stable factorization of the curved-sphere quartic, without a residual scan
 * or interval/bisection fallback.
 */
#pragma once
#include <corsika/accelerator/em/common/EnvironmentPolynomialRoots.hpp>

namespace corsika::gpu::em::environment_tree::polynomial {
  // Stable monic quadratic, with exact degeneracies rather than a dimensional
  // absolute epsilon. Root ordering is not used until the sphere classifier.
  C8_ACCELERATOR_INLINE_FUNCTION inline Roots stableMonicQuadratic(Real b,Real c) {
    Roots out;
    if(c==0){out.append(0.);if(!(b==0))out.append(-b);return out;}
    Real w=4*c,disc=fma(b,b,-w);
    if(disc<0)return out;
    if(disc==0){out.append(-b/2);return out;}
    Real large=-(b+(!(b<0)?sqrt(disc):-sqrt(disc)))/2;
    out.append(large);out.append(c/large);return out;
  }

  // Largest positive real root of m^3+p*m^2+(p^2/4-r)*m-q^2/8=0.
  // The p>=0,B>=0 branch is monotone on m>=0. A scale-aware upper initial
  // value avoids subtracting nearly equal quantities in Cardano's formula.
  // Four fixed Halley updates: no residual loop and no bisection.
  C8_ACCELERATOR_INLINE_FUNCTION inline Real stableResolvent(Real p,Real q,Real r) {
    Real B=p*p/4-r,D=q*q/8;
    // For p<0,r<0 and small D the only real resolvent root is tiny, whereas
    // Cardano subtracts terms O(|p|). Select this coefficient domain directly.
    // x0=D/B <= |p|/64 brackets that root in [x0,2*x0]; D<|p|*|r|/8
    // excludes a second positive crossing near -p/2. No residual trial/scan.
    if(p<0 && r<0 && D<(-p)*B/64 && D<p*r/8){
      Real x=D/B;
      for(unsigned i=0;i<4;++i){
        Real f=((x+p)*x+B)*x-D,df=(3*x+2*p)*x+B,ddf=6*x+2*p;
        x-=2*f*df/(2*df*df-f*ddf);
      }
      return x;
    }
    if(!(p<0) && !(B<0)){
      Real x=cbrt(D);
      if(p>0){Real a=sqrt(D/p);if(a<x)x=a;}
      if(B>0){Real a=D/B;if(a<x)x=a;}
      for(unsigned i=0;i<4;++i){
        Real f=((x+p)*x+B)*x-D,df=(3*x+2*p)*x+B,ddf=6*x+2*p;
        x-=2*f*df/(2*df*df-f*ddf);
      }
      return x;
    }
    Real Q=(3*B-p*p)/9,R=(p*(9*B-2*p*p)+27*D)/54;
    Real disc=fma(R,R,Q*Q*Q),x;
    if(!(disc<0)){
      Real z=cbrt(R+(!(R<0)?sqrt(disc):-sqrt(disc)));
      x=(z==0?Real(0):z-Q/z)-p/3;
    }else{
      Real arg=R/sqrt(-Q*Q*Q);if(arg>1)arg=1;if(arg<-1)arg=-1;
      x=2*sqrt(-Q)*cos(acos(arg)/3)-p/3;
    }
    return x;
  }

  C8_ACCELERATOR_INLINE_FUNCTION inline Roots stableQuartic(double p,double q,double r) {
    Roots out;
    // Normalize with an exact power of two. frexp(0) gives exponent zero;
    // zero coefficients must not dominate a small nonzero polynomial's scale.
    int ep=-4096,eq=-4096,er=-4096;
    if(p!=0){int e;::frexp(p,&e);ep=e>=0?(e+1)/2:e/2;}
    if(q!=0){int e;::frexp(q,&e);eq=e>=0?(e+2)/3:e/3;}
    if(r!=0){int e;::frexp(r,&e);er=e>=0?(e+3)/4:e/4;}
    int exponent=ep>eq?ep:eq;exponent=exponent>er?exponent:er;
    if(exponent==-4096){out.append(0.);return out;}
    Real P=::ldexp(p,-2*exponent),Q=::ldexp(q,-3*exponent),R=::ldexp(r,-4*exponent);
    double scale=::ldexp(1.,exponent);
    if(q==0){
      auto y=stableMonicQuadratic(P,R);
      for(unsigned i=0;i<y.size;++i){
        if(y.values[i]<0)continue;
        if(y.values[i]==0)appendDistinct(out,0.);
        else {double x=static_cast<double>(sqrt(Real(y.values[i])))*scale;
          appendDistinct(out,-x);appendDistinct(out,x);}
      }
    }else{
      Real m=stableResolvent(P,Q,R);
      if(!(m>0)){out.append(infinity());return out;} // explicit failure, not silent no-root
      Real s=sqrt(2*m),A=P/2+m,B=Q/(2*s),u,v;
      // uv=R. Compute the large constant without cancellation, then the small
      // one by division. Do not compute A-B when A and B are nearly equal.
      if(!(abs(A+B)<abs(A-B))){v=A+B;u=R/v;}
      else {u=A-B;v=R/u;}
      auto first=stableMonicQuadratic(s,u),second=stableMonicQuadratic(-s,v);
      for(unsigned i=0;i<first.size;++i)appendDistinct(out,first.values[i]*scale);
      for(unsigned i=0;i<second.size;++i)appendDistinct(out,second.values[i]*scale);
    }
    // Native intersection classification expects its real roots in the
    // factorization order, except the unordered outside classifier retains the
    // last exit. Ascending order preserves the two-real-root entry/exit case.
    for(unsigned i=1;i<out.size;++i){double v=out.values[i];unsigned j=i;
      while(j && v<out.values[j-1]){out.values[j]=out.values[j-1];--j;}out.values[j]=v;}
    return out;
  }
}

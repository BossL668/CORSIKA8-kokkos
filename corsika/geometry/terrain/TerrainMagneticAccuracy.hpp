#pragma once

namespace corsika::terrain {
// The inherited LeapFrog position is r(s)=r0+u*s+q*s^2, with s=v*dt.
// Its chord is longer than s by sqrt(1+|q*s|^2)-1. The original 0.2 rad
// limit can therefore produce resolved acausal radio sources. For the stock
// limiter s <= sin(2*a)*R_perp, |q*s| <= sin(2*a)/2; a=4e-5 bounds this
// relative excess by 8e-10 before input rounding. It also bounds the local
// displacement error against an exact uniform-field helix by about 1.1e-9.
// Both terrain CPU tracking and the independent interface device session use
// this accuracy cap, even with radio disabled. Original air tracking is untouched.
inline constexpr double MaximumQuadraticMagneticDeflection = 4.e-5;
}

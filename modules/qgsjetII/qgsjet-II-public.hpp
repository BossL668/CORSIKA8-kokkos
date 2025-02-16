#pragma once

#include <functional>

#include "qgsjet-II-04-types.hpp"

namespace QGSJetII04 {
  extern QGARR12* const qgarr12_;
  extern QGARR13* const qgarr13_;
  extern QGARR14* const qgarr14_;
  extern QGARR55* const qgarr55_;

  // function pointers to internal (FORTRAN) functions
  extern void (*const qgset_)();
  extern void (*const qgaini_)(char const*);
  extern void (*const qgini_)(double const&, int const&, int const&, int const&);
  extern void (*const qgconf_)();
  extern double (*const qgsect_)(double const&, int const&, int const&, int const&);
  extern double (*const qgran_)(int&);

  extern void (*const set_rng_function)(std::function<void(double*, std::size_t)>);
} // namespace QGSJetII04

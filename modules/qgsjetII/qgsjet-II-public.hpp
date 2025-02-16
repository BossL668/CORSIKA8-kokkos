#pragma once

#include <functional>

#include "qgsjet-II-04-types.hpp"

namespace QGSJetII04 {

  //~ int const* nptmax;
  //~ int const* iapmax;

  extern QGARR12* qgarr12_;
  extern QGARR13* qgarr13_;
  extern QGARR14* qgarr14_;
  extern QGARR55* qgarr55_;

  // function pointers to internal (FORTRAN) symbols
  extern void (*qgset_)();
  extern void (*qgaini_)(char const*);
  extern void (*qgini_)(double const&, int const&, int const&, int const&);
  extern void (*qgconf_)();
  extern double (*qgsect_)(double const&, int const&, int const&, int const&);
  extern double (*qgran_)(int&);

  extern void (*set_rng_function)(std::function<void(double*, std::size_t)>);
} // namespace QGSJetII04

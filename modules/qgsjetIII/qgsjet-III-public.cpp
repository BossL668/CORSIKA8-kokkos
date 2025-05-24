#include <functional>

#include "qgsjet-III-types.hpp"
#include "qgsjet-III-private.hpp"

namespace QGSJetIII {
  extern QGARR12* const qgarr12_ = &::qgarr12_;
  extern QGARR14* const qgarr14_ = &::qgarr14_;
  extern QGARR13* const qgarr13_ = &::qgarr13_;
  extern QGARR55* const qgarr55_ = &::qgarr55_;

  // functions
  extern void (*const qgset_)() = &::qgset_;
  extern void (*const qgaini_)(char const*) = &::qgaini_;
  extern void (*const qgini_)(double const&, int const&, int const&,
                              int const&) = &::qgini_;
  extern void (*const qgconf_)() = &::qgconf_;
  extern double (*const qgsect_)(double const&, int const&, int const&,
                                 int const&) = &::qgsect_;
  extern double (*const qgran_)(int&) = &::qgran_;

  extern void (*const set_rng_function)(std::function<void(double*, std::size_t)>) =
      &::qgsjetIII::set_rng_function;
} // namespace QGSJetII04

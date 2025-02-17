#include <functional>

#include "qgsjet-II-04-types.hpp"
#include "qgsjet-II-04-private.hpp"

namespace QGSJetII04 {
  QGARR12* const qgarr12_ = &::qgarr12_;
  QGARR14* const qgarr14_ = &::qgarr14_;
  QGARR13* const qgarr13_ = &::qgarr13_;
  QGARR55* const qgarr55_ = &::qgarr55_;

  // functions
  void (*const qgset_)() = &::qgset_;
  void (*const qgaini_)(char const*) = &::qgaini_;
  void (*const qgini_)(double const&, int const&, int const&, int const&) = &::qgini_;
  void (*const qgconf_)() = &::qgconf_;
  double (*const qgsect_)(double const&, int const&, int const&, int const&) = &::qgsect_;
  double (*const qgran_)(int&) = &::qgran_;

  void (*set_rng_function)(std::function<void(double*, std::size_t)>) =
      &::qgsjetII::set_rng_function;
} // namespace QGSJetII04

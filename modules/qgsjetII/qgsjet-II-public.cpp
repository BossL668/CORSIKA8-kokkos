#include <functional>

#include "qgsjet-II-04-types.hpp"
#include "qgsjet-II-04-private.hpp"

namespace QGSJetII04 {
  QGARR12* qgarr12_ = &::qgarr12_;
  QGARR14* qgarr14_ = &::qgarr14_;
  QGARR13* qgarr13_ = &::qgarr13_;
  QGARR55* qgarr55_ = &::qgarr55_;

  //~ int const* nptmax = ::n

  // functions
  void (*qgset_)() = &::qgset_;
  void (*qgaini_)(char const*) = &::qgaini_;
  void (*qgini_)(double const&, int const&, int const&, int const&) = &::qgini_;
  void (*qgconf_)() = &::qgconf_;
  double (*qgsect_)(double const&, int const&, int const&, int const&) = &::qgsect_;
  double (*qgran_)(int&) = &::qgran_;

  void (*set_rng_function)(std::function<void(double*, std::size_t)>) =
      &::qgsjetII::set_rng_function;
} // namespace QGSJetII04

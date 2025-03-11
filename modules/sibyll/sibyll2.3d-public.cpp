#include <functional>

#include "sibyll2.3d-types.hpp"
#include "sibyll2.3d-private.hpp"

namespace sibyll23d {
  extern S_PLIST* const s_plist_ = &::s_plist_;
  extern S_CHIST* const s_chist_ = &::s_chist_;
  extern S_CSYDEC* const s_csydec_ = &::s_csydec_;
  extern S_PLIST1* const s_plist1_ = &::s_plist1_;
  extern S_CHP* const s_chp_ = &::s_chp_;
  extern S_MASS1* const s_mass1_ = &::s_mass1_;
  extern S_CNAM* const s_cnam_ = &::s_cnam_;
  extern S_DEBUG* const s_debug_ = &::s_debug_;

  extern void (*const sibyll_)(const int&, const int&, const double&) = &::sibyll_;
  extern void (*const sibyll_ini_)() = &::sibyll_ini_;
  extern void (*const decsib_)() = &::decsib_;
  extern void (*const sib_list_)(int&) = &::sib_list_;
  extern void (*const decpar_sib_)(const int&, const double*, int&, int*,
                                   double*) = &::decpar_sib_;
  extern void (*const sib_sigma_hnuc_)(const int&, const int&, const double&, double&,
                                       double&, double&) = &::sib_sigma_hnuc_;
  extern void (*const sib_sigma_hp_)(const int&, const double&, double&, double&, double&,
                                     double*, double&, double&) = &::sib_sigma_hp_;

  extern void (*const set_rng_function)(std::function<void(double*, std::size_t)>) =
      &::sibyll::set_rng_function;

  extern int (*const get_nwounded)() = &::get_nwounded;
  extern double (*const get_sibyll_mass2)(int&) = &::get_sibyll_mass2;

  extern void (*const nuc_nuc_ini_)() = &::nuc_nuc_ini_;
} // namespace sibyll23d

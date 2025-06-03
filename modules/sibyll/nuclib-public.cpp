#pragma once

#include "nuclib-types.hpp"
#include "nuclib-private.hpp"

namespace sibyll23d::nuclib {
    extern CNUCMS* const cnucms_ = &::cnucms_;
    extern CNUCSIGNUC* const cnucsignuc_ = &::cnucsignuc_;
    extern FRAGMENTS* const fragments_ = &::fragments_;

    // subroutine to initiate nuclib
    extern void (*const nuc_nuc_ini_)() = &::nuc_nuc_ini_;

    // subroutine to sample nuclear interaction structure
    extern void (*const int_nuc_)(const int&, const int&, const double&, const double&) = &::int_nuc_;

    // subroutine to sample nuclear fragments
    extern void (*const fragm_)(const int&, const int&, const int&, const double&, int&, int*) = &::fragm_;

    extern void (*const signuc_)(const int&, const double&, double&) = &::signuc_;

    extern void (*const signuc2_)(const int&, const int&, const double&, double&) = &::signuc2_;

    extern void (*const sigma_mc_)(const int&, const int&, const double&, const double&, const int&, double&,
               double&, double&, double&) = &::sigma_mc_;
}

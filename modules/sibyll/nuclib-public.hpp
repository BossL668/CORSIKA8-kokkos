#pragma once

#include <nuclib-types.hpp>

namespace sibyll23d::nuclib {
    extern CNUCMS* const cnucms_;
    extern CNUCSIGNUC* const cnucsignuc_;
    extern FRAGMENTS* const fragments_;

    // subroutine to initiate nuclib
    extern void (*const nuc_nuc_ini_)();

    // subroutine to sample nuclear interaction structure
    extern void (*const int_nuc_)(const int&, const int&, const double&, const double&);

    // subroutine to sample nuclear fragments
    extern void (*const fragm_)(const int&, const int&, const int&, const double&, int&, int*);

    extern void (*const signuc_)(const int&, const double&, double&);

    extern void (*const signuc2_)(const int&, const int&, const double&, double&);

    extern void (*const sigma_mc_)(const int&, const int&, const double&, const double&, const int&, double&,
               double&, double&, double&);
}

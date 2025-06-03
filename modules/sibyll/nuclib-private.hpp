#pragma once
#include "nuclib-types.hpp"

extern "C" {
    extern CNUCMS cnucms_;
    extern CNUCSIGNUC cnucsignuc_;
    extern FRAGMENTS fragments_;

// subroutine to initiate nuclib
void nuc_nuc_ini_();

// subroutine to sample nuclear interaction structure
void int_nuc_(const int&, const int&, const double&, const double&);

// subroutine to sample nuclear fragments
void fragm_(const int&, const int&, const int&, const double&, int&, int*);

void signuc_(const int&, const double&, double&);

void signuc2_(const int&, const int&, const double&, double&);

void sigma_mc_(const int&, const int&, const double&, const double&, const int&, double&,
               double&, double&, double&);
}

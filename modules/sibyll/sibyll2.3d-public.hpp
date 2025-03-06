#pragma once

#include <functional>

#include <sibyll2.3d-types.hpp>

namespace sibyll23d {
    extern S_PLIST* const s_plist_;
    extern S_CHIST* const s_chist_;
    extern S_CSYDEC* const s_csydec_;
    extern S_PLIST1* const s_plist1_;
    extern S_CHP * const s_chp_;
    extern S_MASS1* const s_mass1_;
    extern S_CNAM* const s_cnam_;
    extern S_DEBUG* const s_debug_;


    extern void (* const sibyll_)(const int&, const int&, const double&);
    extern void (* const sibyll_ini_)();
    extern void (* const decsib_)();
    extern void (* const sib_list_)(int&);
    extern void (* const decpar_sib_)(const int&, const double*, int&, int*, double*);
    extern void (* const sib_sigma_hnuc_)(const int&, const int&, const double&, double&, double&, double&);
    extern void (* const sib_sigma_hp_)(const int&, const double&, double&, double&, double&, double*, double&,
                       double&);

    extern void (*const set_rng_function)(std::function<void(double*, std::size_t)>);
}

#pragma once

#include <functional>

#include "epos-types.hpp"

namespace EPOS_LHC {
    extern void (*const set_rng_function)(std::function<void(double*, std::size_t)>);
    extern void (*const aaset_)(int&) ;
    extern double (*const lhcparameters_)() ;
    extern void (*const crseaaepos_)(float&, float&, float&, float&);
  extern float (*const eposcrse_)(float&, int&, int&, int&);
  extern float (*const eposelacrse_)(float&, int&, int&, int&) ;
  extern void (*const aepos_)(int&) ;
  extern void (*const afinal_)();
  extern void (*const alistf_)(char* str, int str_length) ;
  extern void (*const idmass_)(int&, float&);
  extern int (*const idtrafo_)(char[3], char[3], int&);
  extern void (* const ainit_)();
      
      extern CPTL* const cptl_ ;
      extern NODCY* const nodcy_ ;
      extern PRNT1* const prnt1_;
      extern PRNT3* const prnt3_;
      extern FILES* const files_;
      extern CSEED* const cseed_;
        extern ENRGY* const enrgy_;
        extern HADR6* const hadr6_;
        extern CJINTI * const cjinti_ ;
        extern OTHE1* const othe1_;
        extern NUCL6* const nucl6_ ;
        extern OTHE2* const othe2_ ;
        extern FNAME * const fname_ ;
        extern NFNAME * const nfname_;
        extern LEPT1* const lept1_;
          extern HADR25* const hadr25_;
          extern NUCL1 * const nucl1_;
          extern HADR5 * const hadr5_;
          extern HAD10 * const had10_;
          extern HADR2 * const hadr2_;
          extern HADR1 * const hadr1_;
}

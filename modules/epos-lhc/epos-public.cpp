#include <functional>

#include "epos-private.hpp"
#include "epos-types.hpp"

namespace EPOS_LHC {
    extern void (*const set_rng_function)(std::function<void(double*, std::size_t)>) =
      &::epos::set_rng_function;
      
    extern void (*const aaset_)(int&) = &::aaset_;
    extern double (*const lhcparameters_)() = &::lhcparameters_;
    extern void (*const crseaaepos_)(float&, float&, float&, float&) = &::crseaaepos_;
  extern float (*const eposcrse_)(float&, int&, int&, int&) = &::eposcrse_;
  extern float (*const eposelacrse_)(float&, int&, int&, int&) = &::eposelacrse_ ;
  extern void (*const aepos_)(int&) = &::aepos_ ;
  extern void (*const afinal_)() = &::afinal_ ;
  extern void (*const alistf_)(char* str, int str_length) = &::alistf_ ;
  extern void (*const idmass_)(int&, float&) = &::idmass_;
  extern int (*const idtrafo_)(char[3], char[3], int&) =&::idtrafo_;
  extern void (* const ainit_)() = &::ainit_;
      
      extern CPTL* const cptl_ = &::cptl_;
      extern NODCY* const nodcy_ = &::nodcy_;
      extern PRNT1* const prnt1_ = &::prnt1_;
      extern PRNT3* const prnt3_ = &::prnt3_;
      extern FILES* const files_ = &::files_;
      extern CSEED* const cseed_ = &::cseed_;
        extern ENRGY* const enrgy_ = &::enrgy_;
        extern HADR6* const hadr6_ = &::hadr6_;
        extern CJINTI * const cjinti_ = &::cjinti_;
        extern OTHE1* const othe1_ = &::othe1_;
        extern NUCL6* const nucl6_ = &::nucl6_;
        extern OTHE2* const othe2_ = &::othe2_;
        extern FNAME * const fname_ = &::fname_;
        extern NFNAME * const nfname_ = &::nfname_;
        extern LEPT1* const lept1_ = &::lept1_;
          extern HADR25* const hadr25_ = &::hadr25_;
          extern NUCL1 * const nucl1_ = &::nucl1_;
          extern HADR5 * const hadr5_  = &::hadr5_;
          extern HAD10 * const had10_ = &::had10_;
          extern HADR2 * const hadr2_ = &::hadr2_;
          extern HADR1 * const hadr1_ = &::hadr1_;
}

#include <iostream>

#include "epos-private.hpp"
#include <rng_impl.hpp>

IMPLEMENT_RNG(epos)

extern "C" {
// this is needed as linker object, but it is not needed to do anything
void ranfst_(int&) {}

// this is needed as linker object, but it is not needed to do anything
void ranfgt_(int&) {}

// this is needed as linker object, but it is not needed to do anything
void rmmaqd_(int[3], int&, char*, int) {}

// this is needed as linker object, but it is not needed to do anything
void ranfini_(double&, int&, int&) {}

// this is needed as linker object, but it is not needed to do anything
void ranfcv_(double&) {} // LCOV_EXCL_LINE

void rmmard_(double rvec[], int const* lenv, int const* /*iseq*/) {
  // we ignore iseq and draw all numbers from same C8 sequence
  rng_ptr(rvec, *lenv);
}

float rangen_() {
  float f{};
  do { f = draw_std_rnd(); } while (f == 1.0f);
  return f;
}

double drangen_() { return draw_std_rnd(); }
}

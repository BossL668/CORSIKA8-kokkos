#include "qgsjet-III-private.hpp"
#include <rng_impl.hpp>

IMPLEMENT_RNG(qgsjetIII)

/**
   @function qgran

   link to random number generation
 */
double qgran_(int&) { return draw_std_rnd(); }

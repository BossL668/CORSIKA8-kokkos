#include "qgsjet-II-04-private.hpp"
#include <rng_impl.hpp>

IMPLEMENT_RNG(qgsjetII)

/**
   @function qgran

   link to random number generation
 */
double qgran_(int&) { return draw_std_rnd(); }

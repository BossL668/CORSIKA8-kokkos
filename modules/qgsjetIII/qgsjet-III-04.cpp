#include <qgsjet-III-04.hpp>
#include <rng_impl.hpp>

#include <iostream>

datadir::datadir(std::string const& dir) {
  if (dir.length() > 130) { // LCOV_EXCL_START since we can't test this error message
    std::cerr << "QGSJetII error, will cut datadir \"" << dir
              << "\" to 130 characters: " << std::endl;
  } // LCOV_EXCL_STOP
  int i = 0;
  for (i = 0; i < std::min(130, int(dir.length())); ++i) data[i] = dir[i];
  data[i + 0] = ' ';
  data[i + 1] = '\0';
}

IMPLEMENT_RNG(qgsjetIII)

/**
   @function qgran

   link to random number generation
 */
double qgran_(int&) { return draw_std_rnd(); }

/*
 * (c) Copyright 2018 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include "sibyll2.3d-private.hpp"
#include <rng_impl.hpp>

#include <cmath>

int get_nwounded() { return s_chist_.nwd; }
double get_sibyll_mass2(int& id) { return s_mass1_.am2[std::abs(id) - 1]; }

IMPLEMENT_RNG(sibyll)

double s_rndm_(int&) { return ::draw_std_rnd(); }

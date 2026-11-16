/*
 * (c) Copyright 2022 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include <sophia.hpp>
#include <rng_impl.hpp>

#include <cmath>

double get_sophia_mass2(int& id) { return so_mass1_.am2[std::abs(id) - 1]; }

IMPLEMENT_RNG(sophia)

double rndm_(int&) { return draw_std_rnd(); }

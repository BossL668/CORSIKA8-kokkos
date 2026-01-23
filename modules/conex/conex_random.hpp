/*
 * (c) Copyright 2023 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */
#pragma once

#include <rng_decl.hpp>

DECLARE_RNG(conex)

extern "C" void rmmard_(double[] field, int const* N, int*);
extern "C" void rmmaqd_(int seed[3], int&, char*, int);
extern "C" double drangen_();

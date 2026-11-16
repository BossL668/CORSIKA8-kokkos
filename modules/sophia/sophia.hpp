/*
 * (c) Copyright 2022 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <rng_decl.hpp>

/**
 * \file sophia.hpp
 *
 * Interface definition to link to sophia library.
 *
 */

DECLARE_RNG(sophia)

//----------------------------------------------
//  C++ interface for the SOPHIA event generator
//----------------------------------------------
// wrapper

extern "C" {

/**
   \struct so_plist_

  SOPHIA particle stack (FORTRAN COMMON)
  variables are: np : numer of particles on stack
                  p : 4momentum + mass of particles on stack
              llist : id of particles on stack
 **/
extern struct {
  double p[5][2000];
  int llist[2000];
  int np;
} so_plist_;

extern struct {
  double cbr[102];
  int idb[49];
  int kdec[612];
  int lbarp[49];
} so_csydec_;

// additional particle stack for the mother particles of unstable particles
// stable particles have entry zero
extern struct {
  int llist1[2000];
} so_plist1_;

// tables with particle properties
// charge, strangeness and baryon number
extern struct {
  double s_life_[49];
  int ichp[49];
  int istr[49];
  int ibar[49];
} so_chp_;

// tables with particle properties
// mass and mass squared
extern struct {
  double am[49];
  double am2[49];
} so_mass1_;

// sophia initialization
void initial_(const int&);

// sophia main subroutine
void eventgen_(const int&, const double&, const double&, const double&, int&);

// print event
void print_event_(const int&);

// decay routine (LA,P0,ND,LL,P)
// void decpar_(const int&, const double*, int&, int*, double*);

double rndm_(int&);

double get_sophia_mass2(int&);
}

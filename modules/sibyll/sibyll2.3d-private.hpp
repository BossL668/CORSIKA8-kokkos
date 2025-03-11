/*
 * (c) Copyright 2018 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <rng_decl.hpp> // from modules/common
#include "sibyll2.3d-types.hpp"

DECLARE_RNG(sibyll)

extern "C" {
extern S_PLIST s_plist_;
extern S_CHIST s_chist_;
extern S_CSYDEC s_csydec_;
extern S_PLIST1 s_plist1_;
extern S_CHP s_chp_;
extern S_MASS1 s_mass1_;
extern S_CNAM s_cnam_;
extern S_DEBUG s_debug_;

// lund random generator setup
// extern struct {int mrlu[6]; float rrlu[100]; }ludatr_;

// sibyll main subroutine
void sibyll_(const int&, const int&, const double&);

// subroutine to initiate sibyll
void sibyll_ini_();

void decsib_();

// print event
void sib_list_(int&);

// decay routine (LA,P0,ND,LL,P)
void decpar_sib_(const int&, const double*, int&, int*, double*);

// interaction length
// double fpni_(double&, int&);

void sib_sigma_hnuc_(const int&, const int&, const double&, double&, double&, double&);
void sib_sigma_hp_(const int&, const double&, double&, double&, double&, double*, double&,
                   double&);

double s_rndm_(int&);

int get_nwounded();
double get_sibyll_mass2(int&);

// phojet random generator setup
void pho_rndin_(int&, int&, int&, int&);

void nuc_nuc_ini_();
}

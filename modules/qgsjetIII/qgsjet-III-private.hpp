/*
 * (c) Copyright 2025 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <rng_decl.hpp>

#include "qgsjet-III-types.hpp"

/**
 * @file qgsjet-III.hpp
 *
 * The interface to the fortran code.
 */

DECLARE_RNG(qgsjetIII)

//----------------------------------------------
//  C++ interface for the QGSJetIII event generator
//----------------------------------------------
// wrapper

extern "C" {

extern QGARR12 qgarr12_;

extern QGARR14 qgarr14_;

extern QGARR13 qgarr13_;

extern QGARR55 qgarr55_;

// functions
void qgset_();
void qgaini_(
    const char* datdir); // Note: there is a length limiation 132 from fortran-qgsjet here

/**
 * Additional initialization procedure per event.
 *
 * @param e0n  - interaction energy (per hadron/nucleon),
 * @param icp0 - hadron type (+-1 - pi+-, +-2 - p(p~), +-3 - n(n~), +-4 - K+-, +-5 -
 *               K_l/s),
 * @param iap  - projectile mass number (1 - for a hadron),
 * @param iat  - target mass number
 */
void qgini_(const double& e0n, const int& icp0, const int& iap, const int& iat);

/**
 * Generate one event configuration.
 */
void qgconf_();

/**
 * Hadron-nucleus (hadron-nucleus) particle production cross section.
 *
 * @param e0n lab. energy per projectile nucleon (hadron)
 * @param icz hadron class (1 - pion, 2 - nucleon, 3 - kaon)
 * @param iap0 projectile mass number (1=<iap0<=iapmax),
 * @param iat0 target mass number     (1=<iat0<=iapmax)
 */
double qgsect_(const double& e0n, const int& icz, const int& iap0, const int& iat0);

/**
 * Link to random number generation.
 */
double qgran_(int&);
}

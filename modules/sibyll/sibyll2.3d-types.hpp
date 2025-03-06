#pragma once

extern "C" {
struct S_PLIST {
  double p[5][8000];
  int llist[8000];
  int np;
};

//! additional information about interactions.
//! number of wounded nucleons, number of hard and soft scatterings etc.
struct S_CHIST {
  int nnsof[20], nnjet[20], jdif[20], nwd, njet, nsof;
};

struct S_CSYDEC {
  double cbr[223 + 16 + 12 + 8];
  int kdec[1338 + 6 * (16 + 12 + 8)];
  int lbarp[99];
  int idb[99];
};

//! additional particle stack for the mother particles of unstable particles
//! stable particles have entry zero
struct S_PLIST1 {
  int llist1[8000];
};

//! tables with particle properties
//! charge, strangeness and baryon number
struct S_CHP {
  int ichp[99];
  int istr[99];
  int ibar[99];
};

//! tables with particle properties
//! mass and mass squared
struct S_MASS1 {
  double am[99];
  double am2[99];
};

//! table with particle names
struct S_CNAM {
  char namp[6][99];
};

//! debug info
struct S_DEBUG {
  int ncall;
  int ndebug;
  int lun;
};
}

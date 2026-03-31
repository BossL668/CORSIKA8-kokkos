#pragma once

#ifndef QGSJET_TYPES_HEADER // shared with QGSJet-III, pragma doesn't work
#define QGSJET_TYPES_HEADER

#include <array>

extern "C" {

// data memory layout
struct QGARR12 {
  int nsp;
};

inline constexpr int nptmax = 95000;
inline constexpr int iapmax = 208;

struct QGARR14 {
  double esp[nptmax][4];
  int ich[nptmax];
};

struct QGARR13 {
  // c nsf - number of secondary fragments;
  // c iaf(i) - mass of the i-th fragment
  int nsf;
  int iaf[iapmax];
};

struct QGARR55 {
  int nwt;
  int nwp;
};
}

#endif

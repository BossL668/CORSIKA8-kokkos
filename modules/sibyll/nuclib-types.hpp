#pragma once

extern "C" {
struct CNUCMS {
  double b, bmax;
  int ntry, na, nb, ni, nael, nbel;
  int jja[56], jjb[56], jjint[56][56], jjael[56], jjbel[56];
};

/*
  nuclib common, nuclear FRAGMENTS

  COMMON /FRAGMENTS/ PPP(3,60)
*/
struct FRAGMENTS { double ppp[60][3]; };

//        COMMON /cnucsignuc/SIGMA(6,4,56), SIGQE(6,4,56)
struct CNUCSIGNUC {
  double sigma[56][4][6];
  double sigqe[56][4][6];
};
}

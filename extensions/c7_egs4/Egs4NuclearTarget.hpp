#pragma once
#include "Egs4Interactions.hpp"

namespace c7_egs4 {
struct NuclearTarget {
  InteractionStatus status{InteractionStatus::invalid_input};
  int pdg{},mass_number{},atomic_number{};
};
// Original C7 SDPM, photon-projectile block, LTA=0: literal A^0.91
// weights, not the nucleon selection in PIGEN1/2 or RHOGEN. Consumes
// precisely one hadronic-stream draw after PIGEN has selected branch 4.
// This selects a target only; it does not pretend to generate HDPM hadrons.
template<class Random>
KOKKOS_INLINE_FUNCTION NuclearTarget sampleManyHadronTarget(double const* composition,Random& rng) {
  NuclearTarget out;
  if(!composition)return out;
  for(int j=0;j<3;++j)if(!finite(composition[j])||composition[j]<0.)return out;
  double nitrogen=composition[0]*11.04019;
  double oxygen=nitrogen+composition[1]*12.46663;
  double total=oxygen+composition[2]*28.69952;
  if(!finite(total)||total<=0.)return out;
  double u;if(!uniform(rng,u)){out.status=InteractionStatus::random_failure;return out;}
  if(u*total<=nitrogen){out.mass_number=14;out.atomic_number=7;}
  else if(u*total<=oxygen){out.mass_number=16;out.atomic_number=8;}
  else {out.mass_number=40;out.atomic_number=18;}
  out.pdg=1000000000+10000*out.atomic_number+10*out.mass_number;
  out.status=InteractionStatus::success;return out;
}
} // namespace c7_egs4

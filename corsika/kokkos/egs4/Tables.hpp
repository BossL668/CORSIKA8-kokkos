/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

// Native C++ C7-EGS4 tables. These queries expose HATCH data; the transport
// module applies PHOTON/ELECTR thresholds, LPM and atmospheric stepping.
#include <Kokkos_Core.hpp>
#include <cmath>
#include <cstddef>
#include <iosfwd>
#include <stdexcept>
#include <string>
#include <vector>

namespace corsika::kokkos::egs4 {
  struct ElectronRow {
    float c_[20]{};
  };
  struct PhotonRow {
    float c_[10]{};
  };
  struct Medium {
    double rho_{}, radiationLengthCm_{}, teffCm_{}, blcc_{}, xcc_{};
    double ae_{}, ap_{}, ue_{}, up_{}, eke0_{}, eke1_{}, ge0_{}, ge1_{};
    double bindingEnergyMeV_{}; // HATCH's REAL EBINDA, promoted after reading
  };
  struct Tables {
    Medium medium_;
    std::vector<ElectronRow> electrons_;
    std::vector<PhotonRow> photons_;
    std::vector<float> bremsPairCoefficients_;
    std::string name_;
  };
  Tables read_air_tables(std::string const& path);
  Tables read_air_tables(std::istream& input);

  enum class Status { success, invalid_input, outside_table };
  struct ElectronQuery {
    Status status_{Status::invalid_input};
    int index_{-1};
    double ratePerCm_{}, lossMeVPerCm_{}, maximumStepCm_{};
    double branches_[3]{};
  };
  struct PhotonQuery {
    Status status_{Status::invalid_input};
    int index_{-1};
    double meanFreePathCm_{}, branches_[4]{};
  };
  struct TableView {
    Medium medium_;
    ElectronRow const* electrons_{};
    PhotonRow const* photons_{};
    int electronCount_{}, photonCount_{};
  };
  inline TableView host_view(Tables const& t);
  KOKKOS_INLINE_FUNCTION bool finite(double x);
  KOKKOS_INLINE_FUNCTION double smaller(double a, double b);
  KOKKOS_INLINE_FUNCTION double linear(float const* c, int i, double x);
  // Explicit mass input: do not silently replace C7's particle constants by C8's.
  KOKKOS_INLINE_FUNCTION ElectronQuery electron_query(TableView t, double total_MeV,
                                                      double mass_MeV, int charge);
  KOKKOS_INLINE_FUNCTION PhotonQuery photon_query(TableView t, double energy_MeV);

  struct LocalStep {
    Status status_{Status::invalid_input};
    double lossMeVPerCm_{}, ratePerCm_{}, trueStepCm_{}, projectedStepCm_{};
  };
  // Only ELECTR's local pre-geometry step proposal. Caller supplies RHOR(IRL),
  // C7 Z(NP), STERNCOR and remaining DEMFP. RHOR is NOT necessarily density at
  // the particle position: the later ALTEXP/barometric transformation is absent.
  // Never present this result as a full transported trajectory or decrement the
  // optical depth before geometry, scattering-path and collision corrections.
  KOKKOS_INLINE_FUNCTION LocalStep propose_local_step(
      TableView t, ElectronQuery q, double total_MeV, double mass_MeV,
      double cut_total_MeV, double region_rho, double c7_z_cm, double sterncor,
      double stepfc, double remaining_demfp, double vacuum_step_cm);
  // TVSTEP must already include the C7 path correction; not a straight-line length.
  KOKKOS_INLINE_FUNCTION double continuous_loss(double dedx, double tvstep);

  template <class TExecutionSpace>
  struct DeviceTables;
} // namespace corsika::kokkos::egs4

#include <corsika/detail/kokkos/egs4/Tables.inl>

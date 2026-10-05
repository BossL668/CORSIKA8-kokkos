#pragma once

// Experimental native C++ C7-EGS4 tables. No C7/Fortran/PROPOSAL runtime calls.
// This is NOT a complete electromagnetic backend yet. Queries are the HATCH
// tables before PHOTON/ELECTR process thresholds, LPM and atmospheric stepping.
#include <Kokkos_Core.hpp>
#include <cmath>
#include <cstddef>
#include <iosfwd>
#include <stdexcept>
#include <string>
#include <vector>

namespace c7_egs4 {
struct ElectronRow { float c[20]{}; };
struct PhotonRow { float c[10]{}; };
struct Medium {
  double rho{}, radiation_length_cm{}, teff_cm{}, blcc{}, xcc{};
  double ae{}, ap{}, ue{}, up{}, eke0{}, eke1{}, ge0{}, ge1{};
  double binding_energy_MeV{}; // HATCH's REAL EBINDA, promoted after reading
};
struct Tables {
  Medium medium;
  std::vector<ElectronRow> electrons;
  std::vector<PhotonRow> photons;
  std::vector<float> brems_pair_coefficients;
  std::string name;
};
Tables readAirTables(std::string const& path);
Tables readAirTables(std::istream& input);

enum class Status { success, invalid_input, outside_table };
struct ElectronQuery {
  Status status{Status::invalid_input};
  int index{-1};
  double rate_per_cm{}, loss_MeV_per_cm{}, maximum_step_cm{};
  double branches[3]{};
};
struct PhotonQuery {
  Status status{Status::invalid_input};
  int index{-1};
  double mean_free_path_cm{}, branches[4]{};
};
struct TableView {
  Medium medium;
  ElectronRow const* electrons{};
  PhotonRow const* photons{};
  int electron_count{}, photon_count{};
};
inline TableView hostView(Tables const& t) {
  return {t.medium, t.electrons.data(), t.photons.data(),
          static_cast<int>(t.electrons.size()), static_cast<int>(t.photons.size())};
}
KOKKOS_INLINE_FUNCTION bool finite(double x) {
  return x == x && x <= 1.7976931348623157e308 && x >= -1.7976931348623157e308;
}
KOKKOS_INLINE_FUNCTION double smaller(double a, double b) { return a < b ? a : b; }
KOKKOS_INLINE_FUNCTION double linear(float const* c, int i, double x) {
  return double(c[i + 1]) * x + double(c[i]);
}
// Explicit mass input: do not silently replace C7's particle constants by C8's.
KOKKOS_INLINE_FUNCTION ElectronQuery electronQuery(
    TableView t, double total_MeV, double mass_MeV, int charge) {
  ElectronQuery q;
  if (!finite(total_MeV) || !finite(mass_MeV) || mass_MeV <= 0. ||
      total_MeV <= mass_MeV || (charge != -1 && charge != 1)) return q;
  if (total_MeV < t.medium.ae || total_MeV > t.medium.ue) {
    q.status = Status::outside_table; return q;
  }
  double x = ::log(total_MeV - mass_MeV);
  double coordinate = t.medium.eke1 * x + t.medium.eke0;
  if (!finite(coordinate) || coordinate < 1. || coordinate >= t.electron_count + 1.) {
    q.status = Status::outside_table; return q;
  }
  q.index = int(coordinate) - 1; // Fortran truncation and 1-based -> 0-based.
  auto c = t.electrons[q.index].c;
  q.rate_per_cm = linear(c, charge < 0 ? 0 : 2, x);
  q.loss_MeV_per_cm = linear(c, charge < 0 ? 4 : 6, x);
  q.maximum_step_cm = linear(c, 18, x);
  int offset = charge < 0 ? 8 : 12;
  for (int i = 0; i < (charge < 0 ? 2 : 3); ++i)
    q.branches[i] = linear(c, offset + 2 * i, x);
  q.status = Status::success;
  return q;
}
KOKKOS_INLINE_FUNCTION PhotonQuery photonQuery(TableView t, double energy_MeV) {
  PhotonQuery q;
  if (!finite(energy_MeV) || energy_MeV <= 0.) return q;
  if (energy_MeV < t.medium.ap || energy_MeV > t.medium.up) {
    q.status = Status::outside_table; return q;
  }
  double x = ::log(energy_MeV), coordinate = t.medium.ge1 * x + t.medium.ge0;
  if (!finite(coordinate) || coordinate < 1. || coordinate >= t.photon_count + 1.) {
    q.status = Status::outside_table; return q;
  }
  q.index = int(coordinate) - 1;
  auto c = t.photons[q.index].c;
  q.mean_free_path_cm = linear(c, 0, x);
  for (int i = 0; i < 4; ++i) q.branches[i] = linear(c, 2 + 2 * i, x);
  q.status = Status::success;
  return q;
}

struct LocalStep {
  Status status{Status::invalid_input};
  double loss_MeV_per_cm{}, rate_per_cm{}, true_step_cm{}, projected_step_cm{};
};
// Only ELECTR's local pre-geometry step proposal. Caller supplies RHOR(IRL),
// C7 Z(NP), STERNCOR and remaining DEMFP. RHOR is NOT necessarily density at
// the particle position: the later ALTEXP/barometric transformation is absent.
// Never present this result as a full transported trajectory or decrement the
// optical depth before geometry, scattering-path and collision corrections.
KOKKOS_INLINE_FUNCTION LocalStep proposeLocalStep(
    TableView t, ElectronQuery q, double total_MeV, double mass_MeV,
    double cut_total_MeV, double region_rho, double c7_z_cm,
    double sterncor, double stepfc, double remaining_demfp, double vacuum_step_cm) {
  LocalStep s;
  if (q.status != Status::success) { s.status = q.status; return s; }
  if (!finite(total_MeV) || !finite(mass_MeV) || mass_MeV <= 0. ||
      !finite(cut_total_MeV) || cut_total_MeV < mass_MeV || total_MeV <= cut_total_MeV ||
      !finite(region_rho) || region_rho <= 0. || !finite(c7_z_cm) || !finite(sterncor) ||
      !finite(stepfc) || stepfc <= 0. || !finite(remaining_demfp) || remaining_demfp < 0. ||
      !finite(vacuum_step_cm) || vacuum_step_cm <= 0.) return s;
  double rhofac = region_rho / t.medium.rho, rhofi = 1. / rhofac;
  s.rate_per_cm = q.rate_per_cm * rhofac;
  double loss = q.loss_MeV_per_cm;
  if (total_MeV >= 3.)
    loss = smaller(loss, (86.65 - sterncor - c7_z_cm * 8.e-6) / t.medium.radiation_length_cm);
  s.loss_MeV_per_cm = rhofac * loss;
  if (!finite(s.loss_MeV_per_cm) || s.loss_MeV_per_cm <= 0.) return s;
  double step = s.rate_per_cm > 0. ? remaining_demfp / s.rate_per_cm : vacuum_step_cm;
  double tmx = smaller(q.maximum_step_cm, stepfc * 200. * t.medium.teff_cm) * rhofi;
  double beta2 = 1. - mass_MeV * mass_MeV / (total_MeV * total_MeV);
  if (beta2 < 1.e-8) beta2 = 1.e-8;
  double beta3 = total_MeV * beta2 * .094315;
  double tscat = t.medium.radiation_length_cm * beta3 * beta3 * rhofi;
  double range = (total_MeV - cut_total_MeV + .001) / s.loss_MeV_per_cm;
  s.true_step_cm = smaller(smaller(step, tmx), smaller(.3 * tscat, range));
  s.projected_step_cm = s.true_step_cm * (1. - s.true_step_cm / tscat);
  if (!finite(s.true_step_cm) || s.true_step_cm < 0. || !finite(s.projected_step_cm)) return s;
  s.status = Status::success;
  return s;
}
// TVSTEP must already include the C7 path correction; not a straight-line length.
KOKKOS_INLINE_FUNCTION double continuousLoss(double dedx, double tvstep) {
  return dedx * tvstep;
}

template<class ExecutionSpace> struct DeviceTables {
  Kokkos::View<ElectronRow*, typename ExecutionSpace::memory_space> electrons;
  Kokkos::View<PhotonRow*, typename ExecutionSpace::memory_space> photons;
  Medium medium;
  explicit DeviceTables(Tables const& t)
      : electrons("egs4_electron_tables", t.electrons.size()),
        photons("egs4_photon_tables", t.photons.size()), medium(t.medium) {
    auto e = Kokkos::create_mirror_view(electrons);
    auto p = Kokkos::create_mirror_view(photons);
    for (std::size_t i = 0; i < t.electrons.size(); ++i) e(i) = t.electrons[i];
    for (std::size_t i = 0; i < t.photons.size(); ++i) p(i) = t.photons[i];
    Kokkos::deep_copy(electrons, e); Kokkos::deep_copy(photons, p);
  }
  TableView view() const {
    return {medium, electrons.data(), photons.data(), int(electrons.extent(0)), int(photons.extent(0))};
  }
};
} // namespace c7_egs4

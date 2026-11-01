/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

namespace corsika::kokkos::egs4 {
  inline TableView host_view(Tables const& t) {
    return {t.medium_, t.electrons_.data(), t.photons_.data(),
            static_cast<int>(t.electrons_.size()), static_cast<int>(t.photons_.size())};
  }

  KOKKOS_INLINE_FUNCTION bool finite(double x) {
    return x == x && x <= 1.7976931348623157e308 && x >= -1.7976931348623157e308;
  }

  KOKKOS_INLINE_FUNCTION double smaller(double a, double b) { return a < b ? a : b; }

  KOKKOS_INLINE_FUNCTION double linear(float const* c, int i, double x) {
    return double(c[i + 1]) * x + double(c[i]);
  }

  KOKKOS_INLINE_FUNCTION ElectronQuery electron_query(TableView t, double total_MeV,
                                                      double mass_MeV, int charge) {
    ElectronQuery q;
    if (!finite(total_MeV) || !finite(mass_MeV) || mass_MeV <= 0. ||
        total_MeV <= mass_MeV || (charge != -1 && charge != 1))
      return q;
    if (total_MeV < t.medium_.ae_ || total_MeV > t.medium_.ue_) {
      q.status_ = Status::outside_table;
      return q;
    }
    double x = ::log(total_MeV - mass_MeV);
    double coordinate = t.medium_.eke1_ * x + t.medium_.eke0_;
    if (!finite(coordinate) || coordinate < 1. || coordinate >= t.electronCount_ + 1.) {
      q.status_ = Status::outside_table;
      return q;
    }
    q.index_ = int(coordinate) - 1; // Fortran truncation and 1-based -> 0-based.
    auto c = t.electrons_[q.index_].c_;
    q.ratePerCm_ = linear(c, charge < 0 ? 0 : 2, x);
    q.lossMeVPerCm_ = linear(c, charge < 0 ? 4 : 6, x);
    q.maximumStepCm_ = linear(c, 18, x);
    int offset = charge < 0 ? 8 : 12;
    for (int i = 0; i < (charge < 0 ? 2 : 3); ++i)
      q.branches_[i] = linear(c, offset + 2 * i, x);
    q.status_ = Status::success;
    return q;
  }

  KOKKOS_INLINE_FUNCTION PhotonQuery photon_query(TableView t, double energy_MeV) {
    PhotonQuery q;
    if (!finite(energy_MeV) || energy_MeV <= 0.) return q;
    if (energy_MeV < t.medium_.ap_ || energy_MeV > t.medium_.up_) {
      q.status_ = Status::outside_table;
      return q;
    }
    double x = ::log(energy_MeV), coordinate = t.medium_.ge1_ * x + t.medium_.ge0_;
    if (!finite(coordinate) || coordinate < 1. || coordinate >= t.photonCount_ + 1.) {
      q.status_ = Status::outside_table;
      return q;
    }
    q.index_ = int(coordinate) - 1;
    auto c = t.photons_[q.index_].c_;
    q.meanFreePathCm_ = linear(c, 0, x);
    for (int i = 0; i < 4; ++i) q.branches_[i] = linear(c, 2 + 2 * i, x);
    q.status_ = Status::success;
    return q;
  }

  KOKKOS_INLINE_FUNCTION LocalStep propose_local_step(
      TableView t, ElectronQuery q, double total_MeV, double mass_MeV,
      double cut_total_MeV, double region_rho, double c7_z_cm, double sterncor,
      double stepfc, double remaining_demfp, double vacuum_step_cm) {
    LocalStep s;
    if (q.status_ != Status::success) {
      s.status_ = q.status_;
      return s;
    }
    if (!finite(total_MeV) || !finite(mass_MeV) || mass_MeV <= 0. ||
        !finite(cut_total_MeV) || cut_total_MeV < mass_MeV ||
        total_MeV <= cut_total_MeV || !finite(region_rho) || region_rho <= 0. ||
        !finite(c7_z_cm) || !finite(sterncor) || !finite(stepfc) || stepfc <= 0. ||
        !finite(remaining_demfp) || remaining_demfp < 0. || !finite(vacuum_step_cm) ||
        vacuum_step_cm <= 0.)
      return s;
    double rhofac = region_rho / t.medium_.rho_, rhofi = 1. / rhofac;
    s.ratePerCm_ = q.ratePerCm_ * rhofac;
    double loss = q.lossMeVPerCm_;
    if (total_MeV >= 3.)
      loss = smaller(loss,
                     (86.65 - sterncor - c7_z_cm * 8.e-6) / t.medium_.radiationLengthCm_);
    s.lossMeVPerCm_ = rhofac * loss;
    if (!finite(s.lossMeVPerCm_) || s.lossMeVPerCm_ <= 0.) return s;
    double step = s.ratePerCm_ > 0. ? remaining_demfp / s.ratePerCm_ : vacuum_step_cm;
    double tmx = smaller(q.maximumStepCm_, stepfc * 200. * t.medium_.teffCm_) * rhofi;
    double beta2 = 1. - mass_MeV * mass_MeV / (total_MeV * total_MeV);
    if (beta2 < 1.e-8) beta2 = 1.e-8;
    double beta3 = total_MeV * beta2 * .094315;
    double tscat = t.medium_.radiationLengthCm_ * beta3 * beta3 * rhofi;
    double range = (total_MeV - cut_total_MeV + .001) / s.lossMeVPerCm_;
    s.trueStepCm_ = smaller(smaller(step, tmx), smaller(.3 * tscat, range));
    s.projectedStepCm_ = s.trueStepCm_ * (1. - s.trueStepCm_ / tscat);
    if (!finite(s.trueStepCm_) || s.trueStepCm_ < 0. || !finite(s.projectedStepCm_))
      return s;
    s.status_ = Status::success;
    return s;
  }

  KOKKOS_INLINE_FUNCTION double continuous_loss(double dedx, double tvstep) {
    return dedx * tvstep;
  }
  template <class TExecutionSpace>
  struct DeviceTables {
    Kokkos::View<ElectronRow*, typename TExecutionSpace::memory_space> electrons_;
    Kokkos::View<PhotonRow*, typename TExecutionSpace::memory_space> photons_;
    Medium medium_;
    explicit DeviceTables(Tables const& t)
        : electrons_("egs4_electron_tables", t.electrons_.size())
        , photons_("egs4_photon_tables", t.photons_.size())
        , medium_(t.medium_) {
      auto e = Kokkos::create_mirror_view(electrons_);
      auto p = Kokkos::create_mirror_view(photons_);
      for (std::size_t i = 0; i < t.electrons_.size(); ++i) e(i) = t.electrons_[i];
      for (std::size_t i = 0; i < t.photons_.size(); ++i) p(i) = t.photons_[i];
      Kokkos::deep_copy(electrons_, e);
      Kokkos::deep_copy(photons_, p);
    }
    TableView view() const {
      return {medium_, electrons_.data(), photons_.data(), int(electrons_.extent(0)),
              int(photons_.extent(0))};
    }
  };
} // namespace corsika::kokkos::egs4

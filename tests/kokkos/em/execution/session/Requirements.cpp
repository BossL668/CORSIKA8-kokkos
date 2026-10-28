/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include <corsika/kokkos/em/proposal/NativeRequirements.hpp>
#include <corsika/kokkos/em/tables/ContinuousPhysicsQueries.hpp>
#include "ProductionRequirements.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
  namespace now = corsika::kokkos;
  namespace old = native_table_reference;
  using old_requirements_type = corsika::accelerator::em::AcceleratedPhysicsRequirements;
  struct Inputs {
    now::ProposalNativeTableSet current;
    old::ProposalNativeTableSet original;
    now::PhysicsRequirements requirements{1.e6, 100., 100., .4};
    old_requirements_type oldRequirements{1.e6, 100., 100., .4};
  };
  unsigned cases{};

  Inputs inputs() {
    Inputs result;
    for (int pdg : {22, 11, -11, 13, -13}) {
      now::NativeTotalRateColumn a{};
      old::NativeTotalRateColumn b{};
      a.pdgId_ = b.pdg_id = pdg;
      a.spline_.axis_.low_ = b.spline.axis.low = 0.;
      a.spline_.axis_.high_ = b.spline.axis.high = 1.e7;
      result.current.totalRateColumns_.push_back(a);
      result.original.total_rate_columns.push_back(b);
      now::NativeDndxColumn d{};
      old::NativeDndxColumn e{};
      d.pdgId_ = e.pdg_id = pdg;
      d.rateModel_ = now::NativeRateModel::BicubicSpline;
      e.rate_model = old::NativeRateModel::BicubicSpline;
      d.spline_.energyAxis_.high_ = e.spline.energy_axis.high = 1.e7;
      result.current.dndxColumns_.push_back(d);
      result.original.dndx_columns.push_back(e);
      if (pdg == 22) continue;
      now::NativeUtilityColumn u{};
      old::NativeUtilityColumn v{};
      u.pdgId_ = v.pdg_id = pdg;
      u.lowerEnergyLimitMeV_ = v.lower_energy_limit_MeV = 0.;
      u.spline_.axis_.low_ = v.spline.axis.low = 0.;
      u.spline_.axis_.high_ = v.spline.axis.high = 1.e7;
      result.current.utilityColumns_.push_back(u);
      result.original.utility_columns.push_back(v);
      now::NativeDedxColumn x{};
      old::NativeDedxColumn y{};
      x.pdgId_ = y.pdg_id = pdg;
      x.spline_.axis_.high_ = y.spline.axis.high = 1.e7;
      result.current.dedxColumns_.push_back(x);
      result.original.dedx_columns.push_back(y);
    }
    return result;
  }

  template <typename TFunction>
  std::string outcome(TFunction&& function) {
    try {
      function();
      return "ok";
    } catch (std::runtime_error const& error) { return error.what(); }
  }

  void compare(Inputs const& in, std::string const& label, bool accepted) {
    ++cases;
    auto const expected = outcome([&] {
      corsika::accelerator::em::validateProposalNativeRequirements(in.original,
                                                                   in.oldRequirements);
    });
    auto const actual = outcome(
        [&] { now::validate_proposal_native_requirements(in.current, in.requirements); });
    if (actual != expected || (actual == "ok") != accepted)
      throw std::runtime_error(label + ": expected=" + expected + ", actual=" + actual);
    std::cout << "requirements " << label << " result=" << actual << '\n';
  }
} // namespace

int main() {
  try {
    compare(inputs(), "valid-all-five", true);
    for (unsigned index = 0; index < 5; ++index) {
      auto in = inputs();
      in.current.totalRateColumns_.erase(in.current.totalRateColumns_.begin() + index);
      in.original.total_rate_columns.erase(in.original.total_rate_columns.begin() +
                                           index);
      compare(in, "missing-total-" + std::to_string(index), false);
      in = inputs();
      in.current.dndxColumns_[index].spline_.energyAxis_.high_ =
          in.original.dndx_columns[index].spline.energy_axis.high = 1.e5;
      compare(in, "dndx-upper-" + std::to_string(index), false);
      in.current.dndxColumns_[index].rateModel_ =
          now::NativeRateModel::AnnihilationHeitler;
      in.original.dndx_columns[index].rate_model =
          old::NativeRateModel::AnnihilationHeitler;
      compare(in, "non-bicubic-upper-ignored-" + std::to_string(index), true);
      in = inputs();
      in.current.totalRateColumns_[index].spline_.axis_.high_ =
          in.original.total_rate_columns[index].spline.axis.high = 1.e5;
      compare(in, "total-upper-" + std::to_string(index), false);
      in = inputs();
      in.current.totalRateColumns_[index].spline_.axis_.low_ =
          in.original.total_rate_columns[index].spline.axis.low = 1.e5;
      compare(in, "total-lower-" + std::to_string(index), false);
    }
    for (unsigned index = 0; index < 4; ++index) {
      auto in = inputs();
      in.current.utilityColumns_.erase(in.current.utilityColumns_.begin() + index);
      in.original.utility_columns.erase(in.original.utility_columns.begin() + index);
      compare(in, "missing-utility-" + std::to_string(index), false);
      in = inputs();
      in.current.utilityColumns_[index].lowerEnergyLimitMeV_ =
          in.original.utility_columns[index].lower_energy_limit_MeV = 1.e5;
      compare(in, "utility-model-lower-" + std::to_string(index), false);
      in = inputs();
      in.current.utilityColumns_[index].spline_.axis_.low_ =
          in.original.utility_columns[index].spline.axis.low = 1.e5;
      compare(in, "utility-axis-lower-" + std::to_string(index), false);
      in = inputs();
      in.current.utilityColumns_[index].spline_.axis_.high_ =
          in.original.utility_columns[index].spline.axis.high = 1.e5;
      compare(in, "utility-upper-" + std::to_string(index), false);
      in = inputs();
      in.current.dedxColumns_[index].spline_.axis_.high_ =
          in.original.dedx_columns[index].spline.axis.high = 1.e5;
      compare(in, "dedx-upper-" + std::to_string(index), false);
    }
    auto in = inputs();
    in.current.totalRateColumns_.clear();
    in.original.total_rate_columns.clear();
    in.current.utilityColumns_.clear();
    in.original.utility_columns.clear();
    compare(in, "total-scanned-before-utility", false);

    in = inputs();
    auto duplicate = in.current.totalRateColumns_[0];
    auto oldDuplicate = in.original.total_rate_columns[0];
    duplicate.spline_.axis_.high_ = oldDuplicate.spline.axis.high = 1.;
    in.current.totalRateColumns_.push_back(duplicate);
    in.original.total_rate_columns.push_back(oldDuplicate);
    compare(in, "first-total-match-wins", true);
    std::swap(in.current.totalRateColumns_.front(), in.current.totalRateColumns_.back());
    std::swap(in.original.total_rate_columns.front(),
              in.original.total_rate_columns.back());
    compare(in, "first-total-match-reordered", false);

    in = inputs();
    in.current.dndxColumns_[0].pdgId_ = in.original.dndx_columns[0].pdg_id = 2212;
    in.current.dndxColumns_[0].spline_.energyAxis_.high_ =
        in.original.dndx_columns[0].spline.energy_axis.high = 1.;
    compare(in, "unrouted-pid-column-ignored", true);
    in.requirements.stochasticCutMeV_ = in.oldRequirements.stochastic_cut_MeV = 1.e99;
    compare(in, "unused-stochastic-cut-preserved", true);

    auto const epsilon = std::numeric_limits<double>::epsilon();
    for (double multiples : {-128., -32., 0., 32., 128.}) {
      in = inputs();
      double const endpoint = 1.e7;
      in.requirements.maximumPrimaryEnergyMeV_ =
          in.oldRequirements.maximum_primary_energy_MeV =
              endpoint * (1. + multiples * epsilon);
      compare(in, "upper-epsilon-" + std::to_string(multiples), multiples <= 64.);
      in = inputs();
      in.current.totalRateColumns_[0].spline_.axis_.low_ =
          in.original.total_rate_columns[0].spline.axis.low =
              100. * (1. + multiples * epsilon);
      compare(in, "lower-epsilon-" + std::to_string(multiples), multiples <= 64.);
    }
    for (unsigned index = 1; index < 5; ++index) {
      using namespace corsika;
      in = inputs();
      auto const pdg = in.current.totalRateColumns_[index].pdgId_;
      double const exact =
          get_mass(convert_from_PDG(static_cast<PDGCode>(pdg))) / 1_MeV + .9999 * 100.;
      in.current.totalRateColumns_[index].spline_.axis_.low_ =
          in.original.total_rate_columns[index].spline.axis.low = exact;
      compare(in, "charged-cut-safety-" + std::to_string(pdg), true);
      in.current.totalRateColumns_[index].spline_.axis_.low_ =
          in.original.total_rate_columns[index].spline.axis.low = exact + .001;
      compare(in, "charged-cut-safety-reject-" + std::to_string(pdg), false);
    }
    std::cout << "passed requirements cases=" << cases << '\n';
    return 0;
  } catch (std::exception const& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}

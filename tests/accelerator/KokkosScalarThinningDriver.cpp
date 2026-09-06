/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include "KokkosScalarThinningDriver.hpp"
#include <Kokkos_Core.hpp>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <iostream>
#include <iomanip>

namespace corsika::accelerator::testing {
ThinningComparison compareThinningOnDevice(std::vector<ThinningInput> const& inputs) {
  Kokkos::ScopeGuard runtime;
  using Exec = Kokkos::DefaultExecutionSpace;
  Kokkos::View<ThinningInput*, typename Exec::memory_space> samples("thinning_cases", inputs.size());
  auto host = Kokkos::create_mirror_view(samples);
  for (std::size_t i = 0; i < inputs.size(); ++i) host(i) = inputs[i];
  Kokkos::deep_copy(samples, host);
  Kokkos::View<gpu::em::EmThinningResult*, typename Exec::memory_space> results("thinning_results", inputs.size());
  Kokkos::parallel_for("scalar_EMThinning_oracle", inputs.size(),
      KOKKOS_LAMBDA(int i) {
        auto const& x = samples(i);
        results(i) = gpu::em::applyEmThinning(x.config, x.energy, x.weight,
                                               x.e1, x.e2, x.u1, x.u2);
      });
  auto actual = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), results);
  ThinningComparison report;
  // Scalar HEPEnergyType stores eV, device POD stores GeV. The extra scalar
  // multiplication by 1e9 can round ratios by a few ULP. Decision masks must
  // still match exactly; report weight ULP rather than weakening that check.
  for (std::size_t i = 0; i < inputs.size(); ++i) {
    auto const& a = actual(i);
    auto const& e = inputs[i].expected;
    report.decision_errors += a.keep_mask != e.keep_mask;
    auto compare = [&](double x, double y) {
      if (!std::isfinite(x) || !std::isfinite(y) || x < 0 || y < 0) {
        ++report.weight_errors; return;
      }
      std::uint64_t ix{}, iy{};
      std::memcpy(&ix, &x, sizeof(ix));
      std::memcpy(&iy, &y, sizeof(iy));
      auto ulp = ix > iy ? ix - iy : iy - ix;
      report.maximum_weight_ulp = std::max(report.maximum_weight_ulp, ulp);
      report.non_bitwise_weights += ulp != 0;
      report.weight_errors += ulp > 16;
      if (ulp > 16 && report.weight_errors <= 5) {
        auto const& t = inputs[i];
        std::cerr << std::setprecision(17) << "thinning mismatch i=" << i
                  << " actual=" << x << " expected=" << y
                  << " mask=" << a.keep_mask << '/' << e.keep_mask
                  << " E=" << t.energy << " parent_weight=" << t.weight
                  << " E1=" << t.e1 << " E2=" << t.e2
                  << " u1=" << t.u1 << " u2=" << t.u2
                  << " threshold=" << t.config.threshold_GeV
                  << " maxw=" << t.config.maximum_weight << '\n';
      }
    };
    // An erased scalar particle has no observable weight. Compare weights
    // only for surviving children, after separately checking the exact mask.
    if (e.keep_mask & 1U) compare(a.first_weight, e.first_weight);
    if (e.keep_mask & 2U) compare(a.second_weight, e.second_weight);
  }
  return report;
}
}

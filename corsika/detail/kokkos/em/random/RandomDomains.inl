/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

namespace corsika::kokkos {

  constexpr bool random_process_domains_are_unique() {
    constexpr std::uint32_t domains[]{
        InteractionDistanceRandomProcessId, InteractionColumnRandomProcessId,
        ProposalSelectionRandomProcessId, ContinuousScatteringRandomProcessId,
        MuonDecayRandomProcessId};
    for (std::uint32_t i = 0; i < 5; ++i)
      for (std::uint32_t j = i + 1; j < 5; ++j)
        if (domains[i] == domains[j]) return false;
    return true;
  }
  static_assert(random_process_domains_are_unique(),
                "independent physical draws must not share a Philox process domain");

} // namespace corsika::kokkos

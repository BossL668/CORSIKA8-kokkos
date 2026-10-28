/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

namespace corsika::kokkos {
  inline LeptonVertexSelectionBatchResult
  EmBackend::reselectLeptonInteractionsAtVertexForValidation(
      std::vector<InteractionRecord> const& interactions) {
    return selectLeptonVerticesForValidation(interactions);
  }
} // namespace corsika::kokkos

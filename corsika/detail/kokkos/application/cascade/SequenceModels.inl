/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */
#pragma once

namespace corsika::kokkos::application {
  template <typename TModels, typename TSequence>
  TModels const& complete_models(TModels const& models, TSequence&) {
    return models;
  }

  template <typename TC, typename TZ, typename TE, typename TP, typename TS>
  auto complete_models(SequenceModels<TC, TZ, TE, TP> const& models, TS& sequence) {
    using namespace process_selection;
    return CascadeModels{models.coreas_,
                         models.zhs_,
                         models.emCascade_,
                         models.emContinuousProposal_,
                         select_process<Neutrino>(sequence),
                         select_process<HadronSwitch>(sequence),
                         select_process<DecaySwitch>(sequence),
                         select_process<ContinuousSwitch>(sequence),
                         select_process<Longitudinal>(sequence),
                         select_process<Production>(sequence),
                         select_process<Cut>(sequence)};
  }
} // namespace corsika::kokkos::application

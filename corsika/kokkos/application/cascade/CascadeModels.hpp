/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */
#pragma once

namespace corsika::kokkos::application {
  /** Borrow the physics and output processes for an application run.
   * All referenced processes must outlive the binding. No model is copied or
   * reconstructed. The event-local inspector and thinning process are supplied
   * separately when assembling the process sequence.
   */
  template <typename TC, typename TZ, typename TE, typename TP, typename TN, typename TH,
            typename TD, typename TContinuous, typename TL, typename TProduction,
            typename TCut>
  struct CascadeModels {
    TC& coreas_;
    TZ& zhs_;
    TE& emCascade_;
    TP& emContinuousProposal_;
    TN& neutrinoPrimary_;
    TH& hadronSequence_;
    TD& decaySequence_;
    TContinuous& emContinuous_;
    TL& longitudinal_;
    TProduction& production_;
    TCut& cut_;
  };
  template <typename TC, typename TZ, typename TE, typename TP, typename TN, typename TH,
            typename TD, typename TContinuous, typename TL, typename TProduction,
            typename TCut>
  CascadeModels(TC&, TZ&, TE&, TP&, TN&, TH&, TD&, TContinuous&, TL&, TProduction&, TCut&)
      ->CascadeModels<TC, TZ, TE, TP, TN, TH, TD, TContinuous, TL, TProduction, TCut>;

} // namespace corsika::kokkos::application

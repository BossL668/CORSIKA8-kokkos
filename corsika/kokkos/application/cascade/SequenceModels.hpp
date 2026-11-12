/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */
#pragma once

#include <corsika/kokkos/application/cascade/CascadeModels.hpp>
#include <corsika/kokkos/application/cascade/CascadeProcessSelection.hpp>

namespace corsika::kokkos::application {
  /** Borrow the run-lifetime models needed before an event sequence exists.
   * The remaining models are obtained from that event's actual sequence.
   */
  template <typename TC, typename TZ, typename TE, typename TP>
  struct SequenceModels {
    TC& coreas_;
    TZ& zhs_;
    TE& emCascade_;
    TP& emContinuousProposal_;
  };
  template <typename TC, typename TZ, typename TE, typename TP>
  SequenceModels(TC&, TZ&, TE&, TP&)->SequenceModels<TC, TZ, TE, TP>;

  /** Keep explicitly bound models available to existing application adapters. */
  template <typename TModels, typename TSequence>
  TModels const& complete_models(TModels const&, TSequence&);
  /** Borrow the remaining models from the event sequence without evaluating switches. */
  template <typename TC, typename TZ, typename TE, typename TP, typename TS>
  auto complete_models(SequenceModels<TC, TZ, TE, TP> const&, TS&);
} // namespace corsika::kokkos::application

#include <corsika/detail/kokkos/application/cascade/SequenceModels.inl>

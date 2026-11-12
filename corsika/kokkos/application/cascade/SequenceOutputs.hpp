/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */
#pragma once

#include <corsika/kokkos/application/cascade/CascadeContext.hpp>
#include <corsika/kokkos/application/cascade/CascadeProcessSelection.hpp>

namespace corsika::kokkos::application {
  /** Borrow observers for run setup; retrieve event writers from their processes. */
  template <typename TC, typename TZ>
  struct SequenceOutputs {
    TC& coreasDetector_;
    TZ& zhsDetector_;
  };
  template <typename TC, typename TZ>
  SequenceOutputs(TC&, TZ&)->SequenceOutputs<TC, TZ>;

  /** Keep explicitly bound outputs available to existing application adapters. */
  template <typename TOutputs, typename TSequence>
  TOutputs const& complete_outputs(TOutputs const&, TSequence&);
  /** Borrow writers from the same processes used by the event sequence. */
  template <typename TC, typename TZ, typename TS>
  auto complete_outputs(SequenceOutputs<TC, TZ> const&, TS&);
} // namespace corsika::kokkos::application

#include <corsika/detail/kokkos/application/cascade/SequenceOutputs.inl>

/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */
#pragma once

namespace corsika::kokkos::application {
  template <typename TOutputs, typename TSequence>
  TOutputs const& complete_outputs(TOutputs const& outputs, TSequence&) {
    return outputs;
  }

  template <typename TC, typename TZ, typename TS>
  auto complete_outputs(SequenceOutputs<TC, TZ> const& outputs, TS& sequence) {
    using namespace process_selection;
    return CascadeOutputs{select_process<Cut>(sequence).getWriter(),
                          select_process<Longitudinal>(sequence).getWriter(),
                          select_process<Production>(sequence).getWriter(),
                          select_process<Observation>(sequence),
                          select_process<Interactions>(sequence),
                          outputs.coreasDetector_,
                          outputs.zhsDetector_};
  }
} // namespace corsika::kokkos::application

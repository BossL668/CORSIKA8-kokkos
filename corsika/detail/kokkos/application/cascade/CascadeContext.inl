/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once
namespace corsika::kokkos::application::detail {
  template <typename TContext, typename TFunction>
  decltype(auto) with_cascade_arguments(TContext const& context, TFunction&& function) {
    auto const& g = context.geometry_;
    auto const& o = context.outputs_;
    auto const& p = context.processes_;
    auto const& e = context.execution_;
    auto const& d = context.diagnostics_;
    return std::forward<TFunction>(function)(
        g.environment_, g.coordinates_, g.injectionPosition_, g.surface_,
        g.propagationStep_, o.coreasDetector_, o.zhsDetector_, g.showerAxis_,
        g.depthStep_, g.observationHeightM_, g.coreXM_, g.coreYM_, g.magneticFieldT_,
        o.energyLoss_, o.longitudinal_, o.production_, o.observation_, o.interactions_,
        p.coreas_, p.zhs_, p.emCascade_, p.emContinuousProposal_, p.stackInspector_,
        p.neutrinoPrimary_, p.hadronSequence_, p.decaySequence_, p.emContinuous_,
        p.longitudinalProcess_, p.productionProcess_, p.thinning_, p.cut_, e.sequence_,
        e.tracking_, e.stack_, e.output_, e.hadronicPool_.get_ptr(), e.configurePrimary_,
        d.photoHigh_, d.photoLow_, d.photoHighBefore_, d.photoLowBefore_, d.highCounter_,
        d.lowCounter_, d.highInteractionsBefore_, d.lowInteractionsBefore_,
        d.highTimingsBefore_, d.lowTimingsBefore_, d.poolBefore_, d.photoFallback_);
  }
} // namespace corsika::kokkos::application::detail

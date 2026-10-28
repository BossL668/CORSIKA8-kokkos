/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

namespace corsika::kokkos::application {
  template <typename TCut>
  CascadeEventConfig make_cascade_defaults(CLI::App const& app, std::uint64_t seed,
                                           HEPEnergyType maximum_energy, Code primary,
                                           TCut const& cut,
                                           HEPEnergyType production_threshold,
                                           HEPEnergyType hadronic_threshold,
                                           bool keep_zero_weight) {
    CascadeEventConfig result;
    result.seed_ = seed;
    result.configuredPrimaryEnergyMax_ = maximum_energy;
    result.emCut_ = cut.getElectronKineticECut();
    result.muonCut_ = cut.getMuonKineticECut();
    result.productionThreshold_ = production_threshold;
    result.highEnergyHadronicThreshold_ = hadronic_threshold;
    result.emThinningFraction_ = app["--emthin"]->as<double>();
    result.automaticMaximumWeight_ = app["--max-weight"]->as<double>() <= 0.;
    result.keepZeroWeightParticles_ = keep_zero_weight;
    result.beamCode_ = primary;
    result.highEnergyHadronicModel_ = app["--hadronModel"]->as<std::string>();
    result.maximumMagneticDeflectionRad_ = app["--max-deflection-angle"]->as<double>();
    return result;
  }

  template <typename TE, typename TC, typename TP, typename TS, typename TStep,
            typename TAxis, typename TDepth>
  auto make_cascade_geometry(TE& environment, TC const& coordinates, TP const& injection,
                             TS const& surface, TStep const& step, TAxis const& axis,
                             TDepth const& depth, RunEnvironmentConfig const& run) {
    return CascadeGeometry{environment,
                           coordinates,
                           injection,
                           surface,
                           step,
                           axis,
                           depth,
                           run.observationPlanePointM_[2],
                           run.observationPlanePointM_[0],
                           run.observationPlanePointM_[1],
                           run.magneticFieldT_};
  }

} // namespace corsika::kokkos::application

/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once
#include <corsika/kokkos/application/cascade/CascadeEventConfig.hpp>
#include <corsika/kokkos/application/BackendOptions.hpp>
#include <corsika/kokkos/application/cascade/PreparedCascade.hpp>
#include <corsika/kokkos/application/cascade/atmosphere/AtmospherePolicy.hpp>
#include <corsika/kokkos/application/cascade/ice/HomogeneousIcePolicy.hpp>
#include <corsika/kokkos/application/cascade/firn/FirnPolicy.hpp>
#include <corsika/kokkos/radio/ice/HomogeneousRadioConfig.hpp>
#include <corsika/kokkos/environment/EnvironmentPreparation.hpp>
#include <corsika/kokkos/radio/atmosphere/FlatAtmosphereConfig.hpp>
#include <corsika/media/MediumProperties.hpp>
#include <corsika/modules/proposal/ProposalProcessBase.hpp>
#include <corsika/output/ReplayTrace.hpp>
#include <algorithm>
#include <array>
#include <stdexcept>
namespace corsika::kokkos::application {
  /** Export the actual environment and prepare the unchanged production configuration.
   * No particle is transported and no backend or model is initialized here.
   * Material and radio preparation follow the explicit scene choice. The generic
   * exporter and backend scheduling remain independent of that choice.
   * magneticFieldT retains the production call contract. The field actually used
   * comes from the native environment's node models, not from that argument.
   */
  template <typename TRunSession, typename TEnvironment, typename TInjectionPosition,
            typename TSurface, typename TPropagationStep, typename TDetectorCoREAS,
            typename TDetectorZHS, typename TShowerAxis, typename TDepthStep,
            typename TEnergyLossWriter, typename TLongitudinalWriter>
  auto prepare_cascade(TRunSession const& session, BackendOptions const& options,
                       CascadeEventConfig const& event, TEnvironment& environment,
                       CoordinateSystemPtr const& rootCoordinates,
                       TInjectionPosition const& injectionPosition,
                       TSurface const& surface, TPropagationStep const& propagationStep,
                       TDetectorCoREAS& coreasDetector, TDetectorZHS& zhsDetector,
                       TShowerAxis const& showerAxis, TDepthStep const& depthStep,
                       double observationHeightM, double showerCoreXM,
                       double showerCoreYM, std::array<double, 3> const& magneticFieldT,
                       TEnergyLossWriter& energyLossWriter,
                       TLongitudinalWriter& longitudinalWriter,
                       ObservationPlane const* observationPlane = nullptr);
} // namespace corsika::kokkos::application

#include <corsika/detail/kokkos/application/cascade/CascadePreparation.inl>

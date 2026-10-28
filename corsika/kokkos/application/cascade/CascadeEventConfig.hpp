/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once
#include <corsika/framework/core/ParticleProperties.hpp>
#include <corsika/framework/core/PhysicalUnits.hpp>
#include <corsika/media/CORSIKA7Atmospheres.hpp>
#include <corsika/kokkos/application/cascade/CascadeScene.hpp>
#include <cstdint>
#include <string>
namespace corsika::kokkos::application {
  struct CascadeEventConfig {
    std::uint64_t seed_{};
    std::uint64_t showerOrdinal_{};
    unsigned int outputShowerId_{};
    HEPEnergyType configuredPrimaryEnergyMax_{};
    HEPEnergyType primaryTotalEnergy_{};
    HEPEnergyType emCut_{};
    HEPEnergyType muonCut_{};
    HEPEnergyType productionThreshold_{};
    HEPEnergyType highEnergyHadronicThreshold_{};
    double emThinningFraction_{};
    double maximumWeight_{};
    bool automaticMaximumWeight_{};
    bool thinningCanActivateFromUnitWeight_{};
    bool keepZeroWeightParticles_{};
    Code beamCode_{};
    std::string highEnergyHadronicModel_;
    double maximumMagneticDeflectionRad_{};
    // Appended to preserve existing aggregate initialization and air defaults.
    AtmosphereId atmosphereId_{AtmosphereId::USStdBK};
    // Comparison-only opt-out. Standard air callers retain both algorithms.
    bool radioZhsEnabled_{true};
    CascadeScene scene_{CascadeScene::Atmosphere};
  };
} // namespace corsika::kokkos::application

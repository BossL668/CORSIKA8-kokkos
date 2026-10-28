/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/kokkos/application/BackendOptions.hpp>
#include <corsika/kokkos/em/execution/BackendConfig.hpp>
#include <corsika/kokkos/scheduling/HadronicProcessPool.hpp>
#include <corsika/kokkos/scheduling/HadronicWorkQueue.hpp>
#include <corsika/framework/core/ParticleProperties.hpp>
#include <corsika/framework/core/PhysicalUnits.hpp>
#include <cstddef>
#include <cstdint>
#include <string>

namespace corsika::kokkos::application {
  /** Serialize the complete production hybrid-shower report.
   * This retains diagnostic trace emission, retrospective hadronic work plans,
   * energy-closure checks and the original YAML schema and insertion order.
   * Transport state is borrowed and is not advanced by this report.
   */
  struct ShowerReportBuilder {
    template <typename TStepRegistry, typename TSequence, typename TRunSession,
              typename TBackend, typename TEas, typename TRouter, typename TOutputSink,
              typename TFallbackHandler, typename TPhotoHighEnergy,
              typename TPhotoLowEnergy, typename TPhotoHighStatistics,
              typename TPhotoLowStatistics, typename THighEnergyCounter,
              typename TLowEnergyCounter, typename TPhotoFallback>
    static void record(TRunSession& cudaSession, BackendOptions const& gpuCli,
                       TBackend& backend, TEas& EAS, TRouter& router,
                       TOutputSink& outputSink, TFallbackHandler& fallbackHandler,
                       TPhotoHighEnergy& photoHadronicHighEnergy,
                       TPhotoLowEnergy& photoHadronicLowEnergy,
                       TPhotoHighStatistics const& photoHadronicBefore,
                       TPhotoLowStatistics const& photoHadronicLeBefore,
                       THighEnergyCounter& heCounted, TLowEnergyCounter& leIntCounted,
                       std::uint64_t highEnergyHadronicInteractionsBefore,
                       std::uint64_t lowEnergyHadronicInteractionsBefore,
                       std::size_t highEnergyHadronicTimingsBefore,
                       std::size_t lowEnergyHadronicTimingsBefore,
                       HadronicWorkClassifierConfig const& hadronicWorkClassifier,
                       HadronicProcessPool* hadronicProcessPool,
                       HadronicProcessPoolStatistics const& hadronicPoolStatisticsBefore,
                       TPhotoFallback const& photoHadronicQgsjetFallback,
                       std::string const& highEnergyHadronicModel,
                       HEPEnergyType heHadronModelThreshold, double emthinfrac,
                       double maxWeight, bool automaticMaxWeight,
                       bool thinningCanActivateFromUnitWeight,
                       bool backendReusedForShower, bool gpuMuonTransportAvailable,
                       bool gpuRadioEnabled,
                       corsika::kokkos::BackendConfig const& gpuConfig, Code beamCode,
                       HEPEnergyType primaryTotalEnergy, unsigned int outputShowerId);
  };
} // namespace corsika::kokkos::application

#include <corsika/detail/kokkos/application/cascade/ShowerReport.inl>

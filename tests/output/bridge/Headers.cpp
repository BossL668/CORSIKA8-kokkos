/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */
#ifndef CORSIKA_OUTPUT_BRIDGE_NATIVE_ONLY
#include <corsika/kokkos/output/CorsikaOutputSink.hpp>
#include <corsika/kokkos/output/EmRunOutput.hpp>
#include <corsika/kokkos/output/OutputRecordConversion.hpp>
#include <corsika/kokkos/output/OutputSinkStatistics.hpp>
#include <corsika/kokkos/output/RadioTrackRecord.hpp>
#include <corsika/modules/writers/FirstInteractionSnapshot.hpp>
#include <type_traits>
static_assert(std::is_standard_layout_v<corsika::kokkos::RadioTrackRecord>);
static_assert(std::is_trivially_copyable_v<corsika::kokkos::RadioTrackRecord>);
#else
#include <corsika/modules/writers/LongitudinalWriter.hpp>
#include <corsika/modules/writers/EnergyLossWriter.hpp>
#include <corsika/modules/writers/ProductionWriter.hpp>
#include <corsika/modules/writers/InteractionWriter.hpp>
#include <corsika/modules/writers/FirstInteractionSnapshot.hpp>
#include <corsika/modules/ObservationPlane.hpp>
#include <corsika/modules/radio/observers/TimeDomainObserver.hpp>
#endif

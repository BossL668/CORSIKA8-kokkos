/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/kokkos/radio/firn/GreenResponseBackend.hpp>
#include <corsika/kokkos/radio/firn/UniformAnalyticBackend.hpp>
#include <corsika/kokkos/radio/firn/StraightAverageIndexBackend.hpp>
#include <corsika/kokkos/radio/firn/StratifiedGOBackend.hpp>
#include <corsika/kokkos/radio/firn/StratifiedTurningGOBackend.hpp>
#include <corsika/kokkos/radio/firn/StratifiedSurfaceReflectionBackend.hpp>
#include <corsika/kokkos/radio/firn/CoherentStratifiedGOBackend.hpp>
#include <corsika/kokkos/radio/firn/AnyGreenBackend.hpp>
#include <corsika/kokkos/radio/firn/TabulatedFullWaveBackend.hpp>
#include <corsika/kokkos/radio/firn/TabulatedCoherentFullWaveBackend.hpp>
#include <corsika/kokkos/radio/firn/HybridCoherentFullWaveBackend.hpp>
#include <corsika/detail/kokkos/radio/firn/GreenResponseCommon.inl>

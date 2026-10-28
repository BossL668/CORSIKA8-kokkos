/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include <corsika/kokkos/em/algorithms/NativeMuonPairCutRecord.hpp>
#include <type_traits>
static_assert(std::is_same_v<decltype(corsika::kokkos::native_muon_pair_cut_record(
                                 corsika::kokkos::InteractionRecord{})),
                             corsika::kokkos::LeptonTransportRecord>);

/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */
#pragma once
#include <corsika/kokkos/runtime/BackendKind.hpp>

namespace corsika::kokkos {
  struct BackendCapabilities {
    BackendKind kind_{BackendKind::OpenMP};
    bool photonTransport_{};
    bool electronTransport_{};
    bool positronTransport_{};
    bool muonTransport_{};
    bool residentProfile_{};
    bool coreasRadio_{};
    bool zhsRadio_{};
    bool deterministic_{};
  };
} // namespace corsika::kokkos

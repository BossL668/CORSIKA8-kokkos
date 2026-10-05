#pragma once

// Native C++/Kokkos electron, positron and photon backend. Hadronic processes
// and PROPOSAL muons remain supplied by the C8 application. Linking requires
// CORSIKA8NativeEgs4 (CORSIKA_ENABLE_EGS4=ON); no Fortran EM runtime is used.
#include <corsika/modules/egs4/EmbeddedTables.hpp>
#include <corsika/modules/egs4/Session.hpp>

// Keep this host-facing facade free of Kokkos device headers. Code that queries
// device tables includes <corsika/modules/egs4/Tables.hpp> explicitly and uses
// the matching backend compiler (NVCC for CUDA).

// Selection belongs solely to the independent interface library. A package may
// provide several backends; each build of this module instantiates exactly one.
#pragma once
#include <Kokkos_Core.hpp>
namespace corsika::interfaces::kokkos {
#if defined(C8_INTERFACE_EXECUTION_OPENMP)
#ifndef KOKKOS_ENABLE_OPENMP
#error "Interface OPENMP execution requires an OpenMP-enabled Kokkos package"
#endif
using ExecutionSpace = Kokkos::OpenMP;
#elif defined(C8_INTERFACE_EXECUTION_CUDA)
#ifndef KOKKOS_ENABLE_CUDA
#error "Interface CUDA execution requires a CUDA-enabled Kokkos package"
#endif
using ExecutionSpace = Kokkos::Cuda;
#else
using ExecutionSpace = Kokkos::DefaultExecutionSpace;
#endif
} // namespace corsika::interfaces::kokkos

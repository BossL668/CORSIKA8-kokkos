#pragma once
#include <Kokkos_Core.hpp>
#include <type_traits>

namespace corsika::accelerator::em::kokkos_detail {
// SharedHostPinnedSpace follows the package's default device. In a dual
// package OpenMP must still use ordinary RAM, not CUDA-pinned staging memory.
template <class ExecutionSpace>
using HostStagingSpace = std::conditional_t<
    std::is_same_v<typename ExecutionSpace::memory_space, Kokkos::HostSpace>,
    Kokkos::HostSpace, Kokkos::SharedHostPinnedSpace>;
} // namespace corsika::accelerator::em::kokkos_detail

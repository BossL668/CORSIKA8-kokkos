# The build scripts use -DCMAKE_TOOLCHAIN_FILE, not a Conan CMake preset.
# Read the selected package identity from the toolchain too. Upgrade old
# NONE/native defaults, but never silently override an explicit conflicting
# backend or architecture. Keep CMakeCache.txt and backend.json consistent.
foreach (_c8_option BACKEND ARCHITECTURE CUDA_ARCHITECTURES)
  set (_c8_name "CORSIKA_KOKKOS_${_c8_option}")
  set (_c8_from_conan "C8_CONAN_KOKKOS_${_c8_option}")
  if (DEFINED ${_c8_from_conan})
    set (_c8_unspecified FALSE)
    if ((_c8_option STREQUAL "ARCHITECTURE" AND
         "${${_c8_name}}" STREQUAL "NONE") OR
        (_c8_option STREQUAL "CUDA_ARCHITECTURES" AND
         "${${_c8_name}}" STREQUAL "native"))
      set (_c8_unspecified TRUE)
    endif ()
    if (DEFINED ${_c8_name} AND NOT _c8_unspecified AND
        NOT "${${_c8_name}}" STREQUAL "${${_c8_from_conan}}")
      message (FATAL_ERROR
        "${_c8_name}=${${_c8_name}} conflicts with the Conan profile "
        "(${${_c8_from_conan}}). Use matching options or a fresh build directory.")
    endif ()
    set (${_c8_name} "${${_c8_from_conan}}" CACHE STRING
         "Kokkos package selection from the Conan host profile" FORCE)
  endif ()
endforeach ()

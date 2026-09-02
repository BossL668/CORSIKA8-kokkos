# Internal accelerator code must remain reusable by air, ice, and future
# environments.  Keep application, reporting, and generator dependencies out
# of the execution-space-neutral runner/session layer.
if (NOT DEFINED C8_SOURCE_DIR)
  message(FATAL_ERROR "C8_SOURCE_DIR is required")
endif ()

file(GLOB_RECURSE _c8_kokkos_common_files
  "${C8_SOURCE_DIR}/corsika/accelerator/em/*.hpp"
  "${C8_SOURCE_DIR}/corsika/accelerator/em/detail/*.hpp"
  "${C8_SOURCE_DIR}/corsika/accelerator/em/kokkos/*.hpp"
  "${C8_SOURCE_DIR}/corsika/accelerator/radio/detail/*.hpp"
  "${C8_SOURCE_DIR}/corsika/accelerator/radio/kokkos/*.hpp"
  "${C8_SOURCE_DIR}/src/accelerator/em/kokkos/*.cpp")

set(_c8_forbidden_patterns
  "CLI/CLI.hpp"
  "yaml-cpp"
  "modules/fluka"
  "FLUKA"
  "air_shower"
  "c8_air_shower"
  "c8_ice_cascade")

foreach (_c8_file IN LISTS _c8_kokkos_common_files)
  file(READ "${_c8_file}" _c8_contents)
  foreach (_c8_pattern IN LISTS _c8_forbidden_patterns)
    string(FIND "${_c8_contents}" "${_c8_pattern}" _c8_found)
    if (NOT _c8_found EQUAL -1)
      message(FATAL_ERROR
        "Portable accelerator boundary violation: ${_c8_file} contains ${_c8_pattern}")
    endif ()
  endforeach ()
endforeach ()

message(STATUS "Portable Kokkos accelerator dependency boundary is clean")

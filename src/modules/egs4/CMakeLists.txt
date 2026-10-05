# Optional native EM module. Application wiring belongs in applications/.
if(NOT CORSIKA_ENABLE_KOKKOS)
  message(FATAL_ERROR "CORSIKA_ENABLE_EGS4 requires CORSIKA_ENABLE_KOKKOS")
endif()
if(NOT CORSIKA_KOKKOS_BACKEND MATCHES "^(OPENMP|CUDA|CUDA_OPENMP)$")
  message(FATAL_ERROR "EGS4 currently supports OpenMP and CUDA builds")
endif()
get_filename_component(_c8_egs4_source_root "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)
set(CORSIKA_EGS4_TABLE "${_c8_egs4_source_root}/resources/c7_egs4/EGSDAT6_.4" CACHE FILEPATH
  "AIR-NTP EGSDAT table embedded at build time (defaults to project resources)")
if(NOT EXISTS "${CORSIKA_EGS4_TABLE}")
  message(FATAL_ERROR "EGS4 table missing: ${CORSIKA_EGS4_TABLE}; restore resources/c7_egs4/EGSDAT6_.4 or override CORSIKA_EGS4_TABLE")
endif()
file(READ "${CORSIKA_EGS4_TABLE}" C8_EGSDAT_CONTENT)
if(NOT C8_EGSDAT_CONTENT MATCHES "^ MEDIUM=AIR-NTP" OR C8_EGSDAT_CONTENT MATCHES "\\)C8_EGSDAT\"")
  message(FATAL_ERROR "Invalid AIR-NTP table for embedding")
endif()
file(SHA256 "${CORSIKA_EGS4_TABLE}" C8_EGSDAT_SHA256)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${CORSIKA_EGS4_TABLE}")
configure_file(EmbeddedTables.cpp.in "${CMAKE_CURRENT_BINARY_DIR}/EmbeddedEgs4Tables.cpp" @ONLY)

add_library(CORSIKA8NativeEgs4 STATIC
  Tables.cpp Session.cpp "${CMAKE_CURRENT_BINARY_DIR}/EmbeddedEgs4Tables.cpp")
add_library(CORSIKA8::NativeEgs4 ALIAS CORSIKA8NativeEgs4)
target_compile_features(CORSIKA8NativeEgs4 PUBLIC cxx_std_17)
target_include_directories(CORSIKA8NativeEgs4 PUBLIC
  $<BUILD_INTERFACE:${_c8_egs4_source_root}>
  $<INSTALL_INTERFACE:include>)
target_link_libraries(CORSIKA8NativeEgs4 PUBLIC CORSIKA8 Kokkos::kokkos)
if(CORSIKA_KOKKOS_BACKEND MATCHES "^CUDA")
  set_source_files_properties(Tables.cpp Session.cpp
    "${CMAKE_CURRENT_BINARY_DIR}/EmbeddedEgs4Tables.cpp" PROPERTIES LANGUAGE CUDA)
  target_compile_options(CORSIKA8NativeEgs4 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
  set_target_properties(CORSIKA8NativeEgs4 PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON
    CUDA_ARCHITECTURES "${CORSIKA_KOKKOS_CUDA_ARCHITECTURES}")
  target_link_libraries(CORSIKA8NativeEgs4 PUBLIC CUDA::cudart CUDA::cuda_driver)
endif()

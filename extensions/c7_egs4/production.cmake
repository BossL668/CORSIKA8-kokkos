# Native EM library attached to the existing application. No secondary CLI,
# fixture geometry or external Fortran EM implementation enters this target.
if(NOT CORSIKA_ENABLE_KOKKOS)
  message(FATAL_ERROR "CORSIKA_ENABLE_EGS4 requires CORSIKA_ENABLE_KOKKOS")
endif()
if(NOT CORSIKA_KOKKOS_BACKEND MATCHES "^(OPENMP|CUDA|CUDA_OPENMP)$")
  message(FATAL_ERROR "EGS4 currently supports OpenMP and CUDA builds")
endif()
get_filename_component(_c8_egs4_default_table
  "${CMAKE_CURRENT_LIST_DIR}/../../resources/c7_egs4/EGSDAT6_.4" ABSOLUTE)
set(CORSIKA_EGS4_TABLE "${_c8_egs4_default_table}" CACHE FILEPATH
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
configure_file("${CMAKE_CURRENT_LIST_DIR}/EmbeddedTables.cpp.in"
  "${CMAKE_CURRENT_BINARY_DIR}/EmbeddedEgs4Tables.cpp" @ONLY)
add_library(CORSIKA8NativeEgs4 STATIC
  "${CMAKE_CURRENT_LIST_DIR}/Egs4Tables.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/Egs4C8Session.cpp"
  "${CMAKE_CURRENT_BINARY_DIR}/EmbeddedEgs4Tables.cpp")
target_compile_features(CORSIKA8NativeEgs4 PUBLIC cxx_std_17)
target_include_directories(CORSIKA8NativeEgs4 PUBLIC "${CMAKE_CURRENT_LIST_DIR}")
target_link_libraries(CORSIKA8NativeEgs4 PUBLIC CORSIKA8 Kokkos::kokkos)
if(CORSIKA_KOKKOS_BACKEND MATCHES "^CUDA")
  set_source_files_properties("${CMAKE_CURRENT_LIST_DIR}/Egs4Tables.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/Egs4C8Session.cpp"
    "${CMAKE_CURRENT_BINARY_DIR}/EmbeddedEgs4Tables.cpp" PROPERTIES LANGUAGE CUDA)
  target_compile_options(CORSIKA8NativeEgs4 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
  set_target_properties(CORSIKA8NativeEgs4 PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON
    CUDA_ARCHITECTURES "${CORSIKA_KOKKOS_CUDA_ARCHITECTURES}")
  target_link_libraries(CORSIKA8NativeEgs4 PUBLIC CUDA::cudart CUDA::cuda_driver)
endif()
target_link_libraries(c8_air_shower PRIVATE CORSIKA8NativeEgs4)
target_compile_definitions(c8_air_shower PRIVATE CORSIKA8_WITH_NATIVE_EGS4=1)

option(CORSIKA_EGS4_BUILD_TESTS "Build the embedded-table regression test" OFF)
if(CORSIKA_EGS4_BUILD_TESTS)
  add_executable(test_c8_egs4_embedded "${CMAKE_CURRENT_LIST_DIR}/test_embedded_tables.cpp")
  target_link_libraries(test_c8_egs4_embedded PRIVATE CORSIKA8NativeEgs4)
  if(CORSIKA_KOKKOS_BACKEND MATCHES "^CUDA")
    set_source_files_properties("${CMAKE_CURRENT_LIST_DIR}/test_embedded_tables.cpp" PROPERTIES LANGUAGE CUDA)
    set_target_properties(test_c8_egs4_embedded PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON
      CUDA_ARCHITECTURES "${CORSIKA_KOKKOS_CUDA_ARCHITECTURES}")
  endif()
  add_test(NAME c8_egs4_embedded_tables COMMAND test_c8_egs4_embedded "${CORSIKA_EGS4_TABLE}")
  set_tests_properties(c8_egs4_embedded_tables PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=2;OMP_PROC_BIND=false")
endif()

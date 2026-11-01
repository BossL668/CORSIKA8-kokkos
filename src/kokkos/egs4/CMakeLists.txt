#
# (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
#
# This software is distributed under the terms of the 3-clause BSD license.
# See file LICENSE for a full version of the license.
#

if (NOT CORSIKA_KOKKOS_RESOLVED_BACKEND MATCHES "^(OPENMP|CUDA|CUDA_OPENMP)$")
  message (FATAL_ERROR "The native EGS4 backend currently supports OpenMP and CUDA.")
endif ()
set (CORSIKA_EGS4_TABLE "${PROJECT_SOURCE_DIR}/resources/c7_egs4/EGSDAT6_.4"
  CACHE FILEPATH "AIR-NTP table embedded in the EGS4 backend")
if (NOT EXISTS "${CORSIKA_EGS4_TABLE}")
  message (FATAL_ERROR "EGS4 table not found: ${CORSIKA_EGS4_TABLE}")
endif ()
file (READ "${CORSIKA_EGS4_TABLE}" C8_EGSDAT_CONTENT)
if (NOT C8_EGSDAT_CONTENT MATCHES "^ MEDIUM=AIR-NTP" OR
    C8_EGSDAT_CONTENT MATCHES "\\)C8_EGSDAT\"")
  message (FATAL_ERROR "CORSIKA_EGS4_TABLE must contain a valid AIR-NTP table.")
endif ()
file (SHA256 "${CORSIKA_EGS4_TABLE}" C8_EGSDAT_SHA256)
set_property (DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${CORSIKA_EGS4_TABLE}")
configure_file (EmbeddedTables.cpp.in EmbeddedTables.cpp @ONLY)
add_library (CORSIKA8NativeEgs4 STATIC Tables.cpp Session.cpp
  "${CMAKE_CURRENT_BINARY_DIR}/EmbeddedTables.cpp")
add_library (CORSIKA8::NativeEgs4 ALIAS CORSIKA8NativeEgs4)
set_target_properties (CORSIKA8NativeEgs4 PROPERTIES EXPORT_NAME NativeEgs4)
target_compile_features (CORSIKA8NativeEgs4 PUBLIC cxx_std_17)
target_link_libraries (CORSIKA8NativeEgs4 PUBLIC CORSIKA8 Kokkos::kokkos)
if (CORSIKA_KOKKOS_RESOLVED_BACKEND MATCHES "^CUDA")
  set_source_files_properties (Tables.cpp Session.cpp
    "${CMAKE_CURRENT_BINARY_DIR}/EmbeddedTables.cpp" PROPERTIES LANGUAGE CUDA)
  set_target_properties (CORSIKA8NativeEgs4 PROPERTIES
    CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON CUDA_EXTENSIONS OFF)
  target_compile_options (CORSIKA8NativeEgs4 PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
  target_link_libraries (CORSIKA8NativeEgs4 PUBLIC CUDA::cudart)
endif ()
install (TARGETS CORSIKA8NativeEgs4 EXPORT CORSIKA8PublicTargets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install (FILES "${PROJECT_SOURCE_DIR}/resources/c7_egs4/COPYING.CORSIKA7"
  "${PROJECT_SOURCE_DIR}/resources/c7_egs4/README.md"
  DESTINATION ${CMAKE_INSTALL_DATADIR}/corsika/egs4)

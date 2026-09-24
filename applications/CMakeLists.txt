SET(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")

add_executable (c8_air_shower c8_air_shower.cpp)
target_link_libraries (c8_air_shower PRIVATE CORSIKA8)


if (CORSIKA_ENABLE_KOKKOS)
  add_executable (kokkos_backend_probe kokkos_backend_probe.cpp)
  target_link_libraries (
    kokkos_backend_probe PRIVATE CORSIKA8KokkosEm CLI11::CLI11)
  install (TARGETS kokkos_backend_probe DESTINATION bin)

  add_executable (c8_kokkos_tune c8_kokkos_tune.cpp)
  target_link_libraries (
    c8_kokkos_tune PRIVATE CORSIKA8KokkosEm CLI11::CLI11)

  target_sources (
    c8_air_shower PRIVATE
    detail/air_shower_kokkos/KokkosRunSession.cpp)
  target_link_libraries (c8_air_shower PRIVATE CORSIKA8KokkosEm)
  target_compile_definitions (
    c8_air_shower PRIVATE
    CORSIKA8_WITH_KOKKOS_EM=1
    CORSIKA8_KOKKOS_VERSION="${Kokkos_VERSION}")
endif ()

set_target_properties(c8_air_shower PROPERTIES
  INSTALL_RPATH "$ORIGIN/../${CMAKE_INSTALL_LIBDIR}/corsika"
)



target_compile_definitions (
  c8_air_shower PRIVATE
  CORSIKA8_BUNDLED_IGRF14_FILE="${CMAKE_SOURCE_DIR}/resources/GeoMag/IGRF14.COF"
)

# FLUKA discovery and application selection historically used two independent
# switches.  If a valid FLUKA library was discovered, make it the application
# default so a fresh build cannot silently select UrQMD instead.  An explicit
# -DWITH_FLUKA=OFF still remains available.
if (NOT DEFINED WITH_FLUKA)
  if (TARGET fluka)
    set (WITH_FLUKA ON CACHE BOOL
         "Use FLUKA for low-energy hadronic interactions")
  else ()
    set (WITH_FLUKA OFF CACHE BOOL
         "Use FLUKA for low-energy hadronic interactions")
  endif ()
endif ()

if (WITH_FLUKA AND NOT TARGET fluka)
  message (FATAL_ERROR
           "WITH_FLUKA=ON, but no valid FLUKA library was configured. "
           "Set C8_FLUKALIB or FLUPRO to a directory containing libflukahp.a.")
endif ()

# Isolated C7 comparison application: never changes the standard air target.
# Opt-in until atmosphere/radio/thinning correspondence has been validated.
option(CORSIKA_BUILD_C7_COMPARISON "Build the QGSJET-III/FLUKA C7 comparison" OFF)
if (CORSIKA_BUILD_C7_COMPARISON)
  if (NOT WITH_FLUKA)
    message(FATAL_ERROR "C7 comparison requires WITH_FLUKA=ON")
  endif ()
  add_executable(c8_corsika7_compare c8_corsika7_compare.cpp)
  target_link_libraries(c8_corsika7_compare PRIVATE CORSIKA8)
  target_compile_definitions(c8_corsika7_compare PRIVATE WITH_FLUKA)
  if (CORSIKA_ENABLE_KOKKOS)
    target_sources(c8_corsika7_compare PRIVATE detail/air_shower_kokkos/KokkosRunSession.cpp)
    target_link_libraries(c8_corsika7_compare PRIVATE CORSIKA8KokkosEm)
    target_compile_definitions(c8_corsika7_compare PRIVATE
      CORSIKA8_WITH_KOKKOS_EM=1 CORSIKA8_KOKKOS_VERSION="${Kokkos_VERSION}")
  endif ()
  set_target_properties(c8_corsika7_compare PROPERTIES
    INSTALL_RPATH "$ORIGIN/../${CMAKE_INSTALL_LIBDIR}/corsika")
  install(TARGETS c8_corsika7_compare DESTINATION bin)
endif ()

# A Conda Fortran compiler can contribute an old-sysroot libm linker script
# even when the final C++ driver and all C++ objects use the host glibc. Put
# the matching host multiarch directory first instead of applying the Conda
# sysroot to host objects.
set (_c8_host_multiarch_libdir
     "/usr/lib/${CMAKE_LIBRARY_ARCHITECTURE}")
if (NOT EXISTS "${_c8_host_multiarch_libdir}/libm.so")
  file (GLOB _c8_host_libm_candidates "/usr/lib/*-linux-gnu/libm.so")
  list (LENGTH _c8_host_libm_candidates _c8_host_libm_count)
  if (_c8_host_libm_count GREATER 0)
    list (GET _c8_host_libm_candidates 0 _c8_host_libm)
    get_filename_component (_c8_host_multiarch_libdir
                            "${_c8_host_libm}" DIRECTORY)
  endif ()
endif ()
if (TARGET c8_corsika7_compare AND
    EXISTS "${_c8_host_multiarch_libdir}/libm.so")
  # Conan transitive link directories can contain a foreign-sysroot libm
  # script even with host GCC. Prefer this host's libm for this isolated app.
  target_link_options(c8_corsika7_compare PRIVATE "-L${_c8_host_multiarch_libdir}")
endif ()
if ((CMAKE_Fortran_COMPILER_SYSROOT OR MATH_LIBRARY MATCHES "/sysroot/" OR
     CMAKE_CXX_COMPILER MATCHES "/(conda|mambaforge|miniconda)[^/]*/") AND
    EXISTS "${_c8_host_multiarch_libdir}/libm.so")
  target_link_options (
    c8_air_shower PRIVATE
    "-L${_c8_host_multiarch_libdir}"
    )
  if (TARGET c8_corsika7_compare)
    target_link_options(c8_corsika7_compare PRIVATE "-L${_c8_host_multiarch_libdir}")
  endif ()

endif ()



if (CORSIKA_ENABLE_KOKKOS)
  # The same live-PROPOSAL first-divergence oracle used by native CUDA is
  # compiled against the selected Kokkos execution space.  It remains an
  # opt-in validation target and is never installed.
  add_executable (
    kokkos_em_proposal_native_decision_oracle EXCLUDE_FROM_ALL
    ../validation/accelerator/compare_proposal_native_decisions.cpp
    ../validation/accelerator/KokkosProposalNativeOracleDevice.cpp
    )
  target_compile_definitions (
    kokkos_em_proposal_native_decision_oracle
    PRIVATE C8_PROPOSAL_NATIVE_ORACLE_KOKKOS=1
    )
  target_link_libraries (
    kokkos_em_proposal_native_decision_oracle
    PRIVATE
    CORSIKA8KokkosEm
    CORSIKA8ProposalNative
    PROPOSAL::PROPOSAL
    spdlog::spdlog
    )
  target_include_directories (
    kokkos_em_proposal_native_decision_oracle
    PRIVATE ../validation/accelerator
    )
  add_dependencies (
    kokkos_em_proposal_native_decision_oracle
    GenParticlesHeaders
    GenMediaProperties
    )
  if (CORSIKA_KOKKOS_BACKEND MATCHES "^CUDA(_OPENMP)?$")
    set_source_files_properties (
      ../validation/accelerator/KokkosProposalNativeOracleDevice.cpp
      PROPERTIES LANGUAGE CUDA
      )
    target_compile_options (
      kokkos_em_proposal_native_decision_oracle PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
      )
    set_target_properties (
      kokkos_em_proposal_native_decision_oracle PROPERTIES
      CUDA_ARCHITECTURES "${CORSIKA_KOKKOS_CUDA_ARCHITECTURES}"
      )
  elseif (CORSIKA_KOKKOS_BACKEND STREQUAL "HIP")
    set_source_files_properties (
      ../validation/accelerator/KokkosProposalNativeOracleDevice.cpp
      PROPERTIES LANGUAGE HIP
      )
  endif ()
endif ()

if(WITH_FLUKA)
  message("compiling c8_air_shower.cpp with FLUKA")
  target_compile_definitions(c8_air_shower PRIVATE WITH_FLUKA)
  add_executable (fluka_batch_worker fluka_batch_worker.cpp)
  target_link_libraries (fluka_batch_worker PRIVATE CORSIKA8 fluka)
  # The worker is exec'ed from the installed bin directory and links FLUKA
  # directly.  Keep its runtime lookup consistent with c8_air_shower, since
  # CORSIKA's model libraries are installed below lib/corsika rather than lib.
  set_target_properties(fluka_batch_worker PROPERTIES
    INSTALL_RPATH "$ORIGIN/../${CMAKE_INSTALL_LIBDIR}/corsika"
  )
else()
  message("compiling c8_air_shower.cpp with UrQMD")
endif()

install (
  TARGETS c8_air_shower DESTINATION bin
)
if (TARGET c8_kokkos_tune)
  install (TARGETS c8_kokkos_tune DESTINATION bin)
endif ()
install (
  DIRECTORY "${CMAKE_SOURCE_DIR}/configs/media"
  DESTINATION "${CMAKE_INSTALL_DATADIR}/corsika"
)
install (
  FILES "${CMAKE_SOURCE_DIR}/resources/GeoMag/IGRF14.COF"
  DESTINATION "${CMAKE_INSTALL_DATADIR}/corsika/GeoMag"
)


if (TARGET fluka_batch_worker)
  install (TARGETS fluka_batch_worker DESTINATION bin)
endif ()

# Native coordination in the existing CUDA air executable. No GPU context is
# created by the coordinator; every worker execs the same binary independently.
set(_c8_native_multigpu_default OFF)
if(CORSIKA_ENABLE_KOKKOS AND CORSIKA_KOKKOS_BACKEND MATCHES "^CUDA(_OPENMP)?$")
  set(_c8_native_multigpu_default ON)
endif()
option(CORSIKA_ENABLE_NATIVE_MULTIGPU "Enable native --devices in c8_air_shower"
  ${_c8_native_multigpu_default})
if(CORSIKA_ENABLE_NATIVE_MULTIGPU)
  if(NOT CORSIKA_ENABLE_KOKKOS OR NOT CORSIKA_KOKKOS_BACKEND MATCHES "^CUDA(_OPENMP)?$")
    message(FATAL_ERROR "Native multi-GPU requires a CUDA or CUDA_OPENMP build")
  endif()
  add_subdirectory(detail/air_shower_multigpu)
  target_link_libraries(c8_air_shower PRIVATE c8_multigpu_coordinator)
  target_compile_definitions(c8_air_shower PRIVATE C8_NATIVE_MULTIGPU=1)
endif()

# Keep the old opt-in worker target for existing Python campaigns.
option(CORSIKA_BUILD_MULTIGPU_APPLICATION "Build experimental static multi-GPU workers" OFF)
if (CORSIKA_BUILD_MULTIGPU_APPLICATION)
  if (NOT CORSIKA_ENABLE_KOKKOS)
    message(FATAL_ERROR "Multi-GPU workers require CORSIKA_ENABLE_KOKKOS=ON")
  endif ()
  add_executable(c8_air_shower_multigpu c8_air_shower_multigpu.cpp
    detail/air_shower_kokkos/KokkosRunSession.cpp)
  target_link_libraries(c8_air_shower_multigpu PRIVATE CORSIKA8 CORSIKA8KokkosEm)
  target_compile_definitions(c8_air_shower_multigpu PRIVATE
    CORSIKA8_WITH_KOKKOS_EM=1 CORSIKA8_KOKKOS_VERSION="${Kokkos_VERSION}"
    CORSIKA8_BUNDLED_IGRF14_FILE="${CMAKE_SOURCE_DIR}/resources/GeoMag/IGRF14.COF")
  if (WITH_FLUKA)
    target_compile_definitions(c8_air_shower_multigpu PRIVATE WITH_FLUKA)
  endif ()
  if (EXISTS "${_c8_host_multiarch_libdir}/libm.so")
    target_link_options(c8_air_shower_multigpu PRIVATE "-L${_c8_host_multiarch_libdir}")
  endif ()
  set_target_properties(c8_air_shower_multigpu PROPERTIES
    INSTALL_RPATH "$ORIGIN/../${CMAKE_INSTALL_LIBDIR}/corsika")
  # Not installed or recommended until lifecycle and statistical gates pass.
  add_executable(test_multigpu_frontier ../validation/accelerator/test_multigpu_frontier.cpp)
  target_link_libraries(test_multigpu_frontier PRIVATE CORSIKA8)
  if (EXISTS "${_c8_host_multiarch_libdir}/libm.so")
    target_link_options(test_multigpu_frontier PRIVATE "-L${_c8_host_multiarch_libdir}")
  endif ()
endif ()

# Optional, independent mountain applications. Never alter the atmospheric
# target, its model selection, or its command-line defaults.
option(CORSIKA_BUILD_MOUNTAIN_APPLICATION "Build independent mountain examples and terrain diagnostics" OFF)
if (CORSIKA_BUILD_MOUNTAIN_APPLICATION)
  add_executable(c8_terrain_scalar_magnetic_probe EXCLUDE_FROM_ALL
    ${CMAKE_SOURCE_DIR}/validation/terrain/probe_scalar_magnetic_units.cpp)
  target_link_libraries(c8_terrain_scalar_magnetic_probe PRIVATE CORSIKA8)
  add_executable(c8_terrain_scalar_lepton_probe EXCLUDE_FROM_ALL
    ${CMAKE_SOURCE_DIR}/validation/terrain/probe_scalar_first_lepton.cpp
    ${CMAKE_SOURCE_DIR}/src/accelerator/em/tables/Sha256.cpp)
  target_include_directories(c8_terrain_scalar_lepton_probe PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  target_link_libraries(c8_terrain_scalar_lepton_probe PRIVATE CORSIKA8)
  if(EXISTS "${_c8_host_multiarch_libdir}/libm.so")
    target_link_options(c8_terrain_scalar_magnetic_probe PRIVATE "-L${_c8_host_multiarch_libdir}")
    target_link_options(c8_terrain_scalar_lepton_probe PRIVATE "-L${_c8_host_multiarch_libdir}")
  endif()
  add_executable(c8_terrain_environment c8_terrain_environment.cpp
    ${CMAKE_SOURCE_DIR}/src/accelerator/em/tables/Sha256.cpp)
  target_link_libraries(c8_terrain_environment PRIVATE CORSIKA8)
  install(TARGETS c8_terrain_environment DESTINATION bin)
  if (EXISTS "${_c8_host_multiarch_libdir}/libm.so")
    target_link_options(c8_terrain_environment PRIVATE "-L${_c8_host_multiarch_libdir}")
  endif ()
  if (CORSIKA_ENABLE_KOKKOS)
    add_library(CORSIKA8TerrainQueries STATIC ${CMAKE_SOURCE_DIR}/src/terrain/TerrainDeviceQuery.cpp)
    target_include_directories(CORSIKA8TerrainQueries PUBLIC ${CMAKE_SOURCE_DIR})
    target_compile_features(CORSIKA8TerrainQueries PUBLIC cxx_std_17)
    target_link_libraries(CORSIKA8TerrainQueries PUBLIC Kokkos::kokkos)
    add_executable(c8_terrain_device_probe c8_terrain_device_probe.cpp)
    target_link_libraries(c8_terrain_device_probe PRIVATE CORSIKA8 CORSIKA8TerrainQueries)
    add_executable(c8_terrain_curved_probe c8_terrain_curved_probe.cpp)
    target_link_libraries(c8_terrain_curved_probe PRIVATE CORSIKA8 CORSIKA8TerrainQueries)
    if (CORSIKA_KOKKOS_BACKEND MATCHES "CUDA")
      set_source_files_properties(${CMAKE_SOURCE_DIR}/src/terrain/TerrainDeviceQuery.cpp PROPERTIES LANGUAGE CUDA)
      target_link_libraries(CORSIKA8TerrainQueries PUBLIC CUDA::cudart CUDA::cuda_driver)
      set_target_properties(CORSIKA8TerrainQueries PROPERTIES
        CUDA_ARCHITECTURES "${CORSIKA_KOKKOS_CUDA_ARCHITECTURES}" CUDA_STANDARD 17)
      if (CORSIKA_KOKKOS_BACKEND STREQUAL "CUDA_OPENMP")
        target_compile_options(CORSIKA8TerrainQueries PRIVATE
          $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fopenmp>)
      endif ()
    elseif (CORSIKA_KOKKOS_BACKEND STREQUAL "HIP")
      set_source_files_properties(${CMAKE_SOURCE_DIR}/src/terrain/TerrainDeviceQuery.cpp PROPERTIES LANGUAGE HIP)
    endif ()
    if (EXISTS "${_c8_host_multiarch_libdir}/libm.so")
      target_link_options(c8_terrain_device_probe PRIVATE "-L${_c8_host_multiarch_libdir}")
      target_link_options(c8_terrain_curved_probe PRIVATE "-L${_c8_host_multiarch_libdir}")
    endif ()
  endif ()
endif ()
if (CORSIKA_BUILD_MOUNTAIN_APPLICATION AND WITH_FLUKA)
  # Independent adapter for more than ten elemental targets. Do not change
  # the air wrapper or redistribute the user's licensed FLUKA headers.
  find_path(C8_TERRAIN_FLUKA_INCLUDE_DIR "(FLKMAT)"
    HINTS "$ENV{FLUPRO}/flukapro" REQUIRED)
  # Parenthesized FLUKA filenames are parsed as archive members by GNU make.
  # Normalize only the private build copies, preserving installed headers.
  set(_terrain_fluka_includes "${CMAKE_CURRENT_BINARY_DIR}/fluka-material-includes")
  file(MAKE_DIRECTORY "${_terrain_fluka_includes}")
  foreach(_terrain_fluka_common DIMPAR FLKMAT FLKCMP)
    configure_file("${C8_TERRAIN_FLUKA_INCLUDE_DIR}/(${_terrain_fluka_common})"
      "${_terrain_fluka_includes}/fluka_${_terrain_fluka_common}.inc" COPYONLY)
  endforeach()
  add_library(C8TerrainFlukaMaterialBridge OBJECT detail/mountain/FlukaMaterialBridge.f)
  target_include_directories(C8TerrainFlukaMaterialBridge PRIVATE
    "${_terrain_fluka_includes}")
  add_executable(c8_terrain_cascade c8_terrain_cascade.cpp
    ${CMAKE_SOURCE_DIR}/src/accelerator/em/tables/Sha256.cpp
    $<TARGET_OBJECTS:C8TerrainFlukaMaterialBridge>)
  target_link_libraries(c8_terrain_cascade PRIVATE CORSIKA8 fluka)
  target_compile_definitions(c8_terrain_cascade PRIVATE
    CORSIKA8_TERRAIN_IGRF14_FILE="${CMAKE_SOURCE_DIR}/resources/GeoMag/IGRF14.COF")
  if (CORSIKA_ENABLE_KOKKOS)
    target_link_libraries(c8_terrain_cascade PRIVATE CORSIKA8InterfaceEm)
    if (CORSIKA_KOKKOS_BACKEND MATCHES "CUDA" AND UNIX AND NOT APPLE)
      target_link_options(c8_terrain_cascade PRIVATE "LINKER:--no-as-needed")
    endif ()
    target_compile_definitions(c8_terrain_cascade PRIVATE CORSIKA8_WITH_KOKKOS_EM=1)

    # Opt-in, uninstalled shadow-replay builds. The normal application never
    # allocates audit mirrors or records; no floating-point flags are changed
    # on a production target. The no-contract variant is a diagnosis, not a fix.
    foreach(_terrain_audit_variant audit audit_nocontract)
      set(_terrain_audit_lib CORSIKA8Terrain_${_terrain_audit_variant})
      set(_terrain_audit_app c8_terrain_cascade_${_terrain_audit_variant})
      add_library(${_terrain_audit_lib} STATIC EXCLUDE_FROM_ALL
        ${CMAKE_SOURCE_DIR}/src/transport/InterfaceEmSession.cpp)
      target_link_libraries(${_terrain_audit_lib} PUBLIC CORSIKA8ProposalNative Kokkos::kokkos)
      target_compile_definitions(${_terrain_audit_lib} PRIVATE C8_TERRAIN_STEP_AUDIT=1
        PUBLIC CORSIKA8_WITH_KOKKOS=1 C8_INTERFACE_EXECUTION_${C8_INTERFACE_EXECUTION_SPACE}=1)
      target_include_directories(${_terrain_audit_lib} PRIVATE ${CMAKE_SOURCE_DIR}/validation/terrain)
      target_compile_features(${_terrain_audit_lib} PUBLIC cxx_std_17)
      if(CORSIKA_KOKKOS_BACKEND MATCHES "CUDA")
        set_source_files_properties(${CMAKE_SOURCE_DIR}/src/transport/InterfaceEmSession.cpp PROPERTIES LANGUAGE CUDA)
        target_link_libraries(${_terrain_audit_lib} PUBLIC CUDA::cudart CUDA::cuda_driver)
        target_compile_options(${_terrain_audit_lib} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
        set_target_properties(${_terrain_audit_lib} PROPERTIES
          CUDA_ARCHITECTURES "${CORSIKA_KOKKOS_CUDA_ARCHITECTURES}" CUDA_STANDARD 17)
        if(CORSIKA_KOKKOS_BACKEND STREQUAL "CUDA_OPENMP")
          target_compile_options(${_terrain_audit_lib} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fopenmp>)
        endif()
      endif()
      if(_terrain_audit_variant STREQUAL "audit_nocontract")
        target_compile_options(${_terrain_audit_lib} PRIVATE
          $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>
          $<$<COMPILE_LANGUAGE:CXX>:-ffp-contract=off>)
      endif()
      add_executable(${_terrain_audit_app} EXCLUDE_FROM_ALL c8_terrain_cascade.cpp
        ${CMAKE_SOURCE_DIR}/src/accelerator/em/tables/Sha256.cpp
        $<TARGET_OBJECTS:C8TerrainFlukaMaterialBridge>)
      target_link_libraries(${_terrain_audit_app} PRIVATE CORSIKA8 fluka ${_terrain_audit_lib})
      target_compile_definitions(${_terrain_audit_app} PRIVATE CORSIKA8_WITH_KOKKOS_EM=1
        CORSIKA8_TERRAIN_IGRF14_FILE="${CMAKE_SOURCE_DIR}/resources/GeoMag/IGRF14.COF")
      if(CORSIKA_KOKKOS_BACKEND MATCHES "CUDA" AND UNIX AND NOT APPLE)
        target_link_options(${_terrain_audit_app} PRIVATE "LINKER:--no-as-needed")
      endif()
      if(EXISTS "${_c8_host_multiarch_libdir}/libm.so")
        target_link_options(${_terrain_audit_app} PRIVATE "-L${_c8_host_multiarch_libdir}")
      endif()
    endforeach()
  endif ()
  set_target_properties(c8_terrain_cascade PROPERTIES
    INSTALL_RPATH "$ORIGIN/../${CMAKE_INSTALL_LIBDIR}/corsika")
  install(TARGETS c8_terrain_cascade DESTINATION bin)
  if (EXISTS "${_c8_host_multiarch_libdir}/libm.so")
    target_link_options(c8_terrain_cascade PRIVATE "-L${_c8_host_multiarch_libdir}")
  endif ()
  add_executable(c8_mountain_neutrino c8_mountain_neutrino.cpp)
  target_link_libraries(c8_mountain_neutrino PRIVATE CORSIKA8 fluka)
  set_target_properties(c8_mountain_neutrino PROPERTIES
    INSTALL_RPATH "$ORIGIN/../${CMAKE_INSTALL_LIBDIR}/corsika")
  if (EXISTS "${_c8_host_multiarch_libdir}/libm.so")
    target_link_options(c8_mountain_neutrino PRIVATE "-L${_c8_host_multiarch_libdir}")
  endif ()
  if (CORSIKA_ENABLE_KOKKOS)
    target_link_libraries(c8_mountain_neutrino PRIVATE CORSIKA8KokkosEm)
    target_compile_definitions(c8_mountain_neutrino PRIVATE CORSIKA8_WITH_KOKKOS_EM=1)
  endif ()
  install(TARGETS c8_mountain_neutrino DESTINATION bin)
  install(DIRECTORY "${CMAKE_SOURCE_DIR}/configs/mountain"
    DESTINATION "${CMAKE_INSTALL_DATADIR}/corsika")
elseif (CORSIKA_BUILD_MOUNTAIN_APPLICATION)
  message(STATUS "Skipping mountain application: it requires WITH_FLUKA=ON")
endif ()

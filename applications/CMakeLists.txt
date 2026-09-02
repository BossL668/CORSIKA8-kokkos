SET(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")

add_executable (c8_air_shower c8_air_shower.cpp)
target_link_libraries (c8_air_shower PRIVATE CORSIKA8)

# The homogeneous-ice application has a complete scalar PROPOSAL path and must
# remain buildable on CPU-only validation hosts.  CUDA support is attached
# below when available; it is not a prerequisite for the executable itself.
add_executable (c8_ice_cascade c8_ice_cascade.cpp)
target_link_libraries (c8_ice_cascade PRIVATE CORSIKA8)

if (CORSIKA_ENABLE_KOKKOS)
  add_executable (kokkos_backend_probe kokkos_backend_probe.cpp)
  target_link_libraries (
    kokkos_backend_probe PRIVATE CORSIKA8KokkosEm CLI11::CLI11)
endif ()

set_target_properties(c8_air_shower PROPERTIES
  INSTALL_RPATH "${CMAKE_INSTALL_FULL_LIBDIR}/corsika"
)

set_target_properties(c8_ice_cascade PROPERTIES
  INSTALL_RPATH "${CMAKE_INSTALL_FULL_LIBDIR}/corsika"
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
if ((CMAKE_Fortran_COMPILER_SYSROOT OR MATH_LIBRARY MATCHES "/sysroot/" OR
     CMAKE_CXX_COMPILER MATCHES "/(conda|mambaforge|miniconda)[^/]*/") AND
    EXISTS "${_c8_host_multiarch_libdir}/libm.so")
  target_link_options (
    c8_air_shower PRIVATE
    "-L${_c8_host_multiarch_libdir}"
    )
  target_link_options (
    c8_ice_cascade PRIVATE
    "-L${_c8_host_multiarch_libdir}"
    )
endif ()

if (CORSIKA_ENABLE_CUDA)
  target_sources (
    c8_air_shower PRIVATE
    detail/air_shower_cuda/CudaRunSession.cpp)
  target_link_libraries (c8_air_shower PRIVATE CORSIKA8GpuEm)
  target_compile_definitions (c8_air_shower PRIVATE CORSIKA8_WITH_CUDA_EM=1)

  # Attach the optional CUDA transport/radio implementation to the otherwise
  # CPU-capable ice executable.  Keep the ice environment and CLI in this
  # separate translation unit so c8_air_shower remains an air-only program.
  target_link_libraries (c8_ice_cascade PRIVATE CORSIKA8GpuEm)
  target_compile_definitions (
    c8_ice_cascade PRIVATE CORSIKA8_WITH_CUDA_EM=1)

  # Explicit, opt-in validation binary.  It is excluded from the default
  # build so a long campaign can continue using its already-linked
  # c8_air_shower while output/accounting changes are compiled and tested in
  # isolation.
  add_executable (
    c8_air_shower_parent_profile_validation EXCLUDE_FROM_ALL
    c8_air_shower.cpp
    detail/air_shower_cuda/CudaRunSession.cpp
    )
  target_link_libraries (
    c8_air_shower_parent_profile_validation
    PRIVATE CORSIKA8 CORSIKA8GpuEm
    )
  target_compile_definitions (
    c8_air_shower_parent_profile_validation
    PRIVATE
    CORSIKA8_WITH_CUDA_EM=1
    CORSIKA8_BUNDLED_IGRF14_FILE="${CMAKE_SOURCE_DIR}/resources/GeoMag/IGRF14.COF"
    )
  if ((CMAKE_Fortran_COMPILER_SYSROOT OR MATH_LIBRARY MATCHES "/sysroot/") AND
      EXISTS "${_c8_host_multiarch_libdir}/libm.so")
    target_link_options (
      c8_air_shower_parent_profile_validation PRIVATE
      "-L${_c8_host_multiarch_libdir}"
      )
  endif ()

  add_executable (cuda_decision_replay cuda_decision_replay.cpp)
  target_link_libraries (
    cuda_decision_replay
    PRIVATE
    CORSIKA8
    CORSIKA8GpuEm
    )

  # Validation-only probe.  This executable compares the original scalar
  # PROPOSAL interaction contract with the CUDA table contract while feeding
  # both sides the same process-selection uniforms.  It is deliberately
  # excluded from the default build and is never installed with production
  # applications.
  add_executable (
    gpu_em_same_random_process_oracle EXCLUDE_FROM_ALL
    ../validation/gpu_em/compare_same_random_processes.cpp
    )
  target_link_libraries (
    gpu_em_same_random_process_oracle
    PRIVATE
    CORSIKA8GpuEm
    PROPOSAL::PROPOSAL
    spdlog::spdlog
    yaml-cpp::yaml-cpp
    )
  add_dependencies (
    gpu_em_same_random_process_oracle
    GenParticlesHeaders
    GenMediaProperties
    )

  add_executable (
    gpu_em_proposal_native_decision_oracle EXCLUDE_FROM_ALL
    ../validation/gpu_em/compare_proposal_native_decisions.cpp
    )
  target_link_libraries (
    gpu_em_proposal_native_decision_oracle
    PRIVATE
    CORSIKA8GpuEm
    PROPOSAL::PROPOSAL
    spdlog::spdlog
    )
  add_dependencies (
    gpu_em_proposal_native_decision_oracle
    GenParticlesHeaders
    GenMediaProperties
    )
endif ()

add_executable (gpu_em_tablegen gpu_em_tablegen.cpp)
target_link_libraries (
  gpu_em_tablegen
  PRIVATE
  CORSIKA8GpuEmTables
  CLI11::CLI11
  PROPOSAL::PROPOSAL
  spdlog::spdlog
  )
add_dependencies (gpu_em_tablegen GenParticlesHeaders GenMediaProperties)

# Numerical diagnostic used to compare the production flat table with both
# cached/interpolated and direct PROPOSAL electron loss queries.
add_executable (
  gpu_em_curve_dump
  ../validation/gpu_em/dump_proposal_table_curves.cpp
  )
target_link_libraries (
  gpu_em_curve_dump
  PRIVATE
  CORSIKA8GpuEmTables
  PROPOSAL::PROPOSAL
  spdlog::spdlog
  )
add_dependencies (gpu_em_curve_dump GenParticlesHeaders GenMediaProperties)

add_executable (gpu_em_table_prepare gpu_em_table_prepare.cpp)
target_link_libraries (
  gpu_em_table_prepare
  PRIVATE
  CORSIKA8GpuEmTables
  CLI11::CLI11
  PROPOSAL::PROPOSAL
  yaml-cpp::yaml-cpp
  )

if(WITH_FLUKA)
  message("compiling c8_air_shower.cpp with FLUKA")
  target_compile_definitions(c8_air_shower PRIVATE WITH_FLUKA)
  if (TARGET c8_ice_cascade)
    target_compile_definitions(c8_ice_cascade PRIVATE WITH_FLUKA)
  endif ()
  if (TARGET c8_air_shower_parent_profile_validation)
    target_compile_definitions(
      c8_air_shower_parent_profile_validation PRIVATE WITH_FLUKA)
  endif ()
  add_executable (fluka_batch_worker fluka_batch_worker.cpp)
  target_link_libraries (fluka_batch_worker PRIVATE CORSIKA8 fluka)
  # The worker is exec'ed from the installed bin directory and links FLUKA
  # directly.  Keep its runtime lookup consistent with c8_air_shower, since
  # CORSIKA's model libraries are installed below lib/corsika rather than lib.
  set_target_properties(fluka_batch_worker PROPERTIES
    INSTALL_RPATH "${CMAKE_INSTALL_FULL_LIBDIR}/corsika"
  )
else()
  message("compiling c8_air_shower.cpp with UrQMD")
endif()

install (
  TARGETS c8_air_shower gpu_em_tablegen gpu_em_table_prepare DESTINATION bin
)
if (TARGET c8_ice_cascade)
  install (TARGETS c8_ice_cascade DESTINATION bin)
endif ()
install (
  DIRECTORY "${CMAKE_SOURCE_DIR}/configs/media"
  DESTINATION "${CMAKE_INSTALL_DATADIR}/corsika"
)
install (
  FILES "${CMAKE_SOURCE_DIR}/resources/GeoMag/IGRF14.COF"
  DESTINATION "${CMAKE_INSTALL_DATADIR}/corsika/GeoMag"
)

if (TARGET cuda_decision_replay)
  install (TARGETS cuda_decision_replay DESTINATION bin)
endif ()

if (TARGET fluka_batch_worker)
  install (TARGETS fluka_batch_worker DESTINATION bin)
endif ()

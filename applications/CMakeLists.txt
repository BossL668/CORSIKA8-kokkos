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

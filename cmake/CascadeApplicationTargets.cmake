#
# (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
#
# This software is distributed under the terms of the 3-clause BSD license.
# See file LICENSE for a full version of the license.
#

# Keep application implementation in a library, including the scalar-only build.
# NATIVE_TARGET is CORSIKA8 in-tree and CORSIKA8::CORSIKA8 for installation checks.
function (corsika_add_cascade_application SOURCE_ROOT NATIVE_TARGET)
  add_library (CORSIKA8CascadeApplication STATIC
    ${SOURCE_ROOT}/src/kokkos/application/BackendCli.cpp
    ${SOURCE_ROOT}/src/kokkos/application/cascade/CascadeSupport.cpp
    ${SOURCE_ROOT}/src/kokkos/application/cascade/CascadeExtension.cpp
    ${SOURCE_ROOT}/src/output/ReplayTrace.cpp)
  add_library (CORSIKA8::CascadeApplication ALIAS CORSIKA8CascadeApplication)
  set_target_properties (CORSIKA8CascadeApplication PROPERTIES EXPORT_NAME CascadeApplication)
  target_compile_features (CORSIKA8CascadeApplication PUBLIC cxx_std_17)
  target_include_directories (CORSIKA8CascadeApplication BEFORE PUBLIC
    $<BUILD_INTERFACE:${SOURCE_ROOT}> $<INSTALL_INTERFACE:include>)
  target_link_libraries (CORSIKA8CascadeApplication PUBLIC ${NATIVE_TARGET})

  if (TARGET CORSIKA8::ApplicationSupport)
    target_link_libraries (CORSIKA8CascadeApplication PUBLIC CORSIKA8::ApplicationSupport)
  else ()
    # These host-only CLI utilities do not require Kokkos headers or libraries.
    target_sources (CORSIKA8CascadeApplication PRIVATE
      ${SOURCE_ROOT}/src/kokkos/application/DeviceSelection.cpp
      ${SOURCE_ROOT}/src/kokkos/application/DeviceCliOptions.cpp)
  endif ()

  if (TARGET CORSIKA8::EmRunSession)
    find_package (CubicInterpolation CONFIG REQUIRED)
    target_sources (CORSIKA8CascadeApplication PRIVATE
      ${SOURCE_ROOT}/src/kokkos/application/cascade/CascadeRunSession.cpp)
    target_compile_definitions (CORSIKA8CascadeApplication PUBLIC
      CORSIKA8_WITH_KOKKOS_EM=1 CORSIKA_KOKKOS_COORDINATOR=1
      CORSIKA_KOKKOS_STATIC_MULTIGPU=1)
    target_link_libraries (CORSIKA8CascadeApplication PUBLIC
      CORSIKA8::EmRunSession CORSIKA8::AirShowerCoordinator
      CORSIKA8::HostRadio CORSIKA8::ProfileStorage CubicInterpolation::CubicInterpolation)
  endif ()
  if (TARGET CORSIKA8::FirnRadio)
    target_link_libraries (CORSIKA8CascadeApplication PUBLIC CORSIKA8::FirnRadio)
  else ()
    target_sources (CORSIKA8CascadeApplication PRIVATE
      ${SOURCE_ROOT}/src/kokkos/radio/firn/FullWaveAccumulatorUnavailable.cpp)
  endif ()
  if (WITH_FLUKA)
    target_compile_definitions (CORSIKA8CascadeApplication PUBLIC WITH_FLUKA)
  endif ()

  # OFF keeps the normal installed binary without publishing Kokkos interfaces.
  if (CORSIKA_ENABLE_KOKKOS)
    install (TARGETS CORSIKA8CascadeApplication EXPORT CORSIKA8PublicTargets
      ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
  endif ()
endfunction ()

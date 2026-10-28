#
# (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
#
# This software is distributed under the terms of the 3-clause BSD license.
# See file LICENSE for a full version of the license.
#

# Included after the existing numerical targets and BackendRuntimeTargets.
# This file selects compiled production implementations; it supplies no fallback.
find_package (OpenSSL REQUIRED)
find_package (nlohmann_json CONFIG REQUIRED)
set (CORSIKA_BACKEND_APPLICATION_TARGETS)

add_library (CORSIKA8ApplicationSupport
  application/DeviceSelection.cpp application/DeviceCliOptions.cpp)
add_library (CORSIKA8::ApplicationSupport ALIAS CORSIKA8ApplicationSupport)
set_target_properties (CORSIKA8ApplicationSupport PROPERTIES EXPORT_NAME ApplicationSupport)
target_compile_features (CORSIKA8ApplicationSupport PUBLIC cxx_std_17)
target_link_libraries (CORSIKA8ApplicationSupport PUBLIC CORSIKA8 CLI11::CLI11)
if (NOT DEFINED Kokkos_VERSION OR Kokkos_VERSION STREQUAL "")
  message (FATAL_ERROR "Application metadata requires the actual Kokkos package version")
endif ()
target_compile_definitions (CORSIKA8ApplicationSupport PUBLIC
  CORSIKA_KOKKOS_VERSION="${Kokkos_VERSION}")
if (NOT CORSIKA_KOKKOS_RESOLVED_BACKEND STREQUAL "NONE")
  target_compile_definitions (CORSIKA8ApplicationSupport PUBLIC
    CORSIKA_KOKKOS_BACKEND_${CORSIKA_KOKKOS_RESOLVED_BACKEND}=1
    CORSIKA8_KOKKOS_BACKEND_${CORSIKA_KOKKOS_RESOLVED_BACKEND}=1)
endif ()
list (APPEND CORSIKA_BACKEND_APPLICATION_TARGETS CORSIKA8ApplicationSupport)

add_library (CORSIKA8AirShowerCoordinator
  scheduling/AirShowerCoordinator.cpp scheduling/StaticFrontierIO.cpp)
add_library (CORSIKA8::AirShowerCoordinator ALIAS CORSIKA8AirShowerCoordinator)
set_target_properties (CORSIKA8AirShowerCoordinator PROPERTIES EXPORT_NAME AirShowerCoordinator)
target_compile_features (CORSIKA8AirShowerCoordinator PUBLIC cxx_std_17)
target_link_libraries (CORSIKA8AirShowerCoordinator
  PUBLIC CORSIKA8ApplicationSupport CORSIKA8 nlohmann_json::nlohmann_json
  PRIVATE OpenSSL::Crypto)
# The production conditional is retained; the EGS4 application integration is separate.
if (CORSIKA_ENABLE_EGS4)
  target_compile_definitions (CORSIKA8AirShowerCoordinator PRIVATE CORSIKA8_WITH_NATIVE_EGS4)
endif ()
list (APPEND CORSIKA_BACKEND_APPLICATION_TARGETS CORSIKA8AirShowerCoordinator)

add_library (CORSIKA8NativeRequirements em/proposal/NativeRequirements.cpp)
add_library (CORSIKA8::NativeRequirements ALIAS CORSIKA8NativeRequirements)
set_target_properties (CORSIKA8NativeRequirements PROPERTIES EXPORT_NAME NativeRequirements)
target_compile_features (CORSIKA8NativeRequirements PUBLIC cxx_std_17)
target_link_libraries (CORSIKA8NativeRequirements PUBLIC CORSIKA8NativeTableQueries CORSIKA8)
list (APPEND CORSIKA_BACKEND_APPLICATION_TARGETS CORSIKA8NativeRequirements)

if (TARGET CORSIKA8BackendRuntime)
  add_library (CORSIKA8EmBackend em/EmBackend.cpp em/BackendInstance.cpp)
  add_library (CORSIKA8::EmBackend ALIAS CORSIKA8EmBackend)
  set_target_properties (CORSIKA8EmBackend PROPERTIES EXPORT_NAME EmBackend)
  target_compile_features (CORSIKA8EmBackend PUBLIC cxx_std_17)
  target_link_libraries (CORSIKA8EmBackend PUBLIC
    CORSIKA8BackendRuntime CORSIKA8TuningCache CORSIKA8NativeTableQueries
    CORSIKA8MoliereInterpolation CORSIKA8)
  if (CORSIKA_KOKKOS_RESOLVED_BACKEND MATCHES "OPENMP")
    target_sources (CORSIKA8EmBackend PRIVATE em/OpenMPBackendInstance.cpp)
  endif ()
  if (CORSIKA_KOKKOS_RESOLVED_BACKEND MATCHES "^CUDA")
    target_sources (CORSIKA8EmBackend PRIVATE em/CudaBackendInstance.cpp)
    set_source_files_properties (em/CudaBackendInstance.cpp PROPERTIES LANGUAGE CUDA)
    set_target_properties (CORSIKA8EmBackend PROPERTIES
      CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON CUDA_EXTENSIONS OFF)
    target_compile_options (CORSIKA8EmBackend PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
    if (CORSIKA_KOKKOS_RESOLVED_BACKEND STREQUAL "CUDA_OPENMP")
      target_sources (CORSIKA8EmBackend PRIVATE em/CooperativeBackend.cpp)
    endif ()
  elseif (CORSIKA_KOKKOS_RESOLVED_BACKEND STREQUAL "HIP")
    target_sources (CORSIKA8EmBackend PRIVATE em/HipBackendInstance.cpp)
    set_source_files_properties (em/HipBackendInstance.cpp PROPERTIES LANGUAGE HIP)
    set_target_properties (CORSIKA8EmBackend PROPERTIES
      HIP_STANDARD 17 HIP_STANDARD_REQUIRED ON)
  elseif (CORSIKA_KOKKOS_RESOLVED_BACKEND STREQUAL "SYCL")
    target_sources (CORSIKA8EmBackend PRIVATE em/SyclBackendInstance.cpp)
  endif ()
  list (APPEND CORSIKA_BACKEND_APPLICATION_TARGETS CORSIKA8EmBackend)

  if (CORSIKA_ENABLE_PROPOSAL_EXPORT)
    add_library (CORSIKA8EmRunSession em/EmRunSession.cpp)
    add_library (CORSIKA8::EmRunSession ALIAS CORSIKA8EmRunSession)
    set_target_properties (CORSIKA8EmRunSession PROPERTIES EXPORT_NAME EmRunSession)
    target_compile_features (CORSIKA8EmRunSession PUBLIC cxx_std_17)
    target_link_libraries (CORSIKA8EmRunSession PUBLIC
      CORSIKA8EmBackend CORSIKA8NativeRequirements CORSIKA8ProposalBridge)
    list (APPEND CORSIKA_BACKEND_APPLICATION_TARGETS CORSIKA8EmRunSession)
  endif ()
endif ()

if (CORSIKA_SANITIZERS_ENABLED)
  foreach (TARGET IN LISTS CORSIKA_BACKEND_APPLICATION_TARGETS)
    target_compile_options (${TARGET} PRIVATE
      -O1 -g -fno-omit-frame-pointer -fsanitize=address,undefined -fno-sanitize-recover=all)
    target_link_libraries (${TARGET} PUBLIC -fsanitize=address,undefined)
  endforeach ()
endif ()
install (TARGETS ${CORSIKA_BACKEND_APPLICATION_TARGETS}
  EXPORT CORSIKA8PublicTargets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

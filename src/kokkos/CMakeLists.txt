#
# (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
#
# See file AUTHORS for a list of contributors.
#
# This software is distributed under the terms of the 3-clause BSD license.
# See file LICENSE for a full version of the license.
#

find_package (Kokkos 4.7 CONFIG REQUIRED)
if (NOT Kokkos_VERSION VERSION_LESS 5)
  message (FATAL_ERROR "This C++17 runtime integration requires Kokkos 4.x (>= 4.7).")
endif ()
find_package (Threads REQUIRED)
get_filename_component (CORSIKA_KOKKOS_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)

# Plain transfer records do not include or link Kokkos or any physics library.
add_library (CORSIKA8KokkosData INTERFACE)
add_library (CORSIKA8::KokkosData ALIAS CORSIKA8KokkosData)
set_target_properties (CORSIKA8KokkosData PROPERTIES EXPORT_NAME KokkosData)
target_compile_features (CORSIKA8KokkosData INTERFACE cxx_std_17)
target_include_directories (
  CORSIKA8KokkosData
  INTERFACE
  $<BUILD_INTERFACE:${CORSIKA_KOKKOS_ROOT}>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  )

# Host-side C8 value conversion stays separate from dependency-free records.
add_library (CORSIKA8KokkosParticleConversion INTERFACE)
add_library (CORSIKA8::KokkosParticleConversion ALIAS CORSIKA8KokkosParticleConversion)
set_target_properties (CORSIKA8KokkosParticleConversion PROPERTIES
  EXPORT_NAME KokkosParticleConversion
)
target_link_libraries (CORSIKA8KokkosParticleConversion
  INTERFACE CORSIKA8KokkosData CORSIKA8
)

# Host topology export and borrowed C8 identity mapping do not require a runtime.
add_library (CORSIKA8KokkosEnvironment INTERFACE)
add_library (CORSIKA8::KokkosEnvironment ALIAS CORSIKA8KokkosEnvironment)
set_target_properties (CORSIKA8KokkosEnvironment PROPERTIES EXPORT_NAME KokkosEnvironment)
target_link_libraries (CORSIKA8KokkosEnvironment
  INTERFACE CORSIKA8KokkosData CORSIKA8
)

# Numerical model evaluation uses Kokkos annotations, not host C8 objects.
add_library (CORSIKA8KokkosModelEvaluation INTERFACE)
add_library (CORSIKA8::KokkosModelEvaluation ALIAS CORSIKA8KokkosModelEvaluation)
set_target_properties (CORSIKA8KokkosModelEvaluation PROPERTIES
  EXPORT_NAME KokkosModelEvaluation
)
target_link_libraries (CORSIKA8KokkosModelEvaluation
  INTERFACE CORSIKA8KokkosData Kokkos::kokkos
)

# Device geometry queries use exported values without native C8 object access.
add_library (CORSIKA8KokkosEnvironmentSearch INTERFACE)
add_library (CORSIKA8::KokkosEnvironmentSearch ALIAS CORSIKA8KokkosEnvironmentSearch)
set_target_properties (CORSIKA8KokkosEnvironmentSearch PROPERTIES
  EXPORT_NAME KokkosEnvironmentSearch
)
target_link_libraries (CORSIKA8KokkosEnvironmentSearch
  INTERFACE CORSIKA8KokkosData Kokkos::kokkos
)

add_library (CORSIKA8KokkosRuntime runtime/Runtime.cpp)
add_library (CORSIKA8::KokkosRuntime ALIAS CORSIKA8KokkosRuntime)
set_target_properties (CORSIKA8KokkosRuntime PROPERTIES EXPORT_NAME KokkosRuntime)
target_compile_features (CORSIKA8KokkosRuntime PUBLIC cxx_std_17)
target_include_directories (
  CORSIKA8KokkosRuntime
  PUBLIC
  $<BUILD_INTERFACE:${CORSIKA_KOKKOS_ROOT}>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  )
target_link_libraries (CORSIKA8KokkosRuntime PUBLIC Kokkos::kokkos Threads::Threads)

# Stable integer prefix scans and execution-specific range construction.
add_library (CORSIKA8KokkosScan INTERFACE)
add_library (CORSIKA8::KokkosScan ALIAS CORSIKA8KokkosScan)
set_target_properties (CORSIKA8KokkosScan PROPERTIES EXPORT_NAME KokkosScan)
target_link_libraries (CORSIKA8KokkosScan INTERFACE CORSIKA8KokkosData Kokkos::kokkos)

# Stable particle storage, upload/download and queue ownership.
add_library (CORSIKA8ParticleQueues INTERFACE)
add_library (CORSIKA8::ParticleQueues ALIAS CORSIKA8ParticleQueues)
set_target_properties (CORSIKA8ParticleQueues PROPERTIES EXPORT_NAME ParticleQueues)
target_link_libraries (CORSIKA8ParticleQueues INTERFACE CORSIKA8KokkosData
  CORSIKA8KokkosRuntime)

# Production particle bucketing preserves equal-key source order.
add_library (CORSIKA8WavefrontBucketing INTERFACE)
add_library (CORSIKA8::WavefrontBucketing ALIAS CORSIKA8WavefrontBucketing)
set_target_properties (CORSIKA8WavefrontBucketing PROPERTIES EXPORT_NAME WavefrontBucketing)
target_link_libraries (CORSIKA8WavefrontBucketing INTERFACE CORSIKA8ParticleQueues)

# Own uploaded numerical environment arrays without native C8 object access.
add_library (CORSIKA8EnvironmentStorage INTERFACE)
add_library (CORSIKA8::EnvironmentStorage ALIAS CORSIKA8EnvironmentStorage)
set_target_properties (CORSIKA8EnvironmentStorage PROPERTIES EXPORT_NAME EnvironmentStorage)
target_link_libraries (CORSIKA8EnvironmentStorage INTERFACE CORSIKA8KokkosData
  CORSIKA8KokkosRuntime)

# Native queries do not require a host PROPOSAL object or the full C8 framework.
add_library (CORSIKA8NativeTableQueries
  em/tables/ProposalNativeTable.cpp
  utility/Sha256.cpp
)
add_library (CORSIKA8::NativeTableQueries ALIAS CORSIKA8NativeTableQueries)
set_target_properties (CORSIKA8NativeTableQueries PROPERTIES EXPORT_NAME NativeTableQueries)
target_link_libraries (CORSIKA8NativeTableQueries
  PUBLIC CORSIKA8KokkosData Kokkos::kokkos
)

# The continuous-step facade retains C8's generated transport-mass check in
# ordinary host translation units; device code uses the same scalar constants.
add_library (CORSIKA8EmContinuous INTERFACE)
add_library (CORSIKA8::EmContinuous ALIAS CORSIKA8EmContinuous)
set_target_properties (CORSIKA8EmContinuous PROPERTIES EXPORT_NAME EmContinuous)
target_link_libraries (CORSIKA8EmContinuous
  INTERFACE CORSIKA8NativeTableQueries CORSIKA8
)

# Fixed-input discrete queries borrow numerical tables, not host calculators.
add_library (CORSIKA8EmDiscrete INTERFACE)
add_library (CORSIKA8::EmDiscrete ALIAS CORSIKA8EmDiscrete)
set_target_properties (CORSIKA8EmDiscrete PROPERTIES EXPORT_NAME EmDiscrete)
target_link_libraries (CORSIKA8EmDiscrete INTERFACE CORSIKA8NativeTableQueries)

# Random identities, query routing and fallback records do not run a cascade.
add_library (CORSIKA8EmStochastic INTERFACE)
add_library (CORSIKA8::EmStochastic ALIAS CORSIKA8EmStochastic)
set_target_properties (CORSIKA8EmStochastic PROPERTIES EXPORT_NAME EmStochastic)
target_link_libraries (CORSIKA8EmStochastic
  INTERFACE CORSIKA8EmContinuous CORSIKA8EmDiscrete
)

# Per-particle selection does not own queues, transport or CPU final-state handlers.
add_library (CORSIKA8EmSelection INTERFACE)
add_library (CORSIKA8::EmSelection ALIAS CORSIKA8EmSelection)
set_target_properties (CORSIKA8EmSelection PROPERTIES EXPORT_NAME EmSelection)
target_link_libraries (CORSIKA8EmSelection INTERFACE CORSIKA8EmStochastic)

# Per-particle transport consumes the same exported environment tree queries.
add_library (CORSIKA8EmTransport INTERFACE)
add_library (CORSIKA8::EmTransport ALIAS CORSIKA8EmTransport)
set_target_properties (CORSIKA8EmTransport PROPERTIES EXPORT_NAME EmTransport)
target_link_libraries (CORSIKA8EmTransport
  INTERFACE CORSIKA8EmStochastic CORSIKA8KokkosEnvironmentSearch
)

# Host radio snapshot export and value-only device projection stay separate.
add_library (CORSIKA8RadioSnapshot INTERFACE)
add_library (CORSIKA8::RadioSnapshot ALIAS CORSIKA8RadioSnapshot)
set_target_properties (CORSIKA8RadioSnapshot PROPERTIES EXPORT_NAME RadioSnapshot)
target_link_libraries (CORSIKA8RadioSnapshot INTERFACE CORSIKA8KokkosData CORSIKA8)

add_library (CORSIKA8RadioProjection INTERFACE)
add_library (CORSIKA8::RadioProjection ALIAS CORSIKA8RadioProjection)
set_target_properties (CORSIKA8RadioProjection PROPERTIES EXPORT_NAME RadioProjection)
target_link_libraries (CORSIKA8RadioProjection INTERFACE CORSIKA8KokkosData CORSIKA8 Kokkos::kokkos)

# Production radio storage, host staging and fixed-point endpoint merge.
add_library (CORSIKA8RadioAccumulation INTERFACE)
add_library (CORSIKA8::RadioAccumulation ALIAS CORSIKA8RadioAccumulation)
set_target_properties (CORSIKA8RadioAccumulation PROPERTIES EXPORT_NAME RadioAccumulation)
target_link_libraries (CORSIKA8RadioAccumulation INTERFACE CORSIKA8RadioProjection CORSIKA8KokkosRuntime)

# Production profile records and endpoint projection; no accumulation or queues.
add_library (CORSIKA8ProfileProjection INTERFACE)
add_library (CORSIKA8::ProfileProjection ALIAS CORSIKA8ProfileProjection)
set_target_properties (CORSIKA8ProfileProjection PROPERTIES EXPORT_NAME ProfileProjection)
target_link_libraries (CORSIKA8ProfileProjection INTERFACE CORSIKA8KokkosData CORSIKA8)

# Production fixed-point profile sums; ownership and queue dispatch stay separate.
add_library (CORSIKA8ProfileAccumulation INTERFACE)
add_library (CORSIKA8::ProfileAccumulation ALIAS CORSIKA8ProfileAccumulation)
set_target_properties (CORSIKA8ProfileAccumulation PROPERTIES EXPORT_NAME ProfileAccumulation)
target_link_libraries (CORSIKA8ProfileAccumulation
  INTERFACE CORSIKA8ProfileProjection Kokkos::kokkos)

# Profile owners, checked integer merge and the production memory budget.
add_library (CORSIKA8ProfileStorage INTERFACE)
add_library (CORSIKA8::ProfileStorage ALIAS CORSIKA8ProfileStorage)
set_target_properties (CORSIKA8ProfileStorage PROPERTIES EXPORT_NAME ProfileStorage)
target_link_libraries (CORSIKA8ProfileStorage INTERFACE CORSIKA8ProfileAccumulation)

# Auxiliary model queries use copied numerical records, without a host library.
add_library (CORSIKA8EmModels INTERFACE)
add_library (CORSIKA8::EmModels ALIAS CORSIKA8EmModels)
set_target_properties (CORSIKA8EmModels PROPERTIES EXPORT_NAME EmModels)
target_link_libraries (CORSIKA8EmModels INTERFACE CORSIKA8KokkosData)

# Final-state classification/materialization owns neither queues nor host models.
add_library (CORSIKA8EmFinalState INTERFACE)
add_library (CORSIKA8::EmFinalState ALIAS CORSIKA8EmFinalState)
set_target_properties (CORSIKA8EmFinalState PROPERTIES EXPORT_NAME EmFinalState)
target_link_libraries (CORSIKA8EmFinalState INTERFACE CORSIKA8EmStochastic CORSIKA8EmModels)

# Host construction of the production Moliere interpolation is available even
# when native PROPOSAL export is disabled; it consumes a numerical snapshot.
find_package (CubicInterpolation CONFIG REQUIRED)
add_library (CORSIKA8MoliereInterpolation em/tables/MoliereInterpolation.cpp)
add_library (CORSIKA8::MoliereInterpolation ALIAS CORSIKA8MoliereInterpolation)
set_target_properties (CORSIKA8MoliereInterpolation PROPERTIES
  EXPORT_NAME MoliereInterpolation
)
target_link_libraries (CORSIKA8MoliereInterpolation
  PUBLIC CORSIKA8EmModels
  PRIVATE CubicInterpolation::CubicInterpolation
)

# Shared resident inputs, borrowing tables and the uploaded environment tree.
add_library (CORSIKA8PhysicsContext INTERFACE)
add_library (CORSIKA8::PhysicsContext ALIAS CORSIKA8PhysicsContext)
set_target_properties (CORSIKA8PhysicsContext PROPERTIES EXPORT_NAME PhysicsContext)
target_link_libraries (CORSIKA8PhysicsContext INTERFACE CORSIKA8EmSelection
  CORSIKA8EnvironmentStorage CORSIKA8EmModels CORSIKA8ProfileProjection)

# One resident photon front preserves production kernel and output order.
add_library (CORSIKA8PhotonFront INTERFACE)
add_library (CORSIKA8::PhotonFront ALIAS CORSIKA8PhotonFront)
set_target_properties (CORSIKA8PhotonFront PROPERTIES EXPORT_NAME PhotonFront)
target_link_libraries (CORSIKA8PhotonFront INTERFACE CORSIKA8PhysicsContext
  CORSIKA8WavefrontBucketing CORSIKA8EmTransport CORSIKA8EmFinalState)

# The resident photon loop retains production queues, waits and profile ownership.
add_library (CORSIKA8PhotonCascade INTERFACE)
add_library (CORSIKA8::PhotonCascade ALIAS CORSIKA8PhotonCascade)
set_target_properties (CORSIKA8PhotonCascade PROPERTIES EXPORT_NAME PhotonCascade)
target_link_libraries (CORSIKA8PhotonCascade INTERFACE CORSIKA8PhotonFront CORSIKA8ProfileStorage)

# Production lepton diagnostics and fused fixed-point profile reduction.
add_library (CORSIKA8LeptonStatistics INTERFACE)
add_library (CORSIKA8::LeptonStatistics ALIAS CORSIKA8LeptonStatistics)
set_target_properties (CORSIKA8LeptonStatistics PROPERTIES EXPORT_NAME LeptonStatistics)
target_link_libraries (CORSIKA8LeptonStatistics INTERFACE CORSIKA8LeptonFront CORSIKA8ProfileStorage)

# One resident lepton front preserves production kernel and output order.
add_library (CORSIKA8LeptonFront INTERFACE)
add_library (CORSIKA8::LeptonFront ALIAS CORSIKA8LeptonFront)
set_target_properties (CORSIKA8LeptonFront PROPERTIES EXPORT_NAME LeptonFront)
target_link_libraries (CORSIKA8LeptonFront INTERFACE CORSIKA8PhysicsContext CORSIKA8WavefrontBucketing CORSIKA8EmTransport CORSIKA8EmFinalState)

# Scalar track collection and bounded replay into the radio accumulator.
add_library (CORSIKA8HostRadio INTERFACE)
add_library (CORSIKA8::HostRadio ALIAS CORSIKA8HostRadio)
set_target_properties (CORSIKA8HostRadio PROPERTIES EXPORT_NAME HostRadio)
target_link_libraries (CORSIKA8HostRadio INTERFACE CORSIKA8RadioAccumulation CORSIKA8)

# Production charged-lepton loop and profile/radio ownership.
add_library (CORSIKA8LeptonCascade INTERFACE)
add_library (CORSIKA8::LeptonCascade ALIAS CORSIKA8LeptonCascade)
set_target_properties (CORSIKA8LeptonCascade PROPERTIES EXPORT_NAME LeptonCascade)
target_link_libraries (CORSIKA8LeptonCascade INTERFACE CORSIKA8LeptonFront
  CORSIKA8LeptonStatistics CORSIKA8RadioAccumulation CORSIKA8ProfileStorage)

# Only this host bridge reads the revision-locked dependency exports.
if (CORSIKA_ENABLE_PROPOSAL_EXPORT)
  add_library (CORSIKA8ProposalBridge
    em/proposal/NativeTableExporter.cpp
    em/proposal/ProposalNativeAux.cpp
  )
  add_library (CORSIKA8::ProposalBridge ALIAS CORSIKA8ProposalBridge)
  set_target_properties (CORSIKA8ProposalBridge PROPERTIES EXPORT_NAME ProposalBridge)
  target_link_libraries (CORSIKA8ProposalBridge
    PUBLIC CORSIKA8NativeTableQueries CORSIKA8EmModels
    PRIVATE CORSIKA8
  )
  install (TARGETS CORSIKA8ProposalBridge
    EXPORT CORSIKA8PublicTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  )
endif ()

if (CORSIKA_SANITIZERS_ENABLED)
  if (Kokkos_DEVICES MATCHES "CUDA|HIP|SYCL|OPENMPTARGET")
    message (FATAL_ERROR "Use a host-only Kokkos build for runtime sanitizer tests.")
  endif ()
  target_compile_options (CORSIKA8KokkosRuntime PRIVATE
    -O1 -g -fno-omit-frame-pointer -fsanitize=address,undefined -fno-sanitize-recover=all
    )
  target_link_libraries (CORSIKA8KokkosRuntime PUBLIC -fsanitize=address,undefined)
endif ()

install (
  TARGETS CORSIKA8LeptonCascade CORSIKA8HostRadio CORSIKA8RadioAccumulation CORSIKA8RadioSnapshot CORSIKA8RadioProjection CORSIKA8LeptonStatistics CORSIKA8LeptonFront CORSIKA8PhotonCascade CORSIKA8PhotonFront CORSIKA8WavefrontBucketing CORSIKA8PhysicsContext CORSIKA8KokkosRuntime CORSIKA8EnvironmentStorage CORSIKA8ParticleQueues CORSIKA8KokkosScan CORSIKA8KokkosData CORSIKA8KokkosParticleConversion
    CORSIKA8KokkosEnvironment CORSIKA8KokkosModelEvaluation CORSIKA8KokkosEnvironmentSearch
    CORSIKA8NativeTableQueries CORSIKA8EmModels CORSIKA8MoliereInterpolation
    CORSIKA8EmContinuous CORSIKA8EmDiscrete CORSIKA8EmStochastic CORSIKA8EmSelection
    CORSIKA8EmTransport CORSIKA8EmFinalState CORSIKA8ProfileProjection CORSIKA8ProfileAccumulation CORSIKA8ProfileStorage
  EXPORT CORSIKA8PublicTargets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  )
install (
  DIRECTORY "${CORSIKA_KOKKOS_ROOT}/corsika/kokkos"
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/corsika"
  )
install (
  DIRECTORY "${CORSIKA_KOKKOS_ROOT}/corsika/detail/kokkos"
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/corsika/detail"
  )

# Separate production runtime ownership from the existing generic runtime.
include ("${PROJECT_SOURCE_DIR}/cmake/BackendRuntimeTargets.cmake")

# Complete production backend, session and host application dependencies.
include ("${PROJECT_SOURCE_DIR}/cmake/BackendApplicationTargets.cmake")

# Schema-2 firn full-wave projection; the public interface has no Kokkos headers.
add_library (CORSIKA8FirnRadio radio/firn/FullWaveAccumulator.cpp
  radio/firn/GeometricalOpticsAccumulator.cpp)
add_library (CORSIKA8::FirnRadio ALIAS CORSIKA8FirnRadio)
set_target_properties (CORSIKA8FirnRadio PROPERTIES EXPORT_NAME FirnRadio)
target_link_libraries (CORSIKA8FirnRadio PUBLIC CORSIKA8KokkosRuntime CORSIKA8)
if (CORSIKA_KOKKOS_RESOLVED_BACKEND MATCHES "^CUDA")
  set_source_files_properties (radio/firn/FullWaveAccumulator.cpp
    radio/firn/GeometricalOpticsAccumulator.cpp PROPERTIES LANGUAGE CUDA)
  set_target_properties (CORSIKA8FirnRadio PROPERTIES CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON CUDA_EXTENSIONS OFF)
  target_compile_options (CORSIKA8FirnRadio PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
elseif (CORSIKA_KOKKOS_RESOLVED_BACKEND STREQUAL "HIP")
  set_source_files_properties (radio/firn/FullWaveAccumulator.cpp
    radio/firn/GeometricalOpticsAccumulator.cpp PROPERTIES LANGUAGE HIP)
  set_target_properties (CORSIKA8FirnRadio PROPERTIES HIP_STANDARD 17 HIP_STANDARD_REQUIRED ON)
endif ()
install (TARGETS CORSIKA8FirnRadio EXPORT CORSIKA8PublicTargets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

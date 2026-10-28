# (c) Copyright 2026 CORSIKA Project. Distributed under the 3-clause BSD license.
CORSIKA_ADD_TEST (testProposalFallback SOURCES testProposalFallback.cpp
  ProposalFallbackAdapterHeader.cpp)
target_link_libraries (testProposalFallback PRIVATE CORSIKA8EmStochastic)
set_tests_properties (testProposalFallback PROPERTIES LABELS "kokkos;proposal;fallback"
  TIMEOUT 60 RUN_SERIAL TRUE ENVIRONMENT "OMP_NUM_THREADS=1")

CORSIKA_ADD_TEST (testParticleTransfer SOURCES testParticleTransfer.cpp
  RouterParticleConversionHeader.cpp)
CORSIKA_ADD_TEST (testCpuFallbackHandler SOURCES testCpuFallbackHandler.cpp
  ProposalCpuFallbackHandlerHeader.cpp)
foreach (target testParticleTransfer testCpuFallbackHandler)
  target_link_libraries (${target} PRIVATE CORSIKA8EmStochastic)
  set_tests_properties (${target} PROPERTIES LABELS "kokkos;proposal;fallback"
    TIMEOUT 60 RUN_SERIAL TRUE ENVIRONMENT "OMP_NUM_THREADS=1")
endforeach ()
CORSIKA_ADD_TEST (testCpuFallbackPhysics SOURCES testCpuFallbackPhysics.cpp)
target_link_libraries (testCpuFallbackPhysics PRIVATE CORSIKA8EmStochastic)
file (MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/cpu-handler-data/PROPOSAL")
set_tests_properties (testCpuFallbackPhysics PROPERTIES LABELS "kokkos;proposal;fallback"
  TIMEOUT 600 RUN_SERIAL TRUE
  ENVIRONMENT "OMP_NUM_THREADS=1;CORSIKA_DATA=${CMAKE_CURRENT_BINARY_DIR}/cpu-handler-data")

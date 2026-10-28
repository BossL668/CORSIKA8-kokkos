# Proposed include from tests/kokkos/runtime/backend after copying formal/tests here.
set (CORSIKA_RUNTIME_CONTRACT_PREFIX testBackend)
set (CORSIKA_RUNTIME_CONTRACT_TESTS "${CMAKE_CURRENT_SOURCE_DIR}")
set (CORSIKA_RUNTIME_CONTRACT_REFERENCE "${CMAKE_CURRENT_SOURCE_DIR}/reference")
include ("${PROJECT_SOURCE_DIR}/cmake/BackendRuntimeContracts.cmake")

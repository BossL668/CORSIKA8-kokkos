
set (BC_INSTALLED FALSE)
set (BC_TESTS "${CMAKE_CURRENT_SOURCE_DIR}")
set (BC_REFERENCE "${BC_TESTS}/reference")
set (BC_NATIVE CORSIKA8)
set (BC_COORDINATOR CORSIKA8AirShowerCoordinator)
set (BC_PREFIX native)
include ("${PROJECT_SOURCE_DIR}/cmake/CoordinatorContracts.cmake")

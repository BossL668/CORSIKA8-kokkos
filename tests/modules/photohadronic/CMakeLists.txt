
set (BC_TESTS "${CMAKE_CURRENT_SOURCE_DIR}")
set (BC_REFERENCE "${BC_TESTS}/reference")
set (BC_TEST_HELPERS "${PROJECT_SOURCE_DIR}/tests/common")
set (BC_NATIVE CORSIKA8)
set (BC_SOPHIA_static Sophia_static)
set (BC_SOPHIA_shared Sophia)
set (BC_PREFIX native_photohadronic)
include ("${PROJECT_SOURCE_DIR}/cmake/PhotohadronicContracts.cmake")

#
# (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
#
# This software is distributed under the terms of the 3-clause BSD license.
# See file LICENSE for a full version of the license.
#

set (BC_INSTALLED FALSE)
set (BC_TESTS "${CMAKE_CURRENT_SOURCE_DIR}")
set (BC_NATIVE CORSIKA8)
set (BC_PREFIX application_support)
include ("${PROJECT_SOURCE_DIR}/cmake/ApplicationSupportContracts.cmake")

add_subdirectory (coordinator)

add_subdirectory (bindings)

if (TARGET CORSIKA8::CascadeApplication)
  add_executable (application_execution_selection ExecutionSelection.cpp)
  target_link_libraries (application_execution_selection PRIVATE CORSIKA8::CascadeApplication)
  add_test (NAME application_execution_selection COMMAND application_execution_selection)
  set_tests_properties (application_execution_selection PROPERTIES TIMEOUT 60 RUN_SERIAL TRUE)
endif ()

#
# (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
#
# This software is distributed under the terms of the 3-clause BSD license.
# See file LICENSE for a full version of the license.
#

set (BC_TESTS "${CMAKE_CURRENT_SOURCE_DIR}")
set (BC_TEST_HELPERS "${PROJECT_SOURCE_DIR}/tests/common")
set (BC_NATIVE CORSIKA8)
set (BC_PREFIX native)
set (BC_CUT_NATIVE_ONLY FALSE)
include ("${PROJECT_SOURCE_DIR}/cmake/NativeCutContracts.cmake")

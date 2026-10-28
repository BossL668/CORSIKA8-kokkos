#
# (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
#
# This software is distributed under the terms of the 3-clause BSD license.
# See file LICENSE for a full version of the license.
#

add_executable (testScenePreparation testScenePreparation.cpp)
target_link_libraries (testScenePreparation PRIVATE CORSIKA8)
add_test (NAME testScenePreparation COMMAND testScenePreparation)
set_tests_properties (testScenePreparation PROPERTIES
  LABELS "kokkos;environment;application" TIMEOUT 60)

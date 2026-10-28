#
# (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
#
# See file AUTHORS for a list of contributors.
#
# This software is distributed under the terms of the 3-clause BSD license.
# See file LICENSE for a full version of the license.
#

add_executable (testDecisionReplayTape testDecisionReplayTape.cpp
  DecisionReplayTypesHeader.cpp DecisionReplayTapeHeader.cpp DecisionReplayTapeOdr.cpp)
target_link_libraries (testDecisionReplayTape PRIVATE CORSIKA8 Catch2::Catch2WithMain)
target_compile_features (testDecisionReplayTape PRIVATE cxx_std_17)
if (CORSIKA_SANITIZERS_ENABLED)
  target_compile_options (testDecisionReplayTape PRIVATE -O1 -g -fno-omit-frame-pointer
    -fsanitize=address,undefined -fno-sanitize-recover=all)
  target_link_libraries (testDecisionReplayTape PRIVATE -fsanitize=address,undefined)
endif ()
add_test (NAME testDecisionReplayTape COMMAND testDecisionReplayTape
  -o ${PROJECT_BINARY_DIR}/test_outputs/junit-testDecisionReplayTape.xml -r junit)
set_tests_properties (testDecisionReplayTape PROPERTIES
  LABELS "kokkos;diagnostics;host" TIMEOUT 180 RUN_SERIAL TRUE PROCESSORS 1)

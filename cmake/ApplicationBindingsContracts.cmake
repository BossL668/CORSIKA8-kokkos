#
# (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
#
# This software is distributed under the terms of the 3-clause BSD license.
# See file LICENSE for a full version of the license.
#
add_executable (${BC_PREFIX}_application_bindings "${BC_TESTS}/Bindings.cpp" "${BC_TESTS}/Header.cpp")
target_link_libraries (${BC_PREFIX}_application_bindings PRIVATE ${BC_NATIVE})
target_include_directories (${BC_PREFIX}_application_bindings PRIVATE "${BC_REFERENCE}")
target_compile_features (${BC_PREFIX}_application_bindings PRIVATE cxx_std_17)
target_compile_options (${BC_PREFIX}_application_bindings PRIVATE -Wall -Wextra -Werror)
foreach (SCENARIO IN ITEMS bindings cascade context)
  add_test (NAME ${BC_PREFIX}_application_${SCENARIO}
    COMMAND ${BC_PREFIX}_application_bindings ${SCENARIO})
  set_tests_properties (${BC_PREFIX}_application_${SCENARIO} PROPERTIES
    TIMEOUT 60 RUN_SERIAL TRUE)
endforeach ()

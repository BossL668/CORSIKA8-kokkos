#
# (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
#
# This software is distributed under the terms of the 3-clause BSD license.
# See file LICENSE for a full version of the license.
#
find_package (Python3 REQUIRED COMPONENTS Interpreter)
foreach (KIND IN ITEMS static shared)
  add_executable (${BC_PREFIX}_sophia_mass_${KIND} "${BC_TESTS}/MassAccessor.cpp")
  target_compile_features (${BC_PREFIX}_sophia_mass_${KIND} PRIVATE cxx_std_17)
  target_link_libraries (${BC_PREFIX}_sophia_mass_${KIND} PRIVATE ${BC_SOPHIA_${KIND}})
  add_test (NAME ${BC_PREFIX}_sophia_mass_${KIND}
    COMMAND ${BC_PREFIX}_sophia_mass_${KIND})
endforeach ()
foreach (VARIANT IN ITEMS production migrated)
  add_executable (${BC_PREFIX}_generator_${VARIANT} "${BC_TESTS}/GeneratorProbe.cpp")
  target_compile_features (${BC_PREFIX}_generator_${VARIANT} PRIVATE cxx_std_17)
  target_include_directories (${BC_PREFIX}_generator_${VARIANT} PRIVATE
    "${BC_TEST_HELPERS}" "${BC_TEST_HELPERS}/../..")
  target_link_libraries (${BC_PREFIX}_generator_${VARIANT} PRIVATE
    ${BC_SOPHIA_static} ${BC_NATIVE})
  target_link_options (${BC_PREFIX}_generator_${VARIANT} PRIVATE
    "-Wl,-Map=${CMAKE_CURRENT_BINARY_DIR}/${BC_PREFIX}_generator_${VARIANT}.map")
  if (VARIANT STREQUAL "production")
    target_include_directories (${BC_PREFIX}_generator_${VARIANT} BEFORE PRIVATE "${BC_REFERENCE}")
    target_compile_definitions (${BC_PREFIX}_generator_${VARIANT} PRIVATE ORIGINAL_PRODUCTION)
  endif ()
endforeach ()
add_test (NAME ${BC_PREFIX}_actual_generator_reference COMMAND Python3::Interpreter
  "${BC_TESTS}/compare_generators.py"
  --production $<TARGET_FILE:${BC_PREFIX}_generator_production>
  --migrated $<TARGET_FILE:${BC_PREFIX}_generator_migrated>
  --output "${CMAKE_CURRENT_BINARY_DIR}/${BC_PREFIX}-generator-reference")
set_tests_properties (${BC_PREFIX}_actual_generator_reference PROPERTIES
  RUN_SERIAL TRUE TIMEOUT 2400)

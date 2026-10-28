#
# (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
#
# This software is distributed under the terms of the 3-clause BSD license.
# See file LICENSE for a full version of the license.
#
# Required: BC_TESTS, BC_NATIVE, BC_PREFIX. Installed mode links actual archives.
set (BC_SUPPORT_HEADERS HeaderSimulationTiming.cpp HeaderSetupStack.cpp)
if (NOT BC_INSTALLED OR TARGET CORSIKA8::ApplicationSupport)
  list (APPEND BC_SUPPORT_HEADERS
    HeaderBackendOptions.cpp HeaderDeviceSelection.cpp HeaderDeviceCliOptions.cpp)
endif ()
list (TRANSFORM BC_SUPPORT_HEADERS PREPEND "${BC_TESTS}/")
add_library (${BC_PREFIX}_support_headers OBJECT ${BC_SUPPORT_HEADERS})
target_link_libraries (${BC_PREFIX}_support_headers PRIVATE ${BC_NATIVE} CLI11::CLI11)
target_compile_features (${BC_PREFIX}_support_headers PRIVATE cxx_std_17)

function (bc_application_test TARGET)
  target_link_libraries (${TARGET} PRIVATE ${BC_NATIVE} CLI11::CLI11)
  target_compile_features (${TARGET} PRIVATE cxx_std_17)
  target_sources (${TARGET} PRIVATE $<TARGET_OBJECTS:${BC_PREFIX}_support_headers>)
  add_test (NAME ${TARGET} COMMAND ${TARGET})
  set_tests_properties (${TARGET} PROPERTIES TIMEOUT 180 ENVIRONMENT "OMP_NUM_THREADS=1")
endfunction ()

if (BC_INSTALLED)
  if (TARGET CORSIKA8::ApplicationSupport)
    add_executable (${BC_PREFIX}_device_options "${BC_TESTS}/DeviceOptions.cpp")
    bc_application_test (${BC_PREFIX}_device_options)
    target_link_libraries (${BC_PREFIX}_device_options PRIVATE CORSIKA8::ApplicationSupport)
  endif ()
else ()
  set (BC_POLICY_UNDEFINES
    -UCORSIKA_KOKKOS_BACKEND_CUDA -UCORSIKA_KOKKOS_BACKEND_CUDA_OPENMP
    -UCORSIKA8_KOKKOS_BACKEND_CUDA -UCORSIKA8_KOKKOS_BACKEND_CUDA_OPENMP)
  foreach (MODE IN ITEMS host cuda cuda_openmp)
    add_executable (${BC_PREFIX}_device_options_${MODE} "${BC_TESTS}/DeviceOptions.cpp"
      "${PROJECT_SOURCE_DIR}/src/kokkos/application/DeviceSelection.cpp"
      "${PROJECT_SOURCE_DIR}/src/kokkos/application/DeviceCliOptions.cpp")
    bc_application_test (${BC_PREFIX}_device_options_${MODE})
    target_compile_options (${BC_PREFIX}_device_options_${MODE} PRIVATE ${BC_POLICY_UNDEFINES})
    if (MODE STREQUAL "host")
      target_compile_definitions (${BC_PREFIX}_device_options_${MODE} PRIVATE TEST_DEVICE_QUERY_TIMEOUT)
    elseif (MODE STREQUAL "cuda")
      target_compile_options (${BC_PREFIX}_device_options_${MODE} PRIVATE
        -DCORSIKA_KOKKOS_BACKEND_CUDA -DCORSIKA8_KOKKOS_BACKEND_CUDA)
    else ()
      target_compile_options (${BC_PREFIX}_device_options_${MODE} PRIVATE
        -DCORSIKA_KOKKOS_BACKEND_CUDA_OPENMP -DCORSIKA8_KOKKOS_BACKEND_CUDA_OPENMP)
    endif ()
  endforeach ()
endif ()

add_executable (${BC_PREFIX}_simulation_timing "${BC_TESTS}/SimulationTiming.cpp")
bc_application_test (${BC_PREFIX}_simulation_timing)
foreach (MODE IN ITEMS plain history)
  add_executable (${BC_PREFIX}_hybrid_stack_${MODE} "${BC_TESTS}/HybridStack.cpp")
  bc_application_test (${BC_PREFIX}_hybrid_stack_${MODE})
  target_compile_options (${BC_PREFIX}_hybrid_stack_${MODE} PRIVATE -UWITH_HISTORY)
  if (MODE STREQUAL "history")
    target_compile_options (${BC_PREFIX}_hybrid_stack_${MODE} PRIVATE -DWITH_HISTORY)
  endif ()
endforeach ()

# Small, self-contained production regression; no C7 installation or Python.
add_executable(test_c8_egs4_embedded testEmbeddedTables.cpp)
target_link_libraries(test_c8_egs4_embedded PRIVATE CORSIKA8NativeEgs4)
if(CORSIKA_KOKKOS_BACKEND MATCHES "^CUDA")
  set_source_files_properties(testEmbeddedTables.cpp PROPERTIES LANGUAGE CUDA)
  set_target_properties(test_c8_egs4_embedded PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON
    CUDA_ARCHITECTURES "${CORSIKA_KOKKOS_CUDA_ARCHITECTURES}")
endif()
add_test(NAME c8_egs4_embedded_tables COMMAND test_c8_egs4_embedded "${CORSIKA_EGS4_TABLE}")
set_tests_properties(c8_egs4_embedded_tables PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=2;OMP_PROC_BIND=false")

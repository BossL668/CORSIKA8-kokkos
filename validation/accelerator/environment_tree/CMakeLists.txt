# Independent original C8 search oracle and CPU/device regression tests.
foreach(kind IN ITEMS Tree Curved Lepton)
  add_executable(testEnvironment${kind} EXCLUDE_FROM_ALL
    ${kind}Geometry.cpp GeometryDriver.cpp)
  target_link_libraries(testEnvironment${kind} PRIVATE CORSIKA8 Kokkos::kokkos
    CORSIKA8ProposalNative PROPOSAL::PROPOSAL nlohmann_json::nlohmann_json)
  add_dependencies(testEnvironment${kind} GenParticlesHeaders GenMediaProperties)
  add_test(NAME testEnvironment${kind} COMMAND testEnvironment${kind})
  set_tests_properties(testEnvironment${kind} PROPERTIES
    ENVIRONMENT "OMP_NUM_THREADS=1;OMP_PROC_BIND=false")
endforeach()
add_executable(testEnvironmentPolynomial EXCLUDE_FROM_ALL PolynomialStress.cpp)
target_link_libraries(testEnvironmentPolynomial PRIVATE CORSIKA8 Kokkos::kokkos
  nlohmann_json::nlohmann_json)
add_dependencies(testEnvironmentPolynomial GenParticlesHeaders)
add_test(NAME testEnvironmentPolynomial COMMAND testEnvironmentPolynomial)
if(CORSIKA_KOKKOS_BACKEND MATCHES "^CUDA(_OPENMP)?$")
  set_source_files_properties(GeometryDriver.cpp PolynomialStress.cpp PROPERTIES LANGUAGE CUDA)
  foreach(kind IN ITEMS Tree Curved Lepton Polynomial)
    target_compile_options(testEnvironment${kind} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
    target_link_libraries(testEnvironment${kind} PRIVATE CUDA::cudart CUDA::cuda_driver)
    target_link_options(testEnvironment${kind} PRIVATE "LINKER:--no-as-needed")
    set_target_properties(testEnvironment${kind} PROPERTIES CUDA_STANDARD 17
      CUDA_ARCHITECTURES "${CORSIKA_KOKKOS_CUDA_ARCHITECTURES}")
  endforeach()
elseif(CORSIKA_KOKKOS_BACKEND STREQUAL "HIP")
  set_source_files_properties(GeometryDriver.cpp PolynomialStress.cpp PROPERTIES LANGUAGE HIP)
endif()

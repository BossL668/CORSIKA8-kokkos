add_executable (testFirnRadio testFirnRadio.cpp)
target_link_libraries (testFirnRadio PRIVATE CORSIKA8 Catch2::Catch2WithMain)
target_compile_features (testFirnRadio PRIVATE cxx_std_17)
add_test (NAME testFirnRadio COMMAND testFirnRadio)
set_tests_properties (testFirnRadio PROPERTIES LABELS "kokkos;radio;firn"
  RUN_SERIAL TRUE TIMEOUT 180)

add_executable (testFirnFullWaveAccumulator testFullWaveAccumulator.cpp)
target_link_libraries (testFirnFullWaveAccumulator PRIVATE CORSIKA8FirnRadio Catch2::Catch2WithMain)
add_test (NAME testFirnFullWaveAccumulator COMMAND testFirnFullWaveAccumulator)
set_tests_properties (testFirnFullWaveAccumulator PROPERTIES LABELS "kokkos;radio;firn"
  RUN_SERIAL TRUE TIMEOUT 180 ENVIRONMENT "OMP_NUM_THREADS=2;OMP_PROC_BIND=false")

add_executable (testFirnGeometricalOptics testGeometricalOpticsAccumulator.cpp)
target_link_libraries (testFirnGeometricalOptics PRIVATE CORSIKA8FirnRadio Catch2::Catch2WithMain)
add_test (NAME testFirnGeometricalOptics COMMAND testFirnGeometricalOptics)
set_tests_properties (testFirnGeometricalOptics PROPERTIES LABELS "kokkos;radio;firn"
  RUN_SERIAL TRUE TIMEOUT 600 ENVIRONMENT "OMP_NUM_THREADS=2;OMP_PROC_BIND=false")

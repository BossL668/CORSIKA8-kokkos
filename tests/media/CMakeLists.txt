set (test_media_sources
  TestMain.cpp
  testNuclearComposition.cpp
  testEnvironment.cpp
  testShowerAxis.cpp
  testMedium.cpp
  testRefractiveIndex.cpp
  testMagneticField.cpp
  testCORSIKA7Atmospheres.cpp
  )

CORSIKA_ADD_TEST (testMedia SOURCES ${test_media_sources})
find_package(Boost REQUIRED)
target_link_libraries (testMedia PRIVATE Boost::boost) # for math/quadrature/gauss_kronrod.hpp

target_compile_definitions (
  testMedia
  PRIVATE
  REFDATADIR="${CMAKE_CURRENT_SOURCE_DIR}"
)

# Native model export is independent of the optional Kokkos backend.
CORSIKA_ADD_TEST (testModelExport SOURCES
  TestMain.cpp
  testModelExport.cpp
  DensityModelParametersHeader.cpp
  RefractivityModelParametersHeader.cpp
  IDensityModelExportHeader.cpp
  IRefractivityModelExportHeader.cpp
  IUniformMagneticFieldExportHeader.cpp
)
target_link_libraries (testModelExport PRIVATE Boost::boost)

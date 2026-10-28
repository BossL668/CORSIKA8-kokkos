set (test_output_sources
  testDummyOutputManager.cpp
  testInteractionWriter.cpp
  TestMain.cpp
  testOutputManager.cpp
  testParquetStreamer.cpp
  testWriterObservationPlane.cpp
  testWriterTracks.cpp
  testWriterEnergyLoss.cpp
  testWriterLongitudinal.cpp
  testWriterOff.cpp
  testWritePrimary.cpp
  )

CORSIKA_ADD_TEST (testOutput SOURCES ${test_output_sources})

target_compile_definitions (
  testOutput
  PRIVATE
  REFDATADIR="${CMAKE_CURRENT_SOURCE_DIR}"
)

CORSIKA_ADD_TEST(testShowerSummary SOURCES testShowerSummary.cpp testOutputLifecycle.cpp
  BaseOutputHeader.cpp ParquetStreamerHeader.cpp OutputManagerHeader.cpp ShowerSummaryHeader.cpp)

# Host output adapters also compile without the optional Kokkos runtime.
include ("${CMAKE_CURRENT_SOURCE_DIR}/bridge/OutputBridgeTests.cmake")

find_package (Python3 COMPONENTS Interpreter REQUIRED)
find_package (Threads REQUIRED)
add_executable (testFlukaWorkerProbe FlukaWorkerProbe.cpp FlukaWorkerHeader.cpp)
target_link_libraries (testFlukaWorkerProbe PRIVATE CORSIKA8 Threads::Threads)
foreach (material air ice firn rock)
  add_test (NAME testFlukaWorker_${material} COMMAND ${Python3_EXECUTABLE}
    ${CMAKE_CURRENT_SOURCE_DIR}/compare_workers.py --probe $<TARGET_FILE:testFlukaWorkerProbe>
    --material ${material} --directory ${CMAKE_CURRENT_BINARY_DIR}/results)
  set_tests_properties (testFlukaWorker_${material} PROPERTIES
    LABELS "fluka;worker" RUN_SERIAL TRUE TIMEOUT 9000)
endforeach ()
add_test (NAME testFlukaWorker_production_fixture COMMAND ${Python3_EXECUTABLE}
  ${CMAKE_CURRENT_SOURCE_DIR}/compare_self_tests.py --probe $<TARGET_FILE:testFlukaWorkerProbe>
  --directory ${CMAKE_CURRENT_BINARY_DIR}/self-test-results)
set_tests_properties (testFlukaWorker_production_fixture PROPERTIES
  LABELS "fluka;worker" RUN_SERIAL TRUE TIMEOUT 9000)

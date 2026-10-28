# FLUKA is process-global: each material/mode runs in a fresh private directory.
find_package (Python3 COMPONENTS Interpreter REQUIRED)
add_executable (testFlukaFinalStateProbe FlukaFinalStateProbe.cpp
  InteractionModelHeader.cpp DiagnosticTimerHeader.cpp)
target_link_libraries (testFlukaFinalStateProbe PRIVATE CORSIKA8)
foreach (material air ice firn rock)
  add_test (NAME testFlukaFinalState_${material}
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/compare_final_states.py
      --candidate $<TARGET_FILE:testFlukaFinalStateProbe> --material ${material}
      --directory ${CMAKE_CURRENT_BINARY_DIR}/results)
  set_tests_properties (testFlukaFinalState_${material} PROPERTIES
    LABELS "fluka;final-state" RUN_SERIAL TRUE TIMEOUT 3600)
endforeach ()

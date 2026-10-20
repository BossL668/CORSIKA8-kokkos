add_subdirectory (common)
add_subdirectory (accelerator)
add_subdirectory (framework)
add_subdirectory (media)
add_subdirectory (stack)
add_subdirectory (modules)
add_subdirectory (output)

add_test (NAME testInstallationConfiguration
          COMMAND ${PYTHON_EXECUTABLE}
                  ${CMAKE_CURRENT_SOURCE_DIR}/tools/test_install_configuration.py)

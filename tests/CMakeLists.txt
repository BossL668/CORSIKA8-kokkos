add_subdirectory (common)
add_subdirectory (framework)
add_subdirectory (media)
add_subdirectory (stack)
add_subdirectory (modules)
add_subdirectory (output)

if (CORSIKA_ENABLE_KOKKOS)
  add_subdirectory (kokkos)
endif ()

add_subdirectory (application_support)

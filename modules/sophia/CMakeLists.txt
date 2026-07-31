set (
  MODEL_SOURCES
  eventgen.f
  #inpoutput.f
  jetset74dp.f
  #sampling.f
  #SOPHIA20.f
  sophia.cpp
  )

set (
  MODEL_HEADERS
  sophia.hpp
  )

enable_language (Fortran)
add_library (Sophia_static STATIC ${MODEL_SOURCES})
add_library (Sophia SHARED ${MODEL_SOURCES})

set_target_properties (
  Sophia_static
  PROPERTIES
  POSITION_INDEPENDENT_CODE 1
  )
set_target_properties (
  Sophia
  PROPERTIES
  # Keep the mixed shared-library link inside the active Fortran toolchain.
  # This is required when conda's Fortran runtime uses its own sysroot but
  # the host C++ compiler is the default linker driver.
  LINKER_LANGUAGE Fortran
  )

target_include_directories (
  Sophia_static
  PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<INSTALL_INTERFACE:include/corsika_modules/sophia>
  )
  target_include_directories (
    Sophia
    PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include/corsika_modules/sophia>
    )

target_link_libraries (
  Sophia_static
  PUBLIC
  modules_common
  gfortran
  )
target_link_libraries (
  Sophia
  PUBLIC
  modules_common
  gfortran
  )
  
install (
  FILES
  ${MODEL_HEADERS}
  DESTINATION include/corsika_modules/sophia
  )

install (
  TARGETS Sophia_static Sophia
  EXPORT CORSIKA8PublicTargets
  ARCHIVE DESTINATION lib/corsika
  LIBRARY DESTINATION lib/corsika # just for cmake 3.10.x (ubuntu 18)
  )

# add sophia to corsika8 build
add_dependencies (CORSIKA8 Sophia_static)
target_link_libraries (CORSIKA8 INTERFACE Sophia_static)

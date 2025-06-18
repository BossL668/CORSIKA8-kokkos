set (
  MODEL_SOURCES
  sibyll2.3d-private.cpp
  sibyll2.3d.f
  nuclib.f
  signuc.f
  gasdev.f
  )

set (
  MODEL_HEADERS
  sibyll2.3d-public.hpp
  sibyll2.3d-types.hpp
  nuclib.hpp
  )

enable_language (Fortran)
add_library (SIBYLL23d_static STATIC ${MODEL_SOURCES})
add_library (SIBYLL23d SHARED "sibyll2.3d-public.cpp" "nuclib-public.cpp")

set_target_properties (
  SIBYLL23d_static
  PROPERTIES
  POSITION_INDEPENDENT_CODE 1
  CXX_VISIBILITY_PRESET hidden
  # unfortunately ignored by gfortran:
  Fortran_VISIBILITY_PRESET hidden
)

target_include_directories (
  SIBYLL23d_static
  PRIVATE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
)

target_include_directories (
  SIBYLL23d
  PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<INSTALL_INTERFACE:include/corsika_modules/sibyll>
)

target_link_libraries (
  SIBYLL23d_static
  PUBLIC
  gfortran
  modules_common
)

target_link_libraries(SIBYLL23d PRIVATE SIBYLL23d_static)
target_link_options(SIBYLL23d PRIVATE "LINKER:--exclude-libs,ALL") # hide symbols from static/private lib

install (
  FILES
  ${MODEL_HEADERS}
  DESTINATION include/corsika_modules/sibyll
)

install (
  TARGETS SIBYLL23d
  EXPORT CORSIKA8PublicTargets
  ARCHIVE DESTINATION lib/corsika
  LIBRARY DESTINATION lib/corsika # just for cmake 3.10.x (ubuntu 18)
)

# add sibyll to corsika8 build
add_dependencies (CORSIKA8 SIBYLL23d)
target_link_libraries (CORSIKA8 INTERFACE SIBYLL23d)

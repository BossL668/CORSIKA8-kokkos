set (
  MODEL_SOURCES
  qgsjet-III-04-mod1.f
  qgsjet-III-04.cpp
  )

set (
  MODEL_HEADERS
  qgsjet-III-04.hpp
  )

enable_language (Fortran)

add_library (QGSJetIII_static STATIC ${MODEL_SOURCES})
set_target_properties (
  QGSJetIII_static
  PROPERTIES
  POSITION_INDEPENDENT_CODE 1
  )

target_include_directories (
  QGSJetIII_static
  PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<INSTALL_INTERFACE:include/corsika_modules/qgsjetIII>
  )
target_link_libraries (QGSJetIII_static CorsikaData)
target_link_libraries (QGSJetIII_static modules_common)

install (
  FILES
  ${MODEL_HEADERS}
  DESTINATION include/corsika_modules/qgsjetIII
  )

install (
  TARGETS QGSJetIII_static
  EXPORT CORSIKA8PublicTargets
  DESTINATION lib/corsika
  )

# add qgsjetIII to corsika8 build
add_dependencies (CORSIKA8 QGSJetIII_static)
target_link_libraries (CORSIKA8 INTERFACE QGSJetIII_static)

enable_language (Fortran)

add_library(QGSJetIII_static STATIC
  qgsjet-III-04-mod1.f
  qgsjet-III-private.cpp
)

set_target_properties (
  QGSJetIII_static
  PROPERTIES
  POSITION_INDEPENDENT_CODE 1
  CXX_VISIBILITY_PRESET hidden
  # unfortunately ignored by gfortran:
  Fortran_VISIBILITY_PRESET hidden
)

add_library(QGSJetIII SHARED qgsjet-III-public.cpp)
target_link_libraries(QGSJetIII PRIVATE QGSJetIII_static modules_common)
target_link_options(QGSJetIII PRIVATE "LINKER:--exclude-libs,ALL") # hide symbols from static/private lib


set (
  MODEL_HEADERS
  qgsjet-III-types.hpp
  qgsjet-III-public.hpp
  )

target_include_directories (
  QGSJetIII
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
  TARGETS QGSJetIII
  EXPORT CORSIKA8PublicTargets
  DESTINATION lib/corsika
  )

# add qgsjetII to corsika8 build
add_dependencies (CORSIKA8 QGSJetIII)
target_link_libraries (CORSIKA8 INTERFACE QGSJetIII)

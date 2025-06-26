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

set_target_properties(QGSJetIII PROPERTIES
  FRAMEWORK TRUE
  PUBLIC_HEADER "${MODEL_HEADERS}"
)

target_include_directories (
  QGSJetIII
  PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/corsika_modules/qgsjetIII>
)
target_link_libraries (QGSJetIII_static CorsikaData)
target_link_libraries (QGSJetIII_static modules_common)

install (
  TARGETS QGSJetIII
  EXPORT CORSIKA8PublicTargets
  DESTINATION "${CMAKE_INSTALL_LIBDIR}/corsika"
  PUBLIC_HEADER DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/corsika_modules/qgsjetIII"
  )

# add qgsjetII to corsika8 build
add_dependencies (CORSIKA8 QGSJetIII)
target_link_libraries (CORSIKA8 INTERFACE QGSJetIII)

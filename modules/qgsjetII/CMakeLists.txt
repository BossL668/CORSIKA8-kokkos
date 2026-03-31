enable_language (Fortran)

add_library(QGSJetII_static STATIC
  qgsjet-II-04-mod1.f
  qgsjet-II-04-private.cpp
)

set_target_properties (
  QGSJetII_static
  PROPERTIES
  POSITION_INDEPENDENT_CODE 1
  CXX_VISIBILITY_PRESET hidden
  # unfortunately ignored by gfortran:
  Fortran_VISIBILITY_PRESET hidden
)

add_library(QGSJetII SHARED qgsjet-II-public.cpp)
target_link_libraries(QGSJetII PRIVATE QGSJetII_static modules_common)
target_link_options(QGSJetII PRIVATE "LINKER:--exclude-libs,ALL") # hide symbols from static/private lib

set (
  MODEL_HEADERS
  qgsjet-II-04-types.hpp
  qgsjet-II-public.hpp
  )

set_target_properties(QGSJetII PROPERTIES
  FRAMEWORK TRUE
  PUBLIC_HEADER "${MODEL_HEADERS}"
)

target_include_directories (
  QGSJetII
  PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/corsika_modules/qgsjetII>
)
target_link_libraries (QGSJetII_static CorsikaData)
target_link_libraries (QGSJetII_static modules_common)

install (
  TARGETS QGSJetII
  EXPORT CORSIKA8PublicTargets
  DESTINATION "${CMAKE_INSTALL_LIBDIR}/corsika"
  PUBLIC_HEADER DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/corsika_modules/qgsjetII"
  )


# add qgsjetII to corsika8 build
add_dependencies (CORSIKA8 QGSJetII)
target_link_libraries (CORSIKA8 INTERFACE QGSJetII)

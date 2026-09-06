include (ExternalProject)

# Optional reuse of an existing CORSIKA-layout TAUOLA 1.1.8 installation.
# This is a dependency, not an alternative shower implementation. Fresh hosts
# retain the pinned source build below when this option is empty.
set (C8_TAUOLA_PREFIX "" CACHE PATH "Prebuilt TAUOLA in CORSIKA install layout")
if (C8_TAUOLA_PREFIX)
  foreach (_file include/corsika_modules/Tauola/Tauola.h
                 lib/corsika/libTauolaCxxInterface.a
                 lib/corsika/libTauolaFortran.a)
    if (NOT EXISTS "${C8_TAUOLA_PREFIX}/${_file}")
      message (FATAL_ERROR "Incomplete C8_TAUOLA_PREFIX: missing ${_file}")
    endif ()
  endforeach ()
  add_library (C8::ext::tauola::CxxInterface STATIC IMPORTED GLOBAL)
  add_library (C8::ext::tauola::Fortran STATIC IMPORTED GLOBAL)
  add_library (C8::ext::tauola INTERFACE IMPORTED)
  foreach (_lib CxxInterface Fortran)
    set_target_properties (C8::ext::tauola::${_lib} PROPERTIES
      IMPORTED_LOCATION "${C8_TAUOLA_PREFIX}/lib/corsika/libTauola${_lib}.a"
      INTERFACE_INCLUDE_DIRECTORIES "${C8_TAUOLA_PREFIX}/include/corsika_modules")
  endforeach ()
  set_property (TARGET C8::ext::tauola PROPERTY INTERFACE_LINK_LIBRARIES
                C8::ext::tauola::CxxInterface C8::ext::tauola::Fortran)
  target_link_libraries (CORSIKA8 INTERFACE C8::ext::tauola)
  set (TAUOLA_FOUND 1 PARENT_SCOPE)
  set (TAUOLA_INCDIR "${C8_TAUOLA_PREFIX}/include/corsika_modules" PARENT_SCOPE)
  set (TAUOLA_LIBDIR "${C8_TAUOLA_PREFIX}/lib/corsika" PARENT_SCOPE)
  set (TAUOLA_INCDIR_INSTALL "${CMAKE_INSTALL_PREFIX}/include/corsika_modules" PARENT_SCOPE)
  set (TAUOLA_LIBDIR_INSTALL "${CMAKE_INSTALL_PREFIX}/lib/corsika" PARENT_SCOPE)
  install (DIRECTORY "${C8_TAUOLA_PREFIX}/" DESTINATION "${CMAKE_INSTALL_PREFIX}")
  return ()
endif ()

add_library (C8::ext::tauola::CxxInterface STATIC IMPORTED GLOBAL)
add_library (C8::ext::tauola::Fortran STATIC IMPORTED GLOBAL)
add_library(C8::ext::tauola INTERFACE IMPORTED)
set_property(TARGET C8::ext::tauola 
    PROPERTY
    INTERFACE_LINK_LIBRARIES 
    C8::ext::tauola::CxxInterface
    C8::ext::tauola::Fortran)

set (TAUOLA_PREFIX ${CMAKE_CURRENT_BINARY_DIR}/tauola/install) # this is for build
set (TAUOLA_PREFIX_INSTALL ${CMAKE_INSTALL_PREFIX})            # this is for install

# those are required for creating cmake target
set (TAUOLA_INCDIR ${TAUOLA_PREFIX}/include/corsika_modules)
set (TAUOLA_INCDIR_INSTALL ${TAUOLA_PREFIX_INSTALL}/include/corsika_modules)
set (TAUOLA_LIBDIR ${TAUOLA_PREFIX}/lib/corsika)
set (TAUOLA_LIBDIR_INSTALL ${TAUOLA_PREFIX_INSTALL}/lib/corsika)


# the URL to download the current version of TAUOLA from
set (TAUOLA_URL
  http://tauolapp.web.cern.ch/tauolapp/resources/TAUOLA.1.1.8/TAUOLA.1.1.8-LHC.tar.gz
  )

# and the md5 hash corresponding to this version of TAULA
set (TAUOLA_MD5
  830c24457627d5bdc2c4e995eb208170
  )

# the flags we use to build TAUOLA
set (TAUOLA_FLAGS
  --without-hepmc 
  --without-hepmc3 
  --without-pythia8 
  FFLAGS=-w 
  CFLAGS=-w
  --prefix=${TAUOLA_PREFIX}
  )

# configure the TAUOLA build process
ExternalProject_Add (
  tauola
  PREFIX ${CMAKE_CURRENT_BINARY_DIR}
  URL ${TAUOLA_URL}
  URL_MD5 ${TAUOLA_MD5}
  DOWNLOAD_NO_PROGRESS ON
  DOWNLOAD_NAME tauola
  SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/tauola/source"
  INSTALL_DIR "${CMAKE_CURRENT_BINARY_DIR}/tauola/install"
  CONFIGURE_COMMAND <SOURCE_DIR>/configure ${TAUOLA_FLAGS}
  BUILD_COMMAND make
  UPDATE_COMMAND ""
  INSTALL_COMMAND make install
  COMMAND rm -rf ${TAUOLA_PREFIX}/bin
  COMMAND rm -rf ${TAUOLA_PREFIX}/share/tauola/examples
  COMMAND mkdir -p ${TAUOLA_INCDIR}
  COMMAND mkdir -p ${TAUOLA_LIBDIR}
  COMMAND rm -rf ${TAUOLA_PREFIX}/include/corsika_modules/Tauola
  COMMAND mv -f ${TAUOLA_PREFIX}/include/Tauola ${TAUOLA_INCDIR}
  COMMAND mv ${TAUOLA_PREFIX}/lib/libTauolaCxxInterface.a  ${TAUOLA_LIBDIR}
  COMMAND mv ${TAUOLA_PREFIX}/lib/libTauolaFortran.a  ${TAUOLA_LIBDIR}
  COMMAND rm ${TAUOLA_PREFIX}/lib/libTauolaCxxInterface.la
  COMMAND rm ${TAUOLA_PREFIX}/lib/libTauolaCxxInterface.so
  COMMAND rm ${TAUOLA_PREFIX}/lib/libTauolaCxxInterface.so.0
  COMMAND rm ${TAUOLA_PREFIX}/lib/libTauolaCxxInterface.so.0.0.0
  COMMAND rm ${TAUOLA_PREFIX}/lib/libTauolaFortran.la
  COMMAND rm ${TAUOLA_PREFIX}/lib/libTauolaFortran.so
  COMMAND rm ${TAUOLA_PREFIX}/lib/libTauolaFortran.so.0
  COMMAND rm ${TAUOLA_PREFIX}/lib/libTauolaFortran.so.0.0.0
  COMMAND rm ${TAUOLA_PREFIX}/lib/libTauolaHEPEVT.a
  COMMAND rm ${TAUOLA_PREFIX}/lib/libTauolaHEPEVT.la
  COMMAND rm ${TAUOLA_PREFIX}/lib/libTauolaHEPEVT.so
  COMMAND rm ${TAUOLA_PREFIX}/lib/libTauolaHEPEVT.so.0
  COMMAND rm ${TAUOLA_PREFIX}/lib/libTauolaHEPEVT.so.0.0.0
  )

install (DIRECTORY ${TAUOLA_PREFIX}/ DESTINATION ${TAUOLA_PREFIX_INSTALL})
message (STATUS "TAUOLA build at: ${TAUOLA_PREFIX}, final install to: ${TAUOLA_PREFIX_INSTALL}")
message (STATUS "TAUOLA_INCDIR: ${TAUOLA_INCDIR}")
message (STATUS "TAUOLA_LIBDIR: ${TAUOLA_LIBDIR}")

add_dependencies (C8::ext::tauola tauola)

set_target_properties (
    C8::ext::tauola::CxxInterface
    PROPERTIES
    IMPORTED_LOCATION ${TAUOLA_LIBDIR}/libTauolaCxxInterface.a
    IMPORTED_LINK_INTERFACE_LIBRARIES dl
    INTERFACE_INCLUDE_DIRECTORIES $<BUILD_INTERFACE:${TAUOLA_INCDIR}>
    )

set_target_properties (
    C8::ext::tauola::Fortran
    PROPERTIES
    IMPORTED_LOCATION ${TAUOLA_LIBDIR}/libTauolaFortran.a
    IMPORTED_LINK_INTERFACE_LIBRARIES dl
    INTERFACE_INCLUDE_DIRECTORIES $<BUILD_INTERFACE:${TAUOLA_INCDIR}>
    )

add_dependencies (CORSIKA8 C8::ext::tauola)
target_link_libraries (CORSIKA8 INTERFACE C8::ext::tauola)

# those are required for creating cmake target
set (TAUOLA_FOUND 1 PARENT_SCOPE)
set (TAUOLA_INCDIR ${TAUOLA_INCDIR} PARENT_SCOPE)
set (TAUOLA_INCDIR_INSTALL ${TAUOLA_INCDIR_INSTALL} PARENT_SCOPE)
set (TAUOLA_LIBDIR ${TAUOLA_LIBDIR} PARENT_SCOPE)
set (TAUOLA_LIBDIR_INSTALL ${TAUOLA_LIBDIR_INSTALL} PARENT_SCOPE)

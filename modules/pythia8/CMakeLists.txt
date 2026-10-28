include (ExternalProject)

# eventually add AUTO here, too:
set (ThirdPartyChoiceValues "C8;SYSTEM" CACHE STRING
    "List of possible values for the ThirdParty package choice")
mark_as_advanced (ThirdPartyChoiceValues)

##############################################################################
# check for Pythia8: either use C8 or system-level installation

message (STATUS "***** Configuring Pythia8 version")

set (USE_Pythia8_C8 "C8" CACHE STRING
    "Selection of pythia8 package. Can be \'C8\' or \'SYSTEM\'. Default: \'C8\'.")
set_property (CACHE USE_Pythia8_C8 PROPERTY STRINGS ${ThirdPartyChoiceValues})
if (NOT (${USE_Pythia8_C8} IN_LIST ThirdPartyChoiceValues))
  message (SEND_ERROR "Illegal USE_Pythia8_C8=\"${USE_Pythia8_C8}\" can only be one of: ${ThirdPartyChoiceValues}")
endif (NOT (${USE_Pythia8_C8} IN_LIST ThirdPartyChoiceValues))
message (STATUS "USE_Pythia8_C8='${USE_Pythia8_C8}'")

set (CORSIKA_Pythia8_MODULE_DIR ${PROJECT_BINARY_DIR}/corsika/modules/pythia8)
file (MAKE_DIRECTORY ${CORSIKA_Pythia8_MODULE_DIR})

add_library (C8::ext::pythia8 STATIC IMPORTED GLOBAL)
if ("x_${USE_Pythia8_C8}" STREQUAL "x_SYSTEM")

  find_package (Pythia8 8315 EXACT REQUIRED)
  message (STATUS "Using system-level Pythia8 version ${Pythia8_VERSION} at ${Pythia8_PREFIX}")
  set (Pythia8_INCLUDE_DIRS ${Pythia8_INCLUDE_DIR})
  set_target_properties (
    C8::ext::pythia8 PROPERTIES
    IMPORTED_LOCATION ${Pythia8_LIBRARY}/libpythia8.a
    IMPORTED_LINK_INTERFACE_LIBRARIES dl
    INTERFACE_INCLUDE_DIRECTORIES "${Pythia8_INCLUDE_DIRS}"
    )
  set (Pythia8_FOUND 1 PARENT_SCOPE)

  set (CORSIKA_Pythia8_LOCATION ${Pythia8_PREFIX}) # build and install location, pythia8 is external
  configure_file (Pythia8ConfigurationDirectory.hpp.in ${CORSIKA_Pythia8_MODULE_DIR}/Pythia8ConfigurationDirectory.hpp  @ONLY)
  configure_file (Pythia8ConfigurationDirectory.hpp.in ${CMAKE_CURRENT_BINARY_DIR}/Pythia8ConfigurationDirectory_install.hpp  @ONLY)

  # install and build are both the same here since it is external
  set (Pythia8_PREFIX_INSTALL ${Pythia8_PREFIX})
  set (Pythia8_LIBDIR ${Pythia8_LIBRARY})
  set (Pythia8_LIBDIR_INSTALL ${Pythia8_LIBRARY})
  set (Pythia8_INCDIR ${Pythia8_INCLUDE_DIR})
  set (Pythia8_INCDIR_INSTALL ${Pythia8_INCLUDE_DIR})

  message (STATUS "Pythia8 (system) at ${Pythia8_PREFIX}")

else ()

  set (_C8_Pythia8_Download_Dir "https://pythia.org/download/pythia83")
  set (_C8_Pythia8_VERSION "8315")
  message (STATUS "Building modules/pythia8 using pythia${_C8_Pythia8_VERSION}.tar.bz2")
  message (STATUS "This will take a bit.....")
  # this is not a full Pythia8 install, it is a bit simplified, e.g. no pythia8-config, no examples
  # and also, it is fitted into the normal CORSIKA8 install "include/corsika_modules/Pythia8, lib/corsika"
  ExternalProject_Add (pythia8
    URL ${_C8_Pythia8_Download_Dir}/pythia${_C8_Pythia8_VERSION}.tar.bz2
    URL_MD5 faf2730a959369e4d25e1285ab70d915
    SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/pythia8/source"
    INSTALL_DIR "${CMAKE_CURRENT_BINARY_DIR}/pythia8/install"
    CONFIGURE_COMMAND ./configure --prefix=${CMAKE_CURRENT_BINARY_DIR}/pythia8/install
    INSTALL_COMMAND make install
    COMMAND rm -rf ${CMAKE_CURRENT_BINARY_DIR}/pythia8/install/bin
    COMMAND rm -rf ${CMAKE_CURRENT_BINARY_DIR}/pythia8/install/share/Pythia8/examples
    COMMAND mkdir -p ${CMAKE_CURRENT_BINARY_DIR}/pythia8/install/include/corsika_modules
    COMMAND mkdir -p ${CMAKE_CURRENT_BINARY_DIR}/pythia8/install/lib/corsika
    COMMAND rm -rf ${CMAKE_CURRENT_BINARY_DIR}/pythia8/install/include/corsika_modules/Pythia8
    COMMAND mv -f ${CMAKE_CURRENT_BINARY_DIR}/pythia8/install/include/Pythia8 ${CMAKE_CURRENT_BINARY_DIR}/pythia8/install/include/corsika_modules/
    COMMAND mv ${CMAKE_CURRENT_BINARY_DIR}/pythia8/install/lib/libpythia8.a  ${CMAKE_CURRENT_BINARY_DIR}/pythia8/install/lib/corsika/
    BUILD_IN_SOURCE ON
    EXCLUDE_FROM_ALL TRUE
    )
  set (Pythia8_FOUND 1 PARENT_SCOPE)
  set (Pythia8_PREFIX_INSTALL ${CMAKE_INSTALL_PREFIX}) # this is "make install"
  ExternalProject_Get_Property (pythia8 INSTALL_DIR) # this is "make"
  set (Pythia8_VERSION ${_C8_Pythia8_VERSION} CACHE STRING "Version of Pythia8")
  set (Pythia8_PREFIX ${INSTALL_DIR})
  set (Pythia8_INCLUDE_DIR  ${Pythia8_PREFIX}/include/corsika_modules)
  set (Pythia8_LIBDIR ${Pythia8_PREFIX}/lib/corsika)
  set (Pythia8_LIBDIR_INSTALL ${Pythia8_PREFIX_INSTALL}/lib/corsika)
  set (Pythia8_INCDIR ${Pythia8_PREFIX}/include/corsika_modules)
  set (Pythia8_INCDIR_INSTALL ${Pythia8_PREFIX_INSTALL}/include/corsika_modules)
  add_dependencies (C8::ext::pythia8 pythia8)

  set (CORSIKA_Pythia8_LOCATION ${Pythia8_PREFIX}) # build location
  configure_file (Pythia8ConfigurationDirectory.hpp.in ${CORSIKA_Pythia8_MODULE_DIR}/Pythia8ConfigurationDirectory.hpp  @ONLY)
  set (CORSIKA_Pythia8_LOCATION ${Pythia8_PREFIX_INSTALL}) # install location
  configure_file (Pythia8ConfigurationDirectory.hpp.in ${CMAKE_CURRENT_BINARY_DIR}/Pythia8ConfigurationDirectory_install.hpp  @ONLY)

  install (DIRECTORY ${Pythia8_PREFIX}/ DESTINATION ${Pythia8_PREFIX_INSTALL})

  set (Pythia8_INCLUDE_DIRS ${Pythia8_INCLUDE_DIR})

  set_target_properties (
    C8::ext::pythia8
    PROPERTIES
    IMPORTED_LOCATION ${Pythia8_LIBDIR}/libpythia8.a
    IMPORTED_LINK_INTERFACE_LIBRARIES dl
    )
  set_property (
    TARGET C8::ext::pythia8
    APPEND PROPERTY
    INTERFACE_INCLUDE_DIRECTORIES $<BUILD_INTERFACE:${Pythia8_INCLUDE_DIRS}>
                                  $<INSTALL_INTERFACE:include/corsika/modules/pythia8/include>
    )

    message (STATUS "Pythia8 build at: ${Pythia8_PREFIX}, final install to: ${Pythia8_PREFIX_INSTALL}")
endif ()

# the install location of Pythia8
install (
  FILES ${CMAKE_CURRENT_BINARY_DIR}/Pythia8ConfigurationDirectory_install.hpp
  RENAME Pythia8ConfigurationDirectory.hpp
  DESTINATION include/corsika/modules/pythia8
  )

##### add pythia8 to CORSIKA8 build
add_dependencies (CORSIKA8 C8::ext::pythia8)
target_link_libraries (CORSIKA8 INTERFACE C8::ext::pythia8)

# we need those for exporting targets and dependencies:
set (Pythia8_FOUND 1 PARENT_SCOPE)
set (Pythia8_INCDIR ${Pythia8_INCDIR} PARENT_SCOPE)
set (Pythia8_INCDIR_INSTALL ${Pythia8_INCDIR_INSTALL} PARENT_SCOPE)
set (Pythia8_LIBDIR ${Pythia8_LIBDIR} PARENT_SCOPE)
set (Pythia8_LIBDIR_INSTALL ${Pythia8_LIBDIR_INSTALL} PARENT_SCOPE)

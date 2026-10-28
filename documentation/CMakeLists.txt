find_package (Doxygen OPTIONAL_COMPONENTS dot mscgen dia)
find_package (Sphinx)

if (DOXYGEN_FOUND)
  if (NOT DOXYGEN_DOT_EXECUTABLE)
    message (FATAL_ERROR "Found doxygen but not 'dot' command, please install graphviz or set DOXYGEN_DOT_EXECUTABLE")
  endif (NOT DOXYGEN_DOT_EXECUTABLE)

  set (DOXYGEN_IN ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in)
  set (DOXYGEN_OUT ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)
  set (DOXYGEN_GENERATE_HTML YES)
  set (DOXYGEN_GENERATE_MAN YES)
  configure_file (${DOXYGEN_IN} ${DOXYGEN_OUT} @ONLY)
  message ("Start doxygen with \"make docs\"")

  # note the option ALL which allows to build the docs together with the application
  add_custom_target (docs # ALL
    COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_OUT}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Generating API documentation with Doxygen"
    VERBATIM
    )

  add_custom_command (TARGET docs POST_BUILD
    COMMAND cd ${CMAKE_CURRENT_BINARY_DIR}/latex; pdflatex refman.tex
    )

  install (DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/html DESTINATION ${CMAKE_INSTALL_DATADIR}/corsika/doc OPTIONAL)
  install (FILES ${CMAKE_CURRENT_BINARY_DIR}/latex/refman.pdf DESTINATION ${CMAKE_INSTALL_DATADIR}/corsika/doc OPTIONAL)

  if (SPHINX_FOUND)

    set (SPHINX_SOURCE ${CMAKE_CURRENT_SOURCE_DIR})
    set (SPHINX_BUILD ${CMAKE_CURRENT_BINARY_DIR}/docs/sphinx)

    add_custom_command (TARGET docs POST_BUILD
      COMMAND ${SPHINX_EXECUTABLE} -b html
      # Tell Breathe where to find the Doxygen output
      -Dbreathe_projects.CORSIKA8=${CMAKE_CURRENT_BINARY_DIR}/xml/
      ${SPHINX_SOURCE} ${SPHINX_BUILD}
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
      COMMENT "Generating documentation with Sphinx"
      )

  else (SPHINX_FOUND)

    message ("Sphinx not found;")

  endif (SPHINX_FOUND)

else (DOXYGEN_FOUND)

  message ("Doxygen need to be installed to generate the doxygen documentation")

endif (DOXYGEN_FOUND)

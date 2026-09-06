set (scripts
  plot_crossings.sh
  plot_tracks.sh
  read_hist.py
  )

install (PROGRAMS ${scripts} DESTINATION share/corsika/tools)

# The shared launcher never links a device runtime. Individual installations
# remain <root>/<backend>; a single source tree may populate several of them.
set (CORSIKA_LAUNCHER_PREFIX "" CACHE PATH
     "Optional common install root containing bin/ and backend subdirectories")
if (CORSIKA_ENABLE_KOKKOS)
  string (TOLOWER "${CORSIKA_KOKKOS_BACKEND}" _c8_launcher_backend)
  set (_c8_launcher_host "serial")
  if (_c8_launcher_backend STREQUAL "openmp")
    set (_c8_launcher_host "openmp")
  endif ()
  configure_file (backend.json.in backend.json @ONLY)
  install (FILES "${CMAKE_CURRENT_BINARY_DIR}/backend.json"
           DESTINATION share/corsika COMPONENT Launcher)
  if (CORSIKA_LAUNCHER_PREFIX)
    get_filename_component (_c8_expected_prefix
      "${CORSIKA_LAUNCHER_PREFIX}/${_c8_launcher_backend}" ABSOLUTE)
    if (NOT CMAKE_INSTALL_PREFIX STREQUAL _c8_expected_prefix)
      message (FATAL_ERROR
        "The unified launcher expects CMAKE_INSTALL_PREFIX="
        "${_c8_expected_prefix}, not ${CMAKE_INSTALL_PREFIX}")
    endif ()
    install (PROGRAMS c8_air_shower_launcher.py
      DESTINATION "${CORSIKA_LAUNCHER_PREFIX}/bin"
      RENAME c8_air_shower COMPONENT Launcher)
  endif ()
endif ()

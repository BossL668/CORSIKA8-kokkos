set (scripts
  plot_crossings.sh
  plot_tracks.sh
  read_hist.py
  )

install (PROGRAMS ${scripts} DESTINATION share/corsika/tools)

# Applications are native binaries installed directly in each backend's prefix.
# Keep reading the old option so an existing build cache produces a clear notice.
if (CORSIKA_LAUNCHER_PREFIX)
  message (WARNING
    "CORSIKA_LAUNCHER_PREFIX is obsolete and ignored. Run the native binary at "
    "${CMAKE_INSTALL_PREFIX}/bin/c8_air_shower directly. An old common launcher, "
    "if present, is not updated or used by this installation.")
endif ()
if (CORSIKA_ENABLE_KOKKOS)
  string (TOLOWER "${CORSIKA_KOKKOS_BACKEND}" _c8_manifest_backend)
  set (_c8_manifest_host "serial")
  if (_c8_manifest_backend MATCHES "^(openmp|cuda_openmp)$")
    set (_c8_manifest_host "openmp")
  endif ()
  configure_file (backend.json.in backend.json @ONLY)
  install (FILES "${CMAKE_CURRENT_BINARY_DIR}/backend.json"
           DESTINATION share/corsika COMPONENT Runtime)
endif ()

set (scripts
  plot_crossings.sh
  plot_tracks.sh
  read_hist.py
  )

install (PROGRAMS ${scripts} DESTINATION share/corsika/tools)
